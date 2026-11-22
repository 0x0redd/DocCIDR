#!/usr/bin/env python3
"""Build Zheng-2018-style word / POS embeddings from PubTator (+ CDR).

The 2018 paper trained word2vec on ~22 GB of PubTator bioconcepts *offsets*
text (titles/abstracts with chemical/disease mentions) plus CDR, then POS2vec
on a random 1/5 of that text.

NCBI no longer hosts ``bioconcepts2pubtator_offsets.gz``. The current equivalent
with article text is PubTator3 BioCXML:

  https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/BioCXML.0.tar.gz

This script streams that archive (no full untar), replaces Chemical/Disease
mentions with MeSH tokens, writes sentence files, trains gensim Word2Vec, and
optionally POS-tags a 1/5 subsample with Stanza.

Examples
--------
# Download BioCXML.0 (~18 GB) then train (long-running):
python scripts/build_pubtator_embeddings.py --download

# Use an already-downloaded archive:
python scripts/build_pubtator_embeddings.py \\
  --bioc-tar /path/to/BioCXML.0.tar.gz

# Faster smoke test (first N BioC docs):
python scripts/build_pubtator_embeddings.py --download --max-docs 5000
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import random
import re
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "embeddings"
DEFAULT_TAR = EMB_DIR / "BioCXML.0.tar.gz"
DEFAULT_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/BioCXML.0.tar.gz"
CDR_DIR = ROOT / "Dataset" / "CDR_Data" / "CDR_Data" / "CDR.Corpus.v010516"

WORD_SENTS = EMB_DIR / "pubtator_word_sents.txt"
POS_SENTS = EMB_DIR / "pubtator_pos_sents.txt"
WORD_MODEL = EMB_DIR / "w2v_word.model"
POS_MODEL = EMB_DIR / "w2v_pos.model"

DIGIT_RE = re.compile(r"[\d.,]*\d[\d.,]*")
TOKEN_RE = re.compile(r"[A-Za-z_<>][A-Za-z0-9_<>]*|[^\s]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_token(tok: str) -> str:
    t = tok.lower()
    if DIGIT_RE.fullmatch(t):
        return "<num>"
    return t


def tokenize(text: str) -> list[str]:
    return [normalize_token(t) for t in TOKEN_RE.findall(text) if t.strip()]


def mesh_token(entity_type: str, identifier: str) -> str | None:
    if not identifier or identifier in {"-", "None", "null"}:
        return None
    ident = identifier.split(";")[0].split("|")[0].strip()
    if ident.upper().startswith("MESH:"):
        ident = ident.split(":", 1)[1]
    # Keep Chemical / Disease only (paper CID focus).
    if entity_type == "Chemical":
        return "C_" + ident
    if entity_type == "Disease":
        return "D_" + ident
    return None


def replace_entities(text: str, anns: list[tuple[int, int, str]]) -> str:
    """Replace annotation spans (start, end, replacement) in *text*."""
    if not anns:
        return text
    parts: list[str] = []
    last = 0
    for start, end, repl in sorted(anns, key=lambda x: (x[0], -(x[1] - x[0]))):
        if start < last or end > len(text) or start >= end:
            continue
        parts.append(text[last:start])
        parts.append(repl)
        last = end
    parts.append(text[last:])
    return "".join(parts)


def passage_sentences(text: str, anns: list[tuple[int, int, str]]) -> list[list[str]]:
    marked = replace_entities(text, anns)
    chunks = SENT_SPLIT_RE.split(marked) if marked.strip() else []
    out = []
    for chunk in chunks:
        toks = tokenize(chunk)
        if toks:
            out.append(toks)
    return out


def iter_bioc_xml_bytes(xml_bytes: bytes):
    """Yield (title_or_abstract_text, annotations) for title/abstract passages."""
    # BioC files may contain a <collection> of documents or a single <document>.
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return
    docs = root.findall(".//document")
    if root.tag.endswith("document"):
        docs = [root]
    for doc in docs:
        for passage in doc.findall("passage"):
            ptype = None
            for infon in passage.findall("infon"):
                if infon.get("key") == "type":
                    ptype = (infon.text or "").strip().lower()
            if ptype not in {"title", "abstract"}:
                continue
            text_el = passage.find("text")
            if text_el is None or not (text_el.text or "").strip():
                continue
            text = text_el.text
            off_el = passage.find("offset")
            try:
                base = int(off_el.text) if off_el is not None and off_el.text else 0
            except ValueError:
                base = 0
            anns: list[tuple[int, int, str]] = []
            for ann in passage.findall("annotation"):
                etype = ident = None
                for infon in ann.findall("infon"):
                    k = infon.get("key")
                    if k == "type":
                        etype = (infon.text or "").strip()
                    elif k == "identifier":
                        ident = (infon.text or "").strip()
                repl = mesh_token(etype or "", ident or "")
                if not repl:
                    continue
                for loc in ann.findall("location"):
                    try:
                        abs_off = int(loc.get("offset", "0"))
                        length = int(loc.get("length", "0"))
                    except ValueError:
                        continue
                    start = abs_off - base
                    end = start + length
                    if 0 <= start < end <= len(text):
                        anns.append((start, end, repl))
            yield text, anns


def stream_bioc_tar(tar_path: Path, max_docs: int | None = None):
    """Stream title/abstract sentences from a BioCXML.*.tar.gz archive."""
    n_docs = 0
    n_sents = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith((".xml", ".XML")):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            # One BioC file may hold many documents.
            before = n_docs
            for text, anns in iter_bioc_xml_bytes(raw):
                sents = passage_sentences(text, anns)
                if not sents:
                    continue
                n_docs += 1
                for s in sents:
                    n_sents += 1
                    yield s
                if max_docs is not None and n_docs >= max_docs:
                    print(f"[bioc] stop at max-docs={max_docs} ({n_sents} sents)", flush=True)
                    return
            if n_docs != before and n_docs % 5000 == 0:
                print(f"[bioc] docs={n_docs} sents={n_sents} file={member.name}", flush=True)
    print(f"[bioc] done docs≈{n_docs} sents={n_sents}", flush=True)


def _flush_cdr_doc(title, abstract, entities):
    """Yield tokenized sentences for one CDR PubTator document."""
    if title is None and abstract is None:
        return
    # CDR character offsets are into title+abstract concatenated (no separator).
    joined = (title or "") + (abstract or "")
    anns = []
    for start, end, etype, eid in entities:
        repl = mesh_token(etype, eid)
        if repl and 0 <= start < end <= len(joined):
            anns.append((start, end, repl))
    marked = replace_entities(joined, anns) if anns else joined
    for s in SENT_SPLIT_RE.split(marked):
        toks = tokenize(s)
        if toks:
            yield toks


def iter_cdr_pubtator_sentences(cdr_dir: Path):
    """Tokenize CDR PubTator title/abstracts with MeSH entity replacement."""
    files = [
        cdr_dir / "CDR_TrainingSet.PubTator.txt",
        cdr_dir / "CDR_DevelopmentSet.PubTator.txt",
        cdr_dir / "CDR_TestSet.PubTator.txt",
    ]
    for path in files:
        if not path.is_file():
            print(f"[cdr] missing {path}", flush=True)
            continue
        title = abstract = None
        entities = []  # (start, end, type, id)
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    yield from _flush_cdr_doc(title, abstract, entities)
                    title = abstract = None
                    entities = []
                    continue
                if re.match(r"^\d+\|t\|", line):
                    title = line.split("|", 2)[2]
                elif re.match(r"^\d+\|a\|", line):
                    abstract = line.split("|", 2)[2]
                else:
                    parts = line.split("\t")
                    if len(parts) >= 6 and parts[1] != "CID":
                        try:
                            start, end = int(parts[1]), int(parts[2])
                        except ValueError:
                            continue
                        etype, mesh = parts[4], parts[5]
                        eid = mesh.split("|")[0]
                        if eid and eid != "-1":
                            entities.append((start, end, etype, eid))
            yield from _flush_cdr_doc(title, abstract, entities)


def download_file(url: str, dest: Path, resume: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {}
    mode = "wb"
    if resume and existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[download] resume from {existing:,} bytes → {dest}", flush=True)
    else:
        print(f"[download] fetching {url}", flush=True)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, mode) as out:
        # If server ignored Range and sent full file, restart.
        if mode == "ab" and resp.status == 200 and existing > 0:
            out.close()
            dest.unlink()
            return download_file(url, dest, resume=False)
        total = resp.headers.get("Content-Length")
        total_i = int(total) + (existing if resp.status == 206 else 0) if total else None
        got = existing
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            if total_i:
                pct = 100.0 * got / total_i
                print(f"\r[download] {got/1e9:.2f}/{total_i/1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r[download] {got/1e9:.2f} GB", end="", flush=True)
    print(f"\n[download] saved {dest} ({dest.stat().st_size:,} bytes)", flush=True)


def write_sentence_corpus(tar_path: Path, out_path: Path, max_docs: int | None, include_cdr: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for sent in stream_bioc_tar(tar_path, max_docs=max_docs):
            out.write(" ".join(sent) + "\n")
            n += 1
        if include_cdr:
            print("[cdr] appending CDR sentences", flush=True)
            for sent in iter_cdr_pubtator_sentences(CDR_DIR):
                out.write(" ".join(sent) + "\n")
                n += 1
    print(f"[corpus] wrote {n:,} sentences → {out_path}", flush=True)
    return n


def train_word2vec(sents_path: Path, model_path: Path, dim: int, epochs: int, workers: int) -> None:
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    print(f"[w2v] training dim={dim} epochs={epochs} on {sents_path}", flush=True)
    model = Word2Vec(
        sentences=LineSentence(str(sents_path)),
        vector_size=dim,
        window=5,
        min_count=5,
        workers=workers,
        epochs=epochs,
        seed=42,
        sg=0,  # CBOW (word2vec default tool style)
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"[w2v] saved {model_path} | vocab={len(model.wv):,}", flush=True)


def build_pos_corpus(word_sents_path: Path, pos_sents_path: Path, fraction: float, max_sents: int, seed: int) -> int:
    """Tag a random fraction of word sentences with Stanza POS (paper: ~1/5)."""
    import stanza

    random.seed(seed)
    print("[pos] loading Stanza tokenize+pos (English)...", flush=True)
    stanza.download("en", processors="tokenize,pos", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos",
        tokenize_pretokenized=True,
        verbose=False,
    )

    # Reservoir / two-pass: first count, then sample line numbers, then tag.
    with word_sents_path.open(encoding="utf-8") as f:
        total = sum(1 for _ in f)
    target = min(max_sents, max(1, int(total * fraction)))
    chosen = set(random.sample(range(total), k=min(target, total)))
    print(f"[pos] tagging {len(chosen):,} / {total:,} sentences (fraction≈{fraction})", flush=True)

    n = 0
    with word_sents_path.open(encoding="utf-8") as inp, pos_sents_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(inp):
            if i not in chosen:
                continue
            toks = line.split()
            if not toks:
                continue
            doc = nlp([toks])  # pretokenized
            tags = []
            for sent in doc.sentences:
                for w in sent.words:
                    tags.append(w.xpos or w.upos or "NN")
            if tags:
                out.write(" ".join(tags) + "\n")
                n += 1
            if n % 2000 == 0:
                print(f"[pos] tagged {n:,}", flush=True)
    print(f"[pos] wrote {n:,} tag sequences → {pos_sents_path}", flush=True)
    return n


def train_pos2vec(pos_sents_path: Path, model_path: Path, dim: int, epochs: int, workers: int) -> None:
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    print(f"[pos2v] training dim={dim} on {pos_sents_path}", flush=True)
    model = Word2Vec(
        sentences=LineSentence(str(pos_sents_path)),
        vector_size=dim,
        window=5,
        min_count=1,
        workers=workers,
        epochs=epochs,
        seed=42,
        sg=0,
    )
    model.save(str(model_path))
    print(f"[pos2v] saved {model_path} | vocab={len(model.wv):,}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help=f"Download {DEFAULT_URL}")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--bioc-tar", type=Path, default=DEFAULT_TAR, help="Local BioCXML.*.tar.gz")
    p.add_argument("--max-docs", type=int, default=None, help="Cap BioC documents (debug / partial)")
    p.add_argument("--skip-cdr", action="store_true", help="Do not append CDR texts")
    p.add_argument("--word-dim", type=int, default=100)
    p.add_argument("--pos-dim", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5, help="word2vec epochs (paper used word2vec tool defaults)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--skip-pos", action="store_true", help="Only train word embeddings")
    p.add_argument("--pos-fraction", type=float, default=0.2, help="Paper uses ~1/5 of texts")
    p.add_argument("--pos-max-sents", type=int, default=500_000, help="Safety cap for Stanza POS")
    p.add_argument("--corpus-only", action="store_true", help="Write sentence file only")
    args = p.parse_args(argv)

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    if args.download or not args.bioc_tar.exists():
        if not args.bioc_tar.exists() and not args.download:
            print(f"[warn] {args.bioc_tar} missing — starting download", flush=True)
        download_file(args.url, args.bioc_tar)

    if not args.bioc_tar.is_file():
        print(f"ERROR: BioC archive not found: {args.bioc_tar}", file=sys.stderr)
        return 1

    if not WORD_SENTS.exists() or args.max_docs is not None or args.download:
        write_sentence_corpus(
            args.bioc_tar,
            WORD_SENTS,
            max_docs=args.max_docs,
            include_cdr=not args.skip_cdr,
        )
    else:
        print(f"[corpus] reusing existing {WORD_SENTS}", flush=True)

    if args.corpus_only:
        return 0

    train_word2vec(WORD_SENTS, WORD_MODEL, dim=args.word_dim, epochs=args.epochs, workers=args.workers)

    if not args.skip_pos:
        if not POS_SENTS.exists():
            build_pos_corpus(
                WORD_SENTS,
                POS_SENTS,
                fraction=args.pos_fraction,
                max_sents=args.pos_max_sents,
                seed=42,
            )
        else:
            print(f"[pos] reusing existing {POS_SENTS}", flush=True)
        train_pos2vec(POS_SENTS, POS_MODEL, dim=args.pos_dim, epochs=args.epochs, workers=args.workers)

    print("[done] load these in the notebook:", WORD_MODEL, POS_MODEL, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
