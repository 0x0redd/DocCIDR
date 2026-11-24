#!/usr/bin/env python3
"""Retrain Zheng CID model with recall-oriented settings (class weights, longer train, thr sweep)."""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
CDR_DIR = ROOT / "Dataset" / "CDR_Data" / "CDR_Data" / "CDR.Corpus.v010516"
CORENLP_DIR = ROOT / "stanford-corenlp-4.6.0"
EXTRA_MODELS = ROOT / "corenlp-english-extra" / "stanford-corenlp-models-english-extra.jar"
EMB_DIR = ROOT / "embeddings"
CACHE = EMB_DIR / "prepared_docs.pkl"
OUT_DIR = EMB_DIR / "cid_model_recall"

K_WINDOW, MAX_SENTS, MAX_WORDS = 4, 30, 120
WORD_DIM, POS_DIM = 100, 10
CNN_WINDOW, CNN_FILTERS = 5, 300
DROPOUT, LEARNING_RATE = 0.5, 0.001
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 4, 40, 8
THRESH_GRID = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]


def setup_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("memory_growth:", e)
    print("GPU:", tf.config.list_physical_devices("GPU"))


def read_pubtator(path: Path):
    docs, doc = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if line == "":
            if doc:
                docs.append(doc)
            doc = None
        elif re.match(r"^\d+\|t\|", line):
            pmid, _, title = line.split("|", 2)
            doc = {"pmid": pmid, "title": title, "abstract": "", "entities": [], "cids": set()}
        elif re.match(r"^\d+\|a\|", line):
            doc["abstract"] = line.split("|", 2)[2]
        else:
            parts = line.split("\t")
            if len(parts) < 2 or doc is None:
                continue
            if parts[1] == "CID":
                doc["cids"].add((parts[2], parts[3]))
            else:
                mesh = parts[5] if len(parts) > 5 else ""
                ids = [i for i in mesh.split("|") if i and i != "-1"]
                if ids:
                    doc["entities"].append({
                        "start": int(parts[1]), "end": int(parts[2]),
                        "text": parts[3], "type": parts[4], "ids": ids,
                    })
    if doc:
        docs.append(doc)
    return docs


class CoreNLPBackend:
    def __init__(self, home: Path):
        from stanfordcorenlp import StanfordCoreNLP
        if EXTRA_MODELS.is_file():
            dest = home / EXTRA_MODELS.name
            if not dest.exists():
                try:
                    shutil.copy2(EXTRA_MODELS, dest)
                except Exception:
                    pass
        self.nlp = StanfordCoreNLP(str(home), memory="4g", timeout=120000)
        self.props = {
            "annotators": "tokenize,ssplit,pos",
            "outputFormat": "json",
            "tokenize.options": "ptb3Escaping=false",
        }

    def annotate(self, text):
        raw = self.nlp.annotate(text, properties=self.props)
        data = json.loads(raw)
        sentences = []
        for sent in data.get("sentences", []):
            tokens = []
            for t in sent.get("tokens", []):
                tokens.append({
                    "word": t.get("originalText") or t.get("word"),
                    "pos": t.get("pos", "NN"),
                    "start": int(t["characterOffsetBegin"]),
                    "end": int(t["characterOffsetEnd"]),
                })
            sentences.append(tokens)
        return sentences

    def close(self):
        try:
            self.nlp.close()
        except Exception:
            pass


def normalize_word(w):
    w = w.lower()
    if re.fullmatch(r"[\d.,]*\d[\d.,]*", w):
        return "<num>"
    return w


def merge_entities(abs_tokens, entities):
    merged, i = [], 0
    while i < len(abs_tokens):
        t = abs_tokens[i]
        hit = None
        if t["start"] >= 0:
            for ent in entities:
                if ent["start"] < t["end"] and ent["end"] > t["start"]:
                    hit = ent
                    break
        if hit is None:
            merged.append({"word": normalize_word(t["word"]), "pos": t["pos"]})
            i += 1
            continue
        j = i
        while j < len(abs_tokens):
            tj = abs_tokens[j]
            if tj["start"] < 0 or not (hit["start"] < tj["end"] and hit["end"] > tj["start"]):
                break
            j += 1
        merged.append({"word": None, "type": hit["type"], "ids": hit["ids"], "pos": "NN"})
        i = max(j, i + 1)
    return merged


def prepare_document(doc, nlp):
    entities = sorted(doc["entities"], key=lambda e: (e["start"], e["end"]))

    def annotate_side(text, base_offset):
        if not text:
            return []
        out = []
        for sent in nlp.annotate(text):
            abs_tokens = []
            for t in sent:
                if t["start"] >= 0:
                    abs_tokens.append({
                        "word": t["word"], "pos": t["pos"],
                        "start": base_offset + t["start"], "end": base_offset + t["end"],
                    })
                else:
                    abs_tokens.append({"word": t["word"], "pos": t["pos"], "start": -1, "end": -1})
            out.append(merge_entities(abs_tokens, entities))
        return out

    title_parts = annotate_side(doc["title"], 0)
    if not title_parts:
        title_tokens = [{"word": normalize_word(w), "pos": "NN"} for w in doc["title"].split()]
        title_parts = [title_tokens or [{"word": "<pad>", "pos": "NN"}]]
    title_sent = [t for s in title_parts for t in s]
    abs_sents = annotate_side(doc["abstract"], len(doc["title"]) + 1)
    doc["sentences"] = [title_sent] + abs_sents
    doc["sent_of"] = defaultdict(set)
    for si, sent in enumerate(doc["sentences"]):
        for t in sent:
            if t.get("word") is None:
                for eid in t["ids"]:
                    doc["sent_of"][eid].add(si)
    return doc


def remove_brackets(words, tags):
    out_w, out_t, buffer = [], [], None
    for w, t in zip(words, tags):
        if buffer is None and w in ("(", "["):
            buffer = [(w, t)]
        elif buffer is not None:
            buffer.append((w, t))
            if w in (")", "]"):
                if any(x in ("ch_start", "ds_start") for x, _ in buffer):
                    out_w += [x for x, _ in buffer]
                    out_t += [y for _, y in buffer]
                buffer = None
        else:
            out_w.append(w)
            out_t.append(t)
    if buffer:
        out_w += [x for x, _ in buffer]
        out_t += [y for _, y in buffer]
    return out_w, out_t


def render(doc, chem=None, dis=None):
    result = []
    for sentence in doc["sentences"]:
        words, tags = [], []
        for t in sentence:
            if t["word"] is not None:
                words.append(t["word"]); tags.append(t["pos"])
            elif chem is not None and t["type"] == "Chemical" and chem in t["ids"]:
                words += ["ch_start", "C_" + chem, "ch_end"]; tags += ["MARK", "NN", "MARK"]
            elif dis is not None and t["type"] == "Disease" and dis in t["ids"]:
                words += ["ds_start", "D_" + dis, "ds_end"]; tags += ["MARK", "NN", "MARK"]
            else:
                prefix = "C_" if t["type"] == "Chemical" else "D_"
                words.append(prefix + t["ids"][0]); tags.append("NN")
        if chem is not None or dis is not None:
            words, tags = remove_brackets(words, tags)
        result.append((words, tags))
    return result


def candidate_pairs(doc, use_k_filter):
    chems = {i for e in doc["entities"] if e["type"] == "Chemical" for i in e["ids"]}
    dises = {i for e in doc["entities"] if e["type"] == "Disease" for i in e["ids"]}
    pairs = []
    for c in sorted(chems):
        for d in sorted(dises):
            if c not in doc["sent_of"] or d not in doc["sent_of"]:
                continue
            if not doc["sent_of"][c] or not doc["sent_of"][d]:
                continue
            span = min(abs(i - j) for i in doc["sent_of"][c] for j in doc["sent_of"][d]) + 1
            if use_k_filter and span > K_WINDOW:
                continue
            pairs.append((c, d, int((c, d) in doc["cids"])))
    return pairs


def iter_cd_pairs(doc):
    chems = {i for e in doc["entities"] if e["type"] == "Chemical" for i in e["ids"]}
    dises = {i for e in doc["entities"] if e["type"] == "Disease" for i in e["ids"]}
    for c in sorted(chems):
        for d in sorted(dises):
            if not doc["sent_of"].get(c) or not doc["sent_of"].get(d):
                continue
            span = min(abs(i - j) for i in doc["sent_of"][c] for j in doc["sent_of"][d]) + 1
            yield c, d, int((c, d) in doc["cids"]), span


def corpus_stats(docs, apply_k):
    cid = original = kept = beyond = 0
    for doc in docs:
        for _c, _d, label, span in iter_cd_pairs(doc):
            original += 1
            cid += label
            if apply_k and span > K_WINDOW:
                if label:
                    beyond += 1
            else:
                kept += 1
    return cid, original, kept, beyond


def partition_train_dev(train_docs, dev_docs, seed=100, dev_subset=3):
    """§3.1: 10 equal article subsets of train∪dev; nine train, one dev."""
    union_docs = list(train_docs) + list(dev_docs)
    rng = random.Random(seed)
    rng.shuffle(union_docs)
    fold = len(union_docs) // 10
    subsets = [union_docs[i * fold:(i + 1) * fold] for i in range(10)]
    new_dev = subsets[dev_subset]
    new_train = [d for i, subset in enumerate(subsets) if i != dev_subset for d in subset]
    return new_train, new_dev


def make_instances(docs, use_k_filter):
    instances = []
    for doc in docs:
        for c, d, label in candidate_pairs(doc, use_k_filter):
            sents = render(doc, c, d)
            instances.append({
                "pmid": doc["pmid"], "chem": c, "dis": d, "label": label,
                "title": sents[0] if sents else ([], []),
                "abstract": sents[1:] if len(sents) > 1 else [],
            })
    return instances


def encode_sentence(items, vocab):
    ids = [vocab.get(x, 1) for x in items][:MAX_WORDS]
    return [0] * (MAX_WORDS - len(ids)) + ids


def encode(instances, word2idx, pos2idx):
    n = len(instances)
    doc_words = np.zeros((n, MAX_SENTS, MAX_WORDS), dtype="int32")
    doc_pos = np.zeros((n, MAX_SENTS, MAX_WORDS), dtype="int32")
    title_words = np.zeros((n, MAX_WORDS), dtype="int32")
    title_pos = np.zeros((n, MAX_WORDS), dtype="int32")
    labels = np.zeros(n, dtype="int32")
    for k, inst in enumerate(instances):
        sents = inst["abstract"][:MAX_SENTS]
        first_row = MAX_SENTS - len(sents)
        for s, (words, tags) in enumerate(sents):
            doc_words[k, first_row + s] = encode_sentence(words, word2idx)
            doc_pos[k, first_row + s] = encode_sentence(tags, pos2idx)
        tw, tt = inst["title"]
        title_words[k] = encode_sentence(tw, word2idx)
        title_pos[k] = encode_sentence(tt, pos2idx)
        labels[k] = inst["label"]
    return {
        "doc_words": doc_words, "doc_pos": doc_pos,
        "title_words": title_words, "title_pos": title_pos,
    }, labels


def matrix_from_w2v(model, vocab, dim):
    mat = np.random.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    mat[0] = 0
    hits = 0
    for tok, index in vocab.items():
        if tok in model.wv:
            mat[index] = model.wv[tok]
            hits += 1
    return mat, hits


def build_model(n_words, n_pos, emb_w, emb_p):
    keras.backend.clear_session()
    gc.collect()
    doc_words_in = keras.Input(shape=(MAX_SENTS, MAX_WORDS), dtype="int32", name="doc_words")
    doc_pos_in = keras.Input(shape=(MAX_SENTS, MAX_WORDS), dtype="int32", name="doc_pos")
    title_words_in = keras.Input(shape=(MAX_WORDS,), dtype="int32", name="title_words")
    title_pos_in = keras.Input(shape=(MAX_WORDS,), dtype="int32", name="title_pos")

    word_embedding = layers.Embedding(
        n_words, WORD_DIM, embeddings_initializer=keras.initializers.Constant(emb_w),
        trainable=True, name="word_embedding",
    )
    pos_embedding = layers.Embedding(
        n_pos, POS_DIM, embeddings_initializer=keras.initializers.Constant(emb_p),
        trainable=True, name="pos_embedding",
    )
    doc_x = layers.Concatenate()([word_embedding(doc_words_in), pos_embedding(doc_pos_in)])
    title_x = layers.Concatenate()([word_embedding(title_words_in), pos_embedding(title_pos_in)])

    bilstm_words = layers.Bidirectional(layers.LSTM(110), name="bilstm1_words")
    sentence_vectors = layers.TimeDistributed(bilstm_words)(doc_x)
    sentence_vectors = layers.Dropout(DROPOUT)(sentence_vectors)

    D_l = layers.Bidirectional(layers.LSTM(220), name="bilstm2")(sentence_vectors)
    D_l = layers.Dropout(DROPOUT)(D_l)
    conv = layers.Conv1D(CNN_FILTERS, CNN_WINDOW, activation="relu", padding="same")(sentence_vectors)
    D_c = layers.GlobalMaxPooling1D()(conv)
    D_t = layers.Bidirectional(layers.LSTM(110), name="bilstm3")(title_x)
    D_t = layers.Dropout(DROPOUT)(D_t)

    merged = layers.Concatenate()([D_l, D_c, D_t])
    merged = layers.Dropout(DROPOUT)(merged)
    output = layers.Dense(2, activation="softmax")(merged)
    model = keras.Model(
        inputs=[doc_words_in, doc_pos_in, title_words_in, title_pos_in],
        outputs=output,
    )
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prf(pred, gold_set):
    tp = len(pred & gold_set)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(100 * p, 1), round(100 * r, 1), round(100 * f, 1)


def predict_pairs(instances, probs, thr):
    return {
        (inst["pmid"], inst["chem"], inst["dis"])
        for inst, p in zip(instances, probs) if p >= thr
    }


def rule_1(pred, docs):
    new = set(pred)
    found = {pmid for pmid, _, _ in pred}
    for doc in docs:
        if doc["pmid"] in found:
            continue
        chem_count = Counter(eid for e in doc["entities"] if e["type"] == "Chemical" for eid in e["ids"])
        dis_count = Counter(eid for e in doc["entities"] if e["type"] == "Disease" for eid in e["ids"])
        if chem_count and dis_count:
            new.add((doc["pmid"], chem_count.most_common(1)[0][0], dis_count.most_common(1)[0][0]))
    return new


def load_or_prepare():
    if CACHE.is_file():
        print("Loading cached prepared docs:", CACHE)
        with CACHE.open("rb") as f:
            return pickle.load(f)

    train_docs = read_pubtator(CDR_DIR / "CDR_TrainingSet.PubTator.txt")
    dev_docs = read_pubtator(CDR_DIR / "CDR_DevelopmentSet.PubTator.txt")
    test_docs = read_pubtator(CDR_DIR / "CDR_TestSet.PubTator.txt")
    print(len(train_docs), len(dev_docs), len(test_docs), "docs")

    nlp = CoreNLPBackend(CORENLP_DIR)
    try:
        for name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
            for i, doc in enumerate(docs, 1):
                prepare_document(doc, nlp)
                if i % 100 == 0:
                    print(f"  {name}: {i}/{len(docs)}")
            print(f"{name} prepared ({len(docs)})")
    finally:
        nlp.close()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump({"train": train_docs, "dev": dev_docs, "test": test_docs}, f, protocol=4)
    print("Cached to", CACHE)
    return {"train": train_docs, "dev": dev_docs, "test": test_docs}


def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    setup_gpu()

    data = load_or_prepare()
    train_docs, dev_docs, test_docs = data["train"], data["dev"], data["test"]

    # §3.1: article-level 10-subset split. Seed matches notebook Table 2 CID counts.
    new_train_docs, new_dev_docs = partition_train_dev(train_docs, dev_docs, seed=100, dev_subset=3)
    train_instances = make_instances(new_train_docs, use_k_filter=False)
    val_instances = make_instances(new_dev_docs, use_k_filter=True)
    test_instances = make_instances(test_docs, use_k_filter=True)
    for name, docs, apply_k in (
        ("train", new_train_docs, False),
        ("dev", new_dev_docs, True),
        ("test", test_docs, True),
    ):
        cid, original, kept, beyond = corpus_stats(docs, apply_k)
        fn = "n/a (no K filter)" if not apply_k else str(beyond)
        print(f"Table2 {name}: CID={cid} originalCD={original} afterK={kept} CID>K={fn}")

    word_counts, pos_set = Counter(), set()
    for inst in train_instances:
        for words, tags in [inst["title"]] + inst["abstract"]:
            word_counts.update(words)
            pos_set.update(tags)
    word2idx = {"<pad>": 0, "<unk>": 1}
    for w, _ in word_counts.most_common():
        word2idx[w] = len(word2idx)
    pos2idx = {"<pad>": 0, "<unk>": 1}
    for p in sorted(pos_set):
        pos2idx[p] = len(pos2idx)
    print(len(train_instances), "train /", len(val_instances), "dev |", len(word2idx), "words")

    w2v = Word2Vec.load(str(EMB_DIR / "w2v_word.model"))
    emb_w, wh = matrix_from_w2v(w2v, word2idx, WORD_DIM)
    if (EMB_DIR / "w2v_pos.model").is_file():
        pos2v = Word2Vec.load(str(EMB_DIR / "w2v_pos.model"))
        emb_p, ph = matrix_from_w2v(pos2v, pos2idx, POS_DIM)
    else:
        pos_sents = []
        for doc in train_docs + dev_docs + test_docs:
            for _, tags in render(doc, None, None):
                if tags:
                    pos_sents.append(tags)
        pos2v = Word2Vec(pos_sents, vector_size=POS_DIM, window=5, min_count=1, epochs=20, seed=42)
        emb_p, ph = matrix_from_w2v(pos2v, pos2idx, POS_DIM)
    print(f"emb hits word={wh}/{len(word2idx)} pos={ph}/{len(pos2idx)}")

    X_train, y_train = encode(train_instances, word2idx, pos2idx)
    X_dev, y_dev = encode(val_instances, word2idx, pos2idx)
    X_test, y_test = encode(test_instances, word2idx, pos2idx)

    n_neg, n_pos = np.bincount(y_train, minlength=2)
    class_weight = {
        0: float((n_neg + n_pos) / (2.0 * max(n_neg, 1))),
        1: float((n_neg + n_pos) / (2.0 * max(n_pos, 1))),
    }
    print("class_weight", class_weight, "pos_rate", float(y_train.mean()))

    model = build_model(len(word2idx), len(pos2idx), emb_w, emb_p)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, PATIENCE // 3),
            min_lr=1e-5, verbose=1,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    print("epochs run:", len(history.history["loss"]))

    dev_probs = model.predict(X_dev, batch_size=32)[:, 1]
    test_probs = model.predict(X_test, batch_size=32)[:, 1]
    # §3.1: relations with span > K stay in gold and count as false negatives.
    gold_dev = {(d["pmid"], c, s) for d in new_dev_docs for (c, s) in d["cids"]}
    gold = {(d["pmid"], c, s) for d in test_docs for (c, s) in d["cids"]}

    print("Dev threshold sweep:")
    best = None
    for thr in THRESH_GRID:
        scores = prf(predict_pairs(val_instances, dev_probs, thr), gold_dev)
        print(f"  thr={thr:.2f} P/R/F={scores}")
        key = (scores[2], scores[1], -thr)
        if best is None or key > best[0]:
            best = (key, thr, scores)
    thr = best[1]
    print("Selected thr", thr, best[2])

    predicted = predict_pairs(test_instances, test_probs, thr)
    print("Test @thr no-PP:", prf(predicted, gold), "n=", len(predicted))
    print("Test @0.50 no-PP:", prf(predict_pairs(test_instances, test_probs, 0.5), gold))
    pred_r1 = rule_1(predicted, test_docs)
    print("Test +rule1:", prf(pred_r1, gold))
    print("Paper Table5 no-PP: (54.3, 65.9, 59.5) | Table7+pp: (56.2, 68.0, 61.5)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "model.keras")
    meta = {"threshold": thr, "class_weight": class_weight, "epochs": len(history.history["loss"])}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved", OUT_DIR)


if __name__ == "__main__":
    main()
