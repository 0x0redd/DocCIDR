# Environment: `dl`

Python **3.11** virtualenv for DocCIDR (Zheng 2018 CID).

## Activate

```bash
cd ~/Documents/DocCIDR
source dl/bin/activate
```

In Cursor/Jupyter: select kernel **Python (dl)**.

## Recreate (if needed)

```bash
cd ~/Documents/DocCIDR
uv venv --python ~/.local/bin/python3.11 dl
export TMPDIR="$PWD/.tmp" UV_CACHE_DIR="$HOME/.cache/uv"
mkdir -p "$TMPDIR"
uv pip install --python dl/bin/python -r requirment.txt
# CPU torch so stanza does not pull a second CUDA stack:
uv pip install --python dl/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python dl/bin/python stanza
python -m ipykernel install --user --name=dl --display-name="Python (dl)"
```

## Check GPU

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Expected: `TF 2.21.0` and a non-empty GPU list.

## PubTator embeddings (paper §3.2)

NCBI retired `bioconcepts2pubtator_offsets.gz`. Build from PubTator3 BioCXML + CDR:

```bash
source dl/bin/activate
python scripts/build_pubtator_embeddings.py --download
```

Writes `embeddings/w2v_word.model` and `embeddings/w2v_pos.model` for the notebook.
Progress: `tail -f embeddings/build.log`.
