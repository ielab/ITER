#!/bin/bash
set -euo pipefail

# Builds a BM25 (Lucene/Pyserini) index from corpus.jsonl.
# CORPUS_PATH: path to corpus JSONL (default: data/corpus.jsonl)
# INDEX_DIR:   output directory    (default: data/indexes/bm25)

ROOT=/scratch/user/uqdche12/ITER
CORPUS_PATH=${CORPUS_PATH:-${ROOT}/data/corpus.jsonl}
INDEX_DIR=${INDEX_DIR:-${ROOT}/data/indexes/bm25}

source /usr/share/lmod/lmod/init/bash
module load java/21.0.8
source "${ROOT}/.venv/bin/activate"
cd "${ROOT}"

python src/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path "${CORPUS_PATH}" \
  --save_dir "${INDEX_DIR}"

echo "===== BM25 index done -> ${INDEX_DIR} ====="
