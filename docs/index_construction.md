# Index Construction

Every retriever ITER evaluates needs its own index, because the index stores
document embeddings produced by that specific encoder. Two corpora are used:
Wikipedia (11.2M chunks, HNSW) and BrowseComp-Plus (100,195 documents, flat).

## Corpus preparation

`wiki-25-512` ships columns `{id, contents}`; every stage here reads
`{docid, text}`.

```bash
python src/prepare_corpus.py \
  --dataset Lk123/wiki-25-512 \
  --output /path/to/data/corpus.jsonl
```

## BM25

Used only for trajectory collection, as one of the four retrieval backends.
Requires Java 21+ on `PATH`.

```bash
python src/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path /path/to/corpus.jsonl \
  --save_dir /path/to/index/bm25
```

## Dense embeddings

| Argument | Meaning |
| --- | --- |
| `--model_name_or_path` | the encoder whose index you are building |
| `--dataset_path` | corpus `.jsonl` with `{docid, text}` |
| `--encode_output_path` | output shard, named `index-<NNN>.pkl` |
| `--passage_max_len` | 512 — must match the retriever's training value |
| `--pooling` | `eos` |
| `--normalize` | on, for all ITER retrievers |
| `--dataset_number_of_shards` / `--dataset_shard_index` | split Wikipedia across parallel encoders |

BrowseComp-Plus fits in a single shard:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path /path/to/your/retriever \
  --dataset_path /path/to/browse-comp-plus-corpus.jsonl \
  --encode_output_path /path/to/index/bcp/index-000.pkl \
  --passage_max_len 512 \
  --normalize --pooling eos --passage_prefix "" \
  --per_device_eval_batch_size <BATCH_SIZE> \
  --padding_side left --fp16
```

Wikipedia is encoded in shards; run this once per shard, varying
`--dataset_shard_index`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path /path/to/your/retriever \
  --dataset_path /path/to/corpus.jsonl \
  --encode_output_path /path/to/index/wiki/index-<SHARD>.pkl \
  --dataset_number_of_shards <NUM_SHARDS> \
  --dataset_shard_index <SHARD> \
  --passage_max_len 512 \
  --normalize --pooling eos --passage_prefix "" \
  --per_device_eval_batch_size <BATCH_SIZE> \
  --padding_side left --fp16
```

## HNSW graph

The flat path loads every vector into RAM, which is not viable at 11.2M
documents. Build the graph once all Wikipedia shards exist (CPU only):

```bash
python src/build_ann_index.py \
  --shards "/path/to/index/wiki/index-*.pkl" \
  --out-dir /path/to/index/wiki \
  --index-type HNSW32 \
  --ef-construction 200 \
  --ef-search 256 \
  --retriever /path/to/your/retriever
```

This writes `index.faiss` and `index.lookup.pkl`. Point evaluation at
`index.faiss` for Wikipedia, and at `index-000.pkl` for BrowseComp-Plus.

## Record the encoder

An index only means anything with the encoder that built it, at the same
passage length and pooling. A mismatch raises nothing — retrieval just gets
worse — so stamp every index directory:

```bash
python src/index_meta.py \
  --index-dir /path/to/index/i2_bcp \
  --retriever /path/to/your/retriever
```

`src/build_ann_index.py --retriever <...>` does this for you. `run_eval.py`
reads the resulting `encoder.json` and refuses to start an arm whose index was
built by a different encoder, or with a `passage_max_len` / `pooling` that no
longer matches `config.INDEXING`.
