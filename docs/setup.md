# Environment setup

Python 3.11. From the repository root (`export ITER_ROOT=$(pwd)`):

```bash
pip install uv && uv venv envs && source envs/bin/activate
uv pip install vllm transformers accelerate faiss-cpu peft matplotlib
uv pip install -e third_party/FlagEmbedding   # retriever training (tier-weight mods)
uv pip install -e third_party/tevatron        # corpus encoding
# flash-attn: install the wheel matching your torch/CUDA, or set
# FAISS_ATTN_IMPL=sdpa to run without it.
# BM25 indexing additionally needs Java 21+ (pyserini).
```

Agent backbones are served locally with vLLM inside each eval script; download
the models into `$HF_HOME` beforehand (scripts run with HF_HUB_OFFLINE=1).
