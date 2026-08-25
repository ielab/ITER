#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu
#SBATCH --job-name=diver_warm_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build the shared corpus Arrow cache ONCE (CPU, no GPU). After this, every encode
# task just memory-maps the cache — no per-task "Generating train split". Uses the
# exact load_dataset signature tevatron uses ('json', split='train', same cache
# dir), so the fingerprints match. The encode array depends on this job, then all
# shards run concurrently. Reused by every model; only needed once per corpus.

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
CORPUS_PATH=${CORPUS_PATH:-${ROOT}/data/corpus.jsonl}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${ROOT}/data/hf_cache}
mkdir -p "${ROOT}/logs/indexing" "${HF_DATASETS_CACHE}"

module load miniforge3 2>/dev/null || true
module load cuda 2>/dev/null || true   # CPU partition (cpu) has no cuda module; not needed here
source activate "${CONDA_ENV}"
cd "${ROOT}"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

python - "${CORPUS_PATH}" <<'PY'
import sys
from datasets import load_dataset
ds = load_dataset("json", data_files=sys.argv[1], split="train")
print(f"cache warmed: {len(ds):,} rows", flush=True)
PY

echo "===== corpus cache ready at ${HF_DATASETS_CACHE} ====="
