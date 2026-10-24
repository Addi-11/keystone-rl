#!/usr/bin/env bash
# One-time environment setup on PACE. Run from the repo root on a login node:
#   bash scripts/pace_setup.sh
# Everything big (env, HF cache) goes to scratch, not home, to stay under quota.
set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
ENV_DIR="$SCRATCH_DIR/envs/agentmismatch"
export HF_HOME="$SCRATCH_DIR/hf_cache"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" python=3.11
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

pip install --upgrade pip
pip install -e ".[dev]"
pip install -e ".[gpu]"   # vllm pulls a matching torch; the CUDA 12/13 drivers on PACE run its wheels
pip freeze > requirements.lock

python -m pytest -q
echo
echo "Add these to ~/.bashrc so every job sees them:"
echo "  export HF_HOME=$HF_HOME"
echo "  alias amis='module load anaconda3; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $ENV_DIR'"
