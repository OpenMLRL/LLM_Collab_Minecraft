#!/usr/bin/env bash
set -euo pipefail

# Copy this file to `run.sh` and fill in your own Slurm/conda settings:
#   cp baselines/run.example.sh baselines/run.sh
#   vim baselines/run.sh
#
# Then submit:
#   bash baselines/run.sh baselines/config.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"

# Slurm resources (set these for your cluster)
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NTASKS="${NTASKS:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEM="${MEM:-100g}"
TIME="${TIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-llm_collab_mc_baseline}"

# Conda env
CONDA_ENV="${CONDA_ENV:-comlrl}"
BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCRIPT_DIR}/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

if [[ -z "${ACCOUNT}" || -z "${PARTITION}" ]]; then
  echo "ERROR: set ACCOUNT and PARTITION (env vars) for sbatch." >&2
  echo "Example:" >&2
  echo "  ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION bash baselines/run.sh baselines/config.yaml" >&2
  exit 2
fi

EXTRA_ARGS="${EXTRA_ARGS:-}"

WRAP_CMD="bash -lc 'set -eo pipefail; \
if [ -f \"${BASHRC_PATH}\" ]; then source \"${BASHRC_PATH}\"; fi; \
if ! command -v conda >/dev/null 2>&1; then echo \"ERROR: conda not found (set BASHRC_PATH or load conda module)\" >&2; exit 1; fi; \
conda activate \"${CONDA_ENV}\"; \
cd \"${REPO_DIR}\"; \
python3 -u baselines/main.py --config \"${CONFIG_PATH}\" ${EXTRA_ARGS}'"

echo "Submitting Slurm job:"
echo "  config: ${CONFIG_PATH}"
echo "  wrap:   ${WRAP_CMD}"

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes="${NODES}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --ntasks="${NTASKS}" \
  --ntasks-per-node="${NTASKS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --job-name="${JOB_NAME}" \
  --output="${SLURM_LOG_DIR}/%x-%j.out" \
  --wrap="${WRAP_CMD}"

