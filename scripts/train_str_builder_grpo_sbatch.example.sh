#!/usr/bin/env bash
set -euo pipefail

# Example SLURM wrapper for str_builder GRPO training (CoMLRL MAGRPOTrainer with num_agents=1).
#
# Usage:
#   ACCOUNT=... PARTITION=... ./scripts/train_str_builder_grpo_sbatch.sh
#
# You can override defaults via *_OVERRIDE env vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_REL="configs/str_builder_config.yaml"
TRAIN_REL="train/train.py"

# SLURM defaults (override via env)
ACCOUNT="${ACCOUNT_OVERRIDE:-${ACCOUNT:-}}"
PARTITION="${PARTITION_OVERRIDE:-${PARTITION:-}}"
GPUS="${GPUS_OVERRIDE:-1}"
CPUS="${CPUS_OVERRIDE:-64}"
MEM="${MEM_OVERRIDE:-100g}"
TIME="${TIME_OVERRIDE:-24:00:00}"

if [[ -z "${ACCOUNT}" || -z "${PARTITION}" ]]; then
  echo "ERROR: set ACCOUNT and PARTITION (env vars) for sbatch." >&2
  echo "Example:" >&2
  echo "  ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./scripts/train_str_builder_grpo_sbatch.sh" >&2
  exit 2
fi

# Output dir defaults (YAML placeholders are TODO; set real paths here)
OUTPUT_BASE_DEFAULT="/projects/bevi/${USER}/output/str_builder/[jobid]"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-$OUTPUT_BASE_DEFAULT}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR_OVERRIDE:-$OUTPUT_BASE}"
WAND_OUTPUT_DIR="${WAND_OUTPUT_DIR_OVERRIDE:-${OUTPUT_BASE}/wandb}"
FINAL_SAVE_PATH="${FINAL_SAVE_PATH_OVERRIDE:-${OUTPUT_BASE}/final_model}"

# W&B overrides (optional)
WAND_ENABLED_OVERRIDE="${WAND_ENABLED_OVERRIDE:-}"
WAND_PROJECT_OVERRIDE="${WAND_PROJECT_OVERRIDE:-}"
WAND_ENTITY_OVERRIDE="${WAND_ENTITY_OVERRIDE:-}"

# Build override string (comma-separated key.path=value)
OVERRIDE=""
if [[ -n "${WAND_ENABLED_OVERRIDE}" ]]; then
  OVERRIDE="wandb.enabled=${WAND_ENABLED_OVERRIDE}"
fi
if [[ -n "${WAND_PROJECT_OVERRIDE}" ]]; then
  OVERRIDE="${OVERRIDE:+${OVERRIDE},}wandb.project=${WAND_PROJECT_OVERRIDE}"
fi
if [[ -n "${WAND_ENTITY_OVERRIDE}" ]]; then
  OVERRIDE="${OVERRIDE:+${OVERRIDE},}wandb.entity=${WAND_ENTITY_OVERRIDE}"
fi
OVERRIDE="${OVERRIDE:+${OVERRIDE},}trainer.output_dir=${TRAIN_OUTPUT_DIR}"
OVERRIDE="${OVERRIDE:+${OVERRIDE},}wandb.output_dir=${WAND_OUTPUT_DIR}"
OVERRIDE="${OVERRIDE:+${OVERRIDE},}output.save_path=${FINAL_SAVE_PATH}"

CONFIG_PATH="${CONFIG_PATH_OVERRIDE:-${REPO_DIR}/${CONFIG_REL}}"

WRAP_CMD="cd ${REPO_DIR} \
&& source \$(conda info --base)/etc/profile.d/conda.sh \
&& source ~/.bashrc \
&& conda activate comlrl \
&& export WANDB_CONSOLE=\${WANDB_CONSOLE_OVERRIDE:-off} \
&& export WANDB_SILENT=\${WANDB_SILENT_OVERRIDE:-true} \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\" \
&& python3 -u ${TRAIN_REL} --config ${CONFIG_PATH} --override \"${OVERRIDE}\""

sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --gpus-per-node="${GPUS}" \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --job-name="llm_collab_mc_str_builder_grpo" \
  --wrap="${WRAP_CMD}"

echo "Submitted str_builder GRPO training job with config: ${CONFIG_PATH}"
