#!/usr/bin/env bash
# Fine-tune VGGT on Co3D.
#
# Required environment variables (or edit defaults below):
#   VGGT_REPO_DIR       - path to this cloned repo
#   CO3D_DIR            - path to Co3D dataset
#   CO3D_ANNOTATION_DIR - path to Co3D annotations (from JianyuanWang/co3d_anno on HuggingFace)
#   PRETRAIN_CKPT       - path to facebook/VGGT-1B model.pt
#   PYTHON_BIN          - python executable in your vggt conda env
#
# Example:
#   VGGT_REPO_DIR=/path/to/repo \
#   CO3D_DIR=/path/to/co3d \
#   CO3D_ANNOTATION_DIR=/path/to/co3d_anno \
#   PRETRAIN_CKPT=/path/to/model.pt \
#   PYTHON_BIN=$(which python) \
#   bash finetune/run_training.sh

set -euo pipefail

VGGT_REPO_DIR="${VGGT_REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VGGT_TRAINING_DIR="${VGGT_REPO_DIR}/training"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="config"
CONFIG_DIR="${VGGT_RUNTIME_CONFIG_DIR:-/tmp/vggt-finetune-configs}"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${NPROC_PER_NODE:=4}"
: "${VGGT_LOG_ROOT:=/tmp/vggt-logs}"
: "${HF_HOME:=/tmp/vggt-hf}"
: "${TORCH_HOME:=/tmp/vggt-torch}"

if [[ ! -d "${VGGT_TRAINING_DIR}" ]]; then
  printf 'ERROR: VGGT training directory not found: %s\n' "${VGGT_TRAINING_DIR}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'ERROR: Python not found or not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

# Stage config files to a temp dir (required by Hydra config-dir mode)
mkdir -p "${CONFIG_DIR}" "${VGGT_LOG_ROOT}" "${HF_HOME}" "${TORCH_HOME}"
cp "${VGGT_TRAINING_DIR}/config/default_dataset.yaml" "${CONFIG_DIR}/default_dataset.yaml"

# Substitute dataset paths into config
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sed \
  -e "s|/YOUR/PATH/TO/CO3D|${CO3D_DIR}|g" \
  -e "s|/YOUR/PATH/TO/CO3D_ANNOTATION|${CO3D_ANNOTATION_DIR}|g" \
  -e "s|/YOUR/PATH/TO/facebook_VGGT-1B/model.pt|${PRETRAIN_CKPT}|g" \
  "${SCRIPT_DIR}/config.yaml" > "${CONFIG_DIR}/${CONFIG_NAME}.yaml"

export CUDA_VISIBLE_DEVICES
export VGGT_LOG_ROOT
export HF_HOME
export TORCH_HOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${VGGT_REPO_DIR}:${VGGT_TRAINING_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

printf 'Launching VGGT fine-tuning\n'
printf '  repo:     %s\n' "${VGGT_REPO_DIR}"
printf '  config:   %s/%s.yaml\n' "${CONFIG_DIR}" "${CONFIG_NAME}"
printf '  log root: %s\n' "${VGGT_LOG_ROOT}"
printf '  GPUs:     %s  (nproc=%s)\n' "${CUDA_VISIBLE_DEVICES}" "${NPROC_PER_NODE}"

cd "${VGGT_TRAINING_DIR}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  "${VGGT_TRAINING_DIR}/launch_vggt_training_with_config_dir.py" \
  --config-dir "${CONFIG_DIR}" \
  --config "${CONFIG_NAME}"
