#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ====================== CONFIGURE ======================
MODEL_PATH="${MODEL_PATH:?Must set MODEL_PATH env var}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval/results}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
USE_HF="${USE_HF:-1}"
# =======================================================

SCRIPT="${REPO_ROOT}/eval/eval_vllm.py"

build_cmd() {
  local setting="$1"
  local subset="$2"
  local ratio="$3"

  local run_id="${OUTPUT_DIR}/adv/${setting}/${subset}_r${ratio}"

  local cmd=(
    python "${SCRIPT}"
    --model "${MODEL_PATH}"
    --pretrain
    --run_id "${run_id}"
    --setting "${setting}"
    --task "generation"
    --subset "${subset}"
    --resume False
    --output_dir "${OUTPUT_DIR}"
  )

  if [[ "${USE_HF}" == "1" ]]; then
    cmd+=(--hf)
  else
    cmd+=(--data_file "${REPO_ROOT}/eval/cache/ppu_eval_canonical.parquet")
  fi

  if [[ "${setting}" == "complete" ]]; then
    cmd+=(--ratio "${ratio}")
  fi

  echo "${cmd[@]}"
}

run_eval() {
  local setting="$1"
  local subset="$2"
  local ratio="$3"
  local device="${4:-${CUDA_DEVICE}}"

  local cmd
  cmd="$(build_cmd "${setting}" "${subset}" "${ratio}")"

  echo
  echo "============================================================"
  echo "CUDA_VISIBLE_DEVICES=${device} ${cmd}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${device}" eval "${cmd}"
}

cd "${REPO_ROOT}"

# ====================== EVAL MATRIX ======================
# Uncomment the lines you want to run.

# ---- complete ----
# run_eval "complete" "random_prefix" "30" "0" &
# run_eval "complete" "jailbreak_style_prompt" "30" "1" &
# run_eval "complete" "paraphrase" "30" "2" &

# ---- selective ----
# run_eval "selective" "random_prefix" "" "0" &
# run_eval "selective" "jailbreak_style_prompt" "" "1" &
# run_eval "selective" "paraphrase" "" "2" &

# ---- persona ----
# run_eval "persona" "random_prefix" "" "0" &
# run_eval "persona" "jailbreak_style_prompt" "" "1" &
# run_eval "persona" "paraphrase" "" "2" &

wait
