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
  local task="$2"
  local subset="$3"
  local ratio="$4"

  local run_id="${OUTPUT_DIR}/${setting}/${task}_${subset}_r${ratio}"

  local cmd=(
    python "${SCRIPT}"
    --model "${MODEL_PATH}"
    --pretrain
    --run_id "${run_id}"
    --setting "${setting}"
    --task "${task}"
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
  local task="$2"
  local subset="$3"
  local ratio="$4"
  local device="${5:-${CUDA_DEVICE}}"

  local cmd
  cmd="$(build_cmd "${setting}" "${task}" "${subset}" "${ratio}")"

  echo
  echo "============================================================"
  echo "CUDA_VISIBLE_DEVICES=${device} ${cmd}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${device}" eval "${cmd}"
}

cd "${REPO_ROOT}"

# ====================== EVAL MATRIX ======================
# Uncomment the lines you want to run.
# Run multiple lines with "&" to parallelize across GPUs.

# ---- complete ----
# run_eval "complete" "generation" "forget" "30" "0" &
# run_eval "complete" "generation" "retain" "30" "1" &
# run_eval "complete" "class" "forget" "30" "2" &
# run_eval "complete" "class" "retain" "30" "3" &
# run_eval "complete" "cloze" "forget" "30" "4" &
run_eval "complete" "cloze" "retain" "30" "5" &

# ---- selective ----
# run_eval "selective" "generation" "forget" "" "0" &
# run_eval "selective" "generation" "retain" "" "1" &
# run_eval "selective" "class" "forget" "" "2" &
# run_eval "selective" "class" "retain" "" "3" &
# run_eval "selective" "cloze" "forget" "" "4" &
# run_eval "selective" "cloze" "retain" "" "5" &

# ---- persona ----
# run_eval "persona" "generation" "forget" "" "0" &
# run_eval "persona" "generation" "retain" "" "1" &
# run_eval "persona" "class" "forget" "" "2" &
# run_eval "persona" "class" "retain" "" "3" &
# run_eval "persona" "cloze" "forget" "" "4" &
# run_eval "persona" "cloze" "retain" "" "5" &

wait
