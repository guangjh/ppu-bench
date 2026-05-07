#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ====================== CONFIGURE ======================
MODEL_PATH="${MODEL_PATH:?Must set MODEL_PATH env var}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval/results}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
USE_HF="${USE_HF:-1}"
TESTSETS="${TESTSETS:-2 3 4}"
# =======================================================

SCRIPT="${REPO_ROOT}/eval/eval_vllm.py"

build_cmd() {
  local setting="$1"
  local task="$2"
  local subset="$3"
  local ratio="$4"
  local testset="$5"

  local run_id="${OUTPUT_DIR}/${setting}/TESTSET_${testset}/${task}_${subset}_r${ratio}"

  local cmd=(
    python "${SCRIPT}"
    --model "${MODEL_PATH}"
    --pretrain
    --run_id "${run_id}"
    --setting "${setting}"
    --task "${task}"
    --subset "${subset}"
    --testset "${testset}"
    --resume False
    --output_dir "${OUTPUT_DIR}"
  )

  if [[ "${USE_HF}" == "1" ]]; then
    cmd+=(--hf)
  else
    cmd+=(--data_file "${REPO_ROOT}/eval/cache/ppu_eval_canonical.parquet")
    cmd+=(--testset_root "${REPO_ROOT}/eval/cache/testsets")
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
  local testset="$5"
  local device="${6:-${CUDA_DEVICE}}"

  local cmd
  cmd="$(build_cmd "${setting}" "${task}" "${subset}" "${ratio}" "${testset}")"

  echo
  echo "============================================================"
  echo "CUDA_VISIBLE_DEVICES=${device} ${cmd}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${device}" eval "${cmd}"
}

cd "${REPO_ROOT}"

# ====================== EVAL MATRIX ======================
# Uncomment the lines you want to run.
# Testset loops over TESTSETS (default: 2 3 4).

for ts in ${TESTSETS}; do

  # ---- complete ----
  # run_eval "complete" "generation" "forget" "30" "${ts}" "0" &
  # run_eval "complete" "class" "forget" "30" "${ts}" "1" &
  # run_eval "complete" "cloze" "forget" "30" "${ts}" "2" &

  # ---- selective ----
  # run_eval "selective" "generation" "forget" "" "${ts}" "3" &
  # run_eval "selective" "class" "forget" "" "${ts}" "4" &
  # run_eval "selective" "cloze" "forget" "" "${ts}" "5" &

  # ---- persona ----
  # run_eval "persona" "generation" "forget" "" "${ts}" "6" &
  # run_eval "persona" "class" "forget" "" "${ts}" "7" &
  # run_eval "persona" "cloze" "forget" "" "${ts}" "8" &

  wait

done
