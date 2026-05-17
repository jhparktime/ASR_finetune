#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./eval_outputs

MANIFEST="./artifacts/training_manifests_results/training_manifests(seg-large-025)/test.jsonl"
MODEL_NAME="openai/whisper-large-v3"

run_eval() {
  local name="$1"
  local adapter_path="${2:-}"
  local log_path="./artifacts/eval_results_each_datas/eval_outputs_025/eval_result_${name}.log"

  echo "============================================================"
  echo "[START] ${name}"
  echo "Log: ${log_path}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if [[ -z "${adapter_path}" ]]; then
    torchrun --nproc_per_node=2 eval_lora.py \
      --manifest "${MANIFEST}" \
      --model-name "${MODEL_NAME}" \
      > "${log_path}" 2>&1
  else
    torchrun --nproc_per_node=2 eval_lora.py \
      --manifest "${MANIFEST}" \
      --model-name "${MODEL_NAME}" \
      --adapter-path "${adapter_path}" \
      > "${log_path}" 2>&1
  fi

  echo "============================================================"
  echo "[DONE] ${name}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  echo
}

run_eval "base_000"

run_eval "lora_000" "./artifacts/training_outputs_results/training_outputs(seg-large)/run_006_large(seg-large-000)"
run_eval "lora_010" "./artifacts/training_outputs_results/training_outputs(seg-large)/run_008_large(seg-large-010)"
run_eval "lora_025" "./artifacts/training_outputs_results/training_outputs(seg-large)/run_010_large(seg-large-025)"
run_eval "lora_035" "./artifacts/training_outputs_results/training_outputs(seg-large)/run_004_large(seg-large-035)"
run_eval "lora_050" "./artifacts/training_outputs_results/training_outputs(seg-large)/run_012_large(seg-large-050)"

echo "All evaluations completed."