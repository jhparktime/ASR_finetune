#!/usr/bin/env bash
set -euo pipefail

# H100 메모리 관리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="/home/gpuuser/workspace/yeongbeom/ASR_finetune"
SCRIPT_PATH="${PROJECT_DIR}/build_nbest_dataset.py"

# 데이터 소스 및 모델 설정
MANIFEST_FILE="${PROJECT_DIR}/artifacts/training_manifests(seg-large-035)_feature_shards/all.feature_shards.merged.jsonl"
ADAPTER_PATH="/home/gpuuser/workspace/yeongbeom/ASR_finetune/artifacts/training_outputs/run_004_large(seg-large-035)"
MODEL_NAME="openai/whisper-large-v3"

OUTPUT_DIR="${PROJECT_DIR}/artifacts/nbest_datasets/seg-large-035_feature_shards"
mkdir -p "${OUTPUT_DIR}"

# 하이퍼파라미터 최적화
NUM_SHARDS=2
NUM_BEAMS=1
NUM_RETURN_SEQUENCES=1
BATCH_SIZE=192     # H100에 맞춰 상향 조정 (OOM 발생 시 32로 조정)
NUM_WORKERS=0      # NFS 병목 해결을 위해 0으로 설정 (Main process 로딩)
TOP_P=0.95
TEMPERATURE=0.8

RESUME_FLAG="--resume"

cd "${PROJECT_DIR}"

# GPU 0 프로세스
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_FILE}" \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${ADAPTER_PATH}" \
  --bf16 \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index 0 \
  --num-beams "${NUM_BEAMS}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --top-p "${TOP_P}" \
  --temperature "${TEMPERATURE}" \
  --reorder-for-locality \
  --max-open-feature-shards 2 \
  --output-path "${OUTPUT_DIR}/all.whisper-large-v3.lora.shard0.jsonl" \
  --meta-path "${OUTPUT_DIR}/all.whisper-large-v3.lora.shard0.meta.json" \
  ${RESUME_FLAG} &

PID0=$!

# GPU 1 프로세스
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_PATH}" \
  --manifest "${MANIFEST_FILE}" \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${ADAPTER_PATH}" \
  --bf16 \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index 1 \
  --num-beams "${NUM_BEAMS}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --top-p "${TOP_P}" \
  --temperature "${TEMPERATURE}" \
  --reorder-for-locality \
  --max-open-feature-shards 2 \
  --output-path "${OUTPUT_DIR}/all.whisper-large-v3.lora.shard1.jsonl" \
  --meta-path "${OUTPUT_DIR}/all.whisper-large-v3.lora.shard1.meta.json" \
  ${RESUME_FLAG} &

PID1=$!

wait "${PID0}"
wait "${PID1}"

# 결과 병합 및 정렬
python - <<'PY'
import json
from pathlib import Path

out_dir = Path("/home/gpuuser/workspace/yeongbeom/ASR_finetune/artifacts/nbest_datasets/seg-large-035_feature_shards")
files = [
    out_dir / "all.whisper-large-v3.lora.shard0.jsonl",
    out_dir / "all.whisper-large-v3.lora.shard1.jsonl",
]

rows = []
for fp in files:
    if not fp.exists():
        continue
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

# manifest_index 기준으로 최종 정렬
rows.sort(key=lambda x: int(x["manifest_index"]))

merged = out_dir / "all.whisper-large-v3.lora.merged.jsonl"
with merged.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Merge Complete: {merged}")
PY