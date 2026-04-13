# LoRA_ASR

## 1. 개요

- 목적: 구음장애 음성 데이터에 대해 `Whisper + LoRA` 기반 ASR 학습 파이프라인을 구성한다.
- 입력 데이터: `원천데이터/*.wav`, `라벨링데이터/*.json`
- 처리 방식:
  1. raw 파일 매칭
  2. `faster-whisper` 기반 alignment
  3. usable segment 선별
  4. `Whisper LoRA` 학습
- 기본 저장 정책: segment WAV를 대량 저장하지 않고, `source_audio_path + start_sec + end_sec`만 manifest에 저장한다.
- 추가 기능: `N-best` 추론 스크립트 제공

## 2. 코드 설명

- [config.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/config.py)
  - 데이터 경로, alignment 파라미터, filtering 기준, 학습 기본값, N-best 기본값을 관리한다.
- [dataset_utils.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/dataset_utils.py)
  - JSON 로드, transcript 추출, severity 추출, speaker 메타 파싱, audio 메타 읽기 등 공통 유틸리티를 제공한다.
- [build_file_manifest.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/build_file_manifest.py)
  - raw `wav/json`를 1:1로 매칭하여 file-level manifest를 생성한다.
- [align_segments.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/align_segments.py)
  - `faster-whisper` 추론 결과와 reference transcript를 DP alignment로 맞춰 segment-level manifest를 생성한다.
  - 기본 모드는 segment WAV를 저장하지 않는다.
  - `--write-audio` 사용 시에만 segment WAV를 저장한다.
- [build_training_splits.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/build_training_splits.py)
  - segment score, coverage, duration, source-file 평균 점수를 기준으로 usable segment만 남긴다.
  - speaker 기준 `train/dev/test` split을 생성한다.
  - split 결과 기록 파일도 함께 생성한다.
- [train_lora.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/train_lora.py)
  - training manifest를 읽어 `Whisper + LoRA` fine-tuning을 수행한다.
  - `audio_path`가 없으면 `source_audio_path + start/end`로 원본 파일에서 직접 구간을 읽는다.
- [infer_nbest.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/infer_nbest.py)
  - base 모델 또는 LoRA adapter에 대해 `N-best` 후보 문장을 생성한다.
  - 전체 파일 또는 `start_sec/end_sec` 구간 단위 추론을 지원한다.
- [build_nbest_dataset.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/build_nbest_dataset.py)
  - manifest 전체 세그먼트에 대해 batched `N-best` 추론을 수행한다.
  - row별 `nbest_candidates`를 포함한 JSONL 데이터셋을 생성한다.
  - 긴 실행을 위해 `--resume`, shard 분할 실행을 지원한다.

## 3. 디렉토리 구조

```text
LoRA_ASR/
├── README.md
├── requirements.txt
├── config.py
├── dataset_utils.py
├── build_file_manifest.py
├── align_segments.py
├── build_training_splits.py
├── train_lora.py
├── infer_nbest.py
└── artifacts/
    ├── file_manifests/
    ├── aligned_segments/
    ├── training_manifests/
    └── training_outputs/
```

## 4. 실행 방법

### 4-1. 환경 준비

```bash
cd /Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

사전 준비 사항:

- Python 가상환경을 생성하고 `requirements.txt`를 설치한다.
- 서버의 전체 데이터 경로를 확인한다.
- 모델 캐시 및 checkpoint 저장용 디스크 여유 공간을 확인한다.
- GPU 서버에서는 CUDA 사용 가능 여부를 먼저 확인한다.
- Hugging Face 캐시 경로를 고정하려면 아래와 같이 설정한다.

```bash
export HF_HOME=/path/to/hf_cache
```

### 4-2. 설정 수정

- [config.py](/Users/jaehyun_lab/Desktop/AI_project/LoRA_ASR/config.py)에서 `DATA_ROOT`를 실제 서버 경로로 수정한다.
- 기본 학습 모델은 `openai/whisper-large-v3`이다.

예시:

```python
DATA_ROOT = Path("/실제/서버/경로/01.데이터")
```

또는:

```bash
export LORA_ASR_DATA_ROOT="/실제/서버/경로/01.데이터"
```

모델 캐시 준비:

- 서버에서 처음 실행할 경우 `openai/whisper-large-v3`가 자동 다운로드된다.
- 네트워크 이슈를 줄이려면 학습 전에 한 번 미리 캐시한다.

```bash
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; WhisperProcessor.from_pretrained('openai/whisper-large-v3'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')"
```

- 모델을 미리 캐시한 뒤 오프라인처럼 실행하려면 `--local-files-only` 옵션을 사용한다.

권장:

- 원격 서버에서 반복 실험을 할 경우 `HF_HOME`을 고정해서 모델 캐시를 재사용한다.
- 학습 산출물 경로도 미리 정해 둔다.
  - 예: `./artifacts/training_outputs/run_001`

### 4-3. file-level manifest 생성

```bash
python build_file_manifest.py
```

출력:

- `artifacts/file_manifests/all_files.jsonl`
- `artifacts/file_manifests/summary.json`

확인:

- `summary.json`에서 `records`, `missing_audio`, `empty_transcript`를 먼저 확인한다.
- `missing_audio`가 비정상적으로 많으면 데이터 경로 또는 압축 해제 상태를 다시 확인한다.

### 4-4. alignment 수행

기본 모드:

```bash
python align_segments.py --device cuda
```

샘플 청취용 segment WAV까지 저장할 때:

```bash
python align_segments.py --device cuda --write-audio
```

출력:

- `artifacts/aligned_segments/segments_manifest.jsonl`
- `artifacts/aligned_segments/report.json`
- `artifacts/aligned_segments/segments/` (`--write-audio` 사용 시)

주의:

- 디스크 절약을 위해 기본적으로 `--write-audio`는 사용하지 않는다.
- 샘플 청취 또는 alignment 검수 시에만 `--write-audio`를 사용한다.
- GPU 서버에서는 가능하면 `--device cuda`로 실행한다.

### 4-5. usable segment 선별 및 split 생성

```bash
python build_training_splits.py
```

출력:

- `artifacts/training_manifests/all.jsonl`
- `artifacts/training_manifests/train.jsonl`
- `artifacts/training_manifests/dev.jsonl`
- `artifacts/training_manifests/test.jsonl`
- `artifacts/training_manifests/speaker_stats.json`
- `artifacts/training_manifests/speaker_split.json`

학습 metric 참고:

- `train_lora.py`의 `wer`와 `cer`는 평가 전에 한글 기준 text normalization을 거친다.
- normalization에서는 Unicode NFC, 소문자화, 공백 정리, 구두점/기호 제거를 적용한다.
- `cer`는 공백까지 제거한 문자열로 계산하고, 비교용 원본 값은 `raw_wer`, `raw_cer`로 같이 저장된다.
- `artifacts/training_manifests/split_records.json`
- `artifacts/training_manifests/split_records.csv`
- `artifacts/training_manifests/summary.json`
- `artifacts/training_manifests/summary.md`

주의:

- 전체 데이터가 아닌 소규모 pilot에서는 특정 split이 비어서 실패할 수 있다.
- pilot 확인만 필요하면 아래와 같이 실행한다.

```bash
python build_training_splits.py --allow-empty-splits
```

확인:

- `summary.json`에서 `filtered_segments`, `retained_source_files`, `splits.train.rows`, `splits.dev.rows`를 확인한다.
- `train.jsonl` 또는 `dev.jsonl`이 비어 있으면 학습을 진행하지 않는다.

### 4-6. LoRA 학습

```bash
python train_lora.py \
  --train-manifest ./artifacts/training_manifests/train.jsonl \
  --eval-manifest ./artifacts/training_manifests/dev.jsonl \
  --test-manifest ./artifacts/training_manifests/test.jsonl \
  --output-dir ./artifacts/training_outputs/run_001 \
  --fp16
```

주의:

- `--fp16`은 CUDA GPU 학습 환경 기준 예시이다.
- CPU 환경 또는 `fp16`을 지원하지 않는 환경에서는 `--fp16` 없이 실행한다.
- `train.jsonl` 또는 `dev.jsonl`이 비어 있으면 학습은 시작되지 않는다. 이 경우 split 단계 결과를 먼저 확인한다.
- 모델을 사전 캐시했다면 `--local-files-only`를 추가할 수 있다.
- H100 환경에서는 `bf16`도 검토할 수 있다.

출력:

- `artifacts/training_outputs/run_001`

### 4-7. N-best 추론

전체 파일 추론:

```bash
python infer_nbest.py \
  --audio-path /abs/path/sample.wav \
  --adapter-path ./artifacts/training_outputs/run_001 \
  --num-beams 5 \
  --num-return-sequences 5
```

특정 구간 추론:

```bash
python infer_nbest.py \
  --audio-path /abs/path/source.wav \
  --start-sec 10.0 \
  --end-sec 15.0 \
  --adapter-path ./artifacts/training_outputs/run_001 \
  --num-beams 5 \
  --num-return-sequences 5
```

### 4-8. manifest 전체 N-best 데이터셋 생성

base 모델만 사용할 때:

```bash
python build_nbest_dataset.py \
  --manifest ./artifacts/training_manifests/all.jsonl \
  --model-name openai/whisper-large-v3 \
  --output-path ./artifacts/nbest_datasets/all.whisper_large_v3.jsonl
```

LoRA adapter를 포함할 때:

```bash
python build_nbest_dataset.py \
  --manifest ./artifacts/training_manifests/all.jsonl \
  --model-name openai/whisper-large-v3 \
  --adapter-path ./artifacts/training_outputs/run_004_large \
  --output-path ./artifacts/nbest_datasets/all.run_004_large.jsonl \
  --resume
```

설명:

- 출력 JSONL의 각 row에는 원본 manifest 정보와 `nbest_candidates`가 함께 저장된다.
- 대용량 실행을 대비해 `--resume`으로 이어쓰기할 수 있다.
- 여러 프로세스로 나눌 때는 `--num-shards`와 `--shard-index`를 사용한다.

## 5. 실행 전 체크리스트

- 가상환경 생성 및 `requirements.txt` 설치 완료
- `LORA_ASR_DATA_ROOT` 또는 `config.py`의 `DATA_ROOT` 설정 완료
- `openai/whisper-large-v3` 캐시 준비 완료
- `HF_HOME` 설정 여부 확인
- 전체 데이터 경로에서 `build_file_manifest.py` 실행 완료
- `align_segments.py` 실행 완료
- `build_training_splits.py` 실행 완료
- `train.jsonl`, `dev.jsonl`이 비어 있지 않은지 확인 완료
- 학습 출력 경로와 디스크 여유 공간 확인 완료
