"""Project configuration for Whisper-aligned LoRA training."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Change this path on the actual training machine.
DATA_ROOT = Path(
    os.environ.get(
        "LORA_ASR_DATA_ROOT",
        "/absolute/path/to/01.데이터",
    )
)

# All generated manifests, reports, and checkpoints are written under this root.
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
FILE_MANIFEST_DIR = ARTIFACT_ROOT / "file_manifests"
ALIGNMENT_DIR = ARTIFACT_ROOT / "aligned_segments"
SPLIT_MANIFEST_DIR = ARTIFACT_ROOT / "training_manifests"
TRAINING_OUTPUT_DIR = ARTIFACT_ROOT / "training_outputs" / "whisper_large_v3_lora_segments"

# The scripts scan both source splits and rebuild a new speaker-level split.
SOURCE_SPLIT_DIRS = ("1.Training", "2.Validation")

# The dataset is assumed to contain mirrored label/audio trees.
LABEL_DIR_PREFIX = "라벨링데이터"
AUDIO_DIRNAME = "원천데이터"

# Split ratios for the new speaker-level split.
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# The split builder balances speaker groups across these metadata fields.
# If `severity_label` is missing in the labels, the code still balances by category.
STRATIFY_FIELDS = ("severity_label", "category_code")
UNKNOWN_SEVERITY_LABEL = "unknown"

# Transcript extraction priority.
TRANSCRIPT_FIELD_CANDIDATES = (
    "Transcript",
    "transcript",
    "Text",
    "text",
)

# Update these dotted paths if the full labels expose a known severity field.
# The builder also performs a fallback recursive search for keys containing
# "severity", "grade", "level", or "중증".
SEVERITY_FIELD_CANDIDATES = (
    "Patient_info.Severity",
    "Patient_info.SeverityLevel",
    "Patient_info.SeverityGrade",
    "Patient_info.Grade",
    "Patient_info.Level",
    "Disease_info.Severity",
    "Disease_info.SeverityLevel",
    "Disease_info.SeverityGrade",
    "Disease_info.Grade",
    "Disease_info.Level",
    "Test_info.Severity",
    "Test_info.SeverityLevel",
    "Test_info.SeverityGrade",
    "Test_info.Grade",
    "Meta_info.Severity",
)

# Faster-Whisper alignment defaults.
ALIGN_MODEL_SIZE = "small"
ALIGN_DEVICE = "cpu"
ALIGN_COMPUTE_TYPE = "int8"
ALIGN_BEAM_SIZE = 3
ALIGN_MAX_GROUP_SENTENCES = 3
ALIGN_MIN_SCORE = 0.28
ALIGN_FALLBACK_WORD_CHUNK = 12
ALIGN_PAUSE_BOUNDARY_MS = 250
ALIGN_SKIP_CHUNK_PENALTY = -0.08
ALIGN_SKIP_SENTENCE_PENALTY = -0.12
ALIGN_WRITE_AUDIO = False

# Clean segment filtering and speaker split defaults.
# These defaults intentionally lean inclusive for dysarthric speech so the
# training set keeps harder utterances instead of over-pruning them away.
MIN_SEGMENT_SCORE = 0.35
MIN_SEGMENT_DURATION_SEC = 1.0
MAX_SEGMENT_DURATION_SEC = 30.0
MIN_ALIGNMENT_COVERAGE = 0.45
ENABLE_SOURCE_FILE_FILTER = False
MIN_SOURCE_FILE_AVG_SCORE = 0.35
MIN_SOURCE_FILE_SEGMENTS = 1
MIN_SOURCE_FILE_COVERAGE = 0.45
MIN_SOURCE_FILE_MATCH_RATIO = 0.3
SEVERITY_BINS = 3
ALLOW_EMPTY_SPLITS = False

# Training defaults.
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "Korean"
TASK = "transcribe"
SAMPLING_RATE = 16000
MAX_LABEL_LENGTH = 256
NORMALIZE_TRAIN_TEXT = True

# Keep large-v3 defaults conservative enough for reproducible single-GPU runs.
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 5.0
WARMUP_RATIO = 0.05
LOGGING_STEPS = 25
EVAL_STEPS = 200
SAVE_STEPS = 200

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "v_proj")

# N-best decoding defaults for dysarthric ASR analysis.
N_BEST_BEAMS = 5
N_BEST_RETURN_SEQUENCES = 5
N_BEST_MAX_LENGTH = 256
N_BEST_NUM_BEAM_GROUPS = 5
N_BEST_DIVERSITY_PENALTY = 0.3
N_BEST_SAMPLING_FALLBACK = True
N_BEST_TOP_P = 0.95
N_BEST_TEMPERATURE = 0.8

SEED = 42
FP16 = True
BF16 = False
GRADIENT_CHECKPOINTING = True
LOCAL_FILES_ONLY = False
