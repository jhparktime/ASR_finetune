"""Frozen config that reproduces the strict full-data manifests in training_manifests/."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *  # noqa: F401,F403


# Reproduce the full-data manifests currently checked into `training_manifests/`.
SPLIT_MANIFEST_DIR = PROJECT_ROOT / "training_manifests"
MIN_SEGMENT_SCORE = 0.6
MIN_SEGMENT_DURATION_SEC = 2.0
MAX_SEGMENT_DURATION_SEC = 20.0
MIN_ALIGNMENT_COVERAGE = 0.7
ENABLE_SOURCE_FILE_FILTER = True
MIN_SOURCE_FILE_AVG_SCORE = 0.55
MIN_SOURCE_FILE_SEGMENTS = 3
MIN_SOURCE_FILE_COVERAGE = 0.7
MIN_SOURCE_FILE_MATCH_RATIO = 0.6
