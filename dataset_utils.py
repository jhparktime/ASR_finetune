"""Shared helpers for raw AIHub dysarthria data."""

from __future__ import annotations

import json
import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEVERITY_KEY_RE = re.compile(r"(severity|grade|level|중증)", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


@dataclass
class AudioMeta:
    sample_rate: int
    channels: int
    sample_width: int
    duration_sec: float
    size_bytes: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return SPACE_RE.sub(" ", str(text).replace("\n", " ").strip())


def normalize_asr_text(text: Any, *, remove_space: bool = False) -> str:
    """Normalize transcript text before ASR metric computation.

    This keeps Hangul/letters/numbers, removes punctuation/symbol noise, and
    canonicalizes unicode and whitespace so Korean WER/CER is less sensitive to
    superficial formatting differences.
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""

    normalized = unicodedata.normalize("NFC", normalized).lower()
    normalized_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith(("P", "S", "C")):
            if char.isspace():
                normalized_chars.append(" ")
            continue
        normalized_chars.append(char)

    normalized = SPACE_RE.sub(" ", "".join(normalized_chars)).strip()
    if remove_space:
        return normalized.replace(" ", "")
    return normalized


def get_nested_value(obj: dict[str, Any], dotted_path: str) -> Any:
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def recursive_keyword_lookup(obj: Any, prefix: str = "") -> tuple[str, str]:
    """Return the first non-empty value whose key path looks severity-related."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if SEVERITY_KEY_RE.search(key):
                normalized = normalize_text(value)
                if normalized:
                    return normalized, full_key
            found_value, found_key = recursive_keyword_lookup(value, full_key)
            if found_value:
                return found_value, found_key
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found_value, found_key = recursive_keyword_lookup(value, f"{prefix}[{index}]")
            if found_value:
                return found_value, found_key
    return "", ""


def extract_transcript(label: dict[str, Any], field_candidates: Iterable[str]) -> tuple[str, str]:
    for dotted_path in field_candidates:
        value = get_nested_value(label, dotted_path)
        normalized = normalize_text(value)
        if normalized:
            return normalized, dotted_path
    return "", ""


def extract_severity(label: dict[str, Any], field_candidates: Iterable[str]) -> tuple[str, str]:
    for dotted_path in field_candidates:
        value = get_nested_value(label, dotted_path)
        normalized = normalize_text(value)
        if normalized:
            return normalized, dotted_path
    return recursive_keyword_lookup(label)


def iter_label_dirs(split_root: Path, label_dir_prefix: str) -> list[Path]:
    if not split_root.exists():
        return []
    return sorted(
        path
        for path in split_root.iterdir()
        if path.is_dir() and path.name.startswith(label_dir_prefix)
    )


def infer_audio_path(label_path: Path, audio_dirname: str) -> Path:
    parts = list(label_path.parts)
    for index, part in enumerate(parts):
        if part.startswith("라벨링데이터"):
            parts[index] = audio_dirname
            break
    return Path(*parts).with_suffix(".wav")


def parse_category(category_dirname: str) -> tuple[str, str]:
    if "." not in category_dirname:
        return category_dirname, category_dirname
    code, name = category_dirname.split(".", 1)
    return code.strip(), name.strip()


def parse_speaker_meta(label: dict[str, Any], stem: str) -> dict[str, str]:
    patient_info = label.get("Patient_info", {}) or {}
    parts = stem.split("-")

    speaker_code = normalize_text(patient_info.get("SpeakerName")) or (parts[4] if len(parts) > 4 else stem)
    sex = normalize_text(patient_info.get("Sex"))
    age = normalize_text(patient_info.get("Age"))
    area = normalize_text(patient_info.get("Area"))

    if not sex and len(parts) > 7:
        sex = parts[7]
    if not age and len(parts) > 8:
        age = parts[8]
    if not area and len(parts) > 9:
        area = parts[9]

    speaker_id = "-".join(part for part in [speaker_code, sex, age, area] if part)
    return {
        "speaker_id": speaker_id or speaker_code,
        "speaker_code": speaker_code,
        "sex": sex,
        "age": age,
        "area": area,
    }


def read_audio_meta(path: Path) -> AudioMeta:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
    duration_sec = frames / sample_rate if sample_rate else 0.0
    return AudioMeta(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        duration_sec=duration_sec,
        size_bytes=path.stat().st_size,
    )
