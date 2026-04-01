#!/usr/bin/env python3
"""Build a file-level manifest from raw WAV/JSON pairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

from dataset_utils import (
    extract_severity,
    extract_transcript,
    infer_audio_path,
    iter_label_dirs,
    load_json,
    parse_category,
    parse_speaker_meta,
    read_audio_meta,
    write_jsonl,
)


def load_config(config_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("lora_asr_config", config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.py"),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def build_records(config: ModuleType, data_root: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    missing_audio: list[str] = []
    empty_transcript: list[str] = []
    match_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()

    for split_dirname in config.SOURCE_SPLIT_DIRS:
        split_root = data_root / split_dirname
        label_dirs = iter_label_dirs(split_root, config.LABEL_DIR_PREFIX)
        for label_dir in label_dirs:
            for label_path in sorted(label_dir.rglob("*.json")):
                label = load_json(label_path)
                audio_path = infer_audio_path(label_path, config.AUDIO_DIRNAME)
                if not audio_path.exists():
                    missing_audio.append(str(audio_path))
                    continue

                transcript, transcript_source = extract_transcript(label, config.TRANSCRIPT_FIELD_CANDIDATES)
                if not transcript:
                    empty_transcript.append(str(label_path))
                    continue

                severity_label, severity_source = extract_severity(label, config.SEVERITY_FIELD_CANDIDATES)
                if not severity_label:
                    severity_label = config.UNKNOWN_SEVERITY_LABEL
                    severity_source = "missing"

                audio_meta = read_audio_meta(audio_path)
                speaker_meta = parse_speaker_meta(label, label_path.stem)
                category_code, category_name = parse_category(label_path.parent.name)
                source_split = "training" if split_dirname.startswith("1.") else "validation"

                disease_info = label.get("Disease_info", {}) or {}
                patient_info = label.get("Patient_info", {}) or {}
                test_info = label.get("Test_info", {}) or {}

                records.append(
                    {
                        "audio_path": str(audio_path.resolve()),
                        "label_path": str(label_path.resolve()),
                        "text": transcript,
                        "transcript_source": transcript_source,
                        "source_split": source_split,
                        "source_label_dir": label_dir.name,
                        "category_dirname": label_path.parent.name,
                        "category_code": category_code,
                        "category_name": category_name,
                        "severity_label": severity_label,
                        "severity_source": severity_source,
                        "duration_sec": round(audio_meta.duration_sec, 3),
                        "sample_rate": audio_meta.sample_rate,
                        "channels": audio_meta.channels,
                        "sample_width_bytes": audio_meta.sample_width,
                        "size_bytes": audio_meta.size_bytes,
                        "file_id": str(label.get("File_id", "")).strip(),
                        "play_time": str(label.get("playTime", "")).strip(),
                        "disease_type": str(disease_info.get("Type", "")).strip(),
                        "subcategory1": str(disease_info.get("Subcategory1", "")).strip(),
                        "subcategory2": str(disease_info.get("Subcategory2", "")).strip(),
                        "subcategory3": str(disease_info.get("Subcategory3", "")).strip(),
                        "subcategory6": str(disease_info.get("Subcategory6", "")).strip(),
                        "test_method": str(test_info.get("TestMethod", "")).strip(),
                        "speaker_name": str(patient_info.get("SpeakerName", "")).strip(),
                        **speaker_meta,
                    }
                )
                match_counts["matched"] += 1
                source_split_counts[source_split] += 1
                category_counts[category_code] += 1
                severity_counts[severity_label] += 1

    summary = {
        "data_root": str(data_root.resolve()),
        "records": len(records),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "missing_audio": len(missing_audio),
        "empty_transcript": len(empty_transcript),
        "missing_audio_examples": missing_audio[:20],
        "empty_transcript_examples": empty_transcript[:20],
        "duration_over_30_sec": sum(float(row["duration_sec"]) > 30.0 for row in records),
        "duration_over_300_sec": sum(float(row["duration_sec"]) > 300.0 for row in records),
    }
    return records, summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_root = args.data_root or Path(config.DATA_ROOT)
    out_dir = args.out_dir or Path(config.FILE_MANIFEST_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, summary = build_records(config, data_root)
    manifest_path = out_dir / "all_files.jsonl"
    summary_path = out_dir / "summary.json"

    write_jsonl(manifest_path, records)
    summary["manifest_path"] = str(manifest_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
