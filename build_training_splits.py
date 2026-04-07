#!/usr/bin/env python3
"""Build clean train/dev/test manifests from aligned Whisper segments."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

from dataset_utils import load_jsonl, write_jsonl


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
    parser.add_argument("--segments-manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-duration-sec", type=float, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--dev-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--severity-bins", type=int, default=None)
    parser.add_argument("--allow-empty-splits", action="store_true")
    return parser.parse_args()


def load_report_details(path: Path) -> dict[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {item["audio_path"]: item for item in report.get("files", [])}


def build_file_quality_map(rows: list[dict], report_details: dict[str, dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source_audio_path"]].append(row)

    quality_map = {}
    for source_audio_path, items in grouped.items():
        scores = [float(item["score"]) for item in items]
        durations = [float(item["duration_sec"]) for item in items]
        report_item = report_details.get(source_audio_path, {})
        whisper_segment_count = int(report_item.get("whisper_segment_count", 0) or 0)
        matched_whisper_segments = int(report_item.get("matched_whisper_segments", 0) or 0)
        quality_map[source_audio_path] = {
            "segment_count": len(items),
            "avg_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "median_score": round(statistics.median(scores), 4) if scores else 0.0,
            "max_score": round(max(scores), 4) if scores else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "usable_duration_sec": round(sum(durations), 3),
            "coverage_ratio": round(float(report_item.get("coverage_ratio", 0.0) or 0.0), 4),
            "reference_sentence_count": int(report_item.get("reference_sentence_count", 0) or 0),
            "matched_reference_sentences": int(report_item.get("matched_reference_sentences", 0) or 0),
            "skipped_reference_sentences": int(report_item.get("skipped_reference_sentences", 0) or 0),
            "whisper_segment_count": whisper_segment_count,
            "matched_whisper_segments": matched_whisper_segments,
            "skipped_whisper_segments": int(report_item.get("skipped_whisper_segments", 0) or 0),
            "matched_whisper_ratio": round(
                matched_whisper_segments / whisper_segment_count,
                4,
            )
            if whisper_segment_count > 0
            else 0.0,
        }
    return quality_map


def maybe_actual_severity(values: list[str], unknown_label: str) -> str:
    filtered = [value for value in values if value and value != unknown_label]
    if not filtered:
        return unknown_label
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def compute_speaker_stats(rows: list[dict], unknown_label: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker_id"]].append(row)

    stats = {}
    for speaker_id, items in grouped.items():
        scores = [float(item["score"]) for item in items]
        coverages = [float(item["coverage"]) for item in items]
        durations = [float(item["duration_sec"]) for item in items]
        sample = items[0]
        mean_score = statistics.mean(scores)
        mean_coverage = statistics.mean(coverages)
        severity_proxy = 1.0 - (0.7 * mean_score + 0.3 * mean_coverage)

        stats[speaker_id] = {
            "speaker_id": speaker_id,
            "speaker_code": sample.get("speaker_code", ""),
            "sex": sample.get("sex", ""),
            "age": sample.get("age", ""),
            "area": sample.get("area", ""),
            "category_code": sample.get("category_code", ""),
            "category_name": sample.get("category_name", ""),
            "utterances": len(items),
            "hours": round(sum(durations) / 3600.0, 3),
            "mean_score": round(mean_score, 4),
            "mean_coverage": round(mean_coverage, 4),
            "severity_proxy": round(severity_proxy, 4),
            "actual_severity": maybe_actual_severity(
                [str(item.get("severity_label", "")).strip() for item in items],
                unknown_label=unknown_label,
            ),
        }
    return stats


def assign_severity_bins(speaker_stats: dict[str, dict], severity_bins: int) -> dict[str, str]:
    ordered = sorted(speaker_stats.items(), key=lambda item: item[1]["severity_proxy"])
    total = len(ordered)
    if total == 0:
        return {}
    if severity_bins <= 1:
        labels = ["all"]
    elif severity_bins == 2:
        labels = ["lower_proxy", "higher_proxy"]
    elif severity_bins == 3:
        labels = ["lower_proxy", "mid_proxy", "higher_proxy"]
    else:
        labels = [f"bin_{index:02d}" for index in range(severity_bins)]

    bucket_map = {}
    for index, (speaker_id, _) in enumerate(ordered):
        bucket_index = min((index * len(labels)) // total, len(labels) - 1)
        bucket_map[speaker_id] = labels[bucket_index]
    return bucket_map


def split_counts(total: int, train_ratio: float, dev_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    counts = {
        "train": int(round(total * train_ratio)),
        "dev": int(round(total * dev_ratio)),
        "test": int(round(total * test_ratio)),
    }
    delta = total - sum(counts.values())
    if delta != 0:
        counts["train"] += delta
    if total >= 3:
        for split_name in ("dev", "test"):
            if counts[split_name] == 0:
                counts[split_name] = 1
                counts["train"] = max(1, counts["train"] - 1)
    assigned = counts["train"] + counts["dev"] + counts["test"]
    if assigned != total:
        counts["train"] += total - assigned
    return counts["train"], counts["dev"], counts["test"]


def split_speakers(
    speaker_stats: dict[str, dict],
    severity_bucket_map: dict[str, str],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for speaker_id, item in speaker_stats.items():
        severity_key = item["actual_severity"]
        if not severity_key or severity_key == "unknown":
            severity_key = severity_bucket_map[speaker_id]
        buckets[(severity_key, item["category_code"] or "unknown")].append(speaker_id)

    rng = random.Random(seed)
    mapping = {}
    for bucket_index, bucket in enumerate(sorted(buckets)):
        speakers = sorted(buckets[bucket])
        bucket_rng = random.Random(seed + bucket_index)
        bucket_rng.shuffle(speakers)
        train_count, dev_count, test_count = split_counts(len(speakers), train_ratio, dev_ratio, test_ratio)
        for speaker_id in speakers[:train_count]:
            mapping[speaker_id] = "train"
        for speaker_id in speakers[train_count : train_count + dev_count]:
            mapping[speaker_id] = "dev"
        for speaker_id in speakers[train_count + dev_count : train_count + dev_count + test_count]:
            mapping[speaker_id] = "test"

    counts = Counter(mapping.values())
    for needed_split in ("train", "dev", "test"):
        if counts.get(needed_split, 0) > 0:
            continue
        donor_split = max(("train", "dev", "test"), key=lambda name: counts.get(name, 0))
        donor_candidates = [speaker_id for speaker_id, split_name in mapping.items() if split_name == donor_split]
        if donor_candidates:
            rng.shuffle(donor_candidates)
            mapping[donor_candidates[0]] = needed_split
            counts = Counter(mapping.values())
    return mapping


def summarize_split(rows: list[dict]) -> dict:
    severity_counts = Counter(row.get("resolved_severity", "unknown") for row in rows)
    category_counts = Counter(row.get("category_code", "unknown") for row in rows)
    speaker_ids = {row["speaker_id"] for row in rows}
    hours = sum(float(row["duration_sec"]) for row in rows) / 3600.0
    return {
        "rows": len(rows),
        "speakers": len(speaker_ids),
        "hours": round(hours, 3),
        "severity_counts": dict(sorted(severity_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    segments_manifest = args.segments_manifest or (Path(config.ALIGNMENT_DIR) / "segments_manifest.jsonl")
    report_path = args.report or (Path(config.ALIGNMENT_DIR) / "report.json")
    out_dir = args.out_dir or Path(config.SPLIT_MANIFEST_DIR)
    min_score = float(args.min_score if args.min_score is not None else config.MIN_SEGMENT_SCORE)
    min_duration = float(args.min_duration_sec if args.min_duration_sec is not None else config.MIN_SEGMENT_DURATION_SEC)
    max_duration = float(args.max_duration_sec if args.max_duration_sec is not None else config.MAX_SEGMENT_DURATION_SEC)
    min_coverage = float(args.min_coverage if args.min_coverage is not None else config.MIN_ALIGNMENT_COVERAGE)
    seed = int(args.seed if args.seed is not None else config.SPLIT_SEED)
    train_ratio = float(args.train_ratio if args.train_ratio is not None else config.TRAIN_RATIO)
    dev_ratio = float(args.dev_ratio if args.dev_ratio is not None else config.DEV_RATIO)
    test_ratio = float(args.test_ratio if args.test_ratio is not None else config.TEST_RATIO)
    severity_bins = int(args.severity_bins if args.severity_bins is not None else config.SEVERITY_BINS)
    enable_source_file_filter = bool(getattr(config, "ENABLE_SOURCE_FILE_FILTER", True))
    min_source_file_avg_score = float(config.MIN_SOURCE_FILE_AVG_SCORE)
    min_source_file_segments = int(config.MIN_SOURCE_FILE_SEGMENTS)
    min_source_file_coverage = float(config.MIN_SOURCE_FILE_COVERAGE)
    min_source_file_match_ratio = float(config.MIN_SOURCE_FILE_MATCH_RATIO)
    allow_empty_splits = bool(args.allow_empty_splits or config.ALLOW_EMPTY_SPLITS)

    if not math.isclose(train_ratio + dev_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/dev/test ratios must sum to 1.0")

    rows = load_jsonl(segments_manifest)
    report_details = load_report_details(report_path)
    coverage = {
        audio_path: float(item.get("coverage_ratio", 0.0) or 0.0)
        for audio_path, item in report_details.items()
    }

    filtered = []
    skipped_missing_audio = 0
    for row in rows:
        if float(row["score"]) < min_score:
            continue
        if float(row["duration_sec"]) < min_duration or float(row["duration_sec"]) > max_duration:
            continue
        if coverage.get(row["source_audio_path"], 0.0) < min_coverage:
            continue
        audio_path = str(row.get("audio_path", "")).strip()
        source_audio_path = str(row.get("source_audio_path", "")).strip()
        if audio_path:
            if not Path(audio_path).exists():
                skipped_missing_audio += 1
                continue
        elif not source_audio_path or not Path(source_audio_path).exists():
            skipped_missing_audio += 1
            continue

        item = dict(row)
        item["coverage"] = float(coverage.get(row["source_audio_path"], 0.0))
        filtered.append(item)

    file_quality_map = build_file_quality_map(filtered, report_details)
    if enable_source_file_filter:
        retained_source_files = {
            source_audio_path
            for source_audio_path, quality in file_quality_map.items()
            if quality["segment_count"] >= min_source_file_segments
            and quality["avg_score"] >= min_source_file_avg_score
            and quality["coverage_ratio"] >= min_source_file_coverage
            and quality["matched_whisper_ratio"] >= min_source_file_match_ratio
        }
        dropped_source_files = {
            source_audio_path: quality
            for source_audio_path, quality in file_quality_map.items()
            if source_audio_path not in retained_source_files
        }
        filtered = [row for row in filtered if row["source_audio_path"] in retained_source_files]
    else:
        retained_source_files = set(file_quality_map)
        dropped_source_files = {}

    if not filtered:
        raise ValueError("No usable segments remain after segment and source-file filtering.")

    speaker_stats = compute_speaker_stats(filtered, unknown_label=config.UNKNOWN_SEVERITY_LABEL)
    severity_bucket_map = assign_severity_bins(speaker_stats, severity_bins)
    speaker_split_map = split_speakers(
        speaker_stats=speaker_stats,
        severity_bucket_map=severity_bucket_map,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_rows = {"train": [], "dev": [], "test": []}
    for row in filtered:
        split_name = speaker_split_map[row["speaker_id"]]
        row["split"] = split_name
        resolved_severity = speaker_stats[row["speaker_id"]]["actual_severity"]
        if not resolved_severity or resolved_severity == config.UNKNOWN_SEVERITY_LABEL:
            resolved_severity = severity_bucket_map[row["speaker_id"]]
        row["severity_proxy"] = float(speaker_stats[row["speaker_id"]]["severity_proxy"])
        row["severity_bucket"] = severity_bucket_map[row["speaker_id"]]
        row["resolved_severity"] = resolved_severity
        split_rows[split_name].append(row)

    empty_splits = [split_name for split_name, rows_in_split in split_rows.items() if not rows_in_split]
    if empty_splits and not allow_empty_splits:
        raise ValueError(
            "Empty split detected after speaker assignment. "
            f"Empty splits: {', '.join(empty_splits)}. "
            "Increase data volume or rerun only for pilot with --allow-empty-splits."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all.jsonl"
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    test_path = out_dir / "test.jsonl"
    speaker_stats_path = out_dir / "speaker_stats.json"
    speaker_split_path = out_dir / "speaker_split.json"
    split_records_path = out_dir / "split_records.json"
    split_records_csv_path = out_dir / "split_records.csv"
    summary_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"

    write_jsonl(all_path, filtered)
    write_jsonl(train_path, split_rows["train"])
    write_jsonl(dev_path, split_rows["dev"])
    write_jsonl(test_path, split_rows["test"])

    speaker_stats_payload = []
    for speaker_id in sorted(speaker_stats):
        item = dict(speaker_stats[speaker_id])
        item["severity_bucket"] = severity_bucket_map[speaker_id]
        item["split"] = speaker_split_map[speaker_id]
        speaker_stats_payload.append(item)
    speaker_stats_path.write_text(json.dumps(speaker_stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    speaker_split_path.write_text(json.dumps(speaker_split_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep a flat split assignment table for audit, review, and reruns on remote training machines.
    split_records_payload = []
    for item in speaker_stats_payload:
        split_records_payload.append(
            {
                "speaker_id": item["speaker_id"],
                "split": item["split"],
                "speaker_code": item["speaker_code"],
                "sex": item["sex"],
                "age": item["age"],
                "area": item["area"],
                "category_code": item["category_code"],
                "category_name": item["category_name"],
                "utterances": item["utterances"],
                "hours": item["hours"],
                "mean_score": item["mean_score"],
                "mean_coverage": item["mean_coverage"],
                "severity_bucket": item["severity_bucket"],
                "actual_severity": item["actual_severity"],
                "severity_proxy": item["severity_proxy"],
            }
        )
    split_records_path.write_text(
        json.dumps(split_records_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with split_records_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "speaker_id",
                "split",
                "speaker_code",
                "sex",
                "age",
                "area",
                "category_code",
                "category_name",
                "utterances",
                "hours",
                "mean_score",
                "mean_coverage",
                "severity_bucket",
                "actual_severity",
                "severity_proxy",
            ],
        )
        writer.writeheader()
        writer.writerows(split_records_payload)

    summary = {
        "input_segments": len(rows),
        "filtered_segments": len(filtered),
        "skipped_missing_segment_audio": skipped_missing_audio,
        "unique_speakers": len(speaker_stats),
        "total_hours": round(sum(float(row["duration_sec"]) for row in filtered) / 3600.0, 3),
        "min_score": min_score,
        "min_duration_sec": min_duration,
        "max_duration_sec": max_duration,
        "min_coverage": min_coverage,
        "min_source_file_avg_score": min_source_file_avg_score,
        "min_source_file_segments": min_source_file_segments,
        "min_source_file_coverage": min_source_file_coverage,
        "min_source_file_match_ratio": min_source_file_match_ratio,
        "enable_source_file_filter": enable_source_file_filter,
        "allow_empty_splits": allow_empty_splits,
        "retained_source_files": len(retained_source_files),
        "dropped_source_files": len(dropped_source_files),
        "dropped_source_file_examples": dict(list(sorted(dropped_source_files.items()))[:10]),
        "severity_bins": severity_bins,
        "ratios": {"train": train_ratio, "dev": dev_ratio, "test": test_ratio},
        "splits": {
            "train": summarize_split(split_rows["train"]),
            "dev": summarize_split(split_rows["dev"]),
            "test": summarize_split(split_rows["test"]),
        },
        "paths": {
            "all": str(all_path.resolve()),
            "train": str(train_path.resolve()),
            "dev": str(dev_path.resolve()),
            "test": str(test_path.resolve()),
            "speaker_stats": str(speaker_stats_path.resolve()),
            "speaker_split": str(speaker_split_path.resolve()),
            "split_records_json": str(split_records_path.resolve()),
            "split_records_csv": str(split_records_csv_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Whisper Segment Split Summary",
        "",
        f"- Input segments: {summary['input_segments']}",
        f"- Filtered segments: {summary['filtered_segments']}",
        f"- Unique speakers: {summary['unique_speakers']}",
        f"- Total hours: {summary['total_hours']}",
        f"- Min score: {summary['min_score']}",
        f"- Min coverage: {summary['min_coverage']}",
        f"- Split records JSON: {split_records_path.resolve()}",
        f"- Split records CSV: {split_records_csv_path.resolve()}",
        "",
        "## Splits",
    ]
    for split_name in ("train", "dev", "test"):
        item = summary["splits"][split_name]
        lines.append(f"- {split_name}: {item['rows']} rows, {item['speakers']} speakers, {item['hours']} hours")
        lines.append(f"- {split_name} severity: {json.dumps(item['severity_counts'], ensure_ascii=False)}")
        lines.append(f"- {split_name} categories: {json.dumps(item['category_counts'], ensure_ascii=False)}")
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
