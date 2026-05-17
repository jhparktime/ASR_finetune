#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import librosa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/gpuuser/workspace/yeongbeom/ASR_finetune/artifacts/aligned_segments/segments_manifest.jsonl"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/gpuuser/workspace/yeongbeom/ASR_finetune/artifacts/segmented_flac_dataset"),
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--padding-sec", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--compression-level", type=int, default=1)
    p.add_argument("--shard-dirs", type=int, default=256)
    p.add_argument("--min-duration-sec", type=float, default=0.05)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["manifest_index"] = idx
            rows.append(row)
    return rows


def safe_int_dir(index: int, shard_dirs: int) -> str:
    return f"{index % shard_dirs:03d}"


def output_path_for(row: dict, output_root: Path, shard_dirs: int) -> Path:
    idx = int(row["manifest_index"])
    subdir = safe_int_dir(idx, shard_dirs)
    speaker = str(row.get("speaker_id", "unknown")).replace("/", "_")
    return output_root / "audio" / subdir / f"{idx:08d}_{speaker}.flac"


def process_one_source(args_tuple):
    (
        source_audio_path,
        rows,
        output_root,
        sample_rate,
        padding_sec,
        compression_level,
        shard_dirs,
        min_duration_sec,
        resume,
    ) = args_tuple

    source_audio_path = Path(source_audio_path)
    results = []

    if not source_audio_path.exists():
        for row in rows:
            results.append(
                {
                    "ok": False,
                    "manifest_index": int(row["manifest_index"]),
                    "error": f"source_missing: {source_audio_path}",
                }
            )
        return results

    try:
        audio, sr = librosa.load(source_audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        for row in rows:
            results.append(
                {
                    "ok": False,
                    "manifest_index": int(row["manifest_index"]),
                    "error": f"load_failed: {source_audio_path}: {repr(e)}",
                }
            )
        return results

    total_samples = len(audio)

    for row in rows:
        idx = int(row["manifest_index"])
        out_path = output_path_for(row, output_root, shard_dirs)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and out_path.exists() and out_path.stat().st_size > 0:
            payload = dict(row)
            payload["audio_path"] = str(out_path.resolve())
            payload["original_audio_path"] = str(row.get("source_audio_path", ""))
            payload["extracted_sample_rate"] = sample_rate
            payload["extracted_format"] = "flac"
            payload["extracted_padding_sec"] = padding_sec
            payload["extracted_duration_sec"] = float(row.get("duration_sec", 0.0)) + 2 * padding_sec
            results.append({"ok": True, "manifest_index": idx, "row": payload})
            continue

        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])

        start_sec_padded = max(0.0, start_sec - padding_sec)
        end_sec_padded = max(start_sec_padded + min_duration_sec, end_sec + padding_sec)

        start_sample = max(0, int(math.floor(start_sec_padded * sample_rate)))
        end_sample = min(total_samples, int(math.ceil(end_sec_padded * sample_rate)))

        if end_sample <= start_sample:
            results.append(
                {
                    "ok": False,
                    "manifest_index": idx,
                    "error": f"empty_segment: start={start_sec}, end={end_sec}",
                }
            )
            continue

        segment = audio[start_sample:end_sample]

        try:
            sf.write(
                out_path,
                segment,
                sample_rate,
                format="FLAC",
                subtype="PCM_16",
                compression_level=compression_level,
            )
        except TypeError:
            # older soundfile versions may not support compression_level
            sf.write(
                out_path,
                segment,
                sample_rate,
                format="FLAC",
                subtype="PCM_16",
            )
        except Exception as e:
            results.append(
                {
                    "ok": False,
                    "manifest_index": idx,
                    "error": f"write_failed: {out_path}: {repr(e)}",
                }
            )
            continue

        payload = dict(row)
        payload["audio_path"] = str(out_path.resolve())
        payload["original_audio_path"] = str(row.get("source_audio_path", ""))
        payload["original_start_sec"] = start_sec
        payload["original_end_sec"] = end_sec
        payload["extracted_start_sec"] = start_sec_padded
        payload["extracted_end_sec"] = end_sec_padded
        payload["extracted_duration_sec"] = (end_sample - start_sample) / sample_rate
        payload["extracted_sample_rate"] = sample_rate
        payload["extracted_format"] = "flac"
        payload["extracted_padding_sec"] = padding_sec

        results.append({"ok": True, "manifest_index": idx, "row": payload})

    return results


def main():
    args = parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_out = args.output_root / "segments_manifest.flac.jsonl"
    error_out = args.output_root / "segments_manifest.flac.errors.jsonl"

    print(f"[INFO] manifest     : {args.manifest}")
    print(f"[INFO] output_root  : {args.output_root}")
    print(f"[INFO] workers      : {args.workers}")
    print(f"[INFO] sample_rate  : {args.sample_rate}")
    print(f"[INFO] padding_sec  : {args.padding_sec}")
    print(f"[INFO] format       : FLAC PCM_16")

    rows = read_jsonl(args.manifest)
    print(f"[INFO] loaded rows  : {len(rows)}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["source_audio_path"])].append(row)

    jobs = []
    for source_audio_path, source_rows in grouped.items():
        source_rows.sort(key=lambda r: float(r["start_sec"]))
        jobs.append(
            (
                source_audio_path,
                source_rows,
                args.output_root,
                args.sample_rate,
                args.padding_sec,
                args.compression_level,
                args.shard_dirs,
                args.min_duration_sec,
                args.resume,
            )
        )

    print(f"[INFO] source wavs  : {len(jobs)}")

    ok_rows = []
    err_rows = []
    processed_sources = 0
    processed_segments = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_source, job) for job in jobs]

        for fut in as_completed(futures):
            processed_sources += 1
            results = fut.result()

            for item in results:
                processed_segments += 1
                if item["ok"]:
                    ok_rows.append(item["row"])
                else:
                    err_rows.append(item)

            if processed_sources % 20 == 0 or processed_sources == len(jobs):
                print(
                    f"[progress] sources {processed_sources}/{len(jobs)} | "
                    f"segments {processed_segments}/{len(rows)} | "
                    f"ok {len(ok_rows)} | errors {len(err_rows)}",
                    flush=True,
                )

    ok_rows.sort(key=lambda r: int(r["manifest_index"]))
    err_rows.sort(key=lambda r: int(r["manifest_index"]))

    with manifest_out.open("w", encoding="utf-8") as f:
        for row in ok_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with error_out.open("w", encoding="utf-8") as f:
        for row in err_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_bytes = 0
    audio_dir = args.output_root / "audio"
    if audio_dir.exists():
        for root, _, files in os.walk(audio_dir):
            for name in files:
                if name.endswith(".flac"):
                    total_bytes += (Path(root) / name).stat().st_size

    print(
        json.dumps(
            {
                "status": "done",
                "input_manifest": str(args.manifest.resolve()),
                "output_manifest": str(manifest_out.resolve()),
                "error_manifest": str(error_out.resolve()),
                "output_audio_dir": str((args.output_root / "audio").resolve()),
                "input_rows": len(rows),
                "ok_rows": len(ok_rows),
                "error_rows": len(err_rows),
                "output_size_gb": round(total_bytes / (1024**3), 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()