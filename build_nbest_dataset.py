#!/usr/bin/env python3
"""Build an N-best inference dataset from a segment manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset_utils import load_jsonl
from infer_nbest import dedupe_candidates, load_audio, load_audio_segment
from train_lora import load_config, maybe_limit


class SegmentInferenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        processor: WhisperProcessor,
        sampling_rate: int,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio_path = str(row.get("audio_path", "")).strip()
        if audio_path:
            audio = load_audio(Path(audio_path), self.sampling_rate)
        else:
            audio = load_audio_segment(
                Path(row["source_audio_path"]),
                float(row["start_sec"]),
                float(row["end_sec"]),
                self.sampling_rate,
            )

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features[0]
        return {
            "input_features": input_features,
            "row": row,
            "manifest_index": int(row["manifest_index"]),
        }


@dataclass
class SegmentInferenceCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["rows"] = [feature["row"] for feature in features]
        batch["manifest_indices"] = [feature["manifest_index"] for feature in features]
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.py"),
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest to decode. Defaults to <split_manifest_dir>/all.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to artifacts/nbest_datasets/<manifest>.<model>.jsonl.",
    )
    parser.add_argument("--meta-path", type=Path, default=None, help="Optional metadata JSON path.")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--sampling-rate", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sampling-fallback", action="store_true")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default=None, help="Device name, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output JSONL and skip decoded rows.")
    return parser.parse_args()


def infer_output_path(config, manifest_path: Path, model_name: str, adapter_path: Path | None) -> Path:
    model_tag = adapter_path.name if adapter_path is not None else model_name.replace("/", "--")
    out_dir = Path(config.ARTIFACT_ROOT) / "nbest_datasets"
    return out_dir / f"{manifest_path.stem}.{model_tag}.jsonl"


def load_processed_indices(path: Path) -> set[int]:
    processed: set[int] = set()
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            processed.add(int(item["manifest_index"]))
    return processed


def chunk_by_return_sequences(items: list[str], scores, chunk_size: int) -> list[tuple[list[str], list[float | None]]]:
    grouped = []
    score_list = None
    if scores is not None:
        score_list = [float(value) for value in scores]
    for start in range(0, len(items), chunk_size):
        texts = items[start : start + chunk_size]
        chunk_scores = None
        if score_list is not None:
            chunk_scores = score_list[start : start + chunk_size]
        grouped.append((texts, chunk_scores))
    return grouped


def add_grouped_candidates(
    candidates_per_row: list[list[dict]],
    grouped_texts: list[tuple[list[str], list[float | None]]],
    source: str,
) -> list[list[dict]]:
    for row_index, (texts, scores) in enumerate(grouped_texts):
        seq_scores = scores if scores is not None else None
        candidates_per_row[row_index] = dedupe_candidates(
            candidates_per_row[row_index],
            texts,
            seq_scores,
            source=source,
        )
    return candidates_per_row


def decode_batch(
    *,
    model,
    processor: WhisperProcessor,
    input_features: torch.Tensor,
    num_beams: int,
    num_return_sequences: int,
    max_length: int,
    sampling_fallback: bool,
    top_p: float,
    temperature: float,
) -> list[list[dict]]:
    candidates_per_row = [[] for _ in range(input_features.size(0))]
    with torch.no_grad():
        beam_outputs = model.generate(
            input_features=input_features,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
    beam_texts = processor.tokenizer.batch_decode(beam_outputs.sequences, skip_special_tokens=True)
    beam_scores = getattr(beam_outputs, "sequences_scores", None)
    grouped_beam = chunk_by_return_sequences(beam_texts, beam_scores, num_return_sequences)
    candidates_per_row = add_grouped_candidates(candidates_per_row, grouped_beam, source="beam")

    if sampling_fallback and any(len(items) < num_return_sequences for items in candidates_per_row):
        sample_count = max(num_return_sequences * 2, 4)
        with torch.no_grad():
            sample_outputs = model.generate(
                input_features=input_features,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                num_beams=1,
                num_return_sequences=sample_count,
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sample_texts = processor.tokenizer.batch_decode(sample_outputs.sequences, skip_special_tokens=True)
        sample_scores = getattr(sample_outputs, "sequences_scores", None)
        grouped_sample = chunk_by_return_sequences(sample_texts, sample_scores, sample_count)
        candidates_per_row = add_grouped_candidates(candidates_per_row, grouped_sample, source="sample")

    for index in range(len(candidates_per_row)):
        candidates_per_row[index] = candidates_per_row[index][:num_return_sequences]
    return candidates_per_row


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    indexed_rows = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["manifest_index"] = index
        indexed_rows.append(item)

    selected = maybe_limit(indexed_rows, args.max_samples)
    if args.start_index > 0 or args.end_index is not None:
        selected = selected[args.start_index : args.end_index]
    if args.num_shards > 1:
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards).")
        selected = [row for index, row in enumerate(selected) if index % args.num_shards == args.shard_index]
    return selected


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    manifest_path = args.manifest or (Path(config.SPLIT_MANIFEST_DIR) / "all.jsonl")
    model_name = args.model_name or config.MODEL_NAME
    language = args.language or config.LANGUAGE
    task = args.task or config.TASK
    sampling_rate = int(args.sampling_rate or config.SAMPLING_RATE)
    num_return_sequences = int(args.num_return_sequences or config.N_BEST_RETURN_SEQUENCES)
    num_beams = max(int(args.num_beams or config.N_BEST_BEAMS), num_return_sequences)
    max_length = int(args.max_length or config.N_BEST_MAX_LENGTH)
    sampling_fallback = bool(args.sampling_fallback or config.N_BEST_SAMPLING_FALLBACK)
    top_p = float(args.top_p or config.N_BEST_TOP_P)
    temperature = float(args.temperature or config.N_BEST_TEMPERATURE)
    local_files_only = bool(args.local_files_only or config.LOCAL_FILES_ONLY)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    requested_bf16 = bool(args.bf16 or config.BF16)
    requested_fp16 = bool(args.fp16 or config.FP16)
    use_bf16 = bool(device_name.startswith("cuda") and requested_bf16)
    use_fp16 = bool(device_name.startswith("cuda") and requested_fp16 and not use_bf16)
    adapter_path = args.adapter_path

    output_path = args.output_path or infer_output_path(config, manifest_path, model_name, adapter_path)
    meta_path = args.meta_path or output_path.with_suffix(".meta.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_jsonl(manifest_path), args)
    if not rows:
        raise ValueError(f"No rows selected from manifest: {manifest_path}")

    processed_indices: set[int] = set()
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {output_path}. Pass --resume to continue writing.")
        processed_indices = load_processed_indices(output_path)

    rows_to_decode = [row for row in rows if row["manifest_index"] not in processed_indices]
    if not rows_to_decode:
        print(
            json.dumps(
                {
                    "status": "nothing_to_do",
                    "manifest": str(manifest_path.resolve()),
                    "output_path": str(output_path.resolve()),
                    "rows_selected": len(rows),
                    "rows_remaining": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task,
        local_files_only=local_files_only,
    )

    model_kwargs = {"local_files_only": local_files_only}
    if device_name.startswith("cuda"):
        if use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
    model = WhisperForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    if adapter_path is not None and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.generation_config.language = language
    model.generation_config.task = task
    model.to(device_name)

    dataset = SegmentInferenceDataset(
        rows_to_decode,
        processor=processor,
        sampling_rate=sampling_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=SegmentInferenceCollator(processor),
        pin_memory=device_name.startswith("cuda"),
    )

    rows_written = 0
    with output_path.open("a", encoding="utf-8") as outfile:
        for batch_index, batch in enumerate(dataloader, start=1):
            input_features = batch["input_features"].to(device_name)
            candidates_per_row = decode_batch(
                model=model,
                processor=processor,
                input_features=input_features,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_length=max_length,
                sampling_fallback=sampling_fallback,
                top_p=top_p,
                temperature=temperature,
            )
            for row, manifest_index, candidates in zip(batch["rows"], batch["manifest_indices"], candidates_per_row):
                payload = dict(row)
                payload["manifest_index"] = int(manifest_index)
                payload["nbest_model_name"] = model_name
                payload["nbest_adapter_path"] = str(adapter_path.resolve()) if adapter_path else ""
                payload["nbest_num_beams"] = num_beams
                payload["nbest_num_return_sequences"] = num_return_sequences
                payload["nbest_sampling_fallback"] = sampling_fallback
                payload["nbest_candidates"] = candidates
                outfile.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rows_written += 1
            outfile.flush()
            print(
                f"[batch {batch_index}] wrote {rows_written}/{len(rows_to_decode)} rows "
                f"to {output_path}"
            )

    meta = {
        "manifest": str(manifest_path.resolve()),
        "output_path": str(output_path.resolve()),
        "adapter_path": str(adapter_path.resolve()) if adapter_path else "",
        "model_name": model_name,
        "language": language,
        "task": task,
        "sampling_rate": sampling_rate,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "max_length": max_length,
        "sampling_fallback": sampling_fallback,
        "top_p": top_p,
        "temperature": temperature,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device_name,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "requested_fp16": requested_fp16,
        "requested_bf16": requested_bf16,
        "local_files_only": local_files_only,
        "rows_selected": len(rows),
        "rows_processed_before_resume": len(processed_indices),
        "rows_written_this_run": rows_written,
        "rows_total_after_run": len(processed_indices) + rows_written,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
