#!/usr/bin/env python3
"""Build an N-best inference dataset from a segment manifest."""

from __future__ import annotations

import argparse
import json
import time
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset_utils import load_jsonl
from infer_nbest import dedupe_candidates, load_audio, load_audio_segment
from train_lora import load_config, maybe_limit


class FeatureShardCache:
    def __init__(self, max_open_shards: int = 2) -> None:
        self.max_open_shards = max_open_shards
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.local_paths: dict[str, Path] = {}
        
        # 고유한 RAM Disk 폴더 생성 (PID 기반으로 겹치지 않게)
        self.shm_dir = Path(f"/dev/shm/asr_shards_pid_{os.getpid()}")
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SHM Cache] Init: Temporary RAM dir created at {self.shm_dir}", flush=True)

    def get(self, shard_path: str) -> torch.Tensor:
        shard_path = str(shard_path)
        if shard_path in self.cache:
            # 캐시 히트: 최신 사용 항목으로 갱신
            tensor = self.cache.pop(shard_path)
            self.cache[shard_path] = tensor
            return tensor

        original_path = Path(shard_path)
        # gpu0, gpu1 폴더명이 겹치지 않도록 부모 디렉토리명도 파일명에 포함
        local_name = f"{original_path.parent.name}_{original_path.name}"
        local_path = self.shm_dir / local_name

        # RAM 디스크에 파일이 없다면 복사
        if not local_path.exists():
            print(f"\n[SHM Cache] Loading from NFS... Copying {original_path.name} to RAM disk.", flush=True)
            t0 = time.time()
            shutil.copy2(original_path, local_path)
            print(f"[SHM Cache] Copy complete in {time.time() - t0:.2f}s.", flush=True)

        # 복사된 RAM 디스크에서 초고속 로드 (mmap=True 적용)
        item = torch.load(local_path, map_location="cpu", mmap=True)
        tensor = item["input_features"].float()

        self.cache[shard_path] = tensor
        self.local_paths[shard_path] = local_path

        # 캐시 용량(max_open_shards) 초과 시 가장 오래된 것 삭제
        if len(self.cache) > self.max_open_shards:
            evicted_key, _ = self.cache.popitem(last=False)
            evicted_local_path = self.local_paths.pop(evicted_key, None)
            if evicted_local_path and evicted_local_path.exists():
                print(f"[SHM Cache] Evicting memory. Deleting {evicted_local_path.name} from RAM disk.", flush=True)
                evicted_local_path.unlink()

        return tensor

    def cleanup(self):
        """프로세스 종료 시 RAM 디스크 완전 싹쓸이"""
        if self.shm_dir.exists():
            print(f"[SHM Cache] Cleaning up RAM dir: {self.shm_dir}", flush=True)
            shutil.rmtree(self.shm_dir, ignore_errors=True)


class SegmentInferenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        processor: WhisperProcessor,
        sampling_rate: int,
        max_open_feature_shards: int = 2,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.feature_cache = FeatureShardCache(max_open_shards=max_open_feature_shards)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]

        feature_shard_path = str(row.get("feature_shard_path", "")).strip()
        if feature_shard_path:
            shard_tensor = self.feature_cache.get(feature_shard_path)
            feature_index = int(row["feature_index_in_shard"])
            input_features = shard_tensor[feature_index]
        else:
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
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True,
        )
        batch["rows"] = [feature["row"] for feature in features]
        batch["manifest_indices"] = [feature["manifest_index"] for feature in features]
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.py"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-open-feature-shards", type=int, default=2)
    parser.add_argument("--reorder-for-locality", action="store_true")
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


def chunk_by_return_sequences(items: list[str], chunk_size: int) -> list[list[str]]:
    grouped = []
    for start in range(0, len(items), chunk_size):
        grouped.append(items[start : start + chunk_size])
    return grouped


def add_grouped_candidates(
    candidates_per_row: list[list[dict]],
    grouped_texts: list[list[str]],
    source: str,
) -> list[list[dict]]:
    for row_index, texts in enumerate(grouped_texts):
        candidates_per_row[row_index] = dedupe_candidates(
            candidates_per_row[row_index],
            texts,
            None,
            source=source,
        )
    return candidates_per_row


def decode_batch(
    *,
    model,
    processor: WhisperProcessor,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_beams: int,
    num_return_sequences: int,
    max_length: int,
    sampling_fallback: bool,
    top_p: float,
    temperature: float,
) -> list[list[dict]]:
    candidates_per_row = [[] for _ in range(input_features.size(0))]

    generate_kwargs = dict(
        input_features=input_features,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        return_dict_in_generate=True,
    )
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    with torch.inference_mode():
        beam_outputs = model.generate(**generate_kwargs)

    beam_texts = processor.tokenizer.batch_decode(beam_outputs.sequences, skip_special_tokens=True)
    grouped_beam = chunk_by_return_sequences(beam_texts, num_return_sequences)
    candidates_per_row = add_grouped_candidates(candidates_per_row, grouped_beam, source="beam")

    # ==========================================================
    # [핵심 수정] 2차 샘플링 전에 1차 빔 서치의 잔여 메모리를 폭파시킵니다!
    del beam_outputs
    torch.cuda.empty_cache()
    # ==========================================================

    if sampling_fallback and any(len(items) < num_return_sequences for items in candidates_per_row):
        sample_count = max(num_return_sequences * 2, 4)
        sample_kwargs = dict(
            input_features=input_features,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_beams=1,
            num_return_sequences=sample_count,
            max_length=max_length,
            return_dict_in_generate=True,
        )
        if attention_mask is not None:
            sample_kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            sample_outputs = model.generate(**sample_kwargs)

        sample_texts = processor.tokenizer.batch_decode(sample_outputs.sequences, skip_special_tokens=True)
        grouped_sample = chunk_by_return_sequences(sample_texts, sample_count)
        candidates_per_row = add_grouped_candidates(candidates_per_row, grouped_sample, source="sample")
        
        # [추가] 샘플링 이후에도 바로 메모리 정리
        del sample_outputs
        torch.cuda.empty_cache()

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

    if args.reorder_for_locality:
        def sort_key(row: dict):
            return (
                str(row.get("feature_shard_path", "")),
                int(row.get("feature_index_in_shard", 0)),
                int(row["manifest_index"]),
            )
        selected = sorted(selected, key=sort_key)

    return selected


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        print(json.dumps({"status": "nothing_to_do"}, ensure_ascii=False))
        return 0

    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=task, local_files_only=local_files_only
    )

    model_kwargs = {"local_files_only": local_files_only}
    if device_name.startswith("cuda"):
        if use_bf16: model_kwargs["torch_dtype"] = torch.bfloat16
        elif use_fp16: model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["attn_implementation"] = "sdpa"

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
        max_open_feature_shards=args.max_open_feature_shards,
    )

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=SegmentInferenceCollator(processor),
        pin_memory=device_name.startswith("cuda"),
    )
    dataloader = DataLoader(**dataloader_kwargs)

    rows_written = 0
    start_time = time.time()
    last_log_time = 0.0
    warmup_rows = max(args.batch_size * 16, 128)

    try:
        with output_path.open("a", encoding="utf-8") as outfile:
            for batch in dataloader:
                input_features = batch["input_features"].to(device_name, non_blocking=True)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device_name, non_blocking=True)

                if use_bf16: input_features = input_features.to(torch.bfloat16)
                elif use_fp16: input_features = input_features.to(torch.float16)

                candidates_per_row = decode_batch(
                    model=model,
                    processor=processor,
                    input_features=input_features,
                    attention_mask=attention_mask,
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

                # [OOM 방지] 매 배치마다 텐서 참조 해제 및 캐시 비우기
                del input_features
                if attention_mask is not None:
                    del attention_mask
                torch.cuda.empty_cache()

                now = time.time()
                if now - last_log_time >= 10 or rows_written == len(rows_to_decode):
                    progress = rows_written / len(rows_to_decode) * 100.0
                    if rows_written >= warmup_rows:
                        elapsed = max(now - start_time, 1e-6)
                        rows_per_sec = rows_written / elapsed
                        remaining = len(rows_to_decode) - rows_written
                        eta_seconds = remaining / rows_per_sec if rows_per_sec > 0 else 0.0
                        eta_text = format_eta(eta_seconds)
                    else:
                        eta_text = "warming_up"

                    print(f"[progress] {rows_written}/{len(rows_to_decode)} ({progress:.2f}%) | ETA {eta_text}", flush=True)
                    last_log_time = now

    finally:
        # 정상 종료든 에러(Ctrl+C)든 무조건 RAM 디스크 정리
        dataset.feature_cache.cleanup()

    meta = {
        "manifest": str(manifest_path.resolve()),
        "output_path": str(output_path.resolve()),
        "model_name": model_name,
        "rows_written_this_run": rows_written,
        "device": device_name,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())