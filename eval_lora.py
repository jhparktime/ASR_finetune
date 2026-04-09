#!/usr/bin/env python3
"""Evaluate Whisper or Whisper+LoRA on a manifest and report WER/CER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from train_lora import (
    RawUtteranceDataset,
    WhisperDataCollator,
    build_metrics,
    load_config,
    maybe_limit,
    maybe_load_split_summary,
)
from dataset_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.py"),
        help="Path to the project config file.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for temporary trainer outputs.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save evaluation metrics.")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--sampling-rate", type=int, default=None)
    parser.add_argument("--max-label-length", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--metric-key-prefix", default="test")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    model_name = args.model_name or config.MODEL_NAME
    language = args.language or config.LANGUAGE
    task = args.task or config.TASK
    sampling_rate = int(args.sampling_rate or config.SAMPLING_RATE)
    max_label_length = int(args.max_label_length or config.MAX_LABEL_LENGTH)
    normalize_train_text = bool(getattr(config, "NORMALIZE_TRAIN_TEXT", False))
    per_device_eval_batch_size = int(args.per_device_eval_batch_size or config.PER_DEVICE_EVAL_BATCH_SIZE)
    fp16 = bool(args.fp16 or config.FP16)
    bf16 = bool(args.bf16 or config.BF16)
    local_files_only = bool(args.local_files_only or config.LOCAL_FILES_ONLY)
    output_dir = args.output_dir or (Path(config.ARTIFACT_ROOT) / "eval_outputs" / args.metric_key_prefix)

    rows = maybe_limit(load_jsonl(args.manifest), args.max_samples)
    if not rows:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    split_summary = maybe_load_split_summary(args.manifest, args.manifest, None)

    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task,
        local_files_only=local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    if args.adapter_path is not None and (args.adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(args.adapter_path))

    model.generation_config.language = language
    model.generation_config.task = task

    eval_dataset = RawUtteranceDataset(
        rows,
        processor=processor,
        sampling_rate=sampling_rate,
        max_label_length=max_label_length,
        normalize_train_text=normalize_train_text,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=max_label_length,
        generation_num_beams=1,
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=WhisperDataCollator(processor),
        tokenizer=processor.feature_extractor,
        compute_metrics=build_metrics(processor),
    )

    metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=args.metric_key_prefix)
    payload = {
        "model_name": model_name,
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else "",
        "manifest": str(args.manifest.resolve()),
        "rows": len(rows),
        "language": language,
        "task": task,
        "sampling_rate": sampling_rate,
        "max_label_length": max_label_length,
        "normalize_train_text": normalize_train_text,
        "metric_normalization": {
            "wer": "normalize_asr_text",
            "cer": "normalize_asr_text(remove_space=True)",
            "raw_metrics_logged": True,
        },
        **metrics,
    }
    if split_summary:
        payload["split_summary_path"] = split_summary.pop("_summary_path", "")
        payload["split_summary"] = split_summary

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
