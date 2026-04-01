#!/usr/bin/env python3
"""Train Whisper with LoRA on aligned segment-level manifests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import librosa
import soundfile as sf
import torch
from jiwer import cer, wer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from dataset_utils import load_jsonl


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
        help="Path to the project config file.",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--sampling-rate", type=int, default=None)
    parser.add_argument("--max-label-length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def maybe_limit(rows: list[dict], limit: int) -> list[dict]:
    if limit > 0:
        return rows[:limit]
    return rows


def load_audio(path: str, target_sr: int):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def load_audio_segment(path: str, start_sec: float, end_sec: float, target_sr: int):
    info = sf.info(path)
    start_frame = max(0, int(start_sec * info.samplerate))
    end_frame = min(info.frames, int(end_sec * info.samplerate))
    frames = max(0, end_frame - start_frame)
    audio, sr = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


class RawUtteranceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        processor: WhisperProcessor,
        sampling_rate: int,
        max_label_length: int,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_label_length = max_label_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio_path = str(row.get("audio_path", "")).strip()
        if audio_path:
            audio = load_audio(audio_path, self.sampling_rate)
        else:
            audio = load_audio_segment(
                row["source_audio_path"],
                float(row["start_sec"]),
                float(row["end_sec"]),
                self.sampling_rate,
            )

        # Default mode keeps only segment metadata and slices source audio at
        # read time. Exported segment WAV files remain optional for inspection.
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_label_length,
        ).input_ids
        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and labels.size(1) > 0 and torch.all(labels[:, 0] == bos_id):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": wer(label_str, pred_str),
            "cer": cer(label_str, pred_str),
        }

    return compute_metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    model_name = args.model_name or config.MODEL_NAME
    output_dir = args.output_dir or Path(config.TRAINING_OUTPUT_DIR)
    language = args.language or config.LANGUAGE
    task = args.task or config.TASK
    sampling_rate = int(args.sampling_rate or config.SAMPLING_RATE)
    max_label_length = int(args.max_label_length or config.MAX_LABEL_LENGTH)
    per_device_train_batch_size = int(args.per_device_train_batch_size or config.PER_DEVICE_TRAIN_BATCH_SIZE)
    per_device_eval_batch_size = int(args.per_device_eval_batch_size or config.PER_DEVICE_EVAL_BATCH_SIZE)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps or config.GRADIENT_ACCUMULATION_STEPS)
    learning_rate = float(args.learning_rate or config.LEARNING_RATE)
    num_train_epochs = float(args.num_train_epochs or config.NUM_TRAIN_EPOCHS)
    warmup_ratio = float(args.warmup_ratio or config.WARMUP_RATIO)
    logging_steps = int(args.logging_steps or config.LOGGING_STEPS)
    eval_steps = int(args.eval_steps or config.EVAL_STEPS)
    save_steps = int(args.save_steps or config.SAVE_STEPS)
    seed = int(args.seed or config.SEED)
    lora_r = int(args.lora_r or config.LORA_R)
    lora_alpha = int(args.lora_alpha or config.LORA_ALPHA)
    lora_dropout = float(args.lora_dropout or config.LORA_DROPOUT)
    fp16 = bool(args.fp16 or config.FP16)
    bf16 = bool(args.bf16 or config.BF16)
    gradient_checkpointing = bool(args.gradient_checkpointing or config.GRADIENT_CHECKPOINTING)
    local_files_only = bool(args.local_files_only or config.LOCAL_FILES_ONLY)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = maybe_limit(load_jsonl(args.train_manifest), args.max_train_samples)
    eval_rows = maybe_limit(load_jsonl(args.eval_manifest), args.max_eval_samples)
    test_rows = maybe_limit(load_jsonl(args.test_manifest), args.max_test_samples) if args.test_manifest else []

    if not train_rows:
        raise ValueError(f"Train manifest is empty: {args.train_manifest}")
    if not eval_rows:
        raise ValueError(f"Eval manifest is empty: {args.eval_manifest}")

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
    model.generation_config.language = language
    model.generation_config.task = task
    model.config.use_cache = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(config.LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, lora_config)

    train_dataset = RawUtteranceDataset(
        train_rows,
        processor=processor,
        sampling_rate=sampling_rate,
        max_label_length=max_label_length,
    )
    eval_dataset = RawUtteranceDataset(
        eval_rows,
        processor=processor,
        sampling_rate=sampling_rate,
        max_label_length=max_label_length,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        predict_with_generate=True,
        generation_max_length=max_label_length,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        report_to=[],
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperDataCollator(processor),
        tokenizer=processor.feature_extractor,
        compute_metrics=build_metrics(processor),
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
    metrics = {
        "model_name": model_name,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "test_rows": len(test_rows),
        "trainable_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "total_params": sum(param.numel() for param in model.parameters()),
        **eval_metrics,
    }

    if test_rows:
        test_dataset = RawUtteranceDataset(
            test_rows,
            processor=processor,
            sampling_rate=sampling_rate,
            max_label_length=max_label_length,
        )
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        metrics.update(test_metrics)

    processor.save_pretrained(output_dir)
    trainer.save_model()

    runtime_summary = {
        "model_name": model_name,
        "language": language,
        "task": task,
        "sampling_rate": sampling_rate,
        "max_label_length": max_label_length,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "warmup_ratio": warmup_ratio,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "seed": seed,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": list(config.LORA_TARGET_MODULES),
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "local_files_only": local_files_only,
        "train_manifest": str(args.train_manifest.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "test_manifest": str(args.test_manifest.resolve()) if args.test_manifest else "",
    }

    (output_dir / "run_config.json").write_text(json.dumps(runtime_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
