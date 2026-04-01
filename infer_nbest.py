#!/usr/bin/env python3
"""Run N-best decoding with a base Whisper model or a LoRA-adapted checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import librosa
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


SPACE_RE = re.compile(r"\s+")


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
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--sampling-rate", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=None)
    parser.add_argument("--num-beam-groups", type=int, default=None)
    parser.add_argument("--diversity-penalty", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sampling-fallback", action="store_true")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_audio(path: Path, target_sr: int):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def load_audio_segment(path: Path, start_sec: float, end_sec: float, target_sr: int):
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


def normalize_candidate_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def dedupe_candidates(existing: list[dict], texts: list[str], seq_scores, source: str) -> list[dict]:
    seen = {normalize_candidate_text(item["text"]) for item in existing}
    next_rank = len(existing) + 1
    for index, text in enumerate(texts):
        normalized = normalize_candidate_text(text)
        if not normalized or normalized in seen:
            continue
        item = {"rank": next_rank, "text": normalized, "source": source}
        if seq_scores is not None:
            item["score"] = round(float(seq_scores[index]), 4)
        existing.append(item)
        seen.add(normalized)
        next_rank += 1
    return existing


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    model_name = args.model_name or config.MODEL_NAME
    language = args.language or config.LANGUAGE
    task = args.task or config.TASK
    sampling_rate = int(args.sampling_rate or config.SAMPLING_RATE)
    num_beams = int(args.num_beams or config.N_BEST_BEAMS)
    num_return_sequences = int(args.num_return_sequences or config.N_BEST_RETURN_SEQUENCES)
    num_beam_groups = int(args.num_beam_groups or config.N_BEST_NUM_BEAM_GROUPS)
    diversity_penalty = float(args.diversity_penalty or config.N_BEST_DIVERSITY_PENALTY)
    max_length = int(args.max_length or config.N_BEST_MAX_LENGTH)
    sampling_fallback = bool(args.sampling_fallback or config.N_BEST_SAMPLING_FALLBACK)
    top_p = float(args.top_p or config.N_BEST_TOP_P)
    temperature = float(args.temperature or config.N_BEST_TEMPERATURE)
    local_files_only = bool(args.local_files_only or config.LOCAL_FILES_ONLY)
    effective_num_beams = max(num_beams, num_return_sequences)
    effective_num_beam_groups = min(num_beam_groups, effective_num_beams, num_return_sequences)
    while effective_num_beam_groups > 1 and effective_num_beams % effective_num_beam_groups != 0:
        effective_num_beam_groups -= 1
    if effective_num_beam_groups <= 1:
        effective_num_beam_groups = 1
        diversity_penalty = 0.0

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

    model.eval()
    model.generation_config.language = language
    model.generation_config.task = task

    if args.start_sec is not None and args.end_sec is not None:
        audio = load_audio_segment(args.audio_path, float(args.start_sec), float(args.end_sec), sampling_rate)
    else:
        audio = load_audio(args.audio_path, sampling_rate)
    inputs = processor.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")

    candidates = []
    with torch.no_grad():
        beam_outputs = model.generate(
            input_features=inputs.input_features,
            num_beams=effective_num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=effective_num_beam_groups,
            diversity_penalty=diversity_penalty,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
        )
    beam_texts = processor.tokenizer.batch_decode(beam_outputs.sequences, skip_special_tokens=True)
    beam_scores = getattr(beam_outputs, "sequences_scores", None)
    candidates = dedupe_candidates(candidates, beam_texts, beam_scores, source="beam")

    if sampling_fallback and len(candidates) < num_return_sequences:
        sample_count = max(num_return_sequences * 2, 4)
        with torch.no_grad():
            sample_outputs = model.generate(
                input_features=inputs.input_features,
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
        candidates = dedupe_candidates(candidates, sample_texts, sample_scores, source="sample")
    candidates = candidates[:num_return_sequences]

    payload = {
        "audio_path": str(args.audio_path.resolve()),
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "model_name": model_name,
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else "",
        "num_beams": effective_num_beams,
        "num_return_sequences": num_return_sequences,
        "num_beam_groups": effective_num_beam_groups,
        "diversity_penalty": diversity_penalty,
        "sampling_fallback": sampling_fallback,
        "candidates": candidates,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
