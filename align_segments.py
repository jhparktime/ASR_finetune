#!/usr/bin/env python3
"""Align long-form WAV/JSON pairs into Whisper-trainable segments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType

from faster_whisper import WhisperModel

from dataset_utils import load_jsonl, normalize_text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
SPACE_RE = re.compile(r"\s+")
KEEP_TEXT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
NUM_JUNG = 21
NUM_JONG = 28
CHOSEONG = ["g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
JUNGSEONG = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
JONGSEONG = ["", "g", "gg", "gs", "n", "nj", "nh", "d", "r", "rg", "rm", "rb", "rs", "rt", "rp", "rh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]


@dataclass
class Chunk:
    start_sec: float
    end_sec: float
    text: str
    words: list[dict]


@dataclass
class AlignmentStep:
    action: str
    chunk_index: int | None
    sent_index: int | None
    group_size: int
    score: float


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
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--order-by", choices=("duration", "manifest"), default="duration")
    parser.add_argument("--model-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--max-group-sentences", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--fallback-word-chunk", type=int, default=None)
    parser.add_argument("--pause-boundary-ms", type=int, default=None)
    parser.add_argument("--skip-chunk-penalty", type=float, default=None)
    parser.add_argument("--skip-sentence-penalty", type=float, default=None)
    parser.add_argument("--write-audio", action="store_true")
    return parser.parse_args()


def split_transcript(text: str, fallback_word_chunk: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if re.search(r"[.!?。！？]", text):
        parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if parts:
            return parts
    words = text.split()
    return [
        " ".join(words[index : index + fallback_word_chunk])
        for index in range(0, len(words), fallback_word_chunk)
        if words[index : index + fallback_word_chunk]
    ]


def choose_records(records: list[dict], limit_files: int, start_index: int, order_by: str) -> list[dict]:
    if order_by == "duration":
        ordered = sorted(records, key=lambda row: (float(row["duration_sec"]), row["audio_path"]))
    else:
        ordered = list(records)
    if start_index < 0:
        start_index = 0
    if limit_files <= 0:
        return ordered[start_index:]
    return ordered[start_index : start_index + limit_files]


def decompose_hangul(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if HANGUL_BASE <= code <= HANGUL_END:
            offset = code - HANGUL_BASE
            cho = offset // (NUM_JUNG * NUM_JONG)
            jung = (offset % (NUM_JUNG * NUM_JONG)) // NUM_JONG
            jong = offset % NUM_JONG
            out.append(CHOSEONG[cho])
            out.append(JUNGSEONG[jung])
            out.append(JONGSEONG[jong])
        else:
            out.append(ch.lower())
    return "".join(out)


def normalize_match_text(text: str) -> str:
    text = KEEP_TEXT_RE.sub(" ", normalize_text(text))
    text = SPACE_RE.sub(" ", text).strip().lower()
    return decompose_hangul(text)


def normalized_tokens(text: str) -> list[str]:
    cleaned = KEEP_TEXT_RE.sub(" ", normalize_text(text)).lower()
    return [decompose_hangul(token) for token in cleaned.split() if token]


def sequence_score(asr_text: str, ref_text: str) -> float:
    asr_norm = normalize_match_text(asr_text)
    ref_norm = normalize_match_text(ref_text)
    if not asr_norm or not ref_norm:
        return 0.0
    char_ratio = SequenceMatcher(None, asr_norm, ref_norm).ratio()
    asr_tokens = set(normalized_tokens(asr_text))
    ref_tokens = set(normalized_tokens(ref_text))
    token_score = len(asr_tokens & ref_tokens) / max(1, len(ref_tokens)) if ref_tokens else 0.0
    return 0.8 * char_ratio + 0.2 * token_score


def segment_words(segment) -> list[dict]:
    words = []
    for word in getattr(segment, "words", []) or []:
        if word.start is None or word.end is None:
            continue
        text = normalize_text(getattr(word, "word", ""))
        if not text:
            continue
        words.append({"start": float(word.start), "end": float(word.end), "text": text})
    return words


def build_chunks(segments) -> list[Chunk]:
    chunks = []
    for segment in segments:
        text = normalize_text(segment.text)
        if not text:
            continue
        chunks.append(
            Chunk(
                start_sec=float(segment.start),
                end_sec=float(segment.end),
                text=text,
                words=segment_words(segment),
            )
        )
    return chunks


def build_subchunk(chunk: Chunk, start_sec: float, end_sec: float) -> Chunk:
    sub_words = [
        word
        for word in chunk.words
        if float(word["end"]) > start_sec and float(word["start"]) < end_sec
    ]
    if sub_words:
        sub_text = " ".join(word["text"] for word in sub_words)
    elif not chunk.words:
        sub_text = chunk.text
    else:
        sub_text = ""
    return Chunk(
        start_sec=start_sec,
        end_sec=end_sec,
        text=normalize_text(sub_text),
        words=sub_words,
    )


def group_match_score(chunk: Chunk, ref_sentences: list[str]) -> float:
    ref_text = " ".join(ref_sentences)
    text_score = sequence_score(chunk.text, ref_text)
    chunk_word_count = len(chunk.words) if chunk.words else max(1, len(chunk.text.split()))
    ref_word_count = max(1, len(ref_text.split()))
    word_count_score = 1.0 - min(abs(chunk_word_count - ref_word_count) / ref_word_count, 1.0)

    chunk_char_count = max(1, len(normalize_match_text(chunk.text)))
    ref_char_count = max(1, len(normalize_match_text(ref_text)))
    char_count_score = 1.0 - min(abs(chunk_char_count - ref_char_count) / ref_char_count, 1.0)

    duration = max(0.0, chunk.end_sec - chunk.start_sec)
    expected = ref_char_count / 7.0
    duration_score = 1.0 - min(abs(duration - expected) / max(expected, 1.0), 1.0)

    # Dysarthric speech often hurts exact lexical matching, so keep some text
    # signal but lean more on coarse alignment cues such as word count, length,
    # and duration shape.
    return (
        0.25 * text_score
        + 0.35 * word_count_score
        + 0.20 * char_count_score
        + 0.20 * duration_score
    )


def align_chunks_to_sentences(
    chunks: list[Chunk],
    sentences: list[str],
    max_group_sentences: int,
    min_score: float,
    skip_chunk_penalty: float,
    skip_sentence_penalty: float,
) -> list[AlignmentStep]:
    n = len(chunks)
    m = len(sentences)
    neg_inf = -10**9
    dp = [[neg_inf for _ in range(m + 1)] for _ in range(n + 1)]
    back: list[list[tuple[int, int, AlignmentStep] | None]] = [[None for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 0.0

    for chunk_index in range(n + 1):
        for sent_index in range(m + 1):
            current = dp[chunk_index][sent_index]
            if current <= neg_inf / 2:
                continue
            if chunk_index < n and current + skip_chunk_penalty > dp[chunk_index + 1][sent_index]:
                step = AlignmentStep("skip_chunk", chunk_index, None, 0, skip_chunk_penalty)
                dp[chunk_index + 1][sent_index] = current + skip_chunk_penalty
                back[chunk_index + 1][sent_index] = (chunk_index, sent_index, step)
            if sent_index < m and current + skip_sentence_penalty > dp[chunk_index][sent_index + 1]:
                step = AlignmentStep("skip_sentence", None, sent_index, 1, skip_sentence_penalty)
                dp[chunk_index][sent_index + 1] = current + skip_sentence_penalty
                back[chunk_index][sent_index + 1] = (chunk_index, sent_index, step)
            if chunk_index < n:
                max_group = min(max_group_sentences, m - sent_index)
                for group_size in range(1, max_group + 1):
                    score = group_match_score(chunks[chunk_index], sentences[sent_index : sent_index + group_size])
                    if score < min_score:
                        continue
                    new_score = current + score
                    if new_score > dp[chunk_index + 1][sent_index + group_size]:
                        step = AlignmentStep("match", chunk_index, sent_index, group_size, score)
                        dp[chunk_index + 1][sent_index + group_size] = new_score
                        back[chunk_index + 1][sent_index + group_size] = (chunk_index, sent_index, step)

    steps: list[AlignmentStep] = []
    chunk_index, sent_index = n, m
    while chunk_index != 0 or sent_index != 0:
        prev = back[chunk_index][sent_index]
        if prev is None:
            break
        prev_chunk, prev_sent, step = prev
        steps.append(step)
        chunk_index, sent_index = prev_chunk, prev_sent
    steps.reverse()
    return steps


def split_time_across_sentences(start_sec: float, end_sec: float, sentences: list[str]) -> list[tuple[float, float, str]]:
    if not sentences:
        return []
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return []
    weights = [max(1, len(normalize_match_text(sentence))) for sentence in sentences]
    total_weight = sum(weights)
    results = []
    cursor = start_sec
    for index, sentence in enumerate(sentences):
        next_cursor = end_sec if index == len(sentences) - 1 else cursor + duration * weights[index] / total_weight
        results.append((cursor, next_cursor, sentence))
        cursor = next_cursor
    return results


def split_with_pause_hints(chunk: Chunk, sentences: list[str], pause_boundary_sec: float) -> list[tuple[float, float, str]]:
    if len(sentences) <= 1 or len(chunk.words) < len(sentences):
        return split_time_across_sentences(chunk.start_sec, chunk.end_sec, sentences)

    ref_word_counts = [max(1, len(sentence.split())) for sentence in sentences]
    total_ref_words = sum(ref_word_counts)
    total_asr_words = len(chunk.words)
    boundaries = []
    prev_boundary_index = 0

    cumulative = 0
    for boundary_no, count in enumerate(ref_word_counts[:-1], start=1):
        cumulative += count
        target_ratio = cumulative / total_ref_words
        best_index = None
        best_score = None
        min_index = prev_boundary_index + 1
        max_index = total_asr_words - (len(sentences) - boundary_no)
        for index in range(min_index, max_index + 1):
            prev_word = chunk.words[index - 1]
            next_word = chunk.words[index]
            gap = max(0.0, next_word["start"] - prev_word["end"])
            ratio = index / total_asr_words
            closeness = 1.0 - min(abs(ratio - target_ratio) * 2.0, 1.0)
            pause_bonus = min(gap / pause_boundary_sec, 1.0) if pause_boundary_sec > 0 else 0.0
            score = 0.75 * closeness + 0.25 * pause_bonus
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            return split_time_across_sentences(chunk.start_sec, chunk.end_sec, sentences)
        boundaries.append(best_index)
        prev_boundary_index = best_index

    results = []
    word_ranges = [0] + boundaries + [len(chunk.words)]
    for index, sentence in enumerate(sentences):
        start_index = word_ranges[index]
        end_index = word_ranges[index + 1]
        if start_index >= end_index:
            return split_time_across_sentences(chunk.start_sec, chunk.end_sec, sentences)
        seg_start = chunk.start_sec if index == 0 else max(chunk.start_sec, chunk.words[start_index]["start"])
        seg_end = chunk.end_sec if index == len(sentences) - 1 else min(chunk.end_sec, chunk.words[end_index - 1]["end"])
        if seg_end <= seg_start:
            return split_time_across_sentences(chunk.start_sec, chunk.end_sec, sentences)
        results.append((seg_start, seg_end, sentence))
    return results


def copy_wav_segment(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
    with wave.open(str(src), "rb") as in_wav:
        params = in_wav.getparams()
        frame_rate = in_wav.getframerate()
        start_frame = max(0, int(start_sec * frame_rate))
        end_frame = min(in_wav.getnframes(), int(end_sec * frame_rate))
        frame_count = max(0, end_frame - start_frame)
        in_wav.setpos(start_frame)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as out_wav:
            out_wav.setparams(params)
            remaining = frame_count
            chunk_frames = frame_rate * 10
            while remaining > 0:
                to_read = min(chunk_frames, remaining)
                raw = in_wav.readframes(to_read)
                if not raw:
                    break
                out_wav.writeframes(raw)
                remaining -= to_read


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    manifest_path = args.manifest or (Path(config.FILE_MANIFEST_DIR) / "all_files.jsonl")
    out_dir = args.out_dir or Path(config.ALIGNMENT_DIR)
    model_size = args.model_size or config.ALIGN_MODEL_SIZE
    device = args.device or config.ALIGN_DEVICE
    compute_type = args.compute_type or config.ALIGN_COMPUTE_TYPE
    beam_size = int(args.beam_size if args.beam_size is not None else config.ALIGN_BEAM_SIZE)
    max_group_sentences = int(args.max_group_sentences if args.max_group_sentences is not None else config.ALIGN_MAX_GROUP_SENTENCES)
    min_score = float(args.min_score if args.min_score is not None else config.ALIGN_MIN_SCORE)
    fallback_word_chunk = int(args.fallback_word_chunk if args.fallback_word_chunk is not None else config.ALIGN_FALLBACK_WORD_CHUNK)
    pause_boundary_ms = int(args.pause_boundary_ms if args.pause_boundary_ms is not None else config.ALIGN_PAUSE_BOUNDARY_MS)
    skip_chunk_penalty = float(args.skip_chunk_penalty if args.skip_chunk_penalty is not None else config.ALIGN_SKIP_CHUNK_PENALTY)
    skip_sentence_penalty = float(args.skip_sentence_penalty if args.skip_sentence_penalty is not None else config.ALIGN_SKIP_SENTENCE_PENALTY)
    write_audio = bool(args.write_audio or config.ALIGN_WRITE_AUDIO)

    records = load_jsonl(manifest_path)
    pilot_records = choose_records(records, args.limit_files, args.start_index, args.order_by)
    if not pilot_records:
        print("No input records found.", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "segments_manifest.jsonl"
    out_report = out_dir / "report.json"
    out_audio_dir = out_dir / "segments"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    file_reports: list[dict] = []
    output_segments = 0

    with out_manifest.open("w", encoding="utf-8") as manifest_out:
        for record in pilot_records:
            audio_path = Path(record["audio_path"])
            sentences = split_transcript(str(record["text"]), fallback_word_chunk)
            if not sentences:
                continue

            segments, info = model.transcribe(
                str(audio_path),
                language="ko",
                beam_size=beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
            chunks = build_chunks(list(segments))
            steps = align_chunks_to_sentences(
                chunks=chunks,
                sentences=sentences,
                max_group_sentences=max_group_sentences,
                min_score=min_score,
                skip_chunk_penalty=skip_chunk_penalty,
                skip_sentence_penalty=skip_sentence_penalty,
            )

            matched_sentences = 0
            matched_chunks = 0
            skipped_chunks = 0
            skipped_sentences = 0

            for step_index, step in enumerate(steps, start=1):
                if step.action == "skip_chunk":
                    skipped_chunks += 1
                    continue
                if step.action == "skip_sentence":
                    skipped_sentences += 1
                    continue

                chunk = chunks[step.chunk_index]
                ref_sentences = sentences[step.sent_index : step.sent_index + step.group_size]
                matched_chunks += 1
                matched_sentences += step.group_size

                split_ranges = split_with_pause_hints(chunk, ref_sentences, pause_boundary_ms / 1000.0)
                for sub_index, (sent_start, sent_end, sent_text) in enumerate(split_ranges, start=1):
                    if sent_end <= sent_start:
                        continue
                    subchunk = build_subchunk(chunk, sent_start, sent_end)
                    subchunk_score = round(group_match_score(subchunk, [sent_text]), 4)
                    if write_audio:
                        segment_name = f"{audio_path.stem}_dp_{step_index:04d}_{sub_index:02d}.wav"
                        segment_path = out_audio_dir / audio_path.stem / segment_name
                        copy_wav_segment(audio_path, segment_path, sent_start, sent_end)
                        segment_audio_path = str(segment_path.resolve())
                    else:
                        segment_audio_path = ""

                    row = {
                        "audio_path": segment_audio_path,
                        "text": sent_text,
                        "asr_text": subchunk.text,
                        "score": subchunk_score,
                        "chunk_asr_text": chunk.text,
                        "chunk_score": round(step.score, 4),
                        "start_sec": round(sent_start, 3),
                        "end_sec": round(sent_end, 3),
                        "duration_sec": round(sent_end - sent_start, 3),
                        "too_long_for_whisper": (sent_end - sent_start) > float(config.MAX_SEGMENT_DURATION_SEC),
                        "source_audio_path": str(audio_path.resolve()),
                        "source_label_path": str(record["label_path"]),
                        "source_segment_index": step.chunk_index + 1,
                        "sentence_count_in_group": step.group_size,
                        "asr_word_count": len(subchunk.words),
                        "chunk_asr_word_count": len(chunk.words),
                        "detected_language": getattr(info, "language", ""),
                        "detected_language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
                        "source_split": record.get("source_split", ""),
                        "category_code": record.get("category_code", ""),
                        "category_name": record.get("category_name", ""),
                        "severity_label": record.get("severity_label", ""),
                        "speaker_id": record.get("speaker_id", ""),
                        "speaker_code": record.get("speaker_code", ""),
                        "sex": record.get("sex", ""),
                        "age": record.get("age", ""),
                        "area": record.get("area", ""),
                    }
                    json.dump(row, manifest_out, ensure_ascii=False)
                    manifest_out.write("\n")
                    output_segments += 1

            file_reports.append(
                {
                    "audio_path": str(audio_path.resolve()),
                    "duration_sec": record["duration_sec"],
                    "reference_sentence_count": len(sentences),
                    "whisper_segment_count": len(chunks),
                    "matched_reference_sentences": matched_sentences,
                    "coverage_ratio": round(matched_sentences / len(sentences), 4) if sentences else 0.0,
                    "matched_whisper_segments": matched_chunks,
                    "skipped_whisper_segments": skipped_chunks,
                    "skipped_reference_sentences": skipped_sentences,
                }
            )

    summary = {
        "processed_files": len(file_reports),
        "output_segments": output_segments,
        "avg_coverage_ratio": round(sum(item["coverage_ratio"] for item in file_reports) / len(file_reports), 4)
        if file_reports
        else 0.0,
        "files": file_reports,
        "manifest_path": str(out_manifest.resolve()),
        "segments_dir": str(out_audio_dir.resolve()) if write_audio else "",
        "model_size": model_size,
        "device": device,
        "compute_type": compute_type,
        "write_audio": write_audio,
    }
    out_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["processed_files", "output_segments", "avg_coverage_ratio"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
