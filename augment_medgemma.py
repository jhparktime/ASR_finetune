import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = """당신은 의료 지식이 있는 LLM을 학습시키기 위한 가상 의료 문진 데이터를 생성하는 전문가입니다.
목표는 구음장애 환자 또는 보호자의 발화를 ASR(음성인식)이 교정한 텍스트를 입력으로 받고, 의사가 보기 좋은 구조화된 문진폼을 출력하는 학습 데이터를 만드는 것입니다.

주어진 [의학 문제]의 의학 지식과 정답을 바탕으로, 실제 환자/보호자가 외래, 응급실, 치과, 상담 진료에서 말할 법한 '환자 진술(input)'과 의사용 '구조화된 의료 문진표(output)'를 생성해 주세요.

[조건]
1. input은 ASR로 교정된 구음장애 환자/보호자의 자연스러운 구어체 진술이어야 합니다. 증상, 불편감, 걱정, 복용 중인 약, 이미 들은 검사 이상 정도만 말하게 하세요.
2. input에서 시험문제 흔적을 제거하세요. "무슨 검사가 필요한가요?", "가장 적절한 치료는 무엇인가요?", "정답이 뭔가요?" 같은 문제풀이식 질문을 만들지 마세요.
3. input에는 원본 정답이 되는 진단명, 검사명, 약물명, 치료명을 가능한 한 직접 쓰지 마세요. 단, 환자가 이미 진단받았거나 복용 중이거나 검사 결과를 들었다는 설정이면 자연스럽게 포함할 수 있습니다.
4. output은 반드시 하나의 문자열(string) 값으로 작성하고, 그 문자열 안에 CC:, PI:, A&P: 구조를 포함하세요. output을 JSON 객체, dict, list, 중첩 구조로 만들지 마세요.
5. CC와 PI는 input에서 확인되는 환자 진술만 요약하고, input에 없는 검사/진찰 결과를 PI에 새로 만들어 넣지 마세요.
6. 원본 문제의 정답 또는 핵심 이론은 A&P에 의사의 평가와 계획으로 자연스럽게 반영하세요. 필요한 경우 감별진단, 권장 검사, 치료 방향을 제시하되 실제 임상 판단처럼 표현하세요.
7. 순수 이론 문제(예: 유전 양식, 약리 기전, 해부학 등)는 해당 질환을 의심받아 내원한 환자/보호자 상황으로 바꾸세요. 그래도 input은 환자 호소와 걱정 중심이어야 합니다.
8. 정말로 의학적 맥락으로 도저히 변환이 불가능한 쓰레기 데이터인 경우에만 예외적으로 "SKIP"을 출력하세요.
9. 절대 Thinking Process나 부연 설명을 작성하지 마세요. 반드시 아래의 [출력 JSON 형식]에 맞는 순수 JSON 코드만 출력해야 합니다.
10. 모든 출력은 특정 영어로만 표현이 가능한 단어를 제외하고 한국어로만 작성하세요.

[잘못된 output 예시]
"output": {"CC": "...", "PI": "...", "A&P": {"평가": "...", "계획": "..."}}

[올바른 output 예시]
"output": "CC: ...\nPI: ...\nA&P: ..."

[출력 JSON 형식]
{
  "instruction": "다음 환자의 진술을 바탕으로 핵심 증상을 파악하고, 의사가 보기 편한 구조화된 의료 문진표를 작성하세요.",
  "input": "[환자/보호자의 자연스러운 증상 중심 진술]",
  "output": "CC: ...\nPI: ...\nA&P: ..."
}"""


DEFAULT_INSTRUCTION = "다음 환자의 진술을 바탕으로 핵심 증상을 파악하고, 의사가 보기 편한 구조화된 의료 문진표를 작성하세요."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedGemma를 사용해 raw_medical_data.jsonl을 증강합니다."
    )
    parser.add_argument(
        "--input-file",
        default="raw_medical_data.jsonl",
        help="입력 JSONL 경로",
    )
    parser.add_argument(
        "--output-file",
        default="augmented_medical_medgemma.jsonl",
        help="출력 JSONL 경로",
    )
    parser.add_argument(
        "--model-name",
        default="google/medgemma-27b-text-it",
        help="기본값은 텍스트 전용 MedGemma 27B입니다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="한 번에 생성할 샘플 수. 27B는 1부터 시작하는 편이 안전합니다.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="샘플당 최대 생성 토큰 수",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="생성 온도",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus sampling top-p",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서부터 일부 샘플만 테스트할 때 사용",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="입력 데이터의 시작 인덱스",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="모델 로딩 dtype",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map 값. 기본 auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Hugging Face 캐시에서만 모델을 읽고 네트워크 접근을 막습니다.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="bitsandbytes 4bit 양자화로 모델을 로드합니다. Colab GPU에서 권장됩니다.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="bitsandbytes 8bit 양자화로 모델을 로드합니다.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="4bit 양자화 사용 시 연산 dtype",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="출력 파일이 이미 있어도 덮어씁니다.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 출력과 progress 파일을 기준으로 이어서 실행합니다.",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="resume용 진행 상태 JSON 경로. 기본값은 출력 파일 경로에 .progress.json을 붙입니다.",
    )
    parser.add_argument(
        "--invalid-log-file",
        default=None,
        help="invalid/skip 생성 결과를 저장할 JSONL 경로. 기본값은 출력 파일 경로에 .invalid.jsonl을 붙입니다.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="필요하면 HF 토큰을 직접 전달합니다. 미지정 시 환경변수 HF_TOKEN을 사용합니다.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="필요한 경우 remote code를 신뢰합니다.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def resolve_bnb_compute_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name != "auto":
        return getattr(torch, dtype_name)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("--load-in-4bit과 --load-in-8bit는 동시에 사용할 수 없습니다.")
    if not args.load_in_4bit and not args.load_in_8bit:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes 양자화는 CUDA GPU 환경에서만 사용 가능합니다.")
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_bnb_compute_dtype(args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    response_text = response_text.strip()
    if response_text == "SKIP" or response_text.startswith("SKIP"):
        return "SKIP"  # type: ignore[return-value]

    decoder = json.JSONDecoder()
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    candidates.append(response_text)

    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def get_case_insensitive(data: Dict[str, Any], *names: str) -> Any:
    wanted = {name.lower() for name in names}
    for key, value in data.items():
        if str(key).lower() in wanted:
            return value
    return None


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = flatten_text(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "\n".join(parts).strip()
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value if flatten_text(item)).strip()
    return str(value).strip()


def normalize_output_text(value: Any) -> str:
    if isinstance(value, dict):
        cc = flatten_text(get_case_insensitive(value, "CC", "chief_complaint", "주소증"))
        pi = flatten_text(get_case_insensitive(value, "PI", "present_illness", "현병력"))
        ap = flatten_text(
            get_case_insensitive(value, "A&P", "AP", "assessment_plan", "평가 및 계획")
        )
        sections = []
        if cc:
            sections.append(f"CC: {cc}")
        if pi:
            sections.append(f"PI: {pi}")
        if ap:
            sections.append(f"A&P: {ap}")
        if sections:
            return "\n".join(sections)
    return flatten_text(value)


def has_required_output_sections(output_text: str) -> bool:
    return (
        output_text.startswith("CC:")
        and "\nPI:" in output_text
        and "\nA&P:" in output_text
    )


def is_valid_data(json_data: Dict[str, Any]) -> bool:
    if not isinstance(json_data, dict):
        return False

    keys_lower = {str(key).lower() for key in json_data.keys()}
    required_keys = {"instruction", "input", "output"}
    if not required_keys.issubset(keys_lower):
        return False

    if "qa_id" in keys_lower or "status" in keys_lower:
        return False

    instruction = flatten_text(get_case_insensitive(json_data, "instruction"))
    input_text = flatten_text(get_case_insensitive(json_data, "input"))
    output_text = normalize_output_text(get_case_insensitive(json_data, "output"))

    for text in [instruction, input_text, output_text]:
        if len(text) < 5:
            return False
        if text == "..." or text.upper() == "SKIP":
            return False
        if "[환자의 자연스러운 호소 문장]" in text:
            return False
        if "[구조화된 문진표]" in text:
            return False

    if not has_required_output_sections(output_text):
        return False

    return True


def normalize_result(result: Dict[str, Any], source_item: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "instruction": (
            result.get("instruction")
            or result.get("Instruction")
            or result.get("INSTRUCTION")
            or DEFAULT_INSTRUCTION
        ),
        "input": result.get("input") or result.get("Input") or result.get("INPUT") or "",
        "output": normalize_output_text(
            result.get("output") or result.get("Output") or result.get("OUTPUT") or ""
        ),
    }

    source_key_map = {
        "qa_id": "source_qa_id",
        "domain": "source_domain",
        "q_type": "source_q_type",
    }
    for source_key, output_key in source_key_map.items():
        if source_key in source_item:
            normalized[output_key] = source_item[source_key]

    return normalized


def write_invalid_log(
    handle: Any,
    source_item: Dict[str, Any],
    generated_text: str,
    reason: str,
) -> None:
    if handle is None:
        return
    payload = {
        "reason": reason,
        "source_qa_id": source_item.get("qa_id"),
        "source_domain": source_item.get("domain"),
        "source_q_type": source_item.get("q_type"),
        "generated_text": generated_text,
    }
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_progress(progress_path: Path) -> Dict[str, int]:
    if not progress_path.exists():
        return {}
    with progress_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "processed_count": int(data.get("processed_count", 0)),
        "valid_count": int(data.get("valid_count", 0)),
        "skip_count": int(data.get("skip_count", 0)),
        "invalid_count": int(data.get("invalid_count", 0)),
    }


def save_progress(
    progress_path: Path,
    processed_count: int,
    valid_count: int,
    skip_count: int,
    invalid_count: int,
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    payload = {
        "processed_count": processed_count,
        "valid_count": valid_count,
        "skip_count": skip_count,
        "invalid_count": invalid_count,
    }
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(progress_path)


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"- 입력 데이터: {json.dumps(item, ensure_ascii=False)}"},
    ]


def prepare_text_batch(tokenizer: AutoTokenizer, batch_items: List[Dict[str, Any]]) -> List[str]:
    prompts = []
    for item in batch_items:
        prompt = tokenizer.apply_chat_template(
            build_messages(item),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    progress_path = (
        Path(args.progress_file)
        if args.progress_file
        else output_path.with_suffix(output_path.suffix + ".progress.json")
    )
    invalid_log_path = (
        Path(args.invalid_log_file)
        if args.invalid_log_file
        else output_path.with_suffix(output_path.suffix + ".invalid.jsonl")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite와 --resume은 동시에 사용할 수 없습니다.")
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"출력 파일이 이미 존재합니다: {output_path}. 덮어쓰려면 --overwrite, 이어서 실행하려면 --resume을 사용하세요."
        )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    torch_dtype = resolve_dtype(args.torch_dtype)
    quantization_config = build_quantization_config(args)

    print(f"Loading input data from {input_path} ...")
    with input_path.open("r", encoding="utf-8") as handle:
        raw_data = [json.loads(line) for line in handle]

    if args.start_index:
        raw_data = raw_data[args.start_index :]
    if args.limit is not None:
        raw_data = raw_data[: args.limit]

    selected_count = len(raw_data)
    progress = load_progress(progress_path) if args.resume else {}
    if args.resume and progress:
        expected_output_lines = progress.get("valid_count", 0)
        actual_output_lines = count_jsonl_lines(output_path)
        if actual_output_lines < expected_output_lines:
            raise RuntimeError(
                "progress 파일의 valid_count가 출력 파일 라인 수보다 큽니다. "
                "출력 파일이 삭제/손상되었을 수 있으니 --overwrite로 새로 시작하거나 progress 파일을 확인하세요."
            )
    processed_count = progress.get("processed_count", 0)
    if args.resume and not progress and output_path.exists():
        processed_count = count_jsonl_lines(output_path)
        print(
            "Progress file not found. Falling back to output line count; "
            "this is only exact if previous outputs had no skipped or invalid records."
        )
    if processed_count > selected_count:
        raise ValueError(
            f"progress processed_count({processed_count})가 선택된 입력 수({selected_count})보다 큽니다."
        )
    raw_data = raw_data[processed_count:]

    print(f"Total {selected_count} records selected.")
    if args.resume:
        print(f"Resuming from record offset {processed_count}. Remaining {len(raw_data)} records.")
    if args.resume and not raw_data:
        print("No remaining records. Nothing to generate.")
        print(f"Saved to: {output_path}")
        print(f"Invalid log saved to: {invalid_log_path}")
        print(f"Progress saved to: {progress_path}")
        return
    print(f"Loading model: {args.model_name}")
    print(
        "Runtime:"
        f" cuda={torch.cuda.is_available()}"
        f", mps={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
        f", dtype={args.torch_dtype}"
        f", quantized={'4bit' if args.load_in_4bit else '8bit' if args.load_in_8bit else 'no'}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "token": hf_token,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        with output_path.open("w", encoding="utf-8"):
            pass
        with invalid_log_path.open("w", encoding="utf-8"):
            pass
        if progress_path.exists():
            progress_path.unlink()

    valid_count = progress.get("valid_count", count_jsonl_lines(output_path) if args.resume else 0)
    skip_count = progress.get("skip_count", 0)
    invalid_count = progress.get("invalid_count", 0)

    with output_path.open("a", encoding="utf-8") as out_handle, invalid_log_path.open(
        "a", encoding="utf-8"
    ) as invalid_handle:
        for batch_index, batch_items in enumerate(batched(raw_data, args.batch_size), start=1):
            prompt_texts = prepare_text_batch(tokenizer, batch_items)
            model_inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
            prompt_length = model_inputs["input_ids"].shape[1]

            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for row_index, generated_ids in enumerate(generated):
                source_item = batch_items[row_index]
                new_tokens = generated_ids[prompt_length:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                result = extract_json_from_response(generated_text)

                if result == "SKIP":
                    skip_count += 1
                    write_invalid_log(invalid_handle, source_item, generated_text, "skip")
                    continue
                if result is None:
                    invalid_count += 1
                    write_invalid_log(invalid_handle, source_item, generated_text, "json_parse_failed")
                    continue
                if not is_valid_data(result):
                    invalid_count += 1
                    write_invalid_log(invalid_handle, source_item, generated_text, "validation_failed")
                    continue

                normalized = normalize_result(result, source_item)
                out_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                valid_count += 1

            processed_count += len(batch_items)
            out_handle.flush()
            invalid_handle.flush()
            save_progress(
                progress_path,
                processed_count,
                valid_count,
                skip_count,
                invalid_count,
            )
            print(
                f"[{processed_count}/{selected_count}] valid={valid_count} skip={skip_count} invalid={invalid_count}"
            )

    print("-" * 40)
    print("MedGemma augmentation finished")
    print(f"Input records: {selected_count}")
    print(f"Valid outputs: {valid_count}")
    print(f"Skipped outputs: {skip_count}")
    print(f"Invalid outputs: {invalid_count}")
    print(f"Saved to: {output_path}")
    print(f"Invalid log saved to: {invalid_log_path}")
    print(f"Progress saved to: {progress_path}")
    print("-" * 40)


if __name__ == "__main__":
    main()
