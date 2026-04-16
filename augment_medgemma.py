import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = """당신은 전문 의학 지식을 바탕으로 의료 AI 학습용 가상 데이터를 생성하는 창의적인 전문가입니다.
주어진 [의학 객관식 문제]를 바탕으로, 구음장애 환자의 발음이 ASR(음성인식)을 통해 교정되었다고 가정한 '환자의 구어체 진술'과 의사가 작성할 '구조화된 의료 문진표'를 생성해 주세요.

[조건]
1. 환자의 진술(input)은 너무 딱딱한 의학 용어보다는 일반인이 병원에 와서 증상을 호소하는 자연스러운 구어체로 작성하세요. (ASR로 교정된 깔끔한 텍스트로 가정)
2. 문진표(output)는 주소증(CC), 현병력(PI), 평가 및 계획(A&P) 등 실제 의무기록 양식을 차용하여 간결하게 요약하세요. 원본 문제의 '정답'이나 '핵심 이론'을 의사의 '평가 및 계획'에 자연스럽게 녹여내세요.
3. 순수 이론 문제(예: 유전 양식, 약리 기전, 해부학 등)라도 절대 "SKIP"하지 마세요. 대신 해당 이론과 관련된 질환을 걱정해서 병원에 온 환자 상황으로 창의적으로 변환하세요.
4. 정말로 의학적 맥락으로 도저히 변환이 불가능한 쓰레기 데이터인 경우에만 예외적으로 "SKIP"을 출력하세요.
5. 절대 Thinking Process나 부연 설명을 작성하지 마세요. 반드시 아래의 [출력 JSON 형식]에 맞는 순수 JSON 코드만 출력해야 합니다.
6. 모든 출력은 특정 영어로만 표현이 가능한 단어를 제외하고 한국어로만 작성하세요.

[출력 JSON 형식]
{
  "instruction": "다음 환자의 진술을 바탕으로 핵심 증상을 파악하고, 의사가 보기 편한 구조화된 의료 문진표를 작성하세요.",
  "input": "[이론/증상을 바탕으로 창의적으로 구성한 환자의 자연스러운 호소 문장]",
  "output": "[구조화된 문진표]"
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

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def is_valid_data(json_data: Dict[str, Any]) -> bool:
    if not isinstance(json_data, dict):
        return False

    keys_lower = {str(key).lower() for key in json_data.keys()}
    required_keys = {"instruction", "input", "output"}
    if not required_keys.issubset(keys_lower):
        return False

    if "qa_id" in keys_lower or "status" in keys_lower:
        return False

    for key, value in json_data.items():
        if str(key).lower() not in required_keys:
            continue

        text = str(value).strip()
        if len(text) < 5:
            return False
        if text == "..." or text.upper() == "SKIP":
            return False
        if "[환자의 자연스러운 호소 문장]" in text:
            return False
        if "[구조화된 문진표]" in text:
            return False

    return True


def normalize_result(result: Dict[str, Any]) -> Dict[str, str]:
    return {
        "instruction": (
            result.get("instruction")
            or result.get("Instruction")
            or result.get("INSTRUCTION")
            or DEFAULT_INSTRUCTION
        ),
        "input": result.get("input") or result.get("Input") or result.get("INPUT") or "",
        "output": result.get("output") or result.get("Output") or result.get("OUTPUT") or "",
    }


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

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"출력 파일이 이미 존재합니다: {output_path}. 덮어쓰려면 --overwrite를 사용하세요."
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

    print(f"Total {len(raw_data)} records selected.")
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
    with output_path.open("w", encoding="utf-8"):
        pass

    valid_count = 0
    skip_count = 0
    invalid_count = 0

    with output_path.open("a", encoding="utf-8") as out_handle:
        for batch_index, batch_items in enumerate(batched(raw_data, args.batch_size), start=1):
            prompt_texts = prepare_text_batch(tokenizer, batch_items)
            model_inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
            input_lengths = model_inputs["attention_mask"].sum(dim=1)

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
                input_length = int(input_lengths[row_index])
                new_tokens = generated_ids[input_length:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                result = extract_json_from_response(generated_text)

                if result == "SKIP":
                    skip_count += 1
                    continue
                if result is None or not is_valid_data(result):
                    invalid_count += 1
                    continue

                normalized = normalize_result(result)
                out_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                valid_count += 1

            processed = min(batch_index * args.batch_size, len(raw_data))
            print(
                f"[{processed}/{len(raw_data)}] valid={valid_count} skip={skip_count} invalid={invalid_count}"
            )

    print("-" * 40)
    print("MedGemma augmentation finished")
    print(f"Input records: {len(raw_data)}")
    print(f"Valid outputs: {valid_count}")
    print(f"Skipped outputs: {skip_count}")
    print(f"Invalid outputs: {invalid_count}")
    print(f"Saved to: {output_path}")
    print("-" * 40)


if __name__ == "__main__":
    main()
