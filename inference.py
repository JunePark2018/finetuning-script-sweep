"""
노지 작물 해충 진단 - 추론 스크립트 (전체 데이터셋)
클래스 목록은 모델 디렉토리의 class_names.json에서 로드 (선택적).
사용법: python inference.py --image test.jpg --model pest-lora-<RUN_NAME>
"""

import argparse
import json
import os

from PIL import Image
from unsloth import FastVisionModel

SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    '사진을 보고 해충의 이름만 한국어로 답하세요. '
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)


def load_class_names(model_path):
    path = os.path.join(model_path, "class_names.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict(image_path: str, model_path: str):
    """단일 이미지에 대해 해충 분류 추론"""
    class_names = load_class_names(model_path)
    if class_names:
        print(f"클래스 ({len(class_names)}개): {class_names}")

    print(f"모델 로딩: {model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)

    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": SYSTEM_MSG}
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."},
        ]},
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
        temperature=0.1,
    )
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    image.close()

    print(f"\n이미지: {image_path}")
    print(f"예측: {prediction}")
    if class_names is not None:
        if prediction in class_names:
            print("  → 유효한 클래스")
        else:
            matched = next((c for c in class_names if c in prediction), None)
            print(f"  → 클래스 집합 밖{' (추출: ' + matched + ')' if matched else ''}")
    return prediction


def main():
    parser = argparse.ArgumentParser(description="해충 진단 추론")
    parser.add_argument("--image", required=True, help="이미지 경로")
    parser.add_argument("--model", required=True, help="LoRA 모델 경로 (예: pest-lora-r16_a16_...)")
    args = parser.parse_args()

    predict(args.image, args.model)


if __name__ == "__main__":
    main()
