"""
노지 작물 해충 진단 - 평가 스크립트 (전체 데이터셋)
클래스 목록은 모델 디렉토리의 class_names.json에서 로드 (학습 시 생성됨).
평가 split: EVAL_SPLIT 환경변수 (기본 val).
사용법: python evaluate.py --model pest-lora-<RUN_NAME>
"""

import argparse
import json
import os
import time
import requests

from PIL import Image
from unsloth import FastVisionModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from prompts import USER_PROMPT, build_system_msg  # train/evaluate/inference 공유 프롬프트


DISCORD_BOT = {
    "username": "RunPod",
    "avatar_url": "https://i.imgur.com/0HOIh4r.png",
}
DISCORD_COLOR = 12648430


def notify_discord(message):
    """DISCORD_WEBHOOK_URL이 설정되어 있으면 메시지 전송"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message}, timeout=10)
    except Exception as e:
        print(f"Discord 알림 실패: {e}")


def notify_discord_json(payload):
    """DISCORD_WEBHOOK_URL이 설정되어 있으면 JSON payload를 그대로 전송 (Embed 등)"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord 알림 실패: {e}")


def discord_embed(description):
    """Embed payload 생성 헬퍼"""
    return {**DISCORD_BOT, "embeds": [{"description": description, "color": DISCORD_COLOR}]}


DATA_DIR = os.environ.get("DATA_DIR", "data")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "val")


def load_class_names(model_path):
    """모델 디렉토리의 class_names.json에서 클래스 목록 로드.
    파일이 없으면 DATA_DIR/train/ 하위 폴더명에서 추출."""
    path = os.path.join(model_path, "class_names.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.isdir(train_dir):
        return sorted(d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d)))
    raise FileNotFoundError(
        f"class_names.json 없음 ({path}) & {train_dir}도 없음. 학습 시 class_names.json 저장 여부 확인."
    )


def load_eval_dataset(split):
    """{split}.jsonl에서 (이미지 경로, 정답 라벨) 리스트 반환"""
    jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
    assert os.path.exists(jsonl_path), f"{split}.jsonl이 없습니다: {jsonl_path}"

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            messages = record["messages"]
            label = messages[-1]["content"][0]["text"]

            img_rel_path = None
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image" and "image" in content:
                        img_rel_path = content["image"].replace("\\", "/")
                        break

            if img_rel_path:
                img_path = os.path.join(DATA_DIR, img_rel_path)
                samples.append((img_path, label))

    return samples


def predict_single(model, tokenizer, image_path, system_msg):
    """단일 이미지에 대해 추론하여 예측 라벨 반환"""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": system_msg}
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
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
        max_new_tokens=32,  # 긴 클래스명(예: 톱다리개미허리노린재)도 안 잘리게 여유
        use_cache=True,
        temperature=0.1,
    )

    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    image.close()
    return prediction


def evaluate(model_path):
    """평가 split 전체에 대해 평가 실행"""
    CLASS_NAMES = load_class_names(model_path)
    SYSTEM_MSG = build_system_msg(CLASS_NAMES)
    print(f"클래스 ({len(CLASS_NAMES)}개): {CLASS_NAMES}")

    # 모델 로딩
    notify_discord_json(discord_embed(f"🔍 [1/3] 평가 모델을 로딩합니다. ({len(CLASS_NAMES)}클래스)"))
    try:
        print(f"모델 로딩: {model_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=False,
        )
        FastVisionModel.for_inference(model)
        notify_discord_json(discord_embed("✅ [1/3] 모델 로딩 완료!"))
    except Exception as e:
        notify_discord_json(discord_embed(f"@everyone\n❌ [1/3] 모델 로딩 중 에러 발생: {e}"))
        raise

    # 데이터 로딩 + 추론
    notify_discord_json(discord_embed(f"📊 [2/3] {EVAL_SPLIT} 데이터 추론을 시작합니다."))
    try:
        samples = load_eval_dataset(EVAL_SPLIT)
        # Smoke test용: EVAL_LIMIT=N으로 N개만 평가. seed 고정 random sample이라
        # 클래스 한쪽으로 쏠리지 않고 [8] 코드 path 전체를 빠르게 검증.
        eval_limit = int(os.environ.get("EVAL_LIMIT", 0))
        if eval_limit > 0 and eval_limit < len(samples):
            import random as _r
            _r.Random(42).shuffle(samples)
            samples = samples[:eval_limit]
            print(f"  EVAL_LIMIT={eval_limit} — {len(samples)}건만 평가 (smoke test 모드)")
        print(f"평가 샘플 ({EVAL_SPLIT}): {len(samples)}건\n")

        y_true = []
        y_pred = []
        inference_times = []

        for i, (img_path, label) in enumerate(samples):
            t_start = time.time()
            pred = predict_single(model, tokenizer, img_path, SYSTEM_MSG)
            t_elapsed = time.time() - t_start
            inference_times.append(t_elapsed)

            y_true.append(label)
            y_pred.append(pred)

            status = "O" if label == pred else "X"
            print(f"  [{i+1}/{len(samples)}] {status}  정답: {label:10s}  예측: {pred:10s}  ({t_elapsed:.2f}s)")
    except Exception as e:
        notify_discord_json(discord_embed(f"@everyone\n❌ [2/3] 추론 중 에러 발생: {e}"))
        raise

    # Mode collapse 조기 진단 — 전체 예측이 3개 미만 클래스로 수렴했으면 학습 실패.
    # 2026-04-16 sweep에서 γ=8 조합이 15h 돌고도 단일 클래스만 뱉는 붕괴 사례 확인.
    # metric은 계속 계산·로깅해서 sweep optimizer가 이 영역을 피하도록 정보 제공.
    unique_preds = set(y_pred)
    if len(unique_preds) < 3:
        collapse_msg = (
            f"Mode collapse 감지: {len(y_pred)}건 중 {len(unique_preds)}개 클래스만 예측됨 "
            f"({sorted(unique_preds)})"
        )
        print(f"\n⚠️  {collapse_msg}")
        notify_discord_json(discord_embed(
            f"@everyone\n⚠️ [3/3] {collapse_msg}\n"
            f"학습 실패 판정 — HP 조합 부적절 가능성 (γ 과도, LR 과도 등). "
            f"sweep 중이면 낮은 metric 기록되어 Bayes가 자동 기피."
        ))

    # ════════════════════════════════════════
    # 6개 평가 메트릭
    # ════════════════════════════════════════
    notify_discord_json(discord_embed("📈 [3/3] 평가 결과를 집계합니다."))

    print("\n" + "=" * 60)
    print("평가 결과 (6개 메트릭)")
    print("=" * 60)

    # 1. Confusion Matrix (N×N 동적 렌더링)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    print("\n[1] Confusion Matrix:")
    col_w = max(6, max(len(c) for c in CLASS_NAMES) + 2)
    header = " " * 18 + "".join(f"{('예측:'+c)[:col_w]:>{col_w}}" for c in CLASS_NAMES)
    print(header)
    for i, cls in enumerate(CLASS_NAMES):
        row_label = f"{'실제:'+cls:18s}"
        row_cells = "".join(f"{cm[i][j]:>{col_w}d}" for j in range(len(CLASS_NAMES)))
        print(row_label + row_cells)

    # 2. Accuracy
    acc = accuracy_score(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    print(f"\n[2] Accuracy: {acc:.4f} ({correct}/{len(y_true)})")

    # 3. Precision (per class + macro)
    prec_per_class = precision_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print(f"\n[3] Precision:")
    for cls, p in zip(CLASS_NAMES, prec_per_class):
        print(f"    {cls}: {p:.4f}")
    print(f"    Macro: {prec_macro:.4f}")

    # 4. Recall (per class + macro)
    rec_per_class = recall_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print(f"\n[4] Recall:")
    for cls, r in zip(CLASS_NAMES, rec_per_class):
        print(f"    {cls}: {r:.4f}")
    print(f"    Macro: {rec_macro:.4f}")

    # 5. Macro F1 Score
    f1_per_class = f1_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    print(f"\n[5] Macro F1 Score:")
    for cls, f in zip(CLASS_NAMES, f1_per_class):
        print(f"    {cls}: {f:.4f}")
    print(f"    Macro: {f1_macro:.4f}")

    # 6. 추론 속도
    avg_time = sum(inference_times) / len(inference_times)
    total_time = sum(inference_times)
    print(f"\n[6] 추론 속도:")
    print(f"    총 소요 시간: {total_time:.1f}s")
    print(f"    이미지당 평균: {avg_time:.2f}s")
    print(f"    처리량: {len(samples)/total_time:.1f} img/s")

    # ════════════════════════════════════════
    # 7. 서비스 지표 — Binary pest-vs-normal + Composite (sweep metric)
    # ════════════════════════════════════════
    # 서비스 실패 비용 비대칭:
    #   - 해충 → 정상 FN : 치명적 (농부 무대응 → 피해 확산)
    #   - 정상 → 해충 FP : 나쁨 (불필요 농약)
    #   - 해충 X → 해충 Y: 수용 가능 (어쨌든 알림 전달)
    # pest_gated_f1 = binary_pest_recall × macro_f1_on_18_pest_classes
    # 곱셈이라 둘 다 높아야 점수 남 → sweep이 서비스 비대칭을 직접 최적화.
    NORMAL_CLASS = "정상"
    pest_classes = [c for c in CLASS_NAMES if c != NORMAL_CLASS]

    tp_bin = sum(1 for t, p in zip(y_true, y_pred) if t != NORMAL_CLASS and p != NORMAL_CLASS)
    fn_bin = sum(1 for t, p in zip(y_true, y_pred) if t != NORMAL_CLASS and p == NORMAL_CLASS)
    fp_bin = sum(1 for t, p in zip(y_true, y_pred) if t == NORMAL_CLASS and p != NORMAL_CLASS)
    tn_bin = sum(1 for t, p in zip(y_true, y_pred) if t == NORMAL_CLASS and p == NORMAL_CLASS)

    bin_recall = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0.0
    bin_precision = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else 0.0
    bin_f2 = (5 * bin_precision * bin_recall / (4 * bin_precision + bin_recall)) if (bin_precision + bin_recall) > 0 else 0.0
    normal_specificity = tn_bin / (tn_bin + fp_bin) if (tn_bin + fp_bin) > 0 else 0.0

    f1_macro_pests = f1_score(y_true, y_pred, labels=pest_classes, average="macro", zero_division=0)
    pest_gated_f1 = bin_recall * float(f1_macro_pests)

    print(f"\n[7] 서비스 지표 (농부 관점):")
    print(f"    해충→해충으로 잡은 비율 (binary_pest_recall):     {bin_recall:.4f}   [치명적 FN {fn_bin}건]")
    print(f"    정상→정상으로 잡은 비율 (normal_specificity):      {normal_specificity:.4f}   [과경보 FP {fp_bin}건]")
    print(f"    Binary F2  (recall 4× 가중, pest-vs-normal):    {bin_f2:.4f}")
    print(f"    Macro F1   (18 해충만, 정상 제외):                 {float(f1_macro_pests):.4f}")
    print(f"    ★ pest_gated_f1 (sweep 최적화 metric):           {pest_gated_f1:.4f}")

    # 서비스 건전성 경고 — 해충 클래스 중 F1 < 0.5인 것은 배포 차단 사유.
    # 전체 macro_f1이 높아도 특정 해충 몇 종을 못 잡으면 서비스 부적합 (농부가 해당
    # 해충 사진 올렸을 때 잘못된 대응 유도). winner 선정 시 이 경고 반드시 확인.
    pest_low_f1 = [(cls, float(f)) for cls, f in zip(CLASS_NAMES, f1_per_class)
                   if cls != NORMAL_CLASS and float(f) < 0.5]
    if pest_low_f1:
        warn_lines = "\n".join(f"  • {cls}: F1={f:.3f}" for cls, f in pest_low_f1)
        print(f"\n⚠️  서비스 건전성 경고 — 해충 {len(pest_low_f1)}개 클래스 F1 < 0.5:")
        print(warn_lines)
        notify_discord_json(discord_embed(
            f"⚠️ [진단] 해충 {len(pest_low_f1)}개 클래스 F1<0.5 (서비스 부적합 가능):\n{warn_lines}"
        ))

    # 오답 목록
    wrong = [(t, p, s[0]) for (s, t, p) in zip(samples, y_true, y_pred) if t != p]
    if wrong:
        print(f"\n오답 {len(wrong)}건:")
        for t, p, path in wrong:
            print(f"  정답: {t:10s}  예측: {p:10s}  {os.path.basename(path)}")

    print("=" * 60)

    # 평가 결과를 JSON으로 저장 (모델 디렉토리에 함께 보관)
    from datetime import datetime
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "eval_split": EVAL_SPLIT,
        "num_samples": len(samples),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "accuracy": round(acc, 4),
        "precision": {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, prec_per_class)},
        "precision_macro": round(float(prec_macro), 4),
        "recall": {cls: round(float(r), 4) for cls, r in zip(CLASS_NAMES, rec_per_class)},
        "recall_macro": round(float(rec_macro), 4),
        "f1": {cls: round(float(f), 4) for cls, f in zip(CLASS_NAMES, f1_per_class)},
        "f1_macro": round(float(f1_macro), 4),
        "binary_pest_vs_normal": {
            "true_pest_pred_pest_TP": tp_bin,
            "true_pest_pred_normal_FN": fn_bin,      # ← 치명적 실패 (해충 놓침)
            "true_normal_pred_pest_FP": fp_bin,      # ← 과경보 (불필요 농약)
            "true_normal_pred_normal_TN": tn_bin,
            "recall": round(bin_recall, 4),
            "precision": round(bin_precision, 4),
            "f2": round(bin_f2, 4),
            "normal_specificity": round(normal_specificity, 4),
        },
        "f1_macro_pests_only": round(float(f1_macro_pests), 4),
        "pest_gated_f1": round(pest_gated_f1, 4),
        "inference_speed": {
            "avg_seconds_per_image": round(avg_time, 2),
            "images_per_second": round(len(samples) / total_time, 1),
            "total_seconds": round(total_time, 1),
        },
        "wrong_count": len(wrong),
    }
    eval_path = os.path.join(model_path, "evaluation_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n평가 결과 저장: {eval_path}")

    # W&B에 메트릭 로깅. sweep optimizer는 eval/pest_gated_f1을 읽음 (sweep.yaml 참고).
    try:
        import wandb
        if wandb.run is not None:
            log_dict = {
                # ★ Sweep 최적화 대상
                "eval/pest_gated_f1": pest_gated_f1,
                # Composite 분해 지표 (해석용)
                "eval/binary_pest_recall": float(bin_recall),
                "eval/binary_pest_precision": float(bin_precision),
                "eval/binary_pest_f2": float(bin_f2),
                "eval/normal_specificity": float(normal_specificity),
                "eval/f1_macro_pests_only": float(f1_macro_pests),
                # 기존 일반 지표
                "eval/macro_f1": float(f1_macro),
                "eval/macro_precision": float(prec_macro),
                "eval/macro_recall": float(rec_macro),
                "eval/accuracy": float(acc),
                # 직접 카운트 (절대 숫자)
                "eval/fn_pest_to_normal": fn_bin,
                "eval/fp_normal_to_pest": fp_bin,
            }
            # Per-class F1 — wandb UI에서 슬라이스 분석 가능하도록 클래스별 로깅.
            # 해충 1~2종만 못 잡는 winner를 식별할 때 결정적 (전체 macro는 가려줌).
            for cls, f in zip(CLASS_NAMES, f1_per_class):
                log_dict[f"eval/f1_per_class/{cls}"] = float(f)
            wandb.log(log_dict)
            print("W&B 메트릭 로깅 완료")
    except Exception as e:
        print(f"W&B 로깅 실패 (무시): {e}")

    # Discord 알림 — 서비스 지표 중심
    notify_discord_json(discord_embed(
        f"✅ [3/3] 평가 완료!\n\n"
        f"★ pest_gated_f1 (sweep): {pest_gated_f1:.4f}\n"
        f"해충 잡은 비율 (recall): {bin_recall:.4f}   [FN {fn_bin}건]\n"
        f"정상 지킨 비율 (specificity): {normal_specificity:.4f}   [FP {fp_bin}건]\n"
        f"Macro F1 (전체): {f1_macro:.4f}\n"
        f"Macro F1 (해충만): {float(f1_macro_pests):.4f}\n"
        f"Accuracy: {acc:.4f} ({correct}/{len(y_true)})\n"
        f"추론 속도: {avg_time:.2f}s/img ({len(samples)/total_time:.1f} img/s)"
    ))

    return eval_results, eval_path


def main():
    parser = argparse.ArgumentParser(description="해충 진단 모델 평가")
    parser.add_argument("--model", required=True, help="LoRA 모델 경로 (예: pest-lora-r16_a16_...)")
    args = parser.parse_args()

    evaluate(args.model)


if __name__ == "__main__":
    main()
