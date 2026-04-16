"""
노지 작물 해충 진단 - Qwen3.5-9B LoRA 파인튜닝 스크립트 (전체 데이터셋)
대상: 전체 해충 데이터셋 (클래스 수는 데이터셋에서 동적 추출)
데이터셋: HF Hub의 Himedia-AI-01/pest-detection-korean (gated — HF_TOKEN 필요)
         DATA_DIR이 이미 채워져 있으면 다운로드 건너뜀
환경: 32GB+ VRAM (A5000/A6000), bf16 LoRA
"""

import json
import os
import random
import sys
import time
import requests

from PIL import Image

# ════════════════════════════════════════
# W&B Sweep: CLI args (--KEY=VAL) → env var
# ════════════════════════════════════════
# wandb agent는 sampling한 HP를 `python train.py --LEARNING_RATE=X --LORA_R=Y ...`
# 형태로 전달함. train.py는 argparse를 안 쓰고 os.environ.get으로 읽으므로,
# 여기서 CLI args를 env로 복사해서 이하 모든 os.environ.get이 sampling 결과를 보도록 함.
# 수동 실행(CLI args 없음) 시엔 루프가 비어서 아무 영향 없음.
for _arg in sys.argv[1:]:
    if _arg.startswith("--") and "=" in _arg:
        _k, _, _v = _arg[2:].partition("=")
        os.environ[_k] = _v
        print(f"  [sweep] {_k}={_v}")

# ════════════════════════════════════════
# Discord Webhook 설정
# ════════════════════════════════════════

DISCORD_BOT = {
    "username": "RunPod",
    "avatar_url": "https://i.imgur.com/0HOIh4r.png",
}
DISCORD_COLOR = 12648430
DISCORD_THUMBNAIL = "https://i.imgur.com/3ClKkzk.jpeg"


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


def discord_embed(description, thumbnail=False):
    """Embed payload 생성 헬퍼"""
    embed = {"description": description, "color": DISCORD_COLOR}
    if thumbnail:
        embed["thumbnail"] = {"url": DISCORD_THUMBNAIL}
    return {**DISCORD_BOT, "embeds": [embed]}


# ════════════════════════════════════════
# 하이퍼파라미터 (환경변수로 오버라이드 가능)
# ════════════════════════════════════════

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 6))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 2))

LORA_R = int(os.environ.get("LORA_R", 16))
# α=r 커플링 해제. TML "LoRA Without Regret"에 따라 α를 rank와 분리 고정하면
# optimal LR이 rank에 독립적 → sweep 신호가 더 깨끗해짐.
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
# Dropout은 3 epoch 학습 구간에서 분산만 키운다는 ALLoRA(2410.09692) 근거로 0.05 고정.
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 3))
# 총 step ≈ 2,900 (3 epoch × 967 step/epoch) → 150 step = ~5% (권장 5~10% 구간의 하단)
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 150))
MAX_STEPS = int(os.environ.get("MAX_STEPS", -1))  # > 0이면 num_train_epochs를 무시하고 step 수 직접 고정

print("=" * 60)
print("하이퍼파라미터")
print("=" * 60)
print(f"  BATCH_SIZE     = {BATCH_SIZE}")
print(f"  GRAD_ACCUM     = {GRAD_ACCUM}")
print(f"  Total Batch    = {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LORA_R         = {LORA_R}")
print(f"  LORA_ALPHA     = {LORA_ALPHA}")
print(f"  LORA_DROPOUT   = {LORA_DROPOUT}")
print(f"  LEARNING_RATE  = {LEARNING_RATE}")
if MAX_STEPS > 0:
    print(f"  MAX_STEPS      = {MAX_STEPS} (num_train_epochs 무시됨)")
else:
    print(f"  NUM_EPOCHS     = {NUM_EPOCHS}")
print(f"  WARMUP_STEPS   = {WARMUP_STEPS}")

# run name — 하이퍼파라미터 조합으로 구성. 같은 조합이면 같은 이름 → 같은 OUTPUT_DIR 재사용 → 자동 resume.
# RUN_NAME env로 오버라이드도 가능.
# Sweep의 LR은 log_uniform 연속분포라 완전 동일 샘플 확률 0 — 충돌 걱정 없음.
_epoch_or_step = f"st{MAX_STEPS}" if MAX_STEPS > 0 else f"ep{NUM_EPOCHS}"
_default_run = f"r{LORA_R}_a{LORA_ALPHA}_d{LORA_DROPOUT}_lr{LEARNING_RATE}_bs{BATCH_SIZE}x{GRAD_ACCUM}_{_epoch_or_step}_w{WARMUP_STEPS}"
RUN_NAME = os.environ.get("RUN_NAME") or _default_run
OUTPUT_DIR = f"pest-detector-{RUN_NAME}"
LORA_DIR = f"pest-lora-{RUN_NAME}"

print(f"  RUN_NAME       = {RUN_NAME}{' (env override)' if os.environ.get('RUN_NAME') else ''}")
print(f"  OUTPUT_DIR     = {OUTPUT_DIR}")
print(f"  LORA_DIR       = {LORA_DIR}")
print("=" * 60)

# W&B project 기본값 (환경변수 미설정 시) — 모든 sweep run이 한 project에 모여 비교 가능
os.environ.setdefault("WANDB_PROJECT", "pest-detection-full")

DATASET_REPO = os.environ.get("DATASET_REPO", "Himedia-AI-01/pest-detection-korean")
FORCE_DOWNLOAD = os.environ.get("FORCE_DOWNLOAD", "0") == "1"

Image.MAX_IMAGE_PIXELS = None

# ════════════════════════════════════════
# 1. 데이터셋 준비 (HF Hub에서 자동 다운로드)
# ════════════════════════════════════════

print("\n[1/9] 데이터셋 준비...")
notify_discord_json(discord_embed(f"📂 [1/9] 데이터셋을 준비합니다. ({DATASET_REPO})", thumbnail=True))
try:
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    train_jsonl = os.path.join(DATA_DIR, "train.jsonl")

    if FORCE_DOWNLOAD or not os.path.exists(train_jsonl):
        from huggingface_hub import snapshot_download
        hf_token = os.environ.get("HF_TOKEN") or None
        print(f"  HF Hub에서 다운로드: {DATASET_REPO} → {DATA_DIR}")
        t0 = time.time()
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=DATA_DIR,
            token=hf_token,
        )
        print(f"  다운로드 완료: {time.time() - t0:.1f}s")
    else:
        print(f"  기존 데이터 재사용: {DATA_DIR}/train.jsonl")

    assert os.path.exists(train_jsonl), f"데이터셋 다운로드 실패: {train_jsonl} 없음"

    # 클래스 목록 동적 추출 (train/ 하위 폴더명)
    class_dir = os.path.join(DATA_DIR, "train")
    CLASS_NAMES = sorted(d for d in os.listdir(class_dir)
                         if os.path.isdir(os.path.join(class_dir, d)))
    assert CLASS_NAMES, f"클래스 폴더가 없습니다: {class_dir}"

    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  train.jsonl ✓")
    print(f"  val.jsonl   {'✓' if os.path.exists(os.path.join(DATA_DIR, 'val.jsonl')) else '✗'}")
    print(f"  test.jsonl  {'✓' if os.path.exists(os.path.join(DATA_DIR, 'test.jsonl')) else '✗'}")
    print(f"  클래스 수: {len(CLASS_NAMES)}")
    print(f"  클래스 목록: {CLASS_NAMES}")
    notify_discord_json(discord_embed(
        f"✅ [1/9] 데이터셋 준비 완료. ({len(CLASS_NAMES)}클래스)"
    ))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [1/9] 데이터셋 준비 실패: {e}"))
    raise

# ════════════════════════════════════════
# 2. 이미지 전처리 (크롭 → 디스크 저장)
# ════════════════════════════════════════

print("\n[2/9] 이미지 전처리...")
notify_discord_json(discord_embed("🖼️ [2/9] 이미지 전처리를 시작합니다. (크롭 → 디스크 저장)"))
try:
    # 캐시가 비어 있는 첫 실행에서 50/25/25 분기를 결정적으로 만들기 위한 seed.
    # 캐시가 있으면 분기 자체가 일어나지 않아 무관하지만, 새 Pod에서 캐시를 새로 생성할 때
    # 바이트 단위 재현성이 필요한 경우를 위해 명시.
    random.seed(42)

    # ⚠️ 학습/평가/추론 세 곳에서 바이트 단위로 동일해야 함.
    # 동일 문자열이 evaluate.py, inference.py에도 리터럴로 박혀 있음.
    USER_PROMPT = "이 사진에 있는 해충의 이름을 알려주세요."

    # SYSTEM_MSG는 CLASS_NAMES에서 동적 생성 — 학습/평가/추론 세 곳이 바이트 단위로 동일해야 함.
    # 동일한 build_system_msg 함수가 evaluate.py, inference.py에도 복제돼 있음.
    def build_system_msg(class_names):
        class_list = ", ".join(class_names)
        return (
            "당신은 작물 해충 식별 전문가입니다. "
            "사진 속 해충을 다음 목록에서 하나만 골라 그 단어 그대로 출력하세요:\n"
            f"{class_list}\n\n"
            "출력 규칙 (반드시 준수):\n"
            "- 목록의 단어 하나만, 정확한 철자로\n"
            "- 조사/수식어/구두점/설명/줄바꿈 전부 금지\n"
            '- 해충이 없으면 "정상"'
        )

    SYSTEM_MSG = build_system_msg(CLASS_NAMES)

    BBOX_GROW_STAGE = 33


    def crop_to_bbox(img, bbox, padding_ratio=0.0):
        xtl, ytl = bbox["xtl"], bbox["ytl"]
        xbr, ybr = bbox["xbr"], bbox["ybr"]
        bw, bh = xbr - xtl, ybr - ytl
        pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
        x1 = max(0, xtl - pad_x)
        y1 = max(0, ytl - pad_y)
        x2 = min(img.width, xbr + pad_x)
        y2 = min(img.height, ybr + pad_y)
        return img.crop((x1, y1, x2, y2))


    def find_label_json(split, class_name, img_filename):
        json_path = os.path.join(DATA_DIR, split, class_name, img_filename + ".json")
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            for obj in data["annotations"]["object"]:
                if obj["grow"] == BBOX_GROW_STAGE and obj.get("points"):
                    return obj["points"][0]
        except (json.JSONDecodeError, KeyError, TypeError):
            # 빈 파일, 손상된 JSON, 예상과 다른 구조 → bbox 없이 원본 사용
            pass
        return None


    def preprocess_split(split="train"):
        """원본 이미지를 크롭하여 디스크에 저장하고 새 JSONL 생성"""
        jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
        out_dir = os.path.join(DATA_DIR, f"{split}_cropped")
        out_jsonl = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")

        if os.path.exists(out_jsonl):
            with open(out_jsonl, "r") as f:
                count = sum(1 for _ in f)
            print(f"  [{split}] 이미 전처리 완료: {count}건 (캐시 사용)")
            return count

        with open(jsonl_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
        print(f"  [{split}] 전처리 시작: {total}건")

        out_file = open(out_jsonl, "w", encoding="utf-8")
        count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if (i + 1) % 500 == 0 or (i + 1) == total:
                    print(f"\r  [{split}] {i + 1}/{total} ({(i + 1) * 100 // total}%)", end="", flush=True)

                record = json.loads(line)
                messages = record["messages"]
                label = messages[-1]["content"][0]["text"]

                img_rel_path = None
                for msg in messages:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            img_rel_path = content["image"].replace("\\", "/")
                            break

                if img_rel_path is None:
                    continue

                parts = img_rel_path.split("/")
                class_name = parts[1]
                img_filename = parts[2]

                class_dir = os.path.join(out_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                base_name = os.path.splitext(img_filename)[0]
                out_filename = f"{base_name}.jpg"
                out_path = os.path.join(class_dir, out_filename)
                out_rel_path = f"{split}_cropped/{class_name}/{out_filename}"

                img_path = os.path.join(DATA_DIR, img_rel_path)
                img = Image.open(img_path).convert("RGB")

                # 학습 분포: 원본 50% + tight crop 25% + context crop(pad 0.5) 25%
                # 평가(evaluate.py)는 원본 100%. 이 비대칭은 의도적이다:
                # 서비스 입력은 사용자가 멀리서 찍은 원본 사진이라 eval은 서비스 분포를 반영,
                # 학습의 크롭은 가까이서 찍힌 케이스(특히 minority class) augmentation 용도.
                if label == "정상":
                    result = img
                else:
                    bbox = find_label_json(split, class_name, img_filename)
                    if bbox:
                        r = random.random()
                        if r < 0.5:
                            result = img
                        elif r < 0.75:
                            result = crop_to_bbox(img, bbox, padding_ratio=0.0)
                        else:
                            result = crop_to_bbox(img, bbox, padding_ratio=0.5)
                    else:
                        result = img

                result.save(out_path, "JPEG", quality=95)
                if result is not img:
                    result.close()
                img.close()

                new_record = {
                    "messages": [
                        {"role": "system", "content": [
                            {"type": "text", "text": SYSTEM_MSG}
                        ]},
                        {"role": "user", "content": [
                            {"type": "image", "image": out_rel_path},
                            {"type": "text", "text": USER_PROMPT},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": label}
                        ]},
                    ]
                }
                out_file.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                count += 1

        out_file.close()
        print(f"\n  [{split}] 완료: {count}건 → {out_dir}")
        return count


    random.seed(42)
    t0 = time.time()
    num_train = preprocess_split("train")
    num_val = preprocess_split("val")
    print(f"  전처리 소요 시간: {time.time() - t0:.1f}s")
    notify_discord_json(discord_embed(f"✅ [2/9] 전처리 완료! (train {num_train}건, val {num_val}건)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [2/9] 이미지 전처리 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 3. 데이터 로딩 (경로 기반 — RAM 절약)
# ════════════════════════════════════════

print("\n[3/9] 데이터 로딩...")
notify_discord_json(discord_embed("📊 [3/9] 데이터를 로딩합니다."))
try:
    def load_dataset_from_cropped_jsonl(split="train"):
        jsonl_path = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")
        dataset = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for msg in record["messages"]:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            content["image"] = os.path.join(DATA_DIR, content["image"])
                dataset.append(record)
        random.shuffle(dataset)
        return dataset

    random.seed(42)
    t0 = time.time()
    train_dataset = load_dataset_from_cropped_jsonl("train")
    val_dataset = load_dataset_from_cropped_jsonl("val")
    print(f"  Train: {len(train_dataset)}건, Val: {len(val_dataset)}건")
    print(f"  로딩 소요 시간: {time.time() - t0:.1f}s")
    notify_discord_json(discord_embed(f"✅ [3/9] 데이터 로딩 완료! (Train {len(train_dataset)}건, Val {len(val_dataset)}건)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [3/9] 데이터 로딩 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 4. 모델 로딩
# ════════════════════════════════════════

print("\n[4/9] 모델 로딩...")
notify_discord_json(discord_embed("🤖 [4/9] Qwen3.5-9B 모델을 로딩합니다."))
try:
    # ⚠️ HF cache 환경변수는 torch/unsloth import 전에 설정해야 함
    # (cache 경로가 import 시점에 한 번 읽히기 때문)
    # RunPod(Linux) 기본 경로. Windows나 다른 호스트에서는 HF 기본(~/.cache/huggingface)에 맡김.
    # 명시 경로가 필요하면 HF_HOME env로 오버라이드.
    if os.name != "nt":
        os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")

    import torch
    from unsloth import FastVisionModel

    print(f"  HF_HOME = {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    t0 = time.time()
    # unsloth/Qwen3.5-9B — Unsloth Hub에 실존하는 VL 모델 (오타 아님)
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-9B",
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    print(f"  모델 로딩 소요 시간: {time.time() - t0:.1f}s")
    notify_discord_json(discord_embed("✅ [4/9] 모델 로딩 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [4/9] 모델 로딩 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 5. LoRA 설정
# ════════════════════════════════════════

print(f"\n[5/9] LoRA 설정 (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})...")
notify_discord_json(discord_embed(f"⚙️ [5/9] LoRA 어댑터를 설정합니다. (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})"))
try:
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  학습 가능 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    notify_discord_json(discord_embed("✅ [5/9] LoRA 설정 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [5/9] LoRA 설정 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 6. 학습
# ════════════════════════════════════════

_train_len = f"{MAX_STEPS} steps" if MAX_STEPS > 0 else f"{NUM_EPOCHS} epochs"
print(f"\n[6/9] 학습 시작 ({_train_len}, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, lr={LEARNING_RATE})...")
notify_discord_json(discord_embed(f"@everyone\n🚀 [6/9] 학습을 시작합니다! ({_train_len}, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM})"))
try:
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            max_steps=MAX_STEPS,  # -1 (기본): epoch 사용 / > 0: epoch 무시하고 step 수 직접 제어
            learning_rate=LEARNING_RATE,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,  # 최신 1개만 유지 — sweep 20 run × 9B 체크포인트로 디스크 폭발 방지
            # Smoke test 모드(EVAL_LIMIT set)에선 Trainer 내장 eval 스킵 — val 전체 1595건 돌려서 ~16분 낭비 방지.
            # 실전 sweep에선 epoch당 eval_loss 찍어야 수렴 곡선 관찰 가능하므로 "epoch" 유지.
            eval_strategy="no" if os.environ.get("EVAL_LIMIT") else "epoch",
            optim="adamw_8bit",
            weight_decay=0.01,  # Unsloth 권장 0.01~0.1 (기존 0.001은 10배 낮았음)
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name=RUN_NAME,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            # max_seq_length은 SFTConfig 경로에선 무시됨 (skip_prepare_dataset=True가 TRL tokenization을 우회).
            # 실제로 enforce하려면 UnslothVisionDataCollator(model, tokenizer, max_seq_length=N)로 명시 전달해야 함.
            # 현재는 model 자연 ctx로 흘러감 → VRAM 여유 있는 동안 의도적 미설정.
            # Windows는 fork 미지원 → multiprocessing DataLoader가 spawn으로 뜨면서 에러.
            # RunPod(Linux)에선 8 유지, Windows 로컬에선 0으로 폴백.
            dataloader_num_workers=0 if os.name == "nt" else 8,
            dataloader_pin_memory=True,
        ),
    )

    # 체크포인트 존재 확인 → resume 여부 결정 (빈 OUTPUT_DIR도 안전하게 처리)
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")] if os.path.exists(OUTPUT_DIR) else []
    if ckpts:
        print(f"  체크포인트 발견: {sorted(ckpts)[-1]} → 이어서 학습")
        resume = True
    else:
        print(f"  체크포인트 없음 → 처음부터 학습")
        resume = False

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_time = time.time() - t0
    print(f"  학습 소요 시간: {train_time/60:.1f}분")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Peak VRAM: {peak_gb:.2f} / {total_gb:.1f} GB ({peak_gb/total_gb*100:.0f}%)")
    notify_discord_json(discord_embed("@everyone\n✅ [6/9] 학습 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"@everyone\n❌ [6/9] 학습 중 에러 발생: {e}"))
    raise

# Probe 모드: VRAM 측정만 하고 종료 (평가/저장/업로드 스킵)
if os.environ.get("SKIP_EVAL") == "1":
    print("\nSKIP_EVAL=1 — probe 모드, 평가/저장/업로드 건너뛰고 종료")
    import sys
    sys.exit(0)

# ════════════════════════════════════════
# 7. 모델 저장
# ════════════════════════════════════════

print("\n[7/9] 모델 저장...")
notify_discord_json(discord_embed("💾 [7/9] LoRA 어댑터를 저장합니다."))
try:
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    with open(os.path.join(LORA_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
    lora_files = os.listdir(LORA_DIR)
    lora_size = sum(os.path.getsize(os.path.join(LORA_DIR, f)) for f in lora_files) / 1024**2
    print(f"  저장 경로: {LORA_DIR}/")
    print(f"  파일 수: {len(lora_files)}, 총 크기: {lora_size:.1f} MB")
    print(f"  class_names.json ({len(CLASS_NAMES)}클래스) 저장됨")
    notify_discord_json(discord_embed(f"✅ [7/9] 모델 저장 완료! ({LORA_DIR}/, {len(CLASS_NAMES)}클래스)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [7/9] 모델 저장 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 8. 평가 (test 200건 → evaluation_results.json)
# ════════════════════════════════════════

EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "val")
print(f"\n[8/9] 평가 (split={EVAL_SPLIT})...")
notify_discord_json(discord_embed(f"@everyone\n🔍 [8/9] 학습된 모델을 {EVAL_SPLIT} 데이터셋으로 평가합니다."))
EVAL_JSON_PATH = None
try:
    # 학습 후 GPU 메모리 정리 (LoRA 리로드 전)
    del trainer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from evaluate import evaluate as run_evaluation
    _, EVAL_JSON_PATH = run_evaluation(LORA_DIR)
    print(f"  평가 결과 저장: {EVAL_JSON_PATH}")
    notify_discord_json(discord_embed("@everyone\n✅ [8/9] 평가 완료! evaluation_results.json 저장됨."))
except Exception as e:
    notify_discord_json(discord_embed(f"@everyone\n❌ [8/9] 평가 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 9. HuggingFace Hub에 업로드
# ════════════════════════════════════════

print("\n[9/9] HuggingFace Hub 업로드...")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IS_SWEEP = bool(os.environ.get("WANDB_SWEEP_ID"))

if IS_SWEEP:
    print("  Sweep 모드 감지 — Hub 업로드 건너뜀 (우승자만 수동 재학습·업로드)")
    notify_discord_json(discord_embed("✅ [9/9] Sweep 모드 — Hub 업로드 건너뜀"))
elif HF_TOKEN:
    # org namespace (비워두면 HF_TOKEN 소유자의 개인 계정에 업로드)
    HF_ORG = os.environ.get("HF_ORG", "Himedia-AI-01")
    HUB_REPO = f"{HF_ORG}/pest-{RUN_NAME}" if HF_ORG else f"pest-{RUN_NAME}"
    notify_discord_json(discord_embed(f"☁️ [9/9] HuggingFace Hub에 업로드합니다. ({HUB_REPO})"))
    try:
        from huggingface_hub import HfApi, create_repo

        print(f"  대상 레포: {HUB_REPO}")
        t0 = time.time()
        repo_url = create_repo(HUB_REPO, token=HF_TOKEN, exist_ok=True, private=False)
        api = HfApi(token=HF_TOKEN)
        # LORA_DIR 전체 업로드 (LoRA 어댑터 + tokenizer + evaluation_results.json)
        api.upload_folder(
            folder_path=LORA_DIR,
            repo_id=repo_url.repo_id,
            commit_message=f"Upload {RUN_NAME}",
        )
        uploaded_files = os.listdir(LORA_DIR)
        hub_url = f"https://huggingface.co/{repo_url.repo_id}"
        print(f"  업로드 완료! ({time.time() - t0:.1f}s) — 파일 {len(uploaded_files)}개")
        print(f"  Hub URL: {hub_url}")
        has_eval = "evaluation_results.json" in uploaded_files
        notify_discord_json(discord_embed(
            f"✅ [9/9] 업로드 완료! ({HUB_REPO}) 🎉\n"
            f"🔗 {hub_url}\n"
            f"파일 {len(uploaded_files)}개 (evaluation_results.json {'포함' if has_eval else '없음'})"
        ))
    except Exception as e:
        notify_discord_json(discord_embed(f"❌ [9/9] Hub 업로드 중 에러 발생: {e}"))
        raise
else:
    print("  HF_TOKEN 미설정 — 업로드 건너뜀")
    notify_discord_json(discord_embed("✅ [9/9] Hub 업로드 건너뜀 (HF_TOKEN 미설정). 파이프라인 완료! 🎉"))

# Sweep 모드에서는 run 완료 후 trainer 체크포인트(OUTPUT_DIR) 정리.
# LoRA는 LORA_DIR에 이미 저장되었고 upload_best.py는 LORA_DIR만 스캔하므로
# OUTPUT_DIR의 체크포인트는 디스크만 잡아먹음. 20 run × ~1GB = ~20GB 절감.
# 단일 run 모드에서는 resume 가능성을 위해 유지.
if IS_SWEEP and os.path.exists(OUTPUT_DIR):
    try:
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print(f"Sweep 모드 — OUTPUT_DIR 정리: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"OUTPUT_DIR 정리 실패 (무시): {e}")

# wandb run 명시적 종료 (atexit을 기다리지 않고 즉시 Finished 상태로 전환)
# — sweep 중간 Ctrl+C 등으로 프로세스가 비정상 종료돼도 이 시점까진 깨끗이 마킹됨
if os.environ.get("WANDB_API_KEY"):
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("wandb run finished")
    except Exception as e:
        print(f"wandb finish 실패 (무시): {e}")

print("\n" + "=" * 60)
print("파이프라인 완료!")
print("=" * 60)
