# pest-detection-korean (full dataset)

노지 작물 해충 진단 - **전체 해충 데이터셋** LoRA 파인튜닝 (클래스 수 동적)

데이터셋은 [Himedia-AI-01/pest-detection-korean](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean)에서 학습 시작 시 자동 다운로드됩니다.

## 시작하기 전에 — 환경변수 설정

코드 수정 없이 환경변수만으로 모든 설정을 오버라이드할 수 있습니다.
RunPod 대시보드 → Pod 설정 → **Environment Variables**에 입력하거나, 로컬에서는 `export` / 실행 시 앞에 붙이세요 (`HF_TOKEN=hf_xxx python train.py`).

### ① 인증·알림 (외부 서비스 연결)

| 환경변수 | 필수 | 기본값 | 적용 스크립트 | 설명 |
|---|---|---|---|---|
| `HF_TOKEN` | **필수** | — | train.py, upload_best.py | HF Hub 데이터셋(gated) 다운로드 + LoRA 업로드. [발급](https://huggingface.co/settings/tokens) |
| `HF_ORG` | 선택 | `Himedia-AI-01` | train.py, upload_best.py | Hub 업로드 대상 org. 빈 문자열(`HF_ORG=""`)이면 토큰 소유자 개인 계정에 업로드 |
| `WANDB_API_KEY` | 선택 | — | train.py | W&B 실험 추적. 미설정 시 W&B 없이 학습. [발급](https://wandb.ai/authorize) |
| `WANDB_PROJECT` | 선택 | `pest-detection-full` | train.py | W&B project 이름. `setdefault`라 이미 set돼 있으면 그대로 사용 (sweep 시 `pest-detection-full-sweep` 등으로 덮어씀) |
| `DISCORD_WEBHOOK_URL` | 선택 | — | train.py, evaluate.py | 디스코드 Embed 알림. 미설정 시 조용히 진행. ⚠️ sweep 중엔 20 run × 9단계 = 180개 알림 폭탄이라 unset 권장 |

### ② 학습 하이퍼파라미터 (train.py 전용)

Sweep에서 agent가 주입하는 세 값(`LEARNING_RATE`, `LORA_R`, `LORA_ALPHA`)은 표시해뒀어요. Sweep 밖 단일 학습에서는 전부 수동 오버라이드 가능.

| 환경변수 | 기본값 | Sweep 주입 | 설명 |
|---|---|---|---|
| `BATCH_SIZE` | `6` |  | `per_device_train_batch_size` |
| `GRAD_ACCUM` | `2` |  | `gradient_accumulation_steps`. Total Batch = BATCH_SIZE × GRAD_ACCUM |
| `LORA_R` | `16` | ✅ | LoRA rank |
| `LORA_ALPHA` | `32` | ✅ | LoRA alpha. α=r 커플링 해제 (TML "LoRA Without Regret" 근거). 상세는 [HYPERPARAMETERS.md §5.3](HYPERPARAMETERS.md) |
| `LORA_DROPOUT` | `0.05` |  | LoRA dropout. Sweep에서 제외하고 고정 — ALLoRA(2410.09692) 근거 |
| `LEARNING_RATE` | `2e-4` | ✅ | 학습률. Sweep 범위 `log_uniform(1e-4, 5e-4)` |
| `NUM_EPOCHS` | `3` |  | 학습 에폭 수. `MAX_STEPS > 0`이면 무시됨 |
| `WARMUP_STEPS` | `150` |  | 워밍업 스텝 (총 step의 ~5%, 권장 5~10% 구간) |
| `MAX_STEPS` | `-1` |  | step 수 직접 고정 (예: `MAX_STEPS=250`). `-1`이면 epoch 기반 |

### ③ 경로·캐시·실행 제어

| 환경변수 | 기본값 | 적용 스크립트 | 설명 |
|---|---|---|---|
| `DATA_DIR` | `data` | train.py, evaluate.py | 데이터셋 경로. 비어있으면 HF Hub에서 자동 다운로드. 이미 받아둔 경로 재사용 시 지정 (예: `DATA_DIR=/workspace/pest-data`) |
| `DATASET_REPO` | `Himedia-AI-01/pest-detection-korean` | train.py | HF Hub dataset repo. 다른 데이터셋 재사용 시 변경 (클래스 수 동적) |
| `FORCE_DOWNLOAD` | `0` | train.py | `1`이면 기존 `DATA_DIR`을 무시하고 HF Hub에서 재다운로드 |
| `EVAL_SPLIT` | `val` | train.py, evaluate.py | 평가 split. `test.jsonl`이 있으면 `test`로 설정 가능 |
| `RUN_NAME` | (하이퍼파라미터 기반 자동 생성) | train.py | OUTPUT_DIR / LORA_DIR / W&B run name. 예: `r16_a32_d0.05_lr0.0002_bs6x2_ep3_w150`. 같은 이름 재실행 시 체크포인트 이어받아 resume. **Sweep에선 수동 설정 금지** (20 run이 같은 이름으로 충돌) |
| `HF_HOME` | `/workspace/hf_cache` (Linux) / 기본 `~/.cache/huggingface` (Windows) | train.py | HF cache 경로. ⚠️ torch/unsloth import **전에** 설정해야 반영됨 (train.py가 내부적으로 처리) |
| `TRANSFORMERS_CACHE` | `/workspace/hf_cache` (Linux) / 미설정 (Windows) | train.py | Transformers cache 경로. 대개 `HF_HOME`과 같이 맞춤 |

### ④ Sweep 자동 주입 (손대지 말 것)

| 환경변수 | 주입자 | 설명 |
|---|---|---|
| `LEARNING_RATE`, `LORA_R`, `LORA_ALPHA` | wandb agent | Bayes 샘플링 결과. Sweep 중 수동 export하면 agent 주입값과 충돌 |
| `WANDB_SWEEP_ID` | wandb agent | Sweep run임을 표시하는 핵심 플래그. train.py가 이걸 감지하면 **Hub 업로드 자동 skip** (sweep 후 `upload_best.py`로 winner만 업로드) |

---

## 개요

Qwen3.5-9B 비전-언어 모델을 LoRA로 파인튜닝하여, 작물 사진에서 **전체 19종 해충**(18 해충 + 정상)을 분류합니다. 클래스 이름과 개수는 학습 시 `DATA_DIR/train/` 하위 폴더명으로부터 동적으로 추출되므로, 데이터셋만 교체하면 다른 N-클래스 분류 작업에도 그대로 재사용 가능합니다.

| 항목 | 내용 |
|---|---|
| 모델 | `unsloth/Qwen3.5-9B` (bf16 LoRA) |
| 클래스 | 19개 (18 해충 + 정상, 동적 로딩) |
| 데이터 | Train 11,605건 / Val 1,595건 (test split 없음) |
| 환경 | 32GB+ VRAM (A5000/A6000) |
| 실험 추적 | Weights & Biases (wandb) — 선택 |
| 알림 | Discord Webhook (Embed) — 선택 |

> **⚠️ 주의:** Qwen3.5 모델에서는 **QLoRA (4-bit 양자화)가 권장되지 않습니다.** 반드시 bf16 LoRA를 사용하세요. 코드에는 `load_in_4bit=False` / `bf16=True`로 이미 고정돼 있으니 절대 바꾸지 말 것.

### 클래스 목록 (19)

검거세미밤나방, 꽃노랑총채벌레, 담배가루이, 담배거세미나방, 담배나방, 도둑나방, 먹노린재, 목화바둑명나방, 무잎벌, 배추좀나방, 배추흰나비, 벼룩잎벌레, 비단노린재, 썩덩나무노린재, 알락수염노린재, 정상, 큰28점박이무당벌레, 톱다리개미허리노린재, 파밤나방

## 빠른 시작 (RunPod / Vast.ai)

### 1. 레포 클론

```bash
git clone https://github.com/JunePark2018/finetuning-script.git -b full-dataset
cd finetuning-script
```

### 2. 환경 설정

```bash
bash setup.sh
```

이 스크립트가 하는 일:
- pip 패키지 설치 (unsloth, trl, wandb 등)
- Unsloth 설치 확인
- W&B 활성화 확인 (`WANDB_API_KEY` 설정 시)

### 3. 학습

```bash
HF_TOKEN=hf_xxx python train.py
```

첫 실행 시 `DATA_DIR/train.jsonl`이 없으면 HF Hub에서 데이터셋을 자동 다운로드합니다 (`DATA_DIR`이 비어있을 때만). 체크포인트가 있으면 자동으로 이어서 학습합니다 (동일 하이퍼파라미터 → 동일 `RUN_NAME` → 동일 `OUTPUT_DIR` 재사용).

### 4. 평가

```bash
python evaluate.py --model pest-lora-<RUN_NAME>
```

`EVAL_SPLIT=val`(기본)로 평가, 클래스 목록은 모델 디렉토리의 `class_names.json`을 읽어 동적으로 사용. 6개 메트릭 출력: Confusion Matrix, Accuracy, Precision, Recall, Macro F1, 추론 속도. 결과는 `<LORA_DIR>/evaluation_results.json`에 저장.

### 5. 추론

```bash
python inference.py --image test.jpg --model pest-lora-<RUN_NAME>
```

## 프로젝트 구조

```
.
├── train.py                 # 데이터 준비 + 전처리 + 학습 + 저장 + 평가 (올인원)
├── evaluate.py              # 학습 후 평가 split으로 성능 측정
├── inference.py             # 학습된 모델로 추론
├── upload_best.py           # Sweep 종료 후 최고 성능 모델만 Hub 업로드
├── setup.sh                 # GPU 서버 초기 설정
├── requirements.txt         # 의존성
└── data/                    # 데이터셋 (첫 실행 시 HF Hub에서 자동 다운로드)
    ├── train.jsonl
    ├── val.jsonl
    ├── train/
    │   ├── 검거세미밤나방/
    │   ├── 꽃노랑총채벌레/
    │   ├── ...              # 18 해충 폴더
    │   └── 정상/
    └── val/
        └── (동일 구조)
```

## 데이터셋 구조

각 데이터 샘플은 JSONL 형식의 대화 구조입니다:

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "당신은 작물 해충 식별 전문가입니다..."}]},
    {"role": "user", "content": [
      {"type": "image", "image": "train/검거세미밤나방/xxx.jpg"},
      {"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "검거세미밤나방"}]}
  ]
}
```

- **해충** 이미지에는 `.jpg.json` 파일이 함께 있으며, bbox 좌표(`xtl`, `ytl`, `xbr`, `ybr`)가 포함됩니다.
- **정상** 이미지에는 bbox JSON이 없습니다.
- 전처리 시 해충 이미지는 원본(50%) / bbox tight 크롭(25%) / bbox context 크롭(25%) 비율로 적용됩니다.

## 학습 산출물

학습이 완료되면 `pest-lora-<RUN_NAME>/` 디렉토리에 다음 파일이 생성됩니다:

- LoRA 어댑터 (`adapter_config.json`, `adapter_model.safetensors`)
- Tokenizer 파일
- `class_names.json` — 학습 시 사용된 클래스 목록 (evaluate/inference가 읽음)
- `evaluation_results.json` — 6개 메트릭 결과

HF_TOKEN이 설정되어 있으면 자동으로 `{HF_ORG}/pest-<RUN_NAME>` 레포에 업로드됩니다.

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 오버라이드 |
|---|---|---|
| LoRA r | 16 | `LORA_R=8` |
| LoRA alpha | r과 동일 (자동) | `LORA_ALPHA=32` (명시 시에만 분리) |
| LoRA dropout | 0.0 | `LORA_DROPOUT=0.05` |
| Batch size | 6 (effective 12) | `BATCH_SIZE=4 GRAD_ACCUM=4` |
| Learning rate | 2e-4 | `LEARNING_RATE=1e-4` |
| Epochs | 3 | `NUM_EPOCHS=5` |
| Warmup | 150 steps | `WARMUP_STEPS=100` |
| Max steps | -1 (epoch 사용) | `MAX_STEPS=250` |
| Weight decay | 0.01 | - |
| Optimizer | AdamW 8bit | - |
| Scheduler | Cosine | - |

```bash
# 기본값으로 실행
HF_TOKEN=hf_xxx python train.py

# 파라미터 오버라이드
HF_TOKEN=hf_xxx BATCH_SIZE=4 LORA_R=8 LEARNING_RATE=1e-4 python train.py

# 기존 서브셋 데이터를 재사용하려면 DATA_DIR 지정
DATA_DIR=/path/to/preexisting-data python train.py
```

## Sweep 실행하기 (W&B Sweeps)

하이퍼파라미터 bayes 탐색을 위한 [`sweep.yaml`](sweep.yaml) 설정 포함. `LEARNING_RATE` × `LORA_R` × `LORA_ALPHA` 3차원 공간에서 서비스 composite metric `eval/pest_gated_f1`을 최대화합니다.

### 탐색 공간 (sweep.yaml)

| 파라미터 | 분포/값 |
|---|---|
| `LEARNING_RATE` | log_uniform 1e-4 ~ 5e-4 |
| `LORA_R` | {8, 16, 32} |
| `LORA_ALPHA` | {16, 32, 64} |

- 방법: `method: bayes` (sequential, 단일 Pod 권장)
- 총 run: 20 (`run_cap: 20`)
- Metric: **`eval/pest_gated_f1`** = `binary_pest_recall × macro_f1_on_18_pest_classes` (서비스 실패 비용 비대칭을 직접 반영하는 composite)
- 고정: `LORA_DROPOUT=0.05`, `WARMUP_STEPS=150` (~5%), `weight_decay=0.01`, `NUM_EPOCHS=3`, scheduler/batch 등
- `LORA_DROPOUT`은 3 epoch 학습 구간에서 분산만 키운다는 ALLoRA(2410.09692) 근거로 sweep에서 제외하고 0.05 고정.
- 설계 근거 상세: [HYPERPARAMETERS.md](HYPERPARAMETERS.md)

### Sweep 중 환경변수

| 분류 | 변수 | 비고 |
|---|---|---|
| 🔴 **필수 (수동 설정)** | `HF_TOKEN`, `WANDB_API_KEY` | Pod 시작 후 export |
| 🟡 선택 | `DISCORD_WEBHOOK_URL` | ⚠️ **20 run × 9단계 = 180개 알림** 폭탄. sweep 중엔 unset 권장 |
| 🟡 선택 | `DATA_DIR`, `HF_ORG` | 기본값으로 충분 (`HF_ORG`는 sweep 중엔 업로드 skip이라 무시됨) |
| 🟢 Agent 자동 주입 | `LEARNING_RATE`, `LORA_R`, `LORA_ALPHA` | Bayes 샘플링 결과 주입 |
| 🟢 Agent 자동 주입 | `WANDB_SWEEP_ID` | **핵심 플래그** — train.py가 이걸 감지해 Hub 업로드 skip |
| ⚫ **설정 금지** | `LEARNING_RATE`/`LORA_R`/`LORA_ALPHA` 수동 export | Agent 주입값과 충돌 |
| ⚫ **설정 금지** | `RUN_NAME` | 고정하면 20 run이 같은 이름으로 충돌. 하이퍼파라미터 기반 자동 생성 유지 |

### 실행 절차

**1. Sweep 발급 (한 번만, 어디서든)**
```bash
wandb sweep --project pest-detection-full-sweep sweep.yaml
# 출력:
# wandb: Created sweep with ID: abc123xyz
# wandb: View sweep at: https://wandb.ai/<entity>/pest-detection-full-sweep/sweeps/abc123xyz
```

**2. Pod에서 agent 실행**
```bash
HF_TOKEN=hf_xxx WANDB_API_KEY=... wandb agent <entity>/pest-detection-full-sweep/abc123xyz
```

Agent가 자동으로 수행:
- 컨트롤러에서 다음 파라미터 조합 받음
- env로 주입 (`LEARNING_RATE`, `LORA_R`, `LORA_DROPOUT`)
- `python train.py` 실행
- 학습 종료 후 `eval/pest_gated_f1`을 컨트롤러에 보고
- **Hub 업로드는 자동 skip** (`WANDB_SWEEP_ID` env 감지)
- 다음 조합 pull → 반복 (총 20 run)

**3. 결과 확인**

W&B 대시보드 → `pest-detection-full-sweep` 프로젝트:
- **Parallel coordinates** 뷰: 각 파라미터와 `pest_gated_f1` 관계 시각화
- **Importance** 뷰: 어느 파라미터가 성능에 가장 영향 주는지
- **우승 run 확인**: sort by `eval/pest_gated_f1` desc → 상위 조합의 LR/r/α 값 기록
- **서비스 건전성 진단 (필수)**: 단순 최상위 run을 고르지 말고 `binary_pest_recall`·`normal_specificity` 균형, `fn_pest_to_normal` 절대 카운트 함께 확인. 자세한 체크리스트는 [HYPERPARAMETERS.md §9.0](HYPERPARAMETERS.md).

**4. Sweep 우승자 Hub 업로드 (재학습 불필요)**

Sweep의 각 run은 이미 full 3 epoch 학습 + 평가를 완료한 상태. 재학습 없이 **가장 성능 좋은 모델만 골라서 업로드**:

```bash
HF_TOKEN=hf_xxx python upload_best.py
```

`upload_best.py`가 자동 수행:
- 로컬의 모든 `pest-lora-*/` 디렉토리 스캔
- 각 `evaluation_results.json`에서 `f1_macro` 읽음
- 가장 높은 모델을 `{HF_ORG}/pest-{RUN_NAME}` 레포에 업로드

업로드 전 랭킹만 확인:
```bash
python upload_best.py --dry-run --top-k 10
```

### 주의사항

- **단일 Pod 순차 실행 권장**: bayes는 "이전 결과 보고 다음 추천"이라 Pod 여러 개 동시 실행 시 random으로 퇴화
- **wall-clock**: 6.5h × 20 run ≈ **130h (≈ 5.4일)**. Pod 오래 유지할 계획 세우기
- **Sweep 중 Pod 재시작**: 같은 `wandb agent <ID>` 명령으로 다시 띄우면 중단된 지점부터 sweep 계속
- **Discord 알림**: Sweep 중에도 매 run마다 [1/9]~[9/9] 알림 감. 조용하게 하려면 Pod의 `DISCORD_WEBHOOK_URL`을 sweep 중 unset

## Discord 알림

`DISCORD_WEBHOOK_URL`이 설정되면, 각 단계의 시작/완료/에러를 디스코드 채널에 Embed 형식으로 전송합니다.

| 스크립트 | 단계 | 내용 |
|---|---|---|
| train.py | [1/9]~[9/9] | 데이터 준비 → 전처리 → 로딩 → 모델 → LoRA → 학습 → 저장 → 평가 → 업로드 |
| evaluate.py | [1/3]~[3/3] | 모델 로딩 → 추론 → 결과 집계 |

## W&B (Weights & Biases)

`WANDB_API_KEY`가 설정되면 자동으로 W&B에 기록됩니다.

- 대시보드에서 `train/loss`, `eval/loss`, `learning_rate` 등을 실시간 확인
- **Project**: 기본 `pest-detection-full`. `WANDB_PROJECT` 환경변수로 오버라이드 가능
- **Run name**: `RUN_NAME` (예: `r16_a16_d0.0_lr0.0002_bs6x2_ep3_w150`)
