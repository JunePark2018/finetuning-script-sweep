# pest-detection-korean (full dataset)

노지 작물 해충 진단 - **전체 해충 데이터셋** LoRA 파인튜닝 (클래스 수 동적)

데이터셋은 [Himedia-AI-01/pest-detection-korean](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean)에서 학습 시작 시 자동 다운로드됩니다.

## 시작하기 전에 — 환경변수 설정

RunPod 대시보드 → Pod 설정 → **Environment Variables**에 아래 값을 입력하세요.

| 환경변수 | 필수 | 값 | 설명 |
|---|---|---|---|
| `HF_TOKEN` | **필수** | `hf_xxx...` | 데이터셋(gated) 다운로드 + Hub 업로드. [발급](https://huggingface.co/settings/tokens) |
| `HF_ORG` | 선택 | `Himedia-AI-01` (기본) | Hub 업로드 대상 org. 빈 문자열(`HF_ORG=""`)이면 토큰 소유자의 개인 계정에 업로드 |
| `WANDB_API_KEY` | 선택 | `xxx...` | W&B 실험 추적. 미설정 시 W&B 없이 학습. [발급](https://wandb.ai/authorize) |
| `DISCORD_WEBHOOK_URL` | 선택 | `https://discord.com/api/webhooks/...` | 디스코드 알림. 미설정 시 알림 없이 진행 |

하이퍼파라미터도 환경변수로 오버라이드할 수 있습니다 (train.py 전용):

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `BATCH_SIZE` | `6` | per_device_train_batch_size |
| `GRAD_ACCUM` | `2` | gradient_accumulation_steps (Total Batch = BATCH_SIZE × GRAD_ACCUM) |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha |
| `LEARNING_RATE` | `2e-4` | 학습률 |
| `NUM_EPOCHS` | `3` | 학습 에폭 수 (`MAX_STEPS > 0`일 땐 무시됨) |
| `WARMUP_STEPS` | `50` | 워밍업 스텝 |
| `MAX_STEPS` | `-1` | step 수 직접 고정 (예: `MAX_STEPS=250`). `-1`이면 epoch 기반 |
| `DATA_DIR` | `data` | 데이터셋 경로. 비어있으면 HF Hub에서 자동 다운로드됨 |
| `DATASET_REPO` | `Himedia-AI-01/pest-detection-korean` | HF Hub dataset repo |
| `FORCE_DOWNLOAD` | `0` | `1`이면 기존 `DATA_DIR`을 무시하고 재다운로드 |
| `EVAL_SPLIT` | `val` | 평가 split. 기본 `val` (`test` split이 없는 경우 대비) |

코드 수정 없이 환경변수만으로 모든 설정이 가능합니다.

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

> **노트북 사용 시 주의**: `[1/10] 패키지 설치` 셀 실행 후 **Kernel → Restart Kernel**이 필요합니다. 노트북 안에 안내 셀이 들어 있습니다. 또는 터미널에서 `bash setup.sh`를 먼저 돌리면 재시작 없이 진행 가능.

### 3. 학습

```bash
HF_TOKEN=hf_xxx python train.py
```

첫 실행 시 `DATA_DIR/train.jsonl`이 없으면 HF Hub에서 데이터셋을 자동 다운로드합니다 (`DATA_DIR`이 비어있을 때만). 체크포인트가 있으면 자동으로 이어서 학습합니다.

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
├── pest_detection.ipynb     # 노트북 버전 (단계별 실행)
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
| Batch size | 6 (effective 12) | `BATCH_SIZE=4 GRAD_ACCUM=4` |
| Learning rate | 2e-4 | `LEARNING_RATE=1e-4` |
| Epochs | 3 | `NUM_EPOCHS=5` |
| Warmup | 50 steps | `WARMUP_STEPS=100` |
| Max steps | -1 (epoch 사용) | `MAX_STEPS=250` |
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

하이퍼파라미터 bayes 탐색을 위한 [`sweep.yaml`](sweep.yaml) 설정 포함. `LEARNING_RATE` × `LORA_R` × `WARMUP_STEPS` 3차원 공간에서 `eval/macro_f1`을 최대화합니다.

### 탐색 공간 (sweep.yaml)

| 파라미터 | 분포/값 |
|---|---|
| `LEARNING_RATE` | log_uniform 5e-5 ~ 5e-4 |
| `LORA_R` | {8, 16, 32} |
| `WARMUP_STEPS` | int_uniform 30 ~ 300 |

- 방법: `method: bayes` (sequential, 단일 Pod 권장)
- 총 run: 20 (`run_cap: 20`)
- Metric: `eval/macro_f1` 최대화
- 고정: `LORA_ALPHA = LORA_R` (자동), `NUM_EPOCHS=3`, scheduler/batch 등

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
- env로 주입 (`LEARNING_RATE`, `LORA_R`, `WARMUP_STEPS`)
- `python train.py` 실행
- 학습 종료 후 `eval/macro_f1`을 컨트롤러에 보고
- **Hub 업로드는 자동 skip** (`WANDB_SWEEP_ID` env 감지)
- 다음 조합 pull → 반복 (총 20 run)

**3. 결과 확인**

W&B 대시보드 → `pest-detection-full-sweep` 프로젝트:
- **Parallel coordinates** 뷰: 각 파라미터와 macro_f1 관계 시각화
- **Importance** 뷰: 어느 파라미터가 성능에 가장 영향 주는지
- **우승 run 확인**: sort by `eval/macro_f1` desc → 상위 조합의 LR/r/warmup 값 기록

**4. 우승자로 정식 학습 + Hub 업로드**
```bash
# Sweep 결과에서 best 파라미터 복사
LEARNING_RATE=<best_lr> LORA_R=<best_r> WARMUP_STEPS=<best_warmup> \
  HF_TOKEN=hf_xxx WANDB_API_KEY=... python train.py
# WANDB_SWEEP_ID 없으므로 Hub 업로드도 정상 수행
```

필요하면 `NUM_EPOCHS=5`로 늘려서 더 학습시켜 보기.

### 주의사항

- **단일 Pod 순차 실행 권장**: bayes는 "이전 결과 보고 다음 추천"이라 Pod 여러 개 동시 실행 시 random으로 퇴화
- **wall-clock**: 6.5h × 20 run ≈ **130h (≈ 5.4일)**. Pod 오래 유지할 계획 세우기
- **Sweep 중 Pod 재시작**: 같은 `wandb agent <ID>` 명령으로 다시 띄우면 중단된 지점부터 sweep 계속
- **Discord 알림**: Sweep 중에도 매 run마다 [1/9]~[9/9] 알림 감. 조용하게 하려면 Pod의 `DISCORD_WEBHOOK_URL`을 sweep 중 unset

## Discord 알림

`DISCORD_WEBHOOK_URL`이 설정되면, 각 단계의 시작/완료/에러를 디스코드 채널에 Embed 형식으로 전송합니다. 학습/추론/에러 단계에는 `@everyone` 멘션이 포함됩니다.

| 스크립트 | 단계 | 내용 |
|---|---|---|
| train.py | [1/9]~[9/9] | 데이터 준비 → 전처리 → 로딩 → 모델 → LoRA → 학습 → 저장 → 평가 → 업로드 |
| 노트북 | [1/10]~[10/10] | 패키지 설치 + 평가 포함 |
| evaluate.py | [1/3]~[3/3] | 모델 로딩 → 추론 → 결과 집계 |

## W&B (Weights & Biases)

`WANDB_API_KEY`가 설정되면 자동으로 W&B에 기록됩니다.

- 대시보드에서 `train/loss`, `eval/loss`, `learning_rate` 등을 실시간 확인
- **Project**: 기본 `pest-detection-full`. `WANDB_PROJECT` 환경변수로 오버라이드 가능
- **Run name**: `RUN_NAME` (예: `r16_a16_lr0.0002_bs6x2_ep3_w50`)
