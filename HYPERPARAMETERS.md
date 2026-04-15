# 하이퍼파라미터 설계 리포트

**대상:** `finetuning-script-sweep` (Qwen3.5-9B VLM × 19-class 한국어 해충 분류)
**작성일:** 2026-04-15
**범위:** `sweep.yaml` 탐색 공간 + `train.py` 고정값 결정 근거

---

## 1. TL;DR

- **탐색 대상 3개 (3D Bayes):** `LEARNING_RATE` (log 5e-5~5e-4), `LORA_R` ({8,16,32}), `LORA_DROPOUT` ({0, 0.05, 0.1})
- **고정값:** `LORA_ALPHA=LORA_R`, `WARMUP_STEPS=150` (~5%), `weight_decay=0.01`, `NUM_EPOCHS=3`, `BATCH_SIZE=6×2=12`
- **Metric:** `eval/macro_f1` 최대화
- **Run budget:** 20 run × ~6.5h ≈ 130h
- **설계 원칙:** 연구 권장값 + 데이터셋 특성(11× 불균형, 11,605 샘플) + 태스크 특성(단일 토큰 분류)의 교집합에 탐색 집중. 영향력 작은 파라미터는 고정.

---

## 2. 태스크 특성

| 항목 | 값 | 설계 함의 |
|---|---|---|
| 모델 | Qwen3.5-9B VLM (bf16 LoRA) | ⚠️ QLoRA 비권장 — 4-bit 양자화 금지 |
| 클래스 | 19 (18 해충 + 정상) | 소규모 분류, rank 크게 필요 없음 |
| Instruction | SYSTEM_MSG에 19 클래스 enumeration + strict output rule | 모델이 선택지를 명시적으로 인지 |
| Input | 이미지 + 고정 USER_PROMPT | 프롬프트 분산 없음 → 학습 분포 단순 |
| Output | 단일 클래스명 (한글 7~10자, ~20~30 토큰) | 짧은 출력 → 빠른 수렴 기대 |
| Train 샘플 | 11,605 | 중간 규모 |
| 클래스 불균형 | **최대/최소 = 11×** (정상 957 / 배추좀나방 86) | 정규화(dropout) 중요도 상승 |
| Step/epoch | ~967 (batch 12 기준) | 3 epoch ≈ 2,900 step |

---

## 3. 설계 원칙

1. **영향력 큰 파라미터에 예산 집중** — 연구 일관되게 LR·rank > 기타
2. **차원 수 ≤ 3** — 3D bayes가 20 run 내 수렴 가능한 실질 한계
3. **고정값은 연구 권장값의 중심/하한 채택** — 실패 확률 최소화
4. **태스크 특성 반영** — 불균형·작은 출력·단일 프롬프트

---

## 4. 탐색 파라미터별 근거

### 4.1 `LEARNING_RATE` — log_uniform(5e-5, 5e-4)

**근거:**
- Unsloth 공식 가이드: LoRA 시작점 `2e-4`, 범위 `2e-4 ~ 5e-6`
- Thinking Machines Lab: "LoRA 최적 LR = Full FT 최적 LR × 10"
- 현재 범위는 `2e-4`를 중심으로 한 decade (0.25× ~ 2.5×) 커버
- `log_uniform`으로 오더별 샘플링 → LR의 자연스러운 스케일 특성 반영

**대안 검토:**
- 더 넓게 (1e-5 ~ 1e-3): 탐색 비효율, 하단·상단 winner 가능성 낮음
- 더 좁게 (1e-4 ~ 5e-4): 효율 높으나 하단 샘플 낭비 제거 vs 안전 margin 상실
- 현재 범위는 **안전 margin**과 **탐색 효율** 균형점

**확신도:** ★★★

---

### 4.2 `LORA_R` — {8, 16, 32}

**근거:**
- Unsloth: "simple task는 8~16, complex는 32~64"
- 본 태스크는 19-class 분류 + 짧은 출력 → **simple 구간**
- Thinking Machines: "high-rank LoRA는 loss 감소 곡선이 유사" — 충분한 capacity만 있으면 rank에 민감하지 않음
- 3값 categorical → bayes가 각 값 평균 6~7 run 할당 (20 run 기준)

**대안 검토:**
- `{4, 8, 16}`: 더 공격적 축소. 32를 날리는 건 "한 번쯤은 확인해보자"의 가치 포기
- `{8, 16, 32, 64}`: 4값 시 각 값 5 run 평균 → 신호 약해짐

**확신도:** ★★★★

---

### 4.3 `LORA_DROPOUT` — {0, 0.05, 0.1}  *(신규 추가)*

**근거:**
- Unsloth: "smaller datasets benefit from higher dropout (0.05~0.1)"
- Brenndoerfer: "higher ranks have more capacity to overfit → dropout more valuable"
- 본 태스크: 11,605 샘플 = 중간 규모, **11× 불균형** → majority class (정상 957건) 오버피팅 위험
- 기존 고정값 `0`은 정규화 없음 → overfitting 관찰 시 조치 불가

**대안 검토:**
- 연속 분포(`uniform(0, 0.15)`): categorical이 해석·시각화 용이
- `{0, 0.1}` 2값: dropout 민감도 스윕 해상도 부족

**확신도:** ★★

**한계:** Thinking Machines는 "dropout may be unreliable for short training runs" 언급 — 3 epoch이 short에 해당할 수 있음. 실제 효과는 실험 후 판단 필요.

---

## 5. 고정 파라미터 근거

### 5.1 `WARMUP_STEPS = 150`  *(sweep에서 제외)*

**제외 근거:**
- 영향력 순위에서 LR·rank 대비 후순위 (연구 공통 인식)
- 기존 범위 30~300 중 **하단 30은 총 step의 1%** → warmup 효과 거의 없어 부적절한 탐색값 포함
- 20 run × 3차원 bayes에서 저영향 변수에 차원 할당은 예산 낭비

**150 선택 근거:**
- 총 step ~2,900 × 5% = 145 ≈ **150**
- Unsloth 권장 "5~10% of total steps"의 **하한**
- 하한 선택: cosine schedule에서 warmup 비율이 크면 peak LR 구간이 짧아짐 → 보수적 선택

**한계:** LR이 높을수록 warmup이 더 필요할 수 있는데 (상호작용), 고정 150이 이를 커버하지 못함. 실제로 최적 LR이 5e-4 근처로 나오면 200~250으로 재조정 여지.

---

### 5.2 `weight_decay = 0.01`  *(기본값 상향)*

**근거:**
- 기존 `0.001`은 Unsloth 권장 범위(`0.01~0.1`)의 **10배 낮음**
- `0.01`은 권장 범위 하한 — 안전하고 연구 부합
- LoRA 학습에서 weight_decay는 전체 파라미터 대비 영향 작다는 경험적 리포트 존재 → 공격적으로 높일 이유 없음

**확신도:** ★★★

---

### 5.3 `LORA_ALPHA = LORA_R` (자동 커플링)

**근거:**
- `alpha = r` (scaling factor 1): Unsloth 공식 기본값 중 하나, 보수적 선택
- Thinking Machines는 `alpha=32 + 1/r scaling` 사용 (다른 방식)
- `alpha=2r`도 유효한 관례지만 **별도 차원으로 스윕하면 4D bayes** → 효율 저하

**대안 (향후):**
- Sweep 결과 best run에서 `alpha=2r`로 수동 비교 실험 가능

---

### 5.4 `NUM_EPOCHS = 3`

**근거:**
- Unsloth: "3 epoch 초과 시 overfitting 위험"
- 2 epoch으로 단축하면 run 시간 33% 감소하나, 불균형 minority class 학습 부족 우려
- 3 epoch은 **권장 상한** — 안전하되 공격적

**한계:** 불균형 환경에서 majority class가 2 epoch에 이미 수렴, minority는 3 epoch에서도 부족할 수 있음. Class-weighted loss나 balanced sampler가 근본 해결책이나 범위 밖.

---

### 5.5 기타 고정값

| 파라미터 | 값 | 근거 |
|---|---|---|
| `BATCH_SIZE` × `GRAD_ACCUM` | 6 × 2 = 12 | VRAM 제약, bf16 + gradient checkpointing 기준 |
| `optim` | `adamw_8bit` | VRAM 절감, Unsloth 기본 권장 |
| `lr_scheduler_type` | `cosine` | LoRA 학습의 표준 선택 |
| `bf16` | `True` | **Qwen3.5는 bf16 필수, 4-bit 양자화 비권장** |
| `max_seq_length` | 2048 | 이미지 토큰 + 짧은 텍스트에 여유 |

---

## 6. Sweep Method

### 6.1 `method: bayes`

**근거:**
- 3차원 연속/카테고리 혼합 공간에 효율적
- `random`보다 20 run 한정 예산에서 수렴 속도 우위
- `grid`는 3×3×3=27 조합 초과 시 비실용적
- **제약:** 순차 실행 필수 (동시 agent는 bayes GP를 random으로 퇴화시킴) → 단일 Pod 권장

### 6.2 `run_cap: 20`

**근거:**
- 3D bayes는 보통 10~15 run으로 수렴, 20은 **refinement 여유 포함**
- `LORA_R` 3 카테고리 × `LORA_DROPOUT` 3 카테고리 = 9 grid point → 각 조합 평균 2+ run 확보
- 시간: 20 × 6.5h ≈ **130h** (5.4일). Pod 비용 허용 시 표준 선택.

**대안:** 15 run (3.4일), 12 run (3.3일). 본 설정은 품질 우선.

---

## 7. 한계 및 미검증 가정

1. **본 구성은 Qwen3.5-9B + 한국어 해충 분류에 대한 직접 벤치마크가 아님.** 일반 best practice + 태스크 특성의 교집합.
2. **클래스 불균형(11×)의 근본 해결은 sweep 바깥.** Dropout으로 완화는 되나, WeightedRandomSampler·class-weighted loss가 더 직접적.
3. **WARMUP_STEPS 고정은 LR과의 상호작용 탐색 불가.** 최적 LR이 5e-4 근처이면 warmup 재조정 재실험 필요.
4. **DROPOUT 효과는 3 epoch 학습에서 신뢰도 낮을 수 있음** (Thinking Machines 지적).
5. **Sweep은 winner 1개만 산출.** 여러 seed에서의 분산은 측정 안 됨 — sweep 후 best 조합으로 seed 3~5개 재학습해야 성능 안정성 확인 가능.
6. **LORA_ALPHA 고정은 α=2r 탐색 기회 포기.** Sweep 후 보완 실험 권장.

---

## 8. 후속 실험 우선순위

Sweep 종료 후 성능이 기대 이하이면, 우선순위 순:

1. **Class imbalance 직접 처리** — `WeightedRandomSampler` 또는 `class_weights` loss. Hyperparam 튜닝보다 효과 클 가능성.
2. **`LORA_ALPHA = 2 × LORA_R` 재학습** — winner hyperparam 고정, α만 변경해 A/B
3. **NUM_EPOCHS 2 vs 3 vs 5** — early-stop 도입 또는 epoch 스윕
4. **Seed 분산 측정** — winner 조합으로 seed 3~5개 재학습
5. **Prompt engineering 반복** — SYSTEM_MSG의 포맷 미세 변형이 macro_f1 0.5~1% 움직일 수 있음

---

## 9. 참고문헌

- **Unsloth 공식 LoRA 가이드** — [unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- **Thinking Machines Lab — "LoRA Without Regret"** — [thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/)
- **Michael Brenndoerfer — LoRA Hyperparameters: Rank, Alpha & Target Module Selection** — [mbrenndoerfer.com/writing/lora-hyperparameters-rank-alpha-target-modules](https://mbrenndoerfer.com/writing/lora-hyperparameters-rank-alpha-target-modules)
- **Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA** — [magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- **PLoRA: Efficient LoRA Hyperparameter Tuning for Large Models (2025)** — [arxiv.org/html/2508.02932v1](https://arxiv.org/html/2508.02932v1)
