# 하이퍼파라미터 설계 리포트

**대상:** `finetuning-script-sweep` (Qwen3.5-9B VLM × 19-class 한국어 해충 분류)
**작성일:** 2026-04-15 (2025-2026 VLM LoRA 연구·실무 best practice 웹 조사 반영: α 커플링 해제, LR 범위 축소, dropout 고정, composite sweep metric 도입)
**범위:** `sweep.yaml` 탐색 공간 + `train.py` 고정값 결정 근거

---

## 1. TL;DR

- **서비스 목표:** 농부가 작물 사진을 올려 "해충 있냐?"를 묻는 시나리오. **"해충 → 정상" 오분류(FN)가 "정상 → 해충" 오분류(FP)보다 훨씬 치명적** (피해 확산 vs 불필요한 농약).
- **탐색 대상 3개 (3D Bayes):** `LEARNING_RATE` (log 1e-4~5e-4), `LORA_R` ({8,16,32}), `LORA_ALPHA` ({16,32})
- **고정값:** `LORA_DROPOUT=0.05`, `WARMUP_STEPS=150` (~5%), `weight_decay=0.01`, `NUM_EPOCHS=3`, `BATCH_SIZE=6×2=12`, `max_grad_norm=1.0`
- **Primary metric:** `eval/pest_gated_f1 = binary_pest_recall × macro_f1_on_18_pest_classes` — 서비스 비용 비대칭을 sweep이 직접 최적화. 곱셈이라 "해충 잡기" AND "해충 ID 정확도" 둘 다 높아야 점수 남.
- **Secondary logged metrics:** `binary_pest_recall`, `binary_pest_f2`, `normal_specificity`, `macro_f1` (전체), `macro_f1_on_pests` (해충만), per-class F1 (`eval/f1_per_class/<class>`), confusion matrix의 FN/FP 절대 카운트.
- **Run budget:** 20 run × ~12h ≈ **240h (10일)** — 실측 기준 (Qwen3.5-9B + A6000 + unsloth 2× 가속).
- **설계 원칙:** 2025-2026 VLM LoRA 연구(TML "LoRA Without Regret", ALLoRA, Unsloth guide) + 데이터셋 특성(4.6× 불균형, 11,605 샘플) + 서비스 실패 비용 비대칭성의 교집합.

---

## 2. 서비스 맥락 (설계의 중심축)

파인튜닝의 downstream 목적이 학술 벤치마크가 아닌 **농부용 의사결정 도구**이기 때문에, 몇 가지 일반적인 분류 모델 설계 원칙과 다른 방향이 선택된다.

### 2.1 실패 비용 비대칭

| 오분류 유형 | 농부 행동 | 결과 | 심각도 |
|---|---|---|---|
| 해충 → **정상** (FN) | 무대응 | 해충 방치 → 피해 확산 | 🔴 치명적 |
| **정상** → 해충 (FP) | 농약 살포 | 불필요한 비용·작물 스트레스 | 🟡 나쁨 |
| 해충 X → 해충 Y | 해충 대응 시작 | 어쨌든 조기 발견, 오판 후 재확인 가능 | 🟢 수용 가능 |

### 2.2 이 비대칭이 설계에 미치는 영향

- **Metric 선택:** macro_f1은 클래스 동등 가중 → 정상 다수로의 편향에 자연 페널티. 서비스 목표와 정렬됨. 굳이 weighted_f1로 바꿀 이유 없음.
- **Dropout 탐색 정당화:** 정상(1,381)이 최대 클래스이므로 모델이 정상 쪽으로 편향되기 쉬움 → dropout은 "일반화"보다 **"정상 바이어스 억제"** 맥락에서 필요.
- **Post-sweep 진단 우선순위:** confusion matrix에서 **column="정상" 의 pest rows 합계 (FN)** 가 가장 중요한 건전성 지표. 이 값이 높으면 다른 지표가 좋아도 서비스 부적합.
- **입력 분포 가정:** 농부는 멀리서 원본 사진 촬영 → evaluate.py는 원본 100%. train.py의 50% 원본 + 50% 크롭 혼합은 크기 변이에 대한 augmentation이지 eval 분포를 반영하는 게 아님 (의도적 비대칭).

---

## 3. 태스크 특성

| 항목 | 값 | 설계 함의 |
|---|---|---|
| 모델 | Qwen3.5-9B VLM (bf16 LoRA) | ⚠️ QLoRA 비권장 — 4-bit 양자화 금지 |
| 클래스 | 19 (18 해충 + 정상) | 소규모 분류, rank 크게 필요 없음 |
| Instruction | SYSTEM_MSG에 19 클래스 enumeration + strict output rule | 모델이 선택지를 명시적으로 인지 |
| Input | 이미지 + 고정 USER_PROMPT | 프롬프트 분산 없음 → 학습 분포 단순 |
| Output | 단일 클래스명 (한글 7~10자, ~20~30 토큰) | 짧은 출력 → 빠른 수렴 기대 |
| Train 샘플 | 11,605 | 중간 규모 |
| 클래스 불균형 | **최대/최소 = 4.6×** (정상 1,381 / 최소 300: 검거세미밤나방·도둑나방·무잎벌·비단노린재) | 완만한 불균형. 정규화(dropout)는 일반화 목적이 주, 불균형 보정은 부차 |
| Step/epoch | ~967 (batch 12 기준) | 3 epoch ≈ 2,900 step |

---

## 4. 설계 원칙

1. **영향력 큰 파라미터에 예산 집중** — 연구 일관되게 LR·rank > 기타
2. **차원 수 ≤ 3** — 3D bayes가 20 run 내 수렴 가능한 실질 한계
3. **고정값은 연구 권장값의 중심/하한 채택** — 실패 확률 최소화
4. **태스크 특성 반영** — 불균형·작은 출력·단일 프롬프트

---

## 5. 탐색 파라미터별 근거

### 5.1 `LEARNING_RATE` — log_uniform(1e-4, 5e-4)

**근거:**
- Unsloth Qwen3.5 fine-tune 가이드: LoRA 시작점 `2e-4`
- Thinking Machines Lab ("LoRA Without Regret", 2025): 실측 최적 LR 1e-4 ~ 5e-4. "LoRA 최적 LR ≈ Full FT 최적 LR × 10"
- 이 범위는 `2e-4`를 중심으로 `0.5× ~ 2.5×` 커버 — 실질적 productive zone에 집중
- `log_uniform`으로 오더 내 균등 샘플링 → LR의 자연스러운 스케일 특성 반영

**대안 검토 (및 기각 사유):**
- 이전 설계 `(5e-5, 5e-4)`: 하단 5e-5는 **Full FT 영역**으로 LoRA에 지나치게 낮음 → bayes 예산 낭비. 2025 연구 기반으로 1e-4로 상향.
- 더 넓게 `(1e-5, 1e-3)`: 탐색 비효율, winner 가능성 낮은 구간 포함
- 더 좁게 `(1.5e-4, 3e-4)`: Bayes의 탐색 자유도 너무 제약

**확신도:** ★★★★

---

### 5.2 `LORA_R` — {8, 16, 32}

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

### 5.3 `LORA_ALPHA` — {16, 32}  *(α=r 커플링 해제, γ ≤ 4 안전역)*

**근거:**
- Thinking Machines Lab ("LoRA Without Regret", 2025): α를 rank와 분리해 **고정**하면 optimal LR이 rank에 독립적. α=r로 묶으면 γ=α/r이 항상 1이라 겉보기엔 깔끔하지만, 실제론 LR-rank 상호작용이 sweep 신호에 섞임.
- α를 별도 sweep 차원으로 두면 γ=α/r scaling factor가 **{0.5, 1, 2, 4}** 스펙트럼으로 탐색됨:
  - r=8, α=32: γ=4 (상한, capacity 작고 업데이트 강함)
  - r=8, α=16: γ=2 (Unsloth α=2r 관행)
  - r=16, α=32: γ=2 (표준)
  - r=32, α=16: γ=0.5 (하한, capacity 크고 업데이트 약함)
- 이전 DROPOUT 차원을 α로 교체 — 3 epoch 학습에서 dropout 신호는 LR·rank 대비 작고 분산만 키운다는 ALLoRA(arXiv 2410.09692) 근거로 sweep 대상 제외.

**대안 검토:**
- `{16, 32, 64}` (초기 설계): γ=8 포함(r=8 × α=64). 2026-04-16 첫 sweep에서 이 조합이 **mode collapse 유발**(15h 학습 후 단일 클래스만 예측, eval accuracy ~5% = 1/19 랜덤 수준) → 64 제거.
- `{32}` 단일 고정: sweep 차원 -1로 2D 탐색. 20 run에 bayes가 과수렴 위험.
- `{8, 16, 32, 64, 128}` 5값: 3×5=15 grid → bayes가 run당 1.3개 할당, 신호 약화.
- α=r 복원: LR-rank 혼입 문제 그대로.

**확신도:** ★★★★ (γ=8 실측 실패 근거 추가되면서 상향)

**제거 사유 상세 (γ=8, 2026-04-16 사례):**
- `LORA_R=8, LORA_ALPHA=64, LEARNING_RATE=2.26e-4` 조합 실행
- γ=8이 vision tower LoRA gradient를 8× 증폭 → 유효 LR ≈ 1.8e-3 (full FT 상한의 ~30배)
- 초기 수십 step에서 vision feature가 garbage로 짓눌림 → 모델이 "이미지 신호 무시, class prior만 예측" 전략으로 도망 → mode collapse
- train/loss 2.5 → 0 수백 step 만에 붕괴, eval은 단일 pest class 99% 예측
- 실측값이 "γ=8은 실험적 영역" 한계 주석을 초과해 실패 쪽으로 안착 → 탐색 공간에서 완전 제외 결정
- 방어책으로 `max_grad_norm=1.0` 추가 (§6.6), `evaluate.py`에 mode collapse 탐지기 추가.

---

## 6. 고정 파라미터 근거

### 6.1 `WARMUP_STEPS = 150`  *(sweep에서 제외)*

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

### 6.2 `weight_decay = 0.01`  *(기본값 상향)*

**근거:**
- 기존 `0.001`은 Unsloth 권장 범위(`0.01~0.1`)의 **10배 낮음**
- `0.01`은 권장 범위 하한 — 안전하고 연구 부합
- LoRA 학습에서 weight_decay는 전체 파라미터 대비 영향 작다는 경험적 리포트 존재 → 공격적으로 높일 이유 없음

**확신도:** ★★★

---

### 6.3 `LORA_DROPOUT = 0.05`  *(sweep에서 제외, 고정)*

**제외 근거:**
- ALLoRA (arXiv 2410.09692): "Dropout as a stochastic regularizer has benefits that quickly vanish when fine-tuning ... can introduce detrimental additional variance, particularly when training only employs a minimal amount of training steps."
- 본 학습 ~2,900 step은 mid-length이지만, LR·rank 대비 신호 크기가 작아 3D sweep 예산의 1/3을 할당할 가치 약함.
- 이전 sweep 차원 `{0, 0.05, 0.1}`을 drop하고 그 자리를 더 영향력 큰 `LORA_ALPHA` 차원으로 교체.

**0.05 선택 근거:**
- Unsloth: "smaller datasets benefit from higher dropout (0.05~0.1)"
- 0은 정규화 레버 0 → overfitting 관찰 시 사후 조치 불가
- 0.1은 짧은 학습에서 분산 증가 위험
- 중간값 0.05는 "약한 정규화"로 보수적 선택

---

### 6.4 `NUM_EPOCHS = 3`

**근거:**
- Unsloth: "3 epoch 초과 시 overfitting 위험"
- 2 epoch으로 단축하면 run 시간 33% 감소하나, minority class(300건) 학습 기회 부족 우려
- 3 epoch은 **권장 상한** — 안전하되 공격적

**한계:** 불균형 4.6× 수준에서는 이 이슈가 초기 추정보다 온건. 다만 300건 × 3 epoch = 900 노출이라는 절대량 자체는 여전히 적은 편. Class-weighted loss나 balanced sampler가 근본 해결책이나 범위 밖.

---

### 6.5 기타 고정값

| 파라미터 | 값 | 근거 |
|---|---|---|
| `BATCH_SIZE` × `GRAD_ACCUM` | 6 × 2 = 12 | VRAM 제약, bf16 + gradient checkpointing 기준 |
| `optim` | `adamw_8bit` | VRAM 절감, Unsloth 기본 권장 |
| `lr_scheduler_type` | `cosine` | LoRA 학습의 표준 선택 |
| `bf16` | `True` | **Qwen3.5는 bf16 필수, 4-bit 양자화 비권장** |
| `max_seq_length` | 2048 | 이미지 토큰 + 짧은 텍스트에 여유 |

### 6.6 `max_grad_norm = 1.0`  *(안전장치, 2026-04-17 추가)*

**근거:**
- γ(=α/r) 높은 조합 + `finetune_vision_layers=True` 상태에서 초기 몇 step의 gradient가 폭주하면 모델이 수습 불가 지점까지 밀려나 mode collapse로 도망. 2026-04-16 sweep에서 γ=8 조합이 실제로 이 경로를 탐 (§5.3 참고).
- `max_grad_norm=1.0`은 Transformer 학습의 표준 보수값. 학습 신호는 유지하면서 폭주 구간만 차단.
- 이 sweep의 γ 상한이 4로 축소되어 폭주 확률은 이미 감소했지만, future HP 탐색 확장 또는 LR 상향 시를 대비한 이중 방어막.

**한계:**
- clip이 너무 강하면(예: 0.1) 학습 자체를 억제해 underfit. 1.0은 "정상 gradient는 거의 건드리지 않고 outlier spike만 제어"하는 실용 하한.

**확신도:** ★★★★ (실증 문제에 대한 직접 대응, 표준 보수값)

---

## 7. Sweep Method

### 7.1 `method: bayes`

**근거:**
- 3차원 연속/카테고리 혼합 공간에 효율적
- `random`보다 20 run 한정 예산에서 수렴 속도 우위
- `grid`는 3×3×3=27 조합 초과 시 비실용적
- **제약:** 순차 실행 필수 (동시 agent는 bayes GP를 random으로 퇴화시킴) → 단일 Pod 권장

### 7.2 `run_cap: 20`

**근거:**
- 3D bayes는 보통 10~15 run으로 수렴, 20은 **refinement 여유 포함**
- `LORA_R` 3 카테고리 × `LORA_ALPHA` 2 카테고리 = **6 grid point** → 각 조합 평균 3+ run 확보
- 시간: 20 × 12h ≈ **240h (10일)** — 실측 기준 (Qwen3.5-9B + A6000 + unsloth 2× 가속).

**대안:** 15 run (7.5일), 12 run (6일). 본 설정은 품질 우선.

**참고:** 초기 설계문에는 "6.5h/run × 20 = 130h"로 추정했으나, 실제 첫 run이 15h 이상 걸렸음(grad checkpointing·vision layer 학습 오버헤드 실측). 보수적으로 12h/run으로 재산정.

---

## 8. 한계 및 미검증 가정

1. **본 구성은 Qwen3.5-9B + 한국어 해충 분류에 대한 직접 벤치마크가 아님.** 일반 best practice + 태스크 특성의 교집합.
2. **클래스 불균형(4.6×)은 완만한 편.** 설계 초기엔 11×로 추정했으나 실측 4.6×. IJCV 2024 "Exploring VLMs for Imbalanced Learning"에 따르면 pretrained VLM은 mild imbalance에선 logit-adjustment가 focal/CB loss보다 효과 큼 → §9에 반영.
3. **WARMUP_STEPS 고정은 LR과의 상호작용 탐색 불가.** 최적 LR이 5e-4 근처이면 warmup 재조정 재실험 필요.
4. **α 범위 `{16, 32}` 가 최적 커버리지인지 미검증.** γ=8 극단(α=64)은 2026-04-16 실측에서 붕괴 확인되어 제거. γ=0.5 극단(α=16 with r=32)은 탐색에 포함됨. Winner가 이 하단 극단으로 기울면 α 범위를 {8, 16, 32}로 확장해서 γ={0.25, 0.5, 1, 2}까지 탐색할 여지.
5. **Sweep은 winner 1개만 산출.** 여러 seed에서의 분산은 측정 안 됨 — sweep 후 best 조합으로 seed 3~5개 재학습해야 성능 안정성 확인 가능.
6. **Composite metric `pest_gated_f1`은 곱셈 형태라 한 요소가 0이면 전체 0.** 실전에선 `binary_pest_recall`이 매우 낮은 run을 완전히 제외하는 효과 → 의도된 설계. 단, winner 간 차이가 미세할 때 단순 `macro_f1` 대비 노이즈가 더 클 수 있음.
7. **DoRA·LoRA+·rsLoRA 등 vanilla LoRA 대안 미탐색.** Unsloth 2026 default가 DoRA이지만 α 커플링 해제를 우선. §9 A/B로 별도 검증.
8. **NEFTune / label smoothing 미탐색.** 추가 비용 거의 없는 "free upside" 후보지만 sweep 차원에 넣지 않음 — winner 확정 후 단일 A/B.

---

## 9. 후속 실험 우선순위 (서비스 중심)

### 9.0 Sweep 직후 건전성 진단 (필수)

`upload_best.py` 실행 **전에** 반드시 수행:

- W&B에서 top-5 run의 **`binary_pest_recall`과 `normal_specificity`를 나란히** 확인. `pest_gated_f1` 1등 run이라도 이 두 값의 균형이 극단적이면 재고.
- `evaluation_results.json`의 `binary_pest_vs_normal` 섹션에서 **FN 절대 개수** (`true_pest_pred_normal_FN`) 확인. 예: val 1,276건 중 해충 샘플이 대부분인데 FN이 100건 넘으면 서비스 부적합.
- Confusion matrix에서 **"정상" column이 비정상적으로 진한 행**이 있는지 체크 → 특정 pest 클래스가 정상으로 새고 있는지.
- **Per-class F1 슬라이스 확인** (`eval/f1_per_class/<class>`): 전체 macro가 0.85여도 특정 해충 1\~2종에서 F1<0.5면 서비스 부적합. `evaluate.py`가 학습 직후 이 경고를 자동으로 Discord·콘솔에 띄우므로 놓칠 가능성 낮음.
- **Mode collapse 자동 경고** (신규 안전장치, 2026-04-17): evaluate.py가 예측 클래스 다양성 < 3이면 `@everyone` 경고. sweep 중에는 metric 0 근처로 기록되어 Bayes가 자동 기피.

### 9.0.1 Held-out test로 편향 없는 최종 수치

Sweep winner 확정 후 `upload_best.py` 전에:

```bash
EVAL_SPLIT=test python evaluate.py --model pest-lora-<winner-run-name>
```

Sweep이 val로 HP 선택하면 winner의 val 점수는 낙관적 편향 (val 자체가 간접 학습 신호). `split_val_test.py`로 미리 분리한 held-out test 319건으로 평가한 수치가 발표·보고용.

### 9.1 Winner 확정 후 A/B 우선순위

품질이 기대 이하 또는 추가 개선 여지 탐색 시:

1. **Post-hoc logit bias on "정상"** (최우선 · 재학습 불필요)
   - Menon et al. (ICLR 2021) "Long-tail learning via logit adjustment" 기반.
   - 추론 시 "정상" 클래스의 logit에서 상수 `b_normal`을 뺌 → "정상" 예측을 더 보수적으로 만듦.
   - val set에서 `b_normal` 스칼라 하나만 grid search (예: 0.0, 0.5, 1.0, 1.5, 2.0).
   - 서비스 실패 모드(해충 → 정상 FN)를 **직접적으로** 억제. 재학습 비용 0.
   - 단, Qwen VLM에서 개별 토큰 logit 접근 구현이 필요 (현재 evaluate.py는 generate만 사용 → 수정 필요).

2. **Row-asymmetric CE** (실패 모드가 여전하면)
   - 일반 `class_weights`가 아니라, (`true=pest` AND `pred=정상`) 셀에만 2~3× 가중치.
   - 서비스 비용 매트릭스를 loss에 직접 반영. 일반 class-weighted CE보다 더 정밀.

3. **DoRA A/B** (Unsloth 2026 default)
   - `use_dora=True` 1-run, winner 파라미터 그대로.
   - NVIDIA / Raschka 벤치마크에서 일관된 +0.5~1.5% 개선 보고.

4. **Seed 분산 측정** — winner 조합으로 seed 3~5개 재학습. 단일 run 운 요소 제거.

5. **NEFTune** (`neftune_noise_alpha=5`) — 추가 비용 0, TRL 네이티브 지원. Instruction tuning만큼 효과 크진 않지만 공짜 업사이드.

6. **Label smoothing 0.1** — 19-class classification 관용, 과신뢰 예측 완화.

7. **Minority 클래스 recall 타깃 대응** — per-class recall에서 특정 pest 클래스가 눈에 띄게 낮으면 해당 클래스의 bbox context 크롭 비율 상향 같은 데이터 측 조치.

8. **Prompt engineering 반복** — SYSTEM_MSG 미세 변형. 마지막 수단.

### 9.2 근본적 재설계 카드 (현 구성으로 서비스 요구 미달 시)

- **WeightedRandomSampler**: 4.6× mild imbalance + pretrained backbone 조합에선 ICLR 2025 "Upweighting Easy Samples"가 역효과 가능성 경고. 다른 레버 모두 소진 후 고려.
- **Focal / CB / LDAM / ASL loss**: 대개 20× 이상 severe imbalance용. 본 태스크에 대해 기대 효과 낮음.
- **2-stage 파이프라인**: Stage 1 "pest vs normal" binary, Stage 2 "which pest". 아키텍처 변경, 범위 밖.

---

## 10. 참고문헌

### LoRA / VLM 파인튜닝

- **Unsloth 공식 LoRA 가이드** — [unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- **Unsloth Qwen3.5 fine-tune** — [unsloth.ai/docs/models/qwen3.5/fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- **Thinking Machines Lab — "LoRA Without Regret"** (2025) — [thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/)
- **ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws** (arXiv 2410.09692) — [arxiv.org/html/2410.09692v1](https://arxiv.org/html/2410.09692v1) — dropout의 short-run 분산 이슈
- **Michael Brenndoerfer — LoRA Hyperparameters** — [mbrenndoerfer.com/writing/lora-hyperparameters-rank-alpha-target-modules](https://mbrenndoerfer.com/writing/lora-hyperparameters-rank-alpha-target-modules)
- **Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA** — [magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

### 서비스 metric / 불균형 / post-hoc 보정

- **Menon et al., "Long-tail learning via logit adjustment"** (ICLR 2021) — [arxiv.org/abs/2007.07314](https://arxiv.org/abs/2007.07314) — post-hoc logit bias 근거
- **Wang et al., "Exploring VLMs for Imbalanced Learning"** (IJCV 2024) — [link.springer.com/article/10.1007/s11263-023-01868-w](https://link.springer.com/article/10.1007/s11263-023-01868-w)
- **"Upweighting Easy Samples"** (ICLR 2025) — [arxiv.org/html/2502.02797v2](https://arxiv.org/html/2502.02797v2) — fine-tuning 중 과도한 reweighting 경고
- **F2 score in fraud/defect detection** — [Galileo F1 guide](https://galileo.ai/blog/f1-score-ai-evaluation-precision-recall), [fraud-detection-handbook Ch.4](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/Introduction.html)
- **NEFTune** — [arxiv.org/abs/2310.05914](https://arxiv.org/abs/2310.05914)
