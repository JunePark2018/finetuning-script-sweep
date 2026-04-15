#!/bin/bash
# RunPod / Vast.ai 환경 초기 설정 스크립트
# 사용법: bash setup.sh

set -e

echo "=== 패키지 설치 ==="
pip install --upgrade pip
pip install --upgrade typing_extensions
pip install unsloth
pip install "transformers>=5.2"
pip install trl==0.22.2 datasets Pillow accelerate scikit-learn huggingface_hub wandb requests

echo ""
echo "=== Unsloth 설치 확인 ==="
python -c "from unsloth import FastVisionModel; print('Unsloth OK')"

echo ""
echo "=== W&B 로그인 (선택) ==="
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY 감지 — W&B 활성화"
else
    echo "WANDB_API_KEY 미설정 — W&B 없이 진행합니다."
    echo "사용하려면: export WANDB_API_KEY=your_key"
fi

echo ""
echo "=== 데이터셋 확인 ==="
DATA_DIR="${DATA_DIR:-data}"
if [ -f "$DATA_DIR/train.jsonl" ]; then
    echo "기존 데이터셋 발견: $DATA_DIR/ (다운로드 건너뜀)"
else
    echo "$DATA_DIR/train.jsonl 없음 — train.py 실행 시 HF Hub에서 자동 다운로드됩니다."
    if [ -z "$HF_TOKEN" ]; then
        echo "⚠️  HF_TOKEN이 설정돼 있지 않습니다. 데이터셋이 gated라면 다운로드에 실패합니다."
        echo "   export HF_TOKEN=hf_xxx 로 설정하거나, RunPod Env에 등록하세요."
    else
        echo "HF_TOKEN 감지 — 다운로드 준비 완료"
    fi
fi

echo ""
echo "=== 설정 완료 ==="
echo "학습 시작: python train.py"
