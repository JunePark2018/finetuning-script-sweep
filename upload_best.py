"""
Sweep 종료 후 가장 성능 좋은 LoRA 모델만 HuggingFace Hub에 업로드.

로컬의 모든 `pest-lora-*/` 디렉토리를 스캔하여
`evaluation_results.json`의 `pest_gated_f1`(sweep metric)이 가장 높은 모델을 선정·업로드한다.
재학습 불필요 — sweep의 각 run이 이미 3 epoch full 학습을 마친 상태.

⚠️ 단순 1등 선택이 아닌 서비스 건전성 진단을 병행할 것.
   `--dry-run --top-k 10`으로 상위 랭킹 + binary_pest_recall·normal_specificity 함께 확인 후,
   필요 시 `--pick <N>`으로 N위 모델을 업로드. (HYPERPARAMETERS.md §9.0 참고)

사용법:
    python upload_best.py              # 최고 성능 모델 업로드
    python upload_best.py --dry-run    # 업로드 없이 랭킹만 확인
    python upload_best.py --top-k 10   # 상위 10개 출력

환경변수:
    HF_TOKEN  (필수)
    HF_ORG    (선택, 기본 Himedia-AI-01. 빈 문자열이면 토큰 소유자 개인 계정)
"""

import argparse
import glob
import json
import os
import sys


def scan_candidates():
    """`pest-lora-*/evaluation_results.json`을 모두 읽어 (pest_gated_f1, dir, results) 리스트 반환."""
    candidates = []
    for d in sorted(glob.glob("pest-lora-*")):
        eval_path = os.path.join(d, "evaluation_results.json")
        if not os.path.exists(eval_path):
            continue
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  {d}: evaluation_results.json 읽기 실패 ({e}) — 건너뜀")
            continue
        # sweep metric으로 랭킹. 구버전 evaluation_results.json(f1_macro만)은 fallback.
        score = results.get("pest_gated_f1")
        if score is None:
            score = results.get("f1_macro")
            if score is None:
                print(f"  ⚠️  {d}: pest_gated_f1/f1_macro 둘 다 없음 — 건너뜀")
                continue
            print(f"  ℹ️  {d}: pest_gated_f1 없음, f1_macro로 fallback (구버전 결과)")
        candidates.append((float(score), d, results))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def print_ranking(candidates, top_k):
    print(f"\n=== 전체 {len(candidates)}개 중 상위 {min(top_k, len(candidates))}개 (sorted by pest_gated_f1) ===")
    print(f"  {'#':>3} {'gated_f1':>8} {'pest_rec':>8} {'n_spec':>7} {'FN':>4} {'FP':>4} {'macroF1':>8}  dir")
    for i, (score, d, r) in enumerate(candidates[:top_k]):
        bin_ = r.get("binary_pest_vs_normal", {})
        pest_recall = bin_.get("recall", "—")
        normal_spec = bin_.get("normal_specificity", "—")
        fn = bin_.get("true_pest_pred_normal_FN", "—")
        fp = bin_.get("true_normal_pred_pest_FP", "—")
        macro_f1 = r.get("f1_macro", "—")
        def fmt(v, w, p=4):
            return f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{str(v):>{w}}"
        print(f"  [{i+1:2d}] {fmt(score, 8)} {fmt(pest_recall, 8)} {fmt(normal_spec, 7)} {fmt(fn, 4, 0) if isinstance(fn,int) else f'{str(fn):>4}'} {fmt(fp, 4, 0) if isinstance(fp,int) else f'{str(fp):>4}'} {fmt(macro_f1, 8)}  {d}")


def upload(chosen_dir, chosen_score):
    from huggingface_hub import HfApi, create_repo

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN 환경변수 없음")
        sys.exit(1)
    hf_org = os.environ.get("HF_ORG", "Himedia-AI-01")

    run_name = chosen_dir[len("pest-lora-"):]
    hub_repo = f"{hf_org}/pest-{run_name}" if hf_org else f"pest-{run_name}"
    print(f"\n업로드 대상: {hub_repo}")
    print(f"로컬 경로: {chosen_dir}/")

    repo_url = create_repo(hub_repo, token=hf_token, exist_ok=True, private=False)
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=chosen_dir,
        repo_id=repo_url.repo_id,
        commit_message=f"Upload sweep winner (pest_gated_f1={chosen_score:.4f})",
    )
    hub_url = f"https://huggingface.co/{repo_url.repo_id}"
    print(f"✅ 업로드 완료: {hub_url}")


def main():
    parser = argparse.ArgumentParser(description="Sweep 최고 성능 모델을 HF Hub에 업로드")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 랭킹만 확인")
    parser.add_argument("--top-k", type=int, default=5, help="출력할 상위 N개 (기본 5)")
    parser.add_argument("--pick", type=int, default=1, help="랭킹 N위 모델 업로드 (기본 1위). 서비스 건전성 진단 후 FN 높은 1위 대신 2~3위 선택 시 사용.")
    args = parser.parse_args()

    candidates = scan_candidates()
    if not candidates:
        print("❌ 후보 없음. 현재 디렉토리에서 pest-lora-*/evaluation_results.json을 찾지 못함.")
        print("   Sweep을 끝낸 Pod의 작업 디렉토리에서 실행해야 함.")
        sys.exit(1)

    print_ranking(candidates, args.top_k)

    if args.pick < 1 or args.pick > len(candidates):
        print(f"\n❌ --pick {args.pick} 범위 밖 (1 ~ {len(candidates)})")
        sys.exit(1)

    chosen_score, chosen_dir, _ = candidates[args.pick - 1]
    marker = "★" if args.pick == 1 else f"#{args.pick}"
    print(f"\n{marker} Selected: {chosen_dir} (pest_gated_f1={chosen_score:.4f})")

    if args.dry_run:
        print("--dry-run — 업로드 건너뜀")
        return

    upload(chosen_dir, chosen_score)


if __name__ == "__main__":
    main()
