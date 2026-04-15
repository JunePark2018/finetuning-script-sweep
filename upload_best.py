"""
Sweep 종료 후 가장 성능 좋은 LoRA 모델만 HuggingFace Hub에 업로드.

로컬의 모든 `pest-lora-*/` 디렉토리를 스캔하여
`evaluation_results.json`의 `f1_macro`가 가장 높은 모델을 선정·업로드한다.
재학습 불필요 — sweep의 각 run이 이미 3 epoch full 학습을 마친 상태.

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
    """`pest-lora-*/evaluation_results.json`을 모두 읽어 (f1_macro, dir, results) 리스트 반환."""
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
        f1 = results.get("f1_macro")
        if f1 is None:
            print(f"  ⚠️  {d}: f1_macro 필드 없음 — 건너뜀")
            continue
        candidates.append((float(f1), d, results))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def print_ranking(candidates, top_k):
    print(f"\n=== 전체 {len(candidates)}개 중 상위 {min(top_k, len(candidates))}개 ===")
    for i, (f1, d, r) in enumerate(candidates[:top_k]):
        acc = r.get("accuracy", "?")
        n = r.get("num_samples", "?")
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
        print(f"  [{i+1:2d}] f1_macro={f1:.4f}  accuracy={acc_str}  (n={n})  {d}")


def upload(best_dir, best_f1):
    from huggingface_hub import HfApi, create_repo

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN 환경변수 없음")
        sys.exit(1)
    hf_org = os.environ.get("HF_ORG", "Himedia-AI-01")

    run_name = best_dir[len("pest-lora-"):]
    hub_repo = f"{hf_org}/pest-{run_name}" if hf_org else f"pest-{run_name}"
    print(f"\n업로드 대상: {hub_repo}")
    print(f"로컬 경로: {best_dir}/")

    repo_url = create_repo(hub_repo, token=hf_token, exist_ok=True, private=False)
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=best_dir,
        repo_id=repo_url.repo_id,
        commit_message=f"Upload sweep winner (f1_macro={best_f1:.4f})",
    )
    hub_url = f"https://huggingface.co/{repo_url.repo_id}"
    print(f"✅ 업로드 완료: {hub_url}")


def main():
    parser = argparse.ArgumentParser(description="Sweep 최고 성능 모델을 HF Hub에 업로드")
    parser.add_argument("--dry-run", action="store_true", help="업로드 없이 랭킹만 확인")
    parser.add_argument("--top-k", type=int, default=5, help="출력할 상위 N개 (기본 5)")
    args = parser.parse_args()

    candidates = scan_candidates()
    if not candidates:
        print("❌ 후보 없음. 현재 디렉토리에서 pest-lora-*/evaluation_results.json을 찾지 못함.")
        print("   Sweep을 끝낸 Pod의 작업 디렉토리에서 실행해야 함.")
        sys.exit(1)

    print_ranking(candidates, args.top_k)

    best_f1, best_dir, _ = candidates[0]
    print(f"\n★ Best: {best_dir} (f1_macro={best_f1:.4f})")

    if args.dry_run:
        print("--dry-run — 업로드 건너뜀")
        return

    upload(best_dir, best_f1)


if __name__ == "__main__":
    main()
