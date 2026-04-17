"""
val.jsonl을 stratified split해서 val/test로 분리.

동기:
- sweep이 val 메트릭으로 HP를 선택 → winner의 val 점수는 낙관적 편향.
- 편향 없는 최종 수치를 얻으려면 sweep에 쓰이지 않은 held-out test set 필요.

동작:
1. data/val.jsonl → data/val.jsonl (~80%) + data/test.jsonl (~20%), 클래스 비율 유지
2. 원본은 data/val_original.jsonl로 백업 (idempotent — 재실행 시 이 백업을 소스로 사용)
3. data/val_cropped.jsonl, data/val_cropped/ 캐시 자동 삭제
   → train.py 재실행 시 새 val.jsonl 기준으로 cropped 캐시 재생성

이미지 파일(data/val/<class>/*.jpg) 자체는 이동하지 않음. test.jsonl 레코드의
`image` 필드는 여전히 "val/<class>/*.jpg"를 가리키며 파일은 그 자리에 존재.
evaluate.py는 DATA_DIR/<image_path>로 읽으므로 정상 작동.

2026-04-17: train.py의 [1/9]가 test.jsonl 없을 때 자동 호출하도록 변경 (`AUTO_TEST_SPLIT=1` 기본).
programmatic API로 `split_val_test()` 함수 export — 직접 실행(CLI)과 import 둘 다 지원.

사용법 (RunPod에서):
    # 수동 실행 (CLI)
    python split_val_test.py                # 기본 test_ratio=0.2
    python split_val_test.py --test-ratio 0.15

    # 또는 train.py가 자동 호출 (첫 sweep run의 [1/9])

sweep 후 winner를 test로 최종 평가:
    EVAL_SPLIT=test python evaluate.py --model pest-lora-<winner-run-name>
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter

from sklearn.model_selection import train_test_split


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_label(record):
    """record에서 assistant 응답(클래스명) 추출. train.py/evaluate.py와 동일 경로."""
    return record["messages"][-1]["content"][0]["text"]


def split_val_test(data_dir="data", test_ratio=0.2, seed=42, verbose=True):
    """val.jsonl을 stratified split해서 val/test 생성.

    Programmatic API — train.py의 [1/9]가 test.jsonl 미존재 시 이 함수 호출.

    Args:
        data_dir: jsonl 파일들이 있는 디렉토리 (기본 "data")
        test_ratio: test 비율 (기본 0.2)
        seed: 분할 seed (기본 42)
        verbose: True면 클래스별 분포·진행상황 전체 출력, False면 조용히 수행

    Returns:
        dict: {"val_count": int, "test_count": int, "class_count": int}

    Raises:
        FileNotFoundError: val.jsonl과 val_original.jsonl 둘 다 없을 때
        RuntimeError: 1건만 있는 클래스가 있어 stratified 불가할 때
    """
    val_path = os.path.join(data_dir, "val.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")
    backup_path = os.path.join(data_dir, "val_original.jsonl")

    # 재실행 처리: val_original.jsonl이 이미 있으면 그걸 소스로 (이미 한 번 분할된 상태)
    if os.path.exists(backup_path):
        if verbose:
            print(f"[재실행] 기존 백업 {backup_path} 발견 → 이를 소스로 사용")
        source_path = backup_path
    else:
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"{val_path} 없음. data 디렉토리 위치 확인하세요.")
        shutil.copy(val_path, backup_path)
        if verbose:
            print(f"[백업] {val_path} → {backup_path}")
        source_path = backup_path

    records = load_jsonl(source_path)
    labels = [extract_label(r) for r in records]
    class_counts = Counter(labels)

    if verbose:
        print(f"\n소스: {source_path}")
        print(f"  총 {len(records)}건, {len(class_counts)}클래스")
        print(f"  클래스별 분포:")
        for cls in sorted(class_counts.keys()):
            print(f"    {cls:20s}: {class_counts[cls]:4d}")

    # 희소 클래스 사전 체크 (stratify는 각 클래스 최소 2건 필요)
    singletons = [cls for cls, c in class_counts.items() if c < 2]
    if singletons:
        raise RuntimeError(
            f"1건만 있는 클래스: {singletons} → stratified split 불가. "
            "해당 클래스 샘플 추가하거나 stratify 해제 필요."
        )

    val_records, test_records = train_test_split(
        records,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    save_jsonl(val_records, val_path)
    save_jsonl(test_records, test_path)

    val_labels = Counter(extract_label(r) for r in val_records)
    test_labels = Counter(extract_label(r) for r in test_records)

    if verbose:
        print(f"\n[분할 결과] seed={seed}, test_ratio={test_ratio}")
        print(f"  {val_path}:  {len(val_records):4d}건")
        print(f"  {test_path}: {len(test_records):4d}건")
        print(f"\n  클래스별 val / test / 총:")
        for cls in sorted(class_counts.keys()):
            v = val_labels.get(cls, 0)
            t = test_labels.get(cls, 0)
            print(f"    {cls:20s}: {v:4d} / {t:4d} / {v+t:4d}")

    # val_cropped 캐시 무효화 — 새 val.jsonl로 재생성되도록
    cropped_jsonl = os.path.join(data_dir, "val_cropped.jsonl")
    cropped_dir = os.path.join(data_dir, "val_cropped")
    removed = []
    if os.path.exists(cropped_jsonl):
        os.remove(cropped_jsonl)
        removed.append(cropped_jsonl)
    if os.path.exists(cropped_dir):
        shutil.rmtree(cropped_dir)
        removed.append(f"{cropped_dir}/")
    if removed and verbose:
        print(f"\n[캐시 무효화]")
        for r in removed:
            print(f"  삭제: {r}")
        print(f"  → 다음 train.py 실행 시 새 val.jsonl 기준으로 val_cropped 재생성됨.")

    if verbose:
        print(f"\n완료. 다음 sweep은 val {len(val_records)}건으로 HP 선택,")
        print(f"winner 확정 후 아래로 test {len(test_records)}건 최종 평가:")
        print(f"  EVAL_SPLIT=test python evaluate.py --model pest-lora-<winner-run-name>")

    return {
        "val_count": len(val_records),
        "test_count": len(test_records),
        "class_count": len(class_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="val.jsonl → val/test stratified split")
    parser.add_argument("--data-dir", default="data", help="데이터 디렉토리 (기본 data)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="test 비율 (기본 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="분할 seed (기본 42)")
    args = parser.parse_args()

    try:
        split_val_test(
            data_dir=args.data_dir,
            test_ratio=args.test_ratio,
            seed=args.seed,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"[에러] {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[에러] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
