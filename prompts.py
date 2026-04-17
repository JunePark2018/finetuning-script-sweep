"""
학습·평가·추론 세 곳에서 바이트 단위로 일치해야 하는 프롬프트·메시지 정의.

train.py / evaluate.py / inference.py가 모두 이 모듈을 import하므로
여기만 수정하면 세 파일 자동 동기화. 이전엔 리터럴로 세 파일에 복사돼 있어
휴먼 에러 위험 있었음.

⚠️ 변경 금지 조건:
- USER_PROMPT는 전처리된 train_cropped.jsonl/val_cropped.jsonl에 이미 인코딩됨.
  바꾸면 학습 분포와 추론 분포가 달라져 성능 급락. 기존 학습 데이터 재사용하려면
  이 문자열 건드리지 말 것.
- build_system_msg의 포맷도 동일 이유로 고정.
"""

# 데이터셋에 인코딩된 사용자 질문 — 학습 때와 추론 때 동일해야 모델이 같은 분포로 작동.
USER_PROMPT = "이 사진에 있는 해충의 이름을 알려주세요."


def build_system_msg(class_names):
    """클래스 목록으로 동적 SYSTEM_MSG 생성.
    class_names는 train.py가 data/train/<class>/ 폴더명으로 추출하고
    class_names.json으로 저장 → evaluate.py/inference.py가 그 파일을 읽어 재구성."""
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
