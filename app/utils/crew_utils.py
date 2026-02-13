import json

from pydantic import BaseModel


def parse_crew_output(result, schema: type[BaseModel]):
    data = None
    raw = ""
    try:
        # 1) json_dict 우선 시도
        if hasattr(result, "json_dict") and result.json_dict:
            data = result.json_dict
        else:
            # 2) 문자열에서 JSON 추출
            raw = str(result)
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            data = json.loads(raw)  # ← 이 줄이 빠져있었음

        # 3) "scenario" 래퍼 unwrap
        if isinstance(data, dict) and "scenario" in data:
            data = data["scenario"]

        # 4) dict로 검증 (model_validate_json이 아닌 model_validate)
        return schema.model_validate(data)

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n원본: {raw[:300]}")
    except Exception as e:
        raise ValueError(
            f"CrewAI 출력을 {schema.__name__}으로 파싱할 수 없습니다.\n"
            f"에러: {e}\n"
            f"데이터: {json.dumps(data, ensure_ascii=False)[:300] if data else raw[:300]}"
        )


def parse_crew_output_from_task(task_output, schema: type[BaseModel]):
    """
    개별 Task의 출력(TaskOutput)을 Pydantic 스키마로 파싱한다.
    Crew 전체 결과가 아닌 tasks_output[i]에서 특정 Task 결과를 추출할 때 사용.
    """
    raw = ""
    try:
        # 1순위: 이미 파싱된 pydantic 객체
        if hasattr(task_output, "pydantic") and task_output.pydantic:
            if isinstance(task_output.pydantic, schema):
                return task_output.pydantic
            return schema.model_validate(task_output.pydantic.model_dump())
        
        if hasattr(task_output, "json_dict") and task_output.json_dict:
            return schema.model_validate(task_output.json_dict)

        raw = task_output.raw if hasattr(task_output, "raw") else str(task_output)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        return schema.model_validate_json(raw)
    except Exception:
        raise ValueError(
            f"Task 출력을 {schema.__name__}으로 파싱할 수 없습니다: {raw[:200]}"
        )


def format_conversation_log(log: list[dict]) -> str:
    """대화 기록을 Task description에 주입할 수 있는 문자열로 포맷팅."""
    MAX_TURNS = 10
    recent = log[-MAX_TURNS:] if len(log) > MAX_TURNS else log

    lines = []
    for turn in recent:
        q = turn.get("question", "")
        a = turn.get("answer", "")
        score = turn.get("score", "N/A")
        lines.append(f"Q: {q}\nA: {a}\n[Score: {score}]")

    return "\n---\n".join(lines)


def format_transcript(transcript: list[dict]) -> str:
    """Phase 3용 전체 transcript 포맷팅."""
    lines = []
    for i, turn in enumerate(transcript, 1):
        lines.append(f"== Turn {i} ==")
        lines.append(f"질문 [{turn.get('type', 'MAIN')}]: {turn.get('question', '')}")
        lines.append(f"답변: {turn.get('answer', '')}")
        if "feedback" in turn:
            lines.append(f"피드백: {turn['feedback']}")
        lines.append("")

    return "\n".join(lines)


def format_evaluations(all_evaluations: list[dict]) -> str:
    """Phase 3용 전체 평가 데이터 포맷팅."""
    return json.dumps(all_evaluations, ensure_ascii=False, indent=2)


def extract_keywords(jd_text: str) -> str:
    """JD에서 기술 키워드를 추출 (간단 버전 — JD 텍스트를 그대로 반환)."""
    return jd_text
