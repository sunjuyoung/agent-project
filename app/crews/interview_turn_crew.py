import json

from crewai import Task, Crew, Process

from app.agents.evaluator import create_evaluator
from app.agents.interviewer import create_interviewer
from app.schemas.interview import InterviewDecisionSchema, EvaluationResultSchema


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Crew: 매 턴 답변 평가 + 다음 질문 결정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_interview_turn_crew(
    user_id: str,
    current_question: dict,
    user_answer: str,
    follow_up_count: int,
    remaining_count: int,
    conversation_log: str,
    evaluation_criteria: list,
    scenario: dict,
) -> Crew:
    """Phase 2: 매 턴마다 새로 생성되는 답변 평가 + 다음 질문 결정 Crew."""

    evaluator = create_evaluator(user_id=user_id)
    interviewer = create_interviewer()

    # ── 평가 Task ────────────────────────────────────────────
    eval_task = Task(
        description=(
            "후보자의 면접 답변을 종합적으로 평가합니다.\n\n"
            "**평가 대상:**\n"
            f"- 질문: {current_question.get('text', '')}\n"
            f"- 평가 대상 기술: {current_question.get('skillTarget', current_question.get('skill_target', ''))}\n"
            f"- 난이도: {current_question.get('difficulty', '')}\n"
            f"- 답변: {user_answer}\n\n"
            f"**평가 기준:**\n{json.dumps(evaluation_criteria, ensure_ascii=False)}\n\n"
            f"**이전 대화 기록:**\n{conversation_log}\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **답변 분석**:\n"
            "   - 답변에서 언급된 기술 개념, 키워드, 경험을 추출합니다.\n"
            "   - 평가 기준의 각 항목에 대해 답변이 충족하는지 판단합니다.\n"
            "   - 기술적으로 잘못된 내용(오답)이 있는지 확인합니다.\n"
            "   - '모르겠습니다' 또는 빈 답변인 경우, 이를 감점하되 존중합니다.\n\n"
            "2. **RAG 검색을 통한 학습 노트 비교**:\n"
            "   - 반드시 RAGSearchTool을 호출하여 해당 기술 키워드로 사용자의 노트를 검색합니다.\n"
            "   - **절대로 검색 없이 노트 내용을 추측하거나 생성하지 마세요.**\n"
            "   - 검색 결과를 바탕으로 다음을 구분합니다:\n"
            "     • studied_but_missed: 노트에 학습 내용이 있으나 답변에서 누락된 부분\n"
            "     • not_studied: 노트에도 없어서 아직 학습하지 않은 부분\n"
            "   - 검색 결과가 없으면 해당 주제는 전부 'not_studied'로 분류합니다.\n\n"
            "3. **채점 (1-10점 스케일)**:\n"
            "   - 1-3점: 답변 불가 또는 핵심 개념 오류\n"
            "   - 4-5점: 기본 개념은 이해하나 깊이 부족\n"
            "   - 6-7점: 적절한 수준의 이해와 설명\n"
            "   - 8-9점: 실무 경험이 녹아든 깊이 있는 답변\n"
            "   - 10점: 면접관도 배울 수 있는 전문가 수준 답변\n\n"
            "4. **피드백 및 개선 팁 작성**:\n"
            "   - feedback: 잘한 부분을 먼저 언급하고, 보완할 점을 구체적으로 제시합니다.\n"
            "   - improvement_tip: 이 주제에 대해 학습할 수 있는 구체적인 방향을 제안합니다.\n"
            "   - 피드백은 비판이 아닌 성장 관점으로 작성합니다.\n\n"
            "**주의사항:**\n"
            "- 이전 대화 기록을 참고하여 동일 질문에 대한 보충 답변이 있었는지 확인합니다.\n"
            "- 꼬리질문에 대한 답변인 경우, 메인 질문 답변과 종합하여 평가합니다.\n"
            "- 답변의 길이가 짧다고 무조건 감점하지 않습니다. 핵심을 간결하게 전달했다면 높게 평가합니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "score": 7,
    "hits": [
        "답변에서 정확히 언급한 포인트 1",
        "답변에서 정확히 언급한 포인트 2"
    ],
    "misses": [
        "답변에서 누락되었거나 부정확한 포인트 1",
        "답변에서 누락되었거나 부정확한 포인트 2"
    ],
    "note_comparison": {
        "studied_but_missed": [
            "학습 노트에는 있으나 답변에서 언급하지 않은 내용"
        ],
        "not_studied": [
            "학습 노트에도 없는 내용 (아직 학습하지 않은 영역)"
        ]
    },
    "feedback": "잘한 부분과 보완할 점을 포함한 종합 피드백 (한국어)",
    "improvement_tip": "이 주제에 대해 추가 학습할 수 있는 구체적 방향 제안 (한국어)"
}
""",
        agent=evaluator,
    )

    # ── 면접 진행 Task ───────────────────────────────────────
    # ── 면접 진행 Task ───────────────────────────────────────
    # ★ 변경: 사전 꼬리질문 텍스트 참조 → follow_up_guide 기반 동적 생성
    interview_task = Task(
        description=(
            "평가 결과를 기반으로 면접의 다음 액션을 결정하고, "
            "자연스러운 면접관 멘트를 생성합니다.\n\n"
            f"**현재 면접 진행 상황:**\n"
            f"- 현재 질문 ID: {current_question.get('id', '')}\n"
            f"- 현재 꼬리질문 횟수: {follow_up_count} (이 질문에 대해 이미 진행한 꼬리질문 수)\n"
            f"- 남은 메인 질문 수: {remaining_count}\n\n"
            f"**질문 시나리오 (전체):**\n{json.dumps(scenario, ensure_ascii=False)}\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **액션 결정 (decision)**:\n"
            "   다음 3가지 중 하나를 선택합니다:\n\n"
            "   • FOLLOW_UP (꼬리질문 진행):\n"
            "     - 조건: 꼬리질문 횟수가 1회 미만이고, 답변을 더 깊이 확인할 필요가 있는 경우\n"
            "     - 답변이 모호하거나 핵심을 빗나간 경우 추가 질문으로 기회를 줍니다\n\n"
            "   • NEXT_QUESTION (다음 메인 질문으로 이동):\n"
            "     - 조건: 꼬리질문을 이미 1회 진행했거나, 답변이 충분히 평가 가능한 경우\n"
            "     - 남은 질문이 있는 경우에만 선택 가능\n"
            "     - 시나리오의 questions 리스트에서 다음 순서의 질문을 선택합니다\n\n"
            "   • END (면접 종료):\n"
            "     - 조건: 남은 메인 질문이 0개인 경우\n"
            "     - 면접 마무리 멘트를 자연스럽게 작성합니다\n\n"
            "2. **면접관 멘트 작성 (message)**:\n"
            "   - 후보자의 답변 내용을 구체적으로 참조하여 자연스럽게 연결합니다.\n"
            "   - 예: '말씀하신 WebSocket 활용 경험이 인상적입니다. 그렇다면...'\n"
            "   - 기계적인 진행 멘트 금지: '다음 질문입니다', '잘 답변하셨습니다' 등 지양\n"
            "   - '모르겠습니다' 답변 → '괜찮습니다. 다른 질문으로 넘어가겠습니다' 등 존중하는 톤\n"
            "   - 평가 점수, 평가 기준, 채점 결과를 절대 노출하지 않습니다.\n\n"
            "3. **다음 질문 설정 (nextQuestion)**:\n\n"
            "   - **FOLLOW_UP일 때 — 꼬리질문 동적 생성 (★ 핵심):**\n"
            "     시나리오에는 완성된 꼬리질문 텍스트가 없습니다.\n"
            "     follow_up_guide의 probe_direction(탐색 방향)과 purpose(목적)만 제공됩니다.\n"
            "     다음 과정으로 꼬리질문을 직접 생성하세요:\n"
            "     (1) 후보자의 실제 답변에서 언급한 키워드/기술/경험을 파악\n"
            "     (2) probe_direction에서 답변과 관련된 방향을 선택\n"
            "     (3) 후보자가 언급한 맥락 + 탐색 방향을 결합하여 자연스러운 질문 생성\n"
            "     예시:\n"
            "       probe_direction: '캐시 무효화 전략, TTL 설정 기준'\n"
            "       후보자 답변: 'Ehcache로 로컬 캐시를 구현했습니다'\n"
            "       → 생성: 'Ehcache에서 캐시 무효화는 어떤 방식으로 처리하셨나요?'\n\n"
            "   - **NEXT_QUESTION일 때:**\n"
            "     시나리오 questions 리스트에서 다음 질문을 그대로 사용합니다.\n\n"
            "   - **END일 때:**\n"
            "     nextQuestion은 null로 설정합니다.\n\n"
            "**주의사항:**\n"
            "- 평가 결과(점수, hits, misses)는 내부 참고용이며, 면접관 멘트에 노출하지 않습니다.\n"
            "- 이전 대화 기록의 톤과 흐름을 유지합니다.\n"
            "- 질문 순서는 시나리오에 정의된 순서를 따릅니다. 임의로 순서를 변경하지 않습니다.\n"
            "- 꼬리질문은 메인 질문의 평가 영역(follow_up_guide.purpose)을 벗어나지 않습니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "decision": "FOLLOW_UP" | "NEXT_QUESTION" | "END",
    "message": "면접관의 자연스러운 멘트 (한국어, 후보자 답변 참조)",
    "nextQuestion": {
        "id": "질문 ID (예: q1-f1 또는 q2)",
        "text": "다음 질문 텍스트 (한국어) — FOLLOW_UP일 때는 답변 맥락 기반으로 직접 생성",
        "skillTarget": "평가 대상 기술",
        "difficulty": "EASY/MEDIUM/HARD",
        "evaluationCriteria": ["평가 기준"],
        "followUpGuide": null
    }
}
// decision이 END인 경우 nextQuestion은 null
""",
        agent=interviewer,
        context=[eval_task],
    )

    return Crew(
        agents=[evaluator, interviewer],
        tasks=[eval_task, interview_task],
        process=Process.sequential,
        verbose=True,
    )
