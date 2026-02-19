import json

from crewai import Task, Crew, Process

from app.agents.quiz_master import create_quiz_generator, create_quiz_evaluator
from app.schemas.quiz import QuizGenerateSchema, QuizEvaluateSchema


def create_quiz_generate_crew(
    user_id: str,
    tags: list[str],
    difficulty: str,
    count: int,
) -> Crew:
    """퀴즈 문제 생성 Crew."""
    generator = create_quiz_generator(user_id=user_id)

    generate_task = Task(
        description=(
            "사용자의 학습 노트를 RAG 검색하여, 지정된 태그와 난이도에 맞는 "
            f"4지선다 퀴즈 문제를 {count}개 생성합니다.\n\n"

            "**입력 정보:**\n"
            f"- 사용자 ID: {user_id}\n"
            f"- 태그: {json.dumps(tags, ensure_ascii=False)}\n"
            f"- 난이도: {difficulty}\n"
            f"- 문제 수: {count}\n\n"

            "**필수 작업 순서 (반드시 따라야 함):**\n\n"

            "1. **RAG 검색으로 학습 노트 수집**:\n"
            "   - 각 태그를 키워드로 RAGSearchTool을 호출합니다.\n"
            "   - 검색 결과에서 핵심 개념, 정의, 비교 포인트, 실무 사례를 추출합니다.\n"
            "   - 검색 결과가 부족한 태그는 해당 태그의 기본 개념 위주로 출제합니다.\n\n"

            "2. **난이도별 문제 설계 기준**:\n"
            "   - EASY: 개념 정의, 용어 의미, 기본 동작 원리를 묻는 문제\n"
            "     예: 'Spring Bean의 기본 스코프는 무엇인가?'\n"
            "   - INTERMEDIATE: 개념 간 비교, 동작 차이, 적용 조건을 묻는 문제\n"
            "     예: '@Component와 @Bean의 차이점으로 올바른 것은?'\n"
            "   - HARD: 실무 시나리오, 트러블슈팅, 최적화 판단을 묻는 문제\n"
            "     예: 'N+1 문제가 발생하는 상황에서 가장 적절한 해결 방법은?'\n\n"

            "3. **선택지 설계 원칙**:\n"
            "   - 4개 선택지 모두 그럴듯해야 합니다.\n"
            "   - 오답은 흔히 혼동되는 개념이나 부분적으로 맞는 내용으로 구성합니다.\n"
            "   - '위의 모든 것', '해당 없음' 같은 모호한 선택지는 사용하지 않습니다.\n"
            "   - 선택지 길이와 형식을 균일하게 맞춥니다.\n\n"

            "4. **해설 작성**:\n"
            "   - 정답이 맞는 이유를 명확히 설명합니다.\n"
            "   - 각 오답이 틀린 이유를 간략히 언급합니다.\n"
            "   - 가능하면 사용자의 노트 내용을 참조하여 복습 포인트를 제시합니다.\n\n"

            "5. **출제 다양성**:\n"
            "   - 태그가 여러 개인 경우 균등하게 분배합니다.\n"
            "   - 동일 개념을 다른 각도로 묻는 중복 문제를 피합니다.\n"
            "   - 각 문제에 해당하는 tag와 difficulty를 반드시 명시합니다.\n\n"

            "**주의사항:**\n"
            "- 반드시 RAGSearchTool을 먼저 호출하여 노트를 검색한 후 출제합니다.\n"
            "- 노트에 없는 고급 개념으로 문제를 만들지 않습니다.\n"
            "- 정답은 반드시 options 배열에 포함된 텍스트와 정확히 일치해야 합니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "questions": [
        {
            "question": "문제 텍스트 (한국어)",
            "options": ["선택지 A", "선택지 B", "선택지 C", "선택지 D"],
            "correct_answer": "정답 선택지 텍스트 (options 중 하나와 정확히 일치)",
            "explanation": "정답 근거 및 오답 해설 (한국어)",
            "tag": "이 문제가 평가하는 기술 태그",
            "difficulty": "EASY/INTERMEDIATE/HARD"
        }
    ]
}
""",
        agent=generator,
        output_json=QuizGenerateSchema,
    )

    return Crew(
        agents=[generator],
        tasks=[generate_task],
        process=Process.sequential,
        verbose=False,
    )


def create_quiz_evaluate_crew(
    question_text: str,
    answer: str,
    quiz_attempt_id: str,
    knowledge_note_id: str,
) -> Crew:
    """퀴즈 답변 채점 Crew."""
    evaluator = create_quiz_evaluator()

    evaluate_task = Task(
        description=(
            "퀴즈 답변을 채점하고, 학습에 도움이 되는 상세 피드백을 생성합니다.\n\n"

            "**입력 정보:**\n"
            f"- 퀴즈 시도 ID: {quiz_attempt_id}\n"
            f"- 문제: {question_text}\n"
            f"- 사용자 답변: {answer}\n"
            f"- 관련 노트 ID: {knowledge_note_id}\n\n"

            "**필수 작업 순서 (반드시 따라야 함):**\n\n"

            "1. **정오 판별**:\n"
            "   - 사용자의 답변이 정답인지 판별합니다.\n"
            "   - 부분 정답(핵심은 맞지만 표현이 다른 경우)도 고려합니다.\n\n"

            "2. **해설 작성**:\n"
            "   - 정답인 경우: 왜 이 답이 맞는지 근거를 설명하고, "
            "관련 심화 개념을 간략히 소개합니다.\n"
            "   - 오답인 경우: 사용자가 선택한 답이 틀린 이유를 명확히 설명하고, "
            "정답과의 차이점을 대비하여 설명합니다.\n"
            "   - 흔히 혼동되는 포인트가 있다면 언급합니다.\n\n"

            "3. **학습 팁 제공**:\n"
            "   - 이 개념을 확실히 이해하기 위한 구체적 행동 지침을 제시합니다.\n"
            "   - 좋은 예: 'HashMap vs ConcurrentHashMap의 동기화 메커니즘을 "
            "비교하는 표를 직접 그려보세요'\n"
            "   - 나쁜 예: 'HashMap을 더 공부하세요'\n\n"

            "4. **관련 개념 연결**:\n"
            "   - 이 문제와 함께 복습하면 좋은 관련 개념 2-4개를 제시합니다.\n"
            "   - 단순 나열이 아닌, 왜 함께 복습해야 하는지 맥락을 포함합니다.\n\n"

            "**주의사항:**\n"
            "- 채점은 엄격하되, 피드백은 격려하는 톤으로 작성합니다.\n"
            "- 오답이라고 비난하지 않고, 학습 기회로 전환합니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "is_correct": true,
    "explanation": "정답/오답 해설 (한국어, 정답 근거 + 오답 이유 포함)",
    "study_tip": "구체적이고 실행 가능한 학습 방향 제안 (한국어)",
    "related_concepts": ["관련 개념 1", "관련 개념 2", "관련 개념 3"]
}
""",
        agent=evaluator,
        output_json=QuizEvaluateSchema,
    )

    return Crew(
        agents=[evaluator],
        tasks=[evaluate_task],
        process=Process.sequential,
        verbose=False,
    )
