from crewai import Task, Crew, Process

from app.agents.coach import create_coach
from app.schemas.interview import InterviewReportSchema



def create_report_crew() -> Crew:
    """Phase 3: 전체 면접 transcript → 종합 리포트 생성 Crew."""
    coach = create_coach()

    report_task = Task(
        description=(
            "전체 면접 기록과 턴별 평가 데이터를 종합 분석하여 면접 리포트를 생성합니다.\n\n"
            "**입력 데이터:**\n"
            "- 면접 transcript:\n{transcript}\n"
            "- 각 턴별 평가 데이터:\n{all_evaluations}\n"
            "- JD 요구사항:\n{jd_text}\n\n"

            "**필수 작업 순서 (반드시 따라야 함):**\n\n"

            "1. **턴별 평가 데이터 집계**:\n"
            "   - all_evaluations의 각 턴별 score를 수집합니다.\n"
            "   - 메인 질문 답변(가중치 70%)과 꼬리질문 답변(가중치 30%)을 구분합니다.\n"
            "   - 가중 평균을 계산하여 기초 점수를 산출합니다.\n"
            "   - 이 기초 점수를 1-100 스케일로 변환합니다.\n\n"

            "2. **커뮤니케이션 품질 분석**:\n"
            "   - transcript 전체를 통해 후보자의 답변 패턴을 분석합니다.\n"
            "   - 평가 항목: 구조화 능력, 기술 용어 정확성, 질문 의도 파악, 간결성\n"
            "   - 1-10 스케일로 점수를 매기고, 각 항목에 대한 근거를 기록합니다.\n"
            "   - 커뮤니케이션 점수에 따라 overall_score를 ±5점 보정합니다.\n\n"

            "3. **강점 도출**:\n"
            "   - all_evaluations에서 점수가 높은 턴(7점 이상)을 식별합니다.\n"
            "   - 해당 턴의 hits(정확히 언급한 포인트)를 기반으로 강점을 구성합니다.\n"
            "   - 각 강점에 후보자가 실제 답변한 내용을 구체적으로 인용합니다.\n"
            "   - JD 요구사항과의 연관성을 함께 기술합니다.\n\n"

            "4. **개선점 도출 및 우선순위화**:\n"
            "   - all_evaluations에서 점수가 낮은 턴(5점 이하)을 식별합니다.\n"
            "   - 해당 턴의 misses(누락/부정확 포인트)를 기반으로 개선점을 구성합니다.\n"
            "   - 영향도 기준 우선순위 산정:\n"
            "     • JD 필수 스킬 관련 개선점 → 최우선\n"
            "     • JD 우대 스킬 관련 개선점 → 중간\n"
            "     • 기타 개선점 → 낮음\n"
            "   - 각 개선점에 구체적인 보완 방향을 함께 제시합니다.\n\n"

            "5. **지식 갭 종합 분석**:\n"
            "   - all_evaluations의 각 턴별 note_comparison 데이터를 수집합니다.\n"
            "   - 다음 3개 카테고리로 분류합니다:\n"
            "     • studied_and_strong: 학습 완료 + 면접에서 잘 답변한 영역\n"
            "       (hits에 포함되고 note_comparison에서 studied 기록이 있는 항목)\n"
            "     • studied_but_weak: 학습은 했으나 면접에서 활용 못한 영역\n"
            "       (note_comparison.studied_but_missed 항목 종합)\n"
            "     • not_studied: 아직 학습하지 않은 영역\n"
            "       (note_comparison.not_studied 항목 종합)\n"
            "   - 각 카테고리에 해당하는 구체적 기술/개념을 나열합니다.\n\n"

            "6. **다음 단계(next_steps) 설계**:\n"
            "   - 개선점과 지식 갭을 기반으로 구체적 액션 플랜을 작성합니다.\n"
            "   - 우선순위순으로 최대 5개까지 제시합니다.\n"
            "   - 각 단계는 실행 가능한(actionable) 형태여야 합니다.\n"
            "     • 좋은 예: 'JPA N+1 문제를 @EntityGraph와 Fetch Join으로 해결하는 "
            "실습 프로젝트를 진행하세요'\n"
            "     • 나쁜 예: 'JPA를 더 공부하세요'\n"
            "   - 사용자의 학습 노트(note_comparison)에서 studied_but_weak 항목은 "
            "'복습 및 실전 적용' 방향으로, not_studied 항목은 '신규 학습' 방향으로 안내합니다.\n\n"

            "7. **등급(grade) 부여**:\n"
            "   - overall_score 기준:\n"
            "     S(90-100) / A(80-89) / B(70-79) / C(60-69) / D(50-59) / F(50 미만)\n\n"

            "8. **JD 대비 준비도 평가**:\n"
            "   - JD 필수 스킬 각각에 대해 면접 답변 기반 준비도를 평가합니다.\n"
            "   - 전체적인 JD 매칭도를 서술합니다.\n\n"

            "**주의사항:**\n"
            "- 모든 분석은 all_evaluations와 transcript의 실제 데이터에 기반해야 합니다.\n"
            "- 임의의 점수나 평가를 생성하지 않습니다.\n"
            "- 격려하되 솔직하게: 낮은 점수도 회피하지 않고 건설적으로 전달합니다.\n"
            "- 리포트를 읽는 후보자가 '다음에 무엇을 해야 하는지' 명확히 알 수 있어야 합니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "overall_score": 72,
    "grade": "B",
    "strengths": [
        {
            "skill": "강점 기술/역량",
            "detail": "면접에서 보여준 구체적 역량 설명",
            "evidence": "실제 답변에서 인용한 근거"
        }
    ],
    "improvements": [
        {
            "skill": "개선 필요 기술/역량",
            "detail": "부족했던 부분에 대한 구체적 설명",
            "priority": "HIGH/MEDIUM/LOW",
            "suggestion": "구체적 보완 방향"
        }
    ],
    "knowledge_gap_summary": {
        "studied_and_strong": ["학습 완료 + 면접 답변 우수 영역"],
        "studied_but_weak": ["학습은 했으나 면접에서 활용 못한 영역"],
        "not_studied": ["아직 학습하지 않은 영역"]
    },
    "next_steps": [
        {
            "priority": 1,
            "action": "구체적이고 실행 가능한 다음 단계",
            "reason": "이 단계를 제안하는 이유",
            "related_gap": "연관된 지식 갭 카테고리"
        }
    ],
    "communication_quality": {
        "score": 7,
        "structure": "답변 구조화 능력 평가",
        "terminology": "기술 용어 사용 정확성",
        "comprehension": "질문 의도 파악 정확도",
        "conciseness": "간결성 평가",
        "overall_comment": "종합 커뮤니케이션 피드백"
    }
}
""",
        agent=coach,
        output_json=InterviewReportSchema,
    )

    return Crew(
        agents=[coach],
        tasks=[report_task],
        process=Process.sequential,
        verbose=False,
    )
