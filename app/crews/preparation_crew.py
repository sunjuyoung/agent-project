from crewai import Task, Crew, Process

from app.agents.analyst import create_analyst
from app.agents.planner import create_planner
from app.schemas.interview import InterviewScenarioSchema
from app.tools.rag_search import RAGSearchTool


def create_preparation_crew(user_id: str) -> Crew:
    """Phase 1: 이력서 + JD + RAG → 질문 시나리오 생성 Crew."""
    # 비동기 태스크별 별도 에이전트 인스턴스 생성 (공유 시 race condition 발생)
    
    rag_tool = RAGSearchTool(user_id=user_id)
    
    resume_analyst = create_analyst(user_id=user_id)
    jd_analyst = create_analyst(user_id=user_id)
    rag_analyst = create_analyst(user_id=user_id)
    planner = create_planner()

    # ── Task A: 이력서 분석 ──────────────────────────────────────
    task_a = Task(
        description=(
            "후보자의 이력서를 정밀 분석하여 핵심 정보를 구조화된 형태로 추출합니다.\n\n"
            "**분석 대상 이력서:**\n{resume_text}\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **기술 키워드 추출**:\n"
            "   - 프로그래밍 언어, 프레임워크, 라이브러리, 도구를 모두 식별합니다.\n"
            "   - 예: Java, Spring Boot, React, Docker, Kubernetes 등\n"
            "   - 단순 나열이 아닌, 각 기술의 숙련도(주력/보조/경험)를 구분합니다.\n\n"
            "   - 기술 숙련도 추정이 어려울 경우, 경험 수준을 추정합니다.\n\n"
            "2. **프로젝트 경험 분석**:\n"
            "   - 각 프로젝트의 이름, 기간, 역할, 사용 기술, 주요 성과를 추출합니다.\n"
            "   - 프로젝트에서 본인이 담당한 구체적인 기여도를 파악합니다.\n"
            "   - 팀 규모, 아키텍처 관련 의사결정 경험이 있다면 반드시 포함합니다.\n\n"
            "3. **경력 연차 산출**:\n"
            "   - 총 경력 연차를 계산합니다.\n"
            "   - 기술별 사용 기간이 명시되어 있다면 함께 추출합니다.\n"
            "   - 신입/주니어(0-3년)/미드레벨(3-7년)/시니어(7년+) 수준을 판단합니다.\n\n"
            "**주의사항:**\n"
            "- 이력서에 명시되지 않은 정보를 추측하거나 임의로 생성하지 마세요.\n"
            "- 약어나 줄임말은 풀네임과 함께 기록하세요. (예: k8s → Kubernetes)\n"
            "- 기술 키워드는 가능한 한 공식 명칭을 사용하세요."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "skills": [
        {
            "name": "기술명",
            "level": "주력/보조/경험",
            "usage_period": "사용 기간 (명시된 경우)"
        }
    ],
    "projects": [
        {
            "name": "프로젝트명",
            "duration": "기간",
            "role": "담당 역할",
            "tech_stack": ["사용 기술"],
            "achievements": ["주요 성과"],
            "team_size": "팀 규모 (명시된 경우)"
        }
    ],
    "experience_years": {
        "total": "총 경력 연차",
        "level": "신입/주니어/미드레벨/시니어",
        "by_skill": {"기술명": "사용 기간"} 
    }
}
""",
        agent=resume_analyst,
        async_execution=True,
    )
    
    
    # ── Task B: JD 분석 ──────────────────────────────────────────
    task_b = Task(
        description=(
            "채용 공고(Job Description)를 정밀 분석하여 요구사항을 구조화합니다.\n\n"
            "**분석 대상 JD:**\n{jd_text}\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **필수 스킬 분류**:\n"
            "   - JD에서 '필수', '자격 요건', 'required' 등으로 명시된 기술을 추출합니다.\n"
            "   - 각 필수 스킬의 요구 숙련도 수준을 파악합니다.\n"
            "   - 하드 스킬(기술)과 소프트 스킬(커뮤니케이션, 리더십 등)을 구분합니다.\n\n"
            "2. **우대 스킬 분류**:\n"
            "   - '우대', '선호', 'preferred', 'nice to have' 등으로 명시된 기술을 추출합니다.\n"
            "   - 우대 스킬 중 필수와 겹치는 항목이 있다면 필수로 분류합니다.\n\n"
            "3. **기대 경험 수준 파악**:\n"
            "   - 요구 경력 연차 범위를 파악합니다.\n"
            "   - 직급/포지션 레벨을 식별합니다. (주니어/미드/시니어/리드)\n"
            "   - 특정 도메인 경험 요구사항이 있다면 기록합니다. (핀테크, 이커머스 등)\n\n"
            "4. **직무 핵심 키워드 추출**:\n"
            "   - 면접에서 다뤄질 가능성이 높은 핵심 기술 키워드를 별도로 정리합니다.\n"
            "   - JD에서 반복적으로 강조되는 기술이나 역량을 우선순위로 배치합니다.\n\n"
            "**주의사항:**\n"
            "- JD에 명시되지 않은 요구사항을 추측하지 마세요.\n"
            "- 필수와 우대가 모호한 경우 보수적으로(필수로) 분류하세요.\n"
            "- 기술 키워드는 이력서 분석과 동일한 공식 명칭을 사용하세요."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "required_skills": [
        {
            "name": "스킬명",
            "type": "hard/soft",
            "required_level": "요구 수준"
        }
    ],
    "preferred_skills": [
        {
            "name": "스킬명",
            "type": "hard/soft"
        }
    ],
    "expected_level": {
        "experience_range": "요구 경력 범위",
        "position_level": "직급/포지션 레벨",
        "domain_experience": ["특정 도메인 경험 요구사항"]
    },
    "key_interview_keywords": ["면접 핵심 키워드 (중요도순)"]
}
""",
        agent=jd_analyst,
        async_execution=True,
    )


# ── Task C: RAG 검색 ─────────────────────────────────────────
    task_c = Task(
        description=(
            "JD 핵심 키워드를 기반으로 사용자의 MD 스터디 노트를 RAG 검색하여 "
            "면접 준비에 활용할 수 있는 관련 지식을 수집합니다.\n\n"
            "**검색 키워드:**\n{jd_keywords}\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **키워드별 RAG 검색 실행**:\n"
            "   - 제공된 키워드 각각에 대해 RAGSearchTool을 호출하여 검색합니다.\n"
            "   - 반드시 도구를 실제로 호출하고, 반환된 결과만 사용하세요.\n"
            "   - **절대로 임의의 노트 내용을 생성하지 마세요.**\n\n"
            "2. **검색 결과 정리**:\n"
            "   - 각 검색 결과에서 면접 질문에 활용 가능한 개념, 용어, 설명을 추출합니다.\n"
            "   - 관련성이 낮은 결과는 제외합니다.\n\n"
            "3. **지식 수준 평가**:\n"
            "   - 검색 결과를 바탕으로 사용자가 해당 기술에 대해 어느 정도 학습했는지 추정합니다.\n"
            "   - 이 정보는 면접 질문 난이도 조절에 활용됩니다.\n\n"
            "**경고:**\n"
            "- 반드시 RAGSearchTool을 호출하여 실제 데이터를 가져와야 합니다.\n"
            "- 도구 호출 없이 결과를 지어내는 것은 절대 금지됩니다.\n"
            "- 검색 결과가 없는 키워드는 '검색 결과 없음'으로 명시하세요."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "relevant_notes": [
        {
            "keyword": "검색한 키워드",
            "content_summary": "관련 내용 요약",
            "relevance_score": "1-10 관련성 점수"
        }
    ]
}
""",
        agent=rag_analyst,
        async_execution=True,
        tools=[rag_tool],
    )

    # ── Task D: 후보자 프로필 종합 ──────────────────────────────
    task_d = Task(
        description=(
            "이력서 분석(Task A), JD 분석(Task B), RAG 검색(Task C) 결과를 종합하여 "
            "면접용 후보자 프로필을 생성합니다.\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **강점/약점 매핑**:\n"
            "   - 이력서 스킬과 JD 필수 스킬을 대조하여 매칭되는 강점을 식별합니다.\n"
            "   - JD에서 요구하지만 이력서에 없거나 약한 스킬을 약점으로 분류합니다.\n"
            "   - RAG 검색 결과를 반영하여 학습으로 보완 가능한 약점을 구분합니다.\n"
            "   - 각 강점/약점에 대한 구체적 근거를 이력서 내용에서 인용합니다.\n\n"
            "2. **JD 매칭 점수 산출**:\n"
            "   - 필수 스킬 매칭률 (가중치 70%)을 계산합니다.\n"
            "   - 우대 스킬 매칭률 (가중치 20%)을 계산합니다.\n"
            "   - 경력 수준 적합도 (가중치 10%)를 평가합니다.\n"
            "   - 종합 매칭 점수를 0-100 스케일로 산출합니다.\n\n"
            "3. **기술 매트릭스 생성**:\n"
            "   - JD에서 요구하는 모든 기술에 대해 후보자의 수준을 매핑합니다.\n"
            "   - 각 기술별로 이력서 기반 숙련도 + RAG 기반 학습 수준을 종합합니다.\n"
            "   - 면접에서 깊이 있게 질문할 수 있는 영역과 기본만 확인할 영역을 구분합니다.\n\n"
            "4. **면접 전략 제안**:\n"
            "   - 강점 영역: 심화 질문으로 역량을 검증할 포인트를 제안합니다.\n"
            "   - 약점 영역: 학습 의지와 성장 가능성을 확인할 포인트를 제안합니다.\n"
            "   - 프로젝트 기반: 이력서 프로젝트에서 구체적으로 물어볼 포인트를 제안합니다.\n\n"
            "**주의사항:**\n"
            "- Task A, B, C의 실제 결과 데이터만 사용하세요. 임의로 데이터를 생성하지 마세요.\n"
            "- 매칭 점수는 객관적 기준에 따라 일관성 있게 산출하세요.\n"
            "- 강점과 약점은 면접 질문으로 전환 가능한 수준으로 구체적이어야 합니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "candidate_profile": {
        "strengths": [
            {
                "skill": "강점 기술/역량",
                "evidence": "이력서에서의 근거",
                "interview_point": "면접에서 검증할 포인트"
            }
        ],
        "weaknesses": [
            {
                "skill": "약점 기술/역량",
                "gap_description": "부족한 부분 설명",
                "compensable_by_study": true/false,
                "interview_point": "면접에서 확인할 포인트"
            }
        ],
        "jd_match_score": {
            "required_skills_match": "필수 스킬 매칭률 (%)",
            "preferred_skills_match": "우대 스킬 매칭률 (%)",
            "experience_fit": "경력 수준 적합도 (%)",
            "overall_score": "종합 점수 (0-100)"
        },
        "skill_matrix": [
            {
                "skill": "기술명",
                "jd_requirement": "필수/우대",
                "candidate_level": "상/중/하/없음",
                "study_level": "학습완료/진행중/미학습/해당없음",
                "question_depth": "심화/기본/스킵"
            }
        ],
        "interview_strategy": {
            "deep_dive_areas": ["심화 질문 대상 영역"],
            "basic_check_areas": ["기본 확인 대상 영역"],
            "project_based_points": ["프로젝트 기반 질문 포인트"]
        }
    }
}
""",
        agent=resume_analyst,
        context=[task_a, task_b, task_c],
        
    )

    # ── Task E: 면접 질문 시나리오 설계 ─────────────────────────
    task_e = Task(
        description=(
            "후보자 프로필(Task D)을 기반으로 실전 면접 질문 시나리오를 설계합니다.\n\n"
            "**필수 작업 순서 (반드시 따라야 함):**\n\n"
            "1. **질문 배분 계획 수립**:\n"
            "   - 총 {question_count}개의 메인 질문을 설계합니다.\n"
            "   - 강점 기반 질문: 전체의 40% (후보자가 잘 아는 영역을 깊이 검증)\n"
            "   - 약점 기반 질문: 전체의 40% (부족한 영역의 학습 의지와 이해도 확인)\n"
            "   - 행동(Behavioral) 질문: 전체의 20% (팀워크, 문제해결, 커뮤니케이션 등)\n"
            "   - 소수점이 발생하면 강점/약점 질문에 우선 배분합니다.\n\n"
            "2. **난이도 설계**:\n"
            "   - 기준 난이도: {difficulty}\n"
            "   - 질문 순서에 따라 점진적으로 난이도를 상승시킵니다.\n"
            "   - EASY → MEDIUM → HARD 순서로 배치합니다.\n"
            "   - 기준 난이도가 MEDIUM이면 EASY 1-2개 → MEDIUM 다수 → HARD 1-2개로 구성합니다.\n"
            "   - 기준 난이도가 HARD이면 MEDIUM 1-2개 → HARD 다수로 구성합니다.\n\n"
            "3. **질문별 상세 설계**:\n"
            "   - 각 질문마다 다음을 포함해야 합니다:\n"
            "     • 고유 ID (q1, q2, ... 형식)\n"
            "     • 타겟 스킬 (어떤 기술/역량을 평가하는지)\n"
            "     • 난이도 (EASY/MEDIUM/HARD)\n"
            "     • 질문 텍스트 (구체적이고 명확한 한국어 질문)\n"
            "     • 평가 기준 (이 질문으로 무엇을 평가하며, 좋은 답변의 조건)\n"
            "     • 사전 꼬리질문은 질문별 최대 1개 (메인 질문에 대한 후속 질문)\n\n"
            "     • 사전 꼬리질문은 필수 아님\n\n"
            "4. **질문 품질 기준**:\n"
            "   - 이력서 프로젝트 경험을 직접 언급하는 질문을 2~3개 포함합니다.\n"
            "   - '~에 대해 설명해주세요'와 같은 단순 지식 질문보다 "
            "'~상황에서 어떻게 해결하셨나요?'와 같은 경험 기반 질문을 선호합니다.\n"
            "   - 꼬리질문은 메인 질문의 답변을 심화 또는 연결하는 방향이어야 합니다.\n"
            "   - 각 질문은 서로 중복되지 않는 독립적인 평가 영역을 다뤄야 합니다.\n\n"
            "**주의사항:**\n"
            "- Task D의 후보자 프로필 데이터를 반드시 참조하여 질문을 설계하세요.\n"
            "- 후보자 프로필의 skill_matrix와 interview_strategy를 질문 설계에 활용하세요.\n"
            "- 질문은 모두 한국어로 작성하세요.\n"
            "- 평가 기준은 면접관이 즉시 활용할 수 있을 만큼 구체적이어야 합니다."
        ),
        expected_output="""
다음 형식의 JSON:
{
    "scenario": {
        "total_questions": "총 질문 수",
        "difficulty_base": "기준 난이도",
        "distribution": {
            "strength": "강점 질문 수",
            "weakness": "약점 질문 수",
            "behavioral": "행동 질문 수"
        },
        "questions": [
            {
                "id": "q1",
                "category": "strength/weakness/behavioral",
                "skill_target": "평가 대상 기술/역량",
                "difficulty": "EASY/MEDIUM/HARD",
                "text": "면접 질문 텍스트 (한국어)",
                "evaluation_criteria": [
                    "평가 기준 1: 좋은 답변의 구체적 조건",
                    "평가 기준 2: 확인해야 할 핵심 포인트"
                ],
                "follow_ups": [
                    {
                        "text": "꼬리질문 텍스트 (한국어)",
                        "purpose": "이 꼬리질문의 평가 목적"
                    }
                ]
            }
        ]
    }
}
""",
        agent=planner,
        context=[task_d]
    )

    return Crew(
        agents=[resume_analyst, jd_analyst, rag_analyst, planner],
        tasks=[task_a, task_b, task_c, task_d, task_e],
        process=Process.sequential,
        verbose=True,
    )
