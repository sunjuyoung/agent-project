import json

from fastapi import APIRouter, HTTPException
from crewai import Agent, Task, Crew, Process

from app.schemas.quiz import QuizGenerateRequest, QuizEvaluateRequest
from app.tools.rag_search import RAGSearchTool

router = APIRouter(prefix="/ai/quiz", tags=["quiz"])


@router.post("/generate")
async def generate_quiz(request: QuizGenerateRequest):
    """태그/난이도 기반 퀴즈 문제 생성."""
    try:
        rag_tool = RAGSearchTool()
        quiz_agent = Agent(
            role="퀴즈 출제자",
            goal="사용자의 학습 노트를 기반으로 효과적인 퀴즈 문제를 생성한다",
            backstory=(
                "당신은 기술 학습 퀴즈 전문가입니다. "
                "사용자의 학습 노트를 RAG 검색하여 "
                "해당 태그와 난이도에 맞는 문제를 생성합니다."
            ),
            llm="gpt-4o-mini",
            tools=[rag_tool],
            allow_delegation=False,
            verbose=False,
        )

        generate_task = Task(
            description=(
                f"다음 조건에 맞는 퀴즈 문제를 {request.count}개 생성하라.\n\n"
                f"태그: {json.dumps(request.tags, ensure_ascii=False)}\n"
                f"난이도: {request.difficulty.value}\n"
                f"사용자 ID: {request.user_id}\n\n"
                "RAG로 사용자의 학습 노트를 검색하여 문제를 출제하라.\n"
                "각 문제에는 question, options(4지선다), correct_answer, explanation을 포함하라."
            ),
            expected_output=(
                "JSON: questions[] with question, options[], correct_answer, explanation"
            ),
            agent=quiz_agent,
        )

        crew = Crew(
            agents=[quiz_agent],
            tasks=[generate_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 실패: {str(e)}")


@router.post("/evaluate")
async def evaluate_quiz(request: QuizEvaluateRequest):
    """퀴즈 답변 채점 + 피드백."""
    try:
        quiz_agent = Agent(
            role="퀴즈 채점자",
            goal="퀴즈 답변을 정확히 채점하고 학습에 도움이 되는 피드백을 제공한다",
            backstory=(
                "당신은 기술 학습 퀴즈 채점 전문가입니다. "
                "정답 여부를 판단하고, 오답일 경우 왜 틀렸는지 설명하며 "
                "관련 개념을 복습할 수 있도록 피드백을 제공합니다."
            ),
            llm="gpt-4o-mini",
            allow_delegation=False,
            verbose=False,
        )

        evaluate_task = Task(
            description=(
                f"다음 퀴즈 답변을 채점하라.\n\n"
                f"퀴즈 시도 ID: {request.quiz_attempt_id}\n"
                f"문제: {request.question_text}\n"
                f"사용자 답변: {request.answer}\n"
                f"관련 노트 ID: {request.knowledge_note_id}\n\n"
                "정답 여부, 해설, 관련 개념 복습 팁을 제공하라."
            ),
            expected_output="JSON: is_correct, explanation, study_tip",
            agent=quiz_agent,
        )

        crew = Crew(
            agents=[quiz_agent],
            tasks=[evaluate_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 채점 실패: {str(e)}")
