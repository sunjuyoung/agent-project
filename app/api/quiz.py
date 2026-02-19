from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.quiz import (
    QuizGenerateRequest,
    QuizEvaluateRequest,
    QuizGenerateSchema,
    QuizEvaluateSchema,
)
from app.crews.quiz_crew import create_quiz_generate_crew, create_quiz_evaluate_crew
from app.utils.crew_utils import parse_crew_output

router = APIRouter(prefix="/ai/quiz", tags=["quiz"])


@router.post("/generate", response_model=QuizGenerateSchema, response_class=JSONResponse)
async def generate_quiz(request: QuizGenerateRequest) -> QuizGenerateSchema:
    """태그/난이도 기반 퀴즈 문제 생성."""
    try:
        crew = create_quiz_generate_crew(
            user_id=request.user_id,
            tags=request.tags,
            difficulty=request.difficulty.value,
            count=request.count,
        )
        result = crew.kickoff()

        return parse_crew_output(result, QuizGenerateSchema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 실패: {str(e)}")


@router.post("/evaluate", response_model=QuizEvaluateSchema, response_class=JSONResponse)
async def evaluate_quiz(request: QuizEvaluateRequest) -> QuizEvaluateSchema:
    """퀴즈 답변 채점 + 피드백."""
    try:
        crew = create_quiz_evaluate_crew(
            question_text=request.question_text,
            answer=request.answer,
            quiz_attempt_id=request.quiz_attempt_id,
            knowledge_note_id=request.knowledge_note_id,
        )
        result = crew.kickoff()

        return parse_crew_output(result, QuizEvaluateSchema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 채점 실패: {str(e)}")
