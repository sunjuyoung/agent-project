import json

from fastapi import APIRouter, HTTPException

from app.schemas.interview import (
    PrepareRequest,
    EvaluateRequest,
    ReportRequest,
    InterviewScenarioSchema,
    InterviewDecisionSchema,
    InterviewReportSchema,
    EvaluationResultSchema,
)
from app.crews.preparation_crew import create_preparation_crew
from app.crews.interview_turn_crew import create_interview_turn_crew
from app.crews.report_crew import create_report_crew
from app.utils.crew_utils import (
    parse_crew_output,
    parse_crew_output_from_task,
    format_conversation_log,
    format_transcript,
    extract_keywords,
)

router = APIRouter(prefix="/ai/interview", tags=["interview"])


@router.post("/prepare")
async def prepare_interview(request: PrepareRequest) -> InterviewScenarioSchema:
    """Phase 1: 이력서 + JD + RAG → 질문 시나리오 생성."""
    try:
        crew, task_e = create_preparation_crew(user_id=request.user_id)
        crew.kickoff(
            inputs={
                "resume_text": request.resume_text,
                "jd_text": request.jd_text,
                "jd_keywords": extract_keywords(request.jd_text),
                "question_count": request.question_count,
                "difficulty": request.difficulty.value,
            }
        )
        
        # task_e는 crew.kickoff() 실행 후 output이 채워짐
        scenario_schema: InterviewScenarioSchema = task_e.output.pydantic

        return scenario_schema
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"면접 준비 실패: {str(e)}")

@router.post("/evaluate")
async def evaluate_answer(request: EvaluateRequest):
    """Phase 2: 답변 평가 + 다음 질문 결정."""
    try:
        conversation_log_str = format_conversation_log(request.turn_history)

        # ★ 변경: 시나리오에서 현재 질문의 전체 정보를 추출
        questions = request.question_scenario.get("questions", [])
        matched_question = next(
            (q for q in questions if q.get("id") == request.question_id),
            None,
        )

        # current_question에 eval_task가 참조하는 모든 필드 포함
        current_question = {
            "id": request.question_id,
            "text": request.question_text,
            "skillTarget": (
                matched_question.get("skillTarget", matched_question.get("skill_target", ""))
                if matched_question else request.skill_target
            ),
            "difficulty": (
                matched_question.get("difficulty", "")
                if matched_question else request.difficulty
            ),
        }

        # evaluation_criteria도 시나리오에서 추출
        evaluation_criteria = (
            matched_question.get("evaluationCriteria",
                matched_question.get("evaluation_criteria", []))
            if matched_question else []
        )

        # 남은 질문 수 계산
        total = request.question_scenario.get(
            "totalQuestions",
            request.question_scenario.get("total_questions", 0),
        )
        answered_count = len(request.turn_history)
        remaining_count = max(0, total - answered_count)

        crew = create_interview_turn_crew(
            user_id=request.session_id,
            current_question=current_question,
            user_answer=request.answer,
            follow_up_count=request.follow_up_count,
            remaining_count=remaining_count,
            conversation_log=conversation_log_str,
            evaluation_criteria=evaluation_criteria,
            scenario=request.question_scenario,
        )
        result = crew.kickoff()

        eval_result = parse_crew_output_from_task(
            result.tasks_output[0], EvaluationResultSchema
        )
        decision = parse_crew_output(result, InterviewDecisionSchema)

        next_q = None
        if decision.next_question:
            next_q = decision.next_question.model_dump(by_alias=True)

        return {
            "score": eval_result.score,
            "feedback": eval_result.feedback,
            "decision": decision.decision.value,
            "nextQuestion": next_q,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 평가 실패: {str(e)}")


@router.post("/report")
async def generate_report(request: ReportRequest):
    """Phase 3: 전체 면접 로그 → 종합 리포트."""
    try:
        crew = create_report_crew()
        result = crew.kickoff(
            inputs={
                "transcript": format_transcript(request.turns),
                "all_evaluations": json.dumps(request.turns, ensure_ascii=False, indent=2),
                "jd_text": json.dumps(request.question_scenario, ensure_ascii=False),
            }
        )

        report = parse_crew_output(result, InterviewReportSchema)
        return report.model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 실패: {str(e)}")
