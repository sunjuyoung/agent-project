from fastapi import APIRouter, HTTPException

from app.schemas.resume import ResumeParseRequest, ResumeParseResponse
from app.services.resume_service import parse_resume
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/resume", tags=["resume"])


@router.post("/parse", response_model=ResumeParseResponse)
async def parse_resume_endpoint(request: ResumeParseRequest):
    """PDF 파일에서 텍스트를 추출하고 LLM으로 정제한다."""
    try:
        parsed_text = parse_resume(request.filePath)
        return ResumeParseResponse(parsedText=parsed_text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("이력서 파싱 실패")  # ← 이걸 추가
        raise HTTPException(status_code=500, detail=f"이력서 파싱 실패: {str(e)}")
