from fastapi import APIRouter, HTTPException

from app.schemas.knowledge import EmbedRequest
from app.services.embedding_service import embed_and_store

router = APIRouter(prefix="/ai/knowledge", tags=["knowledge"])


@router.post("/embed")
async def embed_knowledge(request: EmbedRequest):
    """MD 파일 텍스트 → 임베딩 저장."""
    try:
        result = embed_and_store(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 저장 실패: {str(e)}")
