import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


class RAGSearchInput(BaseModel):
    query: str = Field(description="검색할 기술 키워드 또는 질문")
    top_k: int = Field(default=5, description="반환할 최대 결과 수")


class RAGSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = (
        "사용자의 MD 스터디 노트에서 관련 지식을 검색한다. "
        "pgvector similarity_search를 사용하며, "
        "특정 기술 키워드나 개념에 대한 사용자의 학습 내용을 반환한다."
    )
    args_schema: type[BaseModel] = RAGSearchInput
    user_id: str = ""

    def _run(self, query: str, top_k: int = 5) -> str:
        try:
            embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
            vectorstore = PGVector(
                embeddings=embeddings,
                connection=settings.PGVECTOR_CONNECTION_URL,
                collection_name="knowledge_embeddings",
                use_jsonb=True,
            )

            search_filter = {"user_id": self.user_id} if self.user_id else None
            docs = vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=search_filter,
            )

            if not docs:
                return "관련 스터디 노트를 찾지 못했습니다."

            results = []
            for doc in docs:
                results.append(
                    f"[출처: {doc.metadata.get('note_id', 'unknown')}]\n{doc.page_content}"
                )
            return "\n---\n".join(results)
        except Exception as e:
            logger.error(f"RAG 검색 중 오류 발생: {e}")
            return f"RAG 검색 중 오류가 발생했습니다: {str(e)}"
