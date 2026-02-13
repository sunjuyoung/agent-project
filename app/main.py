from fastapi import FastAPI

from app.api import interview, knowledge, quiz, resume

app = FastAPI(
    title="AI Interview Coach - AI Service",
    description="CrewAI 기반 모의면접 AI 오케스트레이션 서비스",
    version="0.1.0",
)

app.include_router(interview.router)
app.include_router(knowledge.router)
app.include_router(quiz.router)
app.include_router(resume.router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-service"}
