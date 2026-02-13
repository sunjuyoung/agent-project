from pydantic_settings import BaseSettings
from pydantic import computed_field
from crewai import LLM


class Settings(BaseSettings):
    model_config = {"env_file": ".env", 
                    "env_file_encoding": "utf-8",
                    "extra": "ignore",
                    "case_sensitive": True,
                    }

    # LLM
    LLM_MODEL: str = "openai/gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3
    OPENAI_API_KEY: str
    FIRECRAWL_API_KEY: str

    # Database (pgvector)
    DATABASE_URL: str

    PGVECTOR_CONNECTION_URL: str

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_CHUNK_SIZE: int = 500

    # CrewAI
    CREW_VERBOSE: bool = False


settings = Settings()
