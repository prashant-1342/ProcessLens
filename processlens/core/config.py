from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    default_llm_provider: str = "groq"

    postgres_db: str = "processlens"
    postgres_user: str = "processlens"
    postgres_password: str = "processlens"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql://processlens:processlens@localhost:5432/processlens"

    embedding_model_name: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
