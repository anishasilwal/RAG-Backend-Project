from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./rag_app.db"
    redis_url: str = "redis://localhost:6379/0"

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    llm_api_key: str = ""
    llm_model: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()