from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq — primary LLM
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"

    # Anthropic — secondary LLM
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # OpenAI — tertiary LLM
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # App
    cors_origins: str = "http://localhost:3000"

    model_config = {"env_file": [".env", "../.env"], "extra": "ignore"}

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
