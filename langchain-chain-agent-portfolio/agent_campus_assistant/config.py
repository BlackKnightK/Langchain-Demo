import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_base_url: str | None
    tavily_api_key: str | None


def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Copy .env.example to .env and add your key.")

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
    )


def build_model(temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()
    kwargs = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)
