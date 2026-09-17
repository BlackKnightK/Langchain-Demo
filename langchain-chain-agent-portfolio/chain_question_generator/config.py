import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass(frozen=True)
class ModelSettings:
    api_key: str
    model: str
    base_url: str | None


def get_settings() -> ModelSettings:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Copy .env.example to .env and add your key.")

    return ModelSettings(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
    )


def build_model(temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()
    kwargs = {
        "model": settings.model,
        "api_key": settings.api_key,
        "temperature": temperature,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(**kwargs)
