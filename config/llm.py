import json
import os
from pathlib import Path

from openai import OpenAI


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "llm_settings.json"


def _load_settings():
    with _SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_llm_settings():
    settings = _load_settings()
    provider = os.getenv("LLM_PROVIDER", settings.get("provider", "mistral"))
    base_url = os.getenv("LLM_BASE_URL", settings.get("base_url"))
    api_key_env = settings.get("api_key_env", "MISTRAL_API_KEY")
    api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "models": settings.get("models", {}),
    }


def get_default_model(kind="default"):
    settings = get_llm_settings()
    models = settings["models"]
    return os.getenv("LLM_MODEL") or models.get(kind) or models.get("default")


def get_llm_client():
    settings = get_llm_settings()
    return OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
