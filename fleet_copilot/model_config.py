"""Configuration shared by the CLI and live-model evaluations."""

import os
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator
from typing import Literal

from .domain import Contract, FleetError

DEFAULT_PATH = Path("fleet.models.toml")
GOOGLE_BACKEND_FLAGS = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE")
NATIVE_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "google_genai": "GOOGLE_API_KEY"}
NATIVE_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "google_genai": "https://generativelanguage.googleapis.com",
}


class ModelSettings(Contract):
    provider: Literal["openai-compatible", "anthropic", "google_genai"] = "openai-compatible"
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z_][A-Z0-9_]*$")
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_output_tokens: int | None = Field(default=None, ge=128, le=8000)
    deepseek_thinking: Literal["enabled", "disabled"] | None = None

    @property
    def is_deepseek(self):
        return self.provider == "openai-compatible" and urlsplit(self.base_url or "").hostname == "api.deepseek.com"

    @model_validator(mode="after")
    def validate_endpoint(self):
        if self.max_output_tokens is not None and self.provider != "openai-compatible":
            raise ValueError("This token limit applies to an OpenAI-compatible endpoint")
        if self.deepseek_thinking and not self.is_deepseek:
            raise ValueError("DeepSeek thinking options require its official OpenAI-compatible endpoint")
        if not self.base_url:
            return self
        if self.provider != "openai-compatible":
            raise ValueError("Custom base URLs use the openai-compatible provider")
        url = urlsplit(self.base_url)
        loopback = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            (url.scheme != "https" and not (url.scheme == "http" and loopback))
            or not url.hostname or url.username is not None or url.query or url.fragment
        ):
            raise ValueError("Use an HTTPS endpoint or loopback HTTP without embedded credentials")
        return self

    def check_backend_environment(self):
        if self.provider == "google_genai":
            for flag in GOOGLE_BACKEND_FLAGS:
                if os.getenv(flag, "").casefold() in {"true", "1"}:
                    raise FleetError(f"Unset {flag} or set it to false for the Gemini Developer API profile.")

    def credential(self):
        key_env = self.api_key_env or NATIVE_KEYS.get(self.provider) or "RAYDA_API_KEY"
        key = os.getenv(key_env)
        if not key and not self.api_key_env and self.provider == "openai-compatible" and not self.base_url:
            key = os.getenv("OPENAI_API_KEY")
        if not key:
            suffix = " explicitly for a custom model endpoint" if self.base_url else " for model mode"
            raise FleetError(f"Set {key_env}{suffix}.")
        return key


def load_model_settings(path: Path | None = None, profile: str | None = None) -> ModelSettings:
    configured = os.getenv("RAYDA_MODEL_CONFIG")
    profile = profile or os.getenv("RAYDA_MODEL_PROFILE")
    if path is None:
        path = Path(configured) if configured else DEFAULT_PATH if DEFAULT_PATH.is_file() else None
    if path is not None:
        try:
            with path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError):
            raise FleetError("Could not read the model TOML configuration.") from None
        profile = profile or config.get("default_profile")
        profiles = config.get("profiles")
        if not isinstance(profile, str) or not isinstance(profiles, dict) or profile not in profiles:
            raise FleetError("Select a model profile declared in the TOML file.")
        values = profiles[profile]
    else:
        if profile:
            raise FleetError("A model profile requires a TOML configuration file.")
        if not os.getenv("RAYDA_MODEL"):
            raise FleetError("Select a model profile or set RAYDA_MODEL for model mode.")
        values = {
            "provider": os.getenv("RAYDA_PROVIDER", "openai-compatible"),
            "model": os.getenv("RAYDA_MODEL"),
            "base_url": os.getenv("RAYDA_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        }
        if values["provider"] != "openai-compatible":
            values["base_url"] = None
    try:
        return ModelSettings.model_validate(values)
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) or "endpoint" for error in exc.errors())
        raise FleetError(f"Invalid model configuration fields: {fields}.") from None
