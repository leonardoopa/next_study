from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reads environment variables first, then the .env file (if present in the cwd).
    # extra="ignore": other keys in .env (e.g. POSTGRES_*) are not fields here, and
    # BaseSettings would raise an error on them by default.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Field name <-> variable name, case-insensitive: database_url <-> DATABASE_URL.
    # No default on purpose: if it is missing, the app fails at startup with a clear error.
    database_url: str

    # SecretStr: the value is hidden in repr/logs; read it with .get_secret_value().
    gemini_api_key: SecretStr
    embedding_model: str = "gemini-embedding-2"
    # Must match the column: chunks.embedding is vector(768).
    embedding_dim: int = 768


@lru_cache
def get_settings() -> Settings:
    # Built once, then reused (lru_cache).
    return Settings()
