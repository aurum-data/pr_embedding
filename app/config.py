from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    server_host: str = Field(default="0.0.0.0", alias="HOST")
    server_port: int = Field(default=8000, alias="PORT")
    model_repo_id: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5-GGUF", alias="MODEL_REPO_ID"
    )
    model_file: str = Field(
        default="nomic-embed-text-v1.5.Q4_K_M.gguf", alias="MODEL_FILE"
    )
    model_cache_dir: Path = Field(default=Path(".models"), alias="MODEL_CACHE_DIR")
    threads: int = Field(default=4, alias="THREADS")
    batch_size: int = Field(default=64, alias="LLM_BATCH_SIZE")
    max_batch_size: int = Field(default=32, alias="MAX_BATCH_SIZE")
    context_window: int = Field(default=8192, alias="CONTEXT_WINDOW")
    server_timeout: int = Field(default=60, alias="SERVER_TIMEOUT")
    embedding_pooling: Literal["mean", "cls", "none"] = Field(
        default="mean", alias="POOLING_STRATEGY"
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
    )

    @property
    def model_path(self) -> Path:
        return (self.model_cache_dir / self.model_file).resolve()


settings = Settings()
