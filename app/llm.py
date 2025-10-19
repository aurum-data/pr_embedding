from __future__ import annotations

import threading
from functools import lru_cache
from inspect import signature
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .config import settings

try:
    from llama_cpp import LlamaEmbeddingPoolingType

    POOLING_OPTIONS = {
        "mean": LlamaEmbeddingPoolingType.MEAN,
        "cls": LlamaEmbeddingPoolingType.CLS,
        "none": LlamaEmbeddingPoolingType.NONE,
    }
except ImportError:  # pragma: no cover - fallback for older llama_cpp
    POOLING_OPTIONS = {"mean": "mean", "cls": "cls", "none": "none"}

try:
    _CREATE_EMBEDDING_ACCEPTS_POOLING = (
        "pooling_type" in signature(Llama.create_embedding).parameters
    )
except (ValueError, TypeError):  # pragma: no cover - defensive guard
    _CREATE_EMBEDDING_ACCEPTS_POOLING = False


class EmbeddingModel:
    """Thread-safe lazy loader for llama.cpp embeddings."""

    _instance: Llama | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model_path: Path | None = None

    def _download_model(self) -> Path:
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(
            hf_hub_download(
                repo_id=settings.model_repo_id,
                filename=settings.model_file,
                local_dir=settings.model_cache_dir,
            )
        )
        self._model_path = model_path
        return model_path

    def _load_llama(self) -> Llama:
        if EmbeddingModel._instance is not None:
            return EmbeddingModel._instance

        with EmbeddingModel._lock:
            if EmbeddingModel._instance is not None:
                return EmbeddingModel._instance

            model_path = self._download_model()
            EmbeddingModel._instance = Llama(
                model_path=str(model_path),
                embedding=True,
                n_ctx=settings.context_window,
                n_threads=settings.threads,
                n_batch=settings.batch_size,
                n_gpu_layers=0,
            )
            return EmbeddingModel._instance

    @staticmethod
    def _ensure_iterable(texts: Iterable[str] | str) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def embed(
        self,
        inputs: Iterable[str] | str,
        *,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        llama = self._load_llama()
        payload = self._ensure_iterable(inputs)

        if not payload:
            return []

        resolved_batch = min(batch_size or settings.max_batch_size, settings.max_batch_size)
        embeddings: list[list[float]] = []
        pooling = POOLING_OPTIONS.get(settings.embedding_pooling, POOLING_OPTIONS["mean"])

        for index in range(0, len(payload), resolved_batch):
            chunk = payload[index : index + resolved_batch]
            kwargs = {"normalize": normalize}
            if _CREATE_EMBEDDING_ACCEPTS_POOLING:
                kwargs["pooling_type"] = pooling
            response = llama.create_embedding(chunk, **kwargs)
            chunk_embeddings = [
                item["embedding"] for item in response["data"] if "embedding" in item
            ]
            embeddings.extend(chunk_embeddings)

        return embeddings


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
