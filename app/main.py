from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .llm import EmbeddingModel, get_embedding_model
from .schemas import EmbeddingData, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _normalize_inputs(raw_input: Sequence[str] | str) -> List[str]:
    if isinstance(raw_input, str):
        return [raw_input]
    return list(raw_input)


async def _warm_model(model: EmbeddingModel) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, model.embed, ["warmup"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = get_embedding_model()

    try:
        await _warm_model(model)
        logger.info("Embedding model loaded successfully.")
    except Exception as exc:
        logger.warning("Model warmup failed: %s", exc)

    yield


app = FastAPI(
    title="Prime Radiant Embedding Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model_file}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    inputs = _normalize_inputs(request.input)

    try:
        embeddings = get_embedding_model().embed(
            inputs,
            normalize=request.normalize,
            batch_size=request.batch_size,
        )
    except Exception as exc:  # pragma: no cover - defensive, ensures clean error
        logger.exception("Failed to generate embeddings.")
        raise HTTPException(status_code=500, detail=f"Embedding failure: {exc}") from exc

    data = [
        EmbeddingData(index=index, embedding=vector)
        for index, vector in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=f"{settings.model_repo_id}:{settings.model_file}",
    )


if __name__ == "__main__":
    # Allow `python -m app.main` to honor configured host/port without extra flags.
    import uvicorn

    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
    )
