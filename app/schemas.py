from __future__ import annotations

from typing import List, Literal, Sequence, Union

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    input: Union[str, Sequence[str]]
    normalize: bool = Field(default=True)
    batch_size: int | None = Field(default=None, ge=1)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: Union[str, Sequence[str]]) -> Union[str, Sequence[str]]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("input must contain at least one string")
            if not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError("each item in input must be a non-empty string")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError("input must be a non-empty string")
        return value


class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]
    object: Literal["embedding"] = "embedding"


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    object: Literal["list"] = "list"
    usage: Usage = Field(default_factory=Usage)
