"""Request and response models of the HTTP API (validated by Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["vector", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(
        default=5, ge=1, le=10, description="how many chunks go into the prompt"
    )
    mode: Mode = Field(
        default="vector",
        description=(
            "vector: embedding search only. hybrid: vector + keyword fused with RRF. "
            "hybrid_rerank: hybrid, then an LLM reorders the candidates."
        ),
    )
    document: str | None = Field(
        default=None,
        max_length=64,
        description="search only this document by name (see GET /documents); null = all",
    )


class Source(BaseModel):
    id: int = Field(description="the [n] used in the answer")
    section: str
    title: str
    content: str
    # How each retrieval step saw this chunk. None when that step did not run.
    similarity: float | None = Field(default=None, description="cosine, vector mode only")
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = Field(default=None, description="0 to 10")


class AskResponse(BaseModel):
    answer: str
    answerable: bool = Field(
        description="false when the sources do not contain the answer"
    )
    sources: list[Source] = Field(
        description="only the sources the answer actually cites"
    )
    mode: Mode
    rerank_fallback: bool = Field(
        default=False, description="true when the rerank failed and the old order was kept"
    )


class DocumentInfo(BaseModel):
    name: str
    chunks: int
