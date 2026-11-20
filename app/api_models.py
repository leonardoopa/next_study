"""Request and response models of the HTTP API (validated by Pydantic)."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10, description="how many chunks to retrieve")


class Source(BaseModel):
    id: int = Field(description="the [n] used in the answer")
    section: str
    title: str
    content: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    answerable: bool = Field(
        description="false when the sources do not contain the answer"
    )
    sources: list[Source] = Field(
        description="only the sources the answer actually cites"
    )
