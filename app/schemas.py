from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document, ready to be embedded and stored.

    Plain data (DTO): produced by the splitters, consumed by the embedding and storage
    steps. frozen=True makes it immutable, so a chunk cannot be changed by accident
    while it moves through the pipeline.
    """

    section: str
    title: str
    text: str

    def for_embedding(self) -> str:
        """Text with context prefix, the version that will be embedded."""
        return f"{self.section} > {self.title}\n{self.text}"


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk plus everything the database needs to store it."""

    position: int  # order inside the document
    chunk: Chunk
    content_hash: str  # sha256 of the exact text that was embedded
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One hit of a similarity search."""

    section: str
    title: str
    content: str
    similarity: float  # cosine similarity: 1 = same direction, 0 = unrelated


@dataclass(frozen=True, slots=True)
class RankedChunk:
    """A hit from lexical or hybrid search, with where each search ranked it (1 = best)."""

    section: str
    title: str
    content: str
    score: float  # lexical: ts_rank; hybrid: RRF score. Not comparable across methods.
    vector_rank: int | None  # None when the vector search did not return it
    lexical_rank: int | None  # None when the keyword search did not return it


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """A candidate after the rerank step."""

    section: str
    title: str
    content: str
    original_rank: int  # position before reranking (1 = best)
    rerank_score: float | None  # 0 to 10; None when the rerank fell back to the old order


@dataclass(frozen=True, slots=True)
class RerankResult:
    chunks: list[RerankedChunk]
    used_fallback: bool  # True when the rerank failed and the original order was kept


class Passage(Protocol):
    """What the generation step needs from a retrieved chunk: any retrieval mode fits."""

    section: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk chosen for the prompt, with how each retrieval step saw it."""

    section: str
    title: str
    content: str
    similarity: float | None = None  # cosine; only set by the vector mode
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class Retrieval:
    chunks: list[RetrievedChunk]
    rerank_fallback: bool = False  # the rerank failed or timed out; original order kept


@dataclass(frozen=True, slots=True)
class Answer:
    """A validated answer: every id in `cited_ids` is a real source (1-based)."""

    text: str
    answerable: bool
    cited_ids: list[int]


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """What an ingestion run did."""

    total: int
    new: int
    existing: int
    removed: int
