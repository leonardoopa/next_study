from dataclasses import dataclass


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
class IngestSummary:
    """What an ingestion run did."""

    total: int
    new: int
    existing: int
    removed: int
