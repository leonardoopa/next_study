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
