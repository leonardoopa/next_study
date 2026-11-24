"""Load a golden set: questions with the articles (chunks) that answer them."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GoldenItem:
    id: str
    category: str
    question: str
    expected: tuple[str, ...]  # article labels such as "Art. 18"; any of them counts as a hit
    answerable: bool
    note: str = ""


def load_golden(path: Path) -> list[GoldenItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            GoldenItem(
                id=raw["id"],
                category=raw["category"],
                question=raw["question"],
                expected=tuple(raw["expected"]),
                answerable=raw["answerable"],
                note=raw.get("note", ""),
            )
        )
    return items
