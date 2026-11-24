"""Structure of the LGPD golden set (no database needed)."""

import json
import re
from collections import Counter
from pathlib import Path

GOLDEN = Path(__file__).parent.parent / "eval" / "lgpd_golden.jsonl"
ITEMS = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
CATEGORIES = {"exato", "parafrase", "tema", "multi", "sem_resposta"}


def test_has_forty_questions_with_unique_ids():
    assert len(ITEMS) == 40
    assert len({q["id"] for q in ITEMS}) == 40


def test_distribution_by_category():
    assert Counter(q["category"] for q in ITEMS) == {
        "exato": 8, "parafrase": 14, "tema": 8, "multi": 5, "sem_resposta": 5,
    }  # fmt: skip


def test_every_item_has_the_expected_fields_and_a_known_category():
    for q in ITEMS:
        assert set(q) >= {"id", "category", "question", "expected", "answerable"}
        assert q["category"] in CATEGORIES
        assert q["question"].endswith("?")


def test_expected_articles_use_the_chunk_label_format():
    for q in ITEMS:
        for label in q["expected"]:
            assert re.fullmatch(r"Art\. \d+(-[A-Z])?", label), (q["id"], label)


def test_answerable_matches_having_expected_articles():
    for q in ITEMS:
        assert q["answerable"] == bool(q["expected"]), q["id"]
    assert all(not q["answerable"] for q in ITEMS if q["category"] == "sem_resposta")


def test_multi_questions_expect_more_than_one_article():
    assert all(len(q["expected"]) >= 2 for q in ITEMS if q["category"] == "multi")
