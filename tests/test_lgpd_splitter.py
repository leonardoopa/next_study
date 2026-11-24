"""LGPD splitter tests on a small synthetic page (the real one is not in the repo)."""

import pytest

from app.ingestion import lgpd_splitter
from app.ingestion.html_text import decode_html, html_to_lines
from app.ingestion.lgpd_splitter import build_lgpd_chunks

PAGE = """<html><head><meta charset="utf-8"><script>var x = "Art. 99 hacked";</script></head>
<body>
<p>LEI Nº 1, DE 2018</p>
<p>Texto compilado</p>
<p>Art. 1º antes do capítulo, deve ser ignorado.</p>
<p>CAPÍTULO I</p><p>DISPOSIÇÕES PRELIMINARES</p>
<p>Art. 1º Esta Lei dispõe sobre o tratamento de dados. (Redação dada pela Lei nº 2, de 2019)</p>
<p>Parágrafo único. As normas são de interesse nacional.</p>
<p>Art. 2º A disciplina tem como fundamentos:</p>
<p>I - o respeito à privacidade;</p>
<p><strike>II - texto antigo e revogado;</strike></p>
<p>II - a autodeterminação informativa.</p>
<p>CAPÍTULO II</p><p>DO TRATAMENTO</p>
<p>Seção I</p><p>Dos Requisitos</p>
<p>Art. 7º O tratamento somente poderá ser realizado:</p>
<p>I - mediante consentimento;</p>
<p>Art. 8º (VETADO).</p>
<p>Art. 9º Texto do nono.</p>
<p>Art. 5 7. (VETADO).</p>
<p>Art. 9-A. Primeira versão do 9-A.</p>
<p>Art. 9-A. Versão nova do artigo 9-A.</p>
<p>Brasília, 14 de agosto de 2018</p>
<p>Art. 100. Texto depois da assinatura.</p>
</body></html>"""
RAW = PAGE.encode("cp1252")  # declared utf-8 but cp1252 bytes, like the real page


@pytest.fixture
def chunks():
    return build_lgpd_chunks(RAW)


def by_title(chunks):
    return {c.title: c for c in chunks}


def test_decodes_cp1252_even_when_the_page_declares_utf8():
    assert decode_html(b"Art. 1\xba") == "Art. 1º"


def test_scripts_and_struck_text_are_dropped_but_nothing_is_executed():
    lines = html_to_lines(RAW)
    assert not any("hacked" in line for line in lines)
    assert not any("antigo" in line for line in lines)


def test_one_chunk_per_real_article_in_order(chunks):
    # Art. 8 and the typo'd "Art. 5 7." (= Art. 57) are vetoed, the text before CAPÍTULO I and
    # after "Brasília" is ignored, and Art. 9-A keeps only one (the newest) version.
    assert [c.title for c in chunks] == ["Art. 1", "Art. 2", "Art. 7", "Art. 9", "Art. 9-A"]


def test_the_newest_version_of_a_repeated_article_wins(chunks):
    assert by_title(chunks)["Art. 9-A"].text == "Versão nova do artigo 9-A."


def test_notes_are_removed_and_the_article_number_is_not_repeated_in_the_text(chunks):
    art1 = by_title(chunks)["Art. 1"]
    assert art1.text == (
        "Esta Lei dispõe sobre o tratamento de dados.\n"
        "Parágrafo único. As normas são de interesse nacional."
    )


def test_incisos_stay_with_their_article_and_revoked_text_is_gone(chunks):
    art2 = by_title(chunks)["Art. 2"].text
    assert "I - o respeito à privacidade;" in art2
    assert "autodeterminação" in art2
    assert "antigo" not in art2


def test_section_carries_chapter_and_subsection(chunks):
    sections = {c.title: c.section for c in chunks}
    assert sections["Art. 1"] == "CAPÍTULO I – DISPOSIÇÕES PRELIMINARES"
    assert sections["Art. 7"] == "CAPÍTULO II – DO TRATAMENTO > Seção I – Dos Requisitos"


def test_long_articles_are_split_and_each_part_repeats_the_caput(monkeypatch):
    monkeypatch.setattr(lgpd_splitter, "MAX_CHARS", 90)
    items = "".join(f"<p>{n} - item numero {n} da lista longa;</p>" for n in range(1, 9))
    page = (
        "<p>CAPÍTULO I</p><p>TESTE</p><p>Art. 5º Para os fins desta Lei, considera-se:</p>"
        + items
    ).encode("cp1252")
    parts = build_lgpd_chunks(page)

    assert len(parts) > 1
    assert parts[0].title == f"Art. 5 (parte 1/{len(parts)})"
    assert parts[1].title == f"Art. 5 (parte 2/{len(parts)})"
    assert parts[1].text.startswith("Para os fins desta Lei, considera-se:\n[...]\n")
    all_text = "\n".join(p.text for p in parts)
    assert all(f"item numero {n} " in all_text for n in range(1, 9))  # nothing was lost


def test_a_later_version_wins_even_when_its_number_is_lower_than_the_previous_article():
    page = (
        "<p>CAPÍTULO I</p><p>TESTE</p>"
        "<p>Art. 55-A. Primeira versão.</p>"
        "<p>Art. 55-K. Texto antigo do K.</p>"
        "<p>Art. 55-A. Versão final do A.</p>"
        "<p>Art. 55-K. Versão final do K.</p>"
    ).encode("cp1252")
    texts = {c.title: c.text for c in build_lgpd_chunks(page)}
    assert texts == {"Art. 55-A": "Versão final do A.", "Art. 55-K": "Versão final do K."}


def test_chapter_title_skips_a_note_line_in_between():
    page = (
        "<p>CAPÍTULO IX</p><p>(Redação dada pela Lei nº 3, de 2020)</p><p>DA FISCALIZAÇÃO</p>"
        "<p>Art. 52. Texto.</p>"
    ).encode("cp1252")
    assert build_lgpd_chunks(page)[0].section == "CAPÍTULO IX – DA FISCALIZAÇÃO"


def test_typo_in_the_article_number_is_repaired_and_the_real_article_is_untouched():
    page = (
        "<p>CAPÍTULO I</p><p>TESTE</p><p>Art. 5º Definições reais.</p>"
        "<p>Art. 5 7. (VETADO).</p>"
    ).encode("cp1252")
    chunks = build_lgpd_chunks(page)
    assert [(c.title, c.text) for c in chunks] == [("Art. 5", "Definições reais.")]
