"""Splitter tests on a synthetic CV (the real one is personal and not in the repo)."""

from app.ingestion.cv_splitter import build_chunks

WIDE = "a" * 130  # a line as wide as the page: it must continue on the next line

TEXT = "\n".join(
    [
        "Fulano de Tal",
        "Engenheiro de Software",
        "contato@example.com",
        "RESUMO",
        WIDE,
        "fim.",
        "COMPETÊNCIAS TÉCNICAS",
        "Linguagens: Python",
        "EXPERIÊNCIA PROFISSIONAL",
        "Engenheiro Back-End set/2023 – out/2026",
        "Acme Corp | Recife, PE | promovido em abr/2025",
        "Fiz coisa A.",
        "Fiz coisa B.",
        "Estagiário jun/2023 – set/2023",
        "Beta Ltda | Recife, PE",
        "Fiz coisa C.",
        "PROJETOS",
        "Projeto Um (Python) — Faz algo.",
        "Projeto Dois (Go) — Faz outra coisa.",
    ]
)


def test_chunk_count_and_sections():
    chunks = build_chunks(TEXT)
    assert [c.section for c in chunks] == [
        "RESUMO",
        "COMPETÊNCIAS TÉCNICAS",
        "EXPERIÊNCIA PROFISSIONAL",
        "EXPERIÊNCIA PROFISSIONAL",
        "PROJETOS",
        "PROJETOS",
    ]


def test_preamble_with_contact_is_dropped():
    assert all("contato@example.com" not in c.text for c in build_chunks(TEXT))


def test_wrapped_line_is_joined():
    assert build_chunks(TEXT)[0].text == WIDE + " fim."


def test_job_title_has_role_company_and_period():
    job = build_chunks(TEXT)[2]
    assert job.title == "Engenheiro Back-End | Acme Corp | set/2023 – out/2026"


def test_job_text_does_not_repeat_the_title():
    job = build_chunks(TEXT)[2]
    assert job.text == "Recife, PE | promovido em abr/2025\nFiz coisa A.\nFiz coisa B."
    assert "Acme Corp" not in job.text


def test_job_without_extra_details_on_company_line():
    job = build_chunks(TEXT)[3]
    assert job.title == "Estagiário | Beta Ltda | jun/2023 – set/2023"
    assert job.text == "Recife, PE\nFiz coisa C."


def test_project_title_is_not_repeated_in_text():
    project = build_chunks(TEXT)[4]
    assert project.title == "Projeto Um"
    assert project.text == "(Python) — Faz algo."
