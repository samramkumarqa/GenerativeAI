import io

import pytest
from openpyxl import load_workbook

from fakes import FakeLLM, HashEmbedder
from tcgen import config
from tcgen.export import build_workbook
from tcgen.loaders import DocumentError, read_upload, read_url
from tcgen.pipeline import extract_requirements, generate_test_cases, traceability
from tcgen.rag import DocumentIndex
from tcgen.schemas import Requirement


def test_sample_pdf_has_requirements(sample_text):
    assert "FR-01" in sample_text and "NFR-04" in sample_text


def test_rejects_empty_and_unknown_files():
    with pytest.raises(DocumentError):
        read_upload("a.txt", b"too short")
    with pytest.raises(DocumentError):
        read_upload("a.exe", b"x" * 500)


@pytest.mark.parametrize("url", ["ftp://example.com/x", "http://localhost/brd", "http://127.0.0.1/x", "nonsense"])
def test_url_loader_blocks_non_public(url):
    with pytest.raises(DocumentError):
        read_url(url)


def test_extraction_dedupes_and_assigns_ids(sample_text):
    reqs = extract_requirements(sample_text, FakeLLM())
    assert [r.id for r in reqs] == ["FR-01", "REQ-001"]
    assert reqs[0].source_excerpt == "below INR 199"


def test_retrieval_finds_relevant_chunk(sample_text):
    index = DocumentIndex(sample_text, HashEmbedder())
    context = index.context_for(["promo code discount alphanumeric"], k=1)
    assert "promo code" in context.lower()


def _reqs(n):
    return [
        Requirement(id=f"R{i}", title=f"Req {i}", description="d", category="Functional", priority="High")
        for i in range(1, n + 1)
    ]


def test_generation_batches_filters_and_numbers(sample_text):
    llm = FakeLLM()
    reqs = _reqs(5)
    tcs = generate_test_cases(reqs, DocumentIndex(sample_text, HashEmbedder()), llm, ["Functional"], 2)

    assert len(llm.calls) == -(-5 // config.GENERATION_BATCH_SIZE)
    assert [tc.requirement_id for tc in tcs] == ["R1", "R2", "R3", "R4", "R5"]  # hallucinated one dropped, order restored
    assert [tc.id for tc in tcs] == ["TC-001", "TC-002", "TC-003", "TC-004", "TC-005"]
    assert tcs[0].steps == ["Open cart", "Tap Checkout"]  # blank step removed
    assert "2 test cases for EACH requirement" in llm.calls[0][0]


def test_traceability_flags_uncovered(sample_text):
    reqs = _reqs(2)
    tcs = generate_test_cases(reqs[:1], DocumentIndex(sample_text, HashEmbedder()), FakeLLM(), ["Functional"])
    rows = traceability(reqs, tcs)
    assert [r["Coverage"] for r in rows] == ["Covered", "Not covered"]
    assert rows[0]["Test Case IDs"] == "TC-001"


def test_workbook_structure(sample_text):
    reqs = _reqs(3)
    tcs = generate_test_cases(reqs[:2], DocumentIndex(sample_text, HashEmbedder()), FakeLLM(), ["Functional"])
    wb = load_workbook(io.BytesIO(build_workbook(reqs, tcs, "sample.pdf", "fake")))

    assert wb.sheetnames == ["Summary", "Test Cases", "Traceability", "Requirements"]
    cases = wb["Test Cases"]
    assert cases["A2"].value == "TC-001"
    assert cases["G2"].value == "1. Open cart\n2. Tap Checkout"
    assert cases["J2"].value == "Not Run"
    assert cases.data_validations.dataValidation  # status dropdown present
    assert wb["Traceability"]["H4"].value == "Not covered"
    assert wb["Summary"]["B9"].value == "=IF(B6=0,0,B8/B6)"
    assert wb["Summary"]["A6"].value == "Requirements" and wb["Summary"]["A8"].value == "Requirements covered"
