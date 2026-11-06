import pytest
from conftest import FIXTURES

from server.parser import location, parse_document, parse_isolated, split_block


@pytest.mark.parametrize(
    "name",
    [
        "shipping.pdf",
        "returns.docx",
        "launch.pptx",
        "metrics.xlsx",
        "products.csv",
        "policy.txt",
        "onboarding.md",
        "incident.html",
    ],
)
def test_all_original_formats(name):
    parsed = parse_document(FIXTURES / name, name.rsplit(".", 1)[1])
    assert parsed["parts"]
    assert all(part["text"] and part["location"] for part in parsed["parts"])
    first = parsed["parts"][0]
    if name == "shipping.pdf":
        assert first["location"]["page"] == 1
    if name == "launch.pptx":
        assert first["location"]["slide"] == 1
    if name == "onboarding.md":
        assert first["location"]["heading"] == "Onboarding"
    if name == "metrics.xlsx":
        assert any(
            "Metric: Revenue; Value: 120000" in p["text"] and p["location"]["row"] == 2
            for p in parsed["parts"]
        )
    if name == "products.csv":
        assert any(
            "product: Beacon; region: West; inventory: 73" in p["text"] and p["location"]["row"] == 2
            for p in parsed["parts"]
        )


def test_chunk_size_and_context_overlap():
    text = " ".join(f"word{i:04}" for i in range(500))
    parts = split_block(text, location("line"), line_numbers=True)
    assert len(parts) > 1
    assert all(len(p["text"]) <= 900 for p in parts)
    assert set(parts[0]["text"].split()) & set(parts[1]["text"].split())
    assert parts[0]["location"]["row"] == 1


def test_parser_subprocess_preserves_result():
    parsed = parse_isolated(FIXTURES / "onboarding.md", "md")
    assert "Priya Shah" in parsed["parts"][0]["text"]


def test_empty_and_broken_files():
    with pytest.raises(ValueError, match="No searchable text"):
        parse_document(FIXTURES / "empty.txt", "txt")
    with pytest.raises(Exception):
        parse_document(FIXTURES / "broken.pdf", "pdf")


def test_scanned_pdf_requires_ocr():
    with pytest.raises(ValueError, match="OCR"):
        parse_document(FIXTURES / "scanned.pdf", "pdf")


def test_markdown_policy_scope_survives_section_splitting():
    from server.parser import markdown_parts

    parts = markdown_parts(
        "# Policy\nApplies after 2026-09-01.\n## Warranty\nCoverage lasts 24 months.\n## Returns\nReturn within 30 days."
    )
    warranty = next(p for p in parts if p["location"]["heading"] == "Warranty")
    assert "Applies after 2026-09-01" in warranty["text"]
    assert "24 months" in warranty["text"]
