import io

import pytest
from pypdf import PdfWriter

from rag_core import SourcePage, extract_pdf_pages, format_context, source_labels, split_pages


def test_extract_pdf_without_text_returns_empty():
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(buffer)
    buffer.seek(0)

    assert extract_pdf_pages([("blank.pdf", buffer)]) == []


def test_split_pages_preserves_metadata():
    pages = [SourcePage("guide.pdf", 3, "A reliable retrieval system keeps source metadata.")]
    documents = split_pages(pages, chunk_size=80, chunk_overlap=10)

    assert documents
    assert documents[0].metadata == {"source": "guide.pdf", "page": 3}


def test_context_and_labels_are_attributable():
    documents = split_pages(
        [
            SourcePage("a.pdf", 1, "First source."),
            SourcePage("b.pdf", 2, "Second source."),
        ],
        chunk_size=80,
        chunk_overlap=10,
    )

    context = format_context(documents)
    assert "[Source 1: a.pdf, page 1]" in context
    assert source_labels(documents) == ["a.pdf — page 1", "b.pdf — page 2"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_chunk_settings_raise(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        split_pages([SourcePage("a.pdf", 1, "text")], chunk_size, chunk_overlap)
