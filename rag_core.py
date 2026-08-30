"""Core, UI-independent functions for the PDF retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass(frozen=True)
class SourcePage:
    filename: str
    page_number: int
    text: str


def extract_pdf_pages(files: Iterable[tuple[str, BinaryIO]]) -> list[SourcePage]:
    """Extract non-empty page text and retain source metadata."""
    pages: list[SourcePage] = []
    for filename, file_obj in files:
        reader = PdfReader(file_obj)
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(SourcePage(filename, index, text))
    return pages


def split_pages(
    pages: Sequence[SourcePage],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split pages into retrieval documents with file and page metadata."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for page in pages:
        documents.extend(
            splitter.create_documents(
                [page.text],
                metadatas=[{"source": page.filename, "page": page.page_number}],
            )
        )
    return documents


def format_context(documents: Sequence[Document]) -> str:
    """Create a numbered, attributable context block for the language model."""
    blocks = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "?")
        blocks.append(f"[Source {index}: {source}, page {page}]\n{document.page_content}")
    return "\n\n".join(blocks)


def source_labels(documents: Sequence[Document]) -> list[str]:
    """Return de-duplicated source labels in retrieval order."""
    labels: list[str] = []
    for document in documents:
        label = f"{document.metadata.get('source', 'unknown')} — page {document.metadata.get('page', '?')}"
        if label not in labels:
            labels.append(label)
    return labels


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the supplied document context. "
            "If the answer is not supported, say that it is not available in the uploaded documents. "
            "Cite supporting passages using [Source N]. Do not invent facts.",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)
