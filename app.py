"""Streamlit interface for private, session-scoped PDF question answering."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_core import PROMPT, extract_pdf_pages, format_context, source_labels, split_pages

MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_MODEL = "gpt-4o-mini"

load_dotenv()
st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")


def get_api_key() -> str | None:
    """Read the API key without displaying or persisting it."""
    try:
        secret = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret = None
    return secret or os.getenv("OPENAI_API_KEY")


def process_documents(uploaded_files, api_key: str) -> None:
    invalid = [f.name for f in uploaded_files if f.size > MAX_FILE_BYTES]
    if invalid:
        raise ValueError("Each PDF must be 20 MB or smaller: " + ", ".join(invalid))

    pages = extract_pdf_pages((f.name, f) for f in uploaded_files)
    if not pages:
        raise ValueError("No extractable text was found. Scanned PDFs require OCR before upload.")

    documents = split_pages(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
    st.session_state.document_count = len(uploaded_files)
    st.session_state.chunk_count = len(documents)


def answer(question: str, api_key: str) -> tuple[str, list[str]]:
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        raise ValueError("Process at least one PDF before asking a question.")

    documents = vector_store.similarity_search(question, k=4)
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    model = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    messages = PROMPT.format_messages(context=format_context(documents), question=question)
    response = model.invoke(messages)
    return str(response.content), source_labels(documents)


st.title("Document Q&A with cited retrieval")
st.caption(
    "Upload PDFs, build a session-scoped FAISS index, and receive answers grounded "
    "in retrieved pages. Uploaded content and indexes are not committed to the repository."
)

api_key = get_api_key()
if not api_key:
    st.warning("Set OPENAI_API_KEY in your environment or Streamlit secrets before processing documents.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 20 MB per file.",
    )
    if st.button("Process documents", type="primary", disabled=not uploads or not api_key):
        try:
            with st.spinner("Extracting and indexing documents..."):
                process_documents(uploads, api_key)
            st.success(
                f"Indexed {st.session_state.document_count} document(s) "
                f"into {st.session_state.chunk_count} chunks."
            )
        except Exception as exc:
            st.error(f"Unable to process the documents: {exc}")

question = st.text_input("Ask a question about the uploaded documents")
if question:
    if not api_key:
        st.error("OPENAI_API_KEY is not configured.")
    else:
        try:
            with st.spinner("Retrieving supporting passages..."):
                response, sources = answer(question, api_key)
            st.subheader("Answer")
            st.write(response)
            st.subheader("Retrieved sources")
            for source in sources:
                st.markdown(f"- {source}")
        except Exception as exc:
            st.error(str(exc))
