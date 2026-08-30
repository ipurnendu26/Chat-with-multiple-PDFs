# Document Q&A with Cited Retrieval

A session-scoped retrieval-augmented generation application for asking grounded questions across multiple PDF documents. The application extracts page text, creates metadata-preserving chunks, retrieves relevant passages with FAISS, and produces answers with file-and-page citations.

## Why this project matters

The project demonstrates a practical RAG workflow while addressing common prototype risks:

- uploaded documents and vector indexes remain in the current application session;
- generated answers are constrained to retrieved context;
- every retrieved passage retains its source filename and page number;
- missing evidence produces an explicit “not available” response;
- secrets and generated indexes are excluded from version control.

## Architecture

1. **Ingestion:** `pypdf` extracts text and page metadata.
2. **Chunking:** LangChain's recursive splitter creates overlapping documents.
3. **Embedding:** OpenAI embeddings represent document chunks.
4. **Retrieval:** FAISS returns the four most relevant chunks.
5. **Generation:** a low-temperature chat model answers from context only.
6. **Attribution:** the UI displays the retrieved file and page references.

## Technology

Python, Streamlit, LangChain, OpenAI, FAISS, pypdf, pytest, Docker and GitHub Actions.

## Run locally

```bash
git clone https://github.com/ipurnendu26/Chat-with-multiple-PDFs.git
cd Chat-with-multiple-PDFs
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Set `OPENAI_API_KEY` in `.env`. Optionally set `OPENAI_MODEL`; the default is `gpt-4o-mini`.

## Test

```bash
pytest -q
```

The tests cover PDF ingestion behavior, chunk validation and preservation of citation metadata. CI runs them on every push and pull request.

## Docker

```bash
docker build -t cited-pdf-rag .
docker run --rm -p 8501:8501 --env-file .env cited-pdf-rag
```

## Security and privacy

- Never commit `.env` or API keys.
- Rotate a credential immediately if it is exposed.
- This demo sends retrieved text to the configured model provider.
- Do not upload confidential documents unless the deployment and provider settings meet your requirements.
- The application does not persist FAISS indexes to disk.

## Limitations

Image-only PDFs require an OCR stage. Retrieval quality depends on document structure, embeddings and the selected model. This project is a portfolio implementation, not a document-compliance system.

## License

MIT
