# Multi-PDF Retrieval-Augmented Question Answering

A Streamlit application that turns one or more PDF documents into a searchable knowledge base. It extracts and chunks document text, creates OpenAI embeddings, stores them in FAISS, and uses retrieved context to answer user questions.

## Why this project matters

Long documents are difficult to search manually. This project demonstrates an end-to-end retrieval-augmented generation workflow that separates document ingestion, semantic retrieval, prompt construction, and answer generation.

## Capabilities

- Upload and process multiple PDF documents
- Extract text with PyPDF2
- Split long content into overlapping semantic chunks
- Build a local FAISS vector index
- Retrieve relevant passages for each question
- Generate context-grounded answers through an OpenAI chat model
- Provide an interactive Streamlit interface

## Architecture

```text
PDF documents
    -> text extraction
    -> recursive chunking
    -> OpenAI embeddings
    -> FAISS vector index
    -> similarity retrieval
    -> context-aware QA prompt
    -> generated answer
```

## Technology stack

- Python
- Streamlit
- LangChain
- OpenAI API
- FAISS
- PyPDF2

## Local setup

1. Clone the repository and enter the project directory.

   ```bash
   git clone https://github.com/ipurnendu26/Chat-with-multiple-PDFs.git
   cd Chat-with-multiple-PDFs
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example configuration and add your own API key.

   ```bash
   cp .env.example .env
   ```

5. Start the application.

   ```bash
   streamlit run app.py
   ```

## Security and data handling

- Never commit API keys or a populated `.env` file.
- Uploaded documents and generated FAISS indexes are excluded from version control.
- Use non-sensitive sample documents for demonstrations.
- Rotate a credential immediately if it has ever been committed to Git history.

## Current limitations

- Retrieval quality depends on PDF extraction quality and chunking strategy.
- Scanned PDFs require an OCR stage that is not currently included.
- The application does not yet provide formal retrieval evaluation, authentication, or persistent multi-user storage.
- Generated answers should be verified against the cited source documents.

## Planned engineering improvements

- Add retrieval evaluation using recall@k and answer-faithfulness metrics
- Add source-page citations in responses
- Add automated tests and continuous integration
- Add OCR support for scanned PDFs
- Upgrade to current LangChain package APIs
- Add containerized deployment

## Author

**Purnendu Kale**  
[LinkedIn](https://www.linkedin.com/in/purnendukale/) · [GitHub](https://github.com/ipurnendu26)
