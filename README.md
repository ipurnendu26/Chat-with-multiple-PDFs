# Chat with PDF using OpenAI GPT

This project is a Streamlit application that allows users to chat with PDF documents using OpenAI's GPT model. Users can upload PDF files, process them, and then ask questions about the content of these documents.

## Features

- PDF text extraction
- Text chunking for efficient processing
- Vector store creation using FAISS
- Question answering using OpenAI's GPT-4 model
- User-friendly interface with Streamlit

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`)

3. Use the sidebar to upload PDF files and process them

4. Once processed, ask questions about the PDF content in the main chat interface

## How it works

1. The app extracts text from uploaded PDF files
2. The extracted text is split into smaller chunks
3. These chunks are embedded and stored in a FAISS vector store
4. When a user asks a question, the app searches for relevant chunks in the vector store
5. The relevant chunks and the user's question are sent to the GPT-4 model
6. The model generates a response based on the provided context and question

## Dependencies

- langchain
- openai
- streamlit
- tiktoken
- unstructured
- pdf2image
- pdfminer
- PyPDF2
- langchain-community
- python-dotenv
- faiss-cpu

## Note

Ensure you have a valid OpenAI API key with access to the GPT-4 model. The application uses the `gpt-4` model for question answering and the `text-embedding-ada-002` model for text embeddings.

