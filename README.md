# Docmind

Docmind is a lightweight document Q&A app built with Streamlit. It lets you upload PDFs, DOCX files, and TXT files, index them, and then ask natural-language questions grounded in the uploaded content.

The app uses a retrieval-augmented generation (RAG) pipeline with FAISS vector search, BM25 keyword retrieval, and a reranker so answers stay grounded in the provided documents and cite relevant sources.

## Features

- Upload and process PDF, DOCX, and TXT files
- Keep documents isolated by workspace
- Search using hybrid retrieval: dense vector search + BM25 + reranking
- Answer questions with source-aware citations such as file name and page number
- Follow-up question rewriting for more natural conversations
- Export chat transcript as a text file
<img width="958" height="456" alt="Screenshot 2026-10-14 031145" src="https://github.com/user-attachments/assets/a6008e45-0a82-43d5-914d-b2652529805c" />
<img width="958" height="452" alt="Screenshot 2026-10-14 024006" src="https://github.com/user-attachments/assets/cc743ccd-e8ba-43b2-9170-0a61fe9084b6" />
<img width="958" height="450" alt="Screenshot 2026-10-14 031124" src="https://github.com/user-attachments/assets/42461a57-29b8-47c7-8433-7dd84e513671" />

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face embeddings and reranking models
- Google Generative AI (Gemini)
- PyPDF and python-docx for document parsing

## Project Structure

- `app.py` — main Streamlit application
- `vector_stores/` — persisted FAISS indexes and workspace data
- `.env` — local environment variables such as your Google API key

## Requirements

- Python 3.10+
- A Google Gemini API key
- Internet access for model inference and package installation

## Setup

1. Clone the repository
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install streamlit python-dotenv pypdf python-docx langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-google-genai faiss-cpu rank-bm25 sentence-transformers
```

4. Create a `.env` file in the project root with your Gemini API key:

```env
GOOGLE_API_KEY=your_key_here
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## How to Use

1. Enter a workspace name in the sidebar.
2. Upload one or more PDF, DOCX, or TXT files.
3. Click `Process` to split, embed, and index the files.
4. Ask a question about the uploaded content in the chat box.
5. Review the answer and the source excerpts for grounding.

## Notes

- Each workspace stores its own index in `vector_stores/<workspace-name>`.
- Uploaded files larger than 20 MB are skipped.
- If a document has no readable text, the app will warn you and skip it.
- The app answers only from the uploaded document context; if the content is unavailable, it responds with a clear "I don't know based on the provided documents" style fallback.

## License

This project is a personal/document Q&A prototype and is provided as-is for local use and experimentation.
