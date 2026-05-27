# DocWizard — RAG System Design
Date: 2026-05-27

## Goal
Local RAG app: upload multiple PDFs, chunk + embed + store in ChromaDB, query via Streamlit chat UI, answered by local Ollama LLM.

## Stack
| Layer | Choice |
|---|---|
| UI | Streamlit |
| LLM | Ollama `llama3.2:1b` |
| Embeddings | Ollama `nomic-embed-text` |
| Vector DB | ChromaDB (local folder `chroma_db/`) |
| PDF parsing | `pypdf` via LangChain `PyPDFLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) |

## Project Structure
```
docWizard/
├── app.py          # Streamlit UI: file upload + chat
├── ingest.py       # PDF → chunks → embed → ChromaDB
├── query.py        # question → ChromaDB search → LLM → answer
├── embeddings.py   # Ollama embedding wrapper
├── config.py       # constants
├── requirements.txt
└── chroma_db/      # auto-created, persisted vector data
```

## Data Flow

### Ingestion
1. User uploads PDF(s) in Streamlit
2. `ingest.py` loads each PDF with `PyPDFLoader`
3. Text split into chunks (500 chars, 50 overlap) via `RecursiveCharacterTextSplitter`
4. Each chunk embedded via Ollama `nomic-embed-text`
5. Embeddings + metadata (filename, page number) stored in ChromaDB collection

### Query
1. User types question in Streamlit chat
2. Question embedded via Ollama `nomic-embed-text`
3. ChromaDB returns top 5 most similar chunks
4. Chunks assembled into a prompt with the question
5. Prompt sent to Ollama `llama3.2:1b`
6. Answer streamed back to Streamlit chat

## Key Decisions
- **No duplicate ingestion**: ChromaDB collection keyed by `filename+page` so re-uploading same PDF skips already-stored chunks.
- **Production path**: `ingest.py` and `query.py` have zero Streamlit imports — wrap them behind FastAPI endpoints when ready to scale.
- **Config-driven**: all model names, chunk sizes, DB path in `config.py` — easy to swap models.
