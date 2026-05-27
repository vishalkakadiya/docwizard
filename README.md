# DocWizard

A fully local **Retrieval-Augmented Generation (RAG)** system that lets you chat with your PDF documents. No cloud, no API keys, no data leaving your machine.

Built with ChromaDB, nomic-embed-text, and Ollama.

---

## What is RAG?

Large Language Models (LLMs) are trained on general knowledge — they know nothing about *your* documents. **RAG** solves this by retrieving relevant content from your files at query time and injecting it into the LLM's prompt as context, so the model answers based on your data rather than hallucinating.

```
PDF → chunk → embed → vector store     (ingestion)
Question → embed → similarity search → top chunks → LLM → answer   (query)
```

DocWizard implements this pipeline entirely locally using Ollama.

---

## Architecture

### Ingestion pipeline

```
PDF file
  │
  ▼
PyPDFLoader          — extracts raw text page by page
  │
  ▼
_dedupe_text()       — removes duplicate text layers (common in low-quality PDFs)
  │
  ▼
RecursiveCharacterTextSplitter  — splits into 500-char chunks with 50-char overlap
  │                               overlap preserves context across chunk boundaries
  ▼
nomic-embed-text     — converts each chunk into a 768-dimensional vector
  │                    prefixed with "search_document:" for task-aware embedding
  ▼
ChromaDB             — persists vectors with cosine similarity index (HNSW)
                       MD5 hash of filename:page:index used as chunk ID to prevent duplicates
```

### Query pipeline

```
User question
  │
  ▼
nomic-embed-text     — embeds the question (prefixed "search_query:" for asymmetric retrieval)
  │
  ▼
ChromaDB ANN search  — approximate nearest-neighbour search over stored vectors
  │                    returns TOP_K=8 most semantically similar chunks
  ▼
Keyword re-rank      — lightweight second pass: boosts chunks that contain
  │                    the question's keywords, compensating for embedding model
  │                    limitations on topically similar documents
  ▼
Top 3 chunks         — only the most relevant chunks sent to the LLM
  │                    (small models degrade with too much context)
  ▼
llama3.2:1b          — generates a grounded answer from the retrieved context
```

### Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Distance metric | Cosine similarity | `nomic-embed-text` vectors are not unit-normalised (L2 norm ≈ 19–22), so Euclidean distance gives wrong rankings |
| Task-aware embeddings | `search_query:` / `search_document:` prefixes | `nomic-embed-text` is an asymmetric model — query and document embeddings live in different sub-spaces without prefixes |
| Hybrid retrieval | Semantic (ChromaDB) + keyword re-rank | Compensates for quantised small embedding models that cluster similar topics too closely |
| Retrieval width vs LLM context | `TOP_K=8` retrieve, `LLM_TOP_K=3` send | Wide retrieval improves recall; narrow LLM context prevents 1B models from losing the relevant chunk |
| Module separation | `ingest.py`, `query.py`, `embeddings.py` have zero Streamlit coupling | Clean path to swap the UI for FastAPI without touching any business logic |

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

Pull the two required models:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2:1b        # LLM for answers
```

---

## Setup

**1. Clone the repo**

```bash
git clone git@github.com:vishalkakadiya/docwizard.git
cd docwizard
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload** — drag a PDF into the sidebar and click **Ingest PDFs**
2. **Ask** — type any question in the chat box
3. **Clear** — use **🗑️ Clear all chunks** in the sidebar to reset and ingest a new file

To inspect what is stored in ChromaDB:

```bash
python inspect_db.py
```

---

## Run tests

```bash
pytest tests/
```

All 7 tests are fully mocked — no Ollama or ChromaDB instance required.

---

## Project structure

```
docwizard/
├── app.py            # Streamlit UI (zero business logic)
├── config.py         # Single source of truth for all settings
├── embeddings.py     # Ollama embedding wrapper (task-type aware)
├── ingest.py         # PDF loading, deduplication, chunking, vector storage
├── query.py          # Semantic search, keyword re-rank, LLM answer generation
├── inspect_db.py     # Dev utility: inspect ChromaDB contents
├── requirements.txt
└── tests/            # Unit tests (mocked, no Ollama needed)
```

---

## Configuration

All tuneable settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_MODEL` | `llama3.2:1b` | Ollama LLM for answers |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `8` | Chunks retrieved from ChromaDB (semantic search) |
| `LLM_TOP_K` | `3` | Chunks passed to the LLM after re-ranking |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) |
| Vector store | [ChromaDB](https://www.trychroma.com) (persistent, cosine similarity) |
| Embeddings | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama |
| LLM | [llama3.2:1b](https://ollama.com/library/llama3.2) via Ollama |
| PDF parsing | LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter` |
| Runtime | 100% local — no external API calls |
