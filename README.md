# DocWizard

Ask questions about your PDFs — powered by local Ollama. No cloud, no API keys.

## How it works

Upload a PDF → DocWizard chunks it, embeds it locally, and stores it in ChromaDB. Ask a question → it finds the relevant chunks and sends them to a local LLM to answer.

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

---

## Run tests

```bash
pytest tests/
```

---

## Project structure

```
docwizard/
├── app.py            # Streamlit UI
├── config.py         # Model names, chunk settings, DB path
├── embeddings.py     # Ollama embedding wrapper
├── ingest.py         # PDF loading, chunking, storing in ChromaDB
├── query.py          # Semantic search + keyword re-rank + LLM answer
├── inspect_db.py     # Dev utility: inspect what's stored in ChromaDB
├── requirements.txt
└── tests/            # Unit tests (mocked, no Ollama needed)
```

---

## Configuration

All tuneable settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_MODEL` | `llama3.2:1b` | Ollama LLM for answers |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `8` | Chunks retrieved from ChromaDB |
| `LLM_TOP_K` | `3` | Chunks actually sent to the LLM |
