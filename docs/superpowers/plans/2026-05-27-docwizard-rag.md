# DocWizard RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local RAG app where users upload multiple PDFs, which get chunked/embedded into ChromaDB, and can be queried through a Streamlit chat UI answered by a local Ollama LLM.

**Architecture:** Streamlit UI calls pure-Python modules (ingest, query, embeddings) with no UI coupling — enabling a future FastAPI wrapper without rewriting logic. ChromaDB persists embeddings locally; Ollama serves both embeddings (`nomic-embed-text`) and LLM (`llama3.2:1b`).

**Tech Stack:** Python 3.11+, Streamlit, ChromaDB, LangChain (PDF loader + splitter), Ollama REST API, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `config.py` | All constants — model names, chunk size, DB path |
| `embeddings.py` | Ollama embedding API wrapper |
| `ingest.py` | PDF → chunks → embeddings → ChromaDB |
| `query.py` | Question → ChromaDB search → Ollama LLM → answer |
| `app.py` | Streamlit UI — upload sidebar + chat interface |
| `requirements.txt` | Dependencies |
| `tests/test_embeddings.py` | Unit tests for embedding wrapper |
| `tests/test_ingest.py` | Unit tests for ingestion pipeline |
| `tests/test_query.py` | Unit tests for query pipeline |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `config.py`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.32.0
chromadb>=0.4.22
langchain>=0.1.0
langchain-community>=0.0.20
pypdf>=4.0.0
requests>=2.31.0
pytest>=8.0.0
```

- [ ] **Step 2: Create config.py**

```python
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "docwizard"
TOP_K = 5
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 4: Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

Expected: `nomic-embed-text` appears in `ollama list`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.py
git commit -m "feat: project setup and config"
```

---

## Task 2: Embeddings Wrapper

**Files:**
- Create: `embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embeddings.py`:

```python
from unittest.mock import patch, MagicMock

def test_get_embedding_returns_list_of_floats():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()

    with patch("embeddings.requests.post", return_value=mock_response) as mock_post:
        from embeddings import get_embedding
        result = get_embedding("hello world")

    mock_post.assert_called_once()
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embeddings.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'embeddings'`

- [ ] **Step 3: Implement embeddings.py**

```python
import requests
from config import OLLAMA_BASE_URL, EMBED_MODEL

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_embeddings.py -v
```

Expected: PASS — `test_get_embedding_returns_list_of_floats PASSED`

- [ ] **Step 5: Commit**

```bash
git add embeddings.py tests/test_embeddings.py
git commit -m "feat: add Ollama embedding wrapper"
```

---

## Task 3: Ingestion Pipeline

**Files:**
- Create: `ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:

```python
from unittest.mock import patch, MagicMock

def test_ingest_pdf_adds_new_chunks(tmp_path):
    fake_pdf = str(tmp_path / "test.pdf")

    mock_doc = MagicMock()
    mock_doc.page_content = "This is test content for the RAG system."
    mock_doc.metadata = {"page": 0}

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}

    with patch("ingest.PyPDFLoader") as mock_loader, \
         patch("ingest.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("ingest.get_collection", return_value=mock_collection):

        mock_loader.return_value.load.return_value = [mock_doc]
        from ingest import ingest_pdf
        count = ingest_pdf(fake_pdf)

    assert count >= 1
    mock_collection.add.assert_called()

def test_ingest_pdf_skips_duplicate_chunks(tmp_path):
    fake_pdf = str(tmp_path / "test.pdf")

    mock_doc = MagicMock()
    mock_doc.page_content = "Duplicate content."
    mock_doc.metadata = {"page": 0}

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": ["existing-id"]}

    with patch("ingest.PyPDFLoader") as mock_loader, \
         patch("ingest.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("ingest.get_collection", return_value=mock_collection):

        mock_loader.return_value.load.return_value = [mock_doc]
        from ingest import ingest_pdf
        count = ingest_pdf(fake_pdf)

    assert count == 0
    mock_collection.add.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'ingest'`

- [ ] **Step 3: Implement ingest.py**

```python
import hashlib
from pathlib import Path
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import get_embedding
from config import CHROMA_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

def ingest_pdf(pdf_path: str) -> int:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    collection = get_collection()
    added = 0

    for i, chunk in enumerate(chunks):
        filename = Path(pdf_path).name
        page = chunk.metadata.get("page", 0)
        chunk_id = hashlib.md5(f"{filename}:{page}:{i}".encode()).hexdigest()

        if collection.get(ids=[chunk_id])["ids"]:
            continue

        embedding = get_embedding(chunk.page_content)
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{"filename": filename, "page": page}]
        )
        added += 1

    return added
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: PASS — both tests pass.

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: add PDF ingestion pipeline"
```

---

## Task 4: Query Pipeline

**Files:**
- Create: `query.py`
- Create: `tests/test_query.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_query.py`:

```python
from unittest.mock import patch, MagicMock

def test_search_returns_chunks():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {"documents": [["chunk1", "chunk2"]]}

    with patch("query.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("query.get_collection", return_value=mock_collection):
        from query import search
        result = search("What is this about?")

    assert isinstance(result, list)
    assert result == ["chunk1", "chunk2"]

def test_search_returns_empty_when_no_documents():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("query.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("query.get_collection", return_value=mock_collection):
        from query import search
        result = search("What is this about?")

    assert result == []

def test_answer_returns_llm_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The answer is 42."}
    mock_response.raise_for_status = MagicMock()

    with patch("query.search", return_value=["relevant context here"]), \
         patch("query.requests.post", return_value=mock_response):
        from query import answer
        result = answer("What is the answer?")

    assert result == "The answer is 42."

def test_answer_returns_fallback_when_no_docs():
    with patch("query.search", return_value=[]):
        from query import answer
        result = answer("What is this about?")

    assert "No documents" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_query.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'query'`

- [ ] **Step 3: Implement query.py**

```python
import requests
import chromadb
from embeddings import get_embedding
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, OLLAMA_BASE_URL, LLM_MODEL

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

def search(question: str) -> list[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(TOP_K, collection.count())
    )
    return results["documents"][0]

def answer(question: str) -> str:
    chunks = search(question)
    if not chunks:
        return "No documents have been ingested yet. Please upload and ingest a PDF first."

    context = "\n\n".join(chunks)
    prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_query.py -v
```

Expected: PASS — all 4 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all 7 tests across 3 files pass.

- [ ] **Step 6: Commit**

```bash
git add query.py tests/test_query.py
git commit -m "feat: add query pipeline with LLM answer generation"
```

---

## Task 5: Streamlit UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Implement app.py**

```python
import streamlit as st
import tempfile
import os
from ingest import ingest_pdf
from query import answer

st.set_page_config(page_title="DocWizard", page_icon="📄")
st.title("DocWizard")
st.caption("Ask questions about your PDFs — powered by local Ollama")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded_files and st.button("Ingest PDFs"):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                count = ingest_pdf(tmp_path)
            os.unlink(tmp_path)
            st.success(f"{uploaded_file.name}: {count} new chunks added")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

- [ ] **Step 2: Start the app**

```bash
streamlit run app.py
```

Expected: browser opens at `http://localhost:8501` showing "DocWizard" title and a sidebar.

- [ ] **Step 3: Manual test — ingest a PDF**

1. Upload any PDF using the sidebar file uploader
2. Click "Ingest PDFs"
3. Expected: green success message showing chunk count (e.g., `report.pdf: 42 new chunks added`)

- [ ] **Step 4: Manual test — query**

1. Type a question in the chat input relevant to the PDF you uploaded
2. Expected: an answer appears in the chat referencing content from the PDF

- [ ] **Step 5: Manual test — duplicate guard**

1. Click "Ingest PDFs" again with the same file
2. Expected: success message shows `0 new chunks added`

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit chat UI"
```

---

## Done

Run the full test suite one last time:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_embeddings.py::test_get_embedding_returns_list_of_floats PASSED
tests/test_ingest.py::test_ingest_pdf_adds_new_chunks PASSED
tests/test_ingest.py::test_ingest_pdf_skips_duplicate_chunks PASSED
tests/test_query.py::test_search_returns_chunks PASSED
tests/test_query.py::test_search_returns_empty_when_no_documents PASSED
tests/test_query.py::test_answer_returns_llm_response PASSED
tests/test_query.py::test_answer_returns_fallback_when_no_docs PASSED
7 passed
```
