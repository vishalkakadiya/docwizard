import re
import hashlib
from pathlib import Path
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embedding
from config import CHROMA_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def _dedupe_text(text: str) -> str:
    """Remove repeated blocks common in poorly-generated PDFs.

    Slides a 15-word window through the text. The first time a window
    reappears (at least 20 words later), we truncate — that marks
    the start of a duplicated passage.
    """
    # Ensure sentence boundaries have a space so "deeply.During" → "deeply. During"
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    words = text.split()
    if len(words) < 30:
        return text

    win, min_gap = 15, 20
    seen: dict = {}

    for i in range(len(words) - win + 1):
        key = tuple(words[i:i + win])
        if key in seen and (i - seen[key]) >= min_gap:
            return ' '.join(words[:i])
        seen.setdefault(key, i)

    return text

COSINE = {"hnsw:space": "cosine"}

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME, metadata=COSINE)

def clear_collection() -> int:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata=COSINE)
    count = collection.count()
    client.delete_collection(COLLECTION_NAME)
    client.get_or_create_collection(COLLECTION_NAME, metadata=COSINE)  # recreate empty
    return count

def ingest_pdf(pdf_path: str) -> int:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Deduplicate repeated sentences before chunking (common in low-quality PDFs)
    for page in pages:
        page.page_content = _dedupe_text(page.page_content)

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
