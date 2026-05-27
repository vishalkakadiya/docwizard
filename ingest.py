import hashlib
from pathlib import Path
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embedding
from config import CHROMA_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

def clear_collection() -> int:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    count = collection.count()
    client.delete_collection(COLLECTION_NAME)
    client.get_or_create_collection(COLLECTION_NAME)  # recreate empty
    return count

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
