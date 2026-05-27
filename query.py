import requests
import chromadb
from embeddings import get_embedding
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, OLLAMA_BASE_URL, LLM_MODEL

COSINE = {"hnsw:space": "cosine"}

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME, metadata=COSINE)

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
