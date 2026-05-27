import re
import requests
import chromadb
from embeddings import get_embedding
from config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, LLM_TOP_K, OLLAMA_BASE_URL, LLM_MODEL

COSINE = {"hnsw:space": "cosine"}

# Common English words that carry no meaning for matching
_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
    'do', 'did', 'will', 'would', 'could', 'should', 'tell', 'me', 'something',
    'about', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'give',
    'explain', 'describe', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by',
    'from', 'and', 'or', 'but', 'not', 'you', 'your', 'it', 'its', 'this',
    'that', 'i', 'my', 'we', 'us', 'can', 'into', 'more', 'some', 'any',
}

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME, metadata=COSINE)

def _keyword_rerank(question: str, chunks: list[str], top_k: int) -> list[str]:
    """Hybrid re-rank: sort semantically-retrieved chunks by keyword overlap.

    nomic-embed-text (small/quantized) ranks similar topics very close together.
    A lightweight keyword boost ensures the chunk that literally contains the
    query terms floats to the top before being sent to the LLM.
    """
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]

    if not keywords:
        return chunks[:top_k]

    def score(chunk: str) -> int:
        lower = chunk.lower()
        return sum(1 for kw in keywords if kw in lower)

    # Stable sort: original semantic rank breaks ties
    ranked = sorted(range(len(chunks)), key=lambda i: score(chunks[i]), reverse=True)
    return [chunks[i] for i in ranked[:top_k]]

def search(question: str) -> list[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    embedding = get_embedding(question, task_type="search_query")
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(TOP_K, collection.count())
    )
    return results["documents"][0]

def answer(question: str) -> str:
    chunks = search(question)
    if not chunks:
        return "No documents have been ingested yet. Please upload and ingest a PDF first."

    # Step 1: retrieve TOP_K chunks semantically (wide net for recall)
    # Step 2: keyword re-rank so the most on-topic chunk rises to the top
    # Step 3: feed only LLM_TOP_K chunks — small models need tight context
    llm_chunks = _keyword_rerank(question, chunks, LLM_TOP_K)
    context = "\n\n".join(llm_chunks)

    prompt = f"""You are a document assistant. Read the context below and answer the question.
Use ONLY information from the context. Be concise and direct.

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
