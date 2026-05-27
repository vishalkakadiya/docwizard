import requests
from config import OLLAMA_BASE_URL, EMBED_MODEL

def get_embedding(text: str, task_type: str = "") -> list[float]:
    """Embed text using Ollama.

    nomic-embed-text performs significantly better when you tell it
    whether it is embedding a *query* or a *document*:
      - task_type="search_query"    → used when embedding user questions
      - task_type="search_document" → used when ingesting PDF chunks
    Passing an empty string disables the prefix (plain embedding).
    """
    prompt = f"{task_type}: {text}" if task_type else text
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": prompt}
    )
    response.raise_for_status()
    return response.json()["embedding"]
