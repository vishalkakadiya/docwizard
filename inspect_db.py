"""Quick ChromaDB inspector — run with: python inspect_db.py"""
import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
col = client.get_collection(COLLECTION_NAME)

total = col.count()
print(f"Collection : {COLLECTION_NAME}")
print(f"Total chunks: {total}")

if total == 0:
    print("No data yet — ingest a PDF first.")
else:
    # Show unique source files
    all_meta = col.get(include=["metadatas"])["metadatas"]
    files = sorted({m["filename"] for m in all_meta})
    print(f"Source files: {files}")

    # Preview first 5 chunks
    preview = col.get(limit=5, include=["documents", "metadatas"])
    print("\n--- First 5 chunks ---")
    for i, (doc, meta) in enumerate(zip(preview["documents"], preview["metadatas"]), 1):
        print(f"\n[{i}] {meta['filename']} | page {meta['page']}")
        print(doc[:300])
        print("...")
