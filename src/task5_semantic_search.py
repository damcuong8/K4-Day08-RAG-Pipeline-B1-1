"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import json
import os
from pathlib import Path

import numpy as np

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "ecommerce_support_docs"

# Thử import cấu hình và hàm embedding từ Task 4 để đảm bảo cùng 1 model & vector store
try:
    from src.task4_chunking_indexing import (
        EMBEDDING_MODEL,
        get_embedding_model,
        embed_texts,
        INDEX_DIR,
        CHUNKS_PATH,
        EMBEDDINGS_PATH,
    )
except (ImportError, Exception):
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    _model = None

    def get_embedding_model():
        """Lazy load embedding model."""
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
        return _model

    def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
        model = get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    INDEX_DIR = Path(__file__).parent.parent / "rag_index"
    CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
    EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"


def get_collection():
    """Get persistent ChromaDB collection."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def generate_hypothetical_document(query: str) -> str:
    """
    Kỹ thuật HyDE (Hypothetical Document Embeddings):
    Sinh một văn bản/câu trả lời giả định từ query trước khi embed để cải thiện độ khớp ngữ nghĩa.
    """
    return (
        f"Văn bản quy định chi tiết giải đáp về câu hỏi: '{query}'. "
        f"Theo quy định của pháp luật Việt Nam và chính sách liên quan đến {query}, "
        f"tổ chức, cá nhân có nghĩa vụ và quyền hạn thực hiện đúng quy trình, trình tự, thủ tục."
    )


def _search_numpy_index(query_vec: list[float], top_k: int) -> list[dict]:
    """Tìm kiếm trên NumPy index nếu Task 4 tạo rag_index/."""
    if not (CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists()):
        return []

    try:
        chunks = []
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        embeddings = np.load(EMBEDDINGS_PATH)
        q_vec = np.array(query_vec, dtype=np.float32)

        # Cosine similarity
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(embeddings, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            c = chunks[idx]
            results.append({
                "content": c.get("content", ""),
                "score": round(float(scores[idx]), 4),
                "metadata": c.get("metadata", {}),
            })
        return results
    except Exception:
        return []


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (Cosine Similarity) & HyDE.

    Args:
        query: Câu truy vấn của người dùng
        top_k: Số lượng kết quả tối đa
        use_hyde: Có áp dụng Hypothetical Document Embeddings hay không

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Nếu dùng HyDE, embed văn bản giả định thay vì query ngắn
    search_text = generate_hypothetical_document(query) if use_hyde else query

    # Embed query dùng hàm embed_texts của Task 4
    try:
        query_vecs = embed_texts([search_text], is_query=True)
        query_vector = query_vecs[0] if query_vecs else []
    except Exception:
        model = get_embedding_model()
        query_vector = model.encode(search_text).tolist()

    if not query_vector:
        return []

    # 1. Tra cứu trên NumPy vector store index của Task 4 nếu file index tồn tại
    numpy_results = _search_numpy_index(query_vector, top_k=top_k)
    if numpy_results:
        return numpy_results

    # 2. Tra cứu ChromaDB nếu collection có dữ liệu
    try:
        collection = get_collection()
        if collection.count() > 0:
            actual_k = min(top_k, collection.count())
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, dists):
                    score = max(0.0, 1.0 - float(dist))
                    output.append({
                        "content": doc,
                        "score": round(score, 4),
                        "metadata": meta or {},
                    })
                output.sort(key=lambda x: x["score"], reverse=True)
                return output[:top_k]
    except Exception:
        pass

    return []



if __name__ == "__main__":
    # Test
    results = semantic_search("quy định ban hành văn bản quy phạm pháp luật", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content'][:100]}...")


