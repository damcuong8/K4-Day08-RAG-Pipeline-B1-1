"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import os
from pathlib import Path

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "ecommerce_support_docs"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None


def get_embedding_model():
    """Lazy load embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


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
    # Mở rộng ngữ nghĩa giả định theo domain pháp luật & thương mại điện tử
    hypothetical_doc = (
        f"Văn bản quy định chi tiết giải đáp về câu hỏi: '{query}'. "
        f"Theo quy định của pháp luật Việt Nam và chính sách thương mại điện tử liên quan đến {query}, "
        f"tổ chức, cá nhân có nghĩa vụ và quyền hạn thực hiện đúng quy trình, trình tự, thủ tục."
    )
    return hypothetical_doc


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
    try:
        collection = get_collection()
        if collection.count() == 0:
            return []
    except Exception:
        return []

    # Nếu dùng HyDE, embed văn bản giả định thay vì query ngắn
    search_text = generate_hypothetical_document(query) if use_hyde else query

    model = get_embedding_model()
    query_vector = model.encode(search_text).tolist()

    actual_k = min(top_k, collection.count())
    if actual_k <= 0:
        return []

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
            # ChromaDB cosine distance d -> similarity score = max(0.0, 1.0 - d)
            score = max(0.0, 1.0 - float(dist))
            output.append({
                "content": doc,
                "score": round(score, 4),
                "metadata": meta or {},
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định ban hành văn bản quy phạm pháp luật", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content'][:100]}...")

