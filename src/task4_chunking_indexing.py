"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options (chọn 1, cân nhắc đánh đổi cài đặt nặng vs cần API key):
    - sentence-transformers/all-MiniLM-L6-v2 hoặc BAAI/bge-m3 — chạy local, không
      cần API key, nhưng cài nặng (~1-2GB vì kéo theo torch)
    - Google models/text-embedding-004 (768 dim) — nhẹ, cần GEMINI_API_KEY
    - OpenAI text-embedding-3-small (1536 dim) — nhẹ, cần OPENAI_API_KEY
    Gợi ý: đọc EMBEDDING_PROVIDER từ .env (os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"))
    để cả nhóm có thể đổi provider mà không sửa code — nhớ đổi provider phải xoá
    chroma_db/ cũ và reindex vì dimension khác nhau (1024/768/1536) không tương thích ngược.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "rag_index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chia trước theo cấu trúc Điều/Mục của văn bản luật, sau đó giới hạn theo ký tự.
# 1.200 ký tự giữ được một ý pháp lý tương đối trọn vẹn; overlap 150 ký tự giúp
# không mất ngữ cảnh tại ranh giới nhưng vẫn tránh lặp quá nhiều.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "legal_structure"

# VietLegal Harrier đã được huấn luyện cho truy hồi pháp luật Việt Nam. Khi model
# local không tồn tại, có thể override bằng LEGAL_RAG_EMBEDDING_MODEL.
_DEFAULT_LOCAL_MODEL = (
    Path(__file__).parents[2]
    / "Legal_assistant"
    / "model_cache"
    / "vietlegal-harrier-0.6b"
)
EMBEDDING_MODEL = os.getenv(
    "LEGAL_RAG_EMBEDDING_MODEL",
    str(_DEFAULT_LOCAL_MODEL)
    if _DEFAULT_LOCAL_MODEL.exists()
    else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIM = 1024

# File NumPy normalized vectors là vector store local nhỏ, reproducible và không
# cần Docker. Corpus production trong folder main sẽ dùng Qdrant + Elasticsearch.
VECTOR_STORE = "numpy_cosine"
COLLECTION_NAME = "vietnamese_legal_docs"

_embedding_model = None


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative.parts[0] if len(relative.parts) > 1 else "legal"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": relative.as_posix(),
                    "type": doc_type,
                },
            }
        )
    return documents


_LEGAL_HEADING_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?(Điều\s+\d+|Mục\s+\d+|Chương\s+[IVXLCDM\d]+|"
    r"PHỤ\s+LỤC(?:\s+[IVXLCDM\d]+)?)(?:[.:\-]?\s*(.*))?$"
)


def _structural_units(text: str) -> list[tuple[str, str, str]]:
    """Return ``(label, title, text)`` units while preserving legal headings."""
    matches = list(_LEGAL_HEADING_RE.finditer(text))
    if not matches:
        return [("Toàn văn", "", text)]
    units = []
    if text[: matches[0].start()].strip():
        units.append(("Phần mở đầu", "", text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            units.append((match.group(1).strip(), (match.group(2) or "").strip(), body))
    return units or [("Toàn văn", "", text)]


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    output = []
    start = 0
    while start < len(text):
        hard_end = min(start + CHUNK_SIZE, len(text))
        end = hard_end
        if hard_end < len(text):
            lower_bound = start + CHUNK_SIZE // 2
            cuts = [
                text.rfind(separator, lower_bound, hard_end)
                for separator in ("\n\n", "\n", ". ", "; ", " ")
            ]
            natural_cut = max(cuts)
            if natural_cut > start:
                end = natural_cut + 1
        chunk = text[start:end].strip()
        if chunk:
            output.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return output


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for document in documents:
        chunk_index = 0
        for label, title, unit_text in _structural_units(document["content"]):
            for part_index, chunk_text in enumerate(_split_text(unit_text)):
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            **document.get("metadata", {}),
                            "article_no": label,
                            "article_title": title,
                            "part_index": part_index,
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1
    return chunks


def get_embedding_model():
    """Load the embedding model lazily and reuse it across task modules."""
    global _embedding_model
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        requested_device = os.getenv("LEGAL_RAG_DEVICE", "cuda")
        device = requested_device if torch.cuda.is_available() else "cpu"
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            model_kwargs={"torch_dtype": torch.float32},
            device=device,
        )
        _embedding_model.max_seq_length = 512
    return _embedding_model


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Embed and L2-normalize texts with the same model used by search."""
    if not texts:
        return []
    values = list(texts)
    if is_query and "vietlegal-harrier" in EMBEDDING_MODEL.lower():
        values = [
            "Instruct: Given a Vietnamese legal question, retrieve relevant legal "
            f"passages that answer the question\nQuery: {text}"
            for text in values
        ]
    vectors = get_embedding_model().encode(
        values,
        batch_size=min(32, len(values)),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    return [
        {**chunk, "embedding": vector}
        for chunk, vector in zip(chunks, vectors)
    ]


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    serializable_chunks = []
    vectors = []
    for index, chunk in enumerate(chunks):
        if "embedding" not in chunk:
            raise ValueError(f"Chunk {index} chưa có embedding")
        serializable_chunks.append(
            {"content": chunk["content"], "metadata": chunk.get("metadata", {})}
        )
        vectors.append(chunk["embedding"])

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 and len(chunks):
        raise ValueError("Embeddings phải là ma trận 2 chiều")
    np.save(EMBEDDINGS_PATH, matrix)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in serializable_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    manifest = {
        "collection": COLLECTION_NAME,
        "vector_store": VECTOR_STORE,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 0,
        "chunk_count": len(serializable_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_vector_index() -> tuple[list[dict], np.ndarray]:
    """Load the persisted chunks and normalized embedding matrix."""
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return [], np.empty((0, 0), dtype=np.float32)
    chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = np.load(EMBEDDINGS_PATH)
    if len(chunks) != len(matrix):
        raise RuntimeError("Vector index không nhất quán; hãy chạy lại Task 4")
    return chunks, matrix


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
