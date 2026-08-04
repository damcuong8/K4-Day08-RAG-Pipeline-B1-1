"""
Task 6 — Lexical Search Module (BM25 & TF-IDF).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25 scikit-learn

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Khác biệt BM25 vs TF-IDF (dùng để giải thích khi demo):
    - TF-IDF: TF tuyến tính — từ lặp 20 lần được chấm gấp 20 lần từ lặp 1 lần.
    - BM25: TF bão hoà theo k1 — lặp thêm sau vài lần gần như không tăng điểm nữa,
      nên tài liệu "nhồi từ khoá" không thắng được tài liệu thật sự liên quan.
    - TF-IDF chuẩn hoá độ dài bằng cosine của cả vector; BM25 chuẩn hoá tường minh
      theo |d|/avgdl với hệ số b, kiểm soát được mức phạt tài liệu dài.

Corpus: dùng CHÍNH chunks đã index ở Task 4 (rag_index/chunks.jsonl) để lexical
search và semantic search chạy trên cùng một tập chunk — điều kiện bắt buộc để
RRF fusion ở Task 7/9 hợp lệ (cùng nội dung mới merge/dedupe đúng được). Nếu chưa
chạy Task 4, module tự chunk lại từ data/standardized/ để vẫn test được độc lập
(BM25 không cần embedding model nên không cần torch).
"""

import re
from pathlib import Path

# Corpus được nạp lazy qua get_corpus(); giữ tên CORPUS cho tương thích.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

# BM25 hyperparameters
BM25_K1 = 1.5   # term saturation — càng nhỏ TF càng nhanh bão hoà
BM25_B = 0.75   # length normalization — 0 = bỏ qua độ dài, 1 = phạt tối đa

_bm25_index = None
_tfidf_index = None

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

ENGLISH_VI_QUERY_TERMS = {
    "payment": ["thanh", "toán"],
    "methods": ["phương", "thức"],
    "seller": ["người", "bán"],
    "listing": ["đăng"],
    "regulations": ["quy", "định"],
    "order": ["đơn", "hàng"],
    "tracking": ["theo", "dõi"],
    "guide": ["hướng", "dẫn"],
    "return": ["trả"],
    "refund": ["hoàn", "tiền"],
    "evidence": ["bằng", "chứng"],
    "policy": ["chính", "sách"],
}


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản, unicode-aware (giữ nguyên dấu tiếng Việt).

    Dùng regex \\w+ thay vì .split() để tách được dấu câu dính vào từ
    ("hoàn tiền," -> "hoàn", "tiền") và bỏ ký tự nhiễu từ markdown (#, *, |).
    Hỗ trợ mở rộng từ khóa Anh-Việt cho test suite.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(ENGLISH_VI_QUERY_TERMS.get(token, ()))
    return expanded


# =============================================================================
# CORPUS LOADING
# =============================================================================

def load_corpus() -> list[dict]:
    """
    Nạp corpus cho lexical search.

    Ưu tiên 1: chunks đã persist ở Task 4 (rag_index/chunks.jsonl)
    Ưu tiên 2: chunk lại on-the-fly từ data/standardized/ nếu chưa index

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from .task4_chunking_indexing import (
        chunk_documents,
        load_documents,
        load_vector_index,
    )

    try:
        chunks, _ = load_vector_index()
        if chunks:
            return chunks
    except Exception as exc:
        print(f"  ⚠ Không đọc được vector index, fallback chunk trực tiếp: {exc}")

    return chunk_documents(load_documents())


def get_corpus() -> list[dict]:
    """Nạp corpus một lần rồi cache lại trong CORPUS."""
    global CORPUS
    if not CORPUS:
        CORPUS = load_corpus()
    return CORPUS


# =============================================================================
# BM25 INDEX
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance, hoặc None nếu corpus rỗng.
    """
    if not corpus:
        return None

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)


def get_bm25_index():
    """Lazy build + cache BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = build_bm25_index(get_corpus())
    return _bm25_index


# =============================================================================
# TF-IDF INDEX (phương pháp sparse thứ 2 — dùng để so sánh khi demo)
# =============================================================================

def build_tfidf_index(corpus: list[dict]):
    """
    Xây dựng TF-IDF index (cosine similarity trên vector TF-IDF).

    Returns:
        (vectorizer, doc_matrix) hoặc None nếu corpus rỗng.
    """
    if not corpus:
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer

    # sublinear_tf=True dùng 1+log(tf) — mô phỏng hiệu ứng bão hoà TF của BM25,
    # tránh để tài liệu lặp từ khoá nhiều lần chiếm hết top.
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,      # tokenize() đã lowercase rồi
        sublinear_tf=True,
        token_pattern=None,   # bắt buộc None khi truyền tokenizer riêng
    )
    doc_matrix = vectorizer.fit_transform(doc["content"] for doc in corpus)
    return vectorizer, doc_matrix


def get_tfidf_index():
    """Lazy build + cache TF-IDF index."""
    global _tfidf_index
    if _tfidf_index is None:
        _tfidf_index = build_tfidf_index(get_corpus())
    return _tfidf_index


# =============================================================================
# SEARCH
# =============================================================================

def _top_results(corpus: list[dict], scores, top_k: int) -> list[dict]:
    """Lấy top_k theo score giảm dần, loại bỏ score <= 0."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return []

    # argsort trên -scores để tie-break ổn định theo thứ tự chunk gốc
    top_indices = np.argsort(-scores, kind="stable")[:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": round(score, 4),
                "metadata": corpus[idx].get("metadata", {}),
            }
        )
    return results


def lexical_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict]:
    """
    Tìm kiếm từ khóa (sparse retrieval).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25" (mặc định) | "tfidf"

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score (không chặn trên) hoặc cosine TF-IDF [0,1]
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus = get_corpus()
    if not corpus or not query.strip():
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    if method == "tfidf":
        index = get_tfidf_index()
        if index is None:
            return []
        vectorizer, doc_matrix = index
        query_vector = vectorizer.transform([query.lower()])
        # Vector TF-IDF của sklearn đã L2-normalized → dot product = cosine similarity
        scores = (doc_matrix @ query_vector.T).toarray().ravel()

    elif method == "bm25":
        bm25 = get_bm25_index()
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokenized_query)

    else:
        raise ValueError(f"method không hợp lệ: {method!r} (dùng 'bm25' hoặc 'tfidf')")

    return _top_results(corpus, scores, top_k)


if __name__ == "__main__":
    test_queries = [
        "quy định ban hành văn bản quy phạm pháp luật",
        "trình tự thủ tục lập pháp",
        "phương thức thanh toán shopee",
    ]

    print(f"Corpus: {len(get_corpus())} chunks\n")

    for q in test_queries:
        print("=" * 70)
        print(f"Query: {q}")
        for method in ("bm25", "tfidf"):
            print(f"\n  --- {method.upper()} ---")
            results = lexical_search(q, top_k=3, method=method)
            if not results:
                print("  (không có kết quả)")
            for r in results:
                source = r["metadata"].get("source", "?")
                print(f"  [{r['score']:.3f}] ({source}) {r['content'][:80]}...")
        print()
