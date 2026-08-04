"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

_reranker_tokenizer = None
_reranker_model = None


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates or top_k <= 0:
        return []
    global _reranker_model, _reranker_tokenizer
    default_path = Path(__file__).parents[2] / "Legal_assistant" / "model_cache" / "ViRanker"
    model_path = Path(os.getenv("LEGAL_RAG_RERANKER_MODEL", str(default_path)))
    if not model_path.exists():
        return _overlap_rerank(query, candidates, top_k)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if _reranker_model is None or _reranker_tokenizer is None:
        _reranker_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        _reranker_model.eval()

    pairs = [[query, item.get("content", "")] for item in candidates]
    inputs = _reranker_tokenizer(
        pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt"
    ).to(_reranker_model.device)
    with torch.inference_mode():
        scores = _reranker_model(**inputs, return_dict=True).logits.view(-1).float()
    output = [
        {**item, "score": float(score)}
        for item, score in zip(candidates, scores.cpu().tolist())
    ]
    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _overlap_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    terms = set(re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", query.lower()))
    output = []
    for candidate in candidates:
        doc_terms = set(
            re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", candidate.get("content", "").lower())
        )
        score = len(terms & doc_terms) / max(1, len(terms))
        output.append({**candidate, "score": float(score)})
    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param phải nằm trong [0, 1]")
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    output = []
    for _ in range(min(top_k, len(candidates))):
        best_index = None
        best_score = float("-inf")
        for index in remaining:
            embedding = candidates[index].get("embedding")
            if embedding is None:
                raise ValueError("MMR yêu cầu mỗi candidate có embedding")
            relevance = _cosine(query_embedding, embedding)
            redundancy = max(
                (
                    _cosine(embedding, candidates[chosen]["embedding"])
                    for chosen in selected
                ),
                default=0.0,
            )
            score = lambda_param * relevance - (1.0 - lambda_param) * redundancy
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)
        output.append({**candidates[best_index], "score": float(best_score)})
    return output


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if k < 0:
        raise ValueError("k phải >= 0")
    scores: dict[str, float] = {}
    documents: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            metadata = item.get("metadata") or {}
            key = str(
                metadata.get("chunk_id")
                or f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{item.get('content', '')}"
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            documents[key] = item
    ranked_keys = sorted(scores, key=scores.get, reverse=True)[: max(0, top_k)]
    return [{**documents[key], "score": float(scores[key])} for key in ranked_keys]


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from .task4_chunking_indexing import embed_texts

        embedded_candidates = candidates
        if any("embedding" not in item for item in embedded_candidates):
            vectors = embed_texts([item.get("content", "") for item in candidates])
            embedded_candidates = [
                {**item, "embedding": vector}
                for item, vector in zip(candidates, vectors)
            ]
        query_embedding = embed_texts([query], is_query=True)[0]
        return rerank_mmr(query_embedding, embedded_candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
