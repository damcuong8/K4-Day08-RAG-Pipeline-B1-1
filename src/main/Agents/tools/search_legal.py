from typing import List, Dict, Any, Tuple
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os
import threading
import json
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import py_vncorenlp
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from Agents.logs.agent_logger import logger
from Agents.llm_client import get_llm

from Agents.config import (
    RERANKER_PATH, 
    ES_HOST, QDRANT_HOST, QDRANT_PORT, INDEX_NAME, 
    COMPRESS_LLM_TEMPERATURE, COMPRESS_LLM_TOP_P, COMPRESS_LLM_TOP_K, COMPRESS_LLM_ENABLE_THINKING,
    RETRIEVER_TOP_K,
    RERANKER_TOP_K,
    SEARCH_LOCAL_DOCS,
    EMBEDDING_MAX_CONCURRENT,
    RERANKER_MAX_CONCURRENT,
    RERANKER_BATCH_SIZE,
    VNCORENLP_MAX_CONCURRENT,
    COMPRESS_TARGET_MAX_WORKERS,
    EMBEDDING_MODEL_PATH, VNCORENLP_DIR
)

reranker_tokenizer = None
reranker_model = None
embedding_model = None
vncorenlp_rdrsegmenter = None
es_client = None
qdrant_client = None
_resource_init_lock = threading.Lock()
_embedding_queue = threading.BoundedSemaphore(EMBEDDING_MAX_CONCURRENT)
_reranker_queue = threading.BoundedSemaphore(RERANKER_MAX_CONCURRENT)
_vncorenlp_queue = threading.BoundedSemaphore(VNCORENLP_MAX_CONCURRENT)
_filter_stats_log_lock = threading.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"

def _acquire_queue(queue: threading.BoundedSemaphore, name: str) -> None:
    if not queue.acquire(blocking=False):
        logger.info(f"[*] {name} đang bận, request sẽ chờ trong queue...")
        queue.acquire()

def _release_queue(queue: threading.BoundedSemaphore) -> None:
    queue.release()

def get_reranker():
    """Trả về reranker đã được nạp sẵn."""
    global reranker_tokenizer, reranker_model
    if reranker_model is None or reranker_tokenizer is None:
        with _resource_init_lock:
            if reranker_model is None or reranker_tokenizer is None:
                logger.info(f"[*] Đang nạp mô hình ViRanker từ: {RERANKER_PATH}")
                reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH)
                reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH).to(device)
                reranker_model.eval()
                logger.info("[*] Nạp ViRanker thành công!")
    return reranker_tokenizer, reranker_model

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        with _resource_init_lock:
            if embedding_model is None:
                logger.info(f"[*] Đang nạp mô hình Embedding từ: {EMBEDDING_MODEL_PATH}")
                embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, model_kwargs={"torch_dtype": torch.float32})
                embedding_model.max_seq_length = 512
                logger.info("[*] Nạp Embedding Model thành công!")
    return embedding_model

def get_vncorenlp():
    global vncorenlp_rdrsegmenter
    if vncorenlp_rdrsegmenter is None:
        with _resource_init_lock:
            if vncorenlp_rdrsegmenter is None:
                logger.info("[*] Đang khởi tạo VnCoreNLP...")
                os.makedirs(VNCORENLP_DIR, exist_ok=True)
                if not os.path.exists(os.path.join(VNCORENLP_DIR, "models")):
                    py_vncorenlp.download_model(save_dir=VNCORENLP_DIR)
                vncorenlp_rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=VNCORENLP_DIR)
                logger.info("[*] Nạp VnCoreNLP thành công!")
    return vncorenlp_rdrsegmenter

def get_es_client():
    global es_client
    if es_client is None:
        with _resource_init_lock:
            if es_client is None:
                logger.info(f"[*] Đang kết nối Elasticsearch tại: {ES_HOST}")
                es_client = Elasticsearch(ES_HOST)
    return es_client

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        with _resource_init_lock:
            if qdrant_client is None:
                logger.info(f"[*] Đang kết nối Qdrant Server tại: {QDRANT_HOST}:{QDRANT_PORT}")
                qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return qdrant_client

def initialize_resources():
    """Nạp toàn bộ resource dùng chung ngay lúc khởi động process."""
    logger.info("[*] Khởi tạo sẵn các model và client truy hồi...")
    get_vncorenlp()
    get_embedding_model()
    get_reranker()
    get_es_client()
    get_qdrant_client()
    logger.info("[*] Hoàn tất khởi tạo resource.")

def tokenize_text(text: str) -> str:
    if not text: return ""
    rdrsegmenter = get_vncorenlp()
    _acquire_queue(_vncorenlp_queue, "VnCoreNLP")
    try:
        sentences = rdrsegmenter.word_segment(text)
        return " ".join(sentences)
    finally:
        _release_queue(_vncorenlp_queue)

def perform_batch_hybrid_search(bm25_queries: List[str], dense_queries: List[str], top_k: int = RETRIEVER_TOP_K) -> List[List[Dict[str, Any]]]:
    """Chạy hybrid search cho nhiều query cùng lúc."""
    es = get_es_client()
    qdrant = get_qdrant_client()
    embed_model = get_embedding_model()
    
    logger.info(f"[*] Batch truy vấn Database - {len(dense_queries)} queries")
    
    instruct_queries = [f"Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: {q}" for q in dense_queries]
    _acquire_queue(_embedding_queue, "Embedding model")
    try:
        query_vectors = embed_model.encode(instruct_queries, batch_size=len(instruct_queries), normalize_embeddings=True, show_progress_bar=False)
    finally:
        _release_queue(_embedding_queue)
    
    all_final_docs = []
    
    es_body = []
    for bm25_query in bm25_queries:
        tokenized_query = tokenize_text(bm25_query)
        es_body.append({"index": INDEX_NAME})
        match_query = {"match": {"content_search": {"query": tokenized_query}}}
        if not SEARCH_LOCAL_DOCS:
            final_query = {
                "bool": {
                    "must": [match_query],
                    "filter": [{"term": {"is_local": False}}]
                }
            }
        else:
            final_query = match_query
            
        es_body.append({"query": final_query, "size": top_k, "_source": False})
        
    try:
        es_res = es.msearch(body=es_body)
        es_responses = [resp.get("hits", {}).get("hits", []) for resp in es_res.get("responses", [])]
    except Exception as e:
        logger.error(f"[!] Lỗi truy vấn Elasticsearch msearch: {e}")
        es_responses = [[] for _ in bm25_queries]

    from qdrant_client.models import QueryRequest, Filter, FieldCondition, MatchValue
    
    qdrant_filter = None
    if not SEARCH_LOCAL_DOCS:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="is_local",
                    match=MatchValue(value=False)
                )
            ]
        )

    qdrant_requests = [
        QueryRequest(
            query=query_vectors[idx].tolist(), 
            limit=top_k,
            filter=qdrant_filter
        )
        for idx in range(len(query_vectors))
    ]
    try:
        qdrant_res_batch = qdrant.query_batch_points(
            collection_name=INDEX_NAME,
            requests=qdrant_requests
        )
        qdrant_responses = [resp.points for resp in qdrant_res_batch]
    except Exception as e:
        logger.error(f"[!] Lỗi truy vấn Qdrant batch: {e}")
        qdrant_responses = [[] for _ in dense_queries]
    
    ranked_chunk_ids_by_query = []
    rrf_scores_by_query = []
    all_chunk_ids = []
    seen_chunk_ids = set()

    for idx, (bm25_query, dense_query) in enumerate(zip(bm25_queries, dense_queries)):
        es_hits = es_responses[idx] if idx < len(es_responses) else []
        qdrant_res = qdrant_responses[idx] if idx < len(qdrant_responses) else []
        
        k = 60
        rrf_scores = {}
        for rank, hit in enumerate(es_hits):
            chunk_id = hit["_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, point in enumerate(qdrant_res):
            chunk_id = point.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        ranked_chunk_ids_by_query.append(sorted_chunk_ids)
        rrf_scores_by_query.append(rrf_scores)

        for chunk_id in sorted_chunk_ids:
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                all_chunk_ids.append(chunk_id)

    docs_by_chunk_id = {}
    if all_chunk_ids:
        try:
            mget_res = es.mget(index=INDEX_NAME, body={"ids": all_chunk_ids})
            for requested_id, doc in zip(all_chunk_ids, mget_res.get("docs", [])):
                if not doc.get("found"):
                    continue

                src = doc.get("_source", {}) or {}
                chunk_id = src.get("chunk_id") or doc.get("_id") or requested_id
                if not chunk_id:
                    continue

                law_title = str(src.get('raw_title') or '').strip()
                document_number = str(src.get('document_number') or '').strip()
                article_no = str(src.get('article_no') or '').strip()
                article_title = str(src.get('article_title') or '').strip()
                raw_content = str(src.get('raw_content') or '').strip()
                url = str(src.get('url') or '').strip()

                prefix = f"{law_title}\n" if law_title else ""
                if article_no: prefix += f"{article_no}: {article_title}\n" if article_title else f"{article_no}\n"
                elif article_title: prefix += f"{article_title}\n"

                prepared_doc = {
                    "id": chunk_id,
                    "text": prefix + raw_content,
                    "url": url,
                    "document_number": document_number,
                    "doc_name": law_title,
                    "article_no": article_no,
                    "article_title": article_title,
                }
                lookup_ids = {requested_id, str(requested_id), chunk_id, str(chunk_id)}
                es_doc_id = doc.get("_id")
                if es_doc_id:
                    lookup_ids.update({es_doc_id, str(es_doc_id)})
                for lookup_id in lookup_ids:
                    docs_by_chunk_id[lookup_id] = prepared_doc
        except Exception as e:
            logger.error(f"[!] Lỗi lấy nội dung Elasticsearch mget: {e}")

    for sorted_chunk_ids, rrf_scores in zip(ranked_chunk_ids_by_query, rrf_scores_by_query):
        final_docs = []
        for chunk_id in sorted_chunk_ids:
            doc = docs_by_chunk_id.get(chunk_id)
            if not doc:
                continue
            final_doc = dict(doc)
            final_doc["rrf_score"] = rrf_scores.get(chunk_id, 0.0)
            final_docs.append(final_doc)

        final_docs = sorted(final_docs, key=lambda x: x.get("rrf_score", 0), reverse=True)
        all_final_docs.append(final_docs)
        
    return all_final_docs


def perform_batch_dense_search(
    dense_queries: List[str],
    top_k: int = RERANKER_TOP_K,
) -> List[List[Dict[str, Any]]]:
    """Dense-only baseline dùng cùng Qdrant index và embedding production.

    ``perform_batch_hybrid_search`` đã chứa toàn bộ logic embed, lọc corpus và
    resolve chunk metadata. Truyền BM25 query rỗng làm nhánh Elasticsearch không
    đóng góp kết quả, vì vậy thứ hạng đầu ra chỉ đến từ Qdrant. Baseline này cố ý
    không chạy ViRanker để tạo đối chứng A/B rõ ràng với pipeline production.
    """
    if not dense_queries:
        return []
    return perform_batch_hybrid_search(
        [""] * len(dense_queries),
        dense_queries,
        top_k=top_k,
    )

def perform_batch_local_rerank(queries: List[str], raw_docs_list: List[List[Dict[str, Any]]], top_k: int = RERANKER_TOP_K) -> List[List[Dict[str, Any]]]:
    """Rerank batch tất cả tài liệu của tất cả queries."""
    tokenizer, model = get_reranker()
    
    flat_pairs = []
    doc_indices = []
    for q_idx, (query, raw_docs) in enumerate(zip(queries, raw_docs_list)):
        for d_idx, doc in enumerate(raw_docs):
            flat_pairs.append([query, doc.get("text", "")])
            doc_indices.append((q_idx, d_idx))
            
    if not flat_pairs:
        return [[] for _ in queries]
        
    logger.info(f"[*] Đang Batch Rerank {len(flat_pairs)} cặp câu hỏi-tài liệu")
    batch_size = RERANKER_BATCH_SIZE
    all_scores = []
    
    _acquire_queue(_reranker_queue, "Reranker model")
    try:
        with torch.inference_mode():
            for i in range(0, len(flat_pairs), batch_size):
                batch_pairs = flat_pairs[i:i+batch_size]
                inputs = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024).to(model.device)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()
                all_scores.extend(scores.tolist())
    finally:
        _release_queue(_reranker_queue)
            
    for (q_idx, d_idx), score in zip(doc_indices, all_scores):
        raw_docs_list[q_idx][d_idx]["rerank_score"] = float(score)
        
    reranked_docs_list = []
    for raw_docs in raw_docs_list:
        sorted_docs = sorted(raw_docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
        reranked_docs_list.append(sorted_docs[:top_k])
        
    return reranked_docs_list

class QueryPair(BaseModel):
    bm25_query: str = Field(description="Từ khóa tìm kiếm cho BM25 (chứa các keyword quan trọng, loại bỏ stopwords).")
    dense_query: str = Field(description="Câu hỏi hoặc văn bản ngữ nghĩa cho Dense/Vector search (giữ nguyên ngữ cảnh).")

class SearchInput(BaseModel):
    queries: List[QueryPair] = Field(description="Danh sách các cặp truy vấn để tìm kiếm chung một mục đích cho mỗi cặp. Mỗi mục cần có 1 bm25_query và 1 dense_query tương ứng.")

class ToolCompressOutput(BaseModel):
    relevant_chunk_ids: List[str] = Field(description="Danh sách ID (Ví dụ: DOC_0, DOC_1) của các tài liệu chứa thông tin liên quan đến mục tiêu tìm kiếm. Bỏ qua các ID vô giá trị.")

TOOL_COMPRESS_SYSTEM_PROMPT = """Bạn là một Thẩm định viên pháp lý chuyên nghiệp. Nhiệm vụ của bạn là LỌC dữ liệu đầu vào để tìm ra căn cứ pháp lý chính xác nhất.
Bạn chỉ đang xử lý MỘT cặp truy vấn tìm kiếm bổ sung duy nhất.

Quy tắc:
1. Đối chiếu từng tài liệu với BM25 query và Dense query của cặp truy vấn hiện tại.
2. Nếu tài liệu không liên quan trực tiếp hoặc không giải quyết được mục đích tìm kiếm hiện tại, bỏ qua ID đó.
3. Nếu tài liệu liên quan trực tiếp và đáp ứng đúng mục đích tìm kiếm hiện tại, đưa ID dạng DOC_X vào danh sách trả về.
4. Không loại bỏ toàn bộ chỉ vì tài liệu chưa đủ trả lời mọi vấn đề trong câu hỏi gốc; cặp truy vấn hiện tại chỉ cần chọn căn cứ tốt nhất cho lát cắt pháp lý của chính nó.
5. Khi có cả văn bản gốc và văn bản hợp nhất/VBHN về cùng một luật và cùng điều, chỉ chọn MỘT nguồn đại diện:
   - Nếu nội dung quy định giống nhau hoặc VBHN không thể hiện sửa đổi, bổ sung liên quan đến điều đó, chọn văn bản gốc và bỏ VBHN.
   - Nếu nội dung khác nhau hoặc VBHN thể hiện nội dung đã được sửa đổi, bổ sung, chọn VBHN và bỏ văn bản gốc cũ.
   - Không trả về đồng thời văn bản gốc và VBHN cho cùng một quy định nếu chúng chỉ trùng lặp căn cứ.

Mục đích của bạn là cung cấp bộ chứng cứ sạch, chính xác và có giá trị pháp lý cao cho truy vấn hiện tại."""

TOOL_COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TOOL_COMPRESS_SYSTEM_PROMPT),
    (
        "human",
        "Truy vấn tìm kiếm bổ sung hiện tại:\n"
        "- Query index: {query_index}\n"
        "- BM25 query: {bm25_query}\n"
        "- Dense query: {dense_query}\n\n"
        "Các tài liệu tìm được cho truy vấn này:\n{context}",
    ),
])

def _log_tool_filter_stats(total: int, kept: int, **extra: Any) -> None:
    total = max(0, int(total or 0))
    kept = max(0, int(kept or 0))
    removed = max(0, total - kept)
    removed_pct = (removed / total * 100.0) if total else 0.0
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": "Tool Compress Stats",
        "input": total,
        "kept": kept,
        "removed": removed,
        "removed_pct": round(removed_pct, 3),
        **extra,
    }
    logger.info(
        "[Tool Compress Stats] input=%s kept=%s removed=%s removed_pct=%.1f%% %s",
        total,
        kept,
        removed,
        removed_pct,
        " ".join(f"{key}={value}" for key, value in extra.items()),
    )

    log_path = os.getenv("FILTER_STATS_LOG_PATH", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with _filter_stats_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Không ghi được tool filter stats log: {e}")

def _normalize_query_pairs(queries: List[QueryPair]) -> List[QueryPair]:
    normalized_queries = []
    for query in queries or []:
        if isinstance(query, QueryPair):
            normalized_queries.append(query)
        elif isinstance(query, dict):
            normalized_queries.append(QueryPair(**query))
    return normalized_queries

def _invoke_structured_with_retries(chain, payload: Dict[str, Any], label: str, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = chain.invoke(payload)
            if result is None:
                raise ValueError("Structured output returned None")
            return result
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                logger.warning(f"{label} structured output lỗi lần {attempt}/{max_attempts}, retry: {e}")
            else:
                logger.error(f"{label} structured output lỗi sau {max_attempts} lần: {e}")
    raise RuntimeError(f"{label} structured output failed after {max_attempts} attempts") from last_error

def _required_structured_tool(llm, schema):
    from langchain_core.output_parsers.openai_tools import PydanticToolsParser

    return llm.bind_tools([schema], tool_choice="required") | PydanticToolsParser(
        tools=[schema],
        first_tool_only=True,
    )

def _build_tool_compress_context(docs: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    context_parts = []
    doc_mapping = {}
    for idx, doc in enumerate(docs or []):
        temp_id = f"DOC_{idx}"
        doc_mapping[temp_id] = doc

        metadata_lines = []
        if doc.get("document_number"):
            metadata_lines.append(f"Số hiệu văn bản: {doc.get('document_number')}")
        if doc.get("doc_name"):
            metadata_lines.append(f"Tên văn bản: {doc.get('doc_name')}")
        if doc.get("article_no"):
            metadata_lines.append(f"Điều: {doc.get('article_no')}")
        if doc.get("article_title"):
            metadata_lines.append(f"Tiêu đề điều: {doc.get('article_title')}")

        metadata_text = "\n".join(metadata_lines)
        if metadata_text:
            metadata_text = f"\nMetadata:\n{metadata_text}"

        context_parts.append(
            f">> Tài liệu: {temp_id} <<{metadata_text}\n"
            f"Nội dung:\n{doc.get('text', '')}"
        )

    return "\n\n".join(context_parts), doc_mapping


def _select_tool_compress_docs(
    *,
    query_index: int,
    query: QueryPair,
    docs: List[Dict[str, Any]],
    chain,
) -> Dict[str, Any]:
    context_text, doc_mapping = _build_tool_compress_context(docs)
    result = _invoke_structured_with_retries(
        chain,
        {
            "query_index": query_index,
            "bm25_query": query.bm25_query,
            "dense_query": query.dense_query,
            "context": context_text,
        },
        "Tool Compress",
    )

    selected_docs = []
    selected_temp_ids = []
    seen_ids = set()
    for temp_id in result.relevant_chunk_ids or []:
        temp_id = str(temp_id).strip()
        doc = doc_mapping.get(temp_id)
        if not doc:
            continue
        doc_id = str(doc.get("id") or "").strip()
        dedupe_key = doc_id or temp_id
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        selected_temp_ids.append(temp_id)
        selected_docs.append(doc)

    mode = "structured"
    if not selected_docs and docs:
        selected_docs = docs[:1]
        mode = "fallback_empty_query_filter"

    return {
        "query_index": query_index,
        "selected_docs": selected_docs,
        "selected_temp_ids": selected_temp_ids,
        "mode": mode,
        "input_count": len(docs or []),
    }

def _run_hybrid_search_docs_with_empty_message(queries: List[QueryPair]) -> Tuple[List[Dict[str, Any]], str]:
    "Chạy logic search thật và trả docs đã lọc kèm thông báo khi rỗng."
    queries = _normalize_query_pairs(queries)
    if not queries:
        _log_tool_filter_stats(0, 0, mode="no_queries", query_count=0)
        return [], "Không có truy vấn tìm kiếm hợp lệ."

    bm25_queries = [q.bm25_query for q in queries]
    dense_queries = [q.dense_query for q in queries]
    logger.info(f"[*] Chạy truy vấn bổ sung với {len(queries)} cặp query.")

    raw_docs_list = perform_batch_hybrid_search(bm25_queries, dense_queries, top_k=RETRIEVER_TOP_K)
    reranked_docs_list = perform_batch_local_rerank(dense_queries, raw_docs_list, top_k=RERANKER_TOP_K)

    total_docs = sum(len(docs or []) for docs in reranked_docs_list)

    if not total_docs:
        _log_tool_filter_stats(0, 0, mode="no_docs", query_count=len(queries))
        return [], "Không tìm thấy kết quả pháp lý nào."

    query_jobs = [
        (idx, query, docs)
        for idx, (query, docs) in enumerate(zip(queries, reranked_docs_list), start=1)
        if docs
    ]
    logger.info(
        "[*] Đang chạy LLM Filter theo %s query bổ sung cho %s tài liệu...",
        len(query_jobs),
        total_docs,
    )

    llm = get_llm(
        temperature=COMPRESS_LLM_TEMPERATURE,
        top_p=COMPRESS_LLM_TOP_P,
        top_k=COMPRESS_LLM_TOP_K,
        enable_thinking=COMPRESS_LLM_ENABLE_THINKING
    )
    structured_llm = _required_structured_tool(llm, ToolCompressOutput)
    chain = TOOL_COMPRESS_PROMPT | structured_llm

    try:
        query_results = []
        max_workers = min(COMPRESS_TARGET_MAX_WORKERS, len(query_jobs)) or 1
        if len(query_jobs) == 1:
            query_index, query, docs = query_jobs[0]
            query_results.append(
                _select_tool_compress_docs(
                    query_index=query_index,
                    query=query,
                    docs=docs,
                    chain=chain,
                )
            )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        _select_tool_compress_docs,
                        query_index=query_index,
                        query=query,
                        docs=docs,
                        chain=chain,
                    ): (query_index, docs)
                    for query_index, query, docs in query_jobs
                }
                results_by_query = {}
                for future in as_completed(future_map):
                    query_index, docs = future_map[future]
                    try:
                        results_by_query[query_index] = future.result()
                    except Exception as e:
                        logger.error("[!] Lỗi Tool Compress query %s: %s", query_index, e)
                        results_by_query[query_index] = {
                            "query_index": query_index,
                            "selected_docs": docs[:1],
                            "selected_temp_ids": [],
                            "mode": "fallback_query_error",
                            "input_count": len(docs or []),
                        }

                for query_index, _, _ in query_jobs:
                    if query_index in results_by_query:
                        query_results.append(results_by_query[query_index])

        unique_docs = []
        seen_ids = set()
        selected_temp_ids_count = 0
        for query_result in query_results:
            selected_temp_ids = query_result.get("selected_temp_ids", []) or []
            selected_docs = query_result.get("selected_docs", []) or []
            selected_temp_ids_count += len(selected_temp_ids)
            _log_tool_filter_stats(
                query_result.get("input_count", 0),
                len(selected_docs),
                mode=query_result.get("mode"),
                query_index=query_result.get("query_index"),
                selected_temp_ids=len(selected_temp_ids),
            )
            for doc in selected_docs:
                doc_id = str(doc.get("id") or "").strip()
                dedupe_key = doc_id or str(id(doc))
                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)
                unique_docs.append(doc)

        if not unique_docs:
            for _, _, docs in query_jobs:
                for doc in docs[:1]:
                    doc_id = str(doc.get("id") or "").strip()
                    dedupe_key = doc_id or str(id(doc))
                    if dedupe_key in seen_ids:
                        continue
                    seen_ids.add(dedupe_key)
                    unique_docs.append(doc)
            _log_tool_filter_stats(
                total_docs,
                len(unique_docs),
                mode="fallback_empty_filter",
                query_count=len(queries),
                queries_with_docs=len(query_jobs),
                workers=max_workers,
                selected_temp_ids=selected_temp_ids_count,
            )
        else:
            _log_tool_filter_stats(
                total_docs,
                len(unique_docs),
                mode="query_batch",
                query_count=len(queries),
                queries_with_docs=len(query_jobs),
                workers=max_workers,
                selected_temp_ids=selected_temp_ids_count,
            )
    except Exception as e:
        logger.error(f"[!] Lỗi LLM nén theo query: {e}")
        unique_docs = []
        seen_ids = set()
        for _, _, docs in query_jobs:
            for doc in docs[:1]:
                doc_id = str(doc.get("id") or "").strip()
                dedupe_key = doc_id or str(id(doc))
                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)
                unique_docs.append(doc)
        _log_tool_filter_stats(
            total_docs,
            len(unique_docs),
            mode="fallback_error",
            query_count=len(queries),
            queries_with_docs=len(query_jobs),
        )

    if not unique_docs:
        return [], "Không tìm thấy kết quả pháp lý nào."

    return unique_docs, "Không tìm thấy kết quả pháp lý nào."


@tool(args_schema=SearchInput)
def hybrid_search_tool(queries: List[QueryPair]) -> List[Dict[str, Any]]:
    "Sử dụng công cụ này để tìm kiếm thêm các điều luật và văn bản pháp luật trên cơ sở dữ liệu."
    docs, _ = _run_hybrid_search_docs_with_empty_message(queries)
    return docs


def _auto_init_enabled() -> bool:
    raw = os.getenv("AUTO_INIT_RETRIEVAL_RESOURCES", "true")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


if _auto_init_enabled():
    initialize_resources()
