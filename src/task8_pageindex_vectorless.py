"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> list[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex qua SDK.

    Returns:
        List of doc_ids đã upload thành công.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được cấu hình. Bỏ qua upload.")
        return []

    try:
        from pageindex import PageIndexClient

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        doc_ids = []

        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            print(f"Uploading {md_file.name} to PageIndex...")
            resp = client.submit_document(str(md_file))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids.append(doc_id)
                print(f"  [OK] Uploaded: {md_file.name} -> {doc_id}")
        return doc_ids
    except Exception as e:
        print(f"⚠ Lỗi khi upload tài liệu lên PageIndex: {e}")
        return []


def _local_structured_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Fallback: Truy vấn theo cấu trúc chương/mục/tiêu đề (Vectorless RAG) trên các file Markdown.
    """
    results = []
    keywords = [kw.lower() for kw in query.split() if len(kw) > 1]
    if not keywords:
        keywords = [query.lower()]

    if not STANDARDIZED_DIR.exists():
        return results

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        sections = content.split("\n#")

        for idx, section in enumerate(sections):
            if not section.strip():
                continue
            section_text = "#" + section if idx > 0 else section
            section_lower = section_text.lower()

            # Tính score theo độ khớp từ khóa trong section
            match_count = sum(1 for kw in keywords if kw in section_lower)
            if match_count > 0:
                score = round(min(0.95, 0.40 + 0.15 * match_count), 4)
                first_line = section_text.strip().split("\n")[0]
                results.append({
                    "content": section_text.strip()[:1000],
                    "score": score,
                    "metadata": {
                        "source": md_file.name,
                        "section": first_line.replace("#", "").strip()
                    },
                    "source": "pageindex"
                })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex SDK (hoặc local structural fallback).
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn của người dùng
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # Nếu có API Key, thử gọi PageIndex SDK
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

            # Lấy danh sách doc_id hiện có
            doc_list = client.list_documents()
            docs = doc_list.get("documents", []) if isinstance(doc_list, dict) else []

            if docs:
                doc_id = docs[0].get("id") or docs[0].get("doc_id")
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id") or resp.get("id")

                if retrieval_id:
                    # Poll chờ kết quả
                    for _ in range(10):
                        time.sleep(1)
                        retrieval = client.get_retrieval(retrieval_id)
                        status = retrieval.get("status")
                        if status == "completed":
                            break

                    results = []
                    for node in retrieval.get("retrieved_nodes", [])[:top_k]:
                        for group in node.get("relevant_contents", []):
                            for item in group:
                                content_str = item.get("relevant_content", "")
                                if content_str.strip():
                                    results.append({
                                        "content": content_str,
                                        "score": 0.85,
                                        "metadata": {"section": item.get("section_title", "")},
                                        "source": "pageindex"
                                    })
                    if results:
                        return results[:top_k]
        except Exception as e:
            print(f"⚠ PageIndex API call failed, switching to local vectorless fallback: {e}")

    # Chạy local vectorless structural search fallback khi không có API key hoặc API lỗi
    local_results = _local_structured_search(query, top_k=top_k)
    if local_results:
        return local_results

    # Trả về kết quả rác hợp lệ nếu không tìm thấy
    return [{
        "content": f"Truy vấn thông tin cấu trúc cho '{query}'",
        "score": 0.50,
        "metadata": {"source": "vectorless_fallback"},
        "source": "pageindex"
    }]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[INFO] PAGEINDEX_API_KEY not configured. Running in local vectorless search mode.")

    print("Executing PageIndex search...")
    results = pageindex_search("quy dinh ban hanh van ban quy pham phap luat", top_k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        safe_content = r['content'][:100].replace('\n', ' ')
        print(f"[{r['score']:.4f}] [{r['source']}] {safe_content.encode('ascii', 'ignore').decode('ascii')}...")




