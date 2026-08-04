from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import urlparse

from elasticsearch import Elasticsearch


DISCLAIMER_TEXT = (
    "Ý kiến tư vấn trên chỉ là hỗ trợ tra cứu ban đầu dựa trên dữ liệu được hệ thống "
    "truy hồi tại thời điểm xử lý. Người dùng cần đối chiếu với văn bản gốc hoặc tham "
    "vấn luật sư/chuyên gia pháp lý trước khi thực hiện hành động trên thực tế."
)
CITATION_PREFIX = "[[cite:"
MAX_CITATION_BODY_CHARS = 200

TECHNICAL_BLOCK_RE = re.compile(
    r"\s*<(?:ANSWER_CHECK_JSON|TECHNICAL_JSON)>.*?</(?:ANSWER_CHECK_JSON|TECHNICAL_JSON)>\s*",
    re.DOTALL,
)
ARTICLE_DIEU_SUFFIX_RE = re.compile(
    r"(\|Điều\s+\d+[A-Za-zĐđ]?)\s*\([^|]*\)\s*$",
    re.IGNORECASE,
)
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_number: str
    doc_name: str
    article_no: str
    article_title: str
    snippet: str
    url: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "document_number": self.document_number,
            "doc_name": self.doc_name,
            "article_no": self.article_no,
            "article_title": self.article_title,
            "snippet": self.snippet,
            "url": self.url,
            "doc_ref": self.doc_ref,
            "article_ref": self.article_ref,
        }

    @property
    def doc_ref(self) -> str:
        if not self.document_number or not self.doc_name:
            return ""
        return f"{self.document_number}|{self.doc_name}"

    @property
    def article_ref(self) -> str:
        if not self.document_number or not self.doc_name or not self.article_no:
            return ""
        return clean_article_ref(f"{self.document_number}|{self.doc_name}|{self.article_no}")


def unique_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def clean_article_ref(ref: str) -> str:
    return ARTICLE_DIEU_SUFFIX_RE.sub(r"\1", str(ref or "").strip())


def final_ai_answer(state: dict[str, Any]) -> str:
    for msg in reversed(state.get("messages", []) or []):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", None):
            answer = str(getattr(msg, "content", "") or "").strip()
            return TECHNICAL_BLOCK_RE.sub("", answer).strip()
    return ""


def answer_check_result(state: dict[str, Any]) -> dict[str, Any]:
    check = state.get("answer_check", {}) or {}
    if not isinstance(check, dict):
        check = {}

    status = str(check.get("status") or "").strip().lower()
    if status not in {"pass", "corrected", "failed"}:
        status = ""

    confidence = str(check.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = ""

    issues = []
    for issue in check.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        issues.append(
            {
                "severity": str(issue.get("severity") or "warning").strip() or "warning",
                "type": str(issue.get("type") or "unknown").strip() or "unknown",
                "quote": str(issue.get("quote") or "").strip(),
                "reason": str(issue.get("reason") or "").strip(),
            }
        )

    corrected_answer = str(check.get("corrected_answer") or "").strip()
    if status != "corrected":
        corrected_answer = ""

    return {
        "status": status,
        "confidence": confidence,
        "corrected_answer": corrected_answer,
        "issues": issues,
    }


def checked_final_answer(state: dict[str, Any]) -> str:
    check = answer_check_result(state)
    if check.get("status") == "corrected" and check.get("corrected_answer"):
        return str(check["corrected_answer"]).strip()
    return final_ai_answer(state)


def choose_chunk_ids(state: dict[str, Any]) -> list[str]:
    candidate_ids = state.get("relevant_chunk_ids", []) or []
    return unique_keep_order(candidate_ids)


def _truncate_snippet(text: str, max_chars: int = 900) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(_string_leaves(item))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_string_leaves(item))
        return output
    return []


def _text_values(value: Any, keys: set[str]) -> list[str]:
    output = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in keys:
                output.extend(_string_leaves(item))
            elif isinstance(item, (dict, list, tuple)):
                output.extend(_text_values(item, keys))
    elif isinstance(value, (list, tuple)):
        for item in value:
            output.extend(_text_values(item, keys))
    return output


def message_content_text(message: Any) -> str:
    return _content_to_text(getattr(message, "content", ""))


def message_reasoning_text(message: Any) -> str:
    reasoning_keys = {"reasoning", "reasoning_content"}
    candidates = []
    for attr in ("additional_kwargs", "response_metadata"):
        candidates.extend(_text_values(getattr(message, attr, {}) or {}, reasoning_keys))

    content = getattr(message, "content", "")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "reasoning" or "reasoning" in item_type:
                candidates.extend(_text_values(item, {"reasoning", "text", "content"}))

    output = []
    seen = set()
    for candidate in candidates:
        text = str(candidate or "")
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return "".join(output)


def is_stream_chunk(message: Any) -> bool:
    return type(message).__name__.endswith("Chunk")


def has_tool_call_chunk(message: Any) -> bool:
    return bool(
        getattr(message, "tool_call_chunks", None)
        or getattr(message, "tool_calls", None)
        or (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls")
    )


def normalize_graph_stream_item(item: Any) -> tuple[str, Any]:
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
    ):
        return item[0], item[1]
    if isinstance(item, dict) and "type" in item and "data" in item:
        return str(item["type"]), item["data"]
    return "values", item


def graph_message_parts(payload: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(payload, tuple) and len(payload) == 2:
        message, metadata = payload
        return message, dict(metadata or {})
    return payload, {}


def parse_citation_doc_ids(body: str) -> list[str] | None:
    if not body or " " in body:
        return None
    output = []
    seen = set()
    for raw_doc_id in body.split(","):
        doc_id = raw_doc_id.strip()
        if not doc_id.startswith("DOC_") or not doc_id[4:].isdigit():
            return None
        if doc_id not in seen:
            seen.add(doc_id)
            output.append(doc_id)
    return output or None


class CitationMarkerAssembler:
    def __init__(self):
        self._state = "text"
        self._prefix_index = 0
        self._prefix_buffer = ""
        self._body = ""
        self._close_seen = False

    def feed(self, text: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        text_buffer: list[str] = []

        def emit_text(value: str) -> None:
            if value:
                text_buffer.append(value)

        def flush_text() -> None:
            if text_buffer:
                output.append({"type": "text", "text": "".join(text_buffer)})
                text_buffer.clear()

        def emit_citation(doc_ids: list[str]) -> None:
            flush_text()
            output.append({"type": "citation", "doc_ids": doc_ids})

        def reset() -> None:
            self._state = "text"
            self._prefix_index = 0
            self._prefix_buffer = ""
            self._body = ""
            self._close_seen = False

        for char in str(text or ""):
            if self._state == "text":
                if char == CITATION_PREFIX[0]:
                    self._state = "prefix"
                    self._prefix_index = 1
                    self._prefix_buffer = char
                else:
                    emit_text(char)
                continue

            if self._state == "prefix":
                expected = CITATION_PREFIX[self._prefix_index]
                self._prefix_buffer += char
                if char == expected:
                    self._prefix_index += 1
                    if self._prefix_index == len(CITATION_PREFIX):
                        self._state = "body"
                        self._body = ""
                        self._close_seen = False
                    continue

                emit_text(self._prefix_buffer)
                reset()
                continue

            if self._state != "body":
                reset()
                emit_text(char)
                continue

            if self._close_seen:
                if char == "]":
                    doc_ids = parse_citation_doc_ids(self._body)
                    if doc_ids:
                        emit_citation(doc_ids)
                    reset()
                    continue
                self._body += "]" + char
                self._close_seen = False
            elif char == "]":
                self._close_seen = True
            else:
                self._body += char

            if len(self._body) > MAX_CITATION_BODY_CHARS:
                reset()

        flush_text()
        return output

    def finish(self) -> list[dict[str, Any]]:
        if self._state == "prefix":
            text = self._prefix_buffer
            self.__init__()
            return [{"type": "text", "text": text}] if text else []
        self.__init__()
        return []


def parse_answer_segments(answer: str) -> list[dict[str, Any]]:
    assembler = CitationMarkerAssembler()
    segments = assembler.feed(answer)
    segments.extend(assembler.finish())
    return merge_text_segments(segments)


def merge_text_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in segments:
        if segment.get("type") == "text":
            text = str(segment.get("text") or "")
            if not text:
                continue
            if output and output[-1].get("type") == "text":
                output[-1]["text"] += text
            else:
                output.append({"type": "text", "text": text})
        elif segment.get("type") == "citation":
            doc_ids = unique_keep_order(segment.get("doc_ids", []) or [])
            if doc_ids:
                output.append({"type": "citation", "doc_ids": doc_ids})
    return output


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _favicon_from_domain(domain: str) -> str:
    return f"https://{domain}/favicon.ico" if domain else ""


def _compact_label(title: str, article_no: str = "", max_chars: int = 42) -> str:
    label = " ".join(str(title or article_no or "Nguồn pháp lý").split())
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 3].rstrip() + "..."


def _legal_basis_line(source: dict[str, Any]) -> str:
    doc_name = str(source.get("doc_name") or source.get("title") or "Văn bản pháp luật").strip()
    document_number = str(source.get("document_number") or "").strip()
    article_no = str(source.get("article_no") or "").strip()
    article_title = str(source.get("article_title") or "").strip()

    doc_label = doc_name
    if document_number and document_number not in doc_name:
        doc_label = f"{doc_name}, {document_number}" if doc_name else document_number
    article_label = article_no or "Điều khoản liên quan"
    if article_title:
        article_label = f"{article_label} ({article_title})"
    return f"**{doc_label} — {article_label}**"


def compose_plain_answer(answer_text: str, legal_basis: list[str], disclaimer: str) -> str:
    parts = [str(answer_text or "").strip()]
    if legal_basis:
        basis = "\n".join(f"- {item}" for item in legal_basis)
        parts.append(f"## Cơ Sở Pháp Lý\n{basis}")
    if disclaimer:
        parts.append(f"## Lưu Ý\n{disclaimer}")
    return "\n\n".join(part for part in parts if part)


def clean_plain_answer_text(text: str) -> str:
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", str(text or ""))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def registry_metadata_loaded(doc: dict[str, Any]) -> bool:
    return any(
        str(doc.get(key) or "").strip()
        for key in (
            "document_number",
            "doc_name",
            "article_no",
            "article_title",
        )
    )


class CitationSourceRegistry:
    def __init__(self, resolver: "CitationResolver"):
        self.resolver = resolver
        self.sources: dict[str, dict[str, Any]] = {}

    def ensure_sources(
        self,
        doc_ids: list[str],
        state: dict[str, Any] | None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        state = state or {}
        doc_registry = state.get("doc_registry", {}) or {}
        normalized_doc_ids = unique_keep_order(doc_ids)
        chunk_ids_to_resolve = []
        registry_by_doc_id = {}

        for doc_id in normalized_doc_ids:
            registry_doc = doc_registry.get(doc_id) or {}
            if not registry_doc:
                continue
            registry_by_doc_id[doc_id] = registry_doc
            chunk_id = str(registry_doc.get("chunk_id") or "").strip()
            if chunk_id and not registry_metadata_loaded(registry_doc):
                chunk_ids_to_resolve.append(chunk_id)

        resolved = self.resolver.resolve(chunk_ids_to_resolve) if chunk_ids_to_resolve else {
            "citations": [],
            "relevant_docs": [],
            "relevant_articles": [],
        }
        resolved_by_chunk = {
            str(citation.get("chunk_id") or ""): citation
            for citation in resolved.get("citations", []) or []
        }

        available_doc_ids = []
        new_sources = []
        for doc_id in normalized_doc_ids:
            registry_doc = registry_by_doc_id.get(doc_id)
            if not registry_doc:
                continue

            chunk_id = str(registry_doc.get("chunk_id") or "").strip()
            citation = resolved_by_chunk.get(chunk_id, {})
            source = self._build_source(doc_id, registry_doc, citation)

            if doc_id not in self.sources:
                new_sources.append(source)
            self.sources[doc_id] = source
            available_doc_ids.append(doc_id)

        return available_doc_ids, new_sources

    def _build_source(
        self,
        doc_id: str,
        registry_doc: dict[str, Any],
        citation: dict[str, Any],
    ) -> dict[str, Any]:
        chunk_id = str(registry_doc.get("chunk_id") or citation.get("chunk_id") or "").strip()
        url = str(
            citation.get("url")
            or registry_doc.get("url")
            or registry_doc.get("source_url")
            or ""
        ).strip()
        doc_name = str(
            citation.get("doc_name")
            or registry_doc.get("doc_name")
            or registry_doc.get("title")
            or ""
        ).strip()
        document_number = str(
            citation.get("document_number")
            or registry_doc.get("document_number")
            or ""
        ).strip()
        article_no = str(citation.get("article_no") or registry_doc.get("article_no") or "").strip()
        article_title = str(
            citation.get("article_title")
            or registry_doc.get("article_title")
            or ""
        ).strip()
        title = doc_name or document_number or article_title or doc_id
        domain = _domain_from_url(url)
        excerpt = str(citation.get("snippet") or "").strip()
        if not excerpt:
            excerpt = _truncate_snippet(str(registry_doc.get("text") or "").strip())

        doc_ref = str(citation.get("doc_ref") or "").strip()
        if not doc_ref and document_number and doc_name:
            doc_ref = f"{document_number}|{doc_name}"

        article_ref = str(citation.get("article_ref") or "").strip()
        if not article_ref and document_number and doc_name and article_no:
            article_ref = clean_article_ref(f"{document_number}|{doc_name}|{article_no}")

        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "title": title,
            "label": _compact_label(title, article_no=article_no),
            "domain": domain,
            "url": url,
            "favicon": _favicon_from_domain(domain),
            "excerpt": excerpt,
            "document_number": document_number,
            "doc_name": doc_name,
            "article_no": article_no,
            "article_title": article_title,
            "snippet": excerpt,
            "doc_ref": doc_ref,
            "article_ref": article_ref,
        }


def build_answer_payload(
    raw_answer: str,
    state: dict[str, Any],
    resolver: "CitationResolver",
) -> dict[str, Any]:
    raw_segments = parse_answer_segments(raw_answer)
    source_registry = CitationSourceRegistry(resolver)
    segments: list[dict[str, Any]] = []

    for segment in raw_segments:
        if segment.get("type") == "text":
            text = str(segment.get("text") or "")
            if text:
                segments.append({"type": "text", "text": text})
            continue

        doc_ids = unique_keep_order(segment.get("doc_ids", []) or [])
        available_doc_ids, _ = source_registry.ensure_sources(doc_ids, state)
        if available_doc_ids:
            segments.append(
                {
                    "type": "citation",
                    "doc_ids": available_doc_ids,
                }
            )

    answer_text = clean_plain_answer_text("".join(
        segment.get("text", "")
        for segment in segments
        if segment.get("type") == "text"
    ))

    if not source_registry.sources:
        fallback_ids = choose_chunk_ids(state)
        fallback_doc_ids = []
        reverse_map = {
            str(real_id): str(fake_id)
            for fake_id, real_id in (state.get("evidence_id_map", {}) or {}).items()
        }
        for chunk_id in fallback_ids:
            doc_id = reverse_map.get(str(chunk_id))
            if doc_id:
                fallback_doc_ids.append(doc_id)
        source_registry.ensure_sources(fallback_doc_ids, state)

    sources = source_registry.sources
    citations = list(sources.values())
    legal_basis = unique_keep_order(
        _legal_basis_line(source)
        for source in citations
    )
    relevant_docs = unique_keep_order(
        str(source.get("doc_ref") or "")
        for source in citations
        if source.get("doc_ref")
    )
    relevant_articles = unique_keep_order(
        clean_article_ref(str(source.get("article_ref") or ""))
        for source in citations
        if source.get("article_ref")
    )

    return {
        "answer_text": answer_text,
        "answer": compose_plain_answer(answer_text, legal_basis, DISCLAIMER_TEXT),
        "segments": segments,
        "sources": sources,
        "citations": citations,
        "legal_basis": legal_basis,
        "disclaimer": DISCLAIMER_TEXT,
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }


def sanitize_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {}

    targets = []
    for target in plan.get("search_targets", []) or []:
        targets.append(
            {
                "target_id": str(target.get("target_id") or ""),
                "purpose": str(target.get("purpose") or ""),
                "expected_evidence_type": str(target.get("expected_evidence_type") or ""),
                "bm25_query": str(target.get("bm25_query") or ""),
                "dense_query": str(target.get("dense_query") or ""),
            }
        )

    return {
        "intent": str(plan.get("intent") or ""),
        "actors": [str(item) for item in plan.get("actors", []) or [] if str(item).strip()],
        "events": [str(item) for item in plan.get("events", []) or [] if str(item).strip()],
        "legal_issues": [
            str(item) for item in plan.get("legal_issues", []) or [] if str(item).strip()
        ],
        "metadata_filters": plan.get("metadata_filters", {}) or {},
        "search_targets": targets,
    }


class CitationResolver:
    def __init__(self, es_host: str, index_name: str):
        self.es = Elasticsearch(es_host)
        self.index_name = index_name
        self._source_cache: dict[str, dict[str, Any] | None] = {}

    def _load_sources(self, chunk_ids: list[str]) -> list[tuple[str, dict[str, Any]]]:
        normalized_ids = unique_keep_order(chunk_ids)
        ids_to_fetch = [cid for cid in normalized_ids if cid not in self._source_cache]
        if ids_to_fetch:
            docs = self.es.mget(index=self.index_name, body={"ids": ids_to_fetch}).get("docs", [])
            for doc in docs:
                doc_id = str(doc.get("_id") or "")
                self._source_cache[doc_id] = doc.get("_source") if doc.get("found") else None

            returned_ids = {str(doc.get("_id") or "") for doc in docs}
            for missing_id in set(ids_to_fetch) - returned_ids:
                self._source_cache[missing_id] = None

        output = []
        for chunk_id in normalized_ids:
            source = self._source_cache.get(chunk_id)
            if source:
                output.append((chunk_id, source))
        return output

    def resolve(self, chunk_ids: list[str]) -> dict[str, Any]:
        citations = []
        doc_refs = []
        article_refs = []

        for chunk_id, src in self._load_sources(chunk_ids):
            document_number = str(src.get("document_number") or "").strip()
            doc_name = str(src.get("raw_title") or "").strip()
            article_no = str(src.get("article_no") or "").strip()
            article_title = str(src.get("article_title") or "").strip()
            snippet = _truncate_snippet(str(src.get("raw_content") or "").strip())

            citation = Citation(
                chunk_id=chunk_id,
                document_number=document_number,
                doc_name=doc_name,
                article_no=article_no,
                article_title=article_title,
                snippet=snippet,
                url=str(src.get("url") or "").strip(),
            )
            citations.append(citation.to_dict())
            if citation.doc_ref:
                doc_refs.append(citation.doc_ref)
            if citation.article_ref:
                article_refs.append(citation.article_ref)

        return {
            "citations": citations,
            "relevant_docs": unique_keep_order(doc_refs),
            "relevant_articles": unique_keep_order(article_refs),
        }


def get_graph_app():
    os.environ.setdefault("CHECKPOINTER_BACKEND", "none")
    os.environ.setdefault("AUTO_INIT_RETRIEVAL_RESOURCES", "false")

    from Agents import graph as graph_module

    graph_mode = os.getenv("CHAT_GRAPH_MODE", "stateless").strip().lower()
    if graph_mode == "persistent":
        return graph_module.app
    return graph_module.stateless_app


def iter_graph_states(question: str, thread_id: str):
    app = get_graph_app()
    question_id = uuid.uuid5(uuid.NAMESPACE_URL, thread_id).int % 2_147_483_647
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"question_id": question_id, "question": question}
    yield from app.stream(payload, config, stream_mode="values")


def iter_graph_events(question: str, thread_id: str):
    app = get_graph_app()
    question_id = uuid.uuid5(uuid.NAMESPACE_URL, thread_id).int % 2_147_483_647
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"question_id": question_id, "question": question}
    for item in app.stream(payload, config, stream_mode=["values", "updates", "messages"]):
        mode, data = normalize_graph_stream_item(item)
        yield SimpleNamespace(mode=mode, data=data)


def initialize_retrieval_resources() -> None:
    os.environ.setdefault("AUTO_INIT_RETRIEVAL_RESOURCES", "false")
    from Agents.tools.search_legal import initialize_resources

    initialize_resources()
