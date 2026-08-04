from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Web import runtime, storage


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
CHAT_MAX_CONCURRENT = int(os.getenv("CHAT_MAX_CONCURRENT", "1"))
CHAT_RATE_LIMIT_PER_MIN = int(os.getenv("CHAT_RATE_LIMIT_PER_MIN", "6"))
CHAT_JOB_TTL_SEC = int(os.getenv("CHAT_JOB_TTL_SEC", "1800"))
WEB_DEMO_TRACE = os.getenv("WEB_DEMO_TRACE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
WEB_CORS_ORIGINS = [
    item.strip()
    for item in os.getenv("WEB_CORS_ORIGINS", "").split(",")
    if item.strip()
]

chat_semaphore = threading.BoundedSemaphore(max(1, CHAT_MAX_CONCURRENT))


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: float


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    last_message: str = ""
    last_status: str = ""
    turn_count: int = 0
    created_at: float
    updated_at: float


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionSummary] = Field(default_factory=list)


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatMessageResponse(BaseModel):
    session_id: str
    user_message_id: str
    message_id: str
    stream_url: str
    status: str


class ChatSyncResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    answer_text: str = ""
    segments: list[dict[str, Any]] = Field(default_factory=list)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    citations: list[dict[str, Any]]
    legal_basis: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    answer_check: dict[str, Any] = Field(default_factory=dict)
    relevant_docs: list[str]
    relevant_articles: list[str]
    elapsed_sec: float


@dataclass
class ChatJob:
    session_id: str
    message_id: str
    question: str
    thread_id: str
    started_at: float = field(default_factory=time.time)
    events: queue.Queue[dict[str, Any] | None] = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_sec: int = 60):
        self.limit = max(1, limit)
        self.window_sec = max(1, window_sec)
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_sec:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = int(max(1, self.window_sec - (now - hits[0])))
                return False, retry_after
            hits.append(now)
            return True, 0


jobs: dict[str, ChatJob] = {}
jobs_lock = threading.Lock()
rate_limiter = InMemoryRateLimiter(CHAT_RATE_LIMIT_PER_MIN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    storage.fail_stale_running_messages(CHAT_JOB_TTL_SEC)
    if os.getenv("WEB_PRELOAD_RESOURCES", "false").strip().lower() in {"1", "true", "yes"}:
        runtime.initialize_retrieval_resources()
    yield


app = FastAPI(
    title="R2AI2026 Legal Assistant Web Chat",
    version="1.0.0",
    lifespan=lifespan,
)

if WEB_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=WEB_CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _cleanup_jobs() -> None:
    cutoff = time.time() - CHAT_JOB_TTL_SEC
    with jobs_lock:
        for message_id, job in list(jobs.items()):
            if job.done.is_set() and job.started_at < cutoff:
                jobs.pop(message_id, None)


def _emit(job: ChatJob, event: str, data: dict[str, Any]) -> None:
    job.events.put({"event": event, "data": data})


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _emit_process(job: ChatJob, stage: str, message: str, **data: Any) -> None:
    payload = {"stage": stage, "message": message, **data}
    _emit(job, "process_step", payload)


def _emit_text_chunks(job: ChatJob, event: str, text: str, **data: Any) -> None:
    if not text:
        return
    max_chars = 1200
    for idx in range(0, len(text), max_chars):
        _emit(job, event, {"delta": text[idx : idx + max_chars], **data})


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _extract_tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    tool_calls = []
    raw_calls = getattr(message, "tool_calls", None) or []
    if not raw_calls:
        kwargs = getattr(message, "additional_kwargs", {}) or {}
        raw_calls = kwargs.get("tool_calls") or []

    for index, raw_call in enumerate(raw_calls, start=1):
        if isinstance(raw_call, dict):
            function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
            name = raw_call.get("name") or function.get("name") or ""
            args = raw_call.get("args")
            if args is None:
                args = function.get("arguments", {})
            call_id = raw_call.get("id") or f"tool_call_{index}"
        else:
            name = getattr(raw_call, "name", "") or ""
            args = getattr(raw_call, "args", {})
            call_id = getattr(raw_call, "id", "") or f"tool_call_{index}"

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw_arguments": args}

        tool_calls.append(
            {
                "id": str(call_id),
                "name": str(name),
                "args": _safe_json_value(args if args is not None else {}),
            }
        )

    return tool_calls


def _extract_tool_calls_from_updates(updates: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for message in updates.get("messages", []) or []:
        calls.extend(_extract_tool_calls_from_message(message))
    return calls


def _handle_answer_segment(
    job: ChatJob,
    segment: dict[str, Any],
    source_registry: runtime.CitationSourceRegistry,
    current_state: dict[str, Any],
) -> None:
    if segment.get("type") == "text":
        text = str(segment.get("text") or "")
        if text:
            _emit(job, "answer_delta", {"delta": text})
        return

    if segment.get("type") != "citation":
        return

    doc_ids = runtime.unique_keep_order(segment.get("doc_ids", []) or [])
    available_doc_ids, new_sources = source_registry.ensure_sources(doc_ids, current_state)
    if not available_doc_ids:
        return
    for source in new_sources:
        _emit(job, "source", source)
    _emit(job, "citation", {"doc_ids": available_doc_ids})


def _run_chat_job(job: ChatJob) -> None:
    acquired = False
    started_at = time.time()
    final_state: dict[str, Any] | None = None
    current_state: dict[str, Any] = {}
    sent = set()
    planner_token_streamed = False
    reasoning_token_streamed = False

    try:
        from Agents.config import (
            ENABLE_ANSWER_CHECKER_NODE,
            ES_HOST,
            INDEX_NAME,
        )

        resolver = runtime.CitationResolver(ES_HOST, INDEX_NAME)
        source_registry = runtime.CitationSourceRegistry(resolver)
        answer_assembler = runtime.CitationMarkerAssembler()

        _emit(job, "status", {"stage": "queued", "message": "Đang xếp hàng xử lý..."})
        _emit_process(job, "queued", "Đang xếp hàng xử lý...")
        chat_semaphore.acquire()
        acquired = True
        _emit(job, "status", {"stage": "running", "message": "Đang phân tích câu hỏi..."})
        _emit_process(job, "running", "Đang phân tích câu hỏi...")

        for graph_event in runtime.iter_graph_events(job.question, job.thread_id):
            if graph_event.mode == "messages":
                message_chunk, metadata = runtime.graph_message_parts(graph_event.data)
                if not runtime.is_stream_chunk(message_chunk):
                    continue

                node_name = str(metadata.get("langgraph_node") or "")
                reasoning_delta = runtime.message_reasoning_text(message_chunk)
                content_delta = runtime.message_content_text(message_chunk)

                if node_name == "planner_node":
                    if WEB_DEMO_TRACE and reasoning_delta:
                        planner_token_streamed = True
                        _emit(job, "planner_token", {"delta": reasoning_delta})
                    continue

                if node_name != "reasoning_node":
                    continue

                if WEB_DEMO_TRACE and reasoning_delta:
                    reasoning_token_streamed = True
                    _emit(job, "reasoning_token", {"delta": reasoning_delta})

                if content_delta and not runtime.has_tool_call_chunk(message_chunk):
                    for segment in answer_assembler.feed(content_delta):
                        _handle_answer_segment(job, segment, source_registry, current_state)
                continue

            if graph_event.mode == "updates":
                if not isinstance(graph_event.data, dict):
                    continue
                for node_name, updates in graph_event.data.items():
                    if not isinstance(updates, dict):
                        continue
                    
                    if node_name == "planner_node":
                        planner_think = updates.get("planner_think")
                        if (
                            WEB_DEMO_TRACE
                            and planner_think
                            and not planner_token_streamed
                            and "planner_think_fallback" not in sent
                        ):
                            sent.add("planner_think_fallback")
                            _emit_text_chunks(job, "planner_token", str(planner_think))

                        plan = updates.get("plan")
                        if plan and "planner" not in sent:
                            sent.add("planner")
                            sanitized_plan = runtime.sanitize_plan(plan)
                            _emit(
                                job,
                                "planner",
                                {
                                    "stage": "planner",
                                    "message": "Đã phân tích tình huống pháp lý.",
                                    "plan": sanitized_plan,
                                },
                            )
                            _emit_process(
                                job,
                                "planner",
                                (
                                    f"Đã xác định {len(sanitized_plan.get('legal_issues', []) or [])} vấn đề pháp lý."
                                    if sanitized_plan.get("legal_issues")
                                    else "Đã phân tích tình huống pháp lý."
                                ),
                                plan=sanitized_plan,
                            )

                    if node_name == "batch_hybrid_search_node":
                        docs = updates.get("retrieved_documents")
                        if docs is not None and "retrieval" not in sent:
                            sent.add("retrieval")
                            _emit(
                                job,
                                "retrieval",
                                {
                                    "stage": "retrieval",
                                    "message": "Đã truy hồi và rerank căn cứ pháp lý.",
                                    "document_count": len(docs or []),
                                },
                            )
                            _emit_process(
                                job,
                                "retrieval",
                                "Đang tìm tài liệu và rerank căn cứ pháp lý.",
                                document_count=len(docs or []),
                            )

                    if node_name == "compress_node":
                        evidence = updates.get("extracted_evidence")
                        if evidence and "compression" not in sent:
                            sent.add("compression")
                            relevant_chunk_count = len(updates.get("relevant_chunk_ids", []) or [])
                            total_docs = len(current_state.get("retrieved_documents", []) or [])
                            msg = f"Đã chọn {relevant_chunk_count} nguồn phù hợp trong {total_docs} nguồn." if total_docs > 0 else f"Đã chọn {relevant_chunk_count} nguồn phù hợp."
                            _emit(
                                job,
                                "compression",
                                {
                                    "stage": "compression",
                                    "message": "Đã lọc căn cứ phù hợp để sinh câu trả lời.",
                                    "evidence_chars": len(str(evidence)),
                                    "relevant_chunk_count": relevant_chunk_count,
                                    "total_chunk_count": total_docs,
                                },
                            )
                            _emit_process(
                                job,
                                "compression",
                                msg,
                                evidence_chars=len(str(evidence)),
                                relevant_chunk_count=relevant_chunk_count,
                                total_chunk_count=total_docs,
                            )

                    if node_name == "reasoning_node":
                        tool_calls = _extract_tool_calls_from_updates(updates)
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or "")
                            key = f"tool_call_{call_id or len(sent)}"
                            if key in sent:
                                continue
                            sent.add(key)
                            _emit(
                                job,
                                "tool_call",
                                {
                                    "stage": "tool_call",
                                    "message": "Model gọi công cụ tìm kiếm bổ sung.",
                                    "tool_call": tool_call,
                                },
                            )

                        reasoning_think = updates.get("reasoning_think")
                        if (
                            WEB_DEMO_TRACE
                            and reasoning_think
                            and not reasoning_token_streamed
                            and "reasoning_think_fallback" not in sent
                        ):
                            sent.add("reasoning_think_fallback")
                            _emit_text_chunks(job, "reasoning_token", str(reasoning_think))

                        if ENABLE_ANSWER_CHECKER_NODE and not tool_calls and updates.get("messages") and "answer_check_started" not in sent:
                            sent.add("answer_check_started")
                            _emit_process(
                                job,
                                "answer_check",
                                "Đang kiểm tra độ tin cậy câu trả lời.",
                            )
                    if node_name == "search_tool_node":
                        tool_msgs = updates.get("messages", [])
                        if tool_msgs:
                            num_queries = sum(
                                ((msg.get("artifact", {}) if isinstance(msg, dict) else getattr(msg, "artifact", {})) or {}).get("num_queries", 0)
                                for msg in tool_msgs
                            )
                            search_run_count = sum(1 for k in sent if k.startswith("tool_search_run_")) + 1
                            key = f"tool_search_run_{search_run_count}"
                            sent.add(key)
                            
                            msg_str = f"Đã chạy tìm kiếm bổ sung lần {search_run_count} với {num_queries} câu truy vấn." if num_queries > 0 else f"Đã chạy tìm kiếm bổ sung lần {search_run_count}."
                            _emit_process(
                                job,
                                "tool_search",
                                msg_str,
                                tool_message_count=search_run_count,
                                num_queries=num_queries,
                            )
                    if node_name == "answer_checker_node":
                        answer_check = updates.get("answer_check")
                        if isinstance(answer_check, dict):
                            _emit(
                                job,
                                "checker_result",
                                runtime.answer_check_result({"answer_check": answer_check}),
                            )
                continue

            if graph_event.mode == "values":
                state = graph_event.data
                if not isinstance(state, dict):
                    continue

                final_state = state
                current_state = state

                continue

        if not final_state:
            raise RuntimeError("Graph finished without state")

        for segment in answer_assembler.finish():
            _handle_answer_segment(job, segment, source_registry, current_state)

        raw_model_answer = runtime.final_ai_answer(final_state)
        if not raw_model_answer:
            raise RuntimeError("Graph finished without final answer")

        answer_check = runtime.answer_check_result(final_state)
        raw_answer = runtime.checked_final_answer(final_state)
        if answer_check.get("status") == "corrected" and answer_check.get("corrected_answer"):
            _emit(job, "answer_reset", {"reason": "answer_checker_corrected"})
            corrected_registry = runtime.CitationSourceRegistry(resolver)
            corrected_assembler = runtime.CitationMarkerAssembler()
            for segment in corrected_assembler.feed(raw_answer):
                _handle_answer_segment(job, segment, corrected_registry, final_state)
            for segment in corrected_assembler.finish():
                _handle_answer_segment(job, segment, corrected_registry, final_state)

        answer_payload = runtime.build_answer_payload(raw_answer, final_state, resolver)
        elapsed_sec = round(time.time() - started_at, 3)

        result = {
            "session_id": job.session_id,
            "message_id": job.message_id,
            "answer": answer_payload["answer"],
            "answer_text": answer_payload["answer_text"],
            "segments": answer_payload["segments"],
            "sources": answer_payload["sources"],
            "citations": answer_payload["citations"],
            "legal_basis": answer_payload["legal_basis"],
            "disclaimer": answer_payload["disclaimer"],
            "answer_check": answer_check,
            "relevant_docs": answer_payload["relevant_docs"],
            "relevant_articles": answer_payload["relevant_articles"],
            "elapsed_sec": elapsed_sec,
        }
        job.result = result
        storage.update_message(
            job.message_id,
            content=result["answer"],
            status="completed",
            citations=result["citations"],
            segments=result["segments"],
            sources=result["sources"],
            legal_basis=result["legal_basis"],
            disclaimer=result["disclaimer"],
            answer_check=result["answer_check"],
            relevant_docs=result["relevant_docs"],
            relevant_articles=result["relevant_articles"],
            elapsed_sec=elapsed_sec,
        )
        _emit(job, "answer_done", result)
        _emit(job, "final", result)
    except Exception as exc:
        job.error = str(exc)
        storage.update_message(
            job.message_id,
            status="failed",
            error=str(exc),
            elapsed_sec=round(time.time() - started_at, 3),
        )
        _emit(
            job,
            "error",
            {
                "message": "Hệ thống chưa thể tạo câu trả lời. Vui lòng thử lại sau.",
                "detail": str(exc),
            },
        )
    finally:
        if acquired:
            chat_semaphore.release()
        job.done.set()
        job.events.put(None)


def _create_or_get_session(request: Request, session_id: str | None) -> str:
    if session_id and storage.session_exists(session_id):
        storage.touch_session(session_id)
        return session_id
    session = storage.create_session(
        client_ip=client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    return session["session_id"]


def _validate_question(message: str) -> str:
    question = " ".join(str(message or "").split())
    if not question:
        raise HTTPException(status_code=422, detail="Câu hỏi không được để trống.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Câu hỏi vượt quá giới hạn {MAX_QUESTION_CHARS} ký tự.",
        )
    return question


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(deep: bool = Query(default=False)) -> dict[str, Any]:
    from Agents.config import (
        EMBEDDING_MODEL_PATH,
        ES_HOST,
        LLM_BASE_URL,
        QDRANT_HOST,
        QDRANT_PORT,
        RERANKER_PATH,
        VNCORENLP_DIR,
    )

    checks: dict[str, Any] = {}

    try:
        response = requests.get(ES_HOST, timeout=2)
        checks["elasticsearch"] = {"ok": response.ok, "status_code": response.status_code}
    except Exception as exc:
        checks["elasticsearch"] = {"ok": False, "error": str(exc)}

    try:
        response = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=2)
        checks["qdrant"] = {"ok": response.ok, "status_code": response.status_code}
    except Exception as exc:
        checks["qdrant"] = {"ok": False, "error": str(exc)}

    try:
        base_url = LLM_BASE_URL.rstrip("/")
        response = requests.get(f"{base_url}/models", timeout=2)
        checks["llm_server"] = {"ok": response.ok, "status_code": response.status_code}
    except Exception as exc:
        checks["llm_server"] = {"ok": False, "error": str(exc)}

    checks["model_paths"] = {
        "embedding": Path(EMBEDDING_MODEL_PATH).exists(),
        "reranker": Path(RERANKER_PATH).exists(),
        "vncorenlp": Path(VNCORENLP_DIR).exists(),
    }

    if deep:
        try:
            runtime.initialize_retrieval_resources()
            checks["retrieval_resources"] = {"ok": True}
        except Exception as exc:
            checks["retrieval_resources"] = {"ok": False, "error": str(exc)}

    ok = all(
        value.get("ok", True) if isinstance(value, dict) else bool(value)
        for key, value in checks.items()
        if key != "model_paths"
    ) and all(checks["model_paths"].values())
    return {"status": "ready" if ok else "degraded", "checks": checks}


@app.post("/api/chat/sessions", response_model=CreateSessionResponse)
def create_chat_session(request: Request) -> dict[str, Any]:
    return storage.create_session(
        client_ip=client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )


@app.get("/api/chat/sessions", response_model=ChatSessionListResponse)
def list_chat_sessions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    return {"sessions": storage.list_sessions(client_ip(request), limit=limit)}


@app.get("/api/chat/messages")
def get_chat_messages(session_id: str, limit: int = 100) -> dict[str, Any]:
    if not storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session không tồn tại.")
    return {"session_id": session_id, "messages": storage.list_messages(session_id, limit=limit)}


@app.post("/api/chat/messages", response_model=ChatMessageResponse)
def create_chat_message(payload: ChatMessageRequest, request: Request) -> dict[str, Any]:
    _cleanup_jobs()
    allowed, retry_after = rate_limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Bạn gửi quá nhanh. Thử lại sau {retry_after} giây.",
            headers={"Retry-After": str(retry_after)},
        )

    question = _validate_question(payload.message)
    session_id = _create_or_get_session(request, payload.session_id)
    user_message = storage.create_message(session_id, "user", question, status="completed")
    assistant_message_id = str(uuid.uuid4())
    storage.create_message(
        session_id,
        "assistant",
        "",
        status="running",
        message_id=assistant_message_id,
    )

    job = ChatJob(
        session_id=session_id,
        message_id=assistant_message_id,
        question=question,
        thread_id=session_id,
    )
    with jobs_lock:
        jobs[assistant_message_id] = job

    thread = threading.Thread(target=_run_chat_job, args=(job,), daemon=True)
    thread.start()

    return {
        "session_id": session_id,
        "user_message_id": user_message["message_id"],
        "message_id": assistant_message_id,
        "stream_url": f"/api/chat/stream?session_id={session_id}&message_id={assistant_message_id}",
        "status": "running",
    }


@app.post("/api/chat", response_model=ChatSyncResponse)
def create_chat_sync(payload: ChatMessageRequest, request: Request) -> dict[str, Any]:
    allowed, retry_after = rate_limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Bạn gửi quá nhanh. Thử lại sau {retry_after} giây.",
            headers={"Retry-After": str(retry_after)},
        )

    question = _validate_question(payload.message)
    session_id = _create_or_get_session(request, payload.session_id)
    storage.create_message(session_id, "user", question, status="completed")
    assistant_message_id = str(uuid.uuid4())
    storage.create_message(
        session_id,
        "assistant",
        "",
        status="running",
        message_id=assistant_message_id,
    )
    job = ChatJob(
        session_id=session_id,
        message_id=assistant_message_id,
        question=question,
        thread_id=session_id,
    )
    _run_chat_job(job)
    if job.error or not job.result:
        raise HTTPException(status_code=500, detail=job.error or "Chat job failed")
    return job.result


@app.get("/api/chat/stream")
def stream_chat(session_id: str, message_id: str) -> StreamingResponse:
    with jobs_lock:
        job = jobs.get(message_id)

    if not job or job.session_id != session_id:
        message = storage.get_message(message_id)
        if not message or message["session_id"] != session_id:
            raise HTTPException(status_code=404, detail="Stream không tồn tại.")

        def replay_completed():
            if message["status"] == "completed":
                result = {
                    "session_id": session_id,
                    "message_id": message_id,
                    "answer": message["content"],
                    "answer_text": "".join(
                        segment.get("text", "")
                        for segment in message.get("segments", []) or []
                        if segment.get("type") == "text"
                    ),
                    "segments": message.get("segments", []) or [],
                    "sources": message.get("sources", {}) or {},
                    "citations": message["citations"],
                    "legal_basis": message.get("legal_basis", []) or [],
                    "disclaimer": message.get("disclaimer", "") or "",
                    "answer_check": message.get("answer_check", {}) or {},
                    "relevant_docs": message["relevant_docs"],
                    "relevant_articles": message["relevant_articles"],
                    "elapsed_sec": message["elapsed_sec"] or 0,
                }
                yield _format_sse("answer_done", result)
                yield _format_sse("final", result)
            else:
                yield _format_sse(
                    "error",
                    {
                        "message": "Không thể khôi phục stream cho câu trả lời này.",
                        "detail": message.get("error") or message["status"],
                    },
                )
            yield _format_sse("done", {"message_id": message_id})

        return StreamingResponse(replay_completed(), media_type="text/event-stream")

    def event_generator():
        try:
            while True:
                try:
                    item = job.events.get(timeout=15)
                except queue.Empty:
                    yield _format_sse("ping", {"ts": time.time()})
                    continue

                if item is None:
                    yield _format_sse("done", {"message_id": message_id})
                    break
                yield _format_sse(str(item["event"]), dict(item["data"]))
        finally:
            if job.done.is_set():
                with jobs_lock:
                    jobs.pop(message_id, None)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
