"""Đánh giá A/B pipeline Vietnamese Legal RAG production.

Config A chạy graph đầy đủ với Elasticsearch + Qdrant + ViRanker.
Config B giữ nguyên planner/compressor/generator nhưng dùng Qdrant dense-only và
không rerank. Bốn metric chuẩn được chấm bằng RAGAS; một lượt LLM-as-a-Judge độc
lập chấm correctness và cung cấp lý do lỗi. Nếu chưa cài RAGAS, chế độ ``auto``
dùng chính LLM judge để chấm cả bốn metric thay vì làm hỏng cả lượt đánh giá.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parents[1]
MAIN_DIR = PROJECT_ROOT / "src" / "main"
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RAW_OUTPUTS_PATH = EVALUATION_DIR / "rag_outputs.json"
DETAILS_PATH = EVALUATION_DIR / "evaluation_details.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
DEFAULT_PRODUCTION_PYTHON = Path("/home/uet/miniconda3/envs/gemma/bin/python")

CONFIGS = {
    "hybrid_rerank": "Config A — Elasticsearch + Qdrant (RRF) + ViRanker",
    "dense_only": "Config B — Qdrant dense-only, không ViRanker",
}
METRIC_NAMES = (
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
    "llm_judge_correctness",
)


class EvaluationDependencyError(RuntimeError):
    """Dependency tùy chọn của evaluator chưa được cài đặt."""


def _ensure_production_python() -> None:
    """Re-exec bằng conda ``gemma`` nếu interpreter hiện tại thiếu LangGraph."""
    if importlib.util.find_spec("langgraph") is not None:
        return

    configured = Path(
        os.getenv("LEGAL_RAG_PYTHON", str(DEFAULT_PRODUCTION_PYTHON))
    ).expanduser()
    current = Path(sys.executable).resolve()
    if configured.exists() and configured.resolve() != current:
        print(
            f"Interpreter {current} thiếu langgraph; chuyển sang {configured}.",
            flush=True,
        )
        os.execv(
            str(configured),
            [str(configured), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    raise EvaluationDependencyError(
        "Environment hiện tại thiếu langgraph và không tìm thấy Python production. "
        "Hãy activate conda env 'gemma' hoặc đặt LEGAL_RAG_PYTHON tới interpreter "
        "đã cài src/main/requirements.txt."
    )


def _load_environment() -> None:
    """Load env project hoặc env production được chỉ định mà không ghi đè shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = os.getenv("LEGAL_ASSISTANT_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(MAIN_DIR / ".env", override=False)


def _configure_production_runtime() -> None:
    """Chuẩn bị imports/env production cho cả generation và ``--score-only``."""
    _load_environment()
    os.environ["CHECKPOINTER_BACKEND"] = "none"
    # Production có thể để max_tokens sát toàn bộ context window. Evaluation
    # luôn có prompt/evidence đi kèm nên cần một giới hạn đầu ra an toàn hơn.
    os.environ["LLM_MAX_TOKENS"] = os.getenv("EVAL_LLM_MAX_TOKENS", "8192")
    os.environ.setdefault("LLM_BASE_URL", "http://localhost:8006/v1")
    os.environ.setdefault("LLM_API_KEY", "EMPTY")
    os.environ.setdefault("LLM_MODEL", "qwen3.5-9b")
    os.environ.setdefault("AUTO_INIT_RETRIEVAL_RESOURCES", "false")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    main_path = str(MAIN_DIR)
    if main_path not in sys.path:
        sys.path.insert(0, main_path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict[str, Any]]:
    """Load và validate golden dataset."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Golden dataset phải là một JSON array không rỗng.")

    normalized = []
    seen_questions = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Mẫu #{index} phải là object.")
        question = str(item.get("question") or "").strip()
        expected_answer = str(item.get("expected_answer") or "").strip()
        expected_context = item.get("expected_context") or []
        if isinstance(expected_context, str):
            expected_context = [expected_context]
        expected_context = [str(value).strip() for value in expected_context if str(value).strip()]
        if not question or not expected_answer or not expected_context:
            raise ValueError(
                f"Mẫu #{index} phải có question, expected_answer và expected_context."
            )
        if question in seen_questions:
            raise ValueError(f"Câu hỏi trùng ở mẫu #{index}: {question}")
        seen_questions.add(question)
        normalized.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_context": expected_context,
            }
        )
    return normalized


def _unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _article_ref(document: dict[str, Any]) -> str:
    number = str(document.get("document_number") or "").strip()
    title = str(document.get("doc_name") or document.get("title") or "").strip()
    article = str(document.get("article_no") or "").strip()
    if number and title and article:
        return f"{number}|{title}|{article}"
    return ""


def _doc_ref(document: dict[str, Any]) -> str:
    number = str(document.get("document_number") or "").strip()
    title = str(document.get("doc_name") or document.get("title") or "").strip()
    if number and title:
        return f"{number}|{title}"
    return ""


def _extract_contexts(state: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Lấy đúng evidence đã được graph cấp cho generator, kèm citation refs."""
    contexts: list[str] = []
    docs: list[dict[str, Any]] = []
    seen_chunk_ids = set()

    for document in (state.get("doc_registry") or {}).values():
        if not isinstance(document, dict):
            continue
        chunk_id = str(document.get("chunk_id") or "").strip()
        text = str(document.get("text") or "").strip()
        identity = chunk_id or text
        if text and identity not in seen_chunk_ids:
            seen_chunk_ids.add(identity)
            contexts.append(text)
            docs.append(document)

    if not contexts:
        relevant_ids = set(state.get("relevant_chunk_ids") or [])
        for document in state.get("retrieved_documents") or []:
            if not isinstance(document, dict):
                continue
            chunk_id = str(document.get("id") or "").strip()
            if relevant_ids and chunk_id not in relevant_ids:
                continue
            text = str(document.get("text") or "").strip()
            identity = chunk_id or text
            if text and identity not in seen_chunk_ids:
                seen_chunk_ids.add(identity)
                contexts.append(text)
                docs.append(document)

    return (
        contexts,
        _unique_strings(_doc_ref(document) for document in docs),
        _unique_strings(_article_ref(document) for document in docs),
    )


class ProductionRAGPipeline:
    """Adapter gọi trực tiếp LangGraph production trong ``src/main``."""

    def __init__(self) -> None:
        _configure_production_runtime()

        from Agents import graph as graph_module
        from Web.runtime import checked_final_answer

        self._app = graph_module.stateless_app
        self._checked_final_answer = checked_final_answer

    def run(self, question: str, retrieval_config: str) -> dict[str, Any]:
        if retrieval_config not in CONFIGS:
            raise ValueError(f"Unknown retrieval config: {retrieval_config}")
        digest = hashlib.sha256(
            f"{retrieval_config}\0{question}".encode("utf-8")
        ).digest()
        question_id = int.from_bytes(digest[:4], "big") % 2_147_483_647
        state = self._app.invoke(
            {
                "question_id": question_id,
                "question": question,
                "retrieval_config": retrieval_config,
            }
        )
        answer = self._checked_final_answer(state)
        contexts, relevant_docs, relevant_articles = _extract_contexts(state)
        if not answer:
            raise RuntimeError("Production graph không trả về câu trả lời cuối.")
        return {
            "answer": answer,
            "contexts": contexts,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
            "answer_check": state.get("answer_check") or {},
        }


def collect_rag_outputs(
    rag_pipeline: ProductionRAGPipeline,
    golden_dataset: list[dict[str, Any]],
    configs: list[str],
    output_path: Path = RAW_OUTPUTS_PATH,
    *,
    resume: bool = True,
    fail_fast: bool = False,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """Chạy graph song song, bỏ qua case hoàn tất và checkpoint từng kết quả."""
    if max_workers <= 0:
        raise ValueError("max_workers phải lớn hơn 0.")

    records: list[dict[str, Any]] = []
    if resume and output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            records = cached

    cache = {
        (str(row.get("config")), str(row.get("question"))): row
        for row in records
        if row.get("answer") and not row.get("error")
    }
    cases = [
        (position, config, item)
        for position, (config, item) in enumerate(
            (config, item) for config in configs for item in golden_dataset
        )
    ]
    results_by_position: dict[int, dict[str, Any]] = {}
    pending = []

    for position, config, item in cases:
        key = (config, item["question"])
        if key in cache:
            # Gold answer/context có thể được chỉnh sau lần chạy RAG; luôn dùng
            # metadata mới nhưng giữ answer/context đã sinh thành công.
            results_by_position[position] = {
                **cache[key],
                "ground_truth": item["expected_answer"],
                "expected_context": item["expected_context"],
            }
            print(f"[skip RAG] {config}: {item['question'][:72]}")
        else:
            pending.append((position, config, item))

    def ordered_results() -> list[dict[str, Any]]:
        return [results_by_position[index] for index in sorted(results_by_position)]

    def run_case(config: str, item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        record = {
            "config": config,
            "question": item["question"],
            "ground_truth": item["expected_answer"],
            "expected_context": item["expected_context"],
            "answer": "",
            "contexts": [],
            "relevant_docs": [],
            "relevant_articles": [],
            "answer_check": {},
            "latency_sec": 0.0,
            "error": "",
        }
        try:
            record.update(rag_pipeline.run(item["question"], config))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if fail_fast:
                raise
        finally:
            record["latency_sec"] = round(time.perf_counter() - started, 3)
        return record

    if pending:
        worker_count = min(max_workers, len(pending))
        print(
            f"Run {len(pending)} case chưa có bằng {worker_count} worker; "
            f"skip {len(cases) - len(pending)} case đã hoàn tất."
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for position, config, item in pending:
                print(f"[queue RAG] {config}: {item['question'][:72]}")
                future = executor.submit(run_case, config, item)
                future_map[future] = (position, config, item["question"])

            try:
                for completed, future in enumerate(as_completed(future_map), start=1):
                    position, config, question = future_map[future]
                    record = future.result()
                    results_by_position[position] = record
                    _write_json(output_path, ordered_results())
                    status = "error" if record.get("error") else "done"
                    print(
                        f"[{completed}/{len(pending)} {status}] "
                        f"{config}: {question[:72]}"
                    )
            except Exception:
                for future in future_map:
                    future.cancel()
                _write_json(output_path, ordered_results())
                raise
    else:
        print(f"Không còn case RAG cần chạy; đã skip toàn bộ {len(cases)} case.")

    output = ordered_results()
    _write_json(output_path, output)
    return output


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return max(0.0, min(1.0, score))


class ProductionLegalEmbeddings:
    """LangChain-compatible adapter cho VietLegal Harrier production."""

    def __init__(self) -> None:
        from Agents.tools.search_legal import get_embedding_model

        self._model = get_embedding_model()

    def _encode(self, texts: list[str], *, queries: bool) -> list[list[float]]:
        if queries:
            texts = [
                "Instruct: Given a Vietnamese legal question, retrieve relevant "
                f"legal passages that answer the question\nQuery: {text}"
                for text in texts
            ]
        vectors = self._model.encode(
            texts,
            batch_size=max(1, min(32, len(texts))),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), queries=False)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], queries=True)[0]


def _get_evaluator_llm():
    """Judge model riêng nếu có, mặc định dùng endpoint/model production."""
    from Agents.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("JUDGE_LLM_MODEL", LLM_MODEL),
        base_url=os.getenv("JUDGE_LLM_BASE_URL", LLM_BASE_URL),
        api_key=os.getenv("JUDGE_LLM_API_KEY", LLM_API_KEY),
        temperature=0.0,
        top_p=0.1,
        max_tokens=int(os.getenv("JUDGE_LLM_MAX_TOKENS", "4096")),
        max_retries=3,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def evaluate_with_ragas(
    records: list[dict[str, Any]],
    *,
    max_workers: int = 10,
) -> dict[int, dict[str, float | None]]:
    """Chấm bốn metric chuẩn bằng RAGAS 0.4.x và model production."""
    try:
        from datasets import Dataset
        from langchain_core.embeddings import Embeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise EvaluationDependencyError(
            "Thiếu RAGAS. Cài dependencies ở requirements.txt hoặc chạy "
            "--evaluator llm-judge."
        ) from exc

    class RagasProductionEmbeddings(ProductionLegalEmbeddings, Embeddings):
        """Gắn adapter production vào interface mà RAGAS kiểm tra kiểu."""

        pass

    llm = _get_evaluator_llm()
    embeddings = RagasProductionEmbeddings()
    scores_by_index: dict[int, dict[str, float | None]] = {}

    for config in CONFIGS:
        indexed = [
            (index, row)
            for index, row in enumerate(records)
            if row.get("config") == config and not row.get("error")
        ]
        if not indexed:
            continue
        dataset = Dataset.from_dict(
            {
                "question": [row["question"] for _, row in indexed],
                "answer": [row["answer"] for _, row in indexed],
                "contexts": [row["contexts"] for _, row in indexed],
                "ground_truth": [row["ground_truth"] for _, row in indexed],
            }
        )
        kwargs = {
            "dataset": dataset,
            "metrics": [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
            "llm": llm,
            "embeddings": embeddings,
            "run_config": RunConfig(max_workers=max_workers),
        }
        try:
            result = evaluate(**kwargs, raise_exceptions=False)
        except TypeError:
            result = evaluate(**kwargs)
        frame = result.to_pandas()
        for (record_index, _), (_, scored) in zip(indexed, frame.iterrows()):
            scores_by_index[record_index] = {
                "faithfulness": _safe_score(scored.get("faithfulness")),
                "answer_relevance": _safe_score(scored.get("answer_relevancy")),
                "context_recall": _safe_score(scored.get("context_recall")),
                "context_precision": _safe_score(scored.get("context_precision")),
            }
    return scores_by_index


JUDGE_PROMPT = """Bạn là giám khảo độc lập đánh giá hệ thống hỏi đáp pháp luật Việt Nam.
Chấm nghiêm ngặt trên thang 0.0 đến 1.0. Chỉ trả về MỘT JSON object hợp lệ.

Định nghĩa:
- faithfulness: mọi khẳng định thực tế trong câu trả lời có được context truy hồi hỗ trợ không.
- answer_relevance: câu trả lời có trực tiếp, đầy đủ và đúng trọng tâm câu hỏi không.
- context_recall: context truy hồi có đủ căn cứ để tái tạo gold answer không.
- context_precision: tỷ lệ context truy hồi thực sự hữu ích; tài liệu thừa làm giảm điểm.
- correctness: mức đúng của câu trả lời so với gold answer, gồm điều kiện, con số, thời hạn và ngoại lệ.

JSON schema bắt buộc:
{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "context_recall": 0.0,
  "context_precision": 0.0,
  "correctness": 0.0,
  "verdict": "pass|borderline|fail",
  "failure_stage": "none|retrieval|generation|both",
  "rationale": "lý do ngắn gọn bằng tiếng Việt"
}
"""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content or "")


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(cleaned[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("LLM judge không trả về JSON object hợp lệ.")


def evaluate_with_llm_judge(
    records: list[dict[str, Any]],
    *,
    max_context_chars: int = 18_000,
    fail_fast: bool = False,
    max_workers: int = 10,
) -> dict[int, dict[str, Any]]:
    """Chấm song song bằng LLM judge, tối đa ``max_workers`` câu cùng lúc."""
    if max_workers <= 0:
        raise ValueError("max_workers phải lớn hơn 0.")

    from langchain_core.messages import HumanMessage, SystemMessage

    judge = _get_evaluator_llm()
    output: dict[int, dict[str, Any]] = {}

    def judge_one(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        context_parts = []
        used_chars = 0
        for context_index, context in enumerate(record.get("contexts") or [], start=1):
            remaining = max_context_chars - used_chars
            if remaining <= 0:
                break
            excerpt = str(context)[:remaining]
            context_parts.append(f"[Context {context_index}]\n{excerpt}")
            used_chars += len(excerpt)

        payload = (
            f"CÂU HỎI:\n{record['question']}\n\n"
            f"GOLD ANSWER:\n{record['ground_truth']}\n\n"
            "CĂN CỨ KỲ VỌNG:\n- "
            + "\n- ".join(record.get("expected_context") or [])
            + "\n\nCÂU TRẢ LỜI HỆ THỐNG:\n"
            + record["answer"]
            + "\n\nCONTEXT ĐÃ TRUY HỒI:\n"
            + ("\n\n".join(context_parts) or "(không có context)")
        )
        try:
            response = judge.invoke(
                [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=payload)]
            )
            parsed = _parse_json_object(_message_text(response))
            result = {
                "faithfulness": _safe_score(parsed.get("faithfulness")),
                "answer_relevance": _safe_score(parsed.get("answer_relevance")),
                "context_recall": _safe_score(parsed.get("context_recall")),
                "context_precision": _safe_score(parsed.get("context_precision")),
                "llm_judge_correctness": _safe_score(parsed.get("correctness")),
                "verdict": str(parsed.get("verdict") or "").strip().lower(),
                "failure_stage": str(parsed.get("failure_stage") or "").strip().lower(),
                "rationale": str(parsed.get("rationale") or "").strip(),
            }
        except Exception as exc:
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "llm_judge_correctness": None,
            }
            if fail_fast:
                raise
        return index, result

    pending = [
        (index, record)
        for index, record in enumerate(records)
        if not record.get("error")
    ]
    if not pending:
        return output

    worker_count = min(max_workers, len(pending))
    print(f"Judge {len(pending)} case bằng {worker_count} worker.")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(judge_one, index, record): (index, record)
            for index, record in pending
        }
        try:
            for completed, future in enumerate(as_completed(future_map), start=1):
                index, result = future.result()
                output[index] = result
                record = records[index]
                status = "error" if result.get("error") else "done"
                print(
                    f"[{completed}/{len(pending)} judge {status}] "
                    f"{record['config']}: {record['question'][:68]}"
                )
        except Exception:
            for future in future_map:
                future.cancel()
            raise
    return output


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("config") or ""), str(record.get("question") or "")


def _evaluation_signature(record: dict[str, Any]) -> str:
    """Phát hiện output/gold đã đổi để không reuse điểm chấm cũ sai dữ liệu."""
    payload = {
        "config": record.get("config"),
        "question": record.get("question"),
        "ground_truth": record.get("ground_truth"),
        "expected_context": record.get("expected_context") or [],
        "answer": record.get("answer"),
        "contexts": record.get("contexts") or [],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metrics_complete(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics") or {}
    return all(_safe_score(metrics.get(name)) is not None for name in METRIC_NAMES)


def score_records(
    records: list[dict[str, Any]],
    *,
    evaluator: str = "auto",
    max_context_chars: int = 18_000,
    fail_fast: bool = False,
    max_workers: int = 10,
    cached_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Kết hợp RAGAS/judge và bỏ qua các điểm còn hợp lệ trong details cache."""
    if max_workers <= 0:
        raise ValueError("max_workers phải lớn hơn 0.")

    cached_by_key = {
        _record_key(record): record
        for record in (cached_records or [])
        if isinstance(record, dict)
    }
    compatible_cache: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        cached = cached_by_key.get(_record_key(record))
        if cached and _evaluation_signature(cached) == _evaluation_signature(record):
            compatible_cache[index] = cached

    judge_scores: dict[int, dict[str, Any]] = {}
    pending_judge_indices = []
    for index, record in enumerate(records):
        cached_judge = (compatible_cache.get(index) or {}).get("judge") or {}
        judge_complete = all(
            _safe_score(cached_judge.get(name)) is not None
            for name in METRIC_NAMES
        )
        if judge_complete:
            judge_scores[index] = cached_judge
            print(f"[skip judge] {record['config']}: {record['question'][:68]}")
        elif not record.get("error"):
            pending_judge_indices.append(index)

    if pending_judge_indices:
        pending_records = [records[index] for index in pending_judge_indices]
        new_judge_scores = evaluate_with_llm_judge(
            pending_records,
            max_context_chars=max_context_chars,
            fail_fast=fail_fast,
            max_workers=max_workers,
        )
        for local_index, score in new_judge_scores.items():
            judge_scores[pending_judge_indices[local_index]] = score
    else:
        print("Không còn case cần LLM judge; đã skip toàn bộ điểm judge hợp lệ.")

    ragas_scores: dict[int, dict[str, float | None]] = {}
    pending_ragas_indices = []
    if evaluator in {"auto", "ragas"}:
        for index, record in enumerate(records):
            cached = compatible_cache.get(index) or {}
            cached_source = str(cached.get("metric_source") or "")
            can_reuse = _metrics_complete(cached) and (
                cached_source == "ragas"
                or (evaluator == "auto" and cached_source == "llm_judge")
            )
            if can_reuse:
                if cached_source == "ragas":
                    ragas_scores[index] = {
                        name: _safe_score((cached.get("metrics") or {}).get(name))
                        for name in METRIC_NAMES[:4]
                    }
                print(f"[skip metrics] {record['config']}: {record['question'][:68]}")
            elif not record.get("error"):
                pending_ragas_indices.append(index)

    if pending_ragas_indices:
        pending_records = [records[index] for index in pending_ragas_indices]
        try:
            new_ragas_scores = evaluate_with_ragas(
                pending_records,
                max_workers=max_workers,
            )
            for local_index, score in new_ragas_scores.items():
                ragas_scores[pending_ragas_indices[local_index]] = score
        except EvaluationDependencyError:
            if evaluator == "ragas":
                raise
            print("[warning] RAGAS chưa được cài; dùng LLM judge cho bốn metric.")

    scored_records = []
    for index, record in enumerate(records):
        judge = judge_scores.get(index, {})
        if index in ragas_scores:
            metrics = dict(ragas_scores[index])
            metric_source = "ragas"
        else:
            metrics = {
                name: judge.get(name)
                for name in METRIC_NAMES[:4]
            }
            metric_source = "llm_judge"
        metrics["llm_judge_correctness"] = judge.get("llm_judge_correctness")
        available = [_safe_score(value) for value in metrics.values()]
        available = [value for value in available if value is not None]
        metrics["average"] = statistics.fmean(available) if available else None
        scored_records.append(
            {
                **record,
                "metrics": metrics,
                "metric_source": metric_source,
                "judge": judge,
                "evaluation_signature": _evaluation_signature(record),
            }
        )
    framework = (
        "RAGAS + LLM-as-a-Judge"
        if any(row.get("metric_source") == "ragas" for row in scored_records)
        else "LLM-as-a-Judge"
    )
    return scored_records, framework


def compare_configs(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tính trung bình theo config cho bảng A/B."""
    comparison: dict[str, dict[str, Any]] = {}
    for config, description in CONFIGS.items():
        rows = [row for row in records if row.get("config") == config]
        metric_means: dict[str, float | None] = {}
        for metric in (*METRIC_NAMES, "average"):
            values = [
                _safe_score(row.get("metrics", {}).get(metric))
                for row in rows
            ]
            values = [value for value in values if value is not None]
            metric_means[metric] = statistics.fmean(values) if values else None
        latencies = [
            float(row.get("latency_sec") or 0.0)
            for row in rows
            if not row.get("error")
        ]
        comparison[config] = {
            "description": description,
            "case_count": len(rows),
            "success_count": sum(not row.get("error") for row in rows),
            "metrics": metric_means,
            "mean_latency_sec": statistics.fmean(latencies) if latencies else None,
        }
    return comparison


def _format_score(value: Any) -> str:
    score = _safe_score(value)
    return "N/A" if score is None else f"{score:.3f}"


def _escape_cell(value: Any, max_chars: int = 180) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _recommendations(comparison: dict[str, dict[str, Any]]) -> list[str]:
    valid = [
        (name, details)
        for name, details in comparison.items()
        if details.get("metrics", {}).get("average") is not None
    ]
    if not valid:
        return ["Khởi động đủ Elasticsearch, Qdrant và LLM server rồi chạy lại evaluation."]
    best_name, best = max(valid, key=lambda item: item[1]["metrics"]["average"])
    metrics = best["metrics"]
    recommendations = []
    if (metrics.get("context_recall") or 0) < 0.75:
        recommendations.append(
            "Tăng độ phủ retrieval: cải thiện query decomposition, synonym pháp lý và top-k ứng viên."
        )
    if (metrics.get("context_precision") or 0) < 0.75:
        recommendations.append(
            "Giảm context nhiễu: hiệu chỉnh ViRanker và ngưỡng lọc evidence theo từng search target."
        )
    if (metrics.get("faithfulness") or 0) < 0.80:
        recommendations.append(
            "Siết grounding/citation ở reasoning prompt và từ chối kết luận khi thiếu căn cứ trực tiếp."
        )
    if (metrics.get("llm_judge_correctness") or 0) < 0.80:
        recommendations.append(
            "Bổ sung kiểm tra con số, thời hạn, điều kiện và ngoại lệ so với điều khoản trước khi trả lời."
        )
    if not recommendations:
        recommendations.append(
            f"Giữ {best_name} làm mặc định và mở rộng golden set bằng các câu đa điều khoản khó hơn."
        )
    return recommendations[:3]


def export_results(
    records: list[dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    framework: str,
    output_path: Path = RESULTS_PATH,
) -> None:
    """Xuất báo cáo Markdown gồm A/B, bottom-3, lỗi và khuyến nghị."""
    a = comparison.get("hybrid_rerank", {})
    b = comparison.get("dense_only", {})
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "llm_judge_correctness": "LLM Judge Correctness",
        "average": "Average",
    }
    lines = [
        "# RAG Evaluation Results",
        "",
        f"- Thời điểm chạy: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Framework: **{framework}**",
        f"- Pipeline: `{MAIN_DIR}`",
        f"- Golden samples: **{len({row['question'] for row in records})}**",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ A−B |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in labels.items():
        a_value = a.get("metrics", {}).get(metric)
        b_value = b.get("metrics", {}).get(metric)
        delta = None if a_value is None or b_value is None else a_value - b_value
        lines.append(
            f"| {label} | {_format_score(a_value)} | {_format_score(b_value)} | "
            f"{('N/A' if delta is None else f'{delta:+.3f}')} |"
        )
    lines.extend(
        [
            "",
            "## A/B Comparison",
            "",
            f"- **Config A:** {a.get('description', CONFIGS['hybrid_rerank'])}.",
            f"- **Config B:** {b.get('description', CONFIGS['dense_only'])}.",
            f"- Mean latency A/B: {a.get('mean_latency_sec') or 0:.2f}s / "
            f"{b.get('mean_latency_sec') or 0:.2f}s.",
        ]
    )
    a_avg = a.get("metrics", {}).get("average")
    b_avg = b.get("metrics", {}).get("average")
    if a_avg is not None and b_avg is not None:
        winner = "Config A" if a_avg >= b_avg else "Config B"
        lines.append(
            f"- **Kết luận:** {winner} có điểm trung bình cao hơn "
            f"({_format_score(max(a_avg, b_avg))} so với {_format_score(min(a_avg, b_avg))})."
        )

    ranked = [row for row in records if _safe_score(row.get("metrics", {}).get("average")) is not None]
    ranked.sort(key=lambda row: row["metrics"]["average"])
    lines.extend(
        [
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Config | Question | Average | Failure Stage | Root Cause |",
            "|---:|---|---|---:|---|---|",
        ]
    )
    for rank, row in enumerate(ranked[:3], start=1):
        judge = row.get("judge") or {}
        lines.append(
            f"| {rank} | {row['config']} | {_escape_cell(row['question'], 100)} | "
            f"{_format_score(row['metrics'].get('average'))} | "
            f"{_escape_cell(judge.get('failure_stage') or 'unknown', 24)} | "
            f"{_escape_cell(judge.get('rationale') or 'Không có nhận xét từ judge')} |"
        )
    if not ranked:
        lines.append("| 1 | N/A | Chưa có lượt chấm thành công | N/A | unknown | Kiểm tra services/dependencies |")

    errors = [row for row in records if row.get("error")]
    if errors:
        lines.extend(["", "## Runtime Errors", ""])
        for row in errors:
            lines.append(
                f"- `{row['config']}` — {_escape_cell(row['question'], 100)}: "
                f"`{_escape_cell(row['error'], 220)}`"
            )

    lines.extend(["", "## Recommendations", ""])
    for index, recommendation in enumerate(_recommendations(comparison), start=1):
        lines.append(f"{index}. {recommendation}")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "cd " + str(PROJECT_ROOT),
            "LEGAL_ASSISTANT_ENV_FILE=/path/to/legal_assistant.env \\",
            "  /home/uet/miniconda3/envs/gemma/bin/python group_project/evaluation/eval_pipeline.py",
            "```",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=list(CONFIGS),
        help="Các cấu hình retrieval cần so sánh.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N mẫu đầu.")
    parser.add_argument(
        "--evaluator",
        choices=("auto", "ragas", "llm-judge"),
        default="auto",
        help="auto ưu tiên RAGAS và fallback sang LLM judge.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Chạy lại toàn bộ, không dùng cache RAG hoặc cache điểm chấm cũ.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Không chạy RAG; chấm lại dữ liệu trong rag_outputs.json.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Dừng ngay khi có lỗi.")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Số câu chạy song song tối đa (mặc định: 10).",
    )
    parser.add_argument("--max-context-chars", type=int, default=18_000)
    parser.add_argument("--golden-dataset", type=Path, default=GOLDEN_DATASET_PATH)
    parser.add_argument("--raw-outputs", type=Path, default=RAW_OUTPUTS_PATH)
    parser.add_argument("--details", type=Path, default=DETAILS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_production_python()
    args = parse_args(argv)
    _configure_production_runtime()
    golden_dataset = load_golden_dataset(args.golden_dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")
        golden_dataset = golden_dataset[: args.limit]
    if args.workers <= 0:
        raise ValueError("--workers phải lớn hơn 0.")

    cached_scored_records: list[dict[str, Any]] = []
    if not args.no_resume and args.details.exists():
        details_payload = json.loads(args.details.read_text(encoding="utf-8"))
        if isinstance(details_payload, dict) and isinstance(
            details_payload.get("records"), list
        ):
            cached_scored_records = details_payload["records"]

    if args.score_only:
        if not args.raw_outputs.exists():
            raise FileNotFoundError(f"Không tìm thấy {args.raw_outputs}")
        records = json.loads(args.raw_outputs.read_text(encoding="utf-8"))
        allowed_questions = {item["question"] for item in golden_dataset}
        records = [
            row
            for row in records
            if row.get("config") in args.configs and row.get("question") in allowed_questions
        ]
    else:
        pipeline = ProductionRAGPipeline()
        records = collect_rag_outputs(
            pipeline,
            golden_dataset,
            args.configs,
            args.raw_outputs,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
            max_workers=args.workers,
        )

    scored, framework = score_records(
        records,
        evaluator=args.evaluator,
        max_context_chars=args.max_context_chars,
        fail_fast=args.fail_fast,
        max_workers=args.workers,
        cached_records=cached_scored_records,
    )
    comparison = compare_configs(scored)
    _write_json(
        args.details,
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "framework": framework,
            "pipeline": str(MAIN_DIR),
            "comparison": comparison,
            "records": scored,
        },
    )
    export_results(scored, comparison, framework, args.results)
    print(f"Saved details: {args.details}")
    print(f"Saved report:  {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
