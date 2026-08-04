import json
import os
import threading
import time
from typing import Any

from Agents.logs.agent_logger import logger


_filter_stats_log_lock = threading.Lock()


def question_preview(question: str, max_chars: int = 160) -> str:
    preview = " ".join(str(question or "").split())
    return preview[:max_chars].rstrip()


def log_filter_stats(label: str, total: int, kept: int, **extra: Any) -> None:
    total = max(0, int(total or 0))
    kept = max(0, int(kept or 0))
    removed = max(0, total - kept)
    removed_pct = (removed / total * 100.0) if total else 0.0
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": label.strip("[]"),
        "input": total,
        "kept": kept,
        "removed": removed,
        "removed_pct": round(removed_pct, 3),
        **extra,
    }
    extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
    if extra_text:
        extra_text = " " + extra_text
    logger.info(
        "%s input=%s kept=%s removed=%s removed_pct=%.1f%%%s",
        label,
        total,
        kept,
        removed,
        removed_pct,
        extra_text,
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
        logger.warning(f"Không ghi được filter stats log: {e}")
