from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "runtime" / "chat_history.db"


def get_db_path() -> Path:
    raw = os.getenv("WEB_DB_PATH", str(DEFAULT_DB_PATH)).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                client_ip TEXT,
                user_agent TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                segments_json TEXT NOT NULL DEFAULT '[]',
                sources_json TEXT NOT NULL DEFAULT '{}',
                legal_basis_json TEXT NOT NULL DEFAULT '[]',
                disclaimer TEXT NOT NULL DEFAULT '',
                answer_check_json TEXT NOT NULL DEFAULT '{}',
                relevant_docs_json TEXT NOT NULL DEFAULT '[]',
                relevant_articles_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                elapsed_sec REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
                ON messages(session_id, created_at);
            """
        )
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        migrations = {
            "segments_json": "ALTER TABLE messages ADD COLUMN segments_json TEXT NOT NULL DEFAULT '[]'",
            "sources_json": "ALTER TABLE messages ADD COLUMN sources_json TEXT NOT NULL DEFAULT '{}'",
            "legal_basis_json": "ALTER TABLE messages ADD COLUMN legal_basis_json TEXT NOT NULL DEFAULT '[]'",
            "disclaimer": "ALTER TABLE messages ADD COLUMN disclaimer TEXT NOT NULL DEFAULT ''",
            "answer_check_json": "ALTER TABLE messages ADD COLUMN answer_check_json TEXT NOT NULL DEFAULT '{}'",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                conn.execute(statement)


def create_session(client_ip: str = "", user_agent: str = "") -> dict[str, Any]:
    now = time.time()
    session_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions(session_id, created_at, updated_at, client_ip, user_agent)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, now, now, client_ip, user_agent),
        )
    return {"session_id": session_id, "created_at": now}


def session_exists(session_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row is not None


def touch_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )


def fail_stale_running_messages(stale_after_sec: int) -> int:
    """Đánh dấu job running đã mất process để lịch sử không hiển thị treo mãi."""
    now = time.time()
    cutoff = now - max(0, int(stale_after_sec or 0))
    recovery_error = "Tiến trình xử lý trước đó đã kết thúc hoặc bị gián đoạn. Bạn có thể thử lại câu hỏi."
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE messages
            SET
                status = 'failed',
                error = CASE
                    WHEN error IS NULL OR error = '' THEN ?
                    ELSE error
                END,
                updated_at = ?
            WHERE status = 'running' AND created_at < ?
            """,
            (recovery_error, now, cutoff),
        )
    return max(0, int(cursor.rowcount or 0))


def list_sessions(client_ip: str, limit: int = 50) -> list[dict[str, Any]]:
    """Liệt kê các cuộc trò chuyện có message của một client, mới nhất trước."""
    normalized_limit = max(1, min(int(limit or 50), 100))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.created_at,
                s.updated_at,
                (
                    SELECT m.content
                    FROM messages AS m
                    WHERE m.session_id = s.session_id AND m.role = 'user'
                    ORDER BY m.created_at ASC
                    LIMIT 1
                ) AS title,
                (
                    SELECT m.content
                    FROM messages AS m
                    WHERE m.session_id = s.session_id AND m.content != ''
                    ORDER BY m.created_at DESC
                    LIMIT 1
                ) AS last_message,
                (
                    SELECT m.status
                    FROM messages AS m
                    WHERE m.session_id = s.session_id
                    ORDER BY m.created_at DESC
                    LIMIT 1
                ) AS last_status,
                (
                    SELECT COUNT(*)
                    FROM messages AS m
                    WHERE m.session_id = s.session_id AND m.role = 'user'
                ) AS turn_count
            FROM sessions AS s
            WHERE s.client_ip = ?
              AND EXISTS (
                  SELECT 1 FROM messages AS m WHERE m.session_id = s.session_id
              )
            ORDER BY s.updated_at DESC, s.created_at DESC
            LIMIT ?
            """,
            (str(client_ip or ""), normalized_limit),
        ).fetchall()

    sessions = []
    for row in rows:
        title = " ".join(str(row["title"] or "Cuộc trò chuyện mới").split())
        last_message = " ".join(str(row["last_message"] or "").split())
        sessions.append(
            {
                "session_id": row["session_id"],
                "title": title[:120],
                "last_message": last_message[:180],
                "last_status": str(row["last_status"] or ""),
                "turn_count": int(row["turn_count"] or 0),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return sessions


def create_message(
    session_id: str,
    role: str,
    content: str,
    status: str = "completed",
    message_id: str | None = None,
) -> dict[str, Any]:
    now = time.time()
    message_id = message_id or str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO messages(
                message_id, session_id, role, content, status,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, role, content, status, now, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
    return {
        "message_id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "status": status,
        "created_at": now,
    }


def update_message(
    message_id: str,
    *,
    content: str | None = None,
    status: str | None = None,
    citations: list[dict[str, Any]] | None = None,
    segments: list[dict[str, Any]] | None = None,
    sources: dict[str, dict[str, Any]] | None = None,
    legal_basis: list[str] | None = None,
    disclaimer: str | None = None,
    answer_check: dict[str, Any] | None = None,
    relevant_docs: list[str] | None = None,
    relevant_articles: list[str] | None = None,
    error: str | None = None,
    elapsed_sec: float | None = None,
) -> None:
    fields = []
    values: list[Any] = []
    if content is not None:
        fields.append("content = ?")
        values.append(content)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if citations is not None:
        fields.append("citations_json = ?")
        values.append(json.dumps(citations, ensure_ascii=False))
    if segments is not None:
        fields.append("segments_json = ?")
        values.append(json.dumps(segments, ensure_ascii=False))
    if sources is not None:
        fields.append("sources_json = ?")
        values.append(json.dumps(sources, ensure_ascii=False))
    if legal_basis is not None:
        fields.append("legal_basis_json = ?")
        values.append(json.dumps(legal_basis, ensure_ascii=False))
    if disclaimer is not None:
        fields.append("disclaimer = ?")
        values.append(disclaimer)
    if answer_check is not None:
        fields.append("answer_check_json = ?")
        values.append(json.dumps(answer_check, ensure_ascii=False))
    if relevant_docs is not None:
        fields.append("relevant_docs_json = ?")
        values.append(json.dumps(relevant_docs, ensure_ascii=False))
    if relevant_articles is not None:
        fields.append("relevant_articles_json = ?")
        values.append(json.dumps(relevant_articles, ensure_ascii=False))
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if elapsed_sec is not None:
        fields.append("elapsed_sec = ?")
        values.append(elapsed_sec)

    updated_at = time.time()
    fields.append("updated_at = ?")
    values.append(updated_at)
    values.append(message_id)

    with _connect() as conn:
        conn.execute(f"UPDATE messages SET {', '.join(fields)} WHERE message_id = ?", values)
        conn.execute(
            """
            UPDATE sessions
            SET updated_at = ?
            WHERE session_id = (
                SELECT session_id FROM messages WHERE message_id = ?
            )
            """,
            (updated_at, message_id),
        )


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "status": row["status"],
        "citations": json.loads(row["citations_json"] or "[]"),
        "segments": json.loads(row["segments_json"] or "[]"),
        "sources": json.loads(row["sources_json"] or "{}"),
        "legal_basis": json.loads(row["legal_basis_json"] or "[]"),
        "disclaimer": row["disclaimer"],
        "answer_check": json.loads(row["answer_check_json"] or "{}"),
        "relevant_docs": json.loads(row["relevant_docs_json"] or "[]"),
        "relevant_articles": json.loads(row["relevant_articles_json"] or "[]"),
        "error": row["error"],
        "elapsed_sec": row["elapsed_sec"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_message(message_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
    return _row_to_message(row) if row else None


def list_messages(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    normalized_limit = max(1, min(int(limit or 100), 500))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM (
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ) AS recent_messages
            ORDER BY created_at ASC
            """,
            (session_id, normalized_limit),
        ).fetchall()
    return [_row_to_message(row) for row in rows]
