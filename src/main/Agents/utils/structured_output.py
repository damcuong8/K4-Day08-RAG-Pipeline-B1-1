from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from Agents.logs.agent_logger import logger


def pydantic_to_dict(model: Any) -> dict:
    if model is None:
        raise ValueError("Structured output returned None")
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def content_to_text(content: Any) -> str:
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


def extract_string_leaves(value: Any) -> list[str]:
    values = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for item in value.values():
            values.extend(extract_string_leaves(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(extract_string_leaves(item))
    return values


def extract_text_values(value: Any, keys: set[str]) -> list[str]:
    values = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in keys:
                values.extend(extract_string_leaves(item))
            elif isinstance(item, (dict, list, tuple)):
                values.extend(extract_text_values(item, keys))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                values.extend(extract_text_values(item, keys))
    return values


def extract_thinking_text(response: Any) -> str:
    """Extract reasoning text from the concrete formats used by vLLM/Qwen/OpenAI/LangChain."""
    reasoning_keys = {
        "reasoning",
        "reasoning_content",
    }
    candidates = []
    for attr in ("additional_kwargs", "response_metadata"):
        candidates.extend(extract_text_values(getattr(response, attr, {}) or {}, reasoning_keys))

    content = getattr(response, "content", response)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "reasoning" or "reasoning" in item_type:
                candidates.extend(extract_text_values(item, {"reasoning", "text", "content"}))

    raw_text = content_to_text(content)
    candidates.extend(
        match.strip()
        for match in re.findall(r"<think>\s*(.*?)\s*</think>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    )
    open_think = re.search(r"<think>\s*(.*)", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if open_think and not re.search(r"</think>", raw_text, flags=re.IGNORECASE):
        candidates.append(open_think.group(1).strip())

    output = []
    seen = set()
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return "\n\n".join(output)


def invoke_structured_tool(chain, payload: dict[str, Any], label: str, max_attempts: int = 3):
    last_error: Exception | None = None
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


def required_structured_parser(schema: type[BaseModel]):
    from langchain_core.output_parsers.openai_tools import PydanticToolsParser

    return PydanticToolsParser(
        tools=[schema],
        first_tool_only=True,
    )


def required_structured_tool(llm, schema: type[BaseModel]):
    return llm.bind_tools([schema], tool_choice="required") | required_structured_parser(schema)


def invoke_structured_tool_with_think(
    prompt,
    llm,
    payload: dict[str, Any],
    schema: type[BaseModel],
    label: str,
    max_attempts: int = 3,
):
    parser = required_structured_parser(schema)
    tool_llm = llm.bind_tools([schema], tool_choice="required")
    messages = prompt.format_messages(**payload)
    last_error: Exception | None = None
    last_think = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = tool_llm.invoke(messages)
            last_think = extract_thinking_text(response) or last_think
            result = parser.invoke(response)
            if result is None:
                raise ValueError("Structured output returned None")
            return result, last_think
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                logger.warning(f"{label} structured output lỗi lần {attempt}/{max_attempts}, retry: {e}")
            else:
                logger.error(f"{label} structured output lỗi sau {max_attempts} lần: {e}")
    raise RuntimeError(f"{label} structured output failed after {max_attempts} attempts") from last_error
