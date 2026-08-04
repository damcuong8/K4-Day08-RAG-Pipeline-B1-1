from typing import Any, Mapping, Optional

from langchain_openai import ChatOpenAI
from Agents.config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_MAX_TOKENS


def _copy_reasoning_fields(message: Any, raw: Mapping[str, Any]) -> Any:
    if not isinstance(raw, Mapping) or not hasattr(message, "additional_kwargs"):
        return message

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return message

    for key in ("reasoning", "reasoning_content"):
        value = raw.get(key)
        if value:
            additional_kwargs.setdefault(key, value)
    return message


def _patch_langchain_openai_reasoning_fields() -> None:
    """Preserve vLLM Chat Completions reasoning fields dropped by ChatOpenAI."""
    try:
        from langchain_openai.chat_models import base as openai_base
    except Exception:
        return

    if getattr(openai_base, "_legal_assistant_reasoning_patch_applied", False):
        return

    original_convert_dict = openai_base._convert_dict_to_message
    original_convert_delta = openai_base._convert_delta_to_message_chunk

    def patched_convert_dict_to_message(raw: Mapping[str, Any]):
        message = original_convert_dict(raw)
        return _copy_reasoning_fields(message, raw)

    def patched_convert_delta_to_message_chunk(raw: Mapping[str, Any], default_class):
        chunk = original_convert_delta(raw, default_class)
        return _copy_reasoning_fields(chunk, raw)

    openai_base._convert_dict_to_message = patched_convert_dict_to_message
    openai_base._convert_delta_to_message_chunk = patched_convert_delta_to_message_chunk
    openai_base._legal_assistant_reasoning_patch_applied = True


_patch_langchain_openai_reasoning_fields()


def get_llm(
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 20,
    enable_thinking: bool = True,
    thinking_token_budget: Optional[int] = None,
) -> ChatOpenAI:
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}
    }
    if top_k > 0:
        extra_body["top_k"] = top_k
    if thinking_token_budget is not None:
        extra_body["thinking_budget_tokens"] = thinking_token_budget

    return ChatOpenAI(
        model=LLM_MODEL, 
        base_url=LLM_BASE_URL, 
        api_key=LLM_API_KEY, 
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=3
    )
