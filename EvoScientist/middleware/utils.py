"""Shared utilities for EvoScientist middleware.

Functions here are used by multiple middleware modules (memory, tool_selector)
and should not depend on any specific middleware class.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


def disable_thinking(model: BaseChatModel) -> BaseChatModel:
    """Return a copy of the model with thinking/reasoning disabled.

    Anthropic's API does not allow extended thinking when ``tool_choice``
    forces tool use (as ``with_structured_output`` does).  Similarly,
    OpenAI reasoning can conflict.  Strip these settings so structured
    output calls work reliably.

    Uses ``model_copy()`` to produce a real new instance — ``bind()`` only
    wraps the model in a ``RunnableBinding`` whose kwargs do NOT override
    first-class Pydantic fields like ``thinking`` on ``ChatAnthropic``.
    """
    updates: dict[str, Any] = {}
    model_kwargs = getattr(model, "model_kwargs", {}) or {}

    if getattr(model, "thinking", None) or "thinking" in model_kwargs:
        updates["thinking"] = None
    if getattr(model, "reasoning", None) or "reasoning" in model_kwargs:
        updates["reasoning"] = None

    if not updates:
        return model

    # Prefer Pydantic model_copy (creates a true new instance with the
    # field cleared) over bind() which only adds invocation kwargs.
    try:
        return model.model_copy(update=updates)
    except Exception:
        # Fallback for non-Pydantic or unusual model classes
        # Note: bind() may not effectively override first-class Pydantic fields
        return model.bind(**updates)


def disable_streaming(model: BaseChatModel) -> BaseChatModel:
    """Return a copy of the model with ``disable_streaming=True``.

    ``BaseChatModel._streaming_disabled()`` (langchain_core
    ``chat_models.py:513``) reads only the instance's Pydantic
    ``disable_streaming`` field — the single gate before langchain routes
    to ``_stream`` / ``_astream`` when a streaming-aware callback handler
    is attached (which langgraph's ``astream_events(v3)`` always does).

    Alternatives that don't work:

    - ``model.bind(disable_streaming=True)``: puts kwargs on a
      ``RunnableBinding``, which ``_streaming_disabled()`` doesn't read.
      Silent no-op.
    - ``model.streaming = False``: only honored when ``streaming`` is
      explicitly in ``model_fields_set``. Provider defaults defeat it
      (``ChatOpenAI.streaming=False`` is already-False and not
      explicit-set; ``ChatGoogleGenerativeAI.streaming=None`` is falsy
      but not ``False``).

    Uses ``model_copy`` to leave the caller's reference untouched. On
    validator failure, falls back to a shallow copy + ``setattr`` so a
    shared/cached model instance (e.g. the main-agent model when
    ``disable_thinking`` returned the original unchanged) is never
    mutated in place.
    """
    try:
        return model.model_copy(update={"disable_streaming": True})
    except Exception:
        import copy as _copy

        copied = _copy.copy(model)
        object.__setattr__(copied, "disable_streaming", True)
        return copied


def append_to_system_message(
    system_message: SystemMessage | None, text: str
) -> SystemMessage:
    """Append a text block to a system message, preserving its metadata.

    Used by the memory and scheduler middleware. Unlike building a fresh
    ``SystemMessage``, ``model_copy`` keeps ``additional_kwargs`` (e.g.
    ``cache_control`` prompt-cache breakpoints), ``id``, ``name`` and
    ``response_metadata`` from the original message.
    """
    existing_blocks = list(system_message.content_blocks) if system_message else []
    new_blocks = [*existing_blocks, {"type": "text", "text": text}]
    if system_message is None:
        return SystemMessage(content=new_blocks)
    return system_message.model_copy(update={"content": new_blocks})
