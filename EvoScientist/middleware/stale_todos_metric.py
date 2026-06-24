"""Stale-todos observability middleware.

Pure observer. Emits a structured log line every time the model returns
a terminal AIMessage (no tool calls) while ``state.todos`` still
contains entries in ``in_progress`` or ``pending`` status. Used to A/B
validate prompt fixes for the "agent ends turn without reconciling
todos" symptom documented in
``notes/todos-stale-after-turn-end.md``.

No request mutation, no decision-making, no fallback. Failure to emit
is swallowed at debug level — observability must never break the
agent loop.

Wired into the main agent only. Sub-agents have their own dispatch
contract and don't manage user-visible todos.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

_logger = logging.getLogger("EvoScientist.metrics.stale_todos")


def _terminal_ai_message(response: ModelResponse) -> AIMessage | None:
    """Return the AIMessage from a terminal model response, else None.

    Terminal = AIMessage with no tool calls. If the model is about to
    invoke a tool, ``state.todos`` will likely be updated on a
    subsequent turn — we only care about the moment the agent stops.
    """
    result = getattr(response, "result", None)
    if not isinstance(result, list) or not result:
        return None
    last = result[-1]
    if not isinstance(last, AIMessage):
        return None
    if last.tool_calls:
        return None
    return last


def _stale_counts(state_todos: Any) -> tuple[int, int] | None:
    """Return ``(in_progress, pending)`` counts, or ``None`` if nothing stale."""
    if not isinstance(state_todos, list) or not state_todos:
        return None
    in_progress = 0
    pending = 0
    for entry in state_todos:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status == "in_progress":
            in_progress += 1
        elif status == "pending":
            pending += 1
    if in_progress == 0 and pending == 0:
        return None
    return in_progress, pending


def _thread_id(request: ModelRequest) -> str | None:
    runtime = getattr(request, "runtime", None)
    if runtime is None:
        return None
    exec_info = getattr(runtime, "execution_info", None)
    if exec_info is not None:
        tid = getattr(exec_info, "thread_id", None)
        if tid:
            return str(tid)
    return None


def _content_len(msg: AIMessage) -> int:
    content = msg.content
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        # v1 multi-part content: sum the str parts.
        return sum(len(p) for p in content if isinstance(p, str))
    return 0


def _emit(request: ModelRequest, response: ModelResponse) -> None:
    msg = _terminal_ai_message(response)
    if msg is None:
        return
    state = getattr(request, "state", None) or {}
    todos = state.get("todos") if isinstance(state, dict) else None
    counts = _stale_counts(todos)
    if counts is None:
        return
    in_progress, pending = counts
    _logger.info(
        "stale_todos_at_turn_end",
        extra={
            "metric": "stale_todos_at_turn_end",
            "thread_id": _thread_id(request),
            "in_progress": in_progress,
            "pending": pending,
            "content_len": _content_len(msg),
        },
    )


class StaleTodosMetricMiddleware(AgentMiddleware):
    """Emit a metric when a terminal turn leaves todos in non-final states."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        try:
            _emit(request, response)
        except Exception:
            _logger.debug("stale_todos metric emit failed", exc_info=True)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await handler(request)
        try:
            _emit(request, response)
        except Exception:
            _logger.debug("stale_todos metric emit failed", exc_info=True)
        return response


def create_stale_todos_metric_middleware() -> StaleTodosMetricMiddleware:
    return StaleTodosMetricMiddleware()
