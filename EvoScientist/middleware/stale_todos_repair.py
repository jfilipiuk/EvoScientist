"""Stale-todos repair middleware.

Defensive intervener. When the model emits a terminal AIMessage (no
tool calls) while ``state.todos`` still contains entries in
``in_progress`` or ``pending`` status, force-flip those entries to
``error`` via a ``Command`` state update. Pairs with the
``TODO_DISCIPLINE`` system-prompt rule that asks the model to
reconcile on its own: prompt asks the model to do the right thing;
this middleware repairs the state when it doesn't, so the WebUI
shows an honest ``error`` badge instead of misleading "in flight" /
"queued" badges.

Why ``error`` and not ``completed``:

The two failure modes we've reproduced are
``notes/todos-stale-after-turn-end.md``:

1. Model forgets to reconcile after a real answer — items may or may
   not actually be done; we don't know.
2. LangGraph checkpoint rollback wiped the original ``write_todos``,
   the retry never re-emitted it — items definitely *aren't* done in
   the sense the user expects.

In both cases marking ``completed`` would lie. ``error`` is honest:
the agent walked away without saying these items finished. The
WebUI can render that distinctly so users see "abandoned" rather
than "in progress".

Wired into the main agent only — async sub-agents have their own
dispatch contract and don't manage user-visible todos.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langgraph.types import Command

_logger = logging.getLogger("EvoScientist.repair.stale_todos")


def _terminal_ai_message(response: ModelResponse) -> AIMessage | None:
    """Return the AIMessage from a terminal model response, else None.

    Terminal = AIMessage with no tool calls. While the model is still
    dispatching tools, ``state.todos`` is in motion; repair only makes
    sense once the agent has stopped.
    """
    # ModelCallResult is a union of ModelResponse | AIMessage |
    # ExtendedModelResponse; only ModelResponse exposes `.result`.
    result = getattr(response, "result", None)
    if not result:
        return None
    last = result[-1]
    # ModelResponse.result is list[BaseMessage]; structured-output
    # turns can append a ToolMessage, so narrow before use.
    if not isinstance(last, AIMessage):
        return None
    if last.tool_calls:
        return None
    return last


def _repair(state_todos: Any) -> tuple[list[dict], int, int] | None:
    """Return ``(repaired_list, in_progress_flipped, pending_flipped)``
    or ``None`` when nothing needs repair.
    """
    if not state_todos:
        return None
    repaired: list[dict] = []
    in_progress = 0
    pending = 0
    for entry in state_todos:
        status = entry.get("status")
        if status == "in_progress":
            repaired.append({**entry, "status": "error"})
            in_progress += 1
        elif status == "pending":
            repaired.append({**entry, "status": "error"})
            pending += 1
        else:
            repaired.append(entry)
    if not (in_progress or pending):
        return None
    return repaired, in_progress, pending


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


def _maybe_command(request: ModelRequest, response: ModelResponse) -> Command | None:
    msg = _terminal_ai_message(response)
    if msg is None:
        return None
    state = request.state or {}
    repaired = _repair(state.get("todos"))
    if repaired is None:
        return None
    repaired_todos, in_progress, pending = repaired
    _logger.info(
        "stale_todos_repaired",
        extra={
            "metric": "stale_todos_repaired",
            "thread_id": _thread_id(request),
            "in_progress_flipped": in_progress,
            "pending_flipped": pending,
        },
    )
    return Command(update={"todos": repaired_todos})


class StaleTodosRepairMiddleware(AgentMiddleware):
    """Flip stale ``state.todos`` entries to ``error`` on terminal turns."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        response = handler(request)
        try:
            cmd = _maybe_command(request, response)
        except Exception:
            _logger.debug("stale_todos repair build failed", exc_info=True)
            cmd = None
        if cmd is None:
            return response
        return ExtendedModelResponse(model_response=response, command=cmd)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        response = await handler(request)
        try:
            cmd = _maybe_command(request, response)
        except Exception:
            _logger.debug("stale_todos repair build failed", exc_info=True)
            cmd = None
        if cmd is None:
            return response
        return ExtendedModelResponse(model_response=response, command=cmd)


def create_stale_todos_repair_middleware() -> StaleTodosRepairMiddleware:
    return StaleTodosRepairMiddleware()
