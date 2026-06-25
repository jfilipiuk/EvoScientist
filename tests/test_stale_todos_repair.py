from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from EvoScientist.middleware.stale_todos_repair import (
    StaleTodosRepairMiddleware,
    create_stale_todos_repair_middleware,
)

_REPAIR_LOGGER = "EvoScientist.repair.stale_todos"


def _request(state: dict, thread_id: str | None = "tid-abc") -> SimpleNamespace:
    exec_info = SimpleNamespace(thread_id=thread_id) if thread_id else None
    runtime = SimpleNamespace(execution_info=exec_info, config=None)
    return SimpleNamespace(state=state, runtime=runtime)


def _response(message: AIMessage | None) -> SimpleNamespace:
    return SimpleNamespace(result=[message] if message is not None else [])


def _run(middleware, request, response):
    async def handler(_req):
        return response

    return asyncio.run(middleware.awrap_model_call(request, handler))


def _command_update(result):
    """Pull the Command.update dict from a returned ExtendedModelResponse,
    or return None when the middleware passed the response through."""
    cmd = getattr(result, "command", None)
    if cmd is None:
        return None
    return getattr(cmd, "update", None)


def test_flips_in_progress_and_pending_to_error_when_terminal(caplog):
    state = {
        "todos": [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "in_progress"},
            {"content": "c", "status": "pending"},
        ]
    }
    msg = AIMessage(content="here's my summary")
    with caplog.at_level(logging.INFO, logger=_REPAIR_LOGGER):
        result = _run(
            create_stale_todos_repair_middleware(), _request(state), _response(msg)
        )
    update = _command_update(result)
    assert update is not None
    todos = update["todos"]
    assert todos[0]["status"] == "completed"
    assert todos[1]["status"] == "error"
    assert todos[2]["status"] == "error"
    assert todos[1]["content"] == "b"
    # Log line fired with the right counts
    recs = [r for r in caplog.records if r.name == _REPAIR_LOGGER]
    assert len(recs) == 1
    assert recs[0].in_progress_flipped == 1
    assert recs[0].pending_flipped == 1


def test_no_op_when_message_has_tool_calls(caplog):
    state = {"todos": [{"content": "a", "status": "in_progress"}]}
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "ls", "args": {}, "id": "x", "type": "tool_call"}],
    )
    with caplog.at_level(logging.INFO, logger=_REPAIR_LOGGER):
        result = _run(
            create_stale_todos_repair_middleware(), _request(state), _response(msg)
        )
    assert _command_update(result) is None
    assert [r for r in caplog.records if r.name == _REPAIR_LOGGER] == []


def test_no_op_when_all_todos_completed(caplog):
    state = {
        "todos": [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
        ]
    }
    msg = AIMessage(content="done")
    with caplog.at_level(logging.INFO, logger=_REPAIR_LOGGER):
        result = _run(
            create_stale_todos_repair_middleware(), _request(state), _response(msg)
        )
    assert _command_update(result) is None
    assert [r for r in caplog.records if r.name == _REPAIR_LOGGER] == []


def test_no_op_when_todos_absent(caplog):
    state: dict = {}
    msg = AIMessage(content="hi")
    with caplog.at_level(logging.INFO, logger=_REPAIR_LOGGER):
        result = _run(
            create_stale_todos_repair_middleware(), _request(state), _response(msg)
        )
    assert _command_update(result) is None
    assert [r for r in caplog.records if r.name == _REPAIR_LOGGER] == []


def test_no_op_when_response_result_empty(caplog):
    state = {"todos": [{"content": "a", "status": "pending"}]}
    with caplog.at_level(logging.INFO, logger=_REPAIR_LOGGER):
        result = _run(
            create_stale_todos_repair_middleware(), _request(state), _response(None)
        )
    assert _command_update(result) is None
    assert [r for r in caplog.records if r.name == _REPAIR_LOGGER] == []


def test_preserves_unrelated_todo_fields():
    state = {
        "todos": [
            {
                "content": "a",
                "status": "in_progress",
                "extra": "preserved",
                "subtasks": [1, 2],
            }
        ]
    }
    msg = AIMessage(content="ok")
    result = _run(
        create_stale_todos_repair_middleware(), _request(state), _response(msg)
    )
    todos = _command_update(result)["todos"]
    assert todos[0]["status"] == "error"
    assert todos[0]["content"] == "a"
    assert todos[0]["extra"] == "preserved"
    assert todos[0]["subtasks"] == [1, 2]


def test_factory_returns_middleware_instance():
    assert isinstance(
        create_stale_todos_repair_middleware(), StaleTodosRepairMiddleware
    )


def _mock_config():
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_mode = False
    cfg.auto_approve = False
    cfg.model_fallbacks = None
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    cfg.code_interpreter_timeout = 60
    cfg.code_interpreter_max_result_chars = 6000
    return cfg


def test_main_agent_middleware_includes_stale_todos_repair():
    from unittest.mock import MagicMock, patch

    with (
        patch(
            "EvoScientist.middleware.create_tool_selector_middleware",
            return_value=[MagicMock(), MagicMock()],
        ),
        patch(
            "EvoScientist.EvoScientist._ensure_chat_model",
            return_value=MagicMock(profile={"max_input_tokens": 200_000}),
        ),
        patch("EvoScientist.EvoScientist._ensure_config", return_value=_mock_config()),
    ):
        from EvoScientist.EvoScientist import _get_default_middleware

        middleware = _get_default_middleware()

    assert any(isinstance(m, StaleTodosRepairMiddleware) for m in middleware)


def test_async_subagent_middleware_excludes_stale_todos_repair():
    from unittest.mock import MagicMock, patch

    with (
        patch(
            "EvoScientist.middleware.create_tool_selector_middleware",
            return_value=[MagicMock(), MagicMock()],
        ),
        patch(
            "EvoScientist.EvoScientist._ensure_chat_model",
            return_value=MagicMock(profile={"max_input_tokens": 200_000}),
        ),
        patch("EvoScientist.EvoScientist._ensure_config", return_value=_mock_config()),
    ):
        from EvoScientist.EvoScientist import _get_default_middleware

        middleware = _get_default_middleware(for_async_subagent=True)

    assert not any(isinstance(m, StaleTodosRepairMiddleware) for m in middleware)


def test_prompt_includes_todo_discipline_section():
    from EvoScientist.prompts import get_system_prompt

    prompt = get_system_prompt()
    assert "Todo Discipline" in prompt
    assert "reconcile" in prompt
    assert "completed" in prompt
    assert "carry-over" in prompt
