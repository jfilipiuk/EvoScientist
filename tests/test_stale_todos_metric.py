from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from EvoScientist.middleware.stale_todos_metric import (
    StaleTodosMetricMiddleware,
    create_stale_todos_metric_middleware,
)

_METRIC_LOGGER = "EvoScientist.metrics.stale_todos"


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


def test_emits_when_terminal_message_leaves_stale_todos(caplog):
    state = {
        "todos": [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "in_progress"},
            {"content": "c", "status": "pending"},
        ]
    }
    msg = AIMessage(content="final summary, 100 chars or so" * 4)
    with caplog.at_level(logging.INFO, logger=_METRIC_LOGGER):
        _run(create_stale_todos_metric_middleware(), _request(state), _response(msg))
    matches = [r for r in caplog.records if r.name == _METRIC_LOGGER]
    assert len(matches) == 1
    rec = matches[0]
    assert rec.in_progress == 1
    assert rec.pending == 1
    assert rec.thread_id == "tid-abc"
    assert rec.content_len > 0
    assert rec.metric == "stale_todos_at_turn_end"


def test_silent_when_message_has_tool_calls(caplog):
    state = {"todos": [{"content": "a", "status": "in_progress"}]}
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "ls", "args": {}, "id": "x", "type": "tool_call"}],
    )
    with caplog.at_level(logging.INFO, logger=_METRIC_LOGGER):
        _run(create_stale_todos_metric_middleware(), _request(state), _response(msg))
    assert [r for r in caplog.records if r.name == _METRIC_LOGGER] == []


def test_silent_when_all_todos_completed(caplog):
    state = {
        "todos": [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
        ]
    }
    msg = AIMessage(content="done")
    with caplog.at_level(logging.INFO, logger=_METRIC_LOGGER):
        _run(create_stale_todos_metric_middleware(), _request(state), _response(msg))
    assert [r for r in caplog.records if r.name == _METRIC_LOGGER] == []


def test_silent_when_todos_absent(caplog):
    state: dict = {}
    msg = AIMessage(content="hi")
    with caplog.at_level(logging.INFO, logger=_METRIC_LOGGER):
        _run(create_stale_todos_metric_middleware(), _request(state), _response(msg))
    assert [r for r in caplog.records if r.name == _METRIC_LOGGER] == []


def test_silent_when_response_result_empty(caplog):
    state = {"todos": [{"content": "a", "status": "pending"}]}
    with caplog.at_level(logging.INFO, logger=_METRIC_LOGGER):
        _run(create_stale_todos_metric_middleware(), _request(state), _response(None))
    assert [r for r in caplog.records if r.name == _METRIC_LOGGER] == []


def test_factory_returns_middleware_instance():
    assert isinstance(
        create_stale_todos_metric_middleware(), StaleTodosMetricMiddleware
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


def test_main_agent_middleware_includes_stale_todos_metric(monkeypatch):
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

    assert any(isinstance(m, StaleTodosMetricMiddleware) for m in middleware)


def test_async_subagent_middleware_excludes_stale_todos_metric():
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

    assert not any(isinstance(m, StaleTodosMetricMiddleware) for m in middleware)
