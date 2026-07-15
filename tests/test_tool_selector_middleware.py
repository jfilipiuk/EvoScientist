"""Tests for LLMToolSelectorMiddleware integration and the event-sink handoff."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.tools import BaseTool, StructuredTool

from EvoScientist.middleware.tool_selector import (
    _ConditionalToolSelectorMiddleware,
    create_tool_selector_middleware,
)
from EvoScientist.stream.emitter import StreamEventEmitter
from EvoScientist.stream.sink import SessionEventSink
from EvoScientist.stream.tool_selection import _ToolSelectionSuppressor


class _RecordingSink:
    """Records selection lifecycle calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.active = False

    def on_tool_selection_started(self, total_tools: int) -> None:
        self.active = True
        self.calls.append(("started", total_tools))

    def on_tool_selection(self, selected: list[str], total_tools: int) -> None:
        self.calls.append(("selection", list(selected), total_tools))

    def on_tool_selection_ended(self) -> None:
        self.active = False
        self.calls.append(("ended",))

    def emit_fallback_notice(self, text: str, style: str = "yellow") -> None:
        pass

    @property
    def tool_selection_active(self) -> bool:
        return self.active


def _tool(name: str) -> BaseTool:
    def _func(value: str = "") -> str:
        return value

    return StructuredTool.from_function(
        func=_func,
        name=name,
        description=f"{name} test tool",
    )


def _request(tools: list[BaseTool | dict[str, Any]]) -> ModelRequest:
    return ModelRequest(model=MagicMock(), messages=[], tools=tools)


def _mock_model():
    """Create a MagicMock model compatible with disable_thinking()."""
    m = MagicMock(profile={"max_input_tokens": 200_000})
    m.thinking = None
    m.reasoning = None
    return m


# Helper: patches needed to call create_tool_selector_middleware without LLM
def _factory_patches():
    return (
        patch(
            "EvoScientist.middleware.tool_selector.disable_thinking",
            return_value=MagicMock(),
            create=True,
        ),
        patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=MagicMock()),
        patch(
            "EvoScientist.middleware.tool_selector._HiddenLLMToolSelectorMiddleware",
            return_value=MagicMock(),
        ),
    )


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_create_tool_selector_returns_single_middleware():
    p1, p2, p3 = _factory_patches()
    with p1, p2, p3:
        result = create_tool_selector_middleware()
        assert isinstance(result, list)
        assert len(result) == 1
        assert type(result[0]).__name__ == "_ConditionalToolSelectorMiddleware"


def test_create_tool_selector_always_include():
    p1, p2, p3 = _factory_patches()
    with p1, p2, p3 as mock_cls:
        result = create_tool_selector_middleware(threshold=0)
        request = _request(
            [
                _tool("think_tool"),
                _tool("search_observations"),
                _tool("read_memory"),
                _tool("unrelated_tool"),
            ]
        )
        result[0].wrap_model_call(request, MagicMock())
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["always_include"] == [
            "read_memory",
            "search_observations",
            "think_tool",
        ]


def test_custom_threshold():
    p1, p2, p3 = _factory_patches()
    with p1, p2, p3:
        result = create_tool_selector_middleware(threshold=5)
        assert result[0]._threshold == 5


# ---------------------------------------------------------------------------
# Conditional selector unit tests
# ---------------------------------------------------------------------------


def test_conditional_skips_below_threshold():
    """When tools <= threshold, selector is skipped and nothing is reported."""
    mock_selector = MagicMock()
    selector_factory = MagicMock(return_value=mock_selector)
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=10,
        events=sink,
    )

    request = MagicMock()
    request.tools = [MagicMock() for _ in range(5)]
    handler = MagicMock()

    cond.wrap_model_call(request, handler)
    handler.assert_called_once_with(request)
    selector_factory.assert_not_called()
    mock_selector.wrap_model_call.assert_not_called()
    assert sink.calls == []  # no selection ran → no events


def test_conditional_runs_above_threshold():
    """When tools > threshold, selector runs."""
    mock_selector = MagicMock()
    selector_factory = MagicMock(return_value=mock_selector)
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=10,
    )

    request = MagicMock()
    request.tools = [MagicMock() for _ in range(15)]
    handler = MagicMock()

    cond.wrap_model_call(request, handler)
    selector_factory.assert_called_once_with([])
    mock_selector.wrap_model_call.assert_called_once()
    handler.assert_not_called()


def test_selection_lifecycle_reported_to_sink():
    """started(total) → selection(selected, total) → ended, reported to the sink."""
    # The fake selector filters the request down to two named tools before
    # calling the downstream handler.
    filtered = _request([_tool("read_file"), _tool("think_tool")])

    def fake_selector_call(request, handler):
        return handler(filtered)

    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = fake_selector_call
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])
    cond.wrap_model_call(request, MagicMock())

    assert sink.calls == [
        ("started", 10),
        ("selection", ["read_file", "think_tool"], 10),
        ("ended",),
    ]


def test_selector_failure_reports_ended_without_selection():
    """A selector that raises before the handler surfaces no selection event."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = RuntimeError("no structured output")
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])
    handler = MagicMock()
    cond.wrap_model_call(request, handler)

    # Falls back to all tools; only started/ended reported, no selection.
    handler.assert_called_once_with(request)
    assert ("started", 10) in sink.calls
    assert not any(c[0] == "selection" for c in sink.calls)
    assert sink.calls[-1] == ("ended",)


def test_selector_failure_ends_before_sync_fallback_handler():
    """All-tools fallback must not run while selector suppression is active."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = RuntimeError("no structured output")
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])

    def handler(req):
        sink.calls.append(("handler", sink.tool_selection_active))
        return MagicMock()

    cond.wrap_model_call(request, handler)

    assert sink.calls == [
        ("started", 10),
        ("ended",),
        ("handler", False),
    ]


@pytest.mark.asyncio
async def test_selector_failure_ends_before_async_fallback_handler():
    """Async all-tools fallback must see selection already closed."""
    mock_selector = MagicMock()
    mock_selector.awrap_model_call.side_effect = RuntimeError("no structured output")
    sink = _RecordingSink()
    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=MagicMock(return_value=mock_selector),
        threshold=5,
        events=sink,
    )

    request = _request([_tool(f"t{i}") for i in range(10)])

    async def handler(req):
        sink.calls.append(("handler", sink.tool_selection_active))
        return MagicMock()

    await cond.awrap_model_call(request, handler)

    assert sink.calls == [
        ("started", 10),
        ("ended",),
        ("handler", False),
    ]


def test_selector_always_includes_available_memory_tools():
    """Adaptive selection must mark available memory tools as mandatory."""
    calls = []

    class FakeSelector:
        def __init__(self, always_include):
            self.always_include = always_include

        def wrap_model_call(self, request, handler):
            calls.append(self.always_include)
            return handler(request)

    def selector_factory(always_include):
        return FakeSelector(always_include)

    request = _request(
        [
            _tool("think_tool"),
            _tool("search_observations"),
            _tool("read_memory"),
            _tool("unrelated_tool"),
        ]
    )

    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=0,
        always_include=frozenset(
            {
                "think_tool",
                "task",
                "search_observations",
                "read_memory",
                "record_observation",
            }
        ),
    )
    handler = MagicMock()

    cond.wrap_model_call(request, handler)

    assert calls == [
        [
            "read_memory",
            "search_observations",
            "think_tool",
        ]
    ]


def test_selector_resolved_once_across_repeated_requests():
    """Agent tools are stable, so build the selector once and reuse it."""
    mock_selector = MagicMock()
    mock_selector.wrap_model_call.side_effect = lambda request, handler: handler(
        request
    )
    selector_factory = MagicMock(return_value=mock_selector)

    cond = _ConditionalToolSelectorMiddleware(
        selector_factory=selector_factory,
        threshold=0,
        always_include=frozenset({"think_tool", "search_observations"}),
    )
    tools = [
        _tool("think_tool"),
        _tool("search_observations"),
        _tool("unrelated_tool"),
    ]

    for _ in range(3):
        cond.wrap_model_call(_request(tools), MagicMock())

    selector_factory.assert_called_once_with(["search_observations", "think_tool"])
    assert mock_selector.wrap_model_call.call_count == 3


# ---------------------------------------------------------------------------
# R1: consume-once + dedup render sequences (sink + suppressor)
# ---------------------------------------------------------------------------


def _drive_selection(sink, suppressor, selected, total):
    """Mimic one selection turn: sink records it, the suppressor observes the
    selector JSON block, then a flush surfaces (or not) the UI event."""
    sink.on_tool_selection_started(total)
    sink.on_tool_selection(selected, total)
    sink.on_tool_selection_ended()
    # Suppressor observes the selector's structured-output tool block.
    suppressor.observe_tool_block("ToolSelectionResponse")
    return suppressor.flush_selection()


def test_render_sequences_table():
    """select → render; same selection again → no repeat; new selection → render."""
    cases = [
        # (label, selected, total, expect_render)
        ("first selection renders", ["read_file", "think_tool"], 5, True),
        ("same selection again does not repeat", ["read_file", "think_tool"], 5, False),
        ("new selection renders", ["execute", "think_tool"], 5, True),
        ("kept-all selection does not render", ["a", "b", "c"], 3, False),
    ]
    sink = SessionEventSink()
    suppressor = _ToolSelectionSuppressor(StreamEventEmitter(), sink)

    for label, selected, total, expect_render in cases:
        events = _drive_selection(sink, suppressor, selected, total)
        rendered = [e for e in events if e.get("type") == "tool_selection"]
        if expect_render:
            assert rendered, f"{label}: expected a tool_selection event"
            assert rendered[0]["tools"] == selected, label
        else:
            assert not rendered, f"{label}: expected no tool_selection event"


def test_consume_is_once_only():
    """A pending selection renders once; a second flush yields nothing."""
    sink = SessionEventSink()
    suppressor = _ToolSelectionSuppressor(StreamEventEmitter(), sink)

    first = _drive_selection(sink, suppressor, ["read_file"], 3)
    assert any(e.get("type") == "tool_selection" for e in first)

    # No new selection recorded; the observation flag was consumed.
    suppressor.observe_tool_block("ToolSelectionResponse")
    second = suppressor.flush_selection()
    assert not any(e.get("type") == "tool_selection" for e in second)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    side_effect=lambda *a, **kw: [
        _ConditionalToolSelectorMiddleware(
            selector_factory=MagicMock(return_value=MagicMock()),
            threshold=20,
        )
    ],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_default_middleware_includes_tool_selector(mock_config, mock_model, mock_ts):
    mock_model.return_value = _mock_model()
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    type_names = [type(m).__name__ for m in mw]
    assert "_ConditionalToolSelectorMiddleware" in type_names


@patch("EvoScientist.EvoScientist._ensure_chat_model")
def test_subagent_no_tool_selector(mock_model):
    mock_model.return_value = _mock_model()

    from EvoScientist.EvoScientist import _inject_subagent_middleware

    subs = [{"name": "test-agent"}]
    _inject_subagent_middleware(subs)

    type_names = [type(m).__name__ for m in subs[0]["middleware"]]
    assert "_ConditionalToolSelectorMiddleware" not in type_names


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    side_effect=lambda *a, **kw: [
        _ConditionalToolSelectorMiddleware(
            selector_factory=MagicMock(return_value=MagicMock()),
            threshold=20,
        )
    ],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_tool_selector_ordering(mock_config, mock_model, mock_ts):
    """ToolSelector should come after ToolErrorHandler and before Memory."""
    mock_model.return_value = _mock_model()
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    from EvoScientist.EvoScientist import _get_default_middleware

    mw = _get_default_middleware()
    type_names = [type(m).__name__ for m in mw]

    ts_idx = type_names.index("_ConditionalToolSelectorMiddleware")
    te_idx = type_names.index("ToolErrorHandlerMiddleware")
    mem_idx = type_names.index("EvoMemoryMiddleware")
    assert te_idx < ts_idx < mem_idx


# ---------------------------------------------------------------------------
# disable_streaming — kills per-chunk selector emissions
# ---------------------------------------------------------------------------


def test_disable_streaming_sets_disable_streaming_field():
    """Helper must set ``disable_streaming=True`` (BaseChatModel's official
    hard-disable field checked by ``_streaming_disabled()``), not the
    model's own ``streaming`` field.
    """
    from EvoScientist.middleware.utils import disable_streaming

    model = MagicMock()
    copied = MagicMock()
    model.model_copy.return_value = copied

    result = disable_streaming(model)

    model.model_copy.assert_called_once_with(update={"disable_streaming": True})
    assert result is copied


def test_disable_streaming_falls_back_to_shallow_copy_on_model_copy_failure():
    """Fallback path: ``model_copy`` fails, so we shallow-copy the passed
    model and ``setattr`` on the copy.

    Shared-instance safety: ``disable_thinking`` returns the original model
    unchanged when no thinking/reasoning update is needed (common for
    Gemini). If the caller's model is the process-global cached main-agent
    model, mutating it here would silently disable streaming on the
    main-agent's model instance — a hard-to-diagnose regression the shallow
    copy prevents.
    """
    from EvoScientist.middleware.utils import disable_streaming

    class _FakeModel:
        disable_streaming = False

        def model_copy(self, update):
            raise RuntimeError("not pydantic")

    model = _FakeModel()
    result = disable_streaming(model)

    assert result is not model
    assert result.disable_streaming is True
    # Caller instance untouched.
    assert model.disable_streaming is False


def test_disable_streaming_defeats_upstream_streaming_dispatch():
    """End-to-end mechanism test: a model copy produced by
    ``disable_streaming`` causes langchain's own ``_streaming_disabled``
    to return True.

    ``_streaming_disabled`` is the single check consulted by
    ``_should_stream`` / ``_should_use_protocol_streaming`` before
    dispatching to ``_stream`` / ``_astream``. If our field-setting fails
    or a future langchain version changes the check key, this test fails
    before the selector floods anything in production — strictly better
    than a runtime canary.
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from EvoScientist.middleware.utils import disable_streaming

    class _FakeModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

    model = _FakeModel()
    assert model._streaming_disabled() is False

    disabled = disable_streaming(model)

    assert disabled._streaming_disabled() is True
    # Original caller instance untouched.
    assert model._streaming_disabled() is False


def test_create_tool_selector_wraps_model_with_disable_streaming():
    """Factory chains ``disable_streaming`` around ``disable_thinking``
    so the selector's internal model call can't dispatch to
    ``_stream`` / ``_astream``.
    """
    thinking_out = MagicMock(name="disable_thinking_output")
    streaming_out = MagicMock(name="disable_streaming_output")

    with (
        patch(
            "EvoScientist.middleware.utils.disable_thinking",
            return_value=thinking_out,
        ) as mock_dt,
        patch(
            "EvoScientist.middleware.utils.disable_streaming",
            return_value=streaming_out,
        ) as mock_ds,
        patch("EvoScientist.EvoScientist._ensure_chat_model", return_value=MagicMock()),
        patch(
            "EvoScientist.middleware.tool_selector._HiddenLLMToolSelectorMiddleware",
            return_value=MagicMock(),
        ) as mock_selector,
    ):
        result = create_tool_selector_middleware(threshold=0)
        # selector_factory is lazy — trigger it via wrap_model_call so the
        # subclass constructor actually fires and we can observe what
        # model was passed.
        result[0].wrap_model_call(_request([_tool("t")]), MagicMock())

    mock_dt.assert_called_once()
    mock_ds.assert_called_once_with(thinking_out)
    assert mock_selector.call_args.kwargs["model"] is streaming_out


# ---------------------------------------------------------------------------
# _HiddenLLMToolSelectorMiddleware — tags selector callback events hidden
# ---------------------------------------------------------------------------


def test_hidden_selector_tags_invoke_with_langsmith_hidden():
    """The subclass passes ``tags=["langsmith:hidden"]`` in the config
    argument to ``structured_model.invoke``, so langgraph_api's SSE layer
    (``langgraph_api/stream.py:327``) drops the selector's callback events
    before they reach WebUI.
    """
    from langchain_core.language_models import BaseChatModel

    from EvoScientist.middleware.tool_selector import (
        _HiddenLLMToolSelectorMiddleware,
    )

    middleware = _HiddenLLMToolSelectorMiddleware(model=MagicMock(spec=BaseChatModel))

    selection_request = MagicMock()
    selection_request.available_tools = [MagicMock(name="tool_a")]
    selection_request.valid_tool_names = {"tool_a"}
    selection_request.system_message = "sys"
    selection_request.last_user_message = {"role": "user", "content": "hi"}

    structured_model = MagicMock()
    structured_model.invoke.return_value = {"tools": ["tool_a"]}
    selection_request.model.with_structured_output.return_value = structured_model

    with (
        patch.object(
            _HiddenLLMToolSelectorMiddleware,
            "_prepare_selection_request",
            return_value=selection_request,
        ),
        patch.object(
            _HiddenLLMToolSelectorMiddleware,
            "_process_selection_response",
            return_value=MagicMock(),
        ),
        patch(
            "EvoScientist.middleware.tool_selector._create_tool_selection_response",
            return_value=MagicMock(json_schema=MagicMock(return_value={})),
        ),
    ):
        middleware.wrap_model_call(MagicMock(), MagicMock())

    structured_model.invoke.assert_called_once()
    call_config = structured_model.invoke.call_args.kwargs.get("config")
    assert call_config is not None
    assert "langsmith:hidden" in call_config.get("tags", [])


@pytest.mark.asyncio
async def test_hidden_selector_tags_ainvoke_with_langsmith_hidden():
    """Async path also tags with ``langsmith:hidden``."""
    from langchain_core.language_models import BaseChatModel

    from EvoScientist.middleware.tool_selector import (
        _HiddenLLMToolSelectorMiddleware,
    )

    middleware = _HiddenLLMToolSelectorMiddleware(model=MagicMock(spec=BaseChatModel))

    selection_request = MagicMock()
    selection_request.available_tools = [MagicMock(name="tool_a")]
    selection_request.valid_tool_names = {"tool_a"}
    selection_request.system_message = "sys"
    selection_request.last_user_message = {"role": "user", "content": "hi"}

    structured_model = MagicMock()

    async def _ainvoke(*args, **kwargs):
        return {"tools": ["tool_a"]}

    structured_model.ainvoke = MagicMock(side_effect=_ainvoke)
    selection_request.model.with_structured_output.return_value = structured_model

    async def _handler(req):
        return MagicMock()

    with (
        patch.object(
            _HiddenLLMToolSelectorMiddleware,
            "_prepare_selection_request",
            return_value=selection_request,
        ),
        patch.object(
            _HiddenLLMToolSelectorMiddleware,
            "_process_selection_response",
            return_value=MagicMock(),
        ),
        patch(
            "EvoScientist.middleware.tool_selector._create_tool_selection_response",
            return_value=MagicMock(json_schema=MagicMock(return_value={})),
        ),
    ):
        await middleware.awrap_model_call(MagicMock(), _handler)

    structured_model.ainvoke.assert_called_once()
    call_config = structured_model.ainvoke.call_args.kwargs.get("config")
    assert call_config is not None
    assert "langsmith:hidden" in call_config.get("tags", [])


# ---------------------------------------------------------------------------
# _SelectorFloodDetector — self-reports the provider quirk
# ---------------------------------------------------------------------------


def test_flood_detector_warns_above_threshold(caplog):
    """Detector emits a WARNING with the count + names when tool_calls
    length hits THRESHOLD. Proves the workaround self-reports so we can
    tell if the provider quirk is still recurring in production."""
    import logging as _logging

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from EvoScientist.middleware.tool_selector import _SelectorFloodDetector

    detector = _SelectorFloodDetector()
    tool_calls = [
        {"name": "ToolSelectionResponse", "args": {}, "id": f"id_{i}"}
        for i in range(_SelectorFloodDetector.THRESHOLD)
    ]
    msg = AIMessage(content="", tool_calls=tool_calls)
    result = LLMResult(generations=[[ChatGeneration(message=msg)]])

    with caplog.at_level(
        _logging.WARNING, logger="EvoScientist.middleware.tool_selector"
    ):
        detector.on_llm_end(result)

    assert any("tool_selector.flood" in rec.message for rec in caplog.records)
    assert any("ToolSelectionResponse" in rec.message for rec in caplog.records)


def test_flood_detector_silent_below_threshold(caplog):
    """Normal selector output (single tool_call) does not emit a warning
    — no noise on the fast path."""
    import logging as _logging

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from EvoScientist.middleware.tool_selector import _SelectorFloodDetector

    detector = _SelectorFloodDetector()
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "ToolSelectionResponse", "args": {"tools": ["x"]}, "id": "id"}
        ],
    )
    result = LLMResult(generations=[[ChatGeneration(message=msg)]])

    with caplog.at_level(
        _logging.WARNING, logger="EvoScientist.middleware.tool_selector"
    ):
        detector.on_llm_end(result)

    assert not any("tool_selector.flood" in rec.message for rec in caplog.records)
