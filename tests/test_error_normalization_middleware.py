"""Tests for ErrorNormalizationMiddleware + ProviderStreamError.

Verifies that provider-SDK exceptions from a chat model call get
wrapped into a non-dataclass ``ProviderStreamError`` at the model
boundary, and that non-provider exceptions pass through unchanged.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from EvoScientist.llm.errors import ProviderStreamError
from EvoScientist.middleware.error_normalization import (
    ErrorNormalizationMiddleware,
    _normalize,
)


def _make_provider_exc(module: str, cls_name: str = "APIError", message: str = "boom"):
    """Build a fake plain-Exception with __module__ set."""
    return type(cls_name, (Exception,), {"__module__": module})(message)


def _make_openai_status_exc(host: str, status_code: int = 401, message: str = "boom"):
    """Build a fake openai-shaped exception with request.url.host set,
    exercising the URL-host branch of _provider_from_exception.
    """

    class _URL:
        def __init__(self, host: str) -> None:
            self.host = host

    class _Request:
        def __init__(self, host: str) -> None:
            self.url = _URL(host)

    cls = type(
        "APIError",
        (Exception,),
        {
            "__module__": "openai",
            "request": _Request(host),
            "status_code": status_code,
        },
    )
    return cls(message)


class TestNormalize:
    def test_openai_module_wraps_to_provider_stream_error(self):
        exc = _make_openai_status_exc("api.openai.com", 429, "rate limited")
        wrapped = _normalize(exc)
        assert isinstance(wrapped, ProviderStreamError)
        assert wrapped.provider == "openai"
        assert wrapped.class_qualname.startswith("openai.")
        assert wrapped.message == "rate limited"
        assert wrapped.status_code == 429
        assert wrapped.original is exc

    def test_openai_routed_host_recovers_concrete_provider(self):
        exc = _make_openai_status_exc("api.deepseek.com", 401, "quota exceeded")
        wrapped = _normalize(exc)
        assert isinstance(wrapped, ProviderStreamError)
        assert wrapped.provider == "deepseek"

    def test_openrouter_module_wraps_to_provider_stream_error(self):
        exc = _make_provider_exc("openrouter.errors.foo", message="User not found.")
        wrapped = _normalize(exc)
        assert isinstance(wrapped, ProviderStreamError)
        assert wrapped.provider == "openrouter"
        assert wrapped.class_qualname == "openrouter.errors.foo.APIError"

    def test_non_provider_module_returns_none(self):
        exc = _make_provider_exc("some.random.package")
        assert _normalize(exc) is None

    def test_builtin_exception_returns_none(self):
        # ValueError is in upstream's whitelist; middleware should not
        # normalize it — upstream's own path is the right thing.
        assert _normalize(ValueError("bad input")) is None


class TestProviderStreamErrorEnvelope:
    def test_envelope_contains_required_fields(self):
        err = ProviderStreamError(
            provider="deepseek",
            class_qualname="openai.RateLimitError",
            message="quota exceeded",
            status_code=429,
            code="insufficient_quota",
        )
        env = err.as_envelope()
        assert env["error"] == "RateLimitError"
        assert env["class"] == "openai.RateLimitError"
        assert env["message"] == "quota exceeded"
        assert env["provider"] == "deepseek"
        assert env["status_code"] == 429
        assert env["code"] == "insufficient_quota"

    def test_envelope_omits_absent_optional_fields(self):
        err = ProviderStreamError(
            provider="openrouter",
            class_qualname="openrouter.errors.foo.UnauthorizedResponseError",
            message="User not found.",
        )
        env = err.as_envelope()
        assert "status_code" not in env
        assert "code" not in env
        assert "type" not in env
        assert "request_id" not in env

    def test_provider_stream_error_is_not_a_dataclass(self):
        """The whole point of the wrapper — must not be a dataclass so
        orjson's OPT_SERIALIZE_DATACLASS fast-path doesn't fire.
        """
        err = ProviderStreamError("x", "y.Z", "msg")
        assert not dataclasses.is_dataclass(err)
        assert not dataclasses.is_dataclass(type(err))


class TestMiddleware:
    def _run_awrap(self, mw, handler):
        async def _go():
            return await mw.awrap_model_call(request=None, handler=handler)

        return asyncio.run(_go())

    def test_awrap_normalizes_provider_exception(self):
        raised = _make_provider_exc("openrouter.errors.foo", message="boom")

        async def handler(_req):
            raise raised

        mw = ErrorNormalizationMiddleware()
        with pytest.raises(ProviderStreamError) as excinfo:
            self._run_awrap(mw, handler)
        assert excinfo.value.provider == "openrouter"
        assert excinfo.value.original is raised

    def test_awrap_passes_through_non_provider_exception(self):
        raised = RuntimeError("internal glitch")

        async def handler(_req):
            raise raised

        mw = ErrorNormalizationMiddleware()
        with pytest.raises(RuntimeError) as excinfo:
            self._run_awrap(mw, handler)
        assert excinfo.value is raised

    def test_sync_wrap_normalizes_provider_exception(self):
        raised = _make_provider_exc("openrouter.errors.foo", message="boom")

        def handler(_req):
            raise raised

        mw = ErrorNormalizationMiddleware()
        with pytest.raises(ProviderStreamError) as excinfo:
            mw.wrap_model_call(request=None, handler=handler)
        assert excinfo.value.provider == "openrouter"

    def test_success_path_returns_handler_result(self):
        async def handler(_req):
            return "ok"

        mw = ErrorNormalizationMiddleware()
        assert self._run_awrap(mw, handler) == "ok"
