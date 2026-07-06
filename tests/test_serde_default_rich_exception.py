"""Regression tests for the serde.default rich-exception-payload patch.

Every provider SDK exception (OpenAI quota, Anthropic auth, Google
timeout, …) used to emerge as ``"An internal error occurred"`` in the
SSE error event because langgraph_api's ``serde.default`` catch-all
only surfaces the class name for a small whitelist. The patch replaces
``serde.default`` with a wrapper that enriches the payload for
exceptions whose module matches a recognized provider allow-list;
everything else (builtins, LangGraph DSL errors, HTTPException,
internal exceptions) falls through to upstream unchanged.
"""

from __future__ import annotations

import os

import langgraph_api.serde as _serde_mod

# Importing patches.py applies the eager module-level monkey-patch.
import EvoScientist.llm.patches  # noqa: F401


def test_whitelist_exception_still_uses_str_exc():
    """Pre-existing whitelist branch must keep working unchanged — the
    patch only replaces the catch-all generic-message branch.
    """
    payload = _serde_mod.default(ValueError("bad input"))
    assert payload["error"] == "ValueError"
    assert payload["message"] == "bad input"


def test_non_provider_exception_falls_through_to_upstream():
    """An exception outside the provider allow-list must not be enriched.
    It falls through to upstream's branch — either the whitelist (for
    builtins / LangGraph DSL / HTTPException) or the catch-all generic
    placeholder. Rationale: non-provider exceptions are rarely
    actionable for the end user; upstream's defensive behavior is the
    right one to preserve.
    """

    class CustomError(Exception):
        pass

    payload = _serde_mod.default(CustomError("something specific went wrong"))
    # Upstream catch-all shape: ``error`` + generic ``message``, no
    # ``class``/``provider`` enrichment.
    assert payload["error"] == "CustomError"
    assert payload["message"] == "An internal error occurred"
    assert "class" not in payload
    assert "provider" not in payload


def test_openai_apierror_payload_has_provider_tag():
    """Exercises the no-host fallback: when a fake exception has no
    ``.request`` attribute, ``_provider_from_exception`` skips the
    URL-host lookup and falls back to the module tag (``openai``). Real
    SDK exceptions always carry ``.request``, so the fallback fires
    mostly for hand-constructed / synthetic exceptions.
    """
    # Fake the class shape without importing openai (so the test works
    # even if the SDK isn't installed).
    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )("You exceeded your current quota, please check your plan.")
    payload = _serde_mod.default(fake_exc)
    assert payload["error"] == "APIError"
    assert payload["class"] == "openai.APIError"
    assert (
        payload["message"] == "You exceeded your current quota, please check your plan."
    )
    assert payload["provider"] == "openai"


def test_anthropic_apierror_payload_has_provider_tag():
    """Same no-host fallback path as the openai case, exercised on the
    ``anthropic`` module tag.
    """
    fake_exc = type(
        "AuthenticationError",
        (Exception,),
        {"__module__": "anthropic"},
    )("invalid x-api-key header")
    payload = _serde_mod.default(fake_exc)
    assert payload["class"] == "anthropic.AuthenticationError"
    assert payload["provider"] == "anthropic"


def test_google_genai_payload_has_provider_tag():
    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "google.genai.errors"},
    )("quota exceeded for project")
    payload = _serde_mod.default(fake_exc)
    assert payload["provider"] == "google_genai"


def test_langchain_openai_inherits_openai_provider_tag():
    """Wrapper-layer module prefixes (``langchain_openai`` etc.) map to
    their upstream provider tag via the allow-list. This test also
    exercises the no-host fallback — a real ``ChatOpenAI`` exception
    would carry ``.request`` and get URL-host refined, but for the
    hand-constructed shape here the module tag stands.
    """
    fake_exc = type(
        "BadRequestError",
        (Exception,),
        {"__module__": "langchain_openai.chat_models"},
    )("model not found")
    payload = _serde_mod.default(fake_exc)
    assert payload["provider"] == "openai"


def _make_openai_exc_with_host(module: str, host: str, message: str = "boom"):
    """Build a fake ``openai.*``-shaped exception whose
    ``.request.url.host`` matches *host*. Used to exercise the URL-host
    provider refinement without needing the real openai SDK.
    """

    class _FakeURL:
        def __init__(self, host: str) -> None:
            self.host = host

    class _FakeRequest:
        def __init__(self, host: str) -> None:
            self.url = _FakeURL(host)

    fake_cls = type(
        "APIError",
        (Exception,),
        {"__module__": module, "request": _FakeRequest(host)},
    )
    return fake_cls(message)


def test_deepseek_via_openai_sdk_tagged_by_host():
    """Regression: providers routed through ``ChatOpenAI`` with a
    custom ``base_url`` raise ``openai.APIError`` — the module prefix
    alone would mis-tag them as ``openai``. The URL host refinement
    must recover the actual provider.
    """
    exc = _make_openai_exc_with_host("openai", "api.deepseek.com", "quota exceeded")
    payload = _serde_mod.default(exc)
    assert payload["provider"] == "deepseek"


def test_moonshot_via_openai_sdk_tagged_by_host():
    exc = _make_openai_exc_with_host("openai", "api.moonshot.cn")
    assert _serde_mod.default(exc)["provider"] == "moonshot"


def test_zhipu_via_openai_sdk_tagged_by_host():
    exc = _make_openai_exc_with_host("openai", "open.bigmodel.cn")
    assert _serde_mod.default(exc)["provider"] == "zhipu"


def test_native_openai_host_tagged_openai():
    exc = _make_openai_exc_with_host("openai", "api.openai.com")
    assert _serde_mod.default(exc)["provider"] == "openai"


def test_unknown_openai_compat_host_tagged_openai_compat():
    """A ``custom-openai`` deploy points at an arbitrary base_url that
    isn't in our host map. Fall back to ``openai_compat`` so the WebUI
    knows "OpenAI SDK, but not native" rather than getting a misleading
    ``openai`` tag.
    """
    exc = _make_openai_exc_with_host("openai", "some.internal.corp")
    assert _serde_mod.default(exc)["provider"] == "openai_compat"


def test_minimax_via_anthropic_sdk_tagged_by_host():
    """Anthropic-routed providers get the same host-refinement treatment
    (minimax, kimi-coding, custom-anthropic).
    """
    exc = _make_openai_exc_with_host("anthropic", "api.minimaxi.com")
    assert _serde_mod.default(exc)["provider"] == "minimax"


def test_unknown_anthropic_compat_host_tagged_anthropic_compat():
    exc = _make_openai_exc_with_host("anthropic", "some.internal.corp")
    assert _serde_mod.default(exc)["provider"] == "anthropic_compat"


def test_host_read_from_response_request_when_top_level_missing():
    """``APIStatusError`` and friends carry the request under
    ``.response.request`` rather than ``.request``. Both paths must be
    consulted.
    """

    class _FakeURL:
        host = "api.deepseek.com"

    class _FakeRequest:
        url = _FakeURL()

    class _FakeResponse:
        request = _FakeRequest()

    exc = type(
        "APIStatusError",
        (Exception,),
        {"__module__": "openai", "response": _FakeResponse()},
    )("rate limit")
    assert _serde_mod.default(exc)["provider"] == "deepseek"


def test_host_extraction_survives_property_that_raises():
    """Regression: ``httpx.Response.request`` raises ``RuntimeError``
    when no request is attached to the response. ``_extract_host``
    must swallow property-access failures and fall back to the module
    tag rather than crashing the error path.
    """

    class _RaisingRequest:
        @property
        def url(self):
            raise RuntimeError("no request attached")

    class _RaisingResponse:
        @property
        def request(self):
            raise RuntimeError("no request attached")

    # Top-level ``.request.url`` raises via the property.
    exc1 = type(
        "APIError",
        (Exception,),
        {"__module__": "openai", "request": _RaisingRequest()},
    )("boom")
    # Should fall back to module tag ("openai"), not crash.
    assert _serde_mod.default(exc1)["provider"] == "openai"

    # ``.response.request`` raises via the property.
    exc2 = type(
        "APIStatusError",
        (Exception,),
        {"__module__": "openai", "response": _RaisingResponse()},
    )("boom")
    assert _serde_mod.default(exc2)["provider"] == "openai"


def test_unknown_module_falls_through_to_upstream():
    """An exception from a module we don't recognize as a provider is
    not enriched at all — it goes through upstream's catch-all and ends
    up as the generic placeholder. Avoids leaking internal exception
    text the user can't act on.
    """
    fake_exc = type(
        "WeirdError",
        (Exception,),
        {"__module__": "some_random_package"},
    )("weird thing happened")
    payload = _serde_mod.default(fake_exc)
    assert payload["error"] == "WeirdError"
    assert payload["message"] == "An internal error occurred"
    assert "class" not in payload
    assert "provider" not in payload


def test_env_deployed_key_redacted_in_message(monkeypatch):
    """A credential exported via env var must be scrubbed if a provider
    echoes it back in an exception message. The redaction table is
    built fresh per ``_redact_api_keys`` call, so ``monkeypatch.setenv``
    alone is enough — no re-import or attribute reassignment needed.
    """
    key = "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", key)

    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )(f"Invalid API key: {key}. Get a new one at https://platform.openai.com/api-keys")
    payload = _serde_mod.default(fake_exc)
    assert key not in payload["message"]
    assert "<redacted>" in payload["message"]
    # The actionable advice survives redaction.
    assert "Invalid API key" in payload["message"]
    assert "platform.openai.com" in payload["message"]


def test_multiple_env_keys_redacted_independently(monkeypatch):
    """Each ``*_API_KEY`` / ``*_TOKEN`` / ``*_SECRET`` env var contributes
    its own prefix to the alternation.
    """
    k1 = "sk-or-aBcDeFg012345678901234"
    k2 = "AIzaABCDEFGHIJ0123456789"
    k3 = "ghp_p4t70k3n0123456789abcdef"
    monkeypatch.setenv("OPENROUTER_API_KEY", k1)
    monkeypatch.setenv("GOOGLE_API_KEY", k2)
    monkeypatch.setenv("GITHUB_TOKEN", k3)

    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )(f"Failures: {k1}, {k2}, {k3}")
    msg = _serde_mod.default(fake_exc)["message"]
    assert k1 not in msg
    assert k2 not in msg
    assert k3 not in msg
    assert msg.count("<redacted>") == 3


def test_base64_suffix_secret_fully_redacted(monkeypatch):
    """A base64-style secret (containing ``/`` ``+`` ``=``) must redact
    end-to-end, not leak its tail. CodeRabbit flagged that the original
    suffix class stopped at base64 padding chars, leaving most of the
    credential visible after the first delimiter.
    """
    key = "AbCdEfGh/secret+tail=="
    monkeypatch.setenv("SOME_SECRET", key)

    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )(f"auth failed with token={key} on retry")
    msg = _serde_mod.default(fake_exc)["message"]
    assert "secret" not in msg
    assert "tail" not in msg
    assert "<redacted>" in msg
    # Surrounding text survives.
    assert "auth failed" in msg
    assert "on retry" in msg


def test_unknown_shape_not_redacted_without_env(monkeypatch):
    """Switch to env-only redaction: a key-shaped string that isn't
    actually deployed via env stays in the message. Tradeoff documented
    on PR #315: we only scrub what we know is a secret.
    """
    import EvoScientist.llm.patches as _p

    # Ensure no credential env vars are visible.
    for k in list(os.environ):
        if k.endswith(_p._API_KEY_ENV_SUFFIXES):
            monkeypatch.delenv(k, raising=False)

    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )("Unknown key seen: sk-or-aBcDeFg012345678901234")
    msg = _serde_mod.default(fake_exc)["message"]
    assert "sk-or-aBcDeFg012345678901234" in msg
    assert "<redacted>" not in msg


def test_env_key_loaded_after_first_call_is_redacted(monkeypatch):
    """Regression: the pattern must rebuild per call so keys loaded
    after ``patches.py`` imports (typical ``load_dotenv`` sequence in
    a main entry point) are still scrubbed on the next exception.
    """
    import EvoScientist.llm.patches as _p

    # No matching env vars visible yet — mirrors "load_dotenv hasn't run".
    for k in list(os.environ):
        if k.endswith(_p._API_KEY_ENV_SUFFIXES):
            monkeypatch.delenv(k, raising=False)
    key = "sk-proj-loaded_after_import_1234567890abcdef"

    fake_exc_type = type("APIError", (Exception,), {"__module__": "openai"})

    # Pass 1: env empty — key leaks (no pattern to match against).
    msg1 = _serde_mod.default(fake_exc_type(f"leak: {key}"))["message"]
    assert key in msg1
    assert "<redacted>" not in msg1

    # Pass 2: after simulated load_dotenv, next exception scrubs the key.
    monkeypatch.setenv("OPENAI_API_KEY", key)
    msg2 = _serde_mod.default(fake_exc_type(f"leak: {key}"))["message"]
    assert key not in msg2
    assert "<redacted>" in msg2


def test_redaction_regex_holds_only_prefix(monkeypatch):
    """Defense-in-depth: the compiled regex must not embed the full key.
    A process-memory leak (traceback locals, debugger) exposes at most
    the first 8 chars — not the secret.
    """
    import EvoScientist.llm.patches as _p

    key = "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890_secret_suffix"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    pattern = _p._build_env_key_redaction_re()
    assert pattern is not None
    # The full key never appears in the regex source.
    assert key not in pattern.pattern
    assert "aBcDeFgHiJkLmNoPqRs" not in pattern.pattern
    # Sanity: the regex still matches the key end-to-end (prefix anchor
    # plus suffix-greedy match).
    m = pattern.search(f"err: {key}")
    assert m is not None
    assert m.group(0) == key


def test_openai_status_code_surfaced():
    """openai/anthropic APIStatusError carries integer ``.status_code`` —
    surface it so the WebUI can switch on 401 vs 429 vs 5xx.
    """
    fake_exc = type(
        "RateLimitError",
        (Exception,),
        {"__module__": "openai", "status_code": 429},
    )("Rate limit exceeded")
    payload = _serde_mod.default(fake_exc)
    assert payload["status_code"] == 429


def test_status_code_via_response_attr():
    """Wrappers that don't promote status to a top-level attr still
    expose it via ``.response.status_code`` (httpx pattern).
    """

    class FakeResponse:
        status_code = 504

    fake_exc = type(
        "HTTPStatusError",
        (Exception,),
        {"__module__": "openai", "response": FakeResponse()},
    )("Gateway Timeout")
    payload = _serde_mod.default(fake_exc)
    assert payload["status_code"] == 504


def test_google_genai_status_via_code_attr():
    """``google.genai.errors.APIError`` stores the HTTP status as
    integer ``.code`` (the SDK's quirky shape). Type-disambiguated from
    openai/anthropic's string ``.code`` (provider error code).
    """
    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "google.genai.errors", "code": 400},
    )("invalid argument")
    payload = _serde_mod.default(fake_exc)
    assert payload["status_code"] == 400
    # An integer ``.code`` is HTTP status, not a provider code — must
    # not bleed into the string ``code`` field.
    assert "code" not in payload


def test_provider_code_surfaced():
    """Provider error code (e.g. ``insufficient_quota``) is higher signal
    than the HTTP status alone — surface as ``code``.
    """
    fake_exc = type(
        "APIError",
        (Exception,),
        {
            "__module__": "openai",
            "code": "insufficient_quota",
            "status_code": 429,
        },
    )("You exceeded your current quota")
    payload = _serde_mod.default(fake_exc)
    assert payload["code"] == "insufficient_quota"
    assert payload["status_code"] == 429


def test_error_type_surfaced():
    """openai exposes a ``type`` label (e.g. ``rate_limit_error``)."""
    fake_exc = type(
        "RateLimitError",
        (Exception,),
        {"__module__": "openai", "type": "rate_limit_error"},
    )("limit")
    payload = _serde_mod.default(fake_exc)
    assert payload["type"] == "rate_limit_error"


def test_extracted_fields_omitted_when_absent():
    """A bare provider exception with none of these attrs — none of the
    optional fields appear in the payload.
    """
    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "anthropic"},
    )("connection error")
    payload = _serde_mod.default(fake_exc)
    assert "status_code" not in payload
    assert "code" not in payload
    assert "type" not in payload


def test_request_id_propagated_when_present():
    """OpenAI/Anthropic SDK exceptions carry a ``.request_id`` — surface
    it so users can cite it in support tickets.
    """
    exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai", "request_id": "req_abc123def456"},
    )("rate limited")
    payload = _serde_mod.default(exc)
    assert payload["request_id"] == "req_abc123def456"


def test_non_exception_objects_delegate_to_upstream():
    """The patch must only intercept ``BaseException`` instances; other
    object types still get upstream's encoding.
    """
    # A datetime is one of the types upstream handles. The patched
    # encoder must delegate to upstream rather than swallow it.
    from datetime import timedelta

    result = _serde_mod.default(timedelta(seconds=42))
    assert result == 42.0


def test_patch_applied_flag_set():
    from EvoScientist.llm.patches import _serde_default_patched

    assert _serde_default_patched is True


# ---------------------------------------------------------------------------
# ProviderStreamError short-circuit — the middleware pre-bakes the envelope
# ---------------------------------------------------------------------------


def test_serde_default_short_circuits_on_provider_stream_error():
    """When ``ErrorNormalizationMiddleware`` has already wrapped an
    exception into ``ProviderStreamError``, ``serde.default`` must
    emit the pre-baked envelope directly rather than re-running
    provider / status / redaction inference.
    """
    from EvoScientist.llm.errors import ProviderStreamError

    err = ProviderStreamError(
        provider="openrouter",
        class_qualname="openrouter.errors.foo.UnauthorizedResponseError",
        message="User not found.",
        status_code=401,
    )
    payload = _serde_mod.default(err)
    assert payload == {
        "error": "UnauthorizedResponseError",
        "class": "openrouter.errors.foo.UnauthorizedResponseError",
        "message": "User not found.",
        "provider": "openrouter",
        "status_code": 401,
    }


def test_provider_stream_error_survives_orjson_dataclass_option():
    """End-to-end guard: a dataclass provider exception, wrapped by
    the middleware into ``ProviderStreamError`` (non-dataclass), must
    NOT bypass ``serde.default`` under orjson's
    ``OPT_SERIALIZE_DATACLASS`` — that was the OpenRouter bug this
    approach is meant to fix.
    """
    import dataclasses

    import orjson

    from EvoScientist.llm.errors import ProviderStreamError

    @dataclasses.dataclass
    class RawDataclassExc(Exception):
        message: str
        status_code: int

        def __init__(self, message: str, status_code: int) -> None:
            super().__init__(message)
            object.__setattr__(self, "message", message)
            object.__setattr__(self, "status_code", status_code)

    RawDataclassExc.__module__ = "openrouter.errors.foo"
    raw = RawDataclassExc("User not found.", 401)
    # Sanity: the raw shape would take orjson's dataclass fast-path.
    assert dataclasses.is_dataclass(raw)

    # Middleware would wrap it; simulate the wrap directly by faking a
    # request whose ``.model`` is an OpenRouter chat model shape.
    from types import SimpleNamespace

    from EvoScientist.middleware.error_normalization import _normalize

    fake_openrouter_cls = type(
        "ChatOpenRouter",
        (),
        {"__module__": "langchain_openrouter.chat_models"},
    )
    request = SimpleNamespace(model=fake_openrouter_cls())
    wrapped = _normalize(request, raw)
    assert isinstance(wrapped, ProviderStreamError)

    wire = orjson.dumps(
        wrapped,
        default=_serde_mod.default,
        option=orjson.OPT_SERIALIZE_DATACLASS,
    )
    decoded = orjson.loads(wire)
    # Envelope, not raw dataclass fields.
    assert decoded["error"] == "RawDataclassExc"
    assert decoded["provider"] == "openrouter"
    assert decoded["status_code"] == 401
    # Raw dataclass fields like ``headers`` / ``raw_response`` / ``data``
    # must not appear — those are the OpenRouter-specific leaks.
    assert "headers" not in decoded
    assert "raw_response" not in decoded
