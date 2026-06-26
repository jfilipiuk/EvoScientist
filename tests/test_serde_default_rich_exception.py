"""Regression tests for the serde.default rich-exception-payload patch.

Reproducer: ``notes/sse-error-event-payload.md`` — every
non-whitelisted exception class hit langgraph_api's catch-all branch
and emerged as ``"An internal error occurred"`` in the SSE error
event, regardless of the real cause (OpenAI quota, Anthropic auth,
Google timeout, …). The patch replaces only the catch-all; whitelist
classes (``ValueError`` etc.) keep their pre-existing behavior.
"""

from __future__ import annotations

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


def test_non_whitelist_exception_exposes_class_and_message():
    """Any ``BaseException`` outside the whitelist should land in our
    enriched branch with ``class`` and a non-generic ``message``.
    """

    class CustomError(Exception):
        pass

    exc = CustomError("something specific went wrong")
    payload = _serde_mod.default(exc)
    assert payload["error"] == "CustomError"
    assert payload["message"] == "something specific went wrong"
    # Class is fully qualified so the WebUI can distinguish
    # ``openai.APIError`` from ``anthropic.APIError`` etc.
    assert payload["class"].endswith(".CustomError")


def test_openai_apierror_payload_has_provider_tag():
    """OpenAI SDK exceptions should be tagged ``provider: openai``
    based on their module prefix.
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
    """Wrapper-layer exceptions (langchain_openai etc.) should still
    map to their upstream provider tag.
    """
    fake_exc = type(
        "BadRequestError",
        (Exception,),
        {"__module__": "langchain_openai.chat_models"},
    )("model not found")
    payload = _serde_mod.default(fake_exc)
    assert payload["provider"] == "openai"


def test_unknown_provider_emits_no_provider_field():
    """Modules we don't recognize shouldn't get a guessed provider —
    omit the field entirely so the WebUI knows the inference failed.
    """
    fake_exc = type(
        "WeirdError",
        (Exception,),
        {"__module__": "some_random_package"},
    )("weird thing happened")
    payload = _serde_mod.default(fake_exc)
    assert "provider" not in payload


def test_api_key_redacted_in_message():
    """If a provider error echoes the API key back in its message, the
    patched encoder must scrub it before serialization.
    """
    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )(
        "Invalid API key: sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890. "
        "Get a new one at https://platform.openai.com/api-keys"
    )
    payload = _serde_mod.default(fake_exc)
    assert "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ" not in payload["message"]
    assert "<redacted>" in payload["message"]
    # The actionable advice and class info survive redaction.
    assert "Invalid API key" in payload["message"]
    assert "platform.openai.com" in payload["message"]


def test_redaction_covers_multiple_key_shapes():
    fake_exc = type(
        "APIError",
        (Exception,),
        {"__module__": "openai"},
    )(
        "Failures: sk-or-aBcDeFg012345678901234, AIzaABCDEFGHIJ0123456789, Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Im"
    )
    payload = _serde_mod.default(fake_exc)
    msg = payload["message"]
    assert "sk-or-" not in msg
    assert "AIza" not in msg
    assert "Bearer eyJ" not in msg
    # Three distinct redactions, one per key shape.
    assert msg.count("<redacted>") == 3


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
