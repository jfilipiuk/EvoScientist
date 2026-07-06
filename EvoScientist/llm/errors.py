"""Provider-error surface for langgraph SSE frames.

Provides :class:`ProviderStreamError` — a normalized, non-dataclass
exception raised by ``ErrorNormalizationMiddleware`` in place of the
provider SDK exception that a chat model call raised. Non-dataclass on
purpose: orjson's ``OPT_SERIALIZE_DATACLASS`` fast-path (enabled by
``langgraph_api.serde.json_dumpb``) enumerates dataclass fields
directly and skips the ``default=`` hook that builds our SSE envelope.
Some provider SDKs (openrouter today) decorate their exceptions with
``@dataclass``, so their errors bypass ``serde.default`` and leak raw
fields on the wire. Wrapping them in a plain ``Exception`` subclass
here forces orjson back onto the ``default=`` path.

The middleware pre-bakes the envelope fields onto the instance so
``langgraph_api.serde.default`` can emit the frame directly from
``.as_envelope()`` without redoing URL-host / module-prefix inference
at emit time.
"""

from __future__ import annotations

from typing import Any


class ProviderStreamError(Exception):
    """Envelope-shaped wrapper for a provider SDK exception raised
    inside a chat model call.

    Attributes mirror the SSE envelope one-for-one:

    - ``provider`` — concrete provider tag (``openai`` / ``anthropic``
      / ``deepseek`` / ``openrouter`` / ``openai_compat`` / …)
    - ``class_qualname`` — fully qualified name of the underlying
      exception's class (e.g. ``openrouter.errors.…``)
    - ``message`` — API-key-redacted ``str(original)``
    - ``status_code`` — HTTP status if the SDK exposed one
    - ``code`` — provider error code (``insufficient_quota``, …)
    - ``err_type`` — provider error type label (openai's ``.type``)
    - ``request_id`` — SDK-provided correlation id
    - ``original`` — the underlying exception, for chaining / debug
    """

    def __init__(
        self,
        provider: str,
        class_qualname: str,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        err_type: str | None = None,
        request_id: str | None = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.class_qualname = class_qualname
        self.message = message
        self.status_code = status_code
        self.code = code
        self.err_type = err_type
        self.request_id = request_id
        self.original = original

    def as_envelope(self) -> dict[str, Any]:
        """Return the SSE envelope dict — the shape ``serde.default``
        emits when it sees this class.
        """
        payload: dict[str, Any] = {
            "error": self.class_qualname.rsplit(".", 1)[-1],
            "class": self.class_qualname,
            "message": self.message,
            "provider": self.provider,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.code is not None:
            payload["code"] = self.code
        if self.err_type is not None:
            payload["type"] = self.err_type
        if self.request_id:
            payload["request_id"] = self.request_id
        return payload
