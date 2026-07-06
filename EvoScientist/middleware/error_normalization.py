"""ErrorNormalizationMiddleware — catch provider-SDK exceptions at the
model boundary and re-raise as a normalized non-dataclass wrapper.

Some provider SDKs (openrouter.errors.* today) decorate their exception
classes with ``@dataclass``. When langgraph_api emits an SSE error
frame via ``json_dumpb`` → ``orjson.dumps(obj, default=default,
option=OPT_SERIALIZE_DATACLASS)``, orjson's dataclass fast-path
enumerates the fields directly and skips the ``default=`` hook that
builds our envelope. The wire payload comes out as
``{"message": …, "status_code": …, "body": …, "headers": null,
"raw_response": null, "data": {…}}`` with no ``error`` / ``class`` /
``provider`` envelope and no way for the WebUI to distinguish quota /
auth / rate-limit / model-not-found.

This middleware sits at the model-call boundary. It catches
``BaseException`` from ``handler()``, and if the exception belongs to
a recognized provider module, wraps it in a
:class:`~EvoScientist.llm.errors.ProviderStreamError` (a plain
``Exception`` subclass, not a dataclass). The wrapper carries the SSE
envelope pre-baked on its instance attributes, so
``serde.default`` emits the envelope from a lookup rather than
re-inferring provider / status / redaction at emit time.

Non-provider exceptions (tool ``ValidationError``, ``RuntimeError``,
``asyncio.TimeoutError``, …) re-raise unchanged so upstream's
whitelist / catch-all behavior is preserved — same policy the
``serde.default`` patch already uses.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from ..llm.errors import ProviderStreamError


def _normalize(exc: BaseException) -> ProviderStreamError | None:
    """Return a :class:`ProviderStreamError` wrapping *exc* if it comes
    from a recognized provider module, or ``None`` if the caller
    should re-raise *exc* unchanged.

    Delegates provider / status / code / redaction extraction to the
    helpers in ``llm.patches`` so the envelope shape stays in one
    place.
    """
    from ..llm.errors import ProviderStreamError
    from ..llm.patches import (
        _extract_error_type,
        _extract_provider_code,
        _extract_status_code,
        _provider_from_exception,
        _redact_api_keys,
    )

    provider = _provider_from_exception(exc)
    if provider is None:
        return None
    cls = type(exc)
    mod = cls.__module__ or ""
    class_qualname = f"{mod}.{cls.__qualname__}" if mod else cls.__qualname__

    request_id_attr = getattr(exc, "request_id", None)
    request_id = (
        request_id_attr
        if isinstance(request_id_attr, str) and request_id_attr
        else None
    )

    return ProviderStreamError(
        provider=provider,
        class_qualname=class_qualname,
        message=_redact_api_keys(str(exc)),
        status_code=_extract_status_code(exc),
        code=_extract_provider_code(exc),
        err_type=_extract_error_type(exc),
        request_id=request_id,
        original=exc,
    )


class ErrorNormalizationMiddleware(AgentMiddleware):
    """Wrap the model call in try/except and normalize provider SDK
    exceptions into a non-dataclass envelope wrapper.

    Place this middleware **outermost** in the chain (first in the
    middleware list) so it catches exceptions raised by inner
    middlewares as well as the model handler itself.
    """

    name = "error_normalization"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        try:
            return handler(request)
        except BaseException as exc:
            normalized = _normalize(exc)
            if normalized is None:
                raise
            raise normalized from exc

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            return await handler(request)
        except BaseException as exc:
            normalized = _normalize(exc)
            if normalized is None:
                raise
            raise normalized from exc
