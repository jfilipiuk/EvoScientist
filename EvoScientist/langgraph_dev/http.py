"""Custom HTTP routes mounted alongside the langgraph dev server.

The langgraph-api host supports a top-level ``http`` key in
``langgraph.json`` that names an ASGI app to mount on the same process
as the graph. We use it to surface the two backends WebUI-style
frontends need:

- ``/api/models`` — model registry for the WebUI ``/model`` picker.
- ``/api/async-notifications/stream/{thread_id}`` — SSE stream of
  async-subagent completion notifications, drained per originating
  thread so the frontend can inject a synthetic "task complete" turn
  without polling for status.

Why this lives here and not as a separate sidecar: the WebUI talks to
``EvoSci deploy``'s langgraph endpoint anyway, so one origin keeps the
WebUI's fetch logic simple — no CORS dance, no extra port to configure.

Why Starlette and not FastAPI: ``langgraph_api`` already depends on
Starlette; adding FastAPI would pull in pydantic v1-vs-v2 reconciliation
the deploy doesn't need. The routes here have no input models, just JSON
or SSE bodies, so the lower-level surface is sufficient.

Lightweight by design — module-level imports stick to ``config``,
``llm.models`` (registry only; no chat-model construction), and
Starlette itself. The ``cli.async_notifier`` import in the SSE handler
is deferred to call time so this module stays cheap on import.
"""

from __future__ import annotations

import asyncio
import json
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from EvoScientist.config import get_effective_config
from EvoScientist.llm.models import list_model_picker_entries

# SSE poll cadence: how often the ``/api/async-notifications/stream`` handler
# checks the per-thread queue between yields. 250 ms is fast enough that a
# completed sub-agent surfaces on the wire within one screen-refresh; low
# enough that an idle stream costs ~4 lock acquisitions/sec, which is
# negligible against the queue's O(1) check.
_SSE_POLL_INTERVAL_S = 0.25

# SSE heartbeat cadence: send a comment-line keepalive every 15 s of
# idle time so intermediate proxies (Cloudflare's 100s, nginx's 60s) don't
# close the connection as stale. SSE spec (§9.2.5) says clients MUST
# silently ignore comment lines, so this is invisible in the browser.
_SSE_HEARTBEAT_INTERVAL_S = 15.0

# Wire fields projected onto each SSE event. Deliberately excludes
# ``origin_cli_thread_id`` — the client already knows the thread it
# opened the stream against, so echoing it is redundant.
_NOTIFICATION_WIRE_FIELDS = (
    "task_id",
    "agent_name",
    "status",
    "received_at",
    "prompt",
    "kind",
)


async def get_models(_request: Request) -> JSONResponse:
    """Return the model registry as ``{entries, default}``.

    ``entries`` preserves the registry order so the WebUI picker can
    rank providers per short name the same way the backend would.
    Mirrors the TUI ``/model`` picker by appending locally-pulled
    Ollama models when ``ollama_base_url`` is configured — same
    ``discover_ollama_models()`` call, same 1.5-s timeout, same
    fail-soft semantics (the probe returns ``[]`` on any error, never
    raises). The TUI's "Custom Ollama model…" sentinel is intentionally
    omitted — that's a widget-specific input affordance, not part of
    the registry surface.

    ``default`` reflects the deployment's currently-configured fallback
    (``config.yaml``'s ``model`` / ``provider`` — what ``/model reset``
    would land on). Returned even when the configured pair isn't in
    the registry, so the picker can still label it.

    Uses ``get_effective_config()`` (not ``load_config()``) so env-var
    overrides like ``OLLAMA_BASE_URL`` from ``_ENV_MAPPINGS`` are
    honored — matching the deploy's actual model-building behavior.
    Offloaded to a thread because ``get_effective_config()`` calls
    ``find_dotenv(usecwd=True)`` which invokes ``os.getcwd()`` — a
    blocking syscall that langgraph-dev's ``blockbuster`` middleware
    refuses to allow on the async event loop (would surface as a 500).
    """
    cfg = await asyncio.to_thread(get_effective_config)
    entries = [
        {"name": name, "model_id": model_id, "provider": provider}
        for name, model_id, provider in await list_model_picker_entries(
            getattr(cfg, "ollama_base_url", None),
            include_custom_ollama=False,
        )
    ]
    return JSONResponse(
        {
            "entries": entries,
            "default": {"name": cfg.model, "provider": cfg.provider},
        }
    )


async def stream_async_notifications(request: Request) -> StreamingResponse:
    """Server-sent event stream of async-task completion notifications.

    One SSE event per :class:`~EvoScientist.cli.async_notifier.AsyncTaskNotification`
    routed to ``thread_id``. Consumers are WebUI-style frontends that need to
    surface a synthetic "task complete" turn to the main agent without polling
    for status; the TUI drains the same notifier queue in-process and does not
    use this endpoint.

    Payload shape (JSON, one event body per SSE ``data:`` line): the six
    fields projected by ``_NOTIFICATION_WIRE_FIELDS`` — ``task_id``,
    ``agent_name``, ``status``, ``received_at``, ``prompt``, ``kind``. The
    dataclass's internal ``origin_cli_thread_id`` is deliberately excluded:
    the client already knows the thread it opened the stream against, so
    echoing it back is redundant.

    Consume semantics: the queue entry is drained as it is committed to
    the wire — there is no read-vs-drain distinction. If the client
    disconnects mid-event, that specific event is already gone from the
    server queue; SSE has no at-least-once redelivery guarantee, and
    downstream idempotency (WebUI-side task_id-keyed dedup) covers the
    resulting gap.

    Per-thread only: ``drain_thread_notifications`` reads from
    ``_notifications_by_thread[thread_id]``. The unrouted bucket
    (notifications with ``origin_cli_thread_id=None``) is intentionally
    excluded from this endpoint to prevent multi-tab clients from receiving
    duplicate notifications for the same task.

    Lifetime: the connection is held open as long as the client is
    connected. There is no server-side cutoff. Client closes when the
    thread has no more active async subagents (WebUI-side policy) or on
    tab close.

    Heartbeat: an SSE comment line (``: keepalive\\n\\n``) fires every
    ``_SSE_HEARTBEAT_INTERVAL_S`` of idle time to keep intermediate
    proxies from evicting the connection. Real notifications reset the
    heartbeat timer since they serve the same keep-alive purpose.

    Not offloaded to a thread: ``drain_thread_notifications`` acquires a
    ``threading.Lock`` briefly and pulls from a ``queue.Queue`` — both
    O(1) and non-blocking. langgraph-dev's ``blockbuster`` middleware
    flags sync file I/O, not brief lock acquisitions.
    """
    from EvoScientist.cli.async_notifier import drain_thread_notifications

    thread_id = request.path_params["thread_id"]

    async def event_stream():
        last_activity_s = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            for notif in drain_thread_notifications(thread_id):
                payload = {k: getattr(notif, k) for k in _NOTIFICATION_WIRE_FIELDS}
                yield f"data: {json.dumps(payload)}\n\n"
                last_activity_s = time.monotonic()
            now_s = time.monotonic()
            if now_s - last_activity_s >= _SSE_HEARTBEAT_INTERVAL_S:
                yield ": keepalive\n\n"
                last_activity_s = now_s
            await asyncio.sleep(_SSE_POLL_INTERVAL_S)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


app = Starlette(
    routes=[
        Route("/api/models", get_models, methods=["GET"]),
        Route(
            "/api/async-notifications/stream/{thread_id}",
            stream_async_notifications,
            methods=["GET"],
        ),
    ]
)
