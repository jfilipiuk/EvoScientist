"""ASGI middleware that deduplicates concurrent POSTs to the threaded
run-creation endpoints: ``/threads/{tid}/runs``, ``.../runs/stream``,
and ``.../runs/wait``.

Background: a client-side race can cause a single user turn to trigger
multiple POSTs against the same thread. Each duplicate arriving while
an earlier run is still processing causes ``langgraph_api`` to cancel
the in-flight run and roll back the checkpoint — but non-transactional
tool side effects (files written by ``execute``, CLI graph mutations,
etc.) are NOT rolled back. The retry re-invokes the tool and creates
duplicated persistent state. Observed shape: a single model turn where
the same tool ran twice under one dispatch, only one of the two
invocations visible in the persisted message list.

The original repro was on ``/runs/stream`` (SSE-timeout-driven client
reconnect); ``.../runs`` and ``.../runs/wait`` share the same server-
side cancel-on-duplicate behaviour and can hit the same race under
different client-side triggers (network hiccup, useEffect double-fire,
user double-click on a scheduled task, etc.). The WebUI uses all three:
``client.runs.create`` for fire-and-forget scheduled tasks,
``client.runs.wait`` for blocking sub-agent runs, ``client.runs.stream``
for the interactive chat path.

This middleware intercepts all three POST endpoints before they reach
``langgraph_api``'s ``protected_mount``. It hashes the request body and
tracks per-thread active runs. When a second POST arrives for the same
thread:

- Same body hash → dedup. Return ``409 Conflict`` and DO NOT forward the
  request downstream. The in-flight run continues untouched — no
  cancellation, no rollback, no side-effect duplication.
- Different body hash → supersede (legitimate cancel-and-resend, e.g.
  user amended the message). Fall through; ``langgraph_api``'s existing
  cancel-and-restart path handles the actual supersession.

Wire-up: registered on the Starlette sub-app in
``EvoScientist/langgraph_dev/http.py``. ``langgraph_api`` harvests
``user_router.user_middleware`` (``langgraph_api/server.py``) and applies
it globally as the outermost layer, so requests flow through here before
any of langgraph_api's routes see them.

**Not the root-cause fix.** The WebUI still needs to eliminate the extra
POSTs. This layer is a persistent-state floor — cheap defense-in-depth
so a race the WebUI hasn't tracked down can't corrupt on-disk state via
non-transactional tool writes.

Logging: hit events (``deduped`` / ``supersede`` / ``evicted``) write
unconditional stderr lines so they surface in ``langgraph_dev.log`` via
the subprocess-stderr redirection set up by
``langgraph_dev.manager.start_langgraph_dev``. Three lines only, each
fires on a meaningful state transition — no per-request debug logging.
Grep contract:

- ``grep run_stream_post_deduped`` — count of race hits
- ``grep run_stream_post_supersede`` — count of legit cancel-resends
- ``grep run_stream_post_evicted`` — count of run completions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Matches the three threaded run-creation endpoints, tolerating an
# arbitrary mount prefix (langgraph_api honors ``MOUNT_PREFIX``).
# Anchoring on ``/threads/`` means a leading prefix of any depth is fine
# while ``notthreads`` doesn't match. All three paths create a run on the
# named thread and are subject to the same client-race → cancel-and-
# restart pattern that this middleware defends against:
#
# - ``.../runs``       -> ``client.runs.create`` (fire-and-forget)
# - ``.../runs/stream`` -> ``client.runs.stream`` (SSE)
# - ``.../runs/wait``  -> ``client.runs.wait`` (synchronous wait)
#
# All three exist in ``langgraph_api.api.runs`` and share the cancel-
# on-duplicate behaviour. Stateless ``/runs`` / ``/runs/stream`` /
# ``/runs/wait`` (no ``/threads/{tid}/`` prefix) aren't deduped — those
# create a fresh thread per request and aren't subject to the race.
_RUN_POST_PATH_RE = re.compile(r"(?:^|/)threads/([^/]+)/runs(?:/stream|/wait)?/?$")


def _audit_log(msg: str) -> None:
    """Unconditional audit line to stderr.

    Fires on every hit-event (``deduped`` / ``supersede`` / ``evicted``).
    Surfaces in ``langgraph_dev.log`` via subprocess-stderr redirection;
    stereotyped ``INFO`` prefix means ``grep 'run_stream_dedup INFO'``
    catches all audit lines and per-event grep catches specific ones.
    Empty audit stream on days when the WebUI race doesn't fire; each
    line means either the race actually happened (``deduped``) or a
    normal run started/finished (``supersede`` / ``evicted``).
    """
    ts = datetime.now(UTC).isoformat(timespec="milliseconds")
    print(
        f"{ts} EvoScientist.middleware.run_stream_dedup INFO: {msg}",
        file=sys.stderr,
        flush=True,
    )


# Upper bound on request-body accumulation before we bypass dedup and let
# the request flow through untouched. ``/runs/stream`` payloads are small
# JSON documents (a few KB in practice); anything over ~8 MiB is almost
# certainly a misdirected upload rather than a legitimate stream request.
# Since this middleware fronts every user's every POST to that path, an
# unbounded buffer would let a single misbehaving client grow RSS by
# whatever they send.
_MAX_BODY_BYTES = 8 * 1024 * 1024


async def _drain_body(
    receive: Receive,
    max_bytes: int = _MAX_BODY_BYTES,
) -> tuple[bytes, list[Message], bool, bool]:
    """Read the full HTTP request body from ASGI ``receive``.

    Returns ``(body, messages, saw_eof, exceeded_max)`` where ``messages``
    is the exact list of ASGI messages consumed so the downstream app can
    be replayed the same stream shape verbatim.

    - ``saw_eof=True`` iff we observed ``more_body=False``. When ``False``,
      the client disconnected mid-body (or the stream ended abnormally) —
      caller must NOT hash+register the body: two truncated bodies both
      hash to the empty-string sha and would falsely dedup each other.
    - ``exceeded_max=True`` iff accumulated bytes exceed ``max_bytes``.
      Caller should bypass dedup and forward the request untouched; the
      downstream app reads the rest of the body via the underlying
      ``receive`` (``_make_replay_receive`` delegates once buffered
      messages are exhausted).
    """
    messages: list[Message] = []
    body = b""
    exceeded_max = False
    while True:
        msg = await receive()
        messages.append(msg)
        msg_type = msg.get("type")
        if msg_type == "http.request":
            body += msg.get("body", b"") or b""
            if len(body) > max_bytes:
                exceeded_max = True
                # Stop buffering — remaining chunks flow through the
                # replay-then-delegate receive without accumulating.
                return body, messages, False, exceeded_max
            if not msg.get("more_body", False):
                return body, messages, True, exceeded_max
        else:
            # ``http.disconnect`` or an unknown type — stream ended without
            # EOF. Return with ``saw_eof=False`` so the caller can decide
            # whether it's safe to hash-and-register.
            return body, messages, False, exceeded_max


def _make_replay_receive(messages: list[Message], receive: Receive) -> Receive:
    """Wrap ``receive`` so the first calls replay ``messages`` verbatim.

    Once the buffered messages are exhausted, subsequent calls delegate
    to the underlying ``receive`` so downstream can still observe a
    client disconnect that arrives after the body was fully sent.
    """
    idx = 0

    async def _replay() -> Message:
        nonlocal idx
        if idx < len(messages):
            m = messages[idx]
            idx += 1
            return m
        return await receive()

    return _replay


def _origin_from_scope(scope: Scope) -> bytes | None:
    """Return the request's ``Origin`` header (bytes) or ``None``.

    Needed to emit CORS headers on the 409 dedup response. This
    middleware is applied OUTSIDE ``langgraph_api``'s ``CORSMiddleware``
    (custom middleware sits outermost per
    ``langgraph_api/server.py``'s composition), so short-circuit
    responses would otherwise lack CORS headers and be dropped by
    cross-origin clients.
    """
    for name, value in scope.get("headers", []) or []:
        if name == b"origin":
            return value
    return None


def _origin_is_cors_allowed(origin: bytes) -> bool:
    """Return whether ``origin`` is on ``langgraph_api``'s CORS allowlist.

    Consulted before echoing ``Origin`` back on the 409 short-circuit so
    our error path doesn't widen the CORS surface beyond what
    ``langgraph_api``'s own ``CORSMiddleware`` would permit on the
    success path. If ``CORS_ALLOW_ORIGINS`` is ``"*"`` (the default),
    all origins are allowed; otherwise the origin must appear verbatim
    in the configured list. Import is deferred to keep the middleware
    importable in test environments where ``langgraph_api`` may not be
    installed; failure to import falls back to permissive behaviour
    (matches the pre-fix shape on non-``langgraph_api`` deploys).
    """
    try:
        from langgraph_api.config import CORS_ALLOW_ORIGINS
    except Exception:
        return True
    allowed = list(CORS_ALLOW_ORIGINS)
    if "*" in allowed:
        return True
    try:
        origin_str = origin.decode("latin-1")
    except Exception:
        return False
    return origin_str in allowed


async def _send_409(scope: Scope, send: Send, thread_id: str, body_hash: str) -> None:
    payload = json.dumps(
        {
            "error": "duplicate_run_stream_post",
            "thread_id": thread_id,
            "body_hash": body_hash,
            "detail": (
                "A run with an identical body is already in flight for this "
                "thread. The in-flight run was preserved; retry after it "
                "completes if a fresh run is really needed."
            ),
        }
    ).encode()
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
    ]
    origin = _origin_from_scope(scope)
    if origin is not None and _origin_is_cors_allowed(origin):
        # Echo the request Origin so browsers accept the response, but
        # only when ``langgraph_api``'s own CORS allowlist would have
        # permitted it — otherwise our error path is wider than the
        # success path (a rejected-origin client would see the 409 with
        # CORS headers while a real 200 from langgraph_api would be
        # blocked). Adds ``allow-credentials=true`` to match the
        # upstream config for cross-origin clients using ``credentials:
        # include``.
        headers.append((b"access-control-allow-origin", origin))
        headers.append((b"access-control-allow-credentials", b"true"))
        headers.append((b"vary", b"Origin"))
    await send(
        {
            "type": "http.response.start",
            "status": 409,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": payload, "more_body": False})


class RunStreamDedupMiddleware:
    """Dedup concurrent ``POST /threads/{tid}/runs/stream`` requests.

    Non-matching paths and non-POST methods pass through unchanged. See
    module docstring for the design rationale.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # ``thread_id`` -> {"body_hash": str, "started_at": float}. This
        # dict is per-middleware-instance and Starlette instantiates the
        # middleware exactly once per app-lifetime (verified in
        # ``starlette/applications.py:build_middleware_stack``, which
        # caches the stack after first request). The persistence-across-
        # requests assumption is load-bearing for the dedup contract — a
        # future refactor that reinstantiates the middleware per request
        # would silently drop all in-flight tracking.
        self._active: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        match = _RUN_POST_PATH_RE.search(scope.get("path", "") or "")
        if match is None:
            await self.app(scope, receive, send)
            return

        thread_id = match.group(1)
        body, buffered, saw_eof, exceeded_max = await _drain_body(receive)

        # Two shapes we refuse to register in ``_active``:
        #
        # 1. Truncated body (``saw_eof=False``) — client disconnected or
        #    stream ended abnormally. Two racing truncated requests both
        #    hash to ``sha256("")[:16]`` and would false-dedup each
        #    other. Forward as-is; downstream sees the disconnect and
        #    aborts naturally.
        # 2. Oversized body (``exceeded_max=True``) — almost certainly a
        #    misdirected upload. Bypass dedup and let the request flow
        #    through un-tracked; downstream reads remaining chunks via
        #    the underlying ``receive`` once buffered messages replay.
        if not saw_eof or exceeded_max:
            replay_receive = _make_replay_receive(buffered, receive)
            await self.app(scope, replay_receive, send)
            return

        # 16 hex chars = 64 bits of the sha256. Enough collision headroom
        # for the tiny key space (active runs per thread, order of 1 in
        # practice) while keeping dict entries small. False-collision
        # would require two POSTs with distinct bodies overlapping in
        # the same in-flight window on the same thread — vanishingly
        # unlikely at production scale.
        body_hash = hashlib.sha256(body).hexdigest()[:16]

        async with self._lock:
            active = self._active.get(thread_id)
            if active is None:
                decision = "insert"
                self._active[thread_id] = {
                    "body_hash": body_hash,
                    "started_at": time.monotonic(),
                }
            elif active["body_hash"] == body_hash:
                decision = "dedup"
                age_ms = int((time.monotonic() - active["started_at"]) * 1000)
                _audit_log(
                    f"run_stream_post_deduped thread_id={thread_id} "
                    f"body_hash={body_hash} age_ms={age_ms}"
                )
            else:
                decision = "supersede"
                old_hash = active["body_hash"]
                old_age_ms = int((time.monotonic() - active["started_at"]) * 1000)
                _audit_log(
                    f"run_stream_post_supersede thread_id={thread_id} "
                    f"old_hash={old_hash} new_hash={body_hash} "
                    f"old_age_ms={old_age_ms}"
                )
                self._active[thread_id] = {
                    "body_hash": body_hash,
                    "started_at": time.monotonic(),
                }

        if decision == "dedup":
            await _send_409(scope, send, thread_id, body_hash)
            return

        # ``insert`` and ``supersede`` both forward. Under ``supersede``,
        # langgraph_api's cancel-and-restart path fires as it does today;
        # the point of this branch is to log the event, not change its
        # behavior.
        replay_receive = _make_replay_receive(buffered, receive)
        started_at = time.monotonic()
        try:
            await self.app(scope, replay_receive, send)
        finally:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            async with self._lock:
                current = self._active.get(thread_id)
                # Only evict if we're still the current owner. Under a
                # supersede, the original request eventually returns here
                # and must NOT wipe the replacement's entry.
                if current is not None and current["body_hash"] == body_hash:
                    del self._active[thread_id]
                    _audit_log(
                        f"run_stream_post_evicted thread_id={thread_id} "
                        f"body_hash={body_hash} duration_ms={duration_ms}"
                    )
