"""Tests for ``EvoScientist.middleware.run_stream_dedup``.

Verifies the ASGI-level dedup layer that defends against the client-side
duplicate-``/runs/stream``-POST race. See the middleware's module
docstring for the mechanism.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pytest

from EvoScientist.middleware.run_stream_dedup import RunStreamDedupMiddleware


def _body_hash(body: bytes) -> str:
    """Same hash the middleware computes — kept in sync for supersede asserts."""
    return hashlib.sha256(body).hexdigest()[:16]


class _StubDownstream:
    """Minimal ASGI app double.

    Records every call and — for HTTP scopes — lets tests drive the
    response timing via a per-invocation ``asyncio.Event``. Non-HTTP
    scopes (lifespan, websocket) return immediately; they don't need to
    simulate an in-flight run.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        release = asyncio.Event()
        entry: dict[str, Any] = {
            "scope": scope,
            "body": b"",
            "release": release,
            "returned": False,
        }
        self.calls.append(entry)
        if scope.get("type") != "http":
            # Pass-through for non-HTTP; don't drain or hold the call open.
            entry["returned"] = True
            return

        # Drain the replayed body so tests can assert it survived the
        # buffering intact.
        while True:
            msg = await receive()
            if msg.get("type") == "http.request":
                entry["body"] += msg.get("body", b"") or b""
                if not msg.get("more_body", False):
                    break
            else:
                break
        # Emit a trivial 200 response so ``send`` is exercised.
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await release.wait()
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        entry["returned"] = True


def _scope(
    path: str = "/threads/tid-A/runs/stream",
    method: str = "POST",
    *,
    origin: bytes | None = None,
) -> dict[str, Any]:
    headers: list[tuple[bytes, bytes]] = [(b"content-type", b"application/json")]
    if origin is not None:
        headers.append((b"origin", origin))
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers,
    }


def _body_receiver(body: bytes):
    """Return an ASGI ``receive`` callable that yields ``body`` once."""
    sent = False

    async def _receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            # Downstream may keep polling for a disconnect; block forever
            # in the test — the stub app never asks after body EOF.
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return _receive


class _Recorder:
    """Collect ASGI ``send`` messages so tests can inspect the response."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return m["status"]
        return None

    @property
    def body(self) -> bytes:
        parts = [
            m.get("body", b"")
            for m in self.messages
            if m.get("type") == "http.response.body"
        ]
        return b"".join(parts)

    @property
    def headers(self) -> dict[bytes, bytes]:
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return dict(m.get("headers", []))
        return {}


def test_non_http_scope_passes_through() -> None:
    """Websocket/lifespan/etc. scopes are forwarded verbatim."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        recorder = _Recorder()

        async def _receive() -> dict[str, Any]:
            return {"type": "lifespan.startup"}

        await mw({"type": "lifespan"}, _receive, recorder)
        assert len(downstream.calls) == 1

    asyncio.run(_run())


def test_non_matching_path_passes_through() -> None:
    """Requests to routes other than ``/runs/stream`` skip dedup logic."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        recorder = _Recorder()

        scope = _scope(path="/api/models", method="GET")
        task = asyncio.create_task(mw(scope, _body_receiver(b""), recorder))
        await asyncio.sleep(0)
        # The downstream app was invoked with no dedup bookkeeping.
        assert len(downstream.calls) == 1
        assert not mw._active
        downstream.calls[0]["release"].set()
        await task

    asyncio.run(_run())


def test_non_post_method_passes_through() -> None:
    """A GET to the stream path (unusual but possible) is not deduped."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        recorder = _Recorder()

        scope = _scope(method="GET")
        task = asyncio.create_task(mw(scope, _body_receiver(b""), recorder))
        await asyncio.sleep(0)
        assert len(downstream.calls) == 1
        assert not mw._active
        downstream.calls[0]["release"].set()
        await task

    asyncio.run(_run())


def test_first_post_registers_and_forwards() -> None:
    """A lone POST to the stream endpoint is forwarded and tracked."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        recorder = _Recorder()

        body = json.dumps({"input": "hello"}).encode()
        task = asyncio.create_task(mw(_scope(), _body_receiver(body), recorder))
        await asyncio.sleep(0)

        assert len(downstream.calls) == 1
        assert downstream.calls[0]["body"] == body
        assert "tid-A" in mw._active

        downstream.calls[0]["release"].set()
        await task
        assert "tid-A" not in mw._active  # evicted after downstream returned

    asyncio.run(_run())


def test_duplicate_same_body_dedups_with_409() -> None:
    """A second POST with an identical body is short-circuited.

    Key invariant: the first (in-flight) call is preserved. Duplicate
    returns 409 without ever touching the downstream app.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body = json.dumps({"input": "hello"}).encode()
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(mw(_scope(), _body_receiver(body), r1))
        await asyncio.sleep(0)
        assert len(downstream.calls) == 1

        # Duplicate arrives; must not spawn a second downstream call.
        await mw(_scope(), _body_receiver(body), r2)
        assert len(downstream.calls) == 1
        assert r2.status == 409
        assert b"duplicate_run_stream_post" in r2.body
        assert json.loads(r2.body)["thread_id"] == "tid-A", (
            "the 409 payload should carry the thread_id"
        )
        # Load-bearing invariant of the whole PR: the first, in-flight
        # run MUST still be running. A regression that responds 409 but
        # also cancels the first stream would silently reintroduce the
        # cancel-restart race this middleware exists to prevent.
        assert downstream.calls[0]["returned"] is False, (
            "the first downstream call was closed/cancelled by the "
            "duplicate — dedup must preserve the in-flight run"
        )

        # First run wraps up cleanly and its entry gets evicted.
        downstream.calls[0]["release"].set()
        await first
        assert "tid-A" not in mw._active

    asyncio.run(_run())


def test_different_body_supersedes_and_forwards() -> None:
    """A second POST with a different body is treated as a legit resend.

    The middleware does NOT block it — ``langgraph_api``'s existing
    cancel-and-restart handles supersession. The second call must reach
    downstream; the ``_active`` entry now belongs to the newer request.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body1 = json.dumps({"input": "hello"}).encode()
        body2 = json.dumps({"input": "world"}).encode()
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(mw(_scope(), _body_receiver(body1), r1))
        await asyncio.sleep(0)
        assert len(downstream.calls) == 1

        second = asyncio.create_task(mw(_scope(), _body_receiver(body2), r2))
        await asyncio.sleep(0)
        assert len(downstream.calls) == 2

        # Active entry tracks the newer body, not the original.
        assert mw._active["tid-A"]["body_hash"] == _body_hash(body2)
        assert mw._active["tid-A"]["body_hash"] != _body_hash(body1)

        # Complete both downstream calls to unblock the tasks.
        downstream.calls[0]["release"].set()
        downstream.calls[1]["release"].set()
        await asyncio.gather(first, second)

        # After both finished, the replacement's eviction runs. The
        # superseded first call's finally block sees a mismatched hash
        # and correctly skips its eviction.
        assert "tid-A" not in mw._active

    asyncio.run(_run())


def test_different_threads_do_not_collide() -> None:
    """Concurrent POSTs to different thread_ids are independent."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body = json.dumps({"input": "hello"}).encode()
        r1 = _Recorder()
        r2 = _Recorder()

        t1 = asyncio.create_task(
            mw(_scope(path="/threads/tid-A/runs/stream"), _body_receiver(body), r1)
        )
        t2 = asyncio.create_task(
            mw(_scope(path="/threads/tid-B/runs/stream"), _body_receiver(body), r2)
        )
        await asyncio.sleep(0)

        assert len(downstream.calls) == 2
        assert set(mw._active) == {"tid-A", "tid-B"}
        # Neither response was short-circuited.
        assert r1.status != 409
        assert r2.status != 409

        for entry in downstream.calls:
            entry["release"].set()
        await asyncio.gather(t1, t2)

    asyncio.run(_run())


def test_evict_only_when_still_owner() -> None:
    """A superseded first request must NOT evict the replacement on exit.

    Simulates the ordering where the original in-flight request finishes
    AFTER the supersede has taken ownership. The finally block in
    ``__call__`` compares body_hash before eviction; the mismatch
    protects the replacement's entry.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body1 = json.dumps({"input": "hello"}).encode()
        body2 = json.dumps({"input": "world"}).encode()
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(mw(_scope(), _body_receiver(body1), r1))
        await asyncio.sleep(0)
        second = asyncio.create_task(mw(_scope(), _body_receiver(body2), r2))
        await asyncio.sleep(0)

        # Finish the SUPERSEDED (first) request FIRST — its eviction
        # attempt should be a no-op because it no longer owns the entry.
        downstream.calls[0]["release"].set()
        await first
        assert "tid-A" in mw._active, "replacement entry must survive first's exit"

        downstream.calls[1]["release"].set()
        await second
        assert "tid-A" not in mw._active

    asyncio.run(_run())


def test_409_includes_cors_headers_when_origin_present() -> None:
    """Cross-origin dedup responses must carry ``Access-Control-Allow-*``.

    Our middleware sits OUTSIDE ``langgraph_api``'s CORSMiddleware in the
    composition, so short-circuited 409s wouldn't otherwise pick up
    those headers. Browsers configured with ``credentials: include``
    would drop the response.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body = b'{"input": "hello"}'
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(
            mw(
                _scope(origin=b"http://localhost:3000"),
                _body_receiver(body),
                r1,
            )
        )
        await asyncio.sleep(0)

        await mw(
            _scope(origin=b"http://localhost:3000"),
            _body_receiver(body),
            r2,
        )
        assert r2.status == 409
        headers = r2.headers
        assert headers.get(b"access-control-allow-origin") == b"http://localhost:3000"
        assert headers.get(b"access-control-allow-credentials") == b"true"
        assert headers.get(b"vary") == b"Origin"

        downstream.calls[0]["release"].set()
        await first

    asyncio.run(_run())


def test_no_cors_headers_when_origin_absent() -> None:
    """Same-origin dedup responses omit CORS headers — no need to echo."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body = b'{"input": "hello"}'
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(mw(_scope(), _body_receiver(body), r1))
        await asyncio.sleep(0)
        await mw(_scope(), _body_receiver(body), r2)

        assert r2.status == 409
        assert b"access-control-allow-origin" not in r2.headers

        downstream.calls[0]["release"].set()
        await first

    asyncio.run(_run())


def test_matches_paths_with_mount_prefix() -> None:
    """Paths under a mount prefix (``MOUNT_PREFIX`` in langgraph_api) match too."""

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        recorder = _Recorder()

        body = b'{"x": 1}'
        task = asyncio.create_task(
            mw(
                _scope(path="/api/v1/threads/tid-X/runs/stream"),
                _body_receiver(body),
                recorder,
            )
        )
        await asyncio.sleep(0)
        assert "tid-X" in mw._active

        downstream.calls[0]["release"].set()
        await task

    asyncio.run(_run())


@pytest.mark.parametrize(
    "path",
    [
        "/threads/tid-A/runs",
        "/threads/tid-A/runs/",
        "/threads/tid-A/runs/wait",
        "/threads/tid-A/runs/wait/",
    ],
)
def test_regex_covers_non_stream_run_endpoints(path: str) -> None:
    """The three run-creation endpoints all get deduped.

    Verifies the widened regex covers ``client.runs.create`` (POST
    ``.../runs``) and ``client.runs.wait`` (POST ``.../runs/wait``)
    in addition to the SSE ``.../runs/stream`` variant. All three share
    the server-side cancel-on-duplicate behaviour that this middleware
    defends against.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body = b'{"x": 1}'
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(mw(_scope(path=path), _body_receiver(body), r1))
        await asyncio.sleep(0)
        # Duplicate arrives; must 409.
        await mw(_scope(path=path), _body_receiver(body), r2)
        assert r2.status == 409, (
            f"{path} was NOT deduped; second request reached downstream"
        )
        assert len(downstream.calls) == 1

        downstream.calls[0]["release"].set()
        await first

    asyncio.run(_run())


@pytest.mark.parametrize(
    "path",
    [
        # GET paths never reach the regex (short-circuited on method), but
        # sibling paths that AREN'T run creation must not accidentally
        # match either.
        "/threads/tid-A/runs/some-run-id",
        "/threads/tid-A/runs/some-run-id/join",
        "/threads/tid-A/state",
        "/threads/tid-A",
    ],
)
def test_regex_does_not_over_match_sibling_paths(path: str) -> None:
    """Non-run-creation paths on the same thread must NOT be deduped.

    Guards against the widened regex accidentally catching per-run
    GETs (``client.runs.get``), state fetches, or anything else that
    isn't the create/stream/wait triplet.
    """
    from EvoScientist.middleware.run_stream_dedup import _RUN_POST_PATH_RE

    assert _RUN_POST_PATH_RE.search(path) is None, (
        f"{path} matched the run-creation regex — over-match"
    )


def test_stateless_runs_stream_is_not_deduped() -> None:
    """``POST /runs/stream`` (no ``/threads/{tid}`` prefix) skips dedup.

    The stateless variant creates a fresh thread per request and is not
    subject to the race. The regex intentionally excludes it.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        r1 = _Recorder()
        r2 = _Recorder()

        body = b'{"x": 1}'
        t1 = asyncio.create_task(
            mw(_scope(path="/runs/stream"), _body_receiver(body), r1)
        )
        await asyncio.sleep(0)
        t2 = asyncio.create_task(
            mw(_scope(path="/runs/stream"), _body_receiver(body), r2)
        )
        await asyncio.sleep(0)

        # Both requests reached downstream — no dedup, no 409.
        assert len(downstream.calls) == 2
        assert not mw._active

        for entry in downstream.calls:
            entry["release"].set()
        await asyncio.gather(t1, t2)

    asyncio.run(_run())


def test_body_streamed_in_chunks_is_hashed_whole() -> None:
    """Chunked ASGI request bodies are drained fully before hashing.

    LangGraph's SSE clients don't chunk the small POST body in practice,
    but the ASGI spec permits it. Verifying the buffer handles it keeps
    the dedup key correct for any ASGI-compliant caller.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)
        r1 = _Recorder()
        r2 = _Recorder()

        chunk_a = b'{"input": '
        chunk_b = b'"hello"}'

        def _chunked_receiver():
            sent_a = False
            sent_b = False

            async def _receive():
                nonlocal sent_a, sent_b
                if not sent_a:
                    sent_a = True
                    return {"type": "http.request", "body": chunk_a, "more_body": True}
                if not sent_b:
                    sent_b = True
                    return {"type": "http.request", "body": chunk_b, "more_body": False}
                await asyncio.Event().wait()
                return {"type": "http.disconnect"}

            return _receive

        first = asyncio.create_task(mw(_scope(), _chunked_receiver(), r1))
        await asyncio.sleep(0)
        # Downstream received the reassembled body verbatim.
        assert downstream.calls[0]["body"] == chunk_a + chunk_b

        # A second call with the same reassembled body dedups.
        await mw(_scope(), _chunked_receiver(), r2)
        assert r2.status == 409

        downstream.calls[0]["release"].set()
        await first

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("audit_line", "expected_key"),
    [
        ("deduped", "run_stream_post_deduped"),
        ("supersede", "run_stream_post_supersede"),
    ],
)
def test_audit_log_grep_contract(
    audit_line: str, expected_key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deduped / supersede audit events each emit a line matching the
    documented grep key.

    The whole point of this middleware is countability from
    ``langgraph_dev.log``. If either key stopped emitting, the audit
    trail would silently vanish. ``evicted`` is exercised by its own
    test below so the three code paths (dedup branch, supersede
    branch, plain insert-and-evict) each have independent coverage.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body1 = b'{"a": 1}'
        body2 = b'{"a": 2}'
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(mw(_scope(), _body_receiver(body1), r1))
        await asyncio.sleep(0)

        if audit_line == "deduped":
            await mw(_scope(), _body_receiver(body1), r2)
        elif audit_line == "supersede":
            second = asyncio.create_task(mw(_scope(), _body_receiver(body2), r2))
            await asyncio.sleep(0)
            downstream.calls[1]["release"].set()
            await second

        downstream.calls[0]["release"].set()
        await first

    asyncio.run(_run())

    captured = capsys.readouterr()
    assert expected_key in captured.err, (
        f"expected {expected_key!r} in stderr; got:\n{captured.err}"
    )


def test_audit_log_evicted_fires_on_normal_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A plain insert-and-clean-return emits ``run_stream_post_evicted``.

    Isolated from the dedup / supersede paths so a future edit that
    skips the finally-block audit is caught here even if the other two
    audit lines still fire.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        r = _Recorder()
        task = asyncio.create_task(mw(_scope(), _body_receiver(b'{"a": 1}'), r))
        await asyncio.sleep(0)
        downstream.calls[0]["release"].set()
        await task
        # Sanity: no dedup / supersede fired.
        assert "tid-A" not in mw._active

    asyncio.run(_run())

    captured = capsys.readouterr()
    assert "run_stream_post_evicted" in captured.err, (
        f"expected 'run_stream_post_evicted' in stderr; got:\n{captured.err}"
    )
    assert "run_stream_post_deduped" not in captured.err
    assert "run_stream_post_supersede" not in captured.err


# ---------------------------------------------------------------------------
# Bypass paths — truncated body (client disconnect) and oversized body.
# Both must NOT register in ``_active`` (registering would create a
# hash-collision surface for false dedups) and must still forward the
# request downstream so langgraph_api sees a normal request lifecycle.
# ---------------------------------------------------------------------------


def _truncated_body_receiver(partial: bytes):
    """ASGI ``receive`` that sends a partial body then a disconnect.

    Simulates the client dropping the connection before ``more_body=False``
    is ever sent — the shape that would otherwise let two racing
    truncated requests both hash to ``sha256("")[:16]`` and dedup.
    """
    step = 0

    async def _receive() -> dict[str, Any]:
        nonlocal step
        step += 1
        if step == 1:
            return {"type": "http.request", "body": partial, "more_body": True}
        if step == 2:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    return _receive


def _oversized_body_receiver(size: int):
    """ASGI ``receive`` that streams ``size`` bytes and completes."""
    remaining = size
    chunk = b"x" * (1024 * 1024)

    async def _receive() -> dict[str, Any]:
        nonlocal remaining
        if remaining <= 0:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}
        send_now = min(remaining, len(chunk))
        remaining -= send_now
        return {
            "type": "http.request",
            "body": chunk[:send_now] if send_now < len(chunk) else chunk,
            "more_body": remaining > 0,
        }

    return _receive


def test_truncated_body_bypasses_dedup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Client-disconnect mid-body must NOT register the request.

    Regression guard: pre-fix, ``_drain_body`` returned whatever bytes it
    saw and the caller hashed the partial. Two clients both dropping
    mid-body would both hash to ``sha256("")[:16]`` (or two identical
    partials) and false-dedup each other, blocking a legitimate follow-up
    to the same thread with a 409.
    """

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        r1 = _Recorder()
        r2 = _Recorder()

        # First request drops mid-body. Downstream drains the replayed
        # (partial+disconnect) stream, opens a response, then waits on
        # release — so wrap in a task and release before awaiting.
        task1 = asyncio.create_task(mw(_scope(), _truncated_body_receiver(b"{par"), r1))
        await asyncio.sleep(0)
        # Truncated request MUST NOT register in _active.
        assert "tid-A" not in mw._active
        # Must have been forwarded (downstream sees the disconnect).
        assert len(downstream.calls) == 1
        downstream.calls[0]["release"].set()
        await task1

        # Second, unrelated truncated request to the same thread must
        # ALSO forward (not receive a 409) — proves no false-dedup on
        # shared empty-string hash.
        task2 = asyncio.create_task(mw(_scope(), _truncated_body_receiver(b"{oth"), r2))
        await asyncio.sleep(0)
        assert len(downstream.calls) == 2
        downstream.calls[1]["release"].set()
        await task2
        assert r2.status != 409, (
            "truncated bodies must not dedup each other; the second "
            "request is being blocked by a stale empty-hash entry"
        )

    asyncio.run(_run())

    captured = capsys.readouterr()
    assert "run_stream_post_deduped" not in captured.err


def test_oversized_body_bypasses_dedup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A body over ``_MAX_BODY_BYTES`` must bypass dedup, not accumulate.

    The middleware fronts every POST to the stream endpoint, so a
    misdirected multi-GB upload would grow RSS unbounded if we tried to
    hash it. Bypass path: forward the request untouched, don't register
    an entry, let the downstream app decide what to do with the body.
    """
    from EvoScientist.middleware.run_stream_dedup import _MAX_BODY_BYTES

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        r = _Recorder()
        # 1 MiB past the cap — enough to trigger the bypass without
        # allocating huge test buffers.
        task = asyncio.create_task(
            mw(
                _scope(),
                _oversized_body_receiver(_MAX_BODY_BYTES + 1024 * 1024),
                r,
            )
        )
        await asyncio.sleep(0)
        # Bypass must not register in _active.
        assert "tid-A" not in mw._active
        # But must still forward the request downstream.
        assert len(downstream.calls) == 1
        downstream.calls[0]["release"].set()
        await task

    asyncio.run(_run())

    captured = capsys.readouterr()
    # Bypass audit surfaces under the debug flag; ensure no unintended
    # dedup or supersede fired.
    assert "run_stream_post_deduped" not in captured.err
    assert "run_stream_post_supersede" not in captured.err


def test_409_omits_origin_when_not_cors_allowed(monkeypatch) -> None:
    """The 409 short-circuit must consult the CORS allowlist before
    echoing ``Origin`` — otherwise our error path is CORS-wider than
    ``langgraph_api``'s success path on non-``*`` deploys.
    """
    import EvoScientist.middleware.run_stream_dedup as mod

    # Force a non-``*`` allowlist that does NOT include the test origin.
    monkeypatch.setattr(
        mod,
        "_origin_is_cors_allowed",
        lambda origin: origin == b"https://allowed.example.com",
    )

    async def _run() -> None:
        downstream = _StubDownstream()
        mw = RunStreamDedupMiddleware(downstream)

        body = b'{"a": 1}'
        r1 = _Recorder()
        r2 = _Recorder()

        first = asyncio.create_task(
            mw(
                _scope(origin=b"https://allowed.example.com"),
                _body_receiver(body),
                r1,
            )
        )
        await asyncio.sleep(0)

        # Duplicate from a DISALLOWED origin: still 409, but NO CORS
        # echo. Browsers will reject the response cross-origin, which is
        # the correct behaviour — the request should not have reached us
        # under a non-``*`` allowlist in the first place.
        await mw(
            _scope(origin=b"https://evil.example.com"),
            _body_receiver(body),
            r2,
        )
        assert r2.status == 409
        header_names = [
            h[0].decode()
            for m in r2.messages
            if m.get("type") == "http.response.start"
            for h in m.get("headers", [])
        ]
        assert "access-control-allow-origin" not in header_names, (
            "disallowed origin was echoed on the 409 — CORS surface widened"
        )

        # Duplicate from an ALLOWED origin: 409 with the echo.
        r3 = _Recorder()
        await mw(
            _scope(origin=b"https://allowed.example.com"),
            _body_receiver(body),
            r3,
        )
        assert r3.status == 409
        allowed_header_names = [
            h[0].decode()
            for m in r3.messages
            if m.get("type") == "http.response.start"
            for h in m.get("headers", [])
        ]
        assert "access-control-allow-origin" in allowed_header_names

        downstream.calls[0]["release"].set()
        await first

    asyncio.run(_run())
