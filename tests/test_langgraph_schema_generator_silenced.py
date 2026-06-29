"""Regression tests for the langgraph_api SchemaGenerator silencing patch.

Reproducer: mounting our ``/api/models`` custom Starlette app makes
langgraph_api call ``update_openapi_spec`` at startup, which iterates
EVERY route (ours + upstream's). Endpoints whose docstrings aren't
valid YAML hit a warning + traceback in the deploy log — purely noise,
since the existing fallback path already produces a usable schema
entry. The patch keeps the fallback shape but silences the log spam.
"""

from __future__ import annotations

import logging
import os

# ``langgraph_api.config`` reads several required env vars at import
# time via starlette's ``Config(...)`` helper. We don't actually use the
# DB or Redis here — any non-empty string keeps the loader happy.
os.environ.setdefault("DATABASE_URI", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URI", "redis://localhost:6379")

# Importing patches.py applies the eager module-level monkey-patch.
import langgraph_api.utils as _lgapi_utils

import EvoScientist.llm.patches as _patches

# Re-invoke the patch after env vars are set. Required because earlier test
# modules (e.g. test_llm.py) import patches.py *without* DATABASE_URI/
# REDIS_URI, which makes ``langgraph_api.utils`` fail to import inside the
# patch's bare ``except``; the loader swallows it and the flag stays False
# forever (Python won't re-run module-level code on subsequent imports).
# The patch function is idempotent (early-return on the flag), so calling
# it here is a no-op when the patch already landed and a successful retry
# when the prior import failed.
_patches._patch_langgraph_schema_generator_silence_warnings()


class _FakeEndpoint:
    """Minimal Starlette-like endpoint info for the schema generator."""

    def __init__(self, path: str, method: str, func):
        self.path = path
        self.http_method = method
        self.func = func


class _DocstringFixture:
    """The kinds of docstrings the patched generator must handle."""

    @staticmethod
    def prose_with_colon():
        """Endpoint summary.

        Query params:
            id: The thing you want.
        """

    @staticmethod
    def valid_yaml():
        """
        summary: A valid YAML docstring.
        description: Stays structured.
        """

    @staticmethod
    def no_docstring():
        pass


def _generator():
    return _lgapi_utils.SchemaGenerator(
        {"openapi": "3.1.0", "info": {"title": "test", "version": "0"}}
    )


def test_prose_docstring_no_longer_logs_warning(caplog):
    """The patched ``get_schema`` must not emit the upstream WARNING
    (with traceback) when a docstring fails YAML parsing.
    """
    routes_stub = []
    gen = _generator()
    endpoint = _FakeEndpoint("/x", "get", _DocstringFixture.prose_with_colon)
    # Bypass get_endpoints — we hand-feed one fake endpoint.
    gen.get_endpoints = lambda _routes: [endpoint]
    with caplog.at_level(logging.WARNING):
        schema = gen.get_schema(routes_stub)
    # Critical: no log records produced
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    # Schema still has the fallback shape
    entry = schema["paths"]["/x"]["get"]
    assert "description" in entry
    assert "Query params" in entry["description"]


def test_valid_yaml_docstring_keeps_structured_parse():
    """Endpoints with parseable YAML keep their structured metadata —
    we only changed the failure branch, not the success path.
    """
    gen = _generator()
    endpoint = _FakeEndpoint("/y", "get", _DocstringFixture.valid_yaml)
    gen.get_endpoints = lambda _routes: [endpoint]
    schema = gen.get_schema([])
    entry = schema["paths"]["/y"]["get"]
    assert entry.get("summary") == "A valid YAML docstring."
    assert entry.get("description") == "Stays structured."


def test_no_docstring_still_handled():
    """Endpoints with ``__doc__ = None`` must not raise — fallback uses
    empty string for ``description``.
    """
    gen = _generator()
    endpoint = _FakeEndpoint("/z", "get", _DocstringFixture.no_docstring)
    gen.get_endpoints = lambda _routes: [endpoint]
    schema = gen.get_schema([])
    entry = schema["paths"]["/z"]["get"]
    # Either description="" (fallback path) or structured (if YAML parse
    # of None happens to succeed somehow — implementation detail).
    # The contract is just "no exception, entry exists".
    assert isinstance(entry, dict)


def test_patch_flag_set():
    from EvoScientist.llm.patches import _langgraph_schema_silenced_patched

    assert _langgraph_schema_silenced_patched is True
