"""Integration tests for the proactive commit shape.

Proves that the three-step commit composition (append via ``as_node="model"``
with stamped config metadata, then clear via ``as_node=END`` with the same
metadata) produces a visible, fresh-start assistant message on the source
thread.

The metadata stamping (``config["metadata"]`` carrying ``agent_name`` via
``build_metadata``) makes the clear's checkpoint the true ``agent_name``-stamped
head, so the proactive message is robustly visible in the head's channel_values
rather than only as a pending write on an older checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from EvoScientist.cli._constants import build_metadata
from EvoScientist.gateway import GraphTarget, LocalGraphGateway
from EvoScientist.sessions import (
    PruningCheckpointer,
    get_thread_messages,
    list_threads,
)


def _build_graph(saver: PruningCheckpointer) -> Any:
    """Compile a minimal graph with a ``messages`` channel and a ``model`` node."""

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    async def model_node(state: State) -> dict:
        return {"messages": [AIMessage(content="model response")]}

    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(checkpointer=saver)


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "proactive_commit.db")
    return db_path


@pytest.fixture
def patched_db_path(temp_db):
    """Patch ``get_db_path`` so ``get_thread_messages`` / ``list_threads`` read
    from the same temp DB the checkpointer writes to.
    """
    with patch("EvoScientist.sessions.get_db_path", return_value=Path(temp_db)):
        yield temp_db


async def test_proactive_commit_visible_with_metadata(patched_db_path, temp_db):
    """Append + clear with stamped metadata → message visible, next empty,
    thread in list_threads."""
    thread_id = "proactive-visible-001"
    meta = build_metadata("/tmp/ws", "test-model")

    async with PruningCheckpointer.from_conn_string_with_keep(
        temp_db, keep_per_ns=10
    ) as saver:
        await saver.setup()
        compiled = _build_graph(saver)
        target = GraphTarget(local_graph=compiled)
        gateway = LocalGraphGateway()

        # Seed a normal turn so the thread has agent_name-stamped checkpoints.
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"thread_id": thread_id}, "metadata": meta},
        )

        # Proactive commit: append tagged AIMessage, then clear next.
        proactive_msg = AIMessage(
            content="proactive update",
            additional_kwargs={"evoscientist": {"is_proactive_push": True}},
        )
        await gateway.update_state_values(
            target,
            thread_id,
            {"messages": [proactive_msg]},
            as_node="model",
            metadata=meta,
        )
        await gateway.update_state_values(
            target,
            thread_id,
            None,
            as_node=END,
            metadata=meta,
        )

    # After closing the checkpointer, read via the patched sessions functions.
    messages = await get_thread_messages(thread_id)
    assert any(m.content == "proactive update" for m in messages), (
        "Proactive message should be visible via get_thread_messages"
    )

    threads = await list_threads()
    assert any(t["thread_id"] == thread_id for t in threads), (
        "Thread should appear in list_threads"
    )

    # Reopen to check snapshot — next should be empty after END clear.
    async with PruningCheckpointer.from_conn_string_with_keep(
        temp_db, keep_per_ns=10
    ) as saver:
        compiled = _build_graph(saver)
        snapshot = await compiled.aget_state({"configurable": {"thread_id": thread_id}})
        assert snapshot.next == (), (
            "next should be empty after END clear so the next turn starts fresh"
        )


async def test_proactive_commit_survives_reopen(patched_db_path, temp_db):
    """The proactive message remains visible after the checkpointer connection
    is closed and reopened — simulates a process restart."""
    thread_id = "proactive-reopen-001"
    meta = build_metadata("/tmp/ws", "test-model")

    async with PruningCheckpointer.from_conn_string_with_keep(
        temp_db, keep_per_ns=10
    ) as saver:
        await saver.setup()
        compiled = _build_graph(saver)
        target = GraphTarget(local_graph=compiled)
        gateway = LocalGraphGateway()

        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"thread_id": thread_id}, "metadata": meta},
        )

        proactive_msg = AIMessage(content="proactive after restart")
        await gateway.update_state_values(
            target,
            thread_id,
            {"messages": [proactive_msg]},
            as_node="model",
            metadata=meta,
        )
        await gateway.update_state_values(
            target,
            thread_id,
            None,
            as_node=END,
            metadata=meta,
        )

    # Close happened. Reopen a fresh connection and verify the message
    # is still visible via get_thread_messages (reads from the DB directly).
    messages = await get_thread_messages(thread_id)
    assert any(m.content == "proactive after restart" for m in messages), (
        "Proactive message should survive checkpointer close/reopen"
    )
