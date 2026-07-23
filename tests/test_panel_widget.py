"""Unit tests for PanelWidget (TUI in-eval fan-out live view).

Widget-mount / compose paths need a Textual App context and are exercised
through the interactive TUI end-to-end. These tests cover construction and
state-transition helpers that do not touch the compositor.
"""

from __future__ import annotations

import unittest


class TestPanelWidgetState:
    """PanelWidget state transitions independent of the compositor."""

    def test_construction(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget

        w = PanelWidget("ci_eval_1")
        assert w.eval_id == "ci_eval_1"
        assert w.dispatch_count == 0
        assert w._is_active is True

    def test_summary_counts_all_running(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget, _DispatchRow

        w = PanelWidget("e1")
        w._rows["d1"] = _DispatchRow("innovator", "a")
        w._rows["d2"] = _DispatchRow("pragmatist", "b")
        running, ok, err = w._summary_counts()
        assert (running, ok, err) == (2, 0, 0)

    def test_summary_counts_mixed(self):
        from EvoScientist.cli.widgets.panel_widget import PanelWidget, _DispatchRow

        w = PanelWidget("e1")
        r1 = _DispatchRow("innovator", "a")
        r2 = _DispatchRow("pragmatist", "b")
        r3 = _DispatchRow("critic", "c")
        r1._status = "ok"
        r2._status = "err"
        w._rows["d1"] = r1
        w._rows["d2"] = r2
        w._rows["d3"] = r3
        running, ok, err = w._summary_counts()
        assert (running, ok, err) == (1, 1, 1)


class TestDispatchRow(unittest.TestCase):
    """_DispatchRow state transitions."""

    def test_construction_sets_running(self):
        from EvoScientist.cli.widgets.panel_widget import _DispatchRow

        row = _DispatchRow("idea-brainstorm", "innovator voice")
        assert row._subagent_type == "idea-brainstorm"
        assert row._label == "innovator voice"
        assert row._status == "running"
        assert row._duration_ms is None

    def test_elapsed_display_uses_recorded_duration(self):
        from EvoScientist.cli.widgets.panel_widget import _DispatchRow

        row = _DispatchRow("x", "y")
        row._duration_ms = 2500
        display = row._elapsed_display()
        assert "2.5" in display
