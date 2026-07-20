"""Slash commands for TUI expert-skill selection.

``/experts`` — list installed expert skills.
``/expert <name>`` — toggle an expert into the current session's
``active_teams`` list; the next turn's ``configurable.active_teams`` picks
this up and ``ActiveTeamMiddleware`` biases the main-agent's delegation
toward the summoned expert(s).
``/expert clear`` — reset the list.

Backing store is ``ChannelRuntime.active_teams`` (see
``EvoScientist/commands/base.py``). WebUI users get the same effect via
its gallery + langgraph-sdk ``config.configurable``; these commands are
the TUI-side equivalent.
"""

from __future__ import annotations

from typing import ClassVar

from rich.table import Table

from ..base import Argument, Command, CommandContext, SubCommand
from ..manager import manager


class ExpertsCommand(Command):
    """List installed expert skills."""

    name: ClassVar[str] = "/experts"
    description: ClassVar[str] = "List installed expert skills"
    category: ClassVar[str] = "Experts"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...tools.skills_manager import list_expert_skills

        experts = list_expert_skills(include_system=True)
        active = _current_active_teams(ctx)

        if not experts:
            ctx.ui.append_system("No expert skills installed.", style="dim")
            ctx.ui.append_system(
                "Install with: /install-skill <path-or-url>", style="dim"
            )
            return

        table = Table(title=f"Expert Skills ({len(experts)})", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="dim")
        table.add_column("Dispatch", style="dim")
        table.add_column("Active", style="green")
        for skill in experts:
            marker = "*" if skill.name in active else ""
            table.add_row(
                skill.name,
                skill.role or skill.description,
                skill.default_dispatch or "sync",
                marker,
            )
        ctx.ui.mount_renderable(table)

        if active:
            ctx.ui.append_system(
                f"Active: {', '.join(active)}. Toggle with `/expert <name>`, "
                "clear with `/expert clear`.",
                style="dim",
            )
        else:
            ctx.ui.append_system(
                "No experts summoned. `/expert <name>` to summon one.",
                style="dim",
            )


class ExpertCommand(Command):
    """Summon, release, or clear expert skills for the current thread."""

    name: ClassVar[str] = "/expert"
    description: ClassVar[str] = "Summon or release an expert skill"
    category: ClassVar[str] = "Experts"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="name_or_clear",
            type=str,
            description="Expert skill name to toggle, or 'clear' to reset",
            required=True,
        )
    ]
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("clear", "Release all summoned experts"),
    ]

    def get_completions(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Complete expert names + the ``clear`` subcommand."""
        prefix = tokens[0].lower() if tokens else ""
        # Only complete at the first positional token; deeper is a usage error.
        if len(tokens) > 1 and tokens[1] != "":
            return []
        try:
            from ...tools.skills_manager import list_expert_skills

            candidates = [
                (s.name, s.role or s.description)
                for s in list_expert_skills(include_system=True)
            ]
        except Exception:
            candidates = []
        candidates.append(("clear", "Release all summoned experts"))
        return [(name, desc) for name, desc in candidates if name.startswith(prefix)]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        runtime = ctx.channel_runtime
        if runtime is None:
            ctx.ui.append_system(
                "/expert requires a session runtime; not available in this context.",
                style="yellow",
            )
            return

        if not args:
            ctx.ui.append_system(
                "Usage: /expert <name>   toggle an expert into the summon list",
                style="yellow",
            )
            ctx.ui.append_system(
                "       /expert clear    release all summoned experts",
                style="dim",
            )
            return

        target = args[0].strip()
        if target.lower() == "clear":
            if not runtime.active_teams:
                ctx.ui.append_system("No experts summoned.", style="dim")
                return
            released = list(runtime.active_teams)
            runtime.active_teams = []
            ctx.ui.append_system(
                f"Released experts: {', '.join(released)}", style="dim"
            )
            return

        from ...tools.skills_manager import list_expert_skills

        available = {s.name for s in list_expert_skills(include_system=True)}
        if target not in available:
            ctx.ui.append_system(
                f"No expert skill named '{target}'. `/experts` lists installed ones.",
                style="red",
            )
            return

        if target in runtime.active_teams:
            runtime.active_teams = [n for n in runtime.active_teams if n != target]
            ctx.ui.append_system(f"Released expert: {target}", style="dim")
        else:
            runtime.active_teams = [*runtime.active_teams, target]
            ctx.ui.append_system(f"Summoned expert: {target}", style="green")
        if runtime.active_teams:
            ctx.ui.append_system(
                f"Active: {', '.join(runtime.active_teams)}", style="dim"
            )


def _current_active_teams(ctx: CommandContext) -> list[str]:
    runtime = ctx.channel_runtime
    return list(runtime.active_teams) if runtime is not None else []


manager.register(ExpertsCommand())
manager.register(ExpertCommand())
