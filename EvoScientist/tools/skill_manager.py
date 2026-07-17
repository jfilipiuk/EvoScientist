"""Skill management tool (LangChain @tool wrapper)."""

from pathlib import Path
from typing import Literal

from langchain_core.tools import tool


def _build_invoke_str(skill_path: Path) -> str:
    """Format the ``Invoke:`` block for a skill given its installed directory.

    The ``/skills/`` virtual mount resolves by directory name, so the mount
    segment is derived from ``skill_path.name`` — never from the SKILL.md
    ``name:`` field, which can diverge when the same skill is checked out
    under multiple directory names.
    """
    mount = skill_path.name
    lines = [f"read_file /skills/{mount}/SKILL.md"]
    if (skill_path / "scripts" / "cli.py").exists():
        lines.append(
            f"execute: uv run python /skills/{mount}/scripts/cli.py <subcommand> ..."
        )
    return "\nInvoke:\n  " + "\n  ".join(lines)


def _format_install_block(entry: dict) -> str:
    """Format one installed-skill entry from an ``install_skill`` result.

    Handles both single-install returns (``{"name": ..., "path": ...}``) and
    the per-entry shape inside batch returns (``{"batch": True,
    "installed": [entry, ...]}``).
    """
    invoke_str = _build_invoke_str(Path(entry["path"]))
    return (
        f"Successfully installed skill: {entry['name']}\n"
        f"Description: {entry.get('description', '(none)')}\n"
        f"{invoke_str.lstrip()}"
    )


@tool(parse_docstring=True)
def skill_manager(
    action: Literal["install", "list", "uninstall", "info", "browse"],
    source: str = "",
    name: str = "",
    tag: str = "",
    include_system: bool = False,
) -> str:
    """Manage user-installable skills: install from GitHub or local path, list available skills, browse remote skills, get details, or uninstall.

    Actions and required parameters:

    action="install" (requires source):
      Install a skill. The source can be:
      - GitHub shorthand: "owner/repo@skill-name" (e.g. "anthropics/skills@peft")
      - GitHub URL: "https://github.com/owner/repo/tree/main/skill-name"
      - Local path: "./my-skill" or "/path/to/skill"
      Nested skills are auto-resolved — if the skill is not at the repo root, subdirectories are searched automatically.

    action="list":
      List installed skills. By default only shows user-installed skills.
      Set include_system=True to also show built-in system skills.
      Built-in skills evolve over time, so use action="list" to see the current set.

    action="browse" (optional tag):
      Browse available skills from the EvoSkills repository (EvoScientist/EvoSkills).
      Set tag to filter by category (e.g. tag="core", tag="writing", tag="experiments").
      Returns skill names, descriptions, tags, and install sources you can pass to action="install".

    action="info" (requires name):
      Get details (description, source, invocation forms, tags) about a specific skill by name.
      Searches both user and system skills.

    action="uninstall" (requires name):
      Remove a user-installed skill by name. System skills cannot be uninstalled.

    Args:
        action: The operation to perform — "install", "list", "browse", "info", or "uninstall"
        source: Required for install — GitHub shorthand, GitHub URL, or local directory path
        name: Required for info and uninstall — the skill name (for example, one returned by action="list")
        tag: Optional for browse — filter by tag (e.g. "core", "writing", "experiments", "research")
        include_system: Only for list — set True to include built-in system skills in the output

    Returns:
        Result message
    """
    from .skills_manager import (
        fetch_remote_skill_index,
        get_skill_info,
        install_skill,
        list_skills,
        uninstall_skill,
    )

    if action == "install":
        if not source:
            return (
                "Error: 'source' is required for install action. "
                "Provide a GitHub shorthand (e.g. source='owner/repo@skill-name'), "
                "a GitHub URL, or a local directory path."
            )
        result = install_skill(source)
        if not result["success"]:
            return f"Failed to install skill: {result['error']}"
        # Host paths are intentionally omitted for the same reason they're
        # omitted from the ``info`` branch below: the sandboxed agent can't
        # ``cd`` there (workspace cwd differs, ``cd`` doesn't persist across
        # execute calls), and surfacing them invites the exact
        # ``cd <host-path> && …`` chain the sandbox-path map exists to
        # prevent. Point at the virtual mount instead. Batch installs (a
        # source directory containing several SKILL.md subdirs) return
        # ``{"batch": True, "installed": [...]}`` without a top-level
        # ``name`` — handle that shape explicitly so the response doesn't
        # KeyError.
        entries = result["installed"] if result.get("batch") else [result]
        blocks = []
        for entry in entries:
            blocks.append(_format_install_block(entry))
        return "\n\n".join(blocks)

    elif action == "list":
        skills = list_skills(include_system=include_system)
        if not skills:
            if include_system:
                return "No skills found."
            return "No user skills installed. Use action='install' to add skills, or set include_system=True to see built-in skills."
        user_skills = [s for s in skills if s.source in ("workspace", "global")]
        system_skills = [s for s in skills if s.source == "builtin"]
        lines = []
        if user_skills:
            lines.append(f"User Skills ({len(user_skills)}):")
            for skill in user_skills:
                tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
                lines.append(f"  - {skill.name}: {skill.description}{tags_str}")
        if system_skills:
            if lines:
                lines.append("")
            lines.append(f"System Skills ({len(system_skills)}):")
            for skill in system_skills:
                tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
                lines.append(f"  - {skill.name}: {skill.description}{tags_str}")
        return "\n".join(lines)

    elif action == "browse":
        try:
            index = fetch_remote_skill_index()
        except Exception as e:
            return f"Failed to fetch skill index: {e}"
        if tag:
            tag_lower = tag.lower()
            index = [
                s for s in index if tag_lower in [t.lower() for t in s.get("tags", [])]
            ]
        if not index:
            return f"No skills found{' with tag: ' + tag if tag else ''}."
        lines = [f"Available Skills ({len(index)}):"]
        for s in index:
            tags_str = " · ".join(s.get("tags", []))
            lines.append(f"  - {s['name']}: {s['description']}")
            if tags_str:
                lines.append(f"    Tags: {tags_str}")
            lines.append(
                f"    Install: skill_manager(action='install', source='{s['install_source']}')"
            )
        return "\n".join(lines)

    elif action == "uninstall":
        if not name:
            return (
                "Error: 'name' is required for uninstall action. "
                "Use action='list' first to see installed skill names."
            )
        result = uninstall_skill(name)
        if result["success"]:
            return f"Successfully uninstalled skill: {name}"
        else:
            return f"Failed to uninstall skill: {result['error']}"

    elif action == "info":
        if not name:
            return (
                "Error: 'name' is required for info action. "
                "Use action='list' with include_system=True to see all available skill names."
            )
        info = get_skill_info(name)
        if not info:
            return (
                f"Skill not found: {name}. "
                f"Use action='list' with include_system=True to see all available skills."
            )
        tags_str = f"\nTags: {', '.join(info.tags)}" if info.tags else ""
        # Host path is intentionally omitted from the agent-visible output.
        # Surfacing it (`Path: /home/.../EvoScientist/skills/<name>`) invited
        # `cd <host-path> && …` chains that fail in the sandbox (the workspace
        # cwd is a different directory, and `cd` doesn't persist across execute
        # calls anyway). The `Invoke:` block below gives the only forms that
        # actually work.
        invoke_str = _build_invoke_str(info.path)
        return (
            f"Name: {info.name}\n"
            f"Description: {info.description}\n"
            f"Source: {info.source}{tags_str}{invoke_str}"
        )

    else:
        return f"Unknown action: {action}. Use 'install', 'list', 'browse', 'uninstall', or 'info'."
