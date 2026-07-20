"""Integration tests for skill-script invocation forms.

Reviewer follow-up on PR#352 (din0s, comment 3612731770) asked whether
``python -m scripts.X`` — which some SKILL.md files document — works in
this sandbox given the prompt's strict ``cd`` ban.

The answer is skill-layout-dependent, so we pin both sides empirically:

- ``skill-creator`` ships a proper ``scripts`` package
  (``scripts/__init__.py`` present, imports use ``from scripts.xxx import``).
  ``uv run --directory /skills/skill-creator python -m scripts.quick_validate``
  works — this is the invocation form its SKILL.md documents.
- ``paper-navigator`` has a flat ``scripts/`` directory with sibling imports
  (``from utils import …``). The same ``-m`` invocation fails with
  ``ModuleNotFoundError`` because ``-m`` sets ``sys.path[0]`` to the skill
  root, not the scripts dir.

The file-path form the sandbox prompt actually documents
(``uv run python /skills/<name>/scripts/X.py …``) works for both layouts —
that's why it's the single form recommended in ``EvoScientist/prompts.py``.

These tests spawn ``uv run`` subprocesses against the real skills shipped
under ``EvoScientist/skills/`` so the assertion is grounded in actual
skill layouts rather than a synthetic fixture.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_CREATOR = REPO_ROOT / "EvoScientist" / "skills" / "skill-creator"
PAPER_NAVIGATOR = REPO_ROOT / "EvoScientist" / "skills" / "paper-navigator"


pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv binary not on PATH — invocation tests require uv",
)


def _run_uv(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.skipif(
    not SKILL_CREATOR.exists(),
    reason="skill-creator skill not present in this checkout",
)
def test_uv_directory_dash_m_works_for_package_layout():
    """``uv run --directory /path python -m scripts.X`` succeeds when the
    skill ships ``scripts/__init__.py`` (skill-creator's shape).

    This is the invocation form skill-creator's own SKILL.md documents.
    """
    assert (SKILL_CREATOR / "scripts" / "__init__.py").exists(), (
        "skill-creator's scripts must be a proper package for -m to work"
    )
    result = _run_uv(
        "--directory",
        str(SKILL_CREATOR),
        "python",
        "-m",
        "scripts.quick_validate",
        "--help",
    )
    assert result.returncode == 0, (
        f"skill-creator's `python -m scripts.quick_validate --help` "
        f"should succeed; stderr:\n{result.stderr}"
    )
    assert "usage:" in result.stdout.lower()


@pytest.mark.skipif(
    not PAPER_NAVIGATOR.exists(),
    reason="paper-navigator skill not present in this checkout",
)
def test_uv_directory_dash_m_fails_for_flat_sibling_imports():
    """``-m scripts.X`` breaks when the skill uses flat sibling imports
    (paper-navigator's shape: ``from utils import …`` inside a script
    whose sibling ``utils.py`` sits in ``scripts/``).

    ``-m`` sets ``sys.path[0]`` to the skill root, not the scripts dir,
    so the sibling module isn't importable.
    """
    result = _run_uv(
        "--directory",
        str(PAPER_NAVIGATOR),
        "python",
        "-m",
        "scripts.download_paper",
        "--help",
    )
    assert result.returncode != 0, (
        "paper-navigator's `-m scripts.download_paper` was expected to fail; "
        "if this now passes, the skill's script layout was fixed and the "
        "prompt's file-path-only guidance can be softened."
    )
    assert "ModuleNotFoundError" in result.stderr
    assert "'utils'" in result.stderr


@pytest.mark.skipif(
    not SKILL_CREATOR.exists(),
    reason="skill-creator skill not present in this checkout",
)
def test_file_path_invocation_works_for_package_layout():
    """The file-path form our prompt documents works regardless of layout.

    ``python /path/to/scripts/X.py`` sets ``sys.path[0]`` to the script's
    directory, so both package-layout and flat-sibling-import skills resolve
    their imports correctly.
    """
    script = SKILL_CREATOR / "scripts" / "quick_validate.py"
    result = _run_uv("python", str(script), "--help")
    assert result.returncode == 0, (
        f"File-path invocation of skill-creator's quick_validate failed; "
        f"stderr:\n{result.stderr}"
    )
    assert "usage:" in result.stdout.lower()
