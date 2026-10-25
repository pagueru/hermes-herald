# ruff: noqa: INP001  # tests/ do upstream não tem __init__.py
"""Tests for loading agent definitions from .md files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hermes_herald import agents

if TYPE_CHECKING:
    from pathlib import Path


def test_load_agents_parses_frontmatter_and_body(tmp_path: Path) -> None:
    """Testa test_load_agents_parses_frontmatter_and_body."""
    (tmp_path / "oracle.md").write_text(
        "---\nname: oracle\ndescription: Revisor\ntools: Read, Grep, Bash\n"
        "model: opus\neffort: high\n---\nCorpo do oracle\n",
        encoding="utf-8",
    )
    specs = agents.load_agents(tmp_path)
    assert specs["oracle"].tools == ["Read", "Grep", "Bash"]
    assert specs["oracle"].model == "opus"
    assert specs["oracle"].effort == "high"
    assert specs["oracle"].body == "Corpo do oracle"
    assert specs["oracle"].description == "Revisor"


def test_load_agents_tools_as_list(tmp_path: Path) -> None:
    """Testa test_load_agents_tools_as_list."""
    (tmp_path / "a.md").write_text("---\nname: a\ntools: [Read, Grep]\n---\nx\n", encoding="utf-8")
    assert agents.load_agents(tmp_path)["a"].tools == ["Read", "Grep"]


def test_load_agents_skips_file_without_name(tmp_path: Path) -> None:
    """Testa test_load_agents_skips_file_without_name."""
    (tmp_path / "x.md").write_text("---\ndescription: sem nome\n---\ncorpo\n", encoding="utf-8")
    (tmp_path / "y.md").write_text("sem frontmatter\n", encoding="utf-8")
    assert agents.load_agents(tmp_path) == {}


def test_load_agents_missing_dir_returns_empty(tmp_path: Path) -> None:
    """Testa test_load_agents_missing_dir_returns_empty."""
    assert agents.load_agents(tmp_path / "nao-existe") == {}


def test_load_agents_skips_malformed_yaml(tmp_path: Path) -> None:
    """Testa que YAML quebrado não derruba os agentes válidos."""
    (tmp_path / "bad.md").write_text("---\nname: bad\ntools: [Read\n---\nx\n", encoding="utf-8")
    (tmp_path / "ok.md").write_text("---\nname: ok\n---\ncorpo\n", encoding="utf-8")
    assert list(agents.load_agents(tmp_path)) == ["ok"]
