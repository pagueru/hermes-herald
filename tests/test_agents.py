# ruff: noqa: INP001  # tests/ do upstream não tem __init__.py
"""Tests for loading agent definitions from .md files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
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


TIERS = [
    {"profile": "agent-ro", "covers": ["Read", "Grep", "Glob", "Bash", "Skill"]},
    {"profile": "agent-web", "covers": ["Bash", "WebSearch", "WebFetch", "Skill"]},
]


def _spec(model: str | None, tools: list[str] | None = None) -> agents.AgentSpec:
    """Monta um AgentSpec mínimo para os testes de rota."""
    return agents.AgentSpec("oracle", "d", "b", tools or ["Read"], model, "high")


def test_select_tier_first_covering_wins() -> None:
    """Testa que o primeiro nível que cobre as tools vence."""
    assert agents.select_tier(["Read", "Bash"], TIERS) == "agent-ro"
    assert agents.select_tier(["Bash", "WebSearch"], TIERS) == "agent-web"


def test_select_tier_none_when_uncovered() -> None:
    """Testa que tools sem cobertura retornam None."""
    assert agents.select_tier(["Read", "WebSearch"], TIERS) is None


def test_resolve_route_maps_model_and_effort() -> None:
    """Testa o mapeamento de modelo e esforço."""
    route = agents.resolve_route(_spec("opus"), {"tiers": TIERS, "model_map": {"opus": "go-pro"}})
    assert (route.profile, route.model, route.reasoning_effort) == ("agent-ro", "go-pro", "high")


def test_resolve_route_inherit_model_is_none() -> None:
    """Testa que model inherit ou ausente vira None."""
    cfg = {"tiers": TIERS, "model_map": {}}
    assert agents.resolve_route(_spec("inherit"), cfg).model is None
    assert agents.resolve_route(_spec(None), cfg).model is None


def test_resolve_route_null_map_entry_is_target_default() -> None:
    """Testa que entrada nula no mapa vira None."""
    route = agents.resolve_route(_spec("opus"), {"tiers": TIERS, "model_map": {"opus": None}})
    assert route.model is None


def test_resolve_route_unmapped_model_raises() -> None:
    """Testa que modelo fora do mapa levanta erro com o nome."""
    with pytest.raises(agents.AgentRouteError, match="sonnet"):
        agents.resolve_route(_spec("sonnet"), {"tiers": TIERS, "model_map": {"opus": "x"}})


def test_resolve_route_uncovered_tools_raises() -> None:
    """Testa que tools sem nível levantam erro citando as tools."""
    with pytest.raises(agents.AgentRouteError, match="WebSearch"):
        agents.resolve_route(_spec(None, ["Read", "WebSearch"]), {"tiers": TIERS})


def test_resolve_route_bad_tiers_raises() -> None:
    """Testa que tiers ausente ou inválido não cobre nada."""
    with pytest.raises(agents.AgentRouteError):
        agents.resolve_route(_spec(None), {})
    with pytest.raises(agents.AgentRouteError):
        agents.resolve_route(_spec(None), {"tiers": "x"})


def test_conversation_id_roundtrip() -> None:
    """Testa criar e interpretar o id de conversa."""
    cid = agents.make_conversation_id("oracle", "sess-123")
    assert cid == "oracle:sess-123"
    assert agents.parse_conversation_id(cid) == ("oracle", "sess-123")


@pytest.mark.parametrize("bad", ["", "oracle", ":x", "oracle:"])
def test_parse_conversation_id_rejects_malformed(bad: str) -> None:
    """Testa que ids malformados levantam erro."""
    with pytest.raises(agents.AgentRouteError):
        agents.parse_conversation_id(bad)
