# ruff: noqa: INP001 (tests do plugin não são pacote; conftest monta o import)
"""dispatch_agent por agente (.md) em run assíncrona; transporte sintético."""

import json
from io import BytesIO
from pathlib import Path

import pytest
from hermes_herald import callback, config, tools

TIERS = [
    {"profile": "agent-ro", "covers": ["Read", "Grep"]},
    {"profile": "agent-web", "covers": ["Read", "Grep", "WebSearch"]},
]


def _write_agent(agents_dir: Path, name: str, tools_line: str, extra: str = "") -> None:
    """Grava um .md de agente no diretório temporário."""
    text = f"---\nname: {name}\ndescription: d\ntools: {tools_line}\n{extra}---\nCorpo do {name}\n"
    (agents_dir / f"{name}.md").write_text(text, encoding="utf-8")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Configura agentes, perfis e transporte de runs sintéticos."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write_agent(agents_dir, "scout", "WebSearch, Read", "model: haiku\neffort: low\n")
    _write_agent(agents_dir, "oracle", "Read, Grep", "model: opus\neffort: high\n")
    _write_agent(agents_dir, "writer", "Read, Edit")
    agents_cfg = {
        "dir": str(agents_dir),
        "tiers": TIERS,
        "model_map": {"haiku": "claude-haiku", "opus": "claude-opus"},
    }
    owner = {"session_key": "chat-1", "session_id": "session-1", "parent_session_id": "session-1"}
    monkeypatch.setattr(config, "get_agents_config", lambda: agents_cfg)
    monkeypatch.setattr(config, "get_ledger_file_path", lambda: tmp_path / "ledger.db")
    monkeypatch.setattr(config, "get_state_file_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(config, "get_active_profile_name", lambda: "origin")
    monkeypatch.setattr(callback, "capture_session_routing", lambda *_a: dict(owner))
    monkeypatch.setattr(callback, "start_listener", lambda **_kw: None)
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
    rec: dict = {"profiles": [], "requests": []}

    def resolve_profile(name: str, **_kwargs: object) -> tuple[dict, None]:
        rec["profiles"].append(name)
        return {"url": f"http://target.invalid/p/{name}", "api_key": "synthetic"}, None

    def verify(_profile: str, _pcfg: dict, requested: str, _provider: object) -> tuple[dict, None]:
        return {"requested_model": requested, "resolved_model": requested}, None

    def transport(req: object, **_kw: object) -> BytesIO:
        rec["requests"].append((req.full_url, json.loads(req.data)))
        return BytesIO(b'{"run_id":"run-1"}')

    monkeypatch.setattr(tools, "_resolve_profile", resolve_profile)
    monkeypatch.setattr(tools, "_verify_run_model_route", verify)
    monkeypatch.setattr(tools, "_get_json", lambda *_a, **_k: {"features": {}})
    monkeypatch.setattr(tools, "urlopen", transport)
    return rec


def _call(args: dict) -> dict:
    return json.loads(tools.handle_dispatch_agent(args))


def _assert_refused_offline(out: dict, env: dict) -> None:
    assert out.get("error")
    assert "run_id" not in out or not out["run_id"]
    assert env["requests"] == []
    assert env["profiles"] == []


def test_agent_resolves_tier_model_effort_and_body(env: dict) -> None:
    """Testa que o agente define perfil, modelo, esforço e instructions da run."""
    out = _call({"agent": "scout", "message": "preço do X"})
    assert out["run_id"] == "run-1"
    assert out["agent"] == "scout"
    assert env["profiles"] == ["agent-web"]
    url, body = env["requests"][0]
    assert url == "http://target.invalid/p/agent-web/v1/runs"
    assert body["input"] == "preço do X"
    assert body["instructions"] == "Corpo do scout"
    assert body["model"] == "claude-haiku"
    assert body["model_options"]["reasoning"]["effort"] == "low"


def test_caller_instructions_follow_agent_body(env: dict) -> None:
    """Testa que o corpo do .md vem antes das instructions do chamador."""
    _call({"agent": "scout", "message": "x", "instructions": "seja breve"})
    instructions = env["requests"][0][1]["instructions"]
    assert instructions.startswith("Corpo do scout")
    assert instructions.endswith("seja breve")


def test_explicit_model_and_effort_override_agent_route(env: dict) -> None:
    """Testa que model e reasoning_effort explícitos vencem a rota do agente."""
    _call({"agent": "oracle", "message": "x", "model": "other", "reasoning_effort": "low"})
    body = env["requests"][0][1]
    assert env["profiles"] == ["agent-ro"]
    assert body["model"] == "other"
    assert body["model_options"]["reasoning"]["effort"] == "low"


def test_agent_with_profile_is_refused(env: dict) -> None:
    """Testa que agent junto com profile é recusado sem rede."""
    _assert_refused_offline(_call({"agent": "scout", "profile": "agent-web", "message": "x"}), env)


@pytest.mark.parametrize(
    "args",
    [
        {"agent": "scout", "conversation": "scout:s1", "message": "x"},
        {"conversation": "scout:s1", "message": "x"},
    ],
)
def test_conversation_is_refused(env: dict, args: dict) -> None:
    """Testa que conversation é recusado: run é sempre sessão nova."""
    out = _call(args)
    _assert_refused_offline(out, env)
    assert "conversation" in out["error"]


def test_unknown_agent_is_refused(env: dict) -> None:
    """Testa que agente desconhecido é recusado sem rede."""
    out = _call({"agent": "ghost", "message": "x"})
    _assert_refused_offline(out, env)
    assert "ghost" in out["error"]


def test_agent_without_tier_is_refused(env: dict) -> None:
    """Testa que agente sem nível que cubra suas tools é recusado sem rede."""
    out = _call({"agent": "writer", "message": "x"})
    _assert_refused_offline(out, env)
    assert "Edit" in out["error"]


def test_agents_not_configured_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Testa que sem config de agentes nada é enviado."""
    monkeypatch.setattr(config, "get_agents_config", dict)
    out = _call({"agent": "scout", "message": "x"})
    _assert_refused_offline(out, env)
    assert "not configured" in out["error"]


def test_without_agent_keeps_profile_behavior(env: dict) -> None:
    """Testa que sem agent o envio segue idêntico ao caminho por perfil."""
    out = _call({"profile": "reviewer", "message": "x"})
    assert out["run_id"] == "run-1"
    assert "agent" not in out
    assert env["profiles"] == ["reviewer"]
    url, body = env["requests"][0]
    assert url == "http://target.invalid/p/reviewer/v1/runs"
    assert body == {"input": "x"}


def test_missing_profile_and_agent_is_refused(env: dict) -> None:
    """Testa que sem profile e sem agent o erro cita os dois."""
    out = _call({"message": "x"})
    assert "'profile' or 'agent'" in out["error"]
    assert env["requests"] == []


def test_schema_exposes_optional_agent() -> None:
    """Testa que o schema aceita agent e não exige profile."""
    params = tools.DISPATCH_AGENT_SCHEMA["parameters"]
    assert params["properties"]["agent"]["type"] == "string"
    assert "profile" not in params["required"]
    assert "conversation" not in params["properties"]
