# ruff: noqa: INP001 (tests do plugin não são pacote; conftest monta o import)
"""dispatch_chat por agente (.md) com conversa retomável; requisições sintéticas."""

import json
from pathlib import Path

import pytest
from hermes_herald import callback, config, tools

CREATED_SESSION = "sess-new"
TIERS = [
    {"profile": "agent-ro", "covers": ["Read", "Grep"]},
    {"profile": "agent-rw", "covers": ["Read", "Grep", "Edit", "Bash"]},
]


def _write_agent(agents_dir: Path, name: str, tools_line: str, extra: str = "") -> None:
    """Grava um .md de agente no diretório temporário."""
    text = f"---\nname: {name}\ndescription: d\ntools: {tools_line}\n{extra}---\nCorpo do {name}\n"
    (agents_dir / f"{name}.md").write_text(text, encoding="utf-8")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Configura agentes, perfil, sessão e streaming sintéticos."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write_agent(agents_dir, "oracle", "Read, Grep", "model: opus\neffort: high\n")
    _write_agent(agents_dir, "gapper", "Read")
    agents_cfg = {"dir": str(agents_dir), "tiers": TIERS, "model_map": {"opus": "go-pro"}}
    monkeypatch.setattr(config, "get_agents_config", lambda: agents_cfg)
    monkeypatch.setattr(config, "get_ledger_file_path", lambda: tmp_path / "ledger.db")
    monkeypatch.setattr(config, "get_state_file_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(tools, "_resolve_parent_agent", lambda *_args: None)
    rec: dict = {"posts": [], "gets": [], "streams": [], "stored": [], "profiles": []}

    def resolve_profile(name: str, **_kwargs: object) -> tuple[dict, None]:
        rec["profiles"].append(name)
        return {"url": "http://target.invalid", "api_key": "k"}, None

    def post_json(url: str, *_args: object, **_kwargs: object) -> dict:
        rec["posts"].append(url)
        return {"session": {"id": CREATED_SESSION}}

    def get_json(url: str, *_args: object, **_kwargs: object) -> dict:
        rec["gets"].append(url)
        if rec.get("get_404"):
            return {"run_id": url.rsplit("/", 1)[-1], "status": "not_found"}
        return {"session": {"id": url.rsplit("/", 1)[-1]}}

    def stream(_url: str, _key: str, body: dict, **kwargs: object) -> dict:
        rec["streams"].append({"body": body, **kwargs})
        if rec.get("stream_exc"):
            raise rec["stream_exc"]
        if rec.get("stream_error"):
            raise RuntimeError(rec["stream_error"])
        return {
            "reply": "ok",
            "session_id": rec.get("result_session") or kwargs["session_id"],
            "usage": {},
        }

    def verify(_profile: str, _pcfg: dict, requested: str, _provider: object) -> tuple[dict, None]:
        return {
            "requested_model": requested,
            "resolved_model": requested,
            "resolution_source": "alias",
        }, None

    monkeypatch.setattr(tools, "_resolve_profile", resolve_profile)
    monkeypatch.setattr(tools, "_post_json", post_json)
    monkeypatch.setattr(tools, "_get_json", get_json)
    monkeypatch.setattr(tools, "_post_streaming_chat", stream)
    monkeypatch.setattr(tools, "_verify_run_model_route", verify)
    real_store = callback.store_profile_session_id

    def store(profile: str, session_id: str) -> None:
        rec["stored"].append((profile, session_id))
        real_store(profile, session_id)

    monkeypatch.setattr(callback, "store_profile_session_id", store)
    return rec


def _call(args: dict) -> dict:
    return json.loads(tools.handle_dispatch_chat(args))


def _is_error(out: dict) -> bool:
    return "error" in out and out.get("status") != "completed"


def test_chat_with_agent_creates_session_and_returns_conversation(env: dict) -> None:
    """Testa que agente cria sessão nova e devolve o id de conversa."""
    out = _call({"agent": "oracle", "message": "revise"})
    assert out["status"] == "completed"
    assert out["conversation"] == f"oracle:{CREATED_SESSION}"
    assert out["agent"] == "oracle"
    sent_body = env["streams"][0]["body"]
    assert sent_body["messages"][0] == {"role": "system", "content": "Corpo do oracle"}
    assert env["profiles"] == ["agent-ro"]
    assert env["posts"] == ["http://target.invalid/api/sessions"]


def test_chat_resume_uses_session_from_conversation_and_resends_body(env: dict) -> None:
    """Testa que a retomada usa a sessão do id e reenvia o corpo."""
    out = _call({"conversation": "oracle:sess-9", "message": "e agora"})
    assert out["status"] == "completed"
    assert out["conversation"] == "oracle:sess-9"
    assert env["posts"] == []
    assert env["gets"] == ["http://target.invalid/api/sessions/sess-9"]
    assert env["streams"][0]["session_id"] == "sess-9"
    assert env["streams"][0]["body"]["messages"][0]["content"] == "Corpo do oracle"
    assert env["stored"] == []


def test_chat_with_agent_never_touches_profile_session_store(env: dict) -> None:
    """Testa que criação e troca de sessão não gravam no store por perfil."""
    env["result_session"] = "sess-swapped"
    out = _call({"agent": "oracle", "message": "x"})
    assert out["conversation"] == "oracle:sess-swapped"
    assert env["stored"] == []


def test_conversation_of_other_agent_is_refused(env: dict) -> None:
    """Testa que conversa de outro agente é recusada sem envio."""
    out = _call({"agent": "gapper", "conversation": "oracle:s", "message": "x"})
    assert _is_error(out)
    assert env["streams"] == []
    assert env["posts"] == []
    assert env["gets"] == []


def test_conversation_with_new_session_is_refused(env: dict) -> None:
    """Testa que conversation junto com new_session é recusado."""
    out = _call({"conversation": "oracle:s", "new_session": True, "message": "x"})
    assert _is_error(out)
    assert env["streams"] == []
    assert env["posts"] == []


def test_agent_with_profile_is_refused(env: dict) -> None:
    """Testa que agent junto com profile é recusado."""
    out = _call({"agent": "oracle", "profile": "marie", "message": "x"})
    assert _is_error(out)
    assert env["streams"] == []
    assert env["profiles"] == []


def test_unknown_session_on_target_returns_conversation_error(env: dict) -> None:
    """Testa que 404 no GET da sessão vira erro de conversa expirada."""
    env["get_404"] = True
    out = _call({"conversation": "oracle:gone", "message": "x"})
    assert "conversation expired; omit conversation to start a new one" in out["error"]
    assert env["streams"] == []
    assert env["posts"] == []


def test_streaming_404_after_session_check_is_not_conversation_error(env: dict) -> None:
    """Testa que 404 no streaming segue como erro de rota ou perfil."""
    env["stream_error"] = "HTTP 404 from http://target.invalid/v1/chat/completions"
    out = _call({"conversation": "oracle:sess-9", "message": "x"})
    assert _is_error(out)
    assert "conversation expired" not in out["error"]
    assert "Route for profile agent-ro returned 404" in out["error"]
    assert "new_session" not in out["error"]
    assert env["stored"] == []


def test_timeout_on_agent_path_keeps_conversation(env: dict) -> None:
    """Testa que timeout no caminho de agente devolve agent e conversation."""
    env["stream_exc"] = TimeoutError("stalled")
    out = _call({"conversation": "oracle:sess-9", "message": "x"})
    assert out["error_code"] == "chat_timeout"
    assert out["agent"] == "oracle"
    assert out["conversation"] == "oracle:sess-9"


def test_timeout_on_legacy_path_has_no_conversation(env: dict) -> None:
    """Testa que timeout no caminho legado não ganha campos de agente."""
    env["stream_exc"] = TimeoutError("stalled")
    out = _call({"profile": "p", "message": "x"})
    assert out["error_code"] == "chat_timeout"
    assert "conversation" not in out
    assert "agent" not in out


def test_session_id_is_escaped_in_existence_check(env: dict) -> None:
    """Testa que o id de sessão é escapado na URL do GET de existência."""
    _call({"conversation": "oracle:a/b", "message": "x"})
    assert env["gets"][0].endswith("/api/sessions/a%2Fb")


def test_caller_instructions_are_appended_after_agent_body(env: dict) -> None:
    """Testa que o corpo vem antes das instructions do chamador."""
    _call({"agent": "oracle", "message": "x", "instructions": "seja breve"})
    msgs = env["streams"][0]["body"]["messages"]
    assert msgs[0] == {"role": "system", "content": "Corpo do oracle"}
    assert msgs[1] == {"role": "system", "content": "seja breve"}
    assert msgs[2]["role"] == "user"


def test_agent_route_model_and_effort_are_used(env: dict) -> None:
    """Testa que modelo e esforço do agente chegam ao corpo."""
    _call({"agent": "oracle", "message": "x"})
    body = env["streams"][0]["body"]
    assert body["model"] == "go-pro"
    assert body["model_options"]["reasoning"]["effort"] == "high"


def test_explicit_model_and_effort_override_agent_route(env: dict) -> None:
    """Testa que model e reasoning_effort explícitos vencem a rota."""
    _call({"agent": "oracle", "message": "x", "model": "other", "reasoning_effort": "low"})
    body = env["streams"][0]["body"]
    assert body["model"] == "other"
    assert body["model_options"]["reasoning"]["effort"] == "low"


def test_agents_not_configured_is_refused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Testa que sem config de agentes nada é enviado."""
    monkeypatch.setattr(config, "get_agents_config", dict)
    out = _call({"agent": "oracle", "message": "x"})
    assert "not configured" in out["error"]
    assert env["streams"] == []


def test_chat_without_agent_keeps_legacy_profile_session(env: dict) -> None:
    """Testa que o caminho por perfil segue gravando a sessão do perfil."""
    callback.store_profile_session_id("legacy-p", "")
    out = _call({"profile": "legacy-p", "message": "x"})
    assert out["status"] == "completed"
    assert "conversation" not in out
    assert "agent" not in out
    assert ("legacy-p", CREATED_SESSION) in env["stored"]
    assert env["gets"] == []
    callback.store_profile_session_id("legacy-p", "")
