"""Carrega definições de agente a partir de arquivos .md com frontmatter."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .yaml_compat import safe_load

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSpec:
    """Definição imutável de um agente."""

    name: str
    description: str
    body: str
    tools: list[str]
    model: str | None
    effort: str | None


def _parse_tools(raw: object) -> list[str]:
    """Normaliza o campo tools (string separada por vírgula ou lista)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    raise ValueError("tools deve ser string ou lista")


def _opt_str(value: object) -> str | None:
    """Converte valor opcional em string não vazia ou None."""
    return str(value).strip() or None if value is not None else None


def _parse_agent(text: str) -> AgentSpec:
    """Interpreta o texto de um .md de agente; levanta ValueError se inválido."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("frontmatter ausente")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("frontmatter sem fechamento") from None
    try:
        meta = safe_load("\n".join(lines[1:end]))
    except Exception as exc:  # classe de erro do parser do host é desconhecida
        msg = f"yaml inválido: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(meta, dict):
        raise TypeError("frontmatter inválido")
    name = _opt_str(meta.get("name"))
    if name is None:
        raise ValueError("name ausente")
    return AgentSpec(
        name=name,
        description=_opt_str(meta.get("description")) or "",
        body="\n".join(lines[end + 1 :]).strip(),
        tools=_parse_tools(meta.get("tools")),
        model=_opt_str(meta.get("model")),
        effort=_opt_str(meta.get("effort")),
    )


def load_agents(agents_dir: Path) -> dict[str, AgentSpec]:
    """Lê os *.md do diretório; arquivo inválido é ignorado com aviso."""
    specs: dict[str, AgentSpec] = {}
    if not agents_dir.is_dir():
        return specs
    for path in sorted(agents_dir.glob("*.md")):
        try:
            spec = _parse_agent(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("agente ignorado %s: %s", path.name, exc)
            continue
        specs[spec.name] = spec
    return specs


class AgentRouteError(ValueError):
    """Erro de roteamento; a mensagem é instrução ao modelo chamador."""


@dataclass(frozen=True)
class AgentRoute:
    """Destino resolvido de um agente: perfil, modelo e esforço."""

    profile: str
    model: str | None
    reasoning_effort: str | None


def select_tier(tools: list[str], tiers: list[dict]) -> str | None:
    """Retorna o perfil do primeiro nível que cobre todas as tools."""
    needed = set(tools)
    for tier in tiers:
        if needed <= set(tier.get("covers", [])):
            return tier["profile"]
    return None


def resolve_route(spec: AgentSpec, agents_cfg: dict) -> AgentRoute:
    """Resolve perfil, modelo e esforço do agente; levanta AgentRouteError se inviável."""
    tiers = agents_cfg.get("tiers")
    profile = select_tier(spec.tools, tiers) if isinstance(tiers, list) else None
    if profile is None:
        msg = (
            f"Agent '{spec.name}' needs tools {spec.tools}; no tier covers them. "
            "No message was sent."
        )
        raise AgentRouteError(msg)
    model: str | None = None
    if spec.model is not None and spec.model != "inherit":
        model_map = agents_cfg.get("model_map")
        if not isinstance(model_map, dict) or spec.model not in model_map:
            msg = (
                f"Agent '{spec.name}' uses model '{spec.model}', which has no entry "
                "in agents.model_map. No message was sent."
            )
            raise AgentRouteError(msg)
        model = model_map[spec.model]
    return AgentRoute(profile=profile, model=model, reasoning_effort=spec.effort)


def make_conversation_id(agent: str, session_id: str) -> str:
    """Monta o id de conversa no formato <agente>:<session_id>."""
    return f"{agent}:{session_id}"


def parse_conversation_id(conversation: str) -> tuple[str, str]:
    """Separa o id de conversa no primeiro ':'; levanta AgentRouteError se malformado."""
    agent, sep, session_id = conversation.partition(":")
    if not sep or not agent or not session_id:
        msg = (
            f"Malformed conversation id '{conversation}'; expected '<agent>:<session_id>'. "
            "No message was sent."
        )
        raise AgentRouteError(msg)
    return agent, session_id


def describe_agents(specs: dict[str, AgentSpec], agents_cfg: dict) -> str:
    """Lista uma linha por agente roteável, ordenada por nome."""
    lines: list[str] = []
    for name in sorted(specs):
        try:
            resolve_route(specs[name], agents_cfg)
        except AgentRouteError:
            continue
        lines.append(f"- {name}: {specs[name].description}")
    return "\n".join(lines)


def with_agents_catalog(schema: dict, specs: dict[str, AgentSpec], agents_cfg: dict) -> dict:
    """Retorna cópia do schema com o catálogo de agentes na descrição.

    Os agentes são lidos no registro: mudança de .md vale após reiniciar o gateway.
    """
    catalog = describe_agents(specs, agents_cfg)
    if not catalog:
        return schema
    out = copy.deepcopy(schema)
    out["description"] = f"{out['description']}\n\nAgents available via agent=:\n{catalog}"
    return out
