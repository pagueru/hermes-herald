"""Carrega definições de agente a partir de arquivos .md com frontmatter."""

from __future__ import annotations

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
    meta = safe_load("\n".join(lines[1:end]))
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
