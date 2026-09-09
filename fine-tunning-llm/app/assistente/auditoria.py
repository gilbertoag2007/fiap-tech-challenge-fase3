"""Auditoria de metadados do assistente, sem conteúdo clínico."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class ServicoAuditoriaAssistente:
    """Registra eventos JSON Lines contendo exclusivamente metadados permitidos."""

    def __init__(
        self,
        caminho_arquivo: Path = Path("app/data/relatorios/auditoria_assistente.jsonl"),
    ) -> None:
        self.caminho_arquivo = caminho_arquivo

    def registrar(
        self,
        id_execucao: str,
        etapa: str,
        situacao: str,
        fontes: Sequence[str] = (),
        alertas: Sequence[str] = (),
        decisao_humana: bool | None = None,
        tipo_erro: str | None = None,
    ) -> None:
        """Acrescenta um evento sem dados clínicos ou identificador de registro."""
        evento: dict[str, object] = {
            "data_hora_utc": datetime.now(timezone.utc).isoformat(),
            "id_execucao": id_execucao,
            "etapa": etapa,
            "situacao": situacao,
            "fontes": sorted(set(fontes)),
            "alertas": sorted(set(alertas)),
        }
        if decisao_humana is not None:
            evento["decisao_humana"] = decisao_humana
        if tipo_erro is not None:
            evento["tipo_erro"] = tipo_erro

        self.caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
        with self.caminho_arquivo.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
