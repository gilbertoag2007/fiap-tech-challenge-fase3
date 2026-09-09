"""Estado transitório das sessões e eventos SSE."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any


class RegistroSessoes:
    """Mantém conversas apenas em memória; nada clínico é persistido."""

    def __init__(self) -> None:
        self._sessoes: dict[str, dict[str, Any]] = {}
        self._trava = RLock()

    def criar(self, id_execucao: str, modelo: str) -> None:
        with self._trava:
            self._sessoes[id_execucao] = {
                "id_execucao": id_execucao,
                "modelo": modelo,
                "status": "iniciando",
                "eventos": [],
            }
        self.adicionar_evento(
            id_execucao,
            {"no": "receber_pergunta", "status": "concluido"},
        )

    def adicionar_evento(
        self,
        id_execucao: str,
        evento: dict[str, object],
    ) -> None:
        with self._trava:
            sessao = self._obter_referencia(id_execucao)
            payload = {
                "sequencia": len(sessao["eventos"]),
                "data_hora_utc": datetime.now(timezone.utc).isoformat(),
                **evento,
            }
            sessao["eventos"].append(payload)
            status = evento.get("status")
            if isinstance(status, str):
                sessao["status"] = status

    def atualizar(self, id_execucao: str, **campos: object) -> None:
        with self._trava:
            self._obter_referencia(id_execucao).update(campos)

    def obter(self, id_execucao: str) -> dict[str, Any]:
        with self._trava:
            return deepcopy(self._obter_referencia(id_execucao))

    def eventos_desde(
        self,
        id_execucao: str,
        sequencia: int,
    ) -> list[dict[str, object]]:
        with self._trava:
            eventos = self._obter_referencia(id_execucao)["eventos"]
            return deepcopy(eventos[sequencia:])

    def _obter_referencia(self, id_execucao: str) -> dict[str, Any]:
        if id_execucao not in self._sessoes:
            raise KeyError(id_execucao)
        return self._sessoes[id_execucao]
