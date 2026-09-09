"""Adaptador LangChain para o modelo Qwen ajustado localmente."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict


@runtime_checkable
class ServicoFineTuningAjustado(Protocol):
    """Contrato mínimo necessário para a inferência do modelo ajustado."""

    NOME_MODELO_BASE: str

    def gerar_resposta_modelo_ajustado(
        self,
        mensagem_system: str,
        mensagem_usuario: str,
        max_novos_tokens: int = 384,
    ) -> str:
        """Gera uma resposta local usando o adaptador LoRA."""


class ModeloChatQwenLocal(BaseChatModel):
    """Expõe o Qwen ajustado como um modelo de chat do LangChain."""

    servico_fine_tuning: ServicoFineTuningAjustado
    max_novos_tokens: int = 384

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "qwen3-06b-lora-local"

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {"modelo_base": self.servico_fine_tuning.NOME_MODELO_BASE}

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        politicas: list[str] = []
        solicitacoes: list[str] = []
        for mensagem in messages:
            if not isinstance(mensagem.content, str):
                raise ValueError("O modelo aceita apenas conteúdo textual.")
            if isinstance(mensagem, SystemMessage):
                politicas.append(mensagem.content)
            else:
                solicitacoes.append(mensagem.content)

        resposta = self.servico_fine_tuning.gerar_resposta_modelo_ajustado(
            mensagem_system="\n\n".join(politicas),
            mensagem_usuario="\n\n".join(solicitacoes),
            max_novos_tokens=self.max_novos_tokens,
        )
        for sequencia in stop or []:
            if sequencia and sequencia in resposta:
                resposta = resposta.split(sequencia, maxsplit=1)[0]
                break

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=resposta))]
        )
