"""Adaptador LangChain para o modelo local ajustado (Llama ou Qwen)."""

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
    chave_modelo: str

    def gerar_resposta_modelo_ajustado(
        self,
        mensagem_system: str,
        mensagem_usuario: str,
        max_novos_tokens: int = 384,
    ) -> str:
        """Gera uma resposta local usando o adaptador LoRA."""


class ModeloChatLocal(BaseChatModel):
    """Expõe o adaptador LoRA local (Llama/Qwen) como chat model LangChain."""

    servico_fine_tuning: ServicoFineTuningAjustado
    max_novos_tokens: int = 384

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        chave = getattr(self.servico_fine_tuning, "chave_modelo", "local")
        return f"{chave}-lora-local"

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {
            "modelo_base": self.servico_fine_tuning.NOME_MODELO_BASE,
            "chave_modelo": getattr(
                self.servico_fine_tuning, "chave_modelo", "local"
            ),
        }

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


# Compatibilidade com o nome trazido da branch feature/langchain-langgraph.
ModeloChatQwenLocal = ModeloChatLocal
