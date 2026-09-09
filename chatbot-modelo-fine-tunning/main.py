"""Ponto de entrada legado — o chatbot foi unificado no menu do pipeline.

Use o assistente médico com HITL (revisão humana) em:

    cd ../fine-tunning-llm
    python main.py

No menu, escolha a opção 11 e selecione Llama ou Qwen.
Os modelos-base devem ser baixados localmente conforme necessário, por exemplo:

    hf download unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
    hf download Qwen/Qwen3-0.6B

Os adaptadores LoRA ficam em fine-tunning-llm/app/modelos/.
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)
    print(
        "Este script não carrega mais o adaptador remoto do Hub. "
        "Execute o menu do pipeline (opção 11)."
    )


if __name__ == "__main__":
    main()
