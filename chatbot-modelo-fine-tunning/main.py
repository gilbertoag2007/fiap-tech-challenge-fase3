import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELO_BASE = "Qwen/Qwen3-0.6B"
ADAPTADOR_LORA = (
    "gilbertoag2007/"
    "qwen3-0.6b-assistente-medico-ptbr-lora-fiap-gp-86"
)

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

TIPO_DADOS = (
    torch.float16
    if DISPOSITIVO == "cuda"
    else torch.float32
)


print("Carregando modelo...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTADOR_LORA)

modelo_base = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    torch_dtype=TIPO_DADOS,
)

modelo = PeftModel.from_pretrained(
    modelo_base,
    ADAPTADOR_LORA,
)

modelo.to(DISPOSITIVO)
modelo.eval()

print(f"Modelo carregado em: {DISPOSITIVO}")


def perguntar(pergunta: str) -> str:
    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um assistente acadêmico de apoio clínico. "
                "Responda em português brasileiro utilizando as seções: "
                "Resposta, Considerações clínicas, Conduta/Orientação "
                "e Limitações. Não faça diagnóstico ou prescrição de "
                "forma autônoma."
            ),
        },
        {
            "role": "user",
            "content": pergunta,
        },
    ]

    # O template é aplicado automaticamente.
    prompt = tokenizer.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    entradas = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DISPOSITIVO)

    with torch.inference_mode():
        resultado = modelo.generate(
            **entradas,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    inicio_resposta = entradas["input_ids"].shape[1]

    resposta = tokenizer.decode(
        resultado[0][inicio_resposta:],
        skip_special_tokens=True,
    )

    return resposta.strip()


# Chat interativo
print("\nDigite uma pergunta médica.")
print("Para encerrar, digite: sair\n")

while True:
    pergunta_usuario = input("Pergunta: ").strip()

    if pergunta_usuario.lower() == "sair":
        print("Chat encerrado.")
        break

    if not pergunta_usuario:
        continue

    resposta_modelo = perguntar(pergunta_usuario)

    print("\nResposta do modelo:")
    print(resposta_modelo)
    print()