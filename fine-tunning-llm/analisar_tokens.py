from math import ceil
from statistics import mean, median

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from app.services.fine_tuning_service import FineTuningService


servico = FineTuningService()
datasets = servico.gerar_datasets_treinamento()

caminho_modelo = snapshot_download(
    repo_id=servico.NOME_MODELO_BASE,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    caminho_modelo,
    local_files_only=True,
)

limites_por_split = {
    "treino": 80,
    "validacao": 10,
    "teste": 10,
}

quantidades = []

for nome_split, dataset in datasets.items():
    quantidade = limites_por_split[nome_split]

    amostra = dataset.shuffle(seed=42).select(
        range(min(quantidade, len(dataset)))
    )

    for exemplo in amostra:
        mensagens = [*exemplo["prompt"], *exemplo["completion"]]
        tokens = tokenizer.apply_chat_template(
            mensagens,
            tokenize=True,
            add_generation_prompt=False,
        )
        quantidades.append(len(tokens["input_ids"]))

valores = sorted(quantidades)
acima_256 = sum(quantidade > 256 for quantidade in valores)

print(f"Registros: {len(valores)}")
print(f"Média: {mean(valores):.2f}")
print(f"Mediana: {median(valores):.2f}")
print(f"Percentil 95: {valores[ceil(len(valores) * 0.95) - 1]}")
print(f"Máximo: {max(valores)}")
print(f"Registros acima de 256: {acima_256}")
print(f"Percentual acima de 256: {acima_256 / len(valores) * 100:.2f}%")