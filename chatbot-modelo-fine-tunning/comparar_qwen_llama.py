"""Compara respostas do chatbot entre Qwen 0.6B (80% q/k/v/o) e Llama 3.1 QLoRA."""
from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parents[1]
DIR_FT = RAIZ / "fine-tunning-llm"
DIR_SAIDA = DIR_FT / "app" / "data" / "relatorios"
DIR_SAIDA.mkdir(parents=True, exist_ok=True)

SISTEMA = (
    "Voce e um assistente especializado em apoio clinico. "
    "Responda somente a solicitacao, sem repetir o papel, o prontuario "
    "ou a pergunta. Use exatamente as secoes: Resposta, Consideracoes "
    "clinicas, Conduta/Orientacao e Limitacoes."
)

PERGUNTAS = [
    {
        "id": "P01",
        "titulo": "Soja e puberdade precoce",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Diagnostico\n"
            "Prontuario: Paciente do Sexo Nao informado. Idade: 7. "
            "Exames/Resultado: Nao informado.\n"
            "Pergunta: O suco de soja por um periodo pode causar "
            "puberdade precoce em criancas?"
        ),
    },
    {
        "id": "P02",
        "titulo": "Vacina COVID e leucopenia",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Paciente do Sexo Feminino. Idade: 12. "
            "Diagnosticos anteriores: leucopenia em acompanhamento.\n"
            "Pergunta: Adolescente com leucopenia pode receber "
            "vacina contra COVID-19?"
        ),
    },
    {
        "id": "P03",
        "titulo": "Hipotireoidismo e drenagem linfatica",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Procedimentos\n"
            "Prontuario: Paciente refere hipotireoidismo em tratamento.\n"
            "Pergunta: Paciente com hipotireoidismo pode fazer "
            "drenagem linfatica?"
        ),
    },
    {
        "id": "P04",
        "titulo": "Dor toracica aguda",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Diagnostico\n"
            "Prontuario: Paciente adulto com dor no peito ha 2 horas, "
            "sudorese e falta de ar. Exames nao informados.\n"
            "Pergunta: Como orientar esse caso de dor toracica aguda?"
        ),
    },
    {
        "id": "P05",
        "titulo": "Pos-colecistectomia",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Procedimentos\n"
            "Prontuario: Paciente 19 anos, cirurgia de emergencia de "
            "vesicula ha poucos dias, com nauseas, vomitos e dor "
            "abdominal.\n"
            "Pergunta: Solicita-se orientacao sobre esse quadro "
            "pos-operatorio."
        ),
    },
    {
        "id": "P06",
        "titulo": "Febre em crianca",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Crianca de 3 anos, febre 39C ha 24h, sem "
            "sinais de alerta descritos. Exames nao informados.\n"
            "Pergunta: Qual a orientacao inicial para febre nessa idade?"
        ),
    },
    {
        "id": "P07",
        "titulo": "Alergia a AAS e analgésico",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Adulto com dor muscular. Alergia relatada a AAS.\n"
            "Pergunta: Pode indicar nimesulida nesse paciente?"
        ),
    },
    {
        "id": "P08",
        "titulo": "Diabetes e jejum de exames",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Exames\n"
            "Prontuario: Paciente com diabetes tipo 2 em uso de "
            "metformina, precisa coletar exames de sangue.\n"
            "Pergunta: Como orientar o jejum e a medicação no dia "
            "da coleta?"
        ),
    },
    {
        "id": "P09",
        "titulo": "Cefaleia recorrente",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Diagnostico\n"
            "Prontuario: Adulto com cefaleia recorrente ha 3 meses, "
            "sem sinais de alarme descritos.\n"
            "Pergunta: Quais consideracoes iniciais e para quem "
            "encaminhar?"
        ),
    },
    {
        "id": "P10",
        "titulo": "Gestante e infeccao urinaria",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Gestante de 20 semanas com disuria e "
            "urgencia urinaria. Exames nao informados.\n"
            "Pergunta: Como orientar a investigacao e conduta "
            "inicial?"
        ),
    },
    {
        "id": "P11",
        "titulo": "Asma e crise leve",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Adolescente asmatico com chiado leve, "
            "sem dificuldade respiratoria grave relatada.\n"
            "Pergunta: Qual orientacao inicial e sinais de alarme?"
        ),
    },
    {
        "id": "P12",
        "titulo": "Hipertensao e exercicio",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Orientacao\n"
            "Prontuario: Adulto com hipertensao controlada, deseja "
            "iniciar academia.\n"
            "Pergunta: Pode praticar atividade fisica e quais "
            "cuidados orientais?"
        ),
    },
    {
        "id": "P13",
        "titulo": "Otite em crianca",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Diagnostico\n"
            "Prontuario: Crianca de 4 anos com otalgia e febre baixa. "
            "Exames nao informados.\n"
            "Pergunta: Quais consideracoes e conduta inicial "
            "sugerida?"
        ),
    },
    {
        "id": "P14",
        "titulo": "Anemia e dieta",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Orientacao\n"
            "Prontuario: Adulto com anemia ferropriva confirmada, "
            "sem sangramento ativo descrito.\n"
            "Pergunta: Quais orientacoes dieteticas e de "
            "acompanhamento sao razoaveis?"
        ),
    },
    {
        "id": "P15",
        "titulo": "Lombalgia mecanica",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Adulto com dor lombar mecanica ha 5 dias, "
            "sem deficit neurologico descrito.\n"
            "Pergunta: Como orientar manejo inicial e quando "
            "investigar mais?"
        ),
    },
    {
        "id": "P16",
        "titulo": "Refluxo e alimentacao",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Orientacao\n"
            "Prontuario: Adulto com pirose frequente apos refeicoes, "
            "sem emagrecimento ou disfagia.\n"
            "Pergunta: Quais orientacoes iniciais de estilo de vida "
            "e quando encaminhar?"
        ),
    },
    {
        "id": "P17",
        "titulo": "Dermatite e corticoides topicos",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Adulto com lesao eczematosa em membro, "
            "sem infeccao evidente descrita.\n"
            "Pergunta: Pode orientar uso de corticoide topico e "
            "quais cuidados?"
        ),
    },
    {
        "id": "P18",
        "titulo": "Ansiedade e sintomas fisicos",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Diagnostico\n"
            "Prontuario: Adulto com palpitacoes e inquietacao em "
            "contexto de estresse, ECG nao informado.\n"
            "Pergunta: Como diferenciar avaliacao clinica inicial "
            "e encaminhamentos?"
        ),
    },
    {
        "id": "P19",
        "titulo": "Insuficiencia renal e contraste",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Exames\n"
            "Prontuario: Adulto com doenca renal cronica, precisa "
            "de tomografia com contraste.\n"
            "Pergunta: Quais cuidados e avaliacoes previamente "
            "necessarios?"
        ),
    },
    {
        "id": "P20",
        "titulo": "Herpes zoster e dor",
        "texto": (
            "Papel do solicitante: Medico(a)\n"
            "Contexto da solicitacao: Conduta e Tratamento\n"
            "Prontuario: Adulto com vesiculas dolorosas em faixa "
            "dermatomerica ha 2 dias.\n"
            "Pergunta: Qual orientacao inicial e quando "
            "encaminhar?"
        ),
    },
]

MODELOS = {
    "qwen_80pct_qkvo": {
        "rotulo": "Qwen3-0.6B 80% (q/k/v/o)",
        "base": "Qwen/Qwen3-0.6B",
        "adapter": str(
            DIR_FT
            / "app/modelos/qwen_06b_lora/qwen3_06b_lora_80pct_more_projections"
        ),
        "tipo": "qwen",
        "max_length": 512,
        "max_new_tokens": 300,
        "enable_thinking": False,
    },
    "llama31_qlora": {
        "rotulo": "Llama 3.1 8B Instruct QLoRA",
        "base": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "adapter": str(DIR_FT / "app/modelos/llama31_8b_instruct_lora"),
        "tipo": "llama",
        "max_length": 512,
        "max_new_tokens": 300,
        "enable_thinking": None,
    },
}


def limpar_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def carregar_qwen(cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(cfg["adapter"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"],
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    modelo = PeftModel.from_pretrained(base, cfg["adapter"], is_trainable=False)
    modelo.eval()
    return tokenizer, modelo


def carregar_llama(cfg: dict):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from huggingface_hub import snapshot_download

    caminho_base = snapshot_download(repo_id=cfg["base"], local_files_only=True)
    modelo, tokenizer = FastLanguageModel.from_pretrained(
        model_name=caminho_base,
        max_seq_length=cfg["max_length"],
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    modelo = PeftModel.from_pretrained(modelo, cfg["adapter"], is_trainable=False)
    FastLanguageModel.for_inference(modelo)
    modelo.eval()
    return tokenizer, modelo


def gerar(tokenizer, modelo, pergunta: str, cfg: dict) -> str:
    mensagens = [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": pergunta},
    ]
    kwargs_template = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if cfg.get("enable_thinking") is not None:
        kwargs_template["enable_thinking"] = cfg["enable_thinking"]
    prompt = tokenizer.apply_chat_template(mensagens, **kwargs_template)
    entradas = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["max_length"],
    )
    dispositivo = next(modelo.parameters()).device
    entradas = entradas.to(dispositivo)
    with torch.inference_mode():
        gerado = modelo.generate(
            **entradas,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    inicio = entradas["input_ids"].shape[1]
    return tokenizer.decode(gerado[0][inicio:], skip_special_tokens=True).strip()


def avaliar_estrutura(texto: str) -> int:
    t = texto.lower()
    return sum(
        1
        for chave in ("resposta", "considera", "conduta", "limita")
        if chave in t
    )


def detectar_loop(texto: str) -> bool:
    """Detecta repeticao excessiva de n-gramas (sintoma tipico de colapso)."""
    palavras = texto.lower().split()
    if len(palavras) < 40:
        return False
    trigramas = [
        " ".join(palavras[i : i + 3]) for i in range(len(palavras) - 2)
    ]
    if not trigramas:
        return False
    mais_comum = max(trigramas.count(t) for t in set(trigramas))
    return mais_comum >= 8


def main() -> None:
    resultados: dict = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "quantidade_perguntas": len(PERGUNTAS),
        "system": SISTEMA,
        "perguntas": PERGUNTAS,
        "modelos": {},
    }

    for chave, cfg in MODELOS.items():
        print(f"\n=== Carregando {cfg['rotulo']} ===")
        limpar_gpu()
        if cfg["tipo"] == "qwen":
            tokenizer, modelo = carregar_qwen(cfg)
        else:
            tokenizer, modelo = carregar_llama(cfg)

        respostas = []
        for item in PERGUNTAS:
            print(f"  > {item['id']} {item['titulo']}")
            texto = gerar(tokenizer, modelo, item["texto"], cfg)
            respostas.append(
                {
                    "id": item["id"],
                    "titulo": item["titulo"],
                    "resposta": texto,
                    "secoes_detectadas": avaliar_estrutura(texto),
                    "chars": len(texto),
                    "loop": detectar_loop(texto),
                }
            )
            print(
                f"    secoes={respostas[-1]['secoes_detectadas']} "
                f"chars={len(texto)} loop={respostas[-1]['loop']}"
            )

        resultados["modelos"][chave] = {
            "rotulo": cfg["rotulo"],
            "adapter": cfg["adapter"],
            "base": cfg["base"],
            "respostas": respostas,
        }

        del modelo, tokenizer
        limpar_gpu()

    n = len(PERGUNTAS)
    caminho_json = DIR_SAIDA / "comparacao_chatbot_qwen_vs_llama_20q.json"
    caminho_md = DIR_SAIDA / "comparacao_chatbot_qwen_vs_llama_20q.md"
    caminho_json.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    linhas = [
        "# Comparacao chatbot 20Q: Qwen 0.6B (80% q/k/v/o) vs Llama 3.1 QLoRA",
        "",
        f"Gerado em: {resultados['gerado_em']}",
        f"Perguntas: {n}",
        "",
        "## Resumo automatico",
        "",
        "| Modelo | Media secoes (0-4) | 4/4 secoes | Loops | Media chars |",
        "|---|---:|---:|---:|---:|",
    ]
    for chave, bloco in resultados["modelos"].items():
        resp = bloco["respostas"]
        media_sec = sum(r["secoes_detectadas"] for r in resp) / n
        quatro = sum(1 for r in resp if r["secoes_detectadas"] == 4)
        loops = sum(1 for r in resp if r["loop"])
        media_chars = sum(r["chars"] for r in resp) / n
        linhas.append(
            f"| {bloco['rotulo']} | {media_sec:.1f} | {quatro}/{n} | "
            f"{loops}/{n} | {media_chars:.0f} |"
        )

        bloco["metricas"] = {
            "media_secoes": round(media_sec, 2),
            "quatro_secoes": quatro,
            "loops": loops,
            "media_chars": round(media_chars, 1),
        }

    # placar simples: mais secoes e sem loop vence; empate se igual
    qwen = resultados["modelos"]["qwen_80pct_qkvo"]["respostas"]
    llama = resultados["modelos"]["llama31_qlora"]["respostas"]
    vitorias = {"qwen": 0, "llama": 0, "empate": 0}
    for rq, rl in zip(qwen, llama):
        score_q = rq["secoes_detectadas"] - (2 if rq["loop"] else 0)
        score_l = rl["secoes_detectadas"] - (2 if rl["loop"] else 0)
        if score_l > score_q:
            vitorias["llama"] += 1
        elif score_q > score_l:
            vitorias["qwen"] += 1
        else:
            vitorias["empate"] += 1
    resultados["placar_estrutura"] = vitorias
    caminho_json.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    linhas.extend(
        [
            "",
            "## Placar estrutural (secoes - penalidade por loop)",
            "",
            f"- Llama: **{vitorias['llama']}/{n}**",
            f"- Qwen: **{vitorias['qwen']}/{n}**",
            f"- Empates: **{vitorias['empate']}/{n}**",
        ]
    )

    for item in PERGUNTAS:
        linhas.extend(
            [
                "",
                f"## {item['id']} — {item['titulo']}",
                "",
                "### Pergunta",
                "",
                "```",
                item["texto"],
                "```",
            ]
        )
        for chave, bloco in resultados["modelos"].items():
            resp = next(r for r in bloco["respostas"] if r["id"] == item["id"])
            linhas.extend(
                [
                    "",
                    f"### {bloco['rotulo']}",
                    "",
                    f"_secoes={resp['secoes_detectadas']}/4 | "
                    f"chars={resp['chars']} | loop={resp['loop']}_",
                    "",
                    resp["resposta"] or "_(vazio)_",
                ]
            )

    caminho_md.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"\nSalvo em:\n- {caminho_json}\n- {caminho_md}")
    print(
        f"Placar: Llama {vitorias['llama']} | "
        f"Qwen {vitorias['qwen']} | empates {vitorias['empate']}"
    )


if __name__ == "__main__":
    main()
