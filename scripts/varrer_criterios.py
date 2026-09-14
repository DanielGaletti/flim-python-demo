#!/usr/bin/env python3
"""
varrer_criterios.py — roda VÁRIAS técnicas de Active Learning na mesma campanha

Duas perguntas, dois runners, e eles NÃO são comparáveis entre si
    Esta é a decisão mais importante do script, e ela vem de um confundidor
    real que quase produziu uma conclusão falsa neste projeto.

    --modo imagem   `paper_selection.py --pool real`
        Seleciona QUAIS imagens anotar, do pool de 31 com marker REAL de
        especialista. Inclui `oracle`, que é o passo 7 do Algoritmo 1 do
        artigo — a seleção do FLIM. É esta campanha, e só ela, que responde
        "aplicar AL melhora sobre o FLIM?", porque os dois lados usam o mesmo
        traço humano.

    --modo regiao   `al_encoder_experiment.py`
        Seleciona ONDE anotar dentro da imagem (região, borda confusa), sobre
        o pool completo. Marker real só existe para 31 imagens, então este
        runner GERA markers sintéticos — o próprio cabeçalho dele diz isso.
        Responde "seleção de região ganha da seleção de imagem?", com o
        controle `random_region` para separar geometria de seleção.

    Por que não misturar: o braço do artigo usa marker real; os braços sobre
    o pool completo usam sintético. Comparar os dois mede QUEM DESENHOU O
    TRAÇO e apresenta como efeito da seleção. Numa primeira versão da tabela
    isso deu "AL perde do FLIM por -0.29, p<0.0001" — número real, conclusão
    falsa. `marker_origem` entrou no pareamento justamente para que esse par
    não se forme.

O pré-registro
    A lista sai declarada ANTES da primeira execução, gravada em
    `evidencia/campanhas/<id>.json` com commit e horário. Escolher a próxima
    técnica depois de ver o resultado da anterior é p-hacking mesmo sem
    intenção; o pré-registro torna isso verificável depois.

Uso
    python scripts/varrer_criterios.py --modo imagem --plano
    python scripts/varrer_criterios.py --modo regiao --plano

    # validar o encanamento em minutos
    python scripts/varrer_criterios.py --modo imagem \\
        --criterios oracle random coreset --splits 1 --seeds 2 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al import evidencia as ev  # noqa: E402

FLIM_AD = os.path.join(RAIZ, "flim_ad")

# Seleção de IMAGEM, pool de markers reais. `oracle` é o Algoritmo 1.
IMAGEM = [
    ("oracle",           "o FLIM do artigo: passo 7, escolhe por pior Fβ [usa GT]"),
    ("random",           "o piso: sorteio, sem critério"),
    ("entropy",          "incerteza pura, entropia do mapa de saliência"),
    ("least_confidence", "incerteza linear, 1 - |p-0.5|*2"),
    ("coreset",          "cobertura, k-center guloso no espaço de features"),
    ("badge",            "incerteza x diversidade, k-means++ nos gradientes"),
]

# Seleção de REGIÃO, pool completo, markers sintéticos.
REGIAO = [
    ("random",         "piso de imagem"),
    ("random_region",  "CONTROLE: seeds sorteados dentro da imagem"),
    ("region_entropy", "incerteza espacialmente resolvida"),
    ("region_margin",  "margem por região"),
    ("region_bald",    "desacordo de comitê por região"),
    ("coreset",        "referência de nível de imagem"),
    ("entropy",        "referência de nível de imagem"),
]

MODOS = {"imagem": IMAGEM, "regiao": REGIAO}


def registrar_campanha(modo, criterios, a) -> str:
    cid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = os.path.join(RAIZ, "evidencia", "campanhas")
    os.makedirs(destino, exist_ok=True)
    pre = {
        "campanha": cid,
        "modo": modo,
        "declarada_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": ev.git_commit_atual(),
        "criterios": criterios,
        "splits": a.splits,
        "seeds": a.seeds,
        "orcamento_max": a.max_images if modo == "imagem" else a.budgets,
        "marker_origem": "real" if modo == "imagem" else "sintetico",
        "observacao": (
            "Lista fixada antes da primeira execução. Alterar depois de ver "
            "resultado invalida a comparação. Campanhas de modos diferentes "
            "NÃO se comparam entre si: a origem do marker difere."),
    }
    caminho = os.path.join(destino, f"{cid}-{modo}.json")
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(pre, fh, ensure_ascii=False, indent=2)
    return caminho


def comando(modo, criterio, a):
    if modo == "imagem":
        return [sys.executable,
                os.path.join(RAIZ, "flim_al", "paper_selection.py"),
                "--criterion", criterio,
                "--pool", "real",
                "--splits", *map(str, a.splits),
                "--seeds", str(a.seeds),
                "--max_images", str(a.max_images),
                "--device", a.device]
    return [sys.executable,
            os.path.join(RAIZ, "flim_al", "al_encoder_experiment.py"),
            "--acquisition", criterio,
            "--splits", *map(str, a.splits),
            "--n_seeds", str(a.seeds),
            "--budgets", *map(str, a.budgets),
            "--markers", a.markers,
            "--device", a.device,
            "--save_dir", os.path.join(a.save_dir, criterio)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=list(MODOS), required=True)
    ap.add_argument("--criterios", nargs="+", default=None)
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--max_images", type=int, default=8,
                    help="modo imagem: até quantas imagens o laço adiciona")
    ap.add_argument("--budgets", nargs="+", type=int, default=[3, 5, 8],
                    help="modo regiao: orçamentos avaliados")
    ap.add_argument("--markers", default="schisto/user_A")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_dir", default="out/varredura")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    tecnicas = MODOS[a.modo]
    disponiveis = [t[0] for t in tecnicas]
    criterios = a.criterios or disponiveis
    desconhecidos = [c for c in criterios if c not in disponiveis]
    if desconhecidos:
        print(f"no modo {a.modo} não existe: {desconhecidos}")
        print(f"disponíveis: {disponiveis}")
        return 1

    origem = "real (31 imagens)" if a.modo == "imagem" else "sintético"
    print(f"modo {a.modo} · marker {origem} · splits {a.splits} · "
          f"{a.seeds} seeds\n")
    for nome, desc in tecnicas:
        marca = "  ✓" if nome in criterios else "   "
        print(f"{marca} {nome:<18} {desc}")

    if a.modo == "imagem":
        execucoes = len(criterios) * len(a.splits) * a.seeds * a.max_images
        print(f"\n{execucoes} execuções de encoder "
              f"(~{execucoes * 40 / 3600:.1f} h de GPU a ~40 s cada).")
        print("\nEsta campanha responde: aplicar AL melhora sobre o FLIM?")
        print("Os dois lados usam o MESMO marker real, então a diferença "
              "isola a seleção.")
        if "oracle" not in criterios:
            print("\nATENÇÃO: sem `oracle` não há a linha do Algoritmo 1, e a "
                  "campanha deixa de responder à pergunta central.")
    else:
        execucoes = len(criterios) * len(a.splits) * a.seeds * len(a.budgets)
        print(f"\n{execucoes} execuções de encoder "
              f"(~{execucoes * 40 / 3600:.1f} h de GPU).")
        print("\nEsta campanha responde: selecionar REGIÃO ganha de "
              "selecionar imagem?")
        print("Marker sintético nos dois lados. NÃO comparável com a campanha "
              "de modo `imagem`.")
        if "random_region" not in criterios:
            print("\nATENÇÃO: sem `random_region` não há controle de "
                  "geometria, e o ganho de região fica creditado à seleção.")

    if a.plano:
        print("\n--plano: nada foi executado.")
        return 0

    if execucoes > 200:
        print("\nCampanha longa. Confirme antes de disparar — horas de GPU "
              "não se recuperam.")

    caminho = registrar_campanha(a.modo, criterios, a)
    print(f"\npré-registro: {os.path.relpath(caminho, RAIZ)}\n")

    falhas = []
    for i, criterio in enumerate(criterios, 1):
        t0 = time.time()
        print(f"[{i}/{len(criterios)}] {criterio} …", flush=True)
        r = subprocess.run(comando(a.modo, criterio, a), cwd=FLIM_AD)
        dur = time.time() - t0
        if r.returncode != 0:
            falhas.append(criterio)
            print(f"    FALHOU (código {r.returncode}) após {dur:.0f}s")
        else:
            print(f"    ok em {dur:.0f}s")

    print()
    if falhas:
        print(f"{len(falhas)} técnica(s) falharam: {', '.join(falhas)}")
        print("A campanha ficou incompleta — diga isso ao reportar, em vez de "
              "apresentar as que rodaram como se fossem a comparação toda.\n")

    print("Agora, nesta ordem:")
    print("  bash scripts/arquivar_resultados.sh")
    print("  python scripts/migrar_evidencia.py")
    print("  python scripts/gerar_tabelas.py")
    print("  python -m pytest")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
