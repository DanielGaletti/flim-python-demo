#!/usr/bin/env python3
"""
fig_margem_captura.py — a margem disponivel, o que se captura, e o custo

Duas figuras, as duas lidas do registro canonico:

  fig_valor_do_local   o pior candidato, o sorteado e o melhor amostrado, lado
                       a lado, para que a AMPLITUDE do efeito de posicao fique
                       visivel de uma vez. E a figura do resultado central.

  fig_margem_captura   por celula, quanto da margem o criterio captura e
                       quanto fica para tras. A parte que fica e o custo do
                       aprendizado ativo definido no trabalho.

As duas existem para impedir uma leitura errada que o texto sozinho nao
impede: o maximo amostrado e identificado COM o resultado de validacao em
maos, e por isso aparece sempre marcado como referencia de potencial, nunca
como barra de desempenho de um metodo.

Uso
    python scripts/fig_margem_captura.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

import flim_al.evidencia as ev                                  # noqa: E402
import flim_al.agregacao as ag                                  # noqa: E402
from flim_al.tabelas import _gm_por_semente, _colapsou          # noqa: E402

NOME = {"schisto": "Parasitas", "brats": "BraTS",
        "conjunctiva": "Conjuntivite"}
CELULAS = [(d, k) for d in ("schisto", "brats", "conjunctiva")
           for k in ("FLIM_lm", "FLIM_pb")]

COR_PIOR, COR_SORT, COR_MELHOR = "#c0392b", "#7f8c8d", "#19a974"
COR_CAPT, COR_CUSTO = "#2d7dd2", "#f4a261"


def coletar(recs, criterio="argmax_entropia"):
    """Por célula: deltas de pior, melhor e do critério, contra o sorteio."""
    saida = {}
    for ds, dec in CELULAS:
        try:
            por = _gm_por_semente(recs, "ganho_marginal", ds, dec)
        except Exception:                                      # noqa: BLE001
            continue
        por = {k: v for k, v in por.items()
               if not _colapsou(v["base"]["fb"])}
        pior, melhor, crit = [], [], []
        for s in sorted(por, key=lambda x: int(x)):
            v = por[s]
            sort = [c for c in v["cands"] if c["papel"] == "sorteado"]
            alvo = next((c for c in v["cands"]
                         if c["papel"] == criterio), None)
            if not sort or alvo is None:
                continue
            ref = sum(c["fb"] for c in sort) / len(sort)
            pior.append(min(c["fb"] for c in sort) - ref)
            melhor.append(max(c["fb"] for c in sort) - ref)
            crit.append(alvo["fb"] - ref)
        if len(melhor) >= 2:
            saida[(ds, dec)] = {"pior": pior, "melhor": melhor, "crit": crit}
    return saida


def fig_valor_do_local(dados, destino):
    rot = [f"{NOME[d]}\n{k.replace('FLIM_', '')}" for d, k in dados]
    x = np.arange(len(rot))
    pior = [np.mean(v["pior"]) for v in dados.values()]
    melhor = [np.mean(v["melhor"]) for v in dados.values()]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    # Barra larga o bastante para o rotulo caber dentro dela: com 0,26 o
    # sinal de menos saia cortado pela borda.
    largura = 0.31
    ax.bar(x - largura, pior, largura, color=COR_PIOR,
           label="pior candidato examinado")
    ax.bar(x, [0] * len(x), largura, color=COR_SORT,
           label="candidato sorteado (referência)")
    ax.bar(x + largura, melhor, largura, color=COR_MELHOR,
           label="melhor candidato examinado (máximo amostrado)")
    for i, (p, m) in enumerate(zip(pior, melhor)):
        # Rotulo DENTRO da barra negativa: fora dele colidia com os rotulos
        # do eixo x quando a barra era longa, como no BraTS lm.
        ax.text(i - largura, p + 0.016, f"{p:+.3f}", ha="center", va="bottom",
                fontsize=8.0, color="white", fontweight="bold")
        ax.text(i + largura, m + 0.012, f"{m:+.3f}", ha="center",
                va="bottom", fontsize=8.5, color=COR_MELHOR,
                fontweight="bold")

    ax.axhline(0, color="#333333", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(rot, fontsize=9)
    ax.set_ylabel(r"$\Delta F_\beta$ contra o candidato sorteado")
    ax.set_title("O valor do local do clique: mesma imagem, mesmo orçamento "
                 "de pixels,\nsó muda qual região recebe a anotação",
                 fontsize=11.5, fontweight="bold", pad=34)
    # Legenda ACIMA da area de plotagem: dentro dela cobria a barra do BraTS.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              fontsize=8.5, framealpha=0.95, borderaxespad=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.045,
             "O melhor candidato é identificado COM o resultado de validação "
             "em mãos: é referência de potencial, não desempenho de um "
             "método.\nA amplitude entre a barra vermelha e a verde é a "
             "margem que a seleção regional teria para capturar.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(destino, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_margem_captura(dados, destino):
    rot = [f"{NOME[d]}\n{k.replace('FLIM_', '')}" for d, k in dados]
    x = np.arange(len(rot))
    capt, custo, ps = [], [], []
    for v in dados.values():
        m = np.mean(v["melhor"])
        c = np.mean(v["crit"])
        capt.append(max(c, 0.0))
        custo.append(m - max(c, 0.0))
        ps.append(ag.teste_pareado(v["melhor"], v["crit"])["p"])
    bh = ag.benjamini_hochberg(ps)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x, capt, 0.52, color=COR_CAPT,
           label="capturado pelo critério automático")
    ax.bar(x, custo, 0.52, bottom=capt, color=COR_CUSTO,
           label="não capturado: o custo do aprendizado ativo")
    for i, (c, k, q, sob) in enumerate(zip(capt, custo, bh["q"],
                                           bh["sobrevive"])):
        marca = "*" if sob else ""
        ax.text(i, c + k + 0.008, f"{k:.3f}{marca}", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
        if c > 0.02:
            ax.text(i, c / 2, f"{c:.3f}", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(rot, fontsize=9)
    ax.set_ylabel(r"$\Delta F_\beta$ contra o candidato sorteado")
    ax.set_title("Margem disponível, parcela capturada e custo do "
                 "aprendizado ativo", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.045,
             "A altura total é o máximo amostrado. O asterisco marca a lacuna "
             "que sobrevive à correção de Benjamini-Hochberg sobre as seis "
             "células.\nOnde o critério entrega valor negativo, a parcela "
             "capturada é mostrada como zero.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(destino, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "DISSERTACAO",
                                                  "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5})

    dados = coletar(ev.carregar())
    if not dados:
        print("sem dados de ganho marginal no registro")
        return 1
    for (ds, dec), v in dados.items():
        print(f"  {NOME[ds]:13s} {dec:8s} n={len(v['melhor']):2d}  "
              f"pior={np.mean(v['pior']):+.4f}  "
              f"melhor={np.mean(v['melhor']):+.4f}  "
              f"critério={np.mean(v['crit']):+.4f}")

    p1 = os.path.join(a.out, "valor_do_local.png")
    p2 = os.path.join(a.out, "margem_captura.png")
    fig_valor_do_local(dados, p1)
    fig_margem_captura(dados, p2)
    print(f"\ngravado em {os.path.relpath(p1, RAIZ)}")
    print(f"gravado em {os.path.relpath(p2, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
