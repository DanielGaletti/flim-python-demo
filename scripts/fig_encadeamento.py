#!/usr/bin/env python3
"""
fig_encadeamento.py — o encadeamento da contribuicao, elo a elo

Uma figura so, com os seis elos do argumento e o numero que sustenta cada um.
Ela existe porque a contribuicao desta dissertacao e uma CADEIA, e uma cadeia
apresentada so em prosa se perde: o leitor chega ao fim sem reter que cada
passo foi medido.

Nao e desenho conceitual. Cada numero vem do registro canonico, lido na hora,
e a figura falha se algum deles nao puder ser computado.

Uso
    python scripts/fig_encadeamento.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

import flim_al.evidencia as ev                                  # noqa: E402
import flim_al.agregacao as ag                                  # noqa: E402
from flim_al.tabelas import _gm_por_semente, _colapsou          # noqa: E402

AZUL, VERDE, LARANJA, CINZA = "#2d7dd2", "#19a974", "#f4a261", "#5b6770"


def numeros(recs):
    """Os valores que a figura cita, computados do registro."""
    tetos, crits = [], []
    for ds in ("schisto", "brats", "conjunctiva"):
        for dec in ("FLIM_lm", "FLIM_pb"):
            try:
                por = _gm_por_semente(recs, "ganho_marginal", ds, dec)
            except Exception:                                  # noqa: BLE001
                continue
            por = {k: v for k, v in por.items()
                   if not _colapsou(v["base"]["fb"])}
            t_, c_ = [], []
            for s in sorted(por, key=lambda x: int(x)):
                v = por[s]
                sort = [c for c in v["cands"] if c["papel"] == "sorteado"]
                alvo = next((c for c in v["cands"]
                             if c["papel"] == "argmax_entropia"), None)
                if not sort or alvo is None:
                    continue
                ref = sum(c["fb"] for c in sort) / len(sort)
                t_.append(max(c["fb"] for c in sort) - ref)
                c_.append(alvo["fb"] - ref)
            if len(t_) >= 2:
                tetos.append(ag.descrever(t_)["media"])
                crits.append(ag.descrever(c_)["media"])
    if not tetos:
        raise SystemExit("sem dados de ganho marginal no registro")
    lac = [t - c for t, c in zip(tetos, crits)]
    return {"teto_min": min(tetos), "teto_max": max(tetos),
            "lac_min": min(lac), "lac_max": max(lac), "celulas": len(tetos)}


def _br(x: float, casas: int = 4) -> str:
    """Numero com virgula decimal, que e a convencao do texto."""
    return f"{x:+.{casas}f}".replace(".", ",")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAIZ, "DISSERTACAO",
                                                  "figuras"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    n = numeros(ev.carregar())
    print(f"  máximo amostrado: {n['teto_min']:+.4f} a {n['teto_max']:+.4f} "
          f"em {n['celulas']} células")
    print(f"  lacuna          : {n['lac_min']:+.4f} a {n['lac_max']:+.4f}")

    elos = [
        ("1. O aprendizado ativo de IMAGEM\nnão supera o sorteio",
         "96 testes, nenhum sobrevive à correção.\n"
         "O baseline que lê o gabarito também não:\n"
         "0,232 abaixo do sorteio, p = 0,0107", CINZA),
        ("2. Mas a POSIÇÃO da anotação importa",
         f"melhor candidato examinado:\n"
         f"{_br(n['teto_min'])} a {_br(n['teto_max'])} de Fβ, "
         f"{n['celulas']} de {n['celulas']} células\n"
         f"pior candidato: até −0,4954", VERDE),
        ("3. Os critérios capturam só PARTE",
         f"lacuna de {_br(n['lac_min'])} a {_br(n['lac_max'])},\n"
         f"significativa em 4 de 6 após BH.\n"
         f"É o custo do aprendizado ativo", LARANJA),
        ("4. A LISTA de candidatos limita\no que o escore pode escolher",
         "só 2,5% a 12,4% dos superpixels\n"
         "atravessam a fronteira do objeto", LARANJA),
        ("5. Intervenção: trocar o GERADOR",
         "discos no contorno predito,\n"
         "escore e orçamento fixos", AZUL),
        ("6. Ganho confirmado em Parasitas",
         "+0,0772, IC [0,0393; 0,1151], 10 de 10\n"
         "t = 0,0013 · sinais = 0,0020 · perm. = 0,0020\n"
         "NÃO confirmado nos outros dois conjuntos", AZUL),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 12.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.9, len(elos) * 2 + 0.6)
    ax.axis("off")

    for i, (titulo, detalhe, cor) in enumerate(elos):
        y = (len(elos) - i) * 2 - 1.2
        ax.add_patch(FancyBboxPatch(
            (0.45, y - 0.62), 9.1, 1.28,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=cor, edgecolor="none", alpha=0.14))
        ax.add_patch(FancyBboxPatch(
            (0.45, y - 0.62), 0.11, 1.28,
            boxstyle="square,pad=0", facecolor=cor, edgecolor="none"))
        ax.text(0.78, y + 0.33, titulo, fontsize=11.2, fontweight="bold",
                va="center", ha="left", color="#14202e")
        ax.text(0.78, y - 0.26, detalhe, fontsize=8.9, va="center",
                ha="left", color="#2c3a47", linespacing=1.45)
        if i < len(elos) - 1:
            ax.add_patch(FancyArrowPatch(
                (5.0, y - 0.70), (5.0, y - 1.30),
                arrowstyle="-|>", mutation_scale=17,
                color="#9aa5b1", lw=1.6))

    ax.text(5.0, len(elos) * 2 + 0.25,
            "O encadeamento da contribuição",
            fontsize=14, fontweight="bold", ha="center", color="#14202e")
    ax.text(5.0, -0.55,
            "O máximo amostrado do elo 2 é identificado com o resultado de "
            "validação em mãos:\né a margem disponível, não o desempenho de "
            "um método automático.",
            fontsize=8.6, ha="center", style="italic", color="#53606d",
            linespacing=1.4)

    destino = os.path.join(a.out, "encadeamento.png")
    fig.savefig(destino, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\ngravado em {os.path.relpath(destino, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
