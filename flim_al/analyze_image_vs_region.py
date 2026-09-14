#!/usr/bin/env python3
"""
analyze_image_vs_region.py
==========================
A tabela central da dissertação: AL por SELEÇÃO DE IMAGEM vs AL por REGIÃO.

Os dois níveis respondem perguntas diferentes:

    imagem   quais imagens o especialista deve anotar?
             entropy, least_confidence, coreset, badge
    região   dentro da imagem, quais regiões vale anotar?
             region_bald (Exp. A), region_entropy (Exp. B)

Para os métodos de região existe o braço de controle `random_region` — imagens
sorteadas com o MESMO gerador de seeds do AL. Ele separa dois efeitos que
antes vinham somados:

    seleção    AL_região − random_região    quais imagens foram escolhidas
    geometria  random_região − random       como os seeds foram desenhados

Sem essa decomposição, o ganho aparente do region_bald era atribuído à
seleção quando a maior parte vem da geometria.

Uso:
    cd flim_ad && python ../flim_al/analyze_image_vs_region.py --out ../TABELA_IMAGEM_VS_REGIAO.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st
from collections import defaultdict

try:
    from scipy import stats as _sps
except ImportError:
    _sps = None

NIVEL = {
    "entropy": "imagem", "least_confidence": "imagem",
    "coreset": "imagem", "badge": "imagem",
    "region_bald": "região", "region_entropy": "região",
}
DECODER_REF = "labeled_marker"   # o decoder do paper


def _p(vals):
    """p do teste t de uma amostra contra zero."""
    if _sps is None or len(vals) < 2:
        return float("nan")
    return float(_sps.ttest_1samp(vals, 0.0).pvalue)


def fmt_p(p):
    if p != p:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def arm_vals(rows, arm, decoder):
    """Fβ de um braço, preferindo as linhas por semente à linha-média."""
    import re
    pat = re.compile(rf"^{re.escape(arm)}_seed\d+$")
    per = [float(r["fb"]) for r in rows
           if r["decoder"] == decoder and pat.match(r["method"])]
    return per or [float(r["fb"]) for r in rows
                   if r["decoder"] == decoder and r["method"] == arm]


# ── Experimento A ─────────────────────────────────────────────────────────────

def tabela_exp_a(base, decoder, L):
    files = sorted(glob.glob(os.path.join(base, "*", "*_encoder_al.csv")))
    if not files:
        L.append(f"_Sem dados em `{base}`._\n")
        return

    L.append("## Experimento A — encoder FLIM retreinado\n")
    L.append(f"Decoder de referência: `{decoder}` (o do paper). "
             "Δ = AL − aleatório, média sobre splits e budgets.\n")
    L.append("| nível | método | AL | aleatório | Δ | p | colapsos |")
    L.append("|---|---|---|---|---|---|---|")

    dados = {}
    for f in files:
        m = os.path.basename(os.path.dirname(f))
        if m not in NIVEL:
            continue
        dados[m] = list(csv.DictReader(open(f)))

    for nivel in ("imagem", "região"):
        for m, rows in sorted(dados.items()):
            if NIVEL[m] != nivel:
                continue
            al = arm_vals(rows, f"al_{m}", decoder)
            rd = arm_vals(rows, "random", decoder)
            if not al or not rd:
                continue
            n = min(len(al), len(rd))
            deltas = [a - b for a, b in zip(al[:n], rd[:n])]
            col = sum(1 for r in rows
                      if r["decoder"] == decoder and r["method"] == f"al_{m}"
                      and abs(float(r["fb"]) - 0.4788) < 0.005)
            tot = len([r for r in rows if r["decoder"] == decoder
                       and r["method"] == f"al_{m}"])
            flag = " ⚠" if col else ""
            L.append(f"| {nivel} | {m} | {st.mean(al):.3f} | {st.mean(rd):.3f} | "
                     f"{st.mean(al) - st.mean(rd):+.3f} | {fmt_p(_p(deltas))} | "
                     f"{col}/{tot}{flag} |")
    L.append("")
    L.append("⚠ = alguma execução do braço AL colapsou (predição toda-fundo, "
             "Fβ=DICE=IoU≈0.479). O 'ganho' dessas linhas não é aprendizado.\n")

    # decomposição
    reg = {m: r for m, r in dados.items()
           if NIVEL[m] == "região" and any(x["method"] == "random_region" for x in r)}
    if reg:
        L.append("### Decomposição do ganho dos métodos de região\n")
        L.append("`seleção` isola quais imagens foram escolhidas; `geometria` "
                 "isola como os seeds foram desenhados. Os dois braços "
                 "aleatórios usam a mesma semente, logo sorteiam as mesmas "
                 "imagens — só muda a anotação.\n")
        L.append("| método | decoder | AL | random_região | random | "
                 "seleção | geometria |")
        L.append("|---|---|---|---|---|---|---|")
        decs = ["labeled_marker", "decoder_2", "decoder_3", "decoder_attention",
                "hybrid_decoder", "vanilla_adaptive_decoder",
                "vanilla_adaptive_decoder_wt"]
        somas = defaultdict(list)
        for m, rows in reg.items():
            for d in decs:
                al = arm_vals(rows, f"al_{m}", d)
                rr = arm_vals(rows, "random_region", d)
                rp = arm_vals(rows, "random", d)
                if not (al and rr and rp):
                    continue
                s, g = st.mean(al) - st.mean(rr), st.mean(rr) - st.mean(rp)
                somas["sel"].append(s)
                somas["geo"].append(g)
                L.append(f"| {m} | {d} | {st.mean(al):.3f} | {st.mean(rr):.3f} | "
                         f"{st.mean(rp):.3f} | {s:+.3f} | {g:+.3f} |")
        if somas["sel"]:
            L.append(f"| **média** | | | | | **{st.mean(somas['sel']):+.3f}** | "
                     f"**{st.mean(somas['geo']):+.3f}** |")
            r = st.mean(somas["geo"]) / max(abs(st.mean(somas["sel"])), 1e-9)
            L.append("")
            L.append(f"> A geometria dos seeds pesa **{r:.1f}×** a seleção de "
                     "imagens. O ganho dos métodos de região vem sobretudo de "
                     "COMO anotar, não de QUAIS imagens anotar.\n")


# ── Experimento B ─────────────────────────────────────────────────────────────

def tabela_exp_b(base, L):
    data = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*_runs.csv")):
            data[os.path.basename(d)] = list(csv.DictReader(open(f)))
    if not data:
        L.append(f"_Sem dados em `{base}`._\n")
        return

    L.append("## Experimento B — encoder fixo, decoder 1×1 por backprop\n")
    L.append("ΔFβ **pareado**: AL e aleatório do mesmo índice compartilham a "
             "semente de inicialização, então a diferença isola a seleção.\n")
    L.append("| nível | método | K | AL | aleatório | ΔFβ pareado | p | colapsos |")
    L.append("|---|---|---|---|---|---|---|---|")

    for nivel in ("imagem", "região"):
        for m, rows in sorted(data.items()):
            if NIVEL.get(m) != nivel:
                continue
            for k in sorted({int(r["budget"]) for r in rows if r["arm"] == "al"}):
                pares, al, rd, col = [], [], [], 0
                for sp in sorted({r["split"] for r in rows}):
                    for sd in sorted({r["seed"] for r in rows}):
                        a = [r for r in rows if r["arm"] == "al"
                             and int(r["budget"]) == k and r["split"] == sp
                             and r["seed"] == sd]
                        b = [r for r in rows if r["arm"] == "rand"
                             and int(r["budget"]) == k and r["split"] == sp
                             and r["seed"] == sd]
                        if a and b:
                            pares.append(float(a[0]["fb"]) - float(b[0]["fb"]))
                            al.append(float(a[0]["fb"]))
                            rd.append(float(b[0]["fb"]))
                            col += int(a[0].get("collapsed", 0) or 0)
                if not pares:
                    continue
                flag = " ⚠" if col else ""
                L.append(f"| {nivel} | {m} | {k} | {st.mean(al):.3f} | "
                         f"{st.mean(rd):.3f} | {st.mean(pares):+.4f} | "
                         f"{fmt_p(_p(pares))} | {col}/{len(pares)}{flag} |")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_a", default="out/al_corrected_results")
    ap.add_argument("--exp_b", default="out/al_backprop_results")
    ap.add_argument("--decoder", default=DECODER_REF)
    ap.add_argument("--dataset", default="Schistosoma (Parasites)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    L = [f"# AL por imagem vs AL por região — {a.dataset}\n"]
    L.append("Dois níveis de granularidade da anotação ativa. O de imagem "
             "escolhe QUAIS imagens anotar; o de região escolhe, dentro da "
             "imagem, ONDE anotar.\n")
    tabela_exp_b(a.exp_b, L)
    tabela_exp_a(a.exp_a, a.decoder, L)

    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print(f"\n[salvo] {a.out}")


if __name__ == "__main__":
    main()
