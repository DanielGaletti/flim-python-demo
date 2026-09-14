#!/usr/bin/env python3
"""
analyze_dissertation.py
=======================
Agrega os CSVs brutos dos dois experimentos e emite tabelas em Markdown
prontas para a dissertação, com média ± desvio e teste PAREADO.

Por que pareado
---------------
No Exp B, o braço AL e o braço Random da semente s compartilham a mesma
inicialização Xavier do decoder. Então

    Δ_s = Fβ(AL, s) − Fβ(Random, s)

cancela o ruído de inicialização e isola o efeito da seleção de imagens.
Medir a média de Δ_s sobre (split × semente) e testar se ela difere de zero
é bem mais sensível do que comparar duas médias independentes — que foi o que
a versão anterior fazia, com um único run de AL contra a média de 3 randoms.

Uso:
    cd flim_ad
    python3 ../flim_al/analyze_dissertation.py
    python3 ../flim_al/analyze_dissertation.py --out ../RESULTADOS_DISSERTACAO.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict


def arm_values(rows: list[dict], arm: str, decoder: str) -> list[float]:
    """
    Fβ de um braço aleatório, preferindo SEMPRE as linhas por semente
    (`random_seed0`, `random_region_seed1`, …) em vez da linha-média.

    Motivo: ao retomar um experimento com mais sementes que a execução
    original, a linha-média fica congelada no número antigo de sementes
    enquanto novas linhas por semente são acrescentadas. Agregar do dado
    bruto elimina essa inconsistência.
    """
    pat = re.compile(rf"^{re.escape(arm)}_seed\d+$")
    per_seed = [float(r["fb"]) for r in rows
                if r["decoder"] == decoder and pat.match(r["method"])]
    if per_seed:
        return per_seed
    return [float(r["fb"]) for r in rows
            if r["decoder"] == decoder and r["method"] == arm]

try:
    from scipy import stats as _sps
except ImportError:
    _sps = None


# ── Estatística ───────────────────────────────────────────────────────────────

def paired_stats(deltas: list[float]) -> dict:
    """
    Estatísticas de uma lista de diferenças pareadas.
    Devolve média, desvio, IC95%, p do teste t pareado e p de Wilcoxon.
    """
    n = len(deltas)
    out = {"n": n, "mean": float("nan"), "sd": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"),
           "p_t": float("nan"), "p_w": float("nan"), "n_pos": 0}
    if n == 0:
        return out

    mean = st.mean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0

    out.update(mean=mean, sd=sd, n_pos=sum(1 for d in deltas if d > 0))

    if n > 1 and se > 0:
        if _sps is not None:
            t_crit = _sps.t.ppf(0.975, n - 1)
            out["p_t"] = float(_sps.ttest_1samp(deltas, 0.0).pvalue)
            try:
                out["p_w"] = float(_sps.wilcoxon(deltas).pvalue)
            except ValueError:
                out["p_w"] = float("nan")
        else:
            t_crit = 1.96
        out["ci_lo"] = mean - t_crit * se
        out["ci_hi"] = mean + t_crit * se
    return out


def fmt_p(p: float) -> str:
    if p != p:
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def sig_marker(s: dict, noise: float) -> str:
    """
    Marca o resultado só se ele passar nos DOIS critérios:
      - IC95% da diferença pareada não cruza zero
      - efeito maior que o piso de ruído de inicialização medido
    """
    if s["n"] < 2 or s["ci_lo"] != s["ci_lo"]:
        return ""
    crosses_zero = s["ci_lo"] <= 0 <= s["ci_hi"]
    if crosses_zero:
        return ""
    if noise == noise and abs(s["mean"]) < noise:
        return "~"          # consistente, mas dentro do ruído
    return "**"


# ── Experimento B ─────────────────────────────────────────────────────────────

def load_runs(base: str) -> dict:
    data = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*_runs.csv")):
            data[os.path.basename(d)] = list(csv.DictReader(open(f)))
    return data


def analyze_exp_b(base: str, lines: list[str]) -> None:
    data = load_runs(base)
    if not data:
        lines.append("_Nenhum CSV de execução encontrado em "
                     f"`{base}` (rode a Fase 2 primeiro)._\n")
        return

    lines.append("## Experimento B — Backprop AL (encoder FIXO)\n")

    # ── Piso de ruído: braço "full", mesmo conjunto, só muda a inicialização ──
    noise_by_method = {}
    lines.append("### B.1 Piso de ruído e teto do decoder\n")
    lines.append("Braço `full`: todas as N imagens do pool, variando apenas a "
                 "semente de inicialização. O desvio é o ruído irredutível; "
                 "nenhum ΔFβ menor que ele deve ser interpretado.\n")
    lines.append("| método | split | N | Fβ (teto) | desvio (ruído) |")
    lines.append("|---|---|---|---|---|")
    for m, rows in data.items():
        sds = []
        for split in sorted({r["split"] for r in rows}):
            f = [float(r["fb"]) for r in rows
                 if r["arm"] == "full" and r["split"] == split]
            if len(f) < 2:
                continue
            sd = st.stdev(f)
            sds.append(sd)
            n_imgs = [r["n_train_imgs"] for r in rows
                      if r["arm"] == "full" and r["split"] == split][0]
            lines.append(f"| {m} | {split} | {n_imgs} | {st.mean(f):.4f} | ±{sd:.4f} |")
        noise_by_method[m] = st.mean(sds) if sds else float("nan")
    lines.append("")

    # ── Δ pareado por budget ─────────────────────────────────────────────────
    lines.append("### B.2 ΔFβ pareado (AL − Random), por budget\n")
    lines.append("Cada par usa a MESMA inicialização; n = splits × sementes.\n")
    lines.append("| método | K | AL Fβ | Random Fβ | ΔFβ pareado | IC95% | p (t) | "
                 "p (Wilcoxon) | Δ>0 | colapsos |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for m, rows in data.items():
        noise = noise_by_method.get(m, float("nan"))
        budgets = sorted({int(r["budget"]) for r in rows if r["arm"] in ("al", "rand")})
        for b in budgets:
            pairs, al_v, rd_v, ncol = [], [], [], 0
            for split in sorted({r["split"] for r in rows}):
                for seed in sorted({r["seed"] for r in rows}):
                    a = [r for r in rows if r["arm"] == "al" and int(r["budget"]) == b
                         and r["split"] == split and r["seed"] == seed]
                    d = [r for r in rows if r["arm"] == "rand" and int(r["budget"]) == b
                         and r["split"] == split and r["seed"] == seed]
                    if not a or not d:
                        continue
                    fa, fd = float(a[0]["fb"]), float(d[0]["fb"])
                    pairs.append(fa - fd)
                    al_v.append(fa)
                    rd_v.append(fd)
                    ncol += int(a[0].get("collapsed", 0) or 0)
            if not pairs:
                continue
            s = paired_stats(pairs)
            mark = sig_marker(s, noise)
            col_txt = f"{ncol}/{len(pairs)}" + (" ⚠" if ncol else "")
            lines.append(
                f"| {m} | {b} | {st.mean(al_v):.4f} | {st.mean(rd_v):.4f} | "
                f"{mark}{s['mean']:+.4f}{mark} | "
                f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] | {fmt_p(s['p_t'])} | "
                f"{fmt_p(s['p_w'])} | {s['n_pos']}/{s['n']} | {col_txt} |"
            )
    lines.append("")
    lines.append("`**` = IC95% não cruza zero **e** efeito acima do piso de ruído. "
                 "`~` = consistente, porém dentro do ruído. "
                 "⚠ = alguma execução do braço AL colapsou (predição toda-fundo): "
                 "nesses casos o 'ganho' não representa aprendizado.\n")

    # ── Agregado por método ──────────────────────────────────────────────────
    lines.append("### B.3 Agregado por método (todos os budgets de anotação)\n")
    lines.append("| método | ΔFβ pareado | IC95% | p (t) | Δ>0 | veredicto |")
    lines.append("|---|---|---|---|---|---|")
    for m, rows in data.items():
        noise = noise_by_method.get(m, float("nan"))
        pairs = []
        for split in sorted({r["split"] for r in rows}):
            for seed in sorted({r["seed"] for r in rows}):
                for b in sorted({int(r["budget"]) for r in rows if r["arm"] == "al"}):
                    a = [r for r in rows if r["arm"] == "al" and int(r["budget"]) == b
                         and r["split"] == split and r["seed"] == seed
                         and not int(r.get("collapsed", 0) or 0)]
                    d = [r for r in rows if r["arm"] == "rand" and int(r["budget"]) == b
                         and r["split"] == split and r["seed"] == seed]
                    if a and d:
                        pairs.append(float(a[0]["fb"]) - float(d[0]["fb"]))
        if not pairs:
            continue
        s = paired_stats(pairs)
        mark = sig_marker(s, noise)
        if mark == "**":
            verdict = "supera o random"
        elif mark == "~":
            verdict = "consistente, mas dentro do ruído"
        else:
            verdict = "indistinguível do random"
        lines.append(
            f"| {m} | {mark}{s['mean']:+.4f}{mark} | "
            f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] | {fmt_p(s['p_t'])} | "
            f"{s['n_pos']}/{s['n']} | {verdict} |"
        )
    lines.append("\n_Execuções com colapso do braço AL são excluídas do agregado._\n")

    # ── B.4 Eficiência de anotação ───────────────────────────────────────────
    lines.append("### B.4 Eficiência de anotação: AL vs pool inteiro\n")
    lines.append("Compara o maior budget de AL contra treinar com TODAS as imagens "
                 "do pool. Diferença dentro do piso de ruído (B.1) significa "
                 "desempenho equivalente com uma fração da anotação.\n")
    lines.append("| método | K | split | N do pool | AL Fβ | pool Fβ | AL − pool | "
                 "veredicto |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m, rows in data.items():
        noise = noise_by_method.get(m, float("nan"))
        al_budgets = sorted({int(r["budget"]) for r in rows if r["arm"] == "al"})
        if not al_budgets:
            continue
        kmax = al_budgets[-1]
        tot_al, tot_fu, n_pool = [], [], "?"
        for split in sorted({r["split"] for r in rows}):
            al = [float(r["fb"]) for r in rows if r["arm"] == "al"
                  and int(r["budget"]) == kmax and r["split"] == split]
            fu = [float(r["fb"]) for r in rows if r["arm"] == "full"
                  and r["split"] == split]
            if not al or not fu:
                continue
            n_pool = [r["n_train_imgs"] for r in rows if r["arm"] == "full"
                      and r["split"] == split][0]
            tot_al += al
            tot_fu += fu
            d = st.mean(al) - st.mean(fu)
            lines.append(f"| {m} | {kmax} | {split} | {n_pool} | {st.mean(al):.4f} | "
                         f"{st.mean(fu):.4f} | {d:+.4f} | |")
        if tot_al:
            d = st.mean(tot_al) - st.mean(tot_fu)
            if noise == noise and abs(d) <= noise:
                verdict = f"**equivalente** (|Δ| ≤ ruído {noise:.3f})"
            elif d > 0:
                verdict = "AL acima do pool"
            else:
                verdict = "AL abaixo do pool"
            lines.append(f"| **{m}** | **{kmax}** | média | | **{st.mean(tot_al):.4f}** | "
                         f"**{st.mean(tot_fu):.4f}** | **{d:+.4f}** | {verdict} |")
    lines.append("")


# ── Experimento A ─────────────────────────────────────────────────────────────

def analyze_exp_a(base: str, lines: list[str]) -> None:
    files = sorted(glob.glob(os.path.join(base, "*", "*_encoder_al.csv")))
    if not files:
        lines.append(f"_Nenhum CSV encontrado em `{base}`._\n")
        return

    lines.append("## Experimento A — Encoder AL (encoder retreinado)\n")

    DECS = ["labeled_marker", "decoder_2", "decoder_3", "decoder_attention",
            "hybrid_decoder", "vanilla_adaptive_decoder",
            "vanilla_adaptive_decoder_wt"]

    per_method = {}
    for f in files:
        m = os.path.basename(os.path.dirname(f))
        per_method[m] = list(csv.DictReader(open(f)))

    # Desbalanceamento de imagens entre braços
    lines.append("### A.1 Imagens efetivamente usadas por braço\n")
    lines.append("Métodos `region_*` descartam imagens sem região de foreground. "
                 "Se o braço AL treinar com menos imagens que o Random, parte do "
                 "Δ vem de ter adicionado menos markers sintéticos, não da seleção.\n")
    lines.append("| método | split | K | AL | random | random_region |")
    lines.append("|---|---|---|---|---|---|")
    any_imbalance = False
    for m, rows in per_method.items():
        for split in sorted({r["split"] for r in rows}):
            for b in sorted({int(r["budget"]) for r in rows}, key=int):
                def _n(meth_prefix):
                    v = [r.get("n_train_imgs", "") for r in rows
                         if r["split"] == split and int(r["budget"]) == b
                         and r["method"].startswith(meth_prefix)
                         and r.get("n_train_imgs", "")]
                    return v[0] if v else "—"
                al_n, rp_n, rr_n = _n("al_"), _n("random_seed"), _n("random_region_seed")
                if al_n == "—" and rp_n == "—":
                    continue
                if al_n != "—" and rp_n != "—" and al_n != rp_n:
                    any_imbalance = True
                    al_n = f"**{al_n}**"
                lines.append(f"| {m} | {split} | {b} | {al_n} | {rp_n} | {rr_n} |")
    if any_imbalance:
        lines.append("\n⚠ Valores em negrito: braço AL treinou com um número de "
                     "imagens diferente do Random no mesmo budget.\n")
    else:
        lines.append("\nBraços balanceados em todas as configurações.\n")

    # Decomposição do efeito
    lines.append("### A.2 Decomposição: seleção vs geometria dos seeds\n")
    lines.append("Só faz sentido para métodos `region_*`, que possuem o braço "
                 "de controle `random_region` (imagens aleatórias + seeds de região).\n")
    lines.append("| método | decoder | AL | random_region | random | "
                 "seleção<br>(AL−rand_reg) | geometria<br>(rand_reg−rand) |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, rows in per_method.items():
        has_ctrl = any(r["method"] == "random_region" for r in rows)
        if not has_ctrl:
            continue
        for dec in DECS:
            _al = [float(r["fb"]) for r in rows
                   if r["decoder"] == dec and r["method"] == f"al_{m}"]
            _rr = arm_values(rows, "random_region", dec)
            _rp = arm_values(rows, "random", dec)
            if not _al or not _rr or not _rp:
                continue
            al, rr, rp = st.mean(_al), st.mean(_rr), st.mean(_rp)
            lines.append(
                f"| {m} | {dec} | {al:.3f} | {rr:.3f} | {rp:.3f} | "
                f"{al - rr:+.3f} | {rr - rp:+.3f} |"
            )
    lines.append("")

    # Tabela principal com desvio entre sementes
    lines.append("### A.3 Fβ por decoder (média entre splits e budgets)\n")
    lines.append("| método | decoder | baseline | AL | random (±dp) | Δ(AL−rand) | colapsos AL |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, rows in per_method.items():
        for dec in DECS:
            base = [float(r["fb"]) for r in rows
                    if r["decoder"] == dec and r["method"] == "original_3imgs"]
            al = [float(r["fb"]) for r in rows
                  if r["decoder"] == dec and r["method"] == f"al_{m}"]
            rp = arm_values(rows, "random", dec)
            if not al or not rp:
                continue
            ncol = sum(
                1 for r in rows
                if r["decoder"] == dec and r["method"] == f"al_{m}"
                and abs(float(r["fb"]) - 0.4788) < 0.005
                and abs(float(r["fb"]) - float(r["dice"])) < 0.01
            )
            b_txt = f"{st.mean(base):.3f}" if base else "—"
            sd_rp = st.stdev(rp) if len(rp) > 1 else 0.0
            lines.append(
                f"| {m} | {dec} | {b_txt} | {st.mean(al):.3f} | "
                f"{st.mean(rp):.3f}±{sd_rp:.3f} | "
                f"{st.mean(al) - st.mean(rp):+.3f} | {ncol}/{len(al)} |"
            )
    lines.append("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_a", default="out/al_corrected_results")
    p.add_argument("--exp_b", default="out/al_backprop_results")
    p.add_argument("--out", default="", help="Escreve Markdown neste caminho")
    a = p.parse_args()

    lines: list[str] = ["# Resultados — FLIM + Active Learning\n"]
    if _sps is None:
        lines.append("_scipy indisponível: valores-p omitidos, IC95% usa "
                     "aproximação normal._\n")

    analyze_exp_b(a.exp_b, lines)
    analyze_exp_a(a.exp_a, lines)

    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n[salvo] {a.out}")


if __name__ == "__main__":
    main()
