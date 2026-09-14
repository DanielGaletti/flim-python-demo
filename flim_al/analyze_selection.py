#!/usr/bin/env python3
"""
analyze_selection.py
====================
Agrega os CSVs do E1 (seleção iterativa) e emite tabelas em Markdown.

Leitura central: `oracle` é o TETO (usa ground truth de todo o pool, como o
Algoritmo 1 do paper) e `random` é o PISO. A contribuição se mede por quanto
do intervalo piso→teto cada critério SEM ground truth recupera:

    recuperação = (critério − random) / (oracle − random)

Uso:
    cd flim_ad && python ../flim_al/analyze_selection.py --pool real
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st
from collections import defaultdict

ORDER = ["oracle", "badge", "coreset", "entropy", "random"]


def load(save_dir: str, pool: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(os.path.join(save_dir, f"selection_*_{pool}.csv"))):
        crit = os.path.basename(f)[len("selection_"):-len(f"_{pool}.csv")]
        out[crit] = list(csv.DictReader(open(f)))
    return out


def best_by_size(rows: list[dict]) -> dict[int, list[float]]:
    """
    Melhor Fβ alcançado com |T| = n, por execução (split, seed).
    Usa só rodadas que melhoraram — as demais são tentativas revertidas pelo
    backtracking, e contá-las puxaria a curva para baixo artificialmente.
    """
    per_run: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if not int(r.get("improved", 1)):
            continue
        k = (r["split"], r["seed"])
        n = int(r["n_images"])
        fb = float(r["fb"])
        per_run[k][n] = max(per_run[k].get(n, -1.0), fb)
    by_n: dict[int, list[float]] = defaultdict(list)
    for run in per_run.values():
        running = -1.0
        for n in sorted(run):
            running = max(running, run[n])
            by_n[n].append(running)
    return by_n


def wasted(rows: list[dict]) -> tuple[float, float]:
    """(média de anotações desperdiçadas, média de rodadas) por execução."""
    per_run: dict[tuple, dict] = defaultdict(lambda: {"rej": set(), "rounds": 0})
    for r in rows:
        k = (r["split"], r["seed"])
        per_run[k]["rounds"] += 1
        rj = r.get("rejected", "") or ""
        if rj:
            per_run[k]["rej"].update(rj.split("|"))
    if not per_run:
        return float("nan"), float("nan")
    return (st.mean(len(v["rej"]) for v in per_run.values()),
            st.mean(v["rounds"] for v in per_run.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="out/paper_selection")
    ap.add_argument("--pool", default="real", choices=["real", "full"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    data = load(a.save_dir, a.pool)
    if not data:
        print(f"Nenhum CSV em {a.save_dir} para pool={a.pool}.")
        return

    crits = [c for c in ORDER if c in data] + [c for c in data if c not in ORDER]
    L: list[str] = [f"# E1 — Seleção iterativa (pool: {a.pool})\n"]
    L.append("Algoritmo 1 de arXiv:2504.20872 com o critério do passo 7 trocado.\n")

    # ── curva Fβ × |T| ───────────────────────────────────────────────────────
    sizes = sorted({int(r["n_images"]) for rows in data.values() for r in rows})
    L.append("## Curva Fβ × |T|\n")
    L.append("Melhor Fβ acumulado com |T| imagens, média ± desvio sobre "
             "(split × semente).\n")
    L.append("| critério | usa GT? | " + " | ".join(f"\\|T\\|={n}" for n in sizes) + " |")
    L.append("|---" * (len(sizes) + 2) + "|")
    curves: dict[str, dict[int, float]] = {}
    for c in crits:
        by_n = best_by_size(data[c])
        curves[c] = {n: st.mean(v) for n, v in by_n.items() if v}
        cells = []
        for n in sizes:
            v = by_n.get(n, [])
            if not v:
                cells.append("—")
            elif len(v) > 1:
                cells.append(f"{st.mean(v):.4f}±{st.stdev(v):.3f}")
            else:
                cells.append(f"{st.mean(v):.4f}")
        gt = "**sim**" if c == "oracle" else "não"
        L.append(f"| {c} | {gt} | " + " | ".join(cells) + " |")
    L.append("")

    # ── recuperação do intervalo piso→teto ───────────────────────────────────
    MIN_GAP = 0.02   # abaixo disso oráculo e random são indistinguíveis
    if "oracle" in curves and "random" in curves:
        L.append("## Diferença para o aleatório, e quanto do oráculo se recupera\n")
        L.append("A coluna principal é `critério − random`, sempre interpretável. "
                 "A recuperação `(critério − random) / (oracle − random)` só é "
                 "reportada quando o oráculo supera o aleatório por pelo menos "
                 f"{MIN_GAP:.2f} — abaixo disso o denominador é ruído e a razão "
                 "explode para valores sem sentido.\n")
        gaps = {n: (curves["oracle"].get(n, float('nan'))
                    - curves["random"].get(n, float('nan'))) for n in sizes}
        L.append("| |T| | oracle − random | " +
                 " | ".join(f"{c} − random" for c in crits
                            if c not in ("oracle", "random")) + " | recuperação |")
        L.append("|---" * (len(crits) + 1) + "|")
        for n in sizes:
            gap = gaps[n]
            cells = []
            for c in crits:
                if c in ("oracle", "random"):
                    continue
                v, r_ = curves[c].get(n), curves["random"].get(n)
                cells.append(f"{v - r_:+.4f}" if None not in (v, r_) else "—")
            if gap == gap and gap >= MIN_GAP:
                rec = []
                for c in crits:
                    if c in ("oracle", "random"):
                        continue
                    v, r_ = curves[c].get(n), curves["random"].get(n)
                    rec.append(f"{c} {100 * (v - r_) / gap:+.0f}%"
                               if None not in (v, r_) else "—")
                rec_txt = ", ".join(rec)
            else:
                rec_txt = "_oráculo ≈ random_"
            gap_txt = f"{gap:+.4f}" if gap == gap else "—"
            L.append(f"| {n} | {gap_txt} | " + " | ".join(cells) + f" | {rec_txt} |")
        L.append("")
        if all((gaps[n] != gaps[n]) or gaps[n] < MIN_GAP for n in sizes):
            L.append("> **Leitura:** o oráculo do paper não supera a seleção "
                     "aleatória neste pool. Como as imagens do pool são "
                     "justamente as que os usuários escolheram anotar, todas já "
                     "são bons candidatos — não há intervalo piso→teto a "
                     "recuperar. Para medir o valor do critério é preciso um "
                     "pool heterogêneo (`--pool full`).\n")

    # ── custo de anotação desperdiçada ───────────────────────────────────────
    L.append("## Anotações desperdiçadas\n")
    L.append("Imagens que o critério mandou anotar, foram anotadas, pioraram o "
             "modelo e acabaram descartadas pelo backtracking. Cada uma é "
             "esforço real do especialista jogado fora.\n")
    L.append("| critério | descartadas por execução | rodadas | taxa |")
    L.append("|---|---|---|---|")
    for c in crits:
        w, rounds = wasted(data[c])
        rate = f"{100 * w / rounds:.0f}%" if rounds and rounds == rounds else "—"
        L.append(f"| {c} | {w:.1f} | {rounds:.1f} | {rate} |")
    L.append("")

    # ── imagens escolhidas ───────────────────────────────────────────────────
    L.append("## Imagens mais escolhidas\n")
    L.append("| critério | top-5 (frequência entre execuções) |")
    L.append("|---|---|")
    for c in crits:
        cnt: dict[str, int] = defaultdict(int)
        seen = set()
        for r in data[c]:
            k = (r["split"], r["seed"], r["round"])
            if k in seen:
                continue
            seen.add(k)
            for img in (r.get("selected", "") or "").split("|"):
                if img:
                    cnt[img] += 1
        top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
        L.append(f"| {c} | " + ", ".join(f"`{i}` ({n})" for i, n in top) + " |")
    L.append("")

    text = "\n".join(L)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n[salvo] {a.out}")


if __name__ == "__main__":
    main()
