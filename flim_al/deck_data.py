#!/usr/bin/env python3
"""
deck_data.py
============
As tabelas do artigo, transcritas do PDF (arXiv:2504.20872), e os leitores dos
nossos CSVs. Separado do construtor de slides para que os números tenham um
lugar único e auditável.

Transcrição conferida contra os recortes em 420 dpi das páginas 8-9 do PDF.
As marcações de cor são as do próprio artigo: verde = melhor, azul = segundo,
vermelho = pior, por coluna e por usuário.
"""
from __future__ import annotations

import csv
import glob
import os
import statistics as st

G, B, R, N = "green", "blue", "red", None

# ── Tabela III: validação Z1\T, coluna Parasites ──────────────────────────────
TABELA_III = [
    ("SAMNet",      "A", ("0.038±0.036", R), ("0.428±0.234", R)),
    ("SAMNet",      "B", ("0.023±0.015", R), ("0.521±0.216", R)),
    ("MSCNet",      "A", ("0.017±0.009", N), ("0.615±0.147", N)),
    ("MSCNet",      "B", ("0.012±0.005", N), ("0.634±0.178", N)),
    ("MEANet",      "A", ("0.023±0.017", N), ("0.538±0.120", N)),
    ("MEANet",      "B", ("0.013±0.005", N), ("0.674±0.094", N)),
    ("U-Net FLIM",  "A", ("0.009±0.002", N), ("0.771±0.028", N)),
    ("U-Net FLIM",  "B", ("0.011±0.004", N), ("0.710±0.087", N)),
    ("FLIM lm",     "A", ("0.005±0.001", G), ("0.860±0.017", G)),
    ("FLIM lm",     "B", ("0.005±0.001", G), ("0.843±0.025", N)),
    ("FLIM bp",     "A", ("0.007±0.000", N), ("0.770±0.030", N)),
    ("FLIM bp",     "B", ("0.008±0.001", N), ("0.737±0.025", N)),
    ("FLIM pb",     "A", ("0.006±0.001", B), ("0.857±0.012", B)),
    ("FLIM pb",     "B", ("0.006±0.001", B), ("0.847±0.015", G)),
    ("FLIM mb",     "A", ("0.006±0.002", N), ("0.843±0.023", N)),
    ("FLIM mb",     "B", ("0.006±0.001", N), ("0.843±0.021", B)),
    ("FLIM at",     "A", ("0.011±0.003", N), ("0.740±0.062", N)),
    ("FLIM at",     "B", ("0.014±0.006", N), ("0.660±0.108", N)),
    ("FLIM ts",     "A", ("0.010±0.000", N), ("0.747±0.015", N)),
    ("FLIM ts",     "B", ("0.013±0.004", N), ("0.687±0.085", N)),
    ("FLIM lt",     "A", ("0.007±0.001", N), ("0.810±0.010", N)),
    ("FLIM lt",     "B", ("0.009±0.004", N), ("0.760±0.089", N)),
]

# ── Tabela IV: teste Z2, coluna Parasites ─────────────────────────────────────
TABELA_IV = [
    ("SAMNet",      "A", ("0.038±0.036", R), ("0.422±0.241", R)),
    ("SAMNet",      "B", ("0.024±0.014", R), ("0.508±0.194", R)),
    ("MSCNet",      "A", ("0.016±0.009", N), ("0.604±0.125", N)),
    ("MSCNet",      "B", ("0.012±0.005", N), ("0.645±0.163", N)),
    ("MEANet",      "A", ("0.023±0.018", N), ("0.538±0.137", N)),
    ("MEANet",      "B", ("0.012±0.004", N), ("0.681±0.085", N)),
    ("U-Net FLIM",  "A", ("0.008±0.001", N), ("0.777±0.025", N)),
    ("U-Net FLIM",  "B", ("0.011±0.004", N), ("0.706±0.063", N)),
    ("FLIM lm",     "A", ("0.005±0.000", G), ("0.857±0.015", B)),
    ("FLIM lm",     "B", ("0.005±0.000", G), ("0.843±0.015", N)),
    ("FLIM bp",     "A", ("0.007±0.001", N), ("0.757±0.012", N)),
    ("FLIM bp",     "B", ("0.008±0.001", N), ("0.733±0.025", N)),
    ("FLIM pb",     "A", ("0.006±0.000", B), ("0.857±0.012", G)),
    ("FLIM pb",     "B", ("0.006±0.001", B), ("0.850±0.010", G)),
    ("FLIM mb",     "A", ("0.006±0.001", N), ("0.847±0.012", N)),
    ("FLIM mb",     "B", ("0.006±0.001", N), ("0.850±0.010", B)),
    ("FLIM at",     "A", ("0.011±0.002", N), ("0.733±0.042", N)),
    ("FLIM at",     "B", ("0.015±0.006", N), ("0.647±0.101", N)),
    ("FLIM ts",     "A", ("0.010±0.001", N), ("0.747±0.015", N)),
    ("FLIM ts",     "B", ("0.013±0.003", N), ("0.647±0.057", N)),
    ("FLIM lt",     "A", ("0.007±0.001", N), ("0.803±0.021", N)),
    ("FLIM lt",     "B", ("0.009±0.005", N), ("0.753±0.086", N)),
]

# nome no código → nome no artigo
COD2PAPER = {
    "labeled_marker":              "FLIM lm",
    "backprop_decoder":            "FLIM bp",
    "decoder_2":                   "FLIM pb",
    "decoder_3":                   "FLIM mb",
    "decoder_attention":           "FLIM at",
    "vanilla_adaptive_decoder":    "FLIM ts",
    "hybrid_decoder":              "FLIM lt",
    "vanilla_adaptive_decoder_wt": "FLIM ts*",
}
# ordem de apresentação nas tabelas novas, pelos nomes do artigo
ORDEM_PAPER = ["FLIM_lm", "FLIM_pb", "FLIM_mb", "FLIM_lt", "FLIM_ts",
               "FLIM_at", "FLIM_ts*"]
ORDEM = ["labeled_marker", "decoder_2", "decoder_3", "hybrid_decoder",
         "backprop_decoder", "vanilla_adaptive_decoder", "decoder_attention",
         "vanilla_adaptive_decoder_wt"]


def _sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def reproducao(base="out/metrics/schisto") -> dict:
    """
    A reprodução do pipeline original: Fβ/MAE no teste Z2, média ± desvio
    sobre os três splits, por usuário e decoder. Otsu + filtro de área, sem
    Dynamic Trees.
    """
    acc: dict[tuple[str, str], list] = {}
    pat = os.path.join(base, "user_*", "test", "split*", "*", "layer_*",
                       "filtered_saliencies", "1000-9000", "metrics.csv")
    for f in glob.glob(pat):
        p = f.replace("\\", "/").split("/")
        user, dec = p[-8], p[-5]
        linhas = open(f).read().strip().splitlines()
        if len(linhas) < 2:
            continue
        v = [x.strip() for x in linhas[1].split("|")]
        acc.setdefault((user, dec), []).append((float(v[2]), float(v[5])))
    out = {}
    for (user, dec), vals in acc.items():
        mae = [x[0] for x in vals]
        fb = [x[1] for x in vals]
        out[(user, dec)] = {
            "mae": st.mean(mae), "mae_sd": _sd(mae),
            "fb": st.mean(fb), "fb_sd": _sd(fb), "n": len(vals),
        }
    return out


def por_modelo(pat="out/final_comparison/per_model_s*.csv") -> dict:
    """
    O efeito do AL por modelo: T do artigo contra T escolhido pelo critério,
    mesma tubulação, teste Z2. Média ± desvio sobre os splits.
    """
    acc: dict[tuple[str, str], list] = {}
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            acc.setdefault((r["arm"], r["decoder"]), []).append(
                (float(r["fb"]), float(r["mae"]), float(r["iou"]),
                 int(r["n_train"])))
    out = {}
    for k, v in acc.items():
        fb = [x[0] for x in v]
        out[k] = {"fb": st.mean(fb), "fb_sd": _sd(fb),
                  "mae": st.mean([x[1] for x in v]),
                  "iou": st.mean([x[2] for x in v]),
                  "n_train": st.mean([x[3] for x in v]),
                  "n": len(v)}
    return out


def tabela_md(path: str) -> list[list[str]]:
    """Extrai a primeira tabela markdown de um arquivo, como lista de linhas."""
    linhas = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if l.startswith("|"):
            cels = [c.strip() for c in l.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cels):
                continue
            linhas.append(cels)
        elif linhas:
            break
    return linhas


def p_pareado(pat="out/final_comparison/per_model_s*.csv") -> dict:
    """
    p do teste t pareado por split, AL contra o T do artigo, por decoder.

    Pareado porque os dois braços compartilham split, validação e conjunto de
    teste — só muda quais imagens entram em T. Com 3 splits o teste é fraco,
    e é por isso que ele está na tabela: para que a leitura não confunda uma
    diferença de 0.03 com um efeito.
    """
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return {}
    por: dict = {}
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            por.setdefault(r["decoder"], {}).setdefault(r["arm"], {})[
                r["split"]] = float(r["fb"])
    out = {}
    for dec, arms in por.items():
        if not {"paper", "al"} <= set(arms):
            continue
        sp = sorted(set(arms["paper"]) & set(arms["al"]))
        if len(sp) < 3:
            continue
        out[dec] = float(ttest_rel([arms["al"][s] for s in sp],
                                   [arms["paper"][s] for s in sp]).pvalue)
    return out


# ── leitores da rodada final (sementes agregadas, dois usuários, TOST) ────────

def resumo_al(delta: float = 0.02) -> dict:
    """
    Por decoder: Fβ dos dois braços, Δ pareado por (usuário, split), IC95%,
    p da diferença e p da equivalência. Delega a estatística ao
    analyze_final para que a tabela do slide e a do documento não possam
    divergir.
    """
    from flim_al import analyze_final as A
    import statistics as _st
    from collections import defaultdict as _dd

    dados = A.ler_schisto()
    if not dados:
        return {}
    ds, ap, aa = _dd(list), _dd(list), _dd(list)
    for k, v in dados.items():
        if v["paper"] is None or not v["al"]:
            continue
        ds[k[0]].append(_st.mean(v["al"]) - v["paper"])
        ap[k[0]].append(v["paper"])
        aa[k[0]].append(_st.mean(v["al"]))
    out = {}
    for dec, d in ds.items():
        lo, hi = A.ic95(d)
        p = (float(A._sps.ttest_1samp(d, 0.0).pvalue)
             if A._sps and len(d) > 1 else float("nan"))
        out[dec] = {"paper": _st.mean(ap[dec]), "al": _st.mean(aa[dec]),
                    "d": _st.mean(d), "lo": lo, "hi": hi, "p": p,
                    "tost": A.tost(d, delta), "n": len(d)}
    return out


def resumo_brats() -> dict:
    """Fβ reproduzido por decoder no BraTS, com o valor do artigo ao lado."""
    from flim_al import analyze_final as A
    import statistics as _st

    dados = A.ler_brats()
    out = {}
    for dec in A.ORDEM:
        vals = [v["paper"] for k, v in dados.items()
                if k[0] == dec and v["paper"] is not None]
        if vals:
            out[dec] = {"fb": _st.mean(vals), "n": len(vals),
                        "artigo": A.PAPER_BRATS_A.get(dec)}
    return out
