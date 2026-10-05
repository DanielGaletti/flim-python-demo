#!/usr/bin/env python3
"""
onde_o_criterio_aponta.py — que tipo de região cada critério escolhe?

A pergunta
    As campanhas anteriores mediram SE os critérios de Active Learning ajudam.
    Esta mede PARA ONDE eles apontam, que é outra coisa e explica a primeira.

    A campanha `artigo_vs_regiao` mediu que anotar na borda do objeto vale de
    +0,126 a +0,166 de Fβ sobre anotar em qualquer outro lugar dentro dele, em
    6 de 6 células. Então a região valiosa é a que ATRAVESSA a borda: ela
    contém objeto e fundo ao mesmo tempo, e é ali que o recorte de 3x3 captura
    a transição que o filtro precisa representar.

    A fração de objeto da região diz exatamente isso:

        fração = 0,0    fundo puro, nenhuma transição
        fração = 1,0    interior puro, nenhuma transição
        fração ~ 0,5    atravessa a borda

Por que não precisa treinar
    Escolher a região é pontuar e ordenar. O custo é um forward pass por
    imagem, que já existe. Isto permite medir em 10 sementes e 3 datasets pelo
    preço de uma fração de uma campanha de treino, e a medida é exata: dado o
    encoder inicial, o argmax de cada critério é determinístico.

    A fração de objeto usa o ground truth, mas apenas para DESCREVER a região
    escolhida. Nenhum critério a enxerga. É medição sobre o comportamento do
    critério, não insumo dele.

Uso
    python scripts/onde_o_criterio_aponta.py --ds schisto brats conjunctiva
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import statistics as st
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from flim_al import cl_flim  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

import ganho_marginal as GM  # noqa: E402
import onde_marcar as OM  # noqa: E402

SAIDA = os.path.join(RAIZ, "evidencia", "bruto", "onde_o_criterio_aponta.json")

PARTICAO = {
    "schisto":     dict(n_pool=40),
    "brats":       dict(n_pool=40),
    "conjunctiva": dict(n_pool=20),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="+",
                    default=["schisto", "brats", "conjunctiva"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sementes", nargs="+", type=int,
                    default=list(range(10)))
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--n_segments", type=int, default=150)
    ap.add_argument("--saida", default=SAIDA)
    a = ap.parse_args()

    from flim_al.region_al import score_regions_by_entropy
    from flim_app import server as SV

    tudo = {}
    if os.path.isfile(a.saida):
        with open(a.saida, encoding="utf-8") as fh:
            tudo = json.load(fh)

    for ds in a.ds:
        part = PARTICAO[ds]
        cfg = DS.resolver(ds)
        cfg["_id"] = ds
        cfg["bloco"] = cfg.get("bloco", 2)
        todas = [os.path.splitext(f)[0] for f in DS.imagens(ds)]
        rng0 = np.random.default_rng(a.semente_particao)
        com = [i for i in todas[:600] if GM._tem_objeto(ds, i)]
        emb = rng0.permutation(len(com))
        pool = sorted(com[i] for i in emb[:part["n_pool"]])

        work = tempfile.mkdtemp(prefix="aponta_")
        try:
            import tabela_k_por_modelo as TK
            cfg["orig"] = TK._orig_em_png(cfg, ds, set(pool), work)
            SV.S["ds"] = ds
            SV.S["block"] = cfg["bloco"]
            SV.S["decoder"] = "labeled_marker"

            escolhas = collections.defaultdict(list)
            dist = []
            for semente in a.sementes:
                r = np.random.default_rng(semente * 7717 + 3)
                sel = [str(x) for x in r.permutation(pool)][:a.k]
                base, gts = {}, {}
                for img in sel:
                    lp = DS.caminho_label(ds, img + ".png")
                    if not lp:
                        continue
                    gt = np.array(Image.open(lp).convert("L")) > 127
                    d = generate_realistic_markers(
                        lp, n_fg_dabs=6, n_bg_dabs=14,
                        seed=OM._semente_marker(img, semente))
                    base[img] = (
                        [tuple(int(v) for v in p) for p in d["fg_seeds"]],
                        [tuple(int(v) for v in p) for p in d["bg_seeds"]],
                        gt.shape[0], gt.shape[1])
                    gts[img] = gt
                if not base:
                    continue
                d0 = os.path.join(work, f"m{semente}")
                enc = os.path.join(work, f"e{semente}.pth")
                GM._escrever(d0, base)
                try:
                    cl_flim.treinar(cfg["arch"], d0, cfg["orig"],
                                    cfg["label"], "cpu", enc)
                    SV.S["encoder"] = enc
                    modelo = SV._montar_modelo(enc)
                except Exception as e:                        # noqa: BLE001
                    print(f"  {ds} s{semente}: {e}")
                    continue

                regioes = {}
                for img in base:
                    gt = gts[img]
                    H, W = gt.shape
                    seg = GM._slic(os.path.join(cfg["orig"],
                                                OM._arquivo(cfg, img)),
                                   a.n_segments)
                    prob = SV._mapa_prob(modelo, img)
                    if prob.shape != (H, W):
                        prob = np.array(Image.fromarray(
                            (prob * 255).astype(np.uint8)).resize(
                                (W, H), Image.BILINEAR)) / 255.0
                    ent = score_regions_by_entropy(seg, prob)
                    lc_map = 1 - np.abs(prob - 0.5) * 2
                    pred = prob > 0.5
                    for rid in np.unique(seg):
                        m = seg == rid
                        if m.sum() < 20:
                            continue
                        fp = float(pred[m].mean())
                        regioes[(img, int(rid))] = dict(
                            ent=float(ent[int(rid)]),
                            lc=float(lc_map[m].mean()),
                            prob=float(prob[m].mean()),
                            # fronteira PREVISTA dentro da regiao, sem GT:
                            # 1 quando metade da regiao e objeto previsto.
                            fronteira=float(1.0 - abs(2.0 * fp - 1.0)),
                            frac=float(gt[m].mean()),
                        )
                if not regioes:
                    continue
                dist += [v["frac"] for v in regioes.values()]
                for nome, chave in (("argmax_entropia", "ent"),
                                    ("argmax_lc", "lc"),
                                    ("argmax_prob", "prob"),
                                    ("argmax_fronteira", "fronteira")):
                    c = max(regioes, key=lambda x: regioes[x][chave])
                    escolhas[nome].append(regioes[c]["frac"])

            def resume(v):
                if not v:
                    return None
                return {"n": len(v), "frac_mediana": round(st.median(v), 4),
                        "frac_media": round(st.fmean(v), 4),
                        "atravessa_borda": round(
                            sum(1 for x in v if 0.05 < x < 0.95) / len(v), 3),
                        "fundo_puro": round(
                            sum(1 for x in v if x <= 0.05) / len(v), 3),
                        "interior_puro": round(
                            sum(1 for x in v if x >= 0.95) / len(v), 3)}

            tudo[ds] = {
                "criterios": {k: resume(v) for k, v in escolhas.items()},
                "todas_as_regioes": resume(dist),
                "n_segments": a.n_segments, "k": a.k,
            }
            print(f"=== {ds} ({len(a.sementes)} sementes)")
            print(f"{'criterio':<20} {'n':>3} {'frac mediana':>13} "
                  f"{'atravessa borda':>16} {'fundo puro':>11} "
                  f"{'interior puro':>14}")
            for nome in ("argmax_entropia", "argmax_lc", "argmax_prob",
                         "argmax_fronteira"):
                s = tudo[ds]["criterios"].get(nome)
                if not s:
                    continue
                print(f"{nome:<20} {s['n']:>3} {s['frac_mediana']:>13.3f} "
                      f"{s['atravessa_borda']:>15.0%} "
                      f"{s['fundo_puro']:>10.0%} {s['interior_puro']:>13.0%}")
            t = tudo[ds]["todas_as_regioes"]
            print(f"{'(todas as regioes)':<20} {t['n']:>3} "
                  f"{t['frac_mediana']:>13.3f} {t['atravessa_borda']:>15.0%} "
                  f"{t['fundo_puro']:>10.0%} {t['interior_puro']:>13.0%}")
            print()
        finally:
            shutil.rmtree(work, ignore_errors=True)

    os.makedirs(os.path.dirname(a.saida), exist_ok=True)
    with open(a.saida, "w", encoding="utf-8") as fh:
        json.dump(tudo, fh, indent=2, ensure_ascii=False)
    print(f"gravado em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
