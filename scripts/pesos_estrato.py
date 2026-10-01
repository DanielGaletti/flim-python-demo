#!/usr/bin/env python3
"""
pesos_estrato.py — quantos superpixels tocam o objeto, por semente

O problema que isto resolve
    `ganho_marginal.py` sorteia candidatos ESTRATIFICADOS: 12 que tocam o
    objeto e 10 de fundo puro. A estratificação é necessária — o ovo ocupa ~3%
    da imagem no Schisto, e um sorteio uniforme de 22 superpixels traria em
    média 1,5 que tocam o objeto, medindo sobretudo o ganho de anotar fundo.

    Mas a estratificação usa o ground truth para decidir de qual estrato cada
    candidato vem. Isso é informação privilegiada, e ela entra na comparação:
    o braço "sorteado" recebe de graça a garantia de que 12 dos 22 candidatos
    tocam o objeto, enquanto o `argmax` de entropia tem de encontrar o objeto
    por conta própria. Comparar os dois direto favorece o sorteio.

    O conserto é reponderar. Com `p` = fração de superpixels que tocam o
    objeto, o Δ esperado de um sorteio UNIFORME (sem privilégio) é

        Δ_uniforme = p · média(Δ | objeto) + (1 − p) · média(Δ | fundo)

    que é o estimador estratificado padrão. Para calculá-lo falta só `p`, e é
    `p` que este script mede.

Por que um arquivo separado
    `tabelas.py` lê o registro e nada mais — é o que torna a geração de tabela
    determinística e auditável. Fazer a tabela abrir imagem e rodar SLIC
    quebraria isso. Então `p` é medido aqui, uma vez, e versionado como dado.

    O cálculo não treina nada: é SLIC mais o ground truth, e é determinístico
    dadas a semente de partição e a lista de imagens — as mesmas do
    `ganho_marginal.py`, replicadas aqui de propósito para que um descasamento
    apareça como divergência e não passe batido.

Uso
    python scripts/pesos_estrato.py --ds schisto brats conjunctiva
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import shutil

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from flim_app import datasets as DS  # noqa: E402
import onde_marcar as OM  # noqa: E402

SAIDA = os.path.join(RAIZ, "evidencia", "bruto", "pesos_estrato.json")

# As partições usadas em cada dataset pela campanha `ganho_marginal`. A
# conjuntivite tem 82 imagens com objeto, e não cabe em pool 40 / val 40.
PARTICAO = {
    "schisto":     dict(n_pool=40, n_val=40, n_test=60),
    "brats":       dict(n_pool=40, n_val=40, n_test=60),
    "conjunctiva": dict(n_pool=20, n_val=25, n_test=30),
}


def _tem_objeto(ds_id, img) -> bool:
    lp = (DS.caminho_label(ds_id, img + ".png")
          or DS.caminho_label(ds_id, img + ".jpg"))
    if not lp:
        return False
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


def _slic(caminho, n_segments):
    from skimage.segmentation import slic
    arr = np.array(Image.open(caminho))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    return slic(arr[:, :, :3], n_segments=n_segments, compactness=10.0,
                sigma=1.0, start_label=0, convert2lab=True).astype(np.int32)


def medir(ds_id, k, sementes, n_segments, semente_particao, part) -> dict:
    cfg = DS.resolver(ds_id)
    cfg["_id"] = ds_id
    todas = [os.path.splitext(f)[0] for f in DS.imagens(ds_id)]
    rng0 = np.random.default_rng(semente_particao)
    com_obj = [i for i in todas[:600] if _tem_objeto(ds_id, i)]
    emb = rng0.permutation(len(com_obj))
    pool = sorted(com_obj[i] for i in emb[:part["n_pool"]])

    work = tempfile.mkdtemp(prefix="pes_")
    try:
        import tabela_k_por_modelo as TK
        cfg["orig"] = TK._orig_em_png(cfg, ds_id, set(pool), work)
        out = {}
        for semente in sementes:
            r = np.random.default_rng(semente * 7717 + 3)
            sel = [str(x) for x in r.permutation(pool)][:k]
            tot = obj = 0
            for img in sel:
                lp = DS.caminho_label(ds_id, img + ".png")
                if not lp:
                    continue
                gt = np.array(Image.open(lp).convert("L")) > 127
                seg = _slic(os.path.join(cfg["orig"], OM._arquivo(cfg, img)),
                            n_segments)
                for rid in np.unique(seg):
                    m = seg == rid
                    if m.sum() < 20:       # o mesmo corte do ganho_marginal
                        continue
                    tot += 1
                    if gt[m].any():
                        obj += 1
            out[str(semente)] = {
                "superpixels": tot, "tocam_objeto": obj,
                "p_objeto": round(obj / tot, 6) if tot else None,
                "imagens": sel,
            }
            print(f"  {ds_id} s{semente}: {obj}/{tot} tocam objeto "
                  f"(p={obj / tot:.4f})" if tot else f"  {ds_id} s{semente}: 0")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", nargs="+",
                    default=["schisto", "brats", "conjunctiva"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sementes", nargs="+", type=int,
                    default=list(range(10)))
    ap.add_argument("--n_segments", type=int, default=150)
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--saida", default=SAIDA)
    a = ap.parse_args()

    tudo = {}
    if os.path.isfile(a.saida):
        with open(a.saida, encoding="utf-8") as fh:
            tudo = json.load(fh)

    for ds_id in a.ds:
        part = PARTICAO.get(ds_id)
        if part is None:
            print(f"{ds_id}: partição não declarada em PARTICAO — pulado")
            continue
        print(f"{ds_id} (pool {part['n_pool']}):")
        tudo[ds_id] = {
            "particao": part,
            "k": a.k,
            "n_segments": a.n_segments,
            "semente_particao": a.semente_particao,
            "por_semente": medir(ds_id, a.k, a.sementes, a.n_segments,
                                 a.semente_particao, part),
        }

    os.makedirs(os.path.dirname(a.saida), exist_ok=True)
    with open(a.saida, "w", encoding="utf-8") as fh:
        json.dump(tudo, fh, indent=2, ensure_ascii=False)
    print(f"\ngravado em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
