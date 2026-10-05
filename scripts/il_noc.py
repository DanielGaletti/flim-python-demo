#!/usr/bin/env python3
"""
il_noc.py — quantos cliques até a qualidade alvo, com e sem proteção do banco

A pergunta
    As campanhas anteriores mediram Fβ com orçamento de anotação FIXO. Esta
    mede a moeda da segmentação interativa: NoC@X, o número de cliques até a
    IoU passar de X. É o eixo de RITM (arXiv:2102.06583) e SimpleClick
    (arXiv:2210.11006).

    O eixo é diferente, e é por isso que ele vale: já se sabe, neste projeto,
    que o Active Learning não melhora o Fβ a orçamento fixo. Resta saber se a
    proteção do banco de filtros reduz o esforço de interação, que é outra
    coisa.

Por que a proteção poderia reduzir o NoC
    Medido em `ganho_marginal`: acrescentar anotação ao FLIM pode derrubar o
    Fβ. Num laço de cliques isso aparece como regressão, o clique seguinte
    gasta-se desfazendo o dano do anterior, e o NoC sobe. Se a proteção
    impedir a regressão, o laço avança monotonicamente e precisa de menos
    cliques. É uma predição concreta, e ela pode falhar: com alpha baixo o
    clique também entra menos, e o laço pode travar sem atingir o alvo. É
    exatamente por isso que `taxa_sucesso` é reportada junto do NoC.

O usuário simulado
    Protocolo de Xu et al. (2016), em `flim_al/il_flim.py`: o clique vai ao
    ponto mais fundo do maior erro. Não há sorteio no laço, então o NoC é
    determinístico dado o encoder inicial.

Limite declarado
    Clique simulado do ground truth mede número de interações sob um usuário
    que acerta sempre. NÃO mede tempo de especialista.

Uso
    python scripts/il_noc.py --plano
    python scripts/il_noc.py --ds schisto --n_imagens 10 --alphas 0.5 1.0
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from flim_al import cl_flim, evidencia as ev, il_flim as IL  # noqa: E402
from flim_al.al_encoder_experiment import (  # noqa: E402
    _official_filter_and_binarize,
)
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

import ganho_marginal as GM  # noqa: E402
import onde_marcar as OM  # noqa: E402

EXPERIMENTO = "il_noc"
DECODER = "labeled_marker"

PARTICAO = {
    "schisto":     dict(n_pool=40, n_val=40),
    "brats":       dict(n_pool=40, n_val=40),
    "conjunctiva": dict(n_pool=20, n_val=25),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--n_imagens", type=int, default=10,
                    help="quantas imagens refinar, uma por execucao de laco")
    ap.add_argument("--pula", type=int, default=0,
                    help="pula as primeiras N imagens da sequencia. Serve "
                         "para retomar: a janela de execucao e limitada, e "
                         "refazer o que ja esta registrado seria so custo")
    ap.add_argument("--alvo", type=float, default=0.75)
    ap.add_argument("--max_cliques", type=int, default=12)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--rotulo", default="noc")
    ap.add_argument("--registro", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    if a.registro:
        ev.ARQUIVO = os.path.abspath(a.registro)
        os.makedirs(os.path.dirname(ev.ARQUIVO), exist_ok=True)
        print(f"registro redirecionado para {ev.ARQUIVO}")

    part = PARTICAO[a.ds]
    cfg = DS.resolver(a.ds)
    cfg["_id"] = a.ds
    cfg["bloco"] = cfg.get("bloco", 2)

    todas = [os.path.splitext(f)[0] for f in DS.imagens(a.ds)]
    rng0 = np.random.default_rng(a.semente_particao)
    com_obj = [i for i in todas[:600] if GM._tem_objeto(a.ds, i)]
    emb = rng0.permutation(len(com_obj))
    pool = sorted(com_obj[i] for i in emb[:part["n_pool"]])
    # A sequencia e fixa pela semente 99, entao `pula` recorta sempre o mesmo
    # trecho e os pedacos de execucoes diferentes se encaixam sem sobrepor.
    alvos = [str(x) for x in np.random.default_rng(99).permutation(pool)]
    alvos = alvos[a.pula:a.pula + a.n_imagens]

    print(f"dataset {a.ds} · {len(alvos)} imagens · alvo IoU {a.alvo} · "
          f"max {a.max_cliques} cliques · alphas {a.alphas}")
    print(f"= ate {len(alvos) * len(a.alphas) * (a.max_cliques + 1)} treinos")
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="ilnoc_")
    import tabela_k_por_modelo as TK
    cfg["orig"] = TK._orig_em_png(cfg, a.ds, set(alvos), work)
    fonte = f"{EXPERIMENTO}/alvo{a.alvo}/{a.rotulo}"
    print(f"fonte: {fonte}\n")

    from flim_app import server as SV
    SV.S["ds"] = a.ds
    SV.S["block"] = cfg["bloco"]
    SV.S["decoder"] = DECODER

    contador = {"n": 0}

    def _escrever(markers, dest):
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest, exist_ok=True)
        for img, (fg, bg, H, W) in markers.items():
            if not fg and not bg:
                continue
            save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": H, "W": W},
                         os.path.join(dest, f"{img}-seeds.txt"))

    def faz_treinar(banco_ini):
        def treinar(markers, alpha, banco):
            contador["n"] += 1
            d = os.path.join(work, f"m{contador['n']}")
            enc = os.path.join(work, f"e{contador['n']}.pth")
            _escrever(markers, d)
            cl_flim.treinar(cfg["arch"], d, cfg["orig"], cfg["label"],
                            a.device, enc,
                            banco=banco, alpha=alpha,
                            gravar_banco=(banco_ini if banco is None
                                          else None))
            return enc
        return treinar

    def prever(enc, img):
        SV.S["encoder"] = enc
        modelo = SV._montar_modelo(enc)
        prob = SV._mapa_prob(modelo, img)
        sal = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        return _official_filter_and_binarize(
            sal, area_range=tuple(cfg["area"])).astype(bool)

    registros, falhas, por_alpha = [], [], {}
    for img in alvos:
        lp = DS.caminho_label(a.ds, img + ".png")
        if not lp:
            continue
        gt = np.array(Image.open(lp).convert("L")) > 127
        d = generate_realistic_markers(
            lp, n_fg_dabs=6, n_bg_dabs=14,
            seed=OM._semente_marker(img, a.semente_particao))
        base = {img: ([tuple(int(v) for v in p) for p in d["fg_seeds"]],
                      [tuple(int(v) for v in p) for p in d["bg_seeds"]],
                      gt.shape[0], gt.shape[1])}

        for alpha in a.alphas:
            banco_ini = os.path.join(work, f"b_{img}_{alpha}.pkl")
            try:
                # O banco inicial sai do PRIMEIRO treino do laco, que usa os
                # marcadores base. O laco entao protege contra ele.
                t0 = time.time()
                tr = faz_treinar(banco_ini)
                enc0 = tr(base, 1.0, None)
                r = IL.laco(
                    lambda m, al, bk: tr(m, al, banco_ini),
                    prever, base, {img: gt}, img,
                    alvo=a.alvo, max_cliques=a.max_cliques, alpha=alpha)
                dt = time.time() - t0
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"{img} a={alpha}: {type(e).__name__}: {e}")
                print(f"  {img} a={alpha}  FALHOU: {e}", flush=True)
                continue
            por_alpha.setdefault(alpha, []).append(r)
            print(f"  {img} a={alpha:<4} IoU {r['iou_inicial']:.3f} -> "
                  f"{r['iou_final']:.3f}  NoC={r['noc']:2d} "
                  f"{'OK' if r['atingiu'] else 'NAO ATINGIU'}  ({dt:.0f}s)",
                  flush=True)
            lote = [ev.execucao(
                experimento=EXPERIMENTO, dataset=a.ds,
                braco="flim" if alpha >= 1.0 else "al",
                variante=(f"alpha={alpha:.2f}|noc={r['noc']}|"
                          f"atingiu={int(r['atingiu'])}|"
                          f"iou0={r['iou_inicial']:.6f}|{img}"),
                criterio=f"il_alpha{alpha:.2f}", decoder=DECODER,
                decoder_paper="FLIM_lm", bloco=cfg["bloco"],
                orcamento=r["noc"],
                orcamento_px=sum(c["px"] for c in r["cliques"]),
                seed=a.semente_particao, imagens=[img],
                marker_origem="sintetico",
                fb=r["iou_final"], dice=r["iou_final"],
                iou=r["iou_final"], mae=1.0 - r["iou_final"],
                segundos=round(dt, 2), fonte=fonte,
            )]
            ev.registrar(lote)
            registros.extend(lote)

    print()
    for alpha, execs in sorted(por_alpha.items()):
        r = IL.resumo(execs, a.alvo)
        print(f"  alpha={alpha:<5} n={r['n']:2d}  NoC medio {r['noc_medio']:5.2f}"
              f"  mediano {r['noc_mediano']:4.1f}"
              f"  sucesso {r['taxa_sucesso']:.0%}"
              f"  IoU {r['iou_inicial_medio']:.3f} -> {r['iou_final_medio']:.3f}")

    if falhas:
        print(f"\n{len(falhas)} falha(s):")
        for f in falhas[:8]:
            print(f"  {f}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nregistrados: {len(registros)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrompido")
