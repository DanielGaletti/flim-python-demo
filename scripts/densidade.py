#!/usr/bin/env python3
"""
densidade.py — poucas imagens densas ou muitas imagens esparsas?

A pergunta
    O orçamento do especialista é contado em PIXELS anotados, fixo. A pergunta
    é onde gastá-lo: concentrando numa imagem só, ou espalhando por oito.

    Ela não é a mesma de `onde_marcar.py`. Lá as imagens eram fixas e variava
    ONDE o traço caía dentro delas. Aqui varia QUANTAS imagens recebem o mesmo
    total de tinta.

Por que esta é a candidata a melhoria
    Medido antes, numa única imagem: com 4.037 pixels marcados o encoder sai
    com 54/54/48/48 kernels; com 23.095, sai com 200/200/183/183. Oito imagens
    esparsas chegam a 200/200/200/200 — mas gastando 32 mil pixels. Uma imagem
    densa chega quase lá com 23 mil.

    Isso é indício de que anotar MENOS imagens, mais densamente, entrega a
    mesma capacidade por menos trabalho. Indício, não resultado: aquilo mediu
    capacidade do encoder, não Fβ. É o que este experimento mede.

    Se der positivo, é a única melhoria defensável do trabalho — e ela preserva
    a premissa do FLIM melhor ainda que o Active Learning, porque usa MENOS
    imagens, não mais.

A referência
    O braço de OITO imagens: é o que mais se aproxima de "espalhar", e é o
    extremo oposto do que se quer testar. Δ positivo num braço de poucas
    imagens significa que concentrar compensou.

O que este experimento NÃO controla
    As imagens são sorteadas, então o braço de uma imagem depende mais da sorte
    daquela imagem única. É por isso que ele roda com várias sementes e o teste
    é pareado: dentro de cada semente os braços compartilham partição, conjunto
    de teste e gerador de traços.

Uso
    python scripts/densidade.py --plano
    python scripts/densidade.py --ds schisto --px_total 8000 24000 --sementes 0 1 2
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

from flim_al import evidencia as ev  # noqa: E402
from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, retrain_encoder,
)
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

import onde_marcar as OM  # noqa: E402

EXPERIMENTO = "densidade"

DECODERS = [
    ("labeled_marker", "FLIM_lm"),
    ("decoder_2",      "FLIM_pb"),
]

# Quantas imagens recebem o orçamento. O 8 é a referência.
N_IMAGENS = [1, 2, 3, 5, 8]

REFERENCIA = 8


def _tem_objeto(ds_id: str, img: str) -> bool:
    lp = (DS.caminho_label(ds_id, img + ".png")
          or DS.caminho_label(ds_id, img + ".jpg"))
    if not lp:
        return False
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


def _markers(cfg, ds_id, imagens, px_por_imagem, dest, semente) -> int:
    """
    Cada imagem recebe `px_por_imagem` pixels, pelo mesmo gerador de traço.

    A semente do traço sai do id da imagem, como em `tabela_k_por_modelo`: a
    mesma imagem recebe sempre o mesmo traço dentro de uma semente de campanha,
    em qualquer braço. Sem isso, o braço de uma imagem e o de oito anotariam a
    imagem que têm em comum de formas diferentes.
    """
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for img in imagens:
        lp = DS.caminho_label(ds_id, img + ".png")
        if not lp:
            continue
        fg, bg = OM._uniforme(lp, px_por_imagem,
                              OM._semente_marker(img, semente))
        if not fg and not bg:
            continue
        with Image.open(lp) as im:
            w, h = im.size
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": h, "W": w},
                     os.path.join(dest, f"{img}-seeds.txt"))
        n += 1
    return n


def _pixels_gravados(dest) -> int:
    t = 0
    for f in os.listdir(dest):
        if f.endswith("-seeds.txt"):
            with open(os.path.join(dest, f)) as fh:
                t += int(fh.readline().split()[0])
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--px_total", nargs="+", type=int, default=[8000, 24000],
                    help="orcamento TOTAL de pixels, dividido entre as imagens")
    ap.add_argument("--n_imagens", nargs="+", type=int, default=N_IMAGENS)
    ap.add_argument("--sementes", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--n_pool", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--decoders", nargs="+", default=None)
    ap.add_argument("--rotulo", default="densidade")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    decoders = DECODERS if not a.decoders else [
        d for d in DECODERS if d[0] in a.decoders]

    cfg = DS.resolver(a.ds)
    cfg["_id"] = a.ds
    cfg["bloco"] = cfg.get("bloco", 2)

    todas = [os.path.splitext(f)[0] for f in DS.imagens(a.ds)]
    rng = np.random.default_rng(a.semente_particao)
    com_obj = [i for i in todas[:600] if _tem_objeto(a.ds, i)]
    emb = rng.permutation(len(com_obj))
    pool = sorted(com_obj[i] for i in emb[:a.n_pool])
    test = sorted(com_obj[i] for i in emb[a.n_pool:a.n_pool + a.n_test])

    n_cel = len(a.n_imagens) * len(a.px_total) * len(a.sementes)
    print(f"dataset {a.ds} · pool {len(pool)} · teste {len(test)}")
    print(f"{len(a.n_imagens)} braços × {len(a.px_total)} orçamentos × "
          f"{len(a.sementes)} sementes = {n_cel} células")
    print(f"orçamento TOTAL de pixels: {a.px_total}")
    print(f"imagens por braço: {a.n_imagens} (referência: {REFERENCIA})")
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="dens_")
    cfg["orig"] = OM_orig_png(cfg, a.ds, set(pool) | set(test), work)
    fonte = (f"{EXPERIMENTO}/pool{len(pool)}/teste{len(test)}/{a.rotulo}")
    print(f"fonte: {fonte}\n")

    registros, falhas = [], []
    for semente in a.sementes:
        # As imagens sao sorteadas UMA vez por semente: o braco de 8 contem as
        # do braco de 5, que contem as do de 3. Assim a diferenca entre os
        # bracos e so quantas imagens receberam tinta, e nao quais.
        r = np.random.default_rng(semente * 977 + 13)
        ordem = [str(x) for x in r.permutation(pool)]
        for px_total in a.px_total:
            for n_img in a.n_imagens:
                if n_img > len(ordem):
                    continue
                sel = ordem[:n_img]
                por_img = max(1, px_total // n_img)
                mdir = os.path.join(work, f"m_{n_img}_{px_total}_{semente}")
                try:
                    if _markers(cfg, a.ds, sel, por_img, mdir, semente) == 0:
                        raise ValueError("nenhum marker gerado")
                    gastos = _pixels_gravados(mdir)
                    enc = os.path.join(
                        work, f"e_{n_img}_{px_total}_{semente}.pth")
                    t0 = time.time()
                    retrain_encoder(cfg["arch"], mdir, cfg["orig"],
                                    cfg["label"], a.device, enc)
                    t_treino = time.time() - t0
                except Exception as e:                        # noqa: BLE001
                    falhas.append(f"{n_img}img px{px_total} s{semente}: "
                                  f"{type(e).__name__}: {e}")
                    print(f"  s{semente} {n_img}img px{px_total}  FALHOU: {e}",
                          flush=True)
                    continue

                print(f"  s{semente} {n_img}img total={px_total} "
                      f"({por_img}/img, gastos {gastos})  "
                      f"treino {t_treino:5.1f}s", end="", flush=True)
                novos = []
                for dec, nome in decoders:
                    t1 = time.time()
                    try:
                        m = evaluate_decoder(
                            enc, dec, cfg["bloco"],
                            [OM._arquivo(cfg, i) for i in test],
                            cfg["orig"], cfg["label"], a.device,
                            area_range=tuple(cfg["area"]))
                    except Exception as e:                    # noqa: BLE001
                        falhas.append(f"{n_img}img px{px_total} {dec}: {e}")
                        continue
                    t_aval = time.time() - t1
                    novos.append(ev.execucao(
                        experimento=EXPERIMENTO, dataset=a.ds,
                        braco="flim" if n_img == REFERENCIA else "densidade",
                        variante=f"{n_img}img_total{px_total}",
                        criterio=f"{n_img}_imagens",
                        decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
                        orcamento=n_img, orcamento_px=por_img,
                        seed=semente, imagens=sel,
                        marker_origem="sintetico",
                        fb=m["fb"], dice=m["dice"], iou=m.get("iou"),
                        mae=m["mae"], segundos=round(t_treino + t_aval, 2),
                        segundos_treino=round(t_treino, 2),
                        segundos_aval=round(t_aval, 2),
                        fonte=fonte,
                    ))
                if novos:
                    res = ev.registrar(novos)
                    registros.extend(novos)
                    if res["divergentes"]:
                        print(f"\n  DIVERGENCIA em "
                              f"{len(res['divergentes'])} run_id(s)", flush=True)
                print(f"  ·  {len(novos)} decoders  "
                      f"[{len(registros)} registrados]", flush=True)

    if falhas:
        print(f"\n{len(falhas)} falha(s):")
        for f in falhas[:8]:
            print(f"  {f}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nregistrados: {len(registros)}")
    return 1 if falhas else 0


def OM_orig_png(cfg, ds_id, imagens, work):
    """Delegado para o conversor de `tabela_k_por_modelo` (datasets .jpg)."""
    import tabela_k_por_modelo as TK
    return TK._orig_em_png(cfg, ds_id, imagens, work)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrompido")
