#!/usr/bin/env python3
"""
clique.py — onde o clique do especialista cai dentro do objeto

A pergunta, em uma frase
    Mesmas imagens, mesmo número de cliques, mesmos traços de fundo. Muda só
    ONDE os cliques de objeto caem. Isso importa?

Os três braços

    flim_puro         clique na BORDA do objeto — é o que o artigo faz e o que
                      a aplicação ensina. Sem AL nenhum.
    al_regiao         clique onde o modelo está mais incerto, dentro do objeto.
                      O Active Learning escolhe o lugar.
    aleatorio_objeto  clique em posição sorteada, dentro do objeto. É o
                      CONTROLE: separa "o AL escolheu bem" de "qualquer lugar
                      serve, desde que seja dentro do objeto".

    Os três estão sempre DENTRO do objeto, então o rótulo é sempre correto. O
    que difere é só a posição.

Por que este desenho é melhor que comparar anotações inteiras
    Ele isola uma variável. O número de cliques, o número de pixels, as
    imagens e todo o traço de fundo são idênticos nos três braços — copiados,
    não regenerados. Se o Fβ mudar, só pode ter sido a posição do clique de
    objeto.

    Nos experimentos anteriores deste projeto, comparações de anotação
    mudavam duas ou três coisas ao mesmo tempo (quais imagens, quantos pixels,
    o balanço entre objeto e fundo), e cada uma exigiu um braço de controle
    próprio. Aqui não é preciso: só uma coisa varia.

De onde o AL tira a incerteza
    De um encoder treinado no braço `flim_puro` das MESMAS imagens. É a ordem
    natural: o especialista anota como sempre, o modelo é treinado, e só então
    ele tem opinião sobre onde faltou informação. O braço `al_regiao` usa essa
    opinião para recolocar os mesmos cliques.

Uso
    python scripts/clique.py --plano
    python scripts/clique.py --ds schisto --ks 3 5 --sementes 0 1 2 3 4
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
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

import onde_marcar as OM  # noqa: E402

EXPERIMENTO = "clique"
BRACOS = ["flim_puro", "al_regiao", "aleatorio_objeto"]
RAIO = 5          # o pincel real do FLIM, medido nos 31 markers dos usuários

DECODERS = [
    ("labeled_marker", "FLIM_lm"),
    ("decoder_2",      "FLIM_pb"),
]


def _tem_objeto(ds_id, img) -> bool:
    lp = (DS.caminho_label(ds_id, img + ".png")
          or DS.caminho_label(ds_id, img + ".jpg"))
    if not lp:
        return False
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


def _disco(cx, cy, gt, vistos, raio=RAIO):
    """Um toque de pincel, recortado no objeto — o rótulo nunca sai errado."""
    H, W = gt.shape
    out = []
    for dy in range(-raio, raio + 1):
        for dx in range(-raio, raio + 1):
            if dx * dx + dy * dy > raio * raio:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < W and 0 <= y < H and gt[y, x] and (x, y) not in vistos:
                vistos.add((x, y))
                out.append((x, y))
    return out


def _cliques_em(centros, gt, n_alvo):
    """Toques nos centros dados, até fechar `n_alvo` pixels."""
    pts, vistos = [], set()
    for cy, cx in centros:
        pts += _disco(int(cx), int(cy), gt, vistos)
        if len(pts) >= n_alvo:
            break
    return pts[:n_alvo]


def _centros_incerteza(gt, seg, prob):
    """Centros dos superpixels que tocam o objeto, do mais incerto ao menos."""
    from flim_al.region_al import score_regions_by_entropy
    scores = score_regions_by_entropy(seg, prob)
    saida = []
    for rid, _ in sorted(scores.items(), key=lambda kv: -kv[1]):
        m = (seg == rid) & gt
        if not m.any():
            continue                     # superpixel fora do objeto: descarta
        ys, xs = np.nonzero(m)
        saida.append((int(ys.mean()), int(xs.mean())))
    return saida


def _centros_aleatorios(gt, rng, n=400):
    ys, xs = np.nonzero(gt)
    idx = rng.permutation(len(ys))[:n]
    return [(int(ys[i]), int(xs[i])) for i in idx]


def _escrever(dest, por_img):
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for img, (fg, bg, H, W) in por_img.items():
        if not fg and not bg:
            continue
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": H, "W": W},
                     os.path.join(dest, f"{img}-seeds.txt"))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--bracos", nargs="+", default=BRACOS)
    ap.add_argument("--sementes", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--n_pool", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_segments", type=int, default=220)
    ap.add_argument("--decoders", nargs="+", default=None)
    ap.add_argument("--rotulo", default="clique")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    decoders = DECODERS if not a.decoders else [
        d for d in DECODERS if d[0] in a.decoders]

    cfg = DS.resolver(a.ds)
    cfg["_id"] = a.ds
    cfg["bloco"] = cfg.get("bloco", 2)

    todas = [os.path.splitext(f)[0] for f in DS.imagens(a.ds)]
    rng0 = np.random.default_rng(a.semente_particao)
    com_obj = [i for i in todas[:600] if _tem_objeto(a.ds, i)]
    emb = rng0.permutation(len(com_obj))
    pool = sorted(com_obj[i] for i in emb[:a.n_pool])
    test = sorted(com_obj[i] for i in emb[a.n_pool:a.n_pool + a.n_test])

    n_cel = len(a.bracos) * len(a.ks) * len(a.sementes)
    print(f"dataset {a.ds} · pool {len(pool)} · teste {len(test)}")
    print(f"{len(a.bracos)} braços × {len(a.ks)} K × {len(a.sementes)} "
          f"sementes = {n_cel} células")
    print("braços:", ", ".join(a.bracos))
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="clq_")
    import tabela_k_por_modelo as TK
    cfg["orig"] = TK._orig_em_png(cfg, a.ds, set(pool) | set(test), work)
    fonte = f"{EXPERIMENTO}/pool{len(pool)}/teste{len(test)}/{a.rotulo}"
    print(f"fonte: {fonte}\n")

    registros, falhas = [], []
    for semente in a.sementes:
        r = np.random.default_rng(semente * 7717 + 3)
        ordem = [str(x) for x in r.permutation(pool)]
        for k in a.ks:
            sel = ordem[:k]

            # ── o traço padrão: é o braço `flim_puro` e a base dos outros ──
            base = {}
            for img in sel:
                lp = DS.caminho_label(a.ds, img + ".png")
                if not lp:
                    continue
                gt = np.array(Image.open(lp).convert("L")) > 127
                d = generate_realistic_markers(
                    lp, n_fg_dabs=6, n_bg_dabs=14,
                    seed=OM._semente_marker(img, semente))
                fg = [tuple(int(v) for v in p) for p in d["fg_seeds"]]
                bg = [tuple(int(v) for v in p) for p in d["bg_seeds"]]
                base[img] = (fg, bg, gt)
            if not base:
                continue

            # O encoder que dá a opinião do AL sai do braço padrão. Ordem
            # natural: anota como sempre, treina, e só então há incerteza.
            d0 = os.path.join(work, f"b_{k}_{semente}")
            _escrever(d0, {i: (f, b, g.shape[0], g.shape[1])
                           for i, (f, b, g) in base.items()})
            enc0 = os.path.join(work, f"b_{k}_{semente}.pth")
            modelo = None
            try:
                retrain_encoder(cfg["arch"], d0, cfg["orig"], cfg["label"],
                                a.device, enc0)
                from flim_app import server as SV
                SV.S["ds"] = a.ds
                SV.S["block"] = cfg["bloco"]
                SV.S["decoder"] = "labeled_marker"
                SV.S["encoder"] = enc0
                modelo = SV._montar_modelo(enc0)
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"encoder base K={k} s={semente}: {e}")

            for braco in a.bracos:
                por_img = {}
                try:
                    for img, (fg0, bg0, gt) in base.items():
                        H, W = gt.shape
                        n_alvo = len(fg0)
                        if braco == "flim_puro":
                            fg = fg0
                        elif braco == "aleatorio_objeto":
                            fg = _cliques_em(_centros_aleatorios(gt, r), gt,
                                             n_alvo)
                        else:
                            if modelo is None:
                                raise RuntimeError("sem encoder base")
                            from skimage.segmentation import slic
                            arr = np.array(Image.open(os.path.join(
                                cfg["orig"], OM._arquivo(cfg, img))))
                            if arr.ndim == 2:
                                arr = np.stack([arr] * 3, axis=2)
                            seg = slic(arr[:, :, :3],
                                       n_segments=a.n_segments,
                                       compactness=10.0, sigma=1.0,
                                       start_label=0,
                                       convert2lab=True).astype(np.int32)
                            prob = SV._mapa_prob(modelo, img)
                            if prob.shape != (H, W):
                                prob = np.array(Image.fromarray(
                                    (prob * 255).astype(np.uint8)).resize(
                                        (W, H), Image.BILINEAR)) / 255.0
                            fg = _cliques_em(
                                _centros_incerteza(gt, seg, prob), gt, n_alvo)
                        # o fundo é COPIADO, nunca regerado: ele não é a
                        # variável deste experimento
                        por_img[img] = (fg, bg0, H, W)

                    mdir = os.path.join(work, f"m_{braco}_{k}_{semente}")
                    if _escrever(mdir, por_img) == 0:
                        raise ValueError("nenhum marker")
                    enc = os.path.join(work, f"e_{braco}_{k}_{semente}.pth")
                    t0 = time.time()
                    retrain_encoder(cfg["arch"], mdir, cfg["orig"],
                                    cfg["label"], a.device, enc)
                    t_treino = time.time() - t0
                except Exception as e:                        # noqa: BLE001
                    falhas.append(f"{braco} K={k} s={semente}: "
                                  f"{type(e).__name__}: {e}")
                    print(f"  s{semente} {braco:<17} K={k}  FALHOU: {e}",
                          flush=True)
                    continue

                px_fg = sum(len(v[0]) for v in por_img.values())
                px_bg = sum(len(v[1]) for v in por_img.values())
                print(f"  s{semente} {braco:<17} K={k}  "
                      f"fg={px_fg} bg={px_bg}  treino {t_treino:5.1f}s",
                      end="", flush=True)
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
                        falhas.append(f"{braco} K={k} {dec}: {e}")
                        continue
                    t_aval = time.time() - t1
                    novos.append(ev.execucao(
                        experimento=EXPERIMENTO, dataset=a.ds,
                        braco="flim" if braco == "flim_puro" else "al",
                        variante=f"{braco}_K{k}", criterio=braco,
                        decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
                        orcamento=k, orcamento_px=px_fg,
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrompido")
