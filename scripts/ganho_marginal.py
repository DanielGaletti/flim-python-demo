#!/usr/bin/env python3
"""
ganho_marginal.py — existe ganho a capturar na escolha da regiao?

A lacuna que isto fecha
    Todas as campanhas anteriores mediram o DESEMPENHO de criterios de Active
    Learning. Nenhuma mediu o TETO que eles tentam atingir. Sabemos que
    entropy, least_confidence, coreset e a escolha de regiao nao batem o
    sorteio — nao sabemos por que.

    Tres causas diferentes produzem exatamente o mesmo resultado negativo:

        (a) existem regioes uteis e o escore nao as encontra
        (b) nenhuma regiao e util neste regime
        (c) o FLIM nao incorpora o marcador adicional

    Cada uma exige uma conclusao diferente na dissertacao. Este experimento
    as separa.

Como
    Fixa tudo — particao, as K imagens, os marcadores base, o encoder inicial
    E0 — e varia UMA coisa: qual regiao recebe o marcador seguinte. Para cada
    candidato r, retreina e mede

        Delta_r = Fbeta(E_r, validacao) - Fbeta(E0, validacao)

    O conjunto de Deltas e a resposta. Sua dispersao diz se ha teto (a vs b);
    sua correlacao com o escore de entropia diz se o criterio o alcanca.

O controle que separa (b) de (c)
    Duas imagens novas inteiras, anotadas pelo gerador padrao, medidas no
    MESMO Delta. Se nenhuma regiao move o Fbeta mas uma imagem move, entao o
    FLIM incorpora marcadores e o que falta e informacao — uma regiao nao
    carrega o suficiente. Se nem a imagem move, o encoder saturou.

Informacao privilegiada — leia antes de citar
    Delta usa o ground truth da validacao para ser calculado. Isto e um
    DIAGNOSTICO, nao um metodo: o "oracle de regiao" nao existe em producao.
    Ele mede o teto, e teto nao se propoe como estrategia.

    O conjunto de TESTE nao e tocado por esta campanha.

Unidade independente de analise
    A semente. Os 24 candidatos dentro de uma semente compartilham encoder,
    imagens e particao — sao correlacionados e nunca entram como n. Toda
    estatistica e pareada por semente.

Uso
    python scripts/ganho_marginal.py --plano
    python scripts/ganho_marginal.py --ds schisto --sementes 0 --n_cand 6   # piloto
    python scripts/ganho_marginal.py --ds schisto --sementes 0 1 2 3 4 5 6 7 8 9
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

EXPERIMENTO = "ganho_marginal"

DECODERS = [
    ("labeled_marker", "FLIM_lm"),
    ("decoder_2",      "FLIM_pb"),
]

# Quantos candidatos de cada estrato. O objeto ocupa ~3% da imagem no Schisto:
# um sorteio uniforme traria ~1 candidato tocando o objeto em 24, e o
# experimento mediria o ganho de anotar fundo. Os estratos sao reportados
# separados, nunca agregados sem peso.
N_OBJ, N_BG = 12, 10

PX_REGIAO = 300      # orcamento identico para todo candidato


def _tem_objeto(ds_id, img) -> bool:
    lp = (DS.caminho_label(ds_id, img + ".png")
          or DS.caminho_label(ds_id, img + ".jpg"))
    if not lp:
        return False
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


def _escrever(dest, por_img) -> int:
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


def _slic(caminho, n_segments):
    from skimage.segmentation import slic
    arr = np.array(Image.open(caminho))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    return slic(arr[:, :, :3], n_segments=n_segments, compactness=10.0,
                sigma=1.0, start_label=0, convert2lab=True).astype(np.int32)


def _pixels_da_regiao(mask, gt, n_alvo, rng):
    """
    Os pixels que o anotador cobriria ao passar o pincel nesta regiao.

    Rotulados pelo ground truth LOCAL — que e o que ele enxerga ali. Nao e
    informacao privilegiada: e a definicao de anotar. O privilegio deste
    experimento esta em ESCOLHER a regiao olhando o Fbeta, nao em rotula-la.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return [], []
    idx = rng.permutation(len(ys))[:n_alvo]
    fg, bg = [], []
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        (fg if gt[y, x] else bg).append((x, y))
    return fg, bg


def _kernels(enc_path) -> list:
    """
    Quantos kernels cada camada conseguiu extrair.

    A arquitetura pede 200 por camada, mas o k-means so entrega tantos
    centroides quantos houver patches distintos sob os markers — medido antes
    neste projeto: traco espalhado da 200, concentrado da 141. Sem isto, o
    ganho por CAPACIDADE (mais pixels, mais kernels) fica indistinguivel do
    ganho por POSICAO (os mesmos pixels em lugar melhor).
    """
    import torch
    try:
        mdl = torch.load(enc_path, map_location="cpu", weights_only=False)
        return [int(v.shape[0]) for v in mdl.state_dict().values()
                if hasattr(v, "ndim") and v.ndim == 4]
    except Exception:                                         # noqa: BLE001
        return []


def _fbeta(enc, cfg, decoders, arquivos, device):
    """Fbeta na validacao, por decoder."""
    out = {}
    for dec, _ in decoders:
        m = evaluate_decoder(enc, dec, cfg["bloco"], arquivos, cfg["orig"],
                             cfg["label"], device,
                             area_range=tuple(cfg["area"]))
        out[dec] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sementes", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--n_pool", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_segments", type=int, default=150)
    ap.add_argument("--n_cand", type=int, default=None,
                    help="corta o numero de candidatos sorteados (piloto)")
    ap.add_argument("--n_imagem_nova", type=int, default=2)
    ap.add_argument("--px_regiao", type=int, default=PX_REGIAO)
    ap.add_argument("--decoders", nargs="+", default=None)
    ap.add_argument("--rotulo", default="diagnostico")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--registro", default=None,
                    help="CSV alternativo. Serve para rodar sem disputar o "
                         "registro canonico com outra campanha; o merge e "
                         "feito depois, sob a trava, por "
                         "scripts/mesclar_registro.py")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    if a.registro:
        # Nao basta passar o caminho a cada `registrar`: o modulo tambem le
        # ARQUIVO para deduplicar. Redirecionar o modulo inteiro mantem as
        # duas coisas consistentes.
        ev.ARQUIVO = os.path.abspath(a.registro)
        os.makedirs(os.path.dirname(ev.ARQUIVO), exist_ok=True)
        print(f"registro redirecionado para {ev.ARQUIVO}")

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
    val = sorted(com_obj[i] for i in emb[a.n_pool:a.n_pool + a.n_val])
    test = sorted(com_obj[i] for i in
                  emb[a.n_pool + a.n_val:a.n_pool + a.n_val + a.n_test])

    n_obj = N_OBJ if a.n_cand is None else max(1, a.n_cand // 2)
    n_bg = N_BG if a.n_cand is None else max(1, a.n_cand - n_obj)
    n_cand_total = 2 + n_obj + n_bg + a.n_imagem_nova

    print(f"dataset {a.ds} · pool {len(pool)} · val {len(val)} · "
          f"teste {len(test)} (NAO tocado)")
    print(f"K={a.k} · {len(a.sementes)} sementes · "
          f"{n_cand_total} candidatos/semente "
          f"(2 argmax + {n_obj} objeto + {n_bg} fundo + "
          f"{a.n_imagem_nova} imagem nova)")
    print(f"= {len(a.sementes) * (1 + n_cand_total)} retreinos, "
          f"{len(decoders)} decoders cada")
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="gmar_")
    import tabela_k_por_modelo as TK
    cfg["orig"] = TK._orig_em_png(
        cfg, a.ds, set(pool) | set(val) | set(test), work)
    arq_val = [OM._arquivo(cfg, i) for i in val]
    fonte = (f"{EXPERIMENTO}/pool{len(pool)}/val{len(val)}/{a.rotulo}")
    print(f"fonte: {fonte}\n")

    registros, falhas = [], []
    for semente in a.sementes:
        r = np.random.default_rng(semente * 7717 + 3)
        ordem = [str(x) for x in r.permutation(pool)]
        sel = ordem[:a.k]
        novas = ordem[a.k:a.k + a.n_imagem_nova]

        # ── o estado inicial: o FLIM como ele e, sem AL nenhum ──────────────
        base, gts = {}, {}
        for img in sel:
            lp = DS.caminho_label(a.ds, img + ".png")
            if not lp:
                continue
            gt = np.array(Image.open(lp).convert("L")) > 127
            d = generate_realistic_markers(
                lp, n_fg_dabs=6, n_bg_dabs=14,
                seed=OM._semente_marker(img, semente))
            base[img] = ([tuple(int(v) for v in p) for p in d["fg_seeds"]],
                         [tuple(int(v) for v in p) for p in d["bg_seeds"]],
                         gt.shape[0], gt.shape[1])
            gts[img] = gt
        if not base:
            falhas.append(f"s{semente}: nenhuma imagem base")
            continue

        d0 = os.path.join(work, f"base_{semente}")
        enc0 = os.path.join(work, f"base_{semente}.pth")
        try:
            _escrever(d0, base)
            t0 = time.time()
            retrain_encoder(cfg["arch"], d0, cfg["orig"], cfg["label"],
                            a.device, enc0)
            t_base = time.time() - t0
            m0 = _fbeta(enc0, cfg, decoders, arq_val, a.device)
        except Exception as e:                                # noqa: BLE001
            falhas.append(f"s{semente} base: {type(e).__name__}: {e}")
            print(f"  s{semente} BASE FALHOU: {e}", flush=True)
            continue

        fb0 = {d: m0[d]["fb"] for d, _ in decoders}
        kn0 = _kernels(enc0)
        print(f"  s{semente} base  treino {t_base:5.1f}s  "
              f"kernels={kn0}  "
              + "  ".join(f"{n}={fb0[d]:.4f}" for d, n in decoders),
              flush=True)

        novos = [ev.execucao(
            experimento=EXPERIMENTO, dataset=a.ds, braco="flim",
            variante="base|k=" + "-".join(str(x) for x in kn0),
            criterio="base",
            decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
            orcamento=a.k, orcamento_px=0, seed=semente, imagens=sel,
            marker_origem="sintetico",
            fb=m0[dec]["fb"], dice=m0[dec]["dice"], iou=m0[dec].get("iou"),
            mae=m0[dec]["mae"], segundos=round(t_base, 2),
            segundos_treino=round(t_base, 2), fonte=fonte,
        ) for dec, nome in decoders]
        ev.registrar(novos)
        registros.extend(novos)

        # ── o escore do AL, calculado de E0 e SEM ground truth ──────────────
        from flim_al.region_al import score_regions_by_entropy
        from flim_app import server as SV
        SV.S["ds"] = a.ds
        SV.S["block"] = cfg["bloco"]
        SV.S["decoder"] = "labeled_marker"
        SV.S["encoder"] = enc0
        try:
            modelo = SV._montar_modelo(enc0)
        except Exception as e:                                # noqa: BLE001
            falhas.append(f"s{semente} modelo: {e}")
            continue

        cands = []          # (img, rid, mask, estrato, entropia, lc, frac_fg)
        por_img_scores = {}
        for img in base:
            gt = gts[img]
            H, W = gt.shape
            seg = _slic(os.path.join(cfg["orig"], OM._arquivo(cfg, img)),
                        a.n_segments)
            prob = SV._mapa_prob(modelo, img)
            if prob.shape != (H, W):
                prob = np.array(Image.fromarray(
                    (prob * 255).astype(np.uint8)).resize(
                        (W, H), Image.BILINEAR)) / 255.0
            ent = score_regions_by_entropy(seg, prob)
            lc_map = 1 - np.abs(prob - 0.5) * 2
            for rid in np.unique(seg):
                mask = seg == rid
                if mask.sum() < 20:
                    continue
                frac = float(gt[mask].mean())
                por_img_scores[(img, int(rid))] = dict(
                    mask=mask,
                    entropia=float(ent[int(rid)]),
                    lc=float(lc_map[mask].mean()),
                    frac_fg=frac,
                    estrato="objeto" if frac > 0 else "fundo",
                )

        chaves = list(por_img_scores)
        # Os dois argmax GLOBAIS: e a regiao que o criterio escolheria de fato.
        top_ent = max(chaves, key=lambda c: por_img_scores[c]["entropia"])
        top_lc = max(chaves, key=lambda c: por_img_scores[c]["lc"])
        k_obj = [c for c in chaves if por_img_scores[c]["estrato"] == "objeto"]
        k_bg = [c for c in chaves if por_img_scores[c]["estrato"] == "fundo"]
        rs = np.random.default_rng(semente * 331 + 7)
        sorteio = (
            [k_obj[i] for i in rs.permutation(len(k_obj))[:n_obj]]
            + [k_bg[i] for i in rs.permutation(len(k_bg))[:n_bg]])

        plano = ([(top_ent, "argmax_entropia"), (top_lc, "argmax_lc")]
                 + [(c, "sorteado") for c in sorteio])

        print(f"    {len(chaves)} superpixels · {len(k_obj)} tocam objeto · "
              f"avaliando {len(plano)} regioes + {len(novas)} imagens",
              flush=True)

        for (img, rid), papel in plano:
            info = por_img_scores[(img, rid)]
            try:
                fg0, bg0, H, W = base[img]
                fgx, bgx = _pixels_da_regiao(info["mask"], gts[img],
                                             a.px_regiao, rs)
                if not fgx and not bgx:
                    continue
                por_img = dict(base)
                por_img[img] = (fg0 + fgx, bg0 + bgx, H, W)
                md = os.path.join(work, f"c_{semente}_{img}_{rid}")
                enc = os.path.join(work, f"c_{semente}_{img}_{rid}.pth")
                _escrever(md, por_img)
                t0 = time.time()
                retrain_encoder(cfg["arch"], md, cfg["orig"], cfg["label"],
                                a.device, enc)
                t_tr = time.time() - t0
                m = _fbeta(enc, cfg, decoders, arq_val, a.device)
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"s{semente} {img}/{rid}: "
                              f"{type(e).__name__}: {e}")
                continue

            dp = m["labeled_marker"]["fb"] - fb0["labeled_marker"]
            kn = _kernels(enc)
            print(f"      {papel:<16} {info['estrato']:<6} "
                  f"ent={info['entropia']:.4f} fg={info['frac_fg']:.3f}  "
                  f"dFb_lm={dp:+.4f}  k={sum(kn)-sum(kn0):+d}  "
                  f"({t_tr:.1f}s)", flush=True)

            # entropia/lc/frac_fg viajam na `variante` porque a analise precisa
            # delas e o registro canonico nao tem coluna para escore de regiao.
            var = (f"regiao|{papel}|{info['estrato']}|"
                   f"ent={info['entropia']:.6f}|lc={info['lc']:.6f}|"
                   f"fg={info['frac_fg']:.6f}|"
                   f"k={'-'.join(str(x) for x in kn)}|{img}#{rid}")
            lote = [ev.execucao(
                experimento=EXPERIMENTO, dataset=a.ds, braco="al",
                variante=var, criterio=papel,
                decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
                orcamento=a.k, orcamento_px=len(fgx) + len(bgx),
                seed=semente, imagens=sel, marker_origem="sintetico",
                fb=m[dec]["fb"], dice=m[dec]["dice"], iou=m[dec].get("iou"),
                mae=m[dec]["mae"], segundos=round(t_tr, 2),
                segundos_treino=round(t_tr, 2), fonte=fonte,
            ) for dec, nome in decoders]
            ev.registrar(lote)
            registros.extend(lote)

        # ── o controle de escala: uma imagem nova inteira ───────────────────
        for nova in novas:
            try:
                lp = DS.caminho_label(a.ds, nova + ".png")
                if not lp:
                    continue
                d = generate_realistic_markers(
                    lp, n_fg_dabs=6, n_bg_dabs=14,
                    seed=OM._semente_marker(nova, semente))
                with Image.open(lp) as im:
                    W, H = im.size
                por_img = dict(base)
                por_img[nova] = (
                    [tuple(int(v) for v in p) for p in d["fg_seeds"]],
                    [tuple(int(v) for v in p) for p in d["bg_seeds"]], H, W)
                md = os.path.join(work, f"img_{semente}_{nova}")
                enc = os.path.join(work, f"img_{semente}_{nova}.pth")
                _escrever(md, por_img)
                t0 = time.time()
                retrain_encoder(cfg["arch"], md, cfg["orig"], cfg["label"],
                                a.device, enc)
                t_tr = time.time() - t0
                m = _fbeta(enc, cfg, decoders, arq_val, a.device)
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"s{semente} img {nova}: "
                              f"{type(e).__name__}: {e}")
                continue
            dp = m["labeled_marker"]["fb"] - fb0["labeled_marker"]
            kn = _kernels(enc)
            px = len(por_img[nova][0]) + len(por_img[nova][1])
            print(f"      {'imagem_nova':<16} {'-':<6} "
                  f"{'':<27}  dFb_lm={dp:+.4f}  "
                  f"k={sum(kn)-sum(kn0):+d}  ({t_tr:.1f}s)",
                  flush=True)
            lote = [ev.execucao(
                experimento=EXPERIMENTO, dataset=a.ds, braco="al",
                variante=f"imagem_nova|k={'-'.join(str(x) for x in kn)}|{nova}",
                criterio="imagem_nova",
                decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
                orcamento=a.k + 1, orcamento_px=px, seed=semente,
                imagens=sel + [nova], marker_origem="sintetico",
                fb=m[dec]["fb"], dice=m[dec]["dice"], iou=m[dec].get("iou"),
                mae=m[dec]["mae"], segundos=round(t_tr, 2),
                segundos_treino=round(t_tr, 2), fonte=fonte,
            ) for dec, nome in decoders]
            ev.registrar(lote)
            registros.extend(lote)

    if falhas:
        print(f"\n{len(falhas)} falha(s):")
        for f in falhas[:10]:
            print(f"  {f}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nregistrados: {len(registros)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrompido")
