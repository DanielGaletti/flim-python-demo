#!/usr/bin/env python3
"""
cl_plasticidade.py — o acréscimo de anotação deixa de ser perigoso?

A pergunta
    A campanha `ganho_marginal` mediu que acrescentar ~300 px de anotação na
    região errada derruba o Fβ de validação em até 0,50, chegando a levar um
    encoder de 0,78 para 0,00. Esquecimento catastrófico, num modelo sem
    backpropagação.

    `flim_al/cl_flim.py` propõe a correção: interpolar o banco de filtros em
    vez de reescrevê-lo, com uma taxa de plasticidade α. Este experimento mede
    se existe α que torne o acréscimo seguro sem matar o aprendizado.

O desenho
    Dentro de cada semente, tudo é idêntico ao `ganho_marginal`: partição,
    imagens, marcadores base, encoder inicial, conjunto de validação. Para
    cada candidato de região, varia-se SÓ o α.

    α = 1 é o FLIM puro, e serve de cruzamento: os Fβ têm de bater com os da
    campanha `ganho_marginal` dentro do ruído entre execuções já medido
    (0,0034). Divergência maior que isso é erro, não resultado.

Os quatro candidatos
    argmax_entropia   o que o Active Learning escolhe, sem ground truth
    pior_sorteada     a sorteada que mais derruba o Fβ  (usa GT: DIAGNÓSTICO)
    melhor_sorteada   a que mais sobe                   (usa GT: DIAGNÓSTICO)
    sorteada          uma qualquer, o caso médio

    Os dois do meio existem para medir os extremos do dano e do ganho. Eles
    leem o Fβ da validação para serem escolhidos, então não são estratégias.

Uso
    python scripts/cl_plasticidade.py --plano
    python scripts/cl_plasticidade.py --ds schisto --sementes 0 1 2
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

from flim_al import cl_flim  # noqa: E402
from flim_al import evidencia as ev  # noqa: E402
from flim_al.al_encoder_experiment import evaluate_decoder  # noqa: E402
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

import ganho_marginal as GM  # noqa: E402
import onde_marcar as OM  # noqa: E402

EXPERIMENTO = "cl_plasticidade"

DECODERS = [("labeled_marker", "FLIM_lm"), ("decoder_2", "FLIM_pb")]
ALPHAS = [0.0, 0.5, 1.0]

PARTICAO = {
    "schisto":     dict(n_pool=40, n_val=40, n_test=60),
    "brats":       dict(n_pool=40, n_val=40, n_test=60),
    "conjunctiva": dict(n_pool=20, n_val=25, n_test=30),
}


def _fb(enc, cfg, decoders, arquivos, device):
    out = {}
    for dec, _ in decoders:
        out[dec] = evaluate_decoder(enc, dec, cfg["bloco"], arquivos,
                                    cfg["orig"], cfg["label"], device,
                                    area_range=tuple(cfg["area"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sementes", nargs="+", type=int,
                    default=list(range(10)))
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--n_sorteadas", type=int, default=8,
                    help="quantas regioes sorteadas examinar para achar a "
                         "pior e a melhor")
    ap.add_argument("--n_segments", type=int, default=150)
    ap.add_argument("--px_regiao", type=int, default=300)
    ap.add_argument("--decoders", nargs="+", default=None)
    ap.add_argument("--rotulo", default="cl")
    ap.add_argument("--registro", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    if a.registro:
        ev.ARQUIVO = os.path.abspath(a.registro)
        os.makedirs(os.path.dirname(ev.ARQUIVO), exist_ok=True)
        print(f"registro redirecionado para {ev.ARQUIVO}")

    decoders = DECODERS if not a.decoders else [
        d for d in DECODERS if d[0] in a.decoders]
    part = PARTICAO[a.ds]

    cfg = DS.resolver(a.ds)
    cfg["_id"] = a.ds
    cfg["bloco"] = cfg.get("bloco", 2)

    todas = [os.path.splitext(f)[0] for f in DS.imagens(a.ds)]
    rng0 = np.random.default_rng(a.semente_particao)
    com_obj = [i for i in todas[:600] if GM._tem_objeto(a.ds, i)]
    emb = rng0.permutation(len(com_obj))
    np_, nv = part["n_pool"], part["n_val"]
    pool = sorted(com_obj[i] for i in emb[:np_])
    val = sorted(com_obj[i] for i in emb[np_:np_ + nv])

    n_cand = 1 + a.n_sorteadas
    print(f"dataset {a.ds} · pool {len(pool)} · val {len(val)} "
          f"(teste NAO tocado)")
    print(f"K={a.k} · {len(a.sementes)} sementes · {n_cand} candidatos "
          f"explorados · alphas {a.alphas}")
    print(f"= {len(a.sementes)} * (1 + {n_cand} + 4*{len(a.alphas)}) "
          f"~ {len(a.sementes) * (1 + n_cand + 4 * len(a.alphas))} treinos")
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="clpl_")
    import tabela_k_por_modelo as TK
    cfg["orig"] = TK._orig_em_png(cfg, a.ds, set(pool) | set(val), work)
    arq_val = [OM._arquivo(cfg, i) for i in val]
    fonte = f"{EXPERIMENTO}/pool{len(pool)}/val{len(val)}/{a.rotulo}"
    print(f"fonte: {fonte}\n")

    registros, falhas = [], []
    for semente in a.sementes:
        r = np.random.default_rng(semente * 7717 + 3)
        sel = [str(x) for x in r.permutation(pool)][:a.k]

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
            falhas.append(f"s{semente}: sem imagem base")
            continue

        # ── E0 e o banco de filtros inicial ────────────────────────────────
        d0 = os.path.join(work, f"base_{semente}")
        enc0 = os.path.join(work, f"base_{semente}.pth")
        banco0 = os.path.join(work, f"banco_{semente}.pkl")
        try:
            GM._escrever(d0, base)
            t0 = time.time()
            cl_flim.treinar(cfg["arch"], d0, cfg["orig"], cfg["label"],
                            a.device, enc0, gravar_banco=banco0)
            t_base = time.time() - t0
            m0 = _fb(enc0, cfg, decoders, arq_val, a.device)
        except Exception as e:                                # noqa: BLE001
            falhas.append(f"s{semente} base: {type(e).__name__}: {e}")
            print(f"  s{semente} BASE FALHOU: {e}", flush=True)
            continue
        fb0 = {d: m0[d]["fb"] for d, _ in decoders}
        print(f"  s{semente} base {t_base:5.1f}s  "
              + "  ".join(f"{n}={fb0[d]:.4f}" for d, n in decoders),
              flush=True)

        novos = [ev.execucao(
            experimento=EXPERIMENTO, dataset=a.ds, braco="flim",
            variante="base", criterio="base", decoder=dec, decoder_paper=nome,
            bloco=cfg["bloco"], orcamento=a.k, orcamento_px=0, seed=semente,
            imagens=sel, marker_origem="sintetico",
            fb=m0[dec]["fb"], dice=m0[dec]["dice"], iou=m0[dec].get("iou"),
            mae=m0[dec]["mae"], segundos=round(t_base, 2),
            segundos_treino=round(t_base, 2), fonte=fonte,
        ) for dec, nome in decoders]
        ev.registrar(novos)
        registros.extend(novos)

        # ── candidatos de regiao, pelo mesmo caminho do ganho_marginal ─────
        try:
            from flim_al.region_al import score_regions_by_entropy
            from flim_app import server as SV
            SV.S["ds"] = a.ds
            SV.S["block"] = cfg["bloco"]
            SV.S["decoder"] = "labeled_marker"
            SV.S["encoder"] = enc0
            modelo = SV._montar_modelo(enc0)
            info = {}
            for img in base:
                gt = gts[img]
                H, W = gt.shape
                seg = GM._slic(
                    os.path.join(cfg["orig"], OM._arquivo(cfg, img)),
                    a.n_segments)
                prob = SV._mapa_prob(modelo, img)
                if prob.shape != (H, W):
                    prob = np.array(Image.fromarray(
                        (prob * 255).astype(np.uint8)).resize(
                            (W, H), Image.BILINEAR)) / 255.0
                ent = score_regions_by_entropy(seg, prob)
                for rid in np.unique(seg):
                    mask = seg == rid
                    if mask.sum() < 20:
                        continue
                    info[(img, int(rid))] = dict(
                        mask=mask, ent=float(ent[int(rid)]),
                        frac=float(gt[mask].mean()))
        except Exception as e:                                # noqa: BLE001
            falhas.append(f"s{semente} candidatos: {e}")
            continue

        chaves = list(info)
        top_ent = max(chaves, key=lambda c: info[c]["ent"])
        rs = np.random.default_rng(semente * 331 + 7)
        k_obj = [c for c in chaves if info[c]["frac"] > 0]
        k_bg = [c for c in chaves if info[c]["frac"] == 0]
        metade = max(1, a.n_sorteadas // 2)
        sorteadas = ([k_obj[i] for i in
                      rs.permutation(len(k_obj))[:metade]]
                     + [k_bg[i] for i in
                        rs.permutation(len(k_bg))[:a.n_sorteadas - metade]])

        def _treina(chave, alpha, tag, so_primeiro=False):
            """
            Acrescenta a regiao `chave` e retreina com plasticidade alpha.

            `so_primeiro` avalia apenas o primeiro decoder. A fase de
            exploracao usa so o `labeled_marker` para decidir qual regiao e a
            pior e qual e a melhor, entao avaliar os dois ali era metade do
            custo jogada fora.
            """
            img, rid = chave
            fg0, bg0, H, W = base[img]
            fgx, bgx = GM._pixels_da_regiao(info[chave]["mask"], gts[img],
                                            a.px_regiao, rs)
            if not fgx and not bgx:
                return None, None, 0
            por = dict(base)
            por[img] = (fg0 + fgx, bg0 + bgx, H, W)
            md = os.path.join(work, f"c_{semente}_{tag}_{alpha}")
            enc = os.path.join(work, f"c_{semente}_{tag}_{alpha}.pth")
            GM._escrever(md, por)
            t = time.time()
            cl_flim.treinar(cfg["arch"], md, cfg["orig"], cfg["label"],
                            a.device, enc, banco=banco0, alpha=alpha)
            quais = decoders[:1] if so_primeiro else decoders
            return (_fb(enc, cfg, quais, arq_val, a.device),
                    time.time() - t, len(fgx) + len(bgx))

        # ── exploracao: achar a pior e a melhor sorteada com alpha=1 ───────
        fb_sort = {}
        for chave in sorteadas:
            try:
                m, t, px = _treina(chave, 1.0, f"exp{chave[1]}",
                                   so_primeiro=True)
                if m is None:
                    continue
                fb_sort[chave] = m["labeled_marker"]["fb"]
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"s{semente} exp {chave}: {e}")
        if not fb_sort:
            falhas.append(f"s{semente}: nenhuma sorteada avaliada")
            continue
        pior = min(fb_sort, key=fb_sort.get)
        melhor = max(fb_sort, key=fb_sort.get)
        qualquer = sorteadas[0]
        print(f"    explorado {len(fb_sort)}: pior={fb_sort[pior]:.4f} "
              f"melhor={fb_sort[melhor]:.4f} (base {fb0['labeled_marker']:.4f})",
              flush=True)

        plano = [(top_ent, "argmax_entropia"), (pior, "pior_sorteada"),
                 (melhor, "melhor_sorteada"), (qualquer, "sorteada")]

        # ── a curva de plasticidade ────────────────────────────────────────
        for chave, papel in plano:
            for alpha in a.alphas:
                try:
                    m, t, px = _treina(chave, alpha, f"{papel}_{chave[1]}")
                    if m is None:
                        continue
                except Exception as e:                        # noqa: BLE001
                    falhas.append(f"s{semente} {papel} a={alpha}: "
                                  f"{type(e).__name__}: {e}")
                    continue
                dp = m["labeled_marker"]["fb"] - fb0["labeled_marker"]
                print(f"      {papel:<16} a={alpha:<5} "
                      f"dFb_lm={dp:+.4f}  ({t:.1f}s)", flush=True)
                var = (f"{papel}|alpha={alpha:.2f}|"
                       f"ent={info[chave]['ent']:.6f}|"
                       f"fg={info[chave]['frac']:.6f}|{chave[0]}#{chave[1]}")
                lote = [ev.execucao(
                    experimento=EXPERIMENTO, dataset=a.ds, braco="al",
                    variante=var, criterio=papel, decoder=dec,
                    decoder_paper=nome, bloco=cfg["bloco"], orcamento=a.k,
                    orcamento_px=px, seed=semente, imagens=sel,
                    marker_origem="sintetico",
                    fb=m[dec]["fb"], dice=m[dec]["dice"],
                    iou=m[dec].get("iou"), mae=m[dec]["mae"],
                    segundos=round(t, 2), segundos_treino=round(t, 2),
                    fonte=fonte,
                ) for dec, nome in decoders]
                res = ev.registrar(lote)
                registros.extend(lote)
                if res["divergentes"]:
                    print(f"      DIVERGENCIA em "
                          f"{len(res['divergentes'])} run_id", flush=True)

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
