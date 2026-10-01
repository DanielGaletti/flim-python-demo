#!/usr/bin/env python3
"""
artigo_vs_regiao.py — a tabela do artigo, e a mesma tabela com AL de região

A pergunta
    O artigo treina o FLIM com os traços que os especialistas A e B
    desenharam. Se o Active Learning escolhesse ONDE esses mesmos cliques
    caem — mesmas imagens, mesmo número de cliques, mesmo traço de fundo —
    a tabela muda?

    É seleção de REGIÃO dentro da imagem, não seleção de imagem. As imagens
    são as que o especialista anotou; não há escolha de imagem em nenhum dos
    dois braços.

Os dois braços

    artigo        os traços REAIS, exatamente como o especialista desenhou.
                  É a reprodução da tabela do artigo.
    al_regiao     mesmas imagens, mesmo traço de FUNDO copiado, e os cliques
                  de objeto realocados para os superpixels onde o modelo está
                  mais incerto — dentro do objeto, com a mesma contagem de
                  pixels.

    A diferença entre eles é, por construção, só a POSIÇÃO dos cliques de
    objeto. Tudo o mais é idêntico: as imagens, a contagem de pixels de
    objeto, a contagem de pixels de fundo e as posições de fundo.

Sobre a origem do traço
    O braço `artigo` tem `marker_origem = real`: é o traço humano intacto. O
    braço `al_regiao` tem `real_realocado`: a quantidade e as imagens vêm do
    especialista, a posição vem da máquina. A distinção fica no registro
    porque ela é real — e porque este projeto já produziu conclusão falsa
    comparando traço real com traço sintético sem perceber.

    O que torna a comparação válida mesmo assim: os dois braços gastam o
    MESMO número de cliques nas MESMAS imagens. O que varia é o tratamento.

Diferenças conhecidas contra o artigo
    1. Sem Dynamic Trees. O binário `iftSMansoniDelineation` é ELF de Linux e
       não executa nesta máquina. O pós-processamento é Otsu + filtro de área,
       igual nos dois braços.
    2. A avaliação usa uma amostra declarada do conjunto de validação Z₁\\T,
       não as 848 imagens inteiras — 848 × 7 decoders × 12 combinações levaria
       dez horas. A amostra é a MESMA em todos os braços e splits.

    Por isso os valores ABSOLUTOS não reproduzem a Tabela III. A comparação
    entre os dois braços é que é o objeto deste experimento.

Uso
    python scripts/artigo_vs_regiao.py --plano
    python scripts/artigo_vs_regiao.py --usuarios A B --splits 1 2 3
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

import clique as CLQ  # noqa: E402

EXPERIMENTO = "artigo_vs_regiao"

# Os sete decoders do artigo, na ordem em que a Tabela III os lista.
DECODERS = [
    ("labeled_marker",              "FLIM_lm"),
    ("decoder_2",                   "FLIM_pb"),
    ("decoder_3",                   "FLIM_mb"),
    ("decoder_attention",           "FLIM_at"),
    ("vanilla_adaptive_decoder",    "FLIM_ts"),
    ("hybrid_decoder",              "FLIM_lt"),
    ("vanilla_adaptive_decoder_wt", "FLIM_ts*"),
]

FLIM_AD = os.path.join(RAIZ, "flim_ad")
ORIG = os.path.join(FLIM_AD, "datasets/schistossoma-eggs/orig")
LABEL = os.path.join(FLIM_AD, "datasets/schistossoma-eggs/label")
SPLITS = os.path.join(FLIM_AD, "datasets/schistossoma-eggs/Splits-5train-70_30")
AREA = (1000, 9000)
BLOCO = 2


def _ler_seeds(caminho):
    """Lê um -seeds.txt e devolve (fg, bg, H, W) — o formato real do FLIM."""
    with open(caminho) as fh:
        cab = fh.readline().split()
        n, W, H = int(cab[0]), int(cab[1]), int(cab[2])
        fg, bg = [], []
        for _ in range(n):
            partes = fh.readline().split()
            if len(partes) < 4:
                continue
            c, r, cls = int(partes[0]), int(partes[1]), int(partes[3])
            (fg if cls == 1 else bg).append((c, r))
    return fg, bg, H, W


def _gt(img, h, w):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(LABEL, img + ext)
        if os.path.isfile(p):
            g = np.array(Image.open(p).convert("L")) > 127
            if g.shape != (h, w):
                g = np.array(Image.fromarray(g.astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            return g
    return None


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
    ap.add_argument("--usuarios", nargs="+", default=["A", "B"])
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n_val", type=int, default=250,
                    help="amostra declarada de Z1\\T; 848 inteiras levariam 10h")
    ap.add_argument("--semente_val", type=int, default=0)
    ap.add_argument("--n_segments", type=int, default=220)
    ap.add_argument("--decoders", nargs="+", default=None)
    ap.add_argument("--rotulo", default="artigo_vs_regiao")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    decoders = DECODERS if not a.decoders else [
        d for d in DECODERS if d[0] in a.decoders]

    n_cel = len(a.usuarios) * len(a.splits) * 3
    print(f"{len(a.usuarios)} usuários × {len(a.splits)} splits × 2 braços "
          f"= {n_cel} encoders")
    print(f"{n_cel * len(decoders)} avaliações de decoder, "
          f"em {a.n_val} imagens de Z1\\T")
    print("braços: artigo (traço real intacto) · al_regiao (realocado por "
          "incerteza) · aleatorio (realocado ao acaso — CONTROLE)")
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="avr_")
    fonte = f"{EXPERIMENTO}/val{a.n_val}/{a.rotulo}"
    print(f"fonte: {fonte}\n")

    registros, falhas = [], []
    for usuario in a.usuarios:
        for split in a.splits:
            mdir_real = os.path.join(FLIM_AD, "data", "schisto",
                                     f"user_{usuario}", f"split{split}",
                                     "markers")
            arch = os.path.join(FLIM_AD, "data", "schisto", f"user_{usuario}",
                                f"split{split}", "arch2D.json")
            if not os.path.isdir(mdir_real) or not os.path.isfile(arch):
                falhas.append(f"user {usuario} split {split}: sem markers/arch")
                continue

            base = {}
            for f in sorted(os.listdir(mdir_real)):
                if not f.endswith("-seeds.txt"):
                    continue
                img = f[:-len("-seeds.txt")]
                fg, bg, H, W = _ler_seeds(os.path.join(mdir_real, f))
                gt = _gt(img, H, W)
                if gt is None or not fg:
                    continue
                base[img] = (fg, bg, H, W, gt)
            if not base:
                falhas.append(f"user {usuario} split {split}: nenhum marker lido")
                continue
            treino = sorted(base)

            # Validação: Z1\T do split, menos as imagens de treino. Amostra
            # declarada, a MESMA para os dois braços.
            vl = os.path.join(SPLITS, f"split{split}-val.txt")
            val = [l.strip() for l in open(vl) if l.strip()]
            val = [v for v in val if os.path.splitext(v)[0] not in treino]
            rv = np.random.default_rng(a.semente_val)
            if len(val) > a.n_val:
                val = [val[i] for i in
                       sorted(rv.permutation(len(val))[:a.n_val])]

            # ── o encoder que dá a opinião do AL sai do braço do ARTIGO ────
            d0 = os.path.join(work, f"base_{usuario}{split}")
            _escrever(d0, {i: (f, b, H, W) for i, (f, b, H, W, _) in base.items()})
            enc0 = os.path.join(work, f"base_{usuario}{split}.pth")
            modelo = None
            try:
                retrain_encoder(arch, d0, ORIG, LABEL, a.device, enc0)
                from flim_app import server as SV
                SV.S["ds"] = "schisto"
                SV.S["block"] = BLOCO
                SV.S["decoder"] = "labeled_marker"
                SV.S["encoder"] = enc0
                modelo = SV._montar_modelo(enc0)
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"encoder base {usuario}{split}: {e}")

            # `aleatorio` e o controle que separa DUAS coisas que o desenho
            # anterior media juntas: realocar o clique (borda -> interior) e
            # a ESCOLHA do AL. Sem ele, um Delta negativo nao diz qual das
            # duas causou.
            for braco in ("artigo", "al_regiao", "aleatorio"):
                por_img = {}
                try:
                    for img, (fg0, bg0, H, W, gt) in base.items():
                        if braco == "artigo":
                            fg = fg0
                        elif braco == "aleatorio":
                            rr = np.random.default_rng(
                                abs(hash((usuario, split, img))) % (2**31))
                            fg = CLQ._cliques_em(
                                CLQ._centros_aleatorios(gt, rr), gt, len(fg0))
                        else:
                            if modelo is None:
                                raise RuntimeError("sem encoder base")
                            from skimage.segmentation import slic
                            arr = np.array(Image.open(
                                os.path.join(ORIG, img + ".png")))
                            if arr.ndim == 2:
                                arr = np.stack([arr] * 3, axis=2)
                            seg = slic(arr[:, :, :3], n_segments=a.n_segments,
                                       compactness=10.0, sigma=1.0,
                                       start_label=0,
                                       convert2lab=True).astype(np.int32)
                            prob = SV._mapa_prob(modelo, img)
                            if prob.shape != (H, W):
                                prob = np.array(Image.fromarray(
                                    (prob * 255).astype(np.uint8)).resize(
                                        (W, H), Image.BILINEAR)) / 255.0
                            fg = CLQ._cliques_em(
                                CLQ._centros_incerteza(gt, seg, prob),
                                gt, len(fg0))
                        por_img[img] = (fg, bg0, H, W)

                    mdir = os.path.join(work, f"m_{braco}_{usuario}{split}")
                    if _escrever(mdir, por_img) == 0:
                        raise ValueError("nenhum marker")
                    enc = os.path.join(work, f"e_{braco}_{usuario}{split}.pth")
                    t0 = time.time()
                    retrain_encoder(arch, mdir, ORIG, LABEL, a.device, enc)
                    t_treino = time.time() - t0
                except Exception as e:                        # noqa: BLE001
                    falhas.append(f"{braco} {usuario}{split}: "
                                  f"{type(e).__name__}: {e}")
                    print(f"  {usuario}{split} {braco:<10} FALHOU: {e}",
                          flush=True)
                    continue

                px_fg = sum(len(v[0]) for v in por_img.values())
                px_bg = sum(len(v[1]) for v in por_img.values())
                print(f"  user {usuario} split {split} {braco:<10} "
                      f"{len(treino)} imgs  fg={px_fg} bg={px_bg}  "
                      f"treino {t_treino:5.1f}s", flush=True)

                novos = []
                for dec, nome in decoders:
                    t1 = time.time()
                    try:
                        m = evaluate_decoder(enc, dec, BLOCO, val, ORIG,
                                             LABEL, a.device, area_range=AREA)
                    except Exception as e:                    # noqa: BLE001
                        falhas.append(f"{braco} {usuario}{split} {dec}: {e}")
                        continue
                    t_aval = time.time() - t1
                    novos.append(ev.execucao(
                        experimento=EXPERIMENTO, dataset="schisto",
                        usuario=usuario, split=split,
                        braco="flim" if braco == "artigo" else "al",
                        variante=f"{braco}_u{usuario}_s{split}",
                        criterio=braco,
                        decoder=dec, decoder_paper=nome, bloco=BLOCO,
                        orcamento=len(treino), orcamento_px=px_fg,
                        seed=split, imagens=treino,
                        marker_origem=("real" if braco == "artigo"
                                       else "real_realocado"),
                        hipotese="H-regiao-vs-acaso",
                        fb=m["fb"], dice=m["dice"], iou=m.get("iou"),
                        mae=m["mae"], segundos=round(t_treino + t_aval, 2),
                        segundos_treino=round(t_treino, 2),
                        segundos_aval=round(t_aval, 2),
                        fonte=fonte,
                    ))
                    print(f"      {nome:<9} fb={m['fb']:.4f} "
                          f"mae={m['mae']:.4f}", flush=True)
                if novos:
                    res = ev.registrar(novos)
                    registros.extend(novos)
                    if res["divergentes"]:
                        print(f"  DIVERGENCIA em {len(res['divergentes'])}",
                              flush=True)

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
