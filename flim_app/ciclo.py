#!/usr/bin/env python3
"""
ciclo.py — o laço AL + IL + CL em tempo real, sobre o FLIM

O que esta tela demonstra
    As outras telas do `flim_app` comparam braços e mostram tabelas. Esta
    mostra uma coisa só, e ao vivo: o que acontece com o modelo quando o
    especialista acrescenta anotação, e como isso muda conforme a proteção do
    banco de filtros.

    Com a plasticidade em 1 (o FLIM puro), a curva de IoU pode DESCER a cada
    clique. Medido em `il_noc`: na imagem 000455 do Schisto, doze cliques
    levaram a IoU de 0,614 para 0,621, ou seja, lugar nenhum. Com proteção em
    0,5, quatro cliques chegaram a 0,772.

    A tela existe para que isso seja visto acontecendo, não lido numa tabela.

Os três componentes, e onde cada um aparece
    AL  `sugerir` aponta a região de maior incerteza. O especialista decide se
        aceita.
    IL  `responder` e `clicar` incorporam a anotação e devolvem a IoU nova. A
        curva acumulada é o que a tela desenha.
    CL  `plasticidade` controla o alpha de `flim_al.cl_flim`. Pode ser mudado
        entre cliques, e é isso que torna a comparação visível na mesma
        sessão.

Aviso que a tela precisa mostrar
    A IoU aqui é calculada contra o ground truth do dataset, que existe porque
    isto é um ambiente de demonstração. Num uso real não haveria IoU para
    mostrar, e o especialista julgaria pela própria máscara. Os números desta
    tela NÃO entram na dissertação: para isso existem as campanhas, com
    semente, partição e registro. O `CLAUDE.md` do projeto diz isso em §2.6, e
    vale aqui.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

from flim_al import cl_flim, il_flim as IL
from flim_al.al_encoder_experiment import _official_filter_and_binarize
from flim_al.marker_generator import save_markers
from flim_al.realistic_markers import generate_realistic_markers
from flim_app import datasets as DS

# Estado do ciclo. Separado do `S` do servidor de proposito: a tela pode ser
# reiniciada sem derrubar o resto da aplicacao.
C = {
    "ativo": False,
    "img": None,
    "alpha": 0.5,
    "work": None,
    "banco": None,
    "enc": None,
    "markers": None,       # {img: (fg, bg, H, W)}
    "gt": None,
    "curva": [],           # [{"n", "iou", "alpha", "rotulo", "x", "y"}]
    "sugestao": None,      # {"x", "y", "ent", "frac_fg"}
    "n_segments": 150,
    "raio": IL.RAIO,
}


def _limpar_work():
    if C["work"]:
        shutil.rmtree(C["work"], ignore_errors=True)
    C["work"] = None


def _escrever(dest):
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    for img, (fg, bg, H, W) in C["markers"].items():
        if not fg and not bg:
            continue
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": H, "W": W},
                     os.path.join(dest, f"{img}-seeds.txt"))


def _treinar(SV, alpha, com_banco: bool):
    """
    Retreina e devolve (caminho do encoder, segundos).

    `com_banco=False` no primeiro treino: ali o banco e GRAVADO, para que os
    cliques seguintes tenham contra o que se proteger. Sem esse primeiro
    treino nao existe banco anterior, e qualquer alpha daria FLIM puro.
    """
    cfg = DS.resolver(SV.S["ds"])
    n = len(C["curva"])
    d = os.path.join(C["work"], f"m{n}")
    enc = os.path.join(C["work"], f"e{n}.pth")
    _escrever(d)
    t0 = time.time()
    cl_flim.treinar(
        cfg["arch"], d, C["orig"], cfg["label"], "cpu", enc,
        banco=(C["banco"] if com_banco else None),
        alpha=alpha,
        gravar_banco=(None if com_banco else C["banco"]),
    )
    return enc, time.time() - t0


def _prever(SV, enc):
    cfg = DS.resolver(SV.S["ds"])
    SV.S["encoder"] = enc
    modelo = SV._montar_modelo(enc)
    prob = SV._mapa_prob(modelo, C["img"])
    H, W = C["gt"].shape
    if prob.shape != (H, W):
        prob = np.array(Image.fromarray(
            (prob * 255).astype(np.uint8)).resize((W, H),
                                                  Image.BILINEAR)) / 255.0
    sal = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    mask = _official_filter_and_binarize(
        sal, area_range=tuple(cfg["area"])).astype(bool)
    return mask, prob


# ── acoes ───────────────────────────────────────────────────────────────────

def iniciar(SV, img: str = None, alpha: float = None):
    """Começa um ciclo: marcadores base, primeiro treino, IoU inicial."""
    ds = SV.S["ds"]
    cfg = DS.resolver(ds)
    if alpha is not None:
        C["alpha"] = float(alpha)

    if not img:
        for cand in [os.path.splitext(f)[0] for f in DS.imagens(ds)][:200]:
            lp = DS.caminho_label(ds, cand + ".png") or \
                 DS.caminho_label(ds, cand + ".jpg")
            if lp and (np.array(Image.open(lp).convert("L")) > 0).any():
                img = cand
                break
    if not img:
        return {"erro": "nenhuma imagem com objeto encontrada"}

    lp = DS.caminho_label(ds, img + ".png") or DS.caminho_label(ds, img + ".jpg")
    if not lp:
        return {"erro": f"sem ground truth para {img}"}

    _limpar_work()
    C["work"] = tempfile.mkdtemp(prefix="ciclo_")
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"))
    import tabela_k_por_modelo as TK
    C["orig"] = TK._orig_em_png(dict(cfg, _id=ds), ds, {img}, C["work"])

    gt = np.array(Image.open(lp).convert("L")) > 127
    d = generate_realistic_markers(lp, n_fg_dabs=6, n_bg_dabs=14, seed=0)
    C.update({
        "ativo": True, "img": img, "gt": gt, "curva": [], "sugestao": None,
        "banco": os.path.join(C["work"], "banco.pkl"),
        "markers": {img: ([tuple(int(v) for v in p) for p in d["fg_seeds"]],
                          [tuple(int(v) for v in p) for p in d["bg_seeds"]],
                          gt.shape[0], gt.shape[1])},
    })

    enc, seg = _treinar(SV, 1.0, com_banco=False)
    C["enc"] = enc
    mask, _ = _prever(SV, enc)
    C["curva"].append({"n": 0, "iou": IL.iou(mask, gt), "alpha": None,
                       "rotulo": None, "x": None, "y": None,
                       "segundos": round(seg, 2)})
    return estado(SV)


def plasticidade(SV, valor: float):
    """
    Muda o alpha entre cliques.

    De proposito: a comparacao fica visivel na MESMA sessao, sobre a mesma
    imagem e o mesmo histórico de cliques. Cada ponto da curva registra o
    alpha com que foi produzido, para que a tela possa marcar onde mudou.
    """
    C["alpha"] = max(0.0, min(1.0, float(valor)))
    return {"alpha": C["alpha"]}


def sugerir(SV):
    """O Active Learning aponta a região de maior incerteza."""
    if not C["ativo"]:
        return {"erro": "ciclo nao iniciado"}
    from skimage.segmentation import slic
    from flim_al.region_al import score_regions_by_entropy

    cfg = DS.resolver(SV.S["ds"])
    arr = np.array(Image.open(os.path.join(
        C["orig"], _arquivo(C["orig"], C["img"]))))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    seg = slic(arr[:, :, :3], n_segments=C["n_segments"], compactness=10.0,
               sigma=1.0, start_label=0, convert2lab=True).astype(np.int32)
    _, prob = _prever(SV, C["enc"])
    ent = score_regions_by_entropy(seg, prob)

    # Regioes ja anotadas saem da lista: sugerir de novo o mesmo lugar e o
    # modo mais rapido de irritar quem esta anotando.
    fg, bg, H, W = C["markers"][C["img"]]
    usados = set(fg) | set(bg)
    melhor, melhor_s = None, -1.0
    for rid in np.unique(seg):
        m = seg == rid
        if m.sum() < 20:
            continue
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        if (cx, cy) in usados:
            continue
        s = float(ent[int(rid)])
        if s > melhor_s:
            melhor_s, melhor = s, (cx, cy, int(rid), m)
    if melhor is None:
        return {"erro": "nenhuma regiao disponivel"}
    cx, cy, rid, m = melhor
    C["sugestao"] = {
        "x": cx, "y": cy, "rid": rid,
        "ent": round(melhor_s, 6),
        # `frac_fg` vem do ground truth e serve de GABARITO da tela, nao de
        # entrada do metodo: e o que permite mostrar se a pessoa acertou o
        # rotulo. Num uso real nao existiria.
        "frac_fg": round(float(C["gt"][m].mean()), 4),
    }
    return {"sugestao": C["sugestao"], "alpha": C["alpha"]}


def responder(SV, rotulo: int):
    """O especialista diz se a região sugerida é objeto (1) ou fundo (0)."""
    if not C["sugestao"]:
        return {"erro": "nenhuma sugestao aberta"}
    s = C["sugestao"]
    return clicar(SV, s["x"], s["y"], rotulo, origem="al")


def clicar(SV, x: int, y: int, rotulo: int, origem: str = "manual"):
    """Incorpora um toque de pincel e retreina com a plasticidade corrente."""
    if not C["ativo"]:
        return {"erro": "ciclo nao iniciado"}
    gt = C["gt"]
    pts = IL.disco(int(x), int(y), gt.shape, C["raio"])
    if not pts:
        return {"erro": "clique fora da imagem"}
    fg, bg, H, W = C["markers"][C["img"]]
    (fg if int(rotulo) else bg).extend(pts)
    C["markers"][C["img"]] = (fg, bg, H, W)

    enc, seg = _treinar(SV, C["alpha"], com_banco=True)
    C["enc"] = enc
    mask, _ = _prever(SV, enc)
    anterior = C["curva"][-1]["iou"] if C["curva"] else None
    novo = IL.iou(mask, gt)
    C["curva"].append({
        "n": len(C["curva"]), "iou": novo, "alpha": C["alpha"],
        "rotulo": int(rotulo), "x": int(x), "y": int(y), "origem": origem,
        "px": len(pts), "segundos": round(seg, 2),
        "delta": None if anterior is None else round(novo - anterior, 6),
    })
    C["sugestao"] = None
    return estado(SV)


def estado(SV):
    if not C["ativo"]:
        return {"ativo": False}
    ious = [p["iou"] for p in C["curva"]]
    cliques = [p for p in C["curva"] if p["n"] > 0]
    pioraram = [p for p in cliques if p.get("delta") is not None
                and p["delta"] < 0]
    return {
        "ativo": True, "img": C["img"], "alpha": C["alpha"],
        "curva": C["curva"],
        "iou_inicial": ious[0], "iou_atual": ious[-1],
        "n_cliques": len(cliques),
        # A contagem de cliques que PIORARAM e o numero que esta tela existe
        # para mostrar. Com plasticidade 1 ele sobe.
        "cliques_que_pioraram": len(pioraram),
        "melhor_iou": max(ious), "sugestao": C["sugestao"],
        "px_anotados": sum(len(v[0]) + len(v[1])
                           for v in C["markers"].values()),
    }


def encerrar(SV):
    _limpar_work()
    C.update({"ativo": False, "img": None, "curva": [], "sugestao": None,
              "markers": None, "gt": None, "enc": None, "banco": None})
    return {"ativo": False}


def _arquivo(pasta: str, img: str) -> str:
    for ext in (".png", ".jpg", ".jpeg"):
        if os.path.exists(os.path.join(pasta, img + ext)):
            return img + ext
    return img + ".png"
