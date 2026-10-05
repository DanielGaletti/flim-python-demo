#!/usr/bin/env python3
"""
il_flim.py — refinamento interativo por cliques, no FLIM

O que isto acrescenta ao projeto
    Todas as campanhas anteriores mediram Fbeta com orcamento de anotacao
    FIXO. A pergunta da segmentacao interativa e outra: quantos cliques sao
    necessarios para chegar a uma qualidade alvo. E um eixo diferente, com
    moeda propria, e e a moeda da literatura de Interactive Learning.

    A metrica e NoC@X (number of clicks), usada por RITM (Sofiiuk et al.,
    2022, arXiv:2102.06583) e SimpleClick (Liu et al., 2023,
    arXiv:2210.11006): quantos cliques ate a IoU passar de X, com um teto de
    cliques. Imagem que nao chega ao alvo conta o teto, e e reportada
    separadamente, porque media que esconde fracasso nao informa.

O usuario simulado
    O protocolo e o de Xu et al. (2016), que a area adotou: o proximo clique
    vai ao centro do MAIOR erro. Erro aqui e a diferenca entre predicao e
    ground truth, separada em falso negativo (faltou objeto, clique de
    foreground) e falso positivo (sobrou objeto, clique de background). Entre
    os dois, escolhe-se a componente conexa de maior area.

    Dentro dessa componente o clique vai no ponto mais distante da borda dela,
    pela transformada de distancia. Isso evita colocar o clique na franja do
    erro, onde ele seria ambiguo, e e o que torna o usuario simulado
    reprodutivel: nao ha sorteio nenhum no laco.

Onde AL e CL entram
    O laco tem dois pontos de insercao, e e por isso que ele serve de
    arcabouco para os tres componentes:

        AL  pode reordenar ou filtrar os candidatos de clique, em vez de
            sempre pegar o maior erro
        CL  controla como o encoder incorpora o clique novo, pelo alpha de
            `cl_flim`

    Com alpha=1 o encoder e reconstruido a cada clique, que e o FLIM puro.
    Com alpha<1 o banco de filtros e protegido, e a pergunta e se isso reduz
    o NoC.

Limite declarado
    Clique simulado do ground truth NAO mede tempo de especialista. Ele mede
    numero de interacoes sob um usuario que acerta sempre, o que e um limite
    superior otimista de qualquer sistema real. Afirmar reducao de tempo
    humano exigiria estudo com pessoas.
"""
from __future__ import annotations

import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(RAIZ, "flim_ad", "libs", "flim-python") not in sys.path:
    sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))

RAIO = 5          # o pincel real do FLIM, medido nos 31 markers dos usuarios


# ── metricas ────────────────────────────────────────────────────────────────

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 1.0


# ── o usuario simulado ──────────────────────────────────────────────────────

def proximo_clique(pred: np.ndarray, gt: np.ndarray, vistos: set = None):
    """
    O centro do maior erro, pelo protocolo de Xu et al. (2016).

    Devolve (x, y, rotulo) com rotulo 1 para foreground e 0 para background,
    ou None quando nao ha erro. `vistos` evita repetir exatamente o mesmo
    pixel, o que travaria o laco quando um clique nao muda a predicao.
    """
    from scipy import ndimage

    p, g = pred.astype(bool), gt.astype(bool)
    fn = g & ~p                       # faltou objeto -> clique de foreground
    fp = p & ~g                       # sobrou objeto -> clique de background

    melhor = None
    for mascara, rotulo in ((fn, 1), (fp, 0)):
        if not mascara.any():
            continue
        lab, n = ndimage.label(mascara)
        if n == 0:
            continue
        areas = ndimage.sum(mascara, lab, index=range(1, n + 1))
        for idx in np.argsort(areas)[::-1]:
            comp = lab == (idx + 1)
            area = float(areas[idx])
            if melhor is not None and area <= melhor[0]:
                break
            # o ponto mais fundo da componente, nao a franja
            dt = ndimage.distance_transform_edt(comp)
            y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
            if vistos is not None and (int(x), int(y)) in vistos:
                continue
            melhor = (area, int(x), int(y), rotulo)
            break
    if melhor is None:
        return None
    return melhor[1], melhor[2], melhor[3]


def disco(cx: int, cy: int, forma, raio: int = RAIO):
    """Os pixels de um toque de pincel, recortados na imagem."""
    H, W = forma
    pts = []
    for dy in range(-raio, raio + 1):
        for dx in range(-raio, raio + 1):
            if dx * dx + dy * dy > raio * raio:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < W and 0 <= y < H:
                pts.append((x, y))
    return pts


# ── o laco ──────────────────────────────────────────────────────────────────

def laco(treinar, prever, markers_base: dict, gts: dict, imagem: str,
         alvo: float = 0.85, max_cliques: int = 20, alpha: float = 1.0,
         banco: str = None, raio: int = RAIO):
    """
    Refina uma imagem por cliques ate a IoU passar de `alvo`.

    `treinar(markers, alpha, banco) -> encoder` e
    `prever(encoder, imagem) -> mascara booleana` sao injetados para que este
    modulo nao dependa do caminho de treino: o mesmo laco serve ao FLIM puro,
    ao FLIM com banco protegido e a qualquer outro modelo que ofereca as duas
    operacoes.

    Devolve um dicionario com a curva de IoU por clique, o NoC e se o alvo foi
    atingido. `noc` vale `max_cliques` quando nao atinge, e `atingiu` diz
    qual dos dois casos e: media de NoC sem essa coluna esconde fracasso.
    """
    markers = {k: (list(v[0]), list(v[1]), v[2], v[3])
               for k, v in markers_base.items()}
    gt = gts[imagem]
    enc = treinar(markers, alpha, banco)
    pred = prever(enc, imagem)
    curva = [iou(pred, gt)]
    vistos = set()
    cliques = []

    for i in range(max_cliques):
        if curva[-1] >= alvo:
            break
        c = proximo_clique(pred, gt, vistos)
        if c is None:
            break
        x, y, rotulo = c
        vistos.add((x, y))
        pts = [(px, py) for px, py in disco(x, y, gt.shape, raio)
               if bool(gt[py, px]) == bool(rotulo)]
        if not pts:
            pts = [(x, y)]
        fg, bg, H, W = markers[imagem]
        (fg if rotulo else bg).extend(pts)
        markers[imagem] = (fg, bg, H, W)
        cliques.append({"n": i + 1, "x": x, "y": y, "rotulo": int(rotulo),
                        "px": len(pts)})
        enc = treinar(markers, alpha, banco)
        pred = prever(enc, imagem)
        curva.append(iou(pred, gt))

    atingiu = curva[-1] >= alvo
    noc = (next((c["n"] for c, v in zip(cliques, curva[1:]) if v >= alvo),
                len(cliques)) if atingiu else max_cliques)
    return {
        "imagem": imagem, "alvo": alvo, "curva": curva, "cliques": cliques,
        "noc": int(noc), "atingiu": bool(atingiu),
        "iou_inicial": curva[0], "iou_final": curva[-1],
        "alpha": alpha,
    }


def resumo(execucoes: list, alvo: float) -> dict:
    """
    Agrega uma lista de resultados de `laco`.

    `noc_medio` inclui as imagens que nao atingiram, contando o teto, e
    `taxa_sucesso` diz quantas atingiram. Os dois juntos: NoC baixo com taxa
    de sucesso baixa e pior que NoC alto com taxa alta, e so a primeira
    coluna nao mostraria isso.
    """
    if not execucoes:
        return {}
    nocs = [e["noc"] for e in execucoes]
    ok = [e for e in execucoes if e["atingiu"]]
    return {
        "n": len(execucoes), "alvo": alvo,
        "noc_medio": float(np.mean(nocs)),
        "noc_mediano": float(np.median(nocs)),
        "taxa_sucesso": len(ok) / len(execucoes),
        "noc_dos_que_atingiram": (float(np.mean([e["noc"] for e in ok]))
                                  if ok else None),
        "iou_inicial_medio": float(np.mean([e["iou_inicial"]
                                            for e in execucoes])),
        "iou_final_medio": float(np.mean([e["iou_final"]
                                          for e in execucoes])),
    }
