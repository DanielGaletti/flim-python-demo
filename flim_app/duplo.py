#!/usr/bin/env python3
"""
duplo.py — comparação lado a lado: com AL e sem AL
==================================================
Dois braços com o MESMO orçamento de anotação. A única diferença é quem
escolhe as imagens:

    com AL   medoide na primeira rodada, CoreSet nas seguintes
    sem AL   sorteio — o controle correto

Uma advertência que precisa aparecer na tela, não só aqui: **uma execução
única não é evidência**. Medimos que o desvio de Fβ entre sorteios diferentes
é de 0.10 a 0.22, enquanto o ganho médio do AL sobre o acaso é de +0.012 com
orçamento pequeno. Numa rodada isolada o sorteio pode perfeitamente ganhar.

Por isso o braço sem AL roda com VÁRIOS sorteios e reporta pior/mediana/melhor.
Isso não é enfeite: é a diferença entre demonstrar um efeito e exibir uma
amostra favorável. E expõe a variância, que é um resultado por si só.

O braço com AL é determinístico (medoide + CoreSet), então uma execução basta —
e essa assimetria é justamente parte do argumento.
"""
from __future__ import annotations

import os
import shutil
import statistics as st
import tempfile
import time
from typing import Callable

import numpy as np
import torch

from flim_al.al_encoder_experiment import evaluate_decoder, retrain_encoder
from flim_al.coreset_badge import coreset_select, extract_encoder_features
from flim_al.marker_generator import save_markers
from flim_al.paper_selection import imagem_medoide
from flim_app import datasets as DS

N_SORTEIOS = 3          # quantos sorteios o braço sem AL executa


def proxima_al(cfg: dict, pool: list[str], anotadas: list[str],
               encoder: str | None, bloco: int) -> tuple[str, str]:
    """A imagem que o AL escolhe, e por quê."""
    restantes = [f for f in pool if f not in anotadas]
    if not restantes:
        return "", "pool esgotado"
    if not anotadas or encoder is None:
        return (imagem_medoide(restantes, cfg["orig"]),
                "a mais típica do pool, medida nos pixels — ainda não há "
                "modelo para consultar")
    enc = torch.load(encoder, map_location="cpu", weights_only=False)
    todas = anotadas + restantes
    feats = extract_encoder_features(
        enc, [os.path.join(cfg["orig"], f) for f in todas], bloco, "cpu")
    esc = coreset_select(feats, budget=1,
                         labeled_indices=list(range(len(anotadas))))
    return (todas[esc[0]],
            f"a mais distante das {len(anotadas)} já anotadas, no espaço de "
            "features (CoreSet)")


def proxima_sem(pool: list[str], anotadas: list[str],
                rng: np.random.Generator) -> tuple[str, str]:
    restantes = [f for f in pool if f not in anotadas]
    if not restantes:
        return "", "pool esgotado"
    return (str(rng.choice(restantes)),
            "sorteada — é o que se faz sem critério de seleção")


def escrever_markers(marcacoes: dict, orig: str, dest: str) -> int:
    """marcacoes: {img_sem_ext: {'fg': [[c,r]], 'bg': [...]}}"""
    from PIL import Image
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for img, m in marcacoes.items():
        if not m["fg"] and not m["bg"]:
            continue
        alvo = None
        for ext in (".png", ".jpg", ".jpeg"):
            if os.path.exists(os.path.join(orig, img + ext)):
                alvo = img + ext
                break
        if alvo is None:
            continue
        with Image.open(os.path.join(orig, alvo)) as im:
            w, h = im.size
        # Rede de seguranca: um seed fora do quadro derruba o treino inteiro
        # com um erro que nao diz nada ("not enough values to unpack"). O
        # pincel da interface ja recorta, mas markers tambem chegam de upload,
        # de sessao antiga e da pre-marcacao -- filtrar aqui cobre todos.
        dentro = lambda pts: [[c, r] for c, r in pts
                              if 0 <= c < w and 0 <= r < h]
        fg, bg = dentro(m["fg"]), dentro(m["bg"])
        fora = (len(m["fg"]) - len(fg)) + (len(m["bg"]) - len(bg))
        if fora:
            print(f"  [markers] {img}: {fora} seed(s) fora de {w}x{h} "
                  f"descartado(s)")
        if not fg and not bg:
            continue
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": h,
                      "W": w}, os.path.join(dest, f"{img}-seeds.txt"))
        n += 1
    return n


def treinar_e_avaliar(cfg: dict, marcacoes: dict, val: list[str],
                      decoder: str, bloco: int, work: str,
                      tag: str) -> dict:
    """Treina um braço com os markers desenhados e mede no mesmo val."""
    mdir = os.path.join(work, f"m_{tag}")
    n = escrever_markers(marcacoes, cfg["orig"], mdir)
    if n == 0:
        return {"erro": "nenhum marker desenhado"}
    enc = os.path.join(work, f"e_{tag}.pth")
    t0 = time.time()
    retrain_encoder(cfg["arch"], mdir, cfg["orig"], cfg["label"] or cfg["orig"],
                    "cpu", enc)
    t_tr = time.time() - t0
    out = {"encoder": enc, "n_imgs": n, "t_treino": round(t_tr, 1)}
    if cfg["label"] and val:
        t1 = time.time()
        m = evaluate_decoder(enc, decoder, bloco, val, cfg["orig"],
                             cfg["label"], "cpu",
                             area_range=tuple(cfg["area"]))
        out.update({"fb": round(m["fb"], 4), "iou": round(m["iou"], 4),
                    "t_aval": round(time.time() - t1, 1)})
    return out


def sem_al_multiplos(cfg: dict, pool: list[str], k: int, val: list[str],
                     decoder: str, bloco: int, ds_id: str, work: str,
                     n_sorteios: int = N_SORTEIOS) -> dict:
    """
    Roda o braço sem AL com vários sorteios, usando markers sintéticos.

    Markers sintéticos aqui são deliberados: pedir ao especialista para anotar
    3 conjuntos aleatórios só para medir a variância seria desrespeitoso com o
    tempo dele. O que se compara com o braço anotado à mão é a MEDIANA, e a
    dispersão vai junto para o leitor ver o tamanho do ruído.
    """
    from flim_al.realistic_markers import generate_realistic_markers
    if not cfg["label"]:
        return {"erro": "sem anotação de referência, não dá para sortear "
                        "vários braços automaticamente"}
    res = []
    for s in range(n_sorteios):
        rng = np.random.default_rng(100 + s)
        sel = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
        mdir = os.path.join(work, f"m_semal_{s}")
        shutil.rmtree(mdir, ignore_errors=True)
        os.makedirs(mdir, exist_ok=True)
        for i, f in enumerate(sel):
            lp = DS.caminho_label(ds_id, f)
            if not lp:
                continue
            d = generate_realistic_markers(lp, n_fg_dabs=6, n_bg_dabs=14,
                                           seed=1000 + 7 * s + i)
            save_markers(d, os.path.join(
                mdir, f"{os.path.splitext(f)[0]}-seeds.txt"))
        enc = os.path.join(work, f"e_semal_{s}.pth")
        retrain_encoder(cfg["arch"], mdir, cfg["orig"], cfg["label"], "cpu",
                        enc)
        m = evaluate_decoder(enc, decoder, bloco, val, cfg["orig"],
                             cfg["label"], "cpu",
                             area_range=tuple(cfg["area"]))
        res.append({"seed": s, "fb": round(m["fb"], 4),
                    "iou": round(m["iou"], 4), "encoder": enc,
                    "imagens": [os.path.splitext(x)[0] for x in sel]})
    fbs = [r["fb"] for r in res]
    ordenado = sorted(res, key=lambda r: r["fb"])
    return {"execucoes": res, "pior": ordenado[0], "melhor": ordenado[-1],
            "mediana": ordenado[len(ordenado) // 2],
            "fb_mediana": st.median(fbs),
            "amplitude": round(max(fbs) - min(fbs), 4)}
