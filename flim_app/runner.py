#!/usr/bin/env python3
"""
runner.py — comparação de critérios de AL, em lote, com tempo medido
====================================================================
Roda o produto (critério × K) e devolve tabela e curva.

O tempo é medido e reportado porque é metade do argumento do FLIM: uma rede
que estima kernels por k-means treina em segundos, e isso só é uma vantagem se
alguém mostrar o número. Separamos treino de teste — o treino é o que o FLIM
promete ser barato; o teste custa o que custa em qualquer rede.

Os critérios implementados aqui não consultam anotação nenhuma:

    coreset   k-center guloso: as K imagens mais distantes entre si no espaço
              de features do encoder. Cobertura, não incerteza.
    entropy   maior entropia binária média do mapa de saliência.
    lc        least confidence: menor distância a 0.5 (para binário é
              equivalente a margin).
    badge     k-means++ sobre embeddings de gradiente.
    medoide   determinístico, direto dos pixels — sem modelo.
    random    controle. Sem ele não dá para dizer se um critério faz algo.

Para o AL, `label/` é dispensável. Ele só é usado para calcular Fβ ao final —
e quando não existe, a avaliação passa a ser o julgamento do especialista.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent

from flim_al.al_encoder_experiment import (  # noqa: E402
    _official_filter_and_binarize, evaluate_decoder, load_input,
    retrain_encoder,
)
from flim_al.coreset_badge import (  # noqa: E402
    badge_select, coreset_select, extract_encoder_features,
)
from flim_al.paper_selection import imagem_medoide  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

CRITERIOS = [
    ("coreset", "CoreSet", "cobertura do espaço de features"),
    ("entropy", "Entropia", "maior incerteza média"),
    ("lc", "Least confidence", "menor confiança na predição"),
    ("badge", "BADGE", "gradiente × diversidade"),
    ("medoide", "Medoide", "as mais típicas, sem modelo"),
    ("random", "Aleatório", "controle"),
]


@torch.no_grad()
def _escores(enc_path: str, orig: str, fnames: list[str], bloco: int,
             modo: str) -> np.ndarray:
    """Escore por imagem para os critérios baseados em saliência."""
    from pyflim import layers
    m = torch.load(enc_path, map_location="cpu", weights_only=False)
    m.device = "cpu"
    for l in range(m.architecture.nlayers):
        ml = getattr(m.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            m.layers[l].marker_labels = ml.to("cpu")
    m.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type="labeled_marker",
        multi_layer=False)
    eps, out = 1e-6, []
    for f in fnames:
        x, _, _ = load_input(os.path.join(orig, f))
        y, _ = m.forward(x, decoder_layer=[bloco - 1])
        p = y[0].float()
        if p.dim() == 3:
            p = p.unsqueeze(0)
        p = np.clip(p.squeeze().numpy() / 255.0, eps, 1 - eps)
        if modo == "entropy":
            out.append(float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean()))
        else:                                   # least confidence
            out.append(float(1.0 - np.abs(p - 0.5).mean() * 2))
    return np.array(out)


def selecionar(criterio: str, k: int, cfg: dict, pool: list[str],
               enc_semente: str | None, rng: np.random.Generator) -> list[str]:
    """As K imagens que o critério manda anotar. Nenhuma consulta a `label/`."""
    if criterio == "random":
        return sorted(rng.choice(pool, size=min(k, len(pool)),
                                 replace=False).tolist())
    if criterio == "medoide" or enc_semente is None:
        # sem encoder não há espaço de features; começa pelo medoide e
        # completa por distância nos próprios pixels
        sel = [imagem_medoide(pool, cfg["orig"])]
        while len(sel) < k and len(sel) < len(pool):
            resto = [f for f in pool if f not in sel]
            sel.append(imagem_medoide(resto, cfg["orig"]))
        return sel
    if criterio in ("entropy", "lc"):
        e = _escores(enc_semente, cfg["orig"], pool, cfg["bloco"], criterio)
        return [pool[i] for i in np.argsort(-e)[:k]]

    enc = torch.load(enc_semente, map_location="cpu", weights_only=False)
    caminhos = [os.path.join(cfg["orig"], f) for f in pool]
    if criterio == "badge":
        feats = extract_encoder_features(enc, caminhos, cfg["bloco"], "cpu")
        return [pool[i] for i in badge_select(feats, budget=k)]
    feats = extract_encoder_features(enc, caminhos, cfg["bloco"], "cpu")
    i0 = pool.index(imagem_medoide(pool, cfg["orig"]))
    idx = [i0] + list(coreset_select(feats, budget=max(0, k - 1),
                                     labeled_indices=[i0]))
    return [pool[i] for i in idx[:k]]


def _markers_sinteticos(cfg: dict, ds_id: str, sel: list[str], dest: str,
                        rng: np.random.Generator) -> int:
    """
    Gera markers a partir da anotação, simulando o especialista.

    Só existe para o modo em lote, onde ninguém está desenhando. É uma
    simulação declarada: no modo interativo os markers vêm do usuário.
    """
    from flim_al.realistic_markers import generate_realistic_markers
    from flim_al.marker_generator import save_markers
    os.makedirs(dest, exist_ok=True)
    n = 0
    for i, f in enumerate(sel):
        lp = DS.caminho_label(ds_id, f)
        if not lp:
            continue
        d = generate_realistic_markers(lp, n_fg_dabs=6, n_bg_dabs=14,
                                       seed=int(rng.integers(1 << 30)) + i)
        save_markers(d, os.path.join(dest, f"{os.path.splitext(f)[0]}-seeds.txt"))
        n += 1
    return n


def executar(ds_id: str, criterios: list[str], ks: list[int], decoder: str,
             n_test: int, semente: int, progresso: Callable[[dict], None],
             cancelado: Callable[[], bool]) -> list[dict]:
    """
    Roda o produto critério × K. Chama `progresso` a cada ponto concluído.
    """
    cfg = DS.resolver(ds_id)
    if not cfg["label"]:
        raise ValueError("este dataset não tem `label/`; a comparação em lote "
                         "precisa dele para calcular Fβ. Use o modo "
                         "interativo.")
    todas = DS.imagens(ds_id)
    rng = np.random.default_rng(semente)

    # Pool e teste restritos a imagens COM objeto — e isto não é conveniência.
    #
    # No Schisto 49% das imagens não têm ovo. Se entrarem no pool, o gerador
    # produz só traços de fundo, o encoder colapsa e o Fβ trava na fração de
    # imagens vazias (0.425 no primeiro teste desta função, idêntico para todo
    # K). Se entrarem no teste, o Fβ é 1.0 nelas por definição (vazio-vazio) e
    # a curva não se move quando o modelo melhora.
    #
    # Em ambos os casos a tabela mediria a patologia do dataset em vez do
    # critério de seleção. O recorte está declarado na interface.
    com_obj = []
    for f in todas:
        lp = DS.caminho_label(ds_id, f)
        if lp and (np.array(Image.open(lp).convert("L")) > 0).any():
            com_obj.append(f)
        if len(com_obj) >= 400:
            break
    base = com_obj or todas

    emb = rng.permutation(len(base))
    test = [base[i] for i in emb[:min(n_test, max(4, len(base) // 3))]]
    pool = [base[i] for i in emb[len(test):len(test) + 60]]
    if len(pool) < max(ks):
        raise ValueError(f"pool de {len(pool)} imagens é menor que o maior "
                         f"K pedido ({max(ks)})")

    work = tempfile.mkdtemp(prefix="runner_")
    resultados: list[dict] = []
    try:
        # Encoder-semente: uma imagem, o medoide. É o que existe na primeira
        # rodada de um AL real — o critério não pode depender de um modelo que
        # ainda não foi construído.
        d0 = os.path.join(work, "m0")
        _markers_sinteticos(cfg, ds_id, [imagem_medoide(pool, cfg["orig"])],
                            d0, rng)
        e0 = os.path.join(work, "e0.pth")
        retrain_encoder(cfg["arch"], d0, cfg["orig"], cfg["label"], "cpu", e0)

        total = len(criterios) * len(ks)
        feito = 0
        for crit in criterios:
            for k in sorted(ks):
                if cancelado():
                    return resultados
                try:
                    t_sel = time.time()
                    sel = selecionar(crit, k, cfg, pool, e0, rng)
                    t_sel = time.time() - t_sel

                    mdir = os.path.join(work, f"m_{crit}_{k}")
                    shutil.rmtree(mdir, ignore_errors=True)
                    n_mk = _markers_sinteticos(cfg, ds_id, sel, mdir, rng)
                    if n_mk == 0:
                        raise ValueError("nenhum marker gerado")

                    t_tr = time.time()
                    enc = os.path.join(work, "e.pth")
                    retrain_encoder(cfg["arch"], mdir, cfg["orig"],
                                    cfg["label"], "cpu", enc)
                    t_tr = time.time() - t_tr

                    t_te = time.time()
                    m = evaluate_decoder(enc, decoder, cfg["bloco"], test,
                                         cfg["orig"], cfg["label"], "cpu",
                                         area_range=tuple(cfg["area"]))
                    t_te = time.time() - t_te

                    r = {
                        "criterio": crit, "k": k,
                        "fb": round(m["fb"], 4), "iou": round(m["iou"], 4),
                        "dice": round(m["dice"], 4), "mae": round(m["mae"], 4),
                        "t_selecao": round(t_sel, 2),
                        "t_treino": round(t_tr, 2),
                        "t_teste": round(t_te, 2),
                        "ms_por_imagem": round(1000 * t_te / max(1, len(test)), 1),
                        "imagens": "|".join(os.path.splitext(x)[0] for x in sel),
                    }
                    resultados.append(r)
                except Exception as e:
                    traceback.print_exc()
                    resultados.append({"criterio": crit, "k": k,
                                       "erro": str(e)[:160]})
                feito += 1
                progresso({"feito": feito, "total": total,
                           "resultados": resultados,
                           "n_test": len(test), "n_pool": len(pool)})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return resultados
