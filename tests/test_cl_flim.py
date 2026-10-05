"""
O mecanismo de aprendizado contínuo do FLIM tem de satisfazer três coisas.

1. REIMPLEMENTAÇÃO FIEL. `cl_flim.treinar` sem banco precisa dar exatamente o
   mesmo Fβ que `retrain_encoder`. Ele existe só porque a interceptação exige
   o objeto modelo antes do `fit`; se os números divergirem, ele não é o mesmo
   treino e nada medido com ele se compara com as campanhas registradas.

2. INTERCEPTAÇÃO REAL. O banco gravado precisa ter uma entrada por camada. Sem
   isso o k-means final não foi interceptado, e os testes de α passariam por
   vacuidade.

3. α=0 É RETENÇÃO TOTAL. Com marcadores NOVOS e α=0, o encoder resultante tem
   de ser idêntico ao anterior. É o teste que de fato prova que o esquecimento
   foi bloqueado, e é o único que falharia se o casamento de centróides
   estivesse errado.

O teste é marcado como lento: ele treina encoders de verdade.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

pytestmark = pytest.mark.lento


def _monta(tmp, n_extra=0):
    """Marcadores de 2 imagens do Schisto; `n_extra` pixels de objeto a mais."""
    from PIL import Image
    from flim_al.marker_generator import save_markers
    from flim_al.realistic_markers import generate_realistic_markers
    from flim_app import datasets as DS
    import onde_marcar as OM

    cfg = DS.resolver("schisto")
    cfg["_id"] = "schisto"
    todas = [os.path.splitext(f)[0] for f in DS.imagens("schisto")]
    sel = []
    for i in todas[:60]:
        lp = DS.caminho_label("schisto", i + ".png")
        if lp and (np.array(Image.open(lp).convert("L")) > 0).any():
            sel.append(i)
        if len(sel) == 2:
            break
    d = os.path.join(tmp, f"m{n_extra}")
    os.makedirs(d, exist_ok=True)
    for img in sel:
        lp = DS.caminho_label("schisto", img + ".png")
        gt = np.array(Image.open(lp).convert("L")) > 127
        m = generate_realistic_markers(lp, n_fg_dabs=6, n_bg_dabs=14,
                                       seed=OM._semente_marker(img, 0))
        fg = [tuple(int(v) for v in p) for p in m["fg_seeds"]]
        bg = [tuple(int(v) for v in p) for p in m["bg_seeds"]]
        if n_extra:
            ys, xs = np.nonzero(gt)
            r = np.random.default_rng(7)
            for k in r.permutation(len(ys))[:n_extra]:
                fg.append((int(xs[k]), int(ys[k])))
        with Image.open(lp) as im:
            w, h = im.size
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": h, "W": w},
                     os.path.join(d, f"{img}-seeds.txt"))
    return cfg, sel, d


def _pesos(caminho):
    import torch
    m = torch.load(caminho, map_location="cpu", weights_only=False)
    return [v.detach().numpy().copy()
            for v in m.state_dict().values()
            if hasattr(v, "ndim") and v.ndim == 4]


@pytest.fixture(scope="module")
def ambiente():
    tmp = tempfile.mkdtemp(prefix="clflim_")
    import tabela_k_por_modelo as TK
    cfg, sel, d0 = _monta(tmp)
    cfg["orig"] = TK._orig_em_png(cfg, "schisto", set(sel), tmp)
    yield tmp, cfg, sel, d0
    shutil.rmtree(tmp, ignore_errors=True)


def test_reimplementacao_bate_com_retrain_encoder(ambiente):
    tmp, cfg, sel, d0 = ambiente
    from flim_al import cl_flim
    from flim_al.al_encoder_experiment import evaluate_decoder, retrain_encoder
    import onde_marcar as OM

    a = os.path.join(tmp, "ref.pth")
    b = os.path.join(tmp, "cl.pth")
    retrain_encoder(cfg["arch"], d0, cfg["orig"], cfg["label"], "cpu", a)
    cl_flim.treinar(cfg["arch"], d0, cfg["orig"], cfg["label"], "cpu", b,
                    gravar_banco=os.path.join(tmp, "banco0.pkl"))

    arq = [OM._arquivo(cfg, i) for i in sel]
    m = {}
    for nome, enc in (("ref", a), ("cl", b)):
        m[nome] = evaluate_decoder(enc, "labeled_marker", cfg.get("bloco", 2),
                                   arq, cfg["orig"], cfg["label"], "cpu",
                                   area_range=tuple(cfg["area"]))["fb"]
    assert abs(m["ref"] - m["cl"]) < 1e-9, (
        f"cl_flim.treinar nao reproduz retrain_encoder: "
        f"{m['ref']:.10f} vs {m['cl']:.10f}")


def test_interceptou_o_kmeans_final(ambiente):
    tmp, cfg, sel, d0 = ambiente
    from flim_al import cl_flim
    _, banco = cl_flim.treinar(cfg["arch"], d0, cfg["orig"], cfg["label"],
                               "cpu", os.path.join(tmp, "i.pth"))
    assert banco.gravado, (
        "nenhum k-means final interceptado: os testes de alpha passariam "
        "por vacuidade")
    n_camadas = len(banco.gravado)
    assert n_camadas >= 2, f"so {n_camadas} camada(s) interceptada(s)"
    for l, c in banco.gravado.items():
        assert c.ndim == 4, f"camada {l}: forma inesperada {c.shape}"


def test_alpha_zero_nao_deixa_entrar_direcao_nova(ambiente):
    """
    Marcadores NOVOS com alpha=0: nenhum filtro pode ser inédito.

    A asserção é sobre o BANCO gravado, não sobre os pesos do Conv2d. Os pesos
    passam por normalização unitária e divisão pelo desvio, estatísticas que a
    rodada nova recalcula, então comparar pesos mediria a normalização e não a
    retenção. O banco é o objeto que o mecanismo controla.

    Com alpha=0, cada centróide de saída é uma cópia de algum centróide
    anterior. A verificação é exatamente essa: toda linha da saída tem de
    existir no banco anterior.
    """
    tmp, cfg, sel, d0 = ambiente
    from flim_al import cl_flim

    banco0 = os.path.join(tmp, "b0.pkl")
    e0 = os.path.join(tmp, "e0.pth")
    cl_flim.treinar(cfg["arch"], d0, cfg["orig"], cfg["label"], "cpu", e0,
                    gravar_banco=banco0)
    ant = cl_flim.carregar_banco(banco0)

    _, _, d1 = _monta(tmp, n_extra=400)
    e1 = os.path.join(tmp, "e1.pth")
    _, b = cl_flim.treinar(cfg["arch"], d1, cfg["orig"], cfg["label"], "cpu",
                           e1, banco=banco0, alpha=0.0)
    assert b.protegidas, "nenhuma camada protegida com banco carregado"

    for l, saida in b.gravado.items():
        a = np.asarray(ant[l]).reshape(len(ant[l]), -1)
        s = np.asarray(saida).reshape(len(saida), -1)
        if a.shape[1] != s.shape[1]:
            continue                     # camada a jusante mudou de dimensao
        d = ((s[:, None, :] - a[None, :, :]) ** 2).sum(-1).min(axis=1)
        assert d.max() < 1e-8, (
            f"camada {l}: alpha=0 deixou entrar filtro inedito "
            f"(maior distancia ao banco anterior {d.max():.3e})")


def test_alpha_um_e_flim_puro(ambiente):
    """alpha=1 tem de devolver exatamente o banco que o FLIM puro produziria."""
    tmp, cfg, sel, d0 = ambiente
    from flim_al import cl_flim

    banco0 = os.path.join(tmp, "b0a.pkl")
    cl_flim.treinar(cfg["arch"], d0, cfg["orig"], cfg["label"], "cpu",
                    os.path.join(tmp, "e0a.pth"), gravar_banco=banco0)
    _, _, d1 = _monta(tmp, n_extra=400)

    puro = os.path.join(tmp, "puro.pth")
    prot = os.path.join(tmp, "prot1.pth")
    cl_flim.treinar(cfg["arch"], d1, cfg["orig"], cfg["label"], "cpu", puro)
    cl_flim.treinar(cfg["arch"], d1, cfg["orig"], cfg["label"], "cpu", prot,
                    banco=banco0, alpha=1.0)

    p, q = _pesos(puro), _pesos(prot)
    assert len(p) == len(q), f"{len(p)} vs {len(q)} camadas"
    for i, (x, y) in enumerate(zip(p, q)):
        assert x.shape == y.shape, f"camada {i}: {x.shape} vs {y.shape}"
        assert np.allclose(x, y, atol=1e-6), (
            f"camada {i}: alpha=1 nao reproduz o FLIM puro "
            f"(maior diferenca {np.abs(x - y).max():.3e})")
