"""
O usuário simulado tem de ser previsível, e sem treinar nada para verificar.

`proximo_clique` implementa o protocolo de Xu et al. (2016): o clique vai ao
centro do maior erro. Os testes abaixo fixam o que "centro do maior erro"
significa, porque é a única parte do laço de Interactive Learning que não
depende do modelo, e portanto a única que pode ser verificada exatamente.
"""
from __future__ import annotations

import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from flim_al import il_flim as IL  # noqa: E402


def test_sem_erro_nao_pede_clique():
    gt = np.zeros((40, 40), bool)
    gt[10:20, 10:20] = True
    assert IL.proximo_clique(gt.copy(), gt) is None


def test_falso_negativo_pede_clique_de_objeto():
    """Faltou objeto: o clique tem de ser de foreground, dentro do que faltou."""
    gt = np.zeros((60, 60), bool)
    gt[10:30, 10:30] = True
    pred = np.zeros_like(gt)            # nao achou nada
    x, y, rotulo = IL.proximo_clique(pred, gt)
    assert rotulo == 1, "faltou objeto, mas pediu clique de fundo"
    assert gt[y, x], "clique de objeto caiu fora do objeto"


def test_falso_positivo_pede_clique_de_fundo():
    """Sobrou objeto: clique de background, dentro do que sobrou."""
    gt = np.zeros((60, 60), bool)
    gt[10:20, 10:20] = True
    pred = np.zeros_like(gt)
    pred[10:20, 10:20] = True
    pred[40:55, 40:55] = True           # uma mancha inteira sobrando
    x, y, rotulo = IL.proximo_clique(pred, gt)
    assert rotulo == 0, "sobrou objeto, mas pediu clique de objeto"
    assert not gt[y, x], "clique de fundo caiu dentro do objeto"
    assert 40 <= x < 55 and 40 <= y < 55, "clique nao caiu na mancha sobrando"


def test_escolhe_o_MAIOR_erro():
    """Com dois erros, o clique vai ao de maior area, nao ao primeiro achado."""
    gt = np.zeros((80, 80), bool)
    gt[5:10, 5:10] = True               # erro pequeno: 25 px
    gt[30:60, 30:60] = True             # erro grande: 900 px
    pred = np.zeros_like(gt)
    x, y, rotulo = IL.proximo_clique(pred, gt)
    assert 30 <= x < 60 and 30 <= y < 60, (
        f"clique em ({x},{y}) foi para o erro pequeno")


def test_clique_vai_ao_ponto_mais_fundo_nao_a_franja():
    """
    Num quadrado de erro, o clique tem de cair perto do centro.

    Protege contra a implementação ingênua que pega o primeiro pixel do erro,
    ou o centróide de uma forma côncava, que pode cair fora dela.
    """
    gt = np.zeros((100, 100), bool)
    gt[20:60, 20:60] = True
    pred = np.zeros_like(gt)
    x, y, _ = IL.proximo_clique(pred, gt)
    assert abs(x - 39.5) <= 2 and abs(y - 39.5) <= 2, (
        f"clique em ({x},{y}) longe do centro (39.5, 39.5)")


def test_vistos_evita_repetir_o_mesmo_pixel():
    """Sem isso, um clique que não muda a predição travaria o laço."""
    gt = np.zeros((60, 60), bool)
    gt[10:40, 10:40] = True
    pred = np.zeros_like(gt)
    a = IL.proximo_clique(pred, gt)
    b = IL.proximo_clique(pred, gt, vistos={(a[0], a[1])})
    assert b is None or (b[0], b[1]) != (a[0], a[1]), (
        "repetiu o mesmo pixel apesar de estar em `vistos`")


def test_disco_fica_dentro_da_imagem():
    pts = IL.disco(1, 1, (30, 30), raio=5)
    assert pts, "disco vazio"
    assert all(0 <= x < 30 and 0 <= y < 30 for x, y in pts), (
        "disco vazou da imagem")
    # num canto, o disco e recortado: menos que a area cheia
    assert len(pts) < len(IL.disco(15, 15, (30, 30), raio=5))


def test_iou_casos_de_borda():
    z = np.zeros((10, 10), bool)
    assert IL.iou(z, z) == 1.0, "duas mascaras vazias sao identicas"
    o = np.ones((10, 10), bool)
    assert IL.iou(o, z) == 0.0
    meio = np.zeros((10, 10), bool)
    meio[:5] = True
    assert abs(IL.iou(meio, o) - 0.5) < 1e-9


def test_resumo_nao_esconde_fracasso():
    """NoC médio tem de contar o teto, e a taxa de sucesso tem de aparecer."""
    execs = [
        {"noc": 3, "atingiu": True, "iou_inicial": 0.1, "iou_final": 0.9},
        {"noc": 20, "atingiu": False, "iou_inicial": 0.1, "iou_final": 0.4},
    ]
    r = IL.resumo(execs, alvo=0.85)
    assert r["n"] == 2
    assert r["taxa_sucesso"] == 0.5
    assert abs(r["noc_medio"] - 11.5) < 1e-9, (
        "noc_medio deveria incluir o teto das que falharam")
    assert abs(r["noc_dos_que_atingiram"] - 3.0) < 1e-9
