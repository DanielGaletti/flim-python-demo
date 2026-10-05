#!/usr/bin/env python3
"""
determinismo.py — o treino do FLIM só é reprodutível com uma thread

O que foi medido (2026-10-05)
    `retrain_encoder` chamado DUAS vezes, com os mesmos arquivos de marcador,
    no mesmo processo:

        markers que geram 105/99/99/99 kernels   Fbeta identico, pesos ~5e-08
        markers que geram 200/200/84/84 kernels  Fbeta difere 2,5e-04,
                                                 pesos da camada 1 em 0,134

    Com as threads de BLAS e OpenMP fixadas em 1, os dois casos dão diferença
    exatamente 0,000e+00 em todas as camadas.

Por que o número de kernels decide
    `FLIMModel.cluster_patches_faiss` tem dois caminhos:

        n > n_clusters   roda k-means sobre os candidatos
        n <= n_clusters  DEVOLVE os próprios candidatos, sem agrupar

    Abaixo do teto da arquitetura não há agrupamento, então não há o que ser
    instável. Acima do teto, o k-means precisa escolher entre muitos
    candidatos, e a ordem de redução em BLAS multithread muda a soma no último
    bit. Isso basta para o k-means cair em outra bacia, e aí o banco de
    filtros inteiro muda, não um pouco.

    O efeito se amplifica camada a camada: a camada 1 recebe como entrada as
    ativações da camada 0, então um desvio de 1e-07 nos pesos da primeira
    reaparece como 0,13 nos pesos da segunda.

Consequência para as campanhas anteriores
    As campanhas registradas antes desta data rodaram SEM fixar as threads.
    Elas são válidas, porque dentro de cada célula os braços compartilham a
    partição e os marcadores, e a comparação é pareada. Mas o ruído entre
    execuções delas não é desprezível em qualquer regime: é praticamente zero
    quando o encoder fica abaixo do teto de kernels e pode ser grande quando
    passa dele. A medição de 2 registros divergentes em 540 na reexecução do
    Schisto subestima o caso acima do teto.

    Toda campanha nova usa `fixar()`.

Uso
    from flim_al.determinismo import fixar

    with fixar():
        modelo.fit(dataset)
"""
from __future__ import annotations

import contextlib
import os

# Para o caso de este módulo ser importado antes de numpy: as variáveis de
# ambiente só têm efeito se lidas na carga das bibliotecas nativas. Elas são
# redundantes com `fixar()`, que funciona em tempo de execução, e existem para
# o caso de alguém rodar um script que não chame o gerenciador de contexto.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


@contextlib.contextmanager
def fixar(n: int = 1):
    """
    Limita BLAS e OpenMP a `n` threads enquanto o bloco roda.

    Usa `threadpoolctl`, que altera os pools já carregados, então funciona
    independentemente da ordem de import. Se o pacote não estiver presente,
    o bloco roda sem limite e a função avisa uma vez: silenciar seria pior,
    porque o resultado deixaria de ser reprodutível sem nenhum sinal.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                                       # noqa: BLE001
        if not getattr(fixar, "_avisou", False):
            print("  [determinismo] threadpoolctl ausente: o treino NAO "
                  "sera reprodutivel entre execucoes acima do teto de "
                  "kernels. pip install threadpoolctl")
            fixar._avisou = True
        yield
        return
    with threadpool_limits(limits=n):
        yield


def conferir(treino, n_rep: int = 2) -> dict:
    """
    Roda `treino()` n_rep vezes e devolve a maior diferença entre os pesos.

    `treino` recebe o índice da repetição e devolve o caminho do encoder.
    Serve de verificação de ambiente: num ambiente pinado, `maxdif` tem de
    ser 0,0.
    """
    import numpy as np
    import torch

    bancos = []
    for i in range(n_rep):
        caminho = treino(i)
        m = torch.load(caminho, map_location="cpu", weights_only=False)
        bancos.append([v.detach().numpy().copy()
                       for v in m.state_dict().values()
                       if hasattr(v, "ndim") and v.ndim == 4])
    difs = []
    for camadas in zip(*bancos):
        base = camadas[0]
        for outro in camadas[1:]:
            if base.shape != outro.shape:
                difs.append(float("inf"))
            else:
                difs.append(float(np.abs(base - outro).max()))
    return {"n_rep": n_rep, "maxdif": max(difs) if difs else 0.0,
            "por_camada": difs}
