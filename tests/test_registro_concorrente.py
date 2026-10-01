"""
Duas campanhas escrevendo no registro ao mesmo tempo não podem perder linha.

Por que este teste existe
    `evidencia.registrar` lê o CSV inteiro e o reescreve inteiro. Sem trava,
    duas campanhas simultâneas — o normal neste projeto, uma na GPU e uma na
    CPU — podem ler o mesmo estado e a segunda escrita apaga as linhas da
    primeira. O registro é a evidência inteira da dissertação; perder linha
    aqui é perder experimento já pago em horas de GPU.

    Já aconteceu: `vendor/REMOVIDOS.txt` documenta dois fragmentos corrompidos
    de `paper_selection` por escrita truncada.

O teste roda processos de verdade, não threads: a trava é entre PROCESSOS, e
o GIL esconderia a corrida que ela existe para impedir.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cada processo registra N execuções com run_id distinto. Ao final, as 2*N
# precisam estar todas lá.
N = 40

FILHO = r'''
import os, sys
sys.path.insert(0, {raiz!r})
from flim_al import evidencia as ev

arq = sys.argv[1]
marca = sys.argv[2]
for i in range({n}):
    r = ev.execucao(experimento="teste_concorrencia", dataset="schisto",
                    braco="al", criterio=marca, decoder="labeled_marker",
                    bloco=2, orcamento=3, seed=i, imagens=["x"],
                    variante=f"{{marca}}_{{i}}", fb=0.5, dice=0.5, iou=0.4,
                    mae=0.01, fonte="teste/concorrencia")
    ev.registrar([r], arquivo=arq)
'''


def test_duas_campanhas_simultaneas_nao_perdem_linha():
    with tempfile.TemporaryDirectory() as d:
        arq = os.path.join(d, "execucoes.csv")
        script = os.path.join(d, "filho.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(FILHO.format(raiz=RAIZ, n=N))

        procs = [subprocess.Popen([sys.executable, script, arq, marca],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
                 for marca in ("campanha_a", "campanha_b")]
        for p in procs:
            saida, erro = p.communicate(timeout=300)
            assert p.returncode == 0, erro.decode("utf-8", "replace")

        sys.path.insert(0, RAIZ)
        from flim_al import evidencia as ev
        recs = ev.carregar(arq)
        por_marca = {}
        for r in recs:
            por_marca.setdefault(r["criterio"], set()).add(r["variante"])

        assert len(por_marca.get("campanha_a", ())) == N, (
            f"campanha_a perdeu linhas: "
            f"{len(por_marca.get('campanha_a', ()))} de {N}")
        assert len(por_marca.get("campanha_b", ())) == N, (
            f"campanha_b perdeu linhas: "
            f"{len(por_marca.get('campanha_b', ()))} de {N}")


def test_trava_e_liberada_mesmo_com_excecao():
    """Uma exceção dentro da trava não pode deixar o registro travado para
    sempre — seria uma campanha de horas morrendo no primeiro `registrar`."""
    sys.path.insert(0, RAIZ)
    from flim_al import evidencia as ev
    with tempfile.TemporaryDirectory() as d:
        arq = os.path.join(d, "x.csv")
        try:
            with ev._trava(arq):
                raise RuntimeError("falha simulada")
        except RuntimeError:
            pass
        assert not os.path.exists(arq + ".lock"), "trava ficou para trás"
        # e ainda é possível adquirir de novo
        with ev._trava(arq):
            pass
