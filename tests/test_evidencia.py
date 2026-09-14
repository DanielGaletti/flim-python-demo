"""
Testes do registro canônico de evidência.

Cada teste aqui corresponde a uma falha que o registro tem de impedir. Um
registro de evidência que aceita lixo em silêncio é pior que nenhum: ele dá
confiança sem dar garantia, e o erro só aparece na tabela da dissertação —
longe da causa.
"""
from __future__ import annotations

import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from flim_al import evidencia as ev  # noqa: E402


# ── construção ──────────────────────────────────────────────────────────────

def test_execucao_preenche_tudo_que_falta_com_unknown():
    r = ev.execucao(experimento="teste", fb=0.8)
    assert set(r) == set(ev.CAMPOS)
    assert r["criterio"] == ev.UNKNOWN
    assert r["seed"] == ev.UNKNOWN


def test_metrica_ausente_fica_vazia_nao_unknown():
    """
    Não medir ≠ perder o registro de quem mediu.

    `dice=""` diz "esta execução não reportou dice". `dice="UNKNOWN"` diria
    "reportou, mas o valor se perdeu". A agregação trata os dois de forma
    diferente, então a distinção não pode colapsar.
    """
    r = ev.execucao(experimento="teste", fb=0.8)
    assert r["dice"] == ""
    assert r["seed"] == ev.UNKNOWN


def test_imagens_aceita_lista():
    r = ev.execucao(experimento="teste", imagens=["000982", "000463"])
    assert r["imagens"] == "000982|000463"


# ── identidade ──────────────────────────────────────────────────────────────

def test_run_id_e_deterministico():
    campos = dict(experimento="t", dataset="schisto", split=1, seed=0,
                  criterio="coreset")
    assert ev.execucao(**campos)["run_id"] == ev.execucao(**campos)["run_id"]


def test_run_id_muda_com_a_seed():
    a = ev.execucao(experimento="t", seed=0)
    b = ev.execucao(experimento="t", seed=1)
    assert a["run_id"] != b["run_id"], "seeds distintas não podem colidir"


def test_run_id_separa_linhas_do_mesmo_arquivo():
    """
    Foi esta colisão que descartou 1.645 registros na primeira migração.

    Duas linhas do mesmo CSV são, por construção, duas execuções — mesmo
    quando a origem não registra o que as diferencia.
    """
    a = ev.execucao(experimento="t", fonte="x.csv", linha_origem=2)
    b = ev.execucao(experimento="t", fonte="x.csv", linha_origem=3)
    assert a["run_id"] != b["run_id"]


def test_run_id_ignora_as_metricas():
    """O id identifica a execução; o resultado dela não entra."""
    a = ev.execucao(experimento="t", seed=0, fb=0.5)
    b = ev.execucao(experimento="t", seed=0, fb=0.9)
    assert a["run_id"] == b["run_id"]


# ── validação ───────────────────────────────────────────────────────────────

def test_recusa_fb_fora_de_faixa():
    """
    Foi esta regra que revelou o CSV com o cabeçalho deslocado: 189 linhas
    com fb entre 2 e 12, porque `n_train_imgs` caía na coluna de fb.
    """
    r = ev.execucao(experimento="t")
    r["fb"] = "6"
    r["run_id"] = ev.run_id(r)
    with pytest.raises(ev.RegistroInvalido, match="fora de"):
        ev.validar(r)


def test_recusa_metrica_nao_numerica():
    r = ev.execucao(experimento="t")
    r["dice"] = "quase bom"
    r["run_id"] = ev.run_id(r)
    with pytest.raises(ev.RegistroInvalido):
        ev.validar(r)


def test_recusa_experimento_vazio():
    r = ev.execucao(experimento="t")
    r["experimento"] = ""
    r["run_id"] = ev.run_id(r)
    with pytest.raises(ev.RegistroInvalido, match="experimento"):
        ev.validar(r)


def test_recusa_campo_identificador_editado_a_mao():
    """
    Editar a seed de um registro já gravado sem recalcular o id deixaria o
    arquivo internamente inconsistente — e ninguém perceberia.
    """
    r = ev.execucao(experimento="t", seed=0)
    r["seed"] = "7"
    with pytest.raises(ev.RegistroInvalido, match="run_id"):
        ev.validar(r)


def test_recusa_campo_desconhecido():
    r = ev.execucao(experimento="t")
    r["gambiarra"] = "1"
    with pytest.raises(ev.RegistroInvalido, match="desconhecidos"):
        ev.validar(r)


# ── escrita ─────────────────────────────────────────────────────────────────

def test_registrar_e_carregar(tmp_path):
    arq = str(tmp_path / "e.csv")
    recs = [ev.execucao(experimento="t", seed=i, fb=0.5 + i / 100)
            for i in range(3)]
    r = ev.registrar(recs, arquivo=arq)
    assert r["novos"] == 3 and r["total"] == 3
    assert len(ev.carregar(arq)) == 3


def test_registrar_e_idempotente(tmp_path):
    """Rodar a migração duas vezes não pode duplicar nada."""
    arq = str(tmp_path / "e.csv")
    recs = [ev.execucao(experimento="t", seed=0, fb=0.5)]
    ev.registrar(recs, arquivo=arq)
    r = ev.registrar(recs, arquivo=arq)
    assert r["novos"] == 0 and r["duplicados"] == 1 and r["total"] == 1


def test_duplicata_com_metrica_diferente_e_sinalizada(tmp_path):
    """
    Mesma configuração, resultado outro = algo mudou fora do que foi
    registrado. Agregá-los como repetições independentes seria errado, então
    o registro precisa gritar.
    """
    arq = str(tmp_path / "e.csv")
    ev.registrar([ev.execucao(experimento="t", seed=0, fb=0.50)], arquivo=arq)
    r = ev.registrar([ev.execucao(experimento="t", seed=0, fb=0.91)],
                     arquivo=arq)
    assert r["divergentes"], "divergência silenciosa"


def test_arquivo_gravado_sobrevive_a_validacao(tmp_path):
    """O que sai do arquivo tem de continuar passando na validação."""
    arq = str(tmp_path / "e.csv")
    ev.registrar([ev.execucao(experimento="t", seed=1, fb=0.7,
                              imagens=["a", "b"])], arquivo=arq)
    for r in ev.carregar(arq):
        ev.validar(r)


# ── o arquivo real do projeto ───────────────────────────────────────────────

def test_registro_do_projeto_e_valido():
    """
    Todo registro em evidencia/execucoes/execucoes.csv passa na validação.

    Este é o teste que impede a base empírica da dissertação de apodrecer sem
    aviso: qualquer edição manual, migração malfeita ou coluna deslocada
    quebra aqui.
    """
    if not os.path.isfile(ev.ARQUIVO):
        pytest.skip("registro ainda não migrado")
    recs = ev.carregar(ev.ARQUIVO)
    assert recs, "registro vazio"
    for i, r in enumerate(recs, 2):
        try:
            ev.validar(r)
        except ev.RegistroInvalido as e:
            pytest.fail(f"linha {i} inválida: {e}")


def test_registro_do_projeto_nao_tem_run_id_repetido():
    if not os.path.isfile(ev.ARQUIVO):
        pytest.skip("registro ainda não migrado")
    ids = [r["run_id"] for r in ev.carregar(ev.ARQUIVO)]
    assert len(ids) == len(set(ids)), (
        f"{len(ids) - len(set(ids))} run_id duplicados — "
        "dois registros distintos estão se passando por um só"
    )
