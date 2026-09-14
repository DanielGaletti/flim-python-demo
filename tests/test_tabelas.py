"""
Testes da agregação e das tabelas.

O teste que dá nome ao arquivo é `test_tabelas_batem_com_o_registro`: ele
falha se qualquer número de uma tabela gerada divergir do que sai do registro
de execuções. É a trava contra o modo de falha mais silencioso de uma
dissertação — a tabela e o dado se separarem, e ninguém notar até a arguição.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from flim_al import agregacao as ag  # noqa: E402
from flim_al import evidencia as ev  # noqa: E402
from flim_al import tabelas as tb  # noqa: E402


# ── estatística ─────────────────────────────────────────────────────────────

def test_descrever_valores_conhecidos():
    r = ag.descrever([1.0, 2.0, 3.0, 4.0])
    assert r["n"] == 4
    assert math.isclose(r["media"], 2.5)
    assert math.isclose(r["desvio"], 1.2909944, rel_tol=1e-6)
    assert r["minimo"] == 1.0 and r["maximo"] == 4.0


def test_descrever_nao_inventa_dispersao_com_uma_amostra():
    """Desvio 0 com n=1 seria uma afirmação de precisão que não existe."""
    r = ag.descrever([0.7])
    assert r["n"] == 1 and r["desvio"] is None and r["ic95"] is None


def test_descrever_ignora_ausentes_sem_virar_zero():
    r = ag.descrever([1.0, None, 3.0, float("nan")])
    assert r["n"] == 2 and math.isclose(r["media"], 2.0)


def test_p_bicaudal_confere_com_scipy():
    """
    A estatística é feita à mão para não depender do scipy no CI. Este teste é
    o que garante que "à mão" não virou "por aproximação".
    """
    stats = pytest.importorskip("scipy.stats")
    for t in (0.5, 1.5, 2.5, 4.0):
        for gl in (3, 8, 20):
            assert math.isclose(ag._p_bicaudal(t, gl), 2 * stats.t.sf(t, gl),
                                abs_tol=1e-9), f"t={t} gl={gl}"


def test_teste_pareado_detecta_diferenca_constante():
    a = [0.7, 0.8, 0.9, 0.75, 0.85]
    b = [x - 0.05 for x in a]
    r = ag.teste_pareado(a, b)
    assert math.isclose(r["delta"], 0.05, abs_tol=1e-9)
    assert r["p"] is not None and r["p"] < 0.001


def test_teste_pareado_sem_pares_nao_inventa_p():
    r = ag.teste_pareado([0.7], [0.6])
    assert r["p"] is None and r["n_pares"] == 1


def test_classificar_e_conservador():
    assert ag.classificar(0.01) == "suportado"
    assert ag.classificar(0.07) == "indicativo"
    assert ag.classificar(0.40) == "sem suporte"
    assert ag.classificar(None) == "sem pares suficientes"


def test_pareamento_nao_inclui_o_que_distingue_os_bracos():
    """
    `criterio` na chave de pareamento tornava o pareamento impossível: os
    braços diferem justamente nele, então nenhum par casava e a tabela dizia
    "sem pares" como se faltasse dado.
    """
    assert "criterio" not in ag.PAREAMENTO
    assert "braco" not in ag.PAREAMENTO


# ── tabela ──────────────────────────────────────────────────────────────────

def test_tabela_exige_proveniencia():
    t = tb.Tabela("t", "T", ["a", "b"])
    with pytest.raises(ValueError, match="run_ids"):
        t.adicionar(["x", "1"], [])


def test_tabela_recusa_linha_com_tamanho_errado():
    t = tb.Tabela("t", "T", ["a", "b"])
    with pytest.raises(ValueError):
        t.adicionar(["x"], ["id1"])


def test_latex_avisa_que_nao_se_edita_a_mao():
    t = tb.Tabela("t", "T", ["a", "b"])
    t.adicionar(["x", "1"], ["id1"])
    tex = t.latex()
    assert "NAO EDITE A MAO" in tex
    assert r"\begin{tabular}" in tex


def test_enfase_so_onde_pedida():
    """Negrito é uma afirmação; não pode aparecer por conta própria."""
    t = tb.Tabela("t", "T", ["a", "b"])
    t.adicionar(["x", "0.9"], ["id1"])
    t.adicionar(["y", "0.5"], ["id2"], enfase=1)
    md = t.markdown()
    assert "**0.5**" in md
    assert "**0.9**" not in md, "o maior valor virou negrito sozinho"


def test_ausencia_vira_travessao_nao_zero():
    assert tb._fmt(None) == "—"
    assert tb._fmt(0.0) == "0.000"


def test_gravar_produz_os_quatro_arquivos(tmp_path):
    t = tb.Tabela("teste", "T", ["a", "b"])
    t.adicionar(["x", "1"], ["id1", "id2"])
    escritos = t.gravar(str(tmp_path))
    assert set(escritos) == {"csv", "md", "tex", "proveniencia"}
    prov = json.load(open(escritos["proveniencia"], encoding="utf-8"))
    assert prov["linhas"][0]["run_ids"] == ["id1", "id2"]
    assert prov["input_latex"] == r"\input{generated/tables/teste.tex}"


# ── a trava: tabela x registro ──────────────────────────────────────────────

def _tabelas_geradas():
    if not os.path.isdir(tb.DESTINO):
        return []
    return [f[:-len(".proveniencia.json")]
            for f in os.listdir(tb.DESTINO)
            if f.endswith(".proveniencia.json")]


def test_toda_tabela_gerada_tem_proveniencia():
    nomes = _tabelas_geradas()
    if not nomes:
        pytest.skip("nenhuma tabela gerada ainda")
    for nome in nomes:
        for ext in ("csv", "md", "tex"):
            caminho = os.path.join(tb.DESTINO, f"{nome}.{ext}")
            assert os.path.isfile(caminho), f"falta {nome}.{ext}"


def test_run_ids_das_tabelas_existem_no_registro():
    """
    Toda célula tem de apontar para execuções que existem.

    Um run_id órfão significa que a tabela foi gerada de um registro que mudou
    depois — e o número na dissertação deixou de ter origem.
    """
    nomes = _tabelas_geradas()
    if not nomes:
        pytest.skip("nenhuma tabela gerada ainda")
    validos = {r["run_id"] for r in ev.carregar()}
    if not validos:
        pytest.skip("registro vazio")
    for nome in nomes:
        prov = json.load(open(
            os.path.join(tb.DESTINO, f"{nome}.proveniencia.json"),
            encoding="utf-8"))
        for linha in prov["linhas"]:
            orfaos = [i for i in linha["run_ids"] if i not in validos]
            assert not orfaos, (
                f"{nome}, linha {linha['linha']} ({linha['rotulo']}): "
                f"{len(orfaos)} run_id sem execução correspondente")


def test_tabelas_batem_com_o_registro():
    """
    Regenerar as tabelas a partir do registro tem de dar o MESMO resultado.

    Este é o teste que impede a tabela e o dado de se separarem. Ele falha se
    alguém editou um `.tex` à mão, se o registro mudou sem regenerar, ou se a
    agregação deixou de ser determinística.
    """
    import subprocess
    import tempfile

    if not os.path.isfile(ev.ARQUIVO):
        pytest.skip("registro ainda não migrado")
    nomes = _tabelas_geradas()
    if not nomes:
        pytest.skip("nenhuma tabela gerada ainda")

    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, os.path.join(RAIZ, "scripts", "gerar_tabelas.py"),
             "--destino", tmp],
            capture_output=True, text=True, cwd=RAIZ, timeout=300)
        assert r.returncode == 0, f"geração falhou:\n{r.stdout}\n{r.stderr}"

        for nome in nomes:
            novo = os.path.join(tmp, f"{nome}.csv")
            antigo = os.path.join(tb.DESTINO, f"{nome}.csv")
            if not os.path.isfile(novo):
                continue
            with open(novo, encoding="utf-8") as fh:
                a = fh.read()
            with open(antigo, encoding="utf-8") as fh:
                b = fh.read()
            assert a == b, (
                f"{nome}.csv não corresponde ao registro.\n"
                "Ou o .csv foi editado à mão, ou o registro mudou sem que as "
                "tabelas fossem regeneradas. Rode:\n"
                "  python scripts/gerar_tabelas.py"
            )
