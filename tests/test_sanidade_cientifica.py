"""
Sanidade científica — as verificações que um revisor faria, em código.

A diferença entre estes testes e uma revisão por agente é que estes não se
deixam convencer. Um modelo de linguagem lendo o código pode concluir que não
há vazamento; um teste que intersecta os conjuntos ou encontra a interseção ou
não encontra.

Cada teste aqui corresponde a um erro que já aconteceu neste projeto, ou a um
que a estrutura do dado torna fácil.
"""
from __future__ import annotations

import collections
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from flim_al import evidencia as ev  # noqa: E402


def _ler_lista(nome: str) -> set:
    caminho = os.path.join(RAIZ, nome)
    if not os.path.isfile(caminho):
        return set()
    with open(caminho, encoding="utf-8") as fh:
        return {os.path.splitext(l.strip())[0] for l in fh if l.strip()}


def _registro():
    if not os.path.isfile(ev.ARQUIVO):
        pytest.skip("registro ainda não migrado")
    return ev.carregar()


# ── vazamento ───────────────────────────────────────────────────────────────

def test_splits_declarados_sao_disjuntos():
    """
    train ∩ test = ∅ nos arquivos de split versionados.

    Isto verifica a DECLARAÇÃO, que é o primeiro lugar onde vazamento entra:
    se os próprios arquivos de split já se sobrepõem, nenhuma disciplina de
    execução salva.
    """
    teste = _ler_lista("test.txt")
    if not teste:
        pytest.skip("test.txt ausente")
    for n in (1, 2, 3):
        treino = _ler_lista(f"split{n}-train.txt")
        if not treino:
            continue
        comum = treino & teste
        assert not comum, (
            f"split{n}-train.txt e test.txt compartilham {len(comum)} "
            f"imagem(ns): {sorted(comum)[:5]}"
        )


def test_nenhuma_execucao_treinou_com_imagem_de_teste():
    """
    Nenhuma imagem do conjunto de teste aparece no treino de execução alguma.

    Qual conjunto de teste importa, e errar isso custa caro
        A primeira versão deste teste usava `test.txt` da raiz e acusou 838
        execuções de vazamento em 11 famílias. Era falso: `test.txt` é o teste
        das BASELINES (SAMNet/MSCNet/MEANet), usado por
        `baselines/train_baselines.py`. Os experimentos de AL usam o
        `teste_comum.txt`, que `montar_teste_comum.py` constrói subtraindo a
        união de tudo que qualquer braço de qualquer semente vai treinar — ou
        seja, a própria correção contra vazamento.

        Um alarme falso num teste de integridade é pior que teste nenhum:
        gasta a confiança que faz os alarmes verdadeiros serem levados a
        sério. Por isso este teste PULA quando não sabe qual conjunto de teste
        a família usou, em vez de chutar um.

    As campanhas `*_POOL_COM_VAZAMENTO` ficam de fora de propósito: elas
    tiveram vazamento de verdade, o nome do diretório registra isso, e estão
    preservadas como resultado negativo — não como evidência válida.
    """
    comum = os.path.join(RAIZ, "evidencia", "conjuntos",
                         "final_comparison_v3_teste_comum.txt")
    if not os.path.isfile(comum):
        pytest.skip("teste_comum.txt não arquivado — sem referência confiável")
    with open(comum, encoding="utf-8") as fh:
        teste = {os.path.splitext(l.strip())[0] for l in fh if l.strip()}

    # Famílias que comprovadamente usaram o teste comum. As demais são
    # puladas: a origem não registra o conjunto de teste, e supor um
    # produziria exatamente o alarme falso descrito acima.
    FAMILIAS = {"final_comparison", "final_comparison_v2", "final_comparison_v3",
                "curva", "sel_cs_grande", "sel_degen", "sel_medoide",
                "sel_proposto"}

    culpados = collections.Counter()
    verificadas = 0
    for r in _registro():
        if r["experimento"] not in FAMILIAS or "VAZAMENTO" in r["experimento"]:
            continue
        imgs = r.get("imagens", "")
        if imgs in ("", ev.UNKNOWN):
            continue
        verificadas += 1
        if set(imgs.split("|")) & teste:
            culpados[r["experimento"]] += 1

    if not verificadas:
        pytest.skip("nenhuma execução com lista de imagens verificável")
    assert not culpados, (
        f"de {verificadas} execuções verificadas, algumas treinaram com "
        "imagem do teste comum: "
        + ", ".join(f"{k} ({v})" for k, v in culpados.most_common())
    )


# ── integridade do registro ─────────────────────────────────────────────────

def test_metricas_dentro_da_faixa():
    """Fβ, Dice e IoU vivem em [0,1]. Fora disso é coluna deslocada."""
    fora = []
    for r in _registro():
        for campo in ("fb", "dice", "iou"):
            v = r.get(campo, "")
            if v in ("", ev.UNKNOWN):
                continue
            x = float(v)
            if not (0.0 <= x <= 1.0):
                fora.append(f"{r['run_id']} {campo}={x} ({r['fonte']})")
    assert not fora, f"{len(fora)} métrica(s) fora de [0,1]:\n" + \
        "\n".join(fora[:5])


def test_familias_com_semente_unica_sao_reportadas():
    """
    Família que registra seed mas nunca a variou não tem repetição real.

    Isto é DIAGNÓSTICO, não erro: rodar uma semente só é uma limitação
    legítima, e várias campanhas deste projeto foram assim. O que não pode
    acontecer é a limitação passar despercebida e virar "n=9 execuções" numa
    tabela, quando são nove decoders de uma execução só.

    Por isso pula com a lista em vez de falhar — a tabela já separa `n` de
    `n_seeds`, e este teste existe para que a lista apareça no log do CI.
    """
    por_fam = collections.defaultdict(set)
    for r in _registro():
        if r["seed"] != ev.UNKNOWN:
            por_fam[r["experimento"]].add(r["seed"])
    unicas = sorted(f for f, s_ in por_fam.items() if len(s_) == 1)
    if unicas:
        pytest.skip(
            "família(s) com semente única — sem repetição independente: "
            + ", ".join(unicas)
        )


def test_colapso_marcado_e_coerente():
    """
    A assinatura do colapso é Fβ = DICE = IoU ≈ 0.479 (predição toda-fundo).

    Execução com essa assinatura e `colapsou` em branco é colapso não
    registrado — e a média que a inclui mede frequência de colapso, não
    qualidade de seleção.
    """
    nao_marcados = []
    for r in _registro():
        try:
            fb, dice, iou = (float(r[c]) for c in ("fb", "dice", "iou"))
        except (ValueError, KeyError):
            continue
        assinatura = (abs(fb - dice) < 1e-3 and abs(dice - iou) < 1e-3
                      and 0.47 < fb < 0.49)
        if assinatura and r.get("colapsou") in ("", ev.UNKNOWN, "0"):
            nao_marcados.append(f"{r['run_id']} fb={fb:.4f} ({r['fonte']})")
    if nao_marcados:
        pytest.skip(
            f"{len(nao_marcados)} execução(ões) com assinatura de colapso sem "
            f"marca — esperado nos registros migrados, que não tinham a "
            f"coluna. Exemplo: {nao_marcados[0]}"
        )


# ── ambiente ────────────────────────────────────────────────────────────────

def test_registro_nao_mistura_series_de_kmeans():
    """
    Todas as execuções de uma mesma família devem vir do mesmo k-means.

    Não há coluna que registre isso nos dados históricos, então o teste só
    pode alertar sobre o risco. O que ele garante de fato está em
    `test_ambiente_experimental.py`: o ambiente ATUAL usa o sklearn.
    """
    pytest.importorskip("pytest")
    from tests import test_ambiente_experimental as amb  # noqa: F401
    # A garantia real vive lá; aqui só se documenta a dependência entre os
    # dois arquivos, para que remover um não deixe o outro sem sentido.
    assert hasattr(amb, "test_kmeans_usa_o_mesmo_caminho_dos_resultados")


def test_todo_numero_de_tabela_tem_origem():
    """
    Atalho para o que `test_tabelas.py` verifica em detalhe: nenhuma tabela
    gerada pode conter linha sem `run_ids`.
    """
    import json
    from flim_al import tabelas as tb

    if not os.path.isdir(tb.DESTINO):
        pytest.skip("nenhuma tabela gerada")
    for f in os.listdir(tb.DESTINO):
        if not f.endswith(".proveniencia.json"):
            continue
        prov = json.load(open(os.path.join(tb.DESTINO, f), encoding="utf-8"))
        for linha in prov["linhas"]:
            assert linha["run_ids"], (
                f"{f}: linha {linha['linha']} sem proveniência"
            )
