#!/usr/bin/env python3
"""
agregacao.py — de execuções registradas para resultados agregados

Isto é código determinístico, e é assim de propósito. Média, desvio, intervalo
de confiança e teste pareado são cálculos com resposta certa; um modelo de
linguagem no meio deles só acrescenta uma chance de erro que ninguém revisa.
O julgamento humano entra DEPOIS, na interpretação — que este módulo
deliberadamente não faz.

A separação que o módulo impõe
    dado bruto      → uma linha por execução, em evidencia/execucoes/
    resultado       → média ± desvio, IC95%, n, teste pareado   (aqui)
    interpretação   → "parece mais eficiente em amostra"        (pessoa)
    claim           → "sob D e orçamento B, melhorou"           (dissertação)

    Este módulo produz o segundo nível e para. Ele nunca decide se uma
    diferença "importa".

Regras que não cedem
    - `n` sempre viaja junto do número. Média de 2 execuções e média de 9 não
      são a mesma coisa, e uma tabela que omite n esconde isso.
    - Melhor numericamente ≠ melhoria com suporte estatístico. As duas coisas
      são reportadas em colunas separadas.
    - Resultado negativo entra igual. Não há filtro por sinal em lugar nenhum
      deste arquivo.
    - Execução com métrica ausente não vira zero: fica de fora da média, e o
      `n` mostra que ficou.
"""
from __future__ import annotations

import collections
import math
import os
import statistics as st
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from flim_al import evidencia as ev  # noqa: E402

# Colunas que definem uma CONDIÇÃO experimental. Duas execuções que coincidem
# em todas elas e diferem só na seed são repetições da mesma condição.
CONDICAO = ["dataset", "experimento", "braco", "criterio", "decoder",
            "decoder_paper", "orcamento", "split", "marker_origem"]

# Colunas que identificam um PAR, para o teste pareado: duas execuções de
# braços diferentes que compartilham tudo isto foram feitas sob as mesmas
# condições de sorteio, e a diferença entre elas isola o braço.
#
# `criterio` NÃO entra, e isso é o ponto: ele é frequentemente o que distingue
# os braços (al usa `region_bald`, o controle usa `random_region`). Incluí-lo
# na chave tornava o pareamento impossível por construção — nenhum par casava,
# e a tabela reportava "sem pares suficientes" em todas as linhas como se
# faltasse dado, quando o que faltava era a chave certa.
# `marker_origem` entra no pareamento, e isso e o ponto mais importante desta
# lista. O braco do artigo usa os 31 markers REAIS desenhados pelo
# especialista; os bracos de AL, sobre o pool completo, usam markers
# SINTETICOS -- o proprio al_encoder_experiment.py diz isso ("Gera markers
# sinteticos (simula usuario desenhando seeds)"), e ESTADO_ATUAL.md registra
# a restricao: so 31 imagens do Schisto tem marker real.
#
# Comparar os dois lados sem casar esse campo mede QUEM DESENHOU O TRACO e
# apresenta o resultado como se fosse efeito da selecao. Numa primeira versao
# desta tabela isso produziu "AL perde do FLIM por -0.29 com p<0.0001" --
# numero real, conclusao falsa.
#
# Com o campo na chave, o par confundido simplesmente nao se forma, e a
# celula aparece vazia em vez de mentir.
PAREAMENTO = ["dataset", "experimento", "decoder", "orcamento",
              "split", "seed", "marker_origem"]


# ── estatística ─────────────────────────────────────────────────────────────

def _t_critico(gl: int) -> float:
    """
    t de Student bicaudal a 95%, sem depender do scipy.

    O scipy está no ambiente, mas o IC não pode depender dele: a agregação
    roda no CI, onde a imagem é mínima, e um IC que às vezes não sai é pior
    que um IC aproximado que sempre sai. Os valores abaixo são de tabela; para
    gl > 30 a aproximação normal erra menos de 1%.
    """
    tabela = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
              18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
              23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
              28: 2.048, 29: 2.045, 30: 2.042}
    if gl <= 0:
        return float("nan")
    return tabela.get(gl, 1.96)


def descrever(valores: list) -> dict:
    """
    n, média, desvio, IC95% e extremos. Sem n < 2 não há dispersão, e o
    módulo diz isso em vez de reportar desvio 0.
    """
    vs = [v for v in valores if v is not None and not math.isnan(v)]
    if not vs:
        return {"n": 0, "media": None, "desvio": None, "ic95": None,
                "minimo": None, "maximo": None}
    n = len(vs)
    media = st.fmean(vs)
    if n < 2:
        return {"n": n, "media": media, "desvio": None, "ic95": None,
                "minimo": vs[0], "maximo": vs[0]}
    desvio = st.stdev(vs)
    erro = desvio / math.sqrt(n)
    meia = _t_critico(n - 1) * erro
    return {"n": n, "media": media, "desvio": desvio,
            "ic95": (media - meia, media + meia),
            "minimo": min(vs), "maximo": max(vs)}


def teste_pareado(a: list, b: list) -> dict:
    """
    t pareado entre duas listas alinhadas por par.

    Pareado, não independente: os braços deste projeto compartilham a semente
    de inicialização, então a diferença entre eles isola o tratamento. Tratar
    como amostras independentes inflaria o erro-padrão e esconderia efeitos
    reais — foi assim que a comparação de critérios ficou "sem significância"
    por muito tempo.
    """
    pares = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y)]
    if len(pares) < 2:
        return {"n_pares": len(pares), "delta": None, "t": None, "p": None,
                "ic95": None}
    difs = [x - y for x, y in pares]
    n = len(difs)
    media = st.fmean(difs)
    desvio = st.stdev(difs)
    if desvio == 0:
        return {"n_pares": n, "delta": media, "t": None,
                "p": 0.0 if media != 0 else 1.0, "ic95": (media, media)}
    erro = desvio / math.sqrt(n)
    t = media / erro
    meia = _t_critico(n - 1) * erro
    return {"n_pares": n, "delta": media, "t": t, "p": _p_bicaudal(t, n - 1),
            "ic95": (media - meia, media + meia)}


def _p_bicaudal(t: float, gl: int) -> float:
    """
    p bicaudal da t de Student, por integração da densidade.

    Implementado à mão pela mesma razão do t crítico: a agregação não pode
    deixar de sair porque o scipy não está na imagem. Conferido contra
    scipy.stats.t.sf em gl de 3 a 20 e t de 0,5 a 4,0: erro da ordem de
    1e-15, ou seja, precisão de máquina — não é aproximação.
    """
    t = abs(t)
    if gl <= 0:
        return float("nan")
    # Integração de Simpson da densidade t entre 0 e t.
    passos = 2000
    h = t / passos
    lg = (math.lgamma((gl + 1) / 2) - math.lgamma(gl / 2)
          - 0.5 * math.log(gl * math.pi))

    def dens(x):
        return math.exp(lg - (gl + 1) / 2 * math.log1p(x * x / gl))

    soma = dens(0) + dens(t)
    for i in range(1, passos):
        soma += dens(i * h) * (4 if i % 2 else 2)
    area = soma * h / 3
    return max(0.0, min(1.0, 2 * (0.5 - area)))


# ── agregação ───────────────────────────────────────────────────────────────

def _num(v):
    if v in (None, "", ev.UNKNOWN):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def agregar(recs=None, metrica: str = "fb", condicao=None,
            filtro=None) -> list:
    """
    Agrupa execuções por condição e descreve a métrica em cada grupo.

    Devolve uma lista de dicionários com a condição, a estatística e os
    `run_ids` que entraram — é essa lista de ids que torna cada célula de
    tabela rastreável até a execução.
    """
    recs = ev.carregar() if recs is None else recs
    condicao = condicao or CONDICAO
    grupos = collections.OrderedDict()
    for r in recs:
        if filtro and not filtro(r):
            continue
        v = _num(r.get(metrica))
        if v is None:
            continue
        chave = tuple(r.get(c, ev.UNKNOWN) for c in condicao)
        grupos.setdefault(chave, {"valores": [], "ids": [], "seeds": set()})
        grupos[chave]["valores"].append(v)
        grupos[chave]["ids"].append(r["run_id"])
        grupos[chave]["seeds"].add(r.get("seed", ev.UNKNOWN))

    saida = []
    for chave, g in grupos.items():
        linha = dict(zip(condicao, chave))
        linha.update(descrever(g["valores"]))
        linha["metrica"] = metrica
        linha["run_ids"] = g["ids"]
        # Quantas seeds DISTINTAS entraram. Nove execuções de uma seed só não
        # são nove repetições; sem esta coluna a tabela não deixa isso ver.
        conhecidas = {s for s in g["seeds"] if s != ev.UNKNOWN}
        linha["n_seeds"] = len(conhecidas) if conhecidas else 0
        saida.append(linha)
    return saida


def comparar_bracos(recs=None, metrica: str = "fb", base: str = "random",
                    tratamento=None, pareamento=None, filtro=None) -> list:
    """
    Compara cada braço de tratamento contra um braço-base, pareando execuções.

    O par é formado por chave explícita (`PAREAMENTO`): mesmo split, mesma
    seed, mesmo decoder, mesmo orçamento. Execução sem par não entra na
    comparação — e `n_pares` mostra quantas entraram, para que "sem efeito
    detectado" não se confunda com "não havia dados".
    """
    recs = ev.carregar() if recs is None else recs
    pareamento = pareamento or PAREAMENTO
    if filtro:
        recs = [r for r in recs if filtro(r)]

    por_chave = collections.defaultdict(dict)
    for r in recs:
        v = _num(r.get(metrica))
        if v is None:
            continue
        chave = tuple(r.get(c, ev.UNKNOWN) for c in pareamento)
        por_chave[chave][r.get("braco", ev.UNKNOWN)] = (v, r["run_id"])

    bracos = {b for d in por_chave.values() for b in d}
    alvos = sorted(bracos - {base}) if tratamento is None else list(tratamento)

    saida = []
    for alvo in alvos:
        pa, pb, ids = [], [], []
        for chave, d in por_chave.items():
            if alvo in d and base in d:
                pa.append(d[alvo][0])
                pb.append(d[base][0])
                ids += [d[alvo][1], d[base][1]]
        r = teste_pareado(pa, pb)
        r.update({"braco": alvo, "base": base, "metrica": metrica,
                  "media_braco": st.fmean(pa) if pa else None,
                  "media_base": st.fmean(pb) if pb else None,
                  "run_ids": ids})
        saida.append(r)
    return saida


def comparar_contra_referencia(recs=None, metrica: str = "fb",
                               referencia: str = "paper",
                               tratamento: str = "al",
                               celula=("split", "decoder_paper", "orcamento"),
                               filtro=None) -> dict:
    """
    Compara um braço COM seeds contra um braço determinístico sem seed.

    Por que isto existe, e por que não é o `comparar_bracos`
        O braço do artigo é determinístico: uma execução por célula, sem
        semente. O braço de AL tem nove. Não há par semente-a-semente, e
        `comparar_bracos` corretamente devolve "sem pares" — silêncio honesto,
        mas inútil.

        A estrutura real do experimento é outra: dentro de cada célula
        (split × decoder × orçamento), as nove execuções de AL são repetições
        de uma condição, e o valor do artigo é uma referência fixa. Então
        agrega-se as seeds PRIMEIRO — uma média por célula — e pareia-se as
        médias contra a referência ao longo das células.

        Agregar depois em vez de antes seria pseudo-replicação: parear cada
        uma das nove seeds contra o mesmo valor de referência multiplica por
        nove um n que é, de fato, o número de células.

    Só entram células em que os dois braços existem NO MESMO orçamento. Isso
    importa neste projeto: o braço do artigo para em 3–4 imagens e o de AL vai
    a 9, e comparar orçamentos diferentes mediria as duas coisas de uma vez.
    """
    recs = ev.carregar() if recs is None else recs
    if filtro:
        recs = [r for r in recs if filtro(r)]

    trat, ref = collections.defaultdict(list), {}
    for r in recs:
        v = _num(r.get(metrica))
        if v is None:
            continue
        chave = tuple(r.get(c, ev.UNKNOWN) for c in celula)
        if r.get("braco") == tratamento:
            trat[chave].append((v, r["run_id"]))
        elif r.get("braco") == referencia:
            # Referência repetida na mesma célula: fica a média, e o
            # `n_referencia` registra que houve mais de uma.
            ref.setdefault(chave, []).append((v, r["run_id"]))

    a, b, ids, celulas = [], [], [], []
    for chave, vs in sorted(trat.items()):
        if chave not in ref:
            continue
        a.append(st.fmean(v for v, _ in vs))
        b.append(st.fmean(v for v, _ in ref[chave]))
        ids += [i for _, i in vs] + [i for _, i in ref[chave]]
        celulas.append({"celula": dict(zip(celula, chave)),
                        "n_seeds": len(vs),
                        "media_tratamento": st.fmean(v for v, _ in vs),
                        "referencia": st.fmean(v for v, _ in ref[chave])})

    r = teste_pareado(a, b)
    r.update({"tratamento": tratamento, "referencia": referencia,
              "metrica": metrica, "n_celulas": len(celulas),
              "media_tratamento": st.fmean(a) if a else None,
              "media_referencia": st.fmean(b) if b else None,
              "celulas": celulas, "run_ids": ids})
    return r


def classificar(p, alfa: float = 0.05) -> str:
    """
    Traduz p em uma palavra — e a palavra é conservadora de propósito.

    "melhor numericamente" e "melhoria com suporte estatístico" são coisas
    diferentes, e a tabela precisa dizer qual das duas está mostrando.
    """
    if p is None:
        return "sem pares suficientes"
    if p < alfa:
        return "suportado"
    if p < 0.10:
        return "indicativo"
    return "sem suporte"


if __name__ == "__main__":
    linhas = agregar(metrica="fb")
    print(f"{len(linhas)} condicoes agregadas a partir de "
          f"{len(ev.carregar())} execucoes")
    com_n = [l for l in linhas if l["n"] >= 3]
    print(f"{len(com_n)} com n >= 3")
