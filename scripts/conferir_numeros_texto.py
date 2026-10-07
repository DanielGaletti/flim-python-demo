#!/usr/bin/env python3
"""
conferir_numeros_texto.py — todo numero do CORPO do texto existe na evidencia?

As tabelas sao geradas e por construcao batem com o registro. O corpo dos
capitulos, nao: ele e escrito a mao, e numero escrito a mao e numero que
envelhece. Isso ja aconteceu duas vezes neste projeto, com a margem do ganho
marginal constando como "0,133 e 0,303" quando o registro dizia 0,1351 e
0,3089.

Esta checagem extrai os decimais do texto corrido dos capitulos e procura cada
um nas tabelas geradas, nos pre-registros das campanhas e no JSON dos modelos
de comparacao.

E uma checagem de TRIAGEM, nao uma prova. Um numero que ela nao encontra pode
ser legitimo: uma porcentagem derivada, um valor citado de um artigo, uma
contagem. O que ela garante e que nenhum numero passe sem alguem ter olhado.
Por isso a lista de excecoes e NOMINAL e cada entrada diz de onde vem.

Uso
    python scripts/conferir_numeros_texto.py
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISS = os.path.join(RAIZ, "DISSERTACAO")

# Numeros do texto que NAO vem do registro. Cada um com a origem declarada.
CONHECIDOS = {
    # medidos e registrados em comentario de codigo ou em campanha, nao em tabela
    "0.997": "fracao de objeto dos candidatos de contorno, campanha do gerador",
    "0.779": "idem, BraTS",
    # parametros e constantes do metodo
    "0.300": "beta ao quadrado da metrica Fbeta",
    "0.500": "limiar de probabilidade",
    # contagens e proporcoes derivadas que o texto explica no lugar
    "0.0034": "desvio maximo entre execucoes; adendo_ganho_marginal_2026-10-01",
    # Os tres abaixo ANTECEDEM o registro de execucoes e nao podem ser
    # recomputados a partir dele. O proprio texto declara essa limitacao na
    # subsecao de reprodutibilidade; a excecao aqui existe para que a checagem
    # nao os acuse a cada rodada, e NAO para dispensar a declaracao.
    "0.134": "divergencia de pesos entre execucoes; anterior ao registro",
    "0.024": "Fbeta mediano antes da correcao do filtro de area; idem",
    "0.594": "Fbeta mediano depois da correcao; idem",
    # NAO reintroduzir 0.126 e 0.166 aqui. Eles constavam do texto como "o
    # valor da anotacao de borda" e NAO vinham da tabela citada: a coluna
    # `aleat-artigo` de artigo_vs_regiao vai de 0,075 a 0,182. Eram numero
    # defasado, e estavam nesta lista por engano, isentados em vez de
    # verificados. Uma excecao sem origem conferida transforma a checagem em
    # carimbo.
    "0.724": "exemplo de regressao no laco interativo",
    "0.702": "idem, valor final",
    "0.1034": "soma 0,0772+0,0262, conferida contra 0,1033 medido",
}


def carregar_universo() -> set:
    txt = []
    for p in glob.glob(os.path.join(RAIZ, "evidencia", "tabelas", "*.md")):
        txt.append(io.open(p, encoding="utf-8", errors="replace").read())
    for p in glob.glob(os.path.join(RAIZ, "evidencia", "campanhas", "*.json")):
        txt.append(io.open(p, encoding="utf-8", errors="replace").read())
    p = os.path.join(RAIZ, "ESTADO_ATUAL.md")
    if os.path.isfile(p):
        txt.append(io.open(p, encoding="utf-8", errors="replace").read())
    universo = "\n".join(txt).replace("\u2212", "-")
    universo = re.sub(r"(?<=\d),(?=\d)", ".", universo)

    numeros = set(re.findall(r"-?\d+\.\d+", universo))
    # O texto costuma citar a MAGNITUDE de um valor negativo ("a dispersao cai
    # 0,1091" para um Delta de -0,1091). Sem isto o verificador acusa o proprio
    # numero que esta certo.
    numeros |= {n.lstrip("-") for n in list(numeros)}
    # os modelos de comparacao guardam float inteiro; arredondar
    jb = os.path.join(RAIZ, "results", "baselines", "baseline_results.json")
    if os.path.isfile(jb):
        def anda(o):
            if isinstance(o, dict):
                for v in o.values():
                    anda(v)
            elif isinstance(o, (int, float)):
                for c in range(2, 7):
                    numeros.add(f"{round(float(o), c):.{c}f}")
        anda(json.load(io.open(jb, encoding="utf-8")))
    return numeros


def sem_estrutura(t: str) -> str:
    """Fora figuras, tabelas, labels e comentarios: so o texto corrido."""
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    t = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", t,
               flags=re.S)
    t = re.sub(r"\\(label|ref|cite[a-zA-Z]*|input|includegraphics)"
               r"(\[[^\]]*\])?\{[^}]*\}", " ", t)
    return t


def main() -> int:
    universo = carregar_universo()
    print(f"{len(universo)} valor(es) distinto(s) na evidencia\n")

    sem_fonte = []
    for f in sorted(glob.glob(os.path.join(DISS, "capitulos", "*.tex"))):
        t = sem_estrutura(io.open(f, encoding="utf-8",
                                  errors="replace").read())
        t = t.replace("\u2212", "-")
        t = re.sub(r"(?<=\d),(?=\d)", ".", t)
        for n, linha in enumerate(t.split("\n"), 1):
            for v in re.findall(r"-?\d+\.\d{3,}", linha):
                alvo = v.lstrip("-")
                if alvo in universo or v in universo:
                    continue
                if alvo in CONHECIDOS:
                    continue
                sem_fonte.append((os.path.basename(f), n, v,
                                  linha.strip()[:88]))

    if not sem_fonte:
        print("ok: todo decimal do corpo do texto tem fonte na evidencia,\n"
              "    ou consta da lista nominal de excecoes declaradas.")
        return 0
    print(f"SEM FONTE: {len(sem_fonte)}\n")
    for arq, n, v, ctx in sem_fonte:
        print(f"  {arq}:{n}  {v}")
        print(f"      ...{ctx}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
