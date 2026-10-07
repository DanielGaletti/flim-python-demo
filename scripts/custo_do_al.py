#!/usr/bin/env python3
"""
custo_do_al.py — a lacuna entre o clique perfeito e o clique que o AL escolhe

Esta e a afirmacao central da dissertacao, e ela tem duas metades que nao
podem ser confundidas:

    teto amostrado   o quanto vale anotar a MELHOR regiao entre as examinadas,
                     escolhida pelo Fbeta da validacao. NAO e estrategia: usa
                     o resultado para escolher. Mede a margem que EXISTE.

    o criterio       o quanto vale anotar a regiao que o escore de Active
                     Learning aponta, sem olhar resultado nenhum. Mede a
                     margem que se CAPTURA.

A diferenca entre as duas e o custo do Active Learning: o que se deixa na mesa
porque o criterio erra o lugar do clique. Escrever "o AL melhorou o Fbeta em
0,1 a 0,3" seria falso; o que vale 0,1 a 0,3 e o clique perfeito.

Ambas as colunas saem da mesma campanha `ganho_marginal`, com a mesma base e a
mesma validacao, entao a subtracao e pareada por semente e nao mistura
condicoes.

Uso
    python scripts/custo_do_al.py
"""
from __future__ import annotations

import collections
import io
import json
import math
import os
import statistics as st
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

import flim_al.evidencia as ev          # noqa: E402
import flim_al.agregacao as ag          # noqa: E402

DEST = os.path.join(RAIZ, "evidencia", "campanhas",
                    "custo_do_al_2026-10-06.json")

# `teto amostrado` nao e um braco gravado: e o MAXIMO de Fbeta entre os
# candidatos sorteados da semente, exatamente como `tabela_ganho_marginal` o
# define. Replicar aqui a definicao dela e o que faz as duas tabelas falarem
# do mesmo numero.
TETO = "_teto_derivado"
PISO = "_piso_derivado"
CRITERIOS = [
    ("argmax_entropia", "argmax entropia (o AL)"),
    ("argmax_lc", "argmax least confidence"),
    ("argmax_prob", "argmax probabilidade"),
]
BASE, SORTEIO = "base", "sorteado"
NOMES = {"schisto": "Parasitas", "brats": "BraTS",
         "conjunctiva": "Conjuntivite"}


def t_pareado(d: list[float]) -> tuple[float, float]:
    n = len(d)
    if n < 2:
        return (st.mean(d) if d else float("nan"), float("nan"))
    m, s = st.mean(d), st.stdev(d)
    if s == 0:
        return m, (0.0 if m else 1.0)
    from scipy import stats
    return m, float(2 * stats.t.sf(abs(m / (s / math.sqrt(n))), n - 1))


def ic95(d: list[float]) -> tuple[float, float]:
    from scipy import stats
    h = stats.t.ppf(0.975, len(d) - 1) * st.stdev(d) / math.sqrt(len(d))
    return st.mean(d) - h, st.mean(d) + h


def main() -> int:
    # Mesmo filtro de `_gm_por_semente`: piloto e smoke ficam fora. Sem a
    # exclusao do piloto o teto do schisto sai +0,3073 em vez dos +0,3089 da
    # tabela publicada, porque o piloto acrescenta candidatos ao pool de
    # sorteados e muda a media e o maximo da semente.
    recs = [r for r in ev.carregar()
            if r.get("experimento") == "ganho_marginal"
            and "piloto" not in (r.get("fonte") or "")
            and "smoke" not in (r.get("fonte") or "")]

    # (dataset, decoder) -> semente -> criterio -> [Fbeta]
    cel: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in recs:
        v = ag._num(r.get("fb"))
        if v is None or not str(r.get("seed", "")).isdigit():
            continue
        cel[(r["dataset"], r["decoder_paper"])][int(r["seed"])][
            r["criterio"]].append(v)

    saida = {}
    print("O custo do Active Learning: teto amostrado menos o que o criterio "
          "captura")
    print("Campanha `ganho_marginal`, base funcional (Fbeta da base > 0), "
          "pareado por semente.\n")

    for (ds, dec) in sorted(cel):
        if dec not in ("FLIM_lm", "FLIM_pb"):
            continue
        # Delta de cada braco em relacao ao sorteio, dentro da semente.
        pares: dict = collections.defaultdict(dict)
        for s, c in cel[(ds, dec)].items():
            if BASE not in c or c[BASE][0] <= 0:
                continue        # base degenerada: Delta a partir de 0 nao cai
            if not c.get(SORTEIO):
                continue
            bz = c[BASE][0]
            cands = c[SORTEIO]
            ref = sum(x - bz for x in cands) / len(cands)
            # o teto: o melhor candidato sorteado, menos a media deles
            pares[s][TETO] = (max(cands) - bz) - ref
            # o piso, pelo mesmo caminho: o risco e simetrico ao premio, e a
            # dissertacao afirma as duas faixas
            pares[s][PISO] = (min(cands) - bz) - ref
            for k, _ in CRITERIOS:
                if c.get(k):
                    pares[s][k] = (sum(x - bz for x in c[k])
                                   / len(c[k])) - ref
        if not pares:
            continue

        linhas = []
        tetos = [pares[s][TETO] for s in sorted(pares) if TETO in pares[s]]
        m_t, p_t = t_pareado(tetos)
        lo, hi = ic95(tetos) if len(tetos) > 1 else (float("nan"),) * 2
        linhas.append({
            "braco": "teto amostrado (melhor regiao entre as examinadas)",
            "n_sementes": len(tetos),
            "delta_vs_sorteio": f"{m_t:+.4f}",
            "ic95": f"[{lo:+.4f}, {hi:+.4f}]",
            "p": f"{p_t:.4f}",
            "o_que_e": "NAO e estrategia: escolhe pelo Fbeta da validacao. "
                       "Mede a margem que EXISTE.",
        })
        for k, nome in CRITERIOS:
            sem = [s for s in sorted(pares) if k in pares[s]
                   and TETO in pares[s]]
            if len(sem) < 2:
                continue
            d = [pares[s][k] for s in sem]
            m, p = t_pareado(d)
            l2, h2 = ic95(d)
            lac = [pares[s][TETO] - pares[s][k] for s in sem]
            m_l, p_l = t_pareado(lac)
            cap = (m / m_t * 100.0) if m_t else float("nan")
            linhas.append({
                "braco": nome,
                "n_sementes": len(d),
                "delta_vs_sorteio": f"{m:+.4f}",
                "ic95": f"[{l2:+.4f}, {h2:+.4f}]",
                "p": f"{p:.4f}",
                "lacuna_ate_o_teto": f"{m_l:+.4f}",
                "p_lacuna": f"{p_l:.4f}",
                "fracao_do_teto_capturada": f"{cap:.0f}%",
            })

        pisos = [pares[s][PISO] for s in sorted(pares) if PISO in pares[s]]
        m_p, p_p = t_pareado(pisos)
        lo_p, hi_p = ic95(pisos) if len(pisos) > 1 else (float("nan"),) * 2
        linhas.append({
            "braco": "pior regiao (o risco, tambem usa o Fbeta da validacao)",
            "n_sementes": len(pisos),
            "delta_vs_sorteio": f"{m_p:+.4f}",
            "ic95": f"[{lo_p:+.4f}, {hi_p:+.4f}]",
            "p": f"{p_p:.4f}",
            "o_que_e": "o quanto custa UMA anotacao mal posicionada.",
        })

        chave = f"{ds}_{dec}"
        saida[chave] = {"dataset": NOMES.get(ds, ds), "decoder": dec,
                        "linhas": linhas}

        print(f"--- {NOMES.get(ds, ds)}, {dec}")
        for l in linhas:
            extra = (f"   lacuna ate o teto {l['lacuna_ate_o_teto']} "
                     f"(p={l['p_lacuna']})   captura {l['fracao_do_teto_capturada']}"
                     if "lacuna_ate_o_teto" in l else "")
            print(f"    {l['braco'][:46]:48s} n={l['n_sementes']:2d}  "
                  f"D={l['delta_vs_sorteio']}  p={l['p']}{extra}")
        print()

    # As FAIXAS que a dissertacao afirma em texto corrido. Ficam aqui porque
    # min/max sobre seis celulas e derivacao, e derivacao montada a mao e o
    # que o texto ja errou uma vez: ele traz "0,133 a 0,303" onde o registro
    # diz 0,1351 a 0,3089.
    def faixa(nome_braco, chave=None):
        vals = []
        for cel_ in saida.values():
            for l in cel_["linhas"]:
                if l["braco"].startswith(nome_braco):
                    vals.append(abs(float(l[chave or "delta_vs_sorteio"])))
        if not vals:
            return None
        return {"menor": f"{min(vals):.4f}", "maior": f"{max(vals):.4f}",
                "celulas": len(vals)}

    faixas = {
        "margem_do_clique_perfeito": faixa("teto amostrado"),
        "custo_de_uma_anotacao_mal_posicionada": faixa("pior regiao"),
        "lacuna_do_argmax_entropia": faixa("argmax entropia",
                                           "lacuna_ate_o_teto"),
    }
    print("FAIXAS sobre as 6 celulas (dataset x decodificador), em modulo:")
    for k, v in faixas.items():
        if v:
            print(f"    {k:42s} {v['menor']} a {v['maior']}  "
                  f"({v['celulas']} celulas)")
    print()

    doc = {
        "id": "custo_do_al_2026-10-06",
        "FAIXAS_PARA_O_TEXTO": faixas,
        "tipo": "RECORTE de campanha pre-registrada, nao desfecho novo",
        "o_que_e": (
            "Recorte de `ganho_marginal` que poe lado a lado o teto amostrado "
            "e o que cada criterio captura, mais a lacuna entre os dois. As "
            "duas colunas ja existiam na campanha; o que esta aqui de novo e "
            "a SUBTRACAO pareada, que e o custo do Active Learning."),
        "o_que_NAO_autoriza": (
            "Escrever que o Active Learning melhorou o Fbeta em 0,1 a 0,3. O "
            "que vale 0,1 a 0,3 e o CLIQUE PERFEITO, medido pelo teto "
            "amostrado. O criterio captura parte disso, e a parte nao "
            "capturada e o custo."),
        "limites_do_teto": (
            "O teto escolhe pelo Fbeta da VALIDACAO, entre os 22 candidatos "
            "examinados de cerca de 380 superpixels. E teto AMOSTRADO, nao "
            "absoluto. As sementes compartilham pool e validacao, entao o "
            "IC95% vale para esta validacao sob sorteio das imagens de "
            "treino e nao generaliza para o dataset."),
        "unidade_de_analise": (
            "a semente. Os candidatos de uma mesma semente compartilham "
            "encoder e sao correlacionados."),
        "fonte": "evidencia/execucoes/execucoes.csv, experimento ganho_marginal",
        "reproduz_com": "python scripts/custo_do_al.py",
        "RESULTADO": saida,
    }
    io.open(DEST, "w", encoding="utf-8").write(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"gravado em {os.path.relpath(DEST, RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
