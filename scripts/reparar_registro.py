#!/usr/bin/env python3
"""
reparar_registro.py — repara o registro pela UNIÃO com cópias conhecidas

Quando isto é necessário
    `registrar` lia o CSV inteiro e o reescrevia inteiro, sem trava. Com duas
    campanhas rodando ao mesmo tempo, a leitura de uma podia intercalar com a
    escrita da outra e a segunda sobrescrevia as linhas que a primeira acabara
    de gravar. Em 2026-10-01 isso custou 293 registros, 17 deles de uma
    campanha já concluída (`artigo_vs_regiao`).

    A trava em `evidencia._trava` impede que volte a acontecer. Este script
    conserta o dano já feito.

Por que a união é segura
    `run_id` é derivado dos campos identificadores, e o treino é determinístico
    (verificado: três repetições dão Fβ idêntico até a décima casa). Então duas
    cópias do mesmo `run_id` descrevem a mesma execução, e unir não inventa
    dado nenhum — só devolve linhas que existiam.

    Quando as duas cópias divergem, o script **não escolhe em silêncio**:
    prefere a menos truncada (mais campos preenchidos), imprime o que fez, e
    aborta se a divergência for numa MÉTRICA. Métrica divergente para o mesmo
    `run_id` significa que algo mudou fora da configuração registrada, e isso
    é uma questão científica, não de arquivo.

Uso
    python scripts/reparar_registro.py --com backup.csv --conferir
    python scripts/reparar_registro.py --com backup.csv a.csv
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al import evidencia as ev  # noqa: E402

# Campos de tempo/orçamento: divergência aqui é truncamento de escrita, e a
# cópia mais completa ganha. Divergência em MÉTRICA é outra coisa — aborta.
METADADO = {"segundos", "segundos_treino", "segundos_aval", "orcamento_px",
            "gravado_em", "git_commit"}


def _preenchidos(r: dict) -> int:
    return sum(1 for c in ev.CAMPOS if str(r.get(c, "")).strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvo", default=None, help="padrão: registro canônico")
    ap.add_argument("--com", nargs="+", required=True,
                    help="cópias de onde recuperar linhas perdidas")
    ap.add_argument("--conferir", action="store_true",
                    help="só relata; não escreve")
    a = ap.parse_args()

    alvo = a.alvo or ev.ARQUIVO
    atual = {r["run_id"]: r for r in ev.carregar(alvo)}
    print(f"alvo: {alvo}  ({len(atual)} execuções)")

    recuperados, conflitos_metrica, preferidos = {}, [], []
    for copia in a.com:
        if not os.path.isfile(copia):
            print(f"  {copia}: NÃO EXISTE")
            continue
        outros = ev.carregar(copia)
        novos = 0
        for r in outros:
            rid = r["run_id"]
            if rid not in atual and rid not in recuperados:
                recuperados[rid] = r
                novos += 1
                continue
            base = atual.get(rid) or recuperados.get(rid)
            difs = [c for c in ev.CAMPOS
                    if str(base.get(c, "")) != str(r.get(c, ""))]
            if not difs:
                continue
            # Vazio contra preenchido é TRUNCAMENTO, não conflito. Uma escrita
            # interrompida deixa o resto da linha em branco — foi o que
            # aconteceu com 474d88f24de2ece5 em 2026-10-01, cujas métricas
            # ficaram vazias no arquivo e estão intactas no git. Tratar isso
            # como divergência científica bloquearia justamente o reparo.
            # Conflito de verdade é quando os DOIS lados têm valor e diferem.
            metricas = [
                c for c in difs
                if c in ev.NUMERICOS and c not in METADADO
                and str(base.get(c, "")).strip()
                and str(r.get(c, "")).strip()
            ]
            if metricas:
                conflitos_metrica.append((rid, metricas, copia))
                continue
            # truncamento: a cópia mais completa ganha
            if _preenchidos(r) > _preenchidos(base):
                if rid in atual:
                    atual[rid] = r
                else:
                    recuperados[rid] = r
                preferidos.append((rid, difs))
        print(f"  {copia}: {len(outros)} execuções, "
              f"{novos} recuperáveis")

    print(f"\nlinhas a recuperar:     {len(recuperados)}")
    print(f"linhas destruncadas:    {len(preferidos)}")
    for rid, difs in preferidos[:5]:
        print(f"    {rid}: {', '.join(difs)}")

    if conflitos_metrica:
        print(f"\n{len(conflitos_metrica)} CONFLITO(S) DE MÉTRICA — "
              f"nada foi escrito:")
        for rid, cs, copia in conflitos_metrica[:10]:
            print(f"    {rid}  campos: {', '.join(cs)}  em {copia}")
        print("\nMétrica divergente no mesmo run_id não é problema de arquivo. "
              "Investigue antes de reparar.")
        return 2

    if a.conferir:
        print("\n--conferir: nada escrito.")
        return 0
    if not recuperados and not preferidos:
        print("\nnada a fazer.")
        return 0

    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sombra = f"{alvo}.antes_do_reparo_{marca}"
    shutil.copy2(alvo, sombra)
    print(f"\ncópia do estado danificado: {sombra}")

    tudo = list(atual.values()) + list(recuperados.values())

    # O reparo NAO passa por `registrar`, e isto e deliberado: `validar`
    # recomputa o run_id a partir de CAMPOS_ID, e o arquivo carrega registros
    # gravados sob um CAMPOS_ID anterior, cujo id nao bate mais. `registrar` so
    # valida o que ENTRA, nunca o que ja esta no arquivo, e foi por isso que
    # eles sobreviveram. Restaurar linha que existia nao pode se autoinvalidar
    # num criterio que a linha nunca atendeu.
    #
    # Em vez de portao, censo: quantas linhas nao batem com o CAMPOS_ID atual.
    # Isso e informacao sobre a proveniencia do arquivo, e tem de ficar visivel.
    legado = 0
    for r in tudo:
        try:
            ev.validar(r)
        except ev.RegistroInvalido:
            legado += 1
    if legado:
        print(f"\n{legado} de {len(tudo)} execuções têm run_id gravado sob um "
              f"CAMPOS_ID anterior ao atual.")
        print("  Elas já estavam assim no arquivo; o reparo não as altera.")
        print("  Consequência: o run_id delas identifica a execução dentro da "
              "campanha que as gravou, mas não é recomputável hoje.")

    with ev._trava(alvo):
        ev._escrever_atomico(alvo, tudo)
    print(f"\ngravado: {len(tudo)} execuções")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
