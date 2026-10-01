#!/usr/bin/env python3
"""
mesclar_registro.py — traz um registro paralelo para dentro do canônico

Por que existe
    `evidencia.registrar` lê o CSV inteiro e o reescreve inteiro. Duas
    campanhas simultâneas no mesmo arquivo disputam essa escrita, e desde que
    a trava existe elas se serializam — mas serializar custa espera, e uma
    campanha que roda em paralelo com outra já em andamento (iniciada por um
    processo que não conhece a trava) ainda corre risco.

    O caminho seguro é cada campanha escrever no seu próprio CSV e o merge
    acontecer uma vez, no fim, sob a trava.

O que este script garante
    Nada entra duas vezes: a deduplicação é por `run_id`, a mesma do
    `registrar`. E nada entra em silêncio: divergência de métrica no mesmo
    `run_id` é impressa, nunca sobrescrita, porque duas medidas diferentes
    para a mesma configuração significam que algo mudou fora do que o registro
    descreve.

Uso
    python scripts/mesclar_registro.py --de evidencia/tmp/reg_brats.csv
    python scripts/mesclar_registro.py --de a.csv b.csv --conferir
"""
from __future__ import annotations

import argparse
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al import evidencia as ev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--de", nargs="+", required=True,
                    help="CSVs de origem, no formato canônico")
    ap.add_argument("--para", default=None, help="destino (padrão: canônico)")
    ap.add_argument("--conferir", action="store_true",
                    help="só relata; não escreve nada")
    a = ap.parse_args()

    destino = a.para or ev.ARQUIVO
    antes = len(ev.carregar(destino))
    print(f"destino: {destino}  ({antes} execuções)")

    total_novos = 0
    for origem in a.de:
        if not os.path.isfile(origem):
            print(f"  {origem}: NÃO EXISTE — ignorado")
            continue
        recs = ev.carregar(origem)
        if not recs:
            print(f"  {origem}: vazio")
            continue
        fontes = sorted({r.get("fonte", "") for r in recs})
        print(f"  {origem}: {len(recs)} execuções, fonte(s): "
              + ", ".join(fontes[:3]) + ("..." if len(fontes) > 3 else ""))
        if a.conferir:
            existentes = {r["run_id"] for r in ev.carregar(destino)}
            novos = [r for r in recs if r["run_id"] not in existentes]
            print(f"    entrariam {len(novos)} novos, "
                  f"{len(recs) - len(novos)} já presentes")
            continue
        res = ev.registrar(recs, arquivo=destino)
        total_novos += res["novos"]
        print(f"    novos {res['novos']}  duplicados {res['duplicados']}  "
              f"total {res['total']}")
        if res["divergentes"]:
            # Nunca silenciar: run_id igual com métrica diferente quer dizer
            # que algo fora da configuração registrada mudou.
            print(f"    DIVERGENTES ({len(res['divergentes'])}): "
                  f"{res['divergentes'][:5]}")
            print("    o registrar NÃO sobrescreveu; confira antes de "
                  "agregar estes run_id")

    if not a.conferir:
        print(f"\nacrescentados {total_novos}; destino agora com "
              f"{len(ev.carregar(destino))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
