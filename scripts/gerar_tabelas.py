#!/usr/bin/env python3
"""
gerar_tabelas.py — regenera todas as tabelas a partir do registro

Este é o único caminho pelo qual um número experimental chega à dissertação.
Não há passo manual entre o resultado e a tabela, e é isso que torna a
pergunta "de onde veio este número?" respondível: cada `.tex` tem um
`.proveniencia.json` ao lado listando os `run_id` de cada linha.

Uso
    python scripts/gerar_tabelas.py
    python scripts/gerar_tabelas.py --destino evidencia/tabelas

No Overleaf, cada tabela entra assim — nunca copiada:

    \\input{generated/tables/comparacao_final.tex}
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al import evidencia as ev  # noqa: E402
from flim_al import tabelas as tb  # noqa: E402

# Cada entrada: (nome, função que devolve uma Tabela).
# Uma tabela que falha não derruba as outras — a saída diz qual falhou e por
# quê, em vez de deixar o diretório pela metade sem explicação.
CATALOGO = [
    ("comparacao_final",
     lambda recs: tb.tabela_por_decoder(
         "final_comparison_v3", "comparacao_final",
         "FLIM puro contra seleção por Active Learning, por decoder", recs)),
    ("regiao_vs_imagem",
     lambda recs: tb.tabela_por_decoder(
         "al_corrected_results", "regiao_vs_imagem",
         "Active Learning por região: braços e controle, por decoder", recs)),
    ("comparacao_criterios",
     lambda recs: tb.tabela_comparacao_criterios(recs)),
    ("k_por_modelo", lambda recs: tb.tabela_k_por_modelo(recs)),
    ("al_vs_flim", lambda recs: tb.tabela_al_vs_flim(recs)),
    ("onde_marcar", lambda recs: tb.tabela_onde_marcar(recs)),
    ("artigo_vs_regiao", lambda recs: tb.tabela_artigo_vs_regiao(recs)),
    ("geral_al_pb", lambda recs: tb.tabela_geral_al(recs, decoder="FLIM_pb")),
    ("geral_al_lm", lambda recs: tb.tabela_geral_al(recs, decoder="FLIM_lm")),
    ("dissertacao_lm", lambda recs: tb.tabela_dissertacao(recs,
                                                          decoder="FLIM_lm")),
    ("dissertacao_pb", lambda recs: tb.tabela_dissertacao(recs,
                                                          decoder="FLIM_pb")),
    ("curva_orcamento", lambda recs: tb.tabela_curva_orcamento(recs)),
    # O diagnostico roda nos tres datasets e nos dois decoders. O regime
    # `colapsada` so existe no BraTS, onde 3 das 10 sementes saem com
    # Fbeta inicial zero -- uma tabela que falha por nao ter semente naquele
    # regime nao derruba as outras, o CATALOGO trata isso.
    ("ganho_marginal_schisto_lm",
     lambda recs, d="schisto", k="FLIM_lm": tb.tabela_ganho_marginal(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_mecanismo_schisto_lm",
     lambda recs, d="schisto", k="FLIM_lm": tb.tabela_ganho_mecanismo(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_marginal_schisto_pb",
     lambda recs, d="schisto", k="FLIM_pb": tb.tabela_ganho_marginal(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_mecanismo_schisto_pb",
     lambda recs, d="schisto", k="FLIM_pb": tb.tabela_ganho_mecanismo(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_marginal_brats_lm",
     lambda recs, d="brats", k="FLIM_lm": tb.tabela_ganho_marginal(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_mecanismo_brats_lm",
     lambda recs, d="brats", k="FLIM_lm": tb.tabela_ganho_mecanismo(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_marginal_brats_pb",
     lambda recs, d="brats", k="FLIM_pb": tb.tabela_ganho_marginal(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_mecanismo_brats_pb",
     lambda recs, d="brats", k="FLIM_pb": tb.tabela_ganho_mecanismo(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_marginal_conjunctiva_lm",
     lambda recs, d="conjunctiva", k="FLIM_lm": tb.tabela_ganho_marginal(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_mecanismo_conjunctiva_lm",
     lambda recs, d="conjunctiva", k="FLIM_lm": tb.tabela_ganho_mecanismo(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_marginal_conjunctiva_pb",
     lambda recs, d="conjunctiva", k="FLIM_pb": tb.tabela_ganho_marginal(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_mecanismo_conjunctiva_pb",
     lambda recs, d="conjunctiva", k="FLIM_pb": tb.tabela_ganho_mecanismo(
         recs, dataset=d, decoder=k, regime="funcional")),
    ("ganho_marginal_brats_lm_colapsada",
     lambda recs, k="FLIM_lm": tb.tabela_ganho_marginal(
         recs, dataset="brats", decoder=k, regime="colapsada")),
    ("ganho_marginal_brats_pb_colapsada",
     lambda recs, k="FLIM_pb": tb.tabela_ganho_marginal(
         recs, dataset="brats", decoder=k, regime="colapsada")),
    ("noc", lambda recs: tb.tabela_noc(recs, rotulo="noc")),
    ("noc_confirmatorio",
     lambda recs: tb.tabela_noc(recs, rotulo="confirmatorio")),
    ("custo_do_al", lambda recs: tb.tabela_custo_do_al(recs)),
    ("gerador", lambda recs: tb.tabela_gerador(recs)),
    ("proveniencia_registro",
     lambda recs: tb.tabela_resumo_registro(recs)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=tb.DESTINO)
    a = ap.parse_args()

    recs = ev.carregar()
    if not recs:
        print("registro vazio — rode scripts/migrar_evidencia.py antes")
        return 1
    print(f"{len(recs)} execuções no registro\n")

    ok, falhas = 0, []
    for nome, construir in CATALOGO:
        try:
            t = construir(recs)
            escritos = t.gravar(a.destino)
            print(f"  {nome:<26} {len(t.linhas):>3} linhas  →  "
                  f"{os.path.basename(escritos['tex'])}")
            ok += 1
        except Exception as e:                                # noqa: BLE001
            falhas.append((nome, f"{type(e).__name__}: {e}"))

    if falhas:
        print(f"\n{len(falhas)} tabela(s) não geradas:")
        for nome, erro in falhas:
            print(f"  {nome:<26} {erro}")
        if os.environ.get("TABELAS_DEBUG"):
            traceback.print_exc()

    print(f"\n{ok} tabela(s) em {os.path.relpath(a.destino, RAIZ)}")
    print("\nNo Overleaf:")
    for nome, _ in CATALOGO[:2]:
        print(f"  \\input{{{tb.PREFIXO_LATEX}/{nome}.tex}}")
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
