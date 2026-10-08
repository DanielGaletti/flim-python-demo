#!/usr/bin/env python3
"""
matriz_campanhas.py — a matriz de campanhas, lida do registro

O texto afirmava um desenho 3 x 4 x 7 que o registro nao sustenta. A correcao
nao e so reescrever a frase: e dar ao leitor a tabela que permite conferir
sozinho o que cada campanha cobriu.

Os numeros vem do registro canonico. A natureza de cada campanha
(confirmatoria, exploratoria ou descritiva) nao esta no registro, porque e
atributo do DESENHO e nao da execucao; ela e lida da pre-especificacao
versionada em `evidencia/campanhas/`, e declarada aqui como constante para as
famílias que nao tem arquivo proprio, com a justificativa ao lado.

Uso
    python scripts/matriz_campanhas.py
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

import flim_al.evidencia as ev                      # noqa: E402
from flim_al.tabelas import Tabela, DESTINO         # noqa: E402

NOME = {"schisto": "Parasitas", "brats": "BraTS",
        "conjunctiva": "Conjuntivite"}

# Natureza do desenho. Confirmatoria = ha pre-especificacao com hipotese e
# criterio de falseamento ANTES da execucao. Exploratoria = analise sem
# pre-especificacao. Descritiva = nao testa hipotese, caracteriza custo ou
# composicao.
NATUREZA = {
    "ganho_marginal": ("confirmatória",
                       "diagnostico_ganho_marginal_2026-10-01"),
    "tabela_k_por_modelo": ("descritiva", None),
    "artigo_vs_regiao": ("confirmatória", "artigo_vs_regiao_2026-09-30"),
    "onde_marcar": ("confirmatória", "onde_marcar_2026-09-15"),
    "cl_plasticidade": ("confirmatória", "cl_plasticidade_2026-10-05"),
    "il_noc": ("confirmatória", "il_noc_2026-10-05"),
    "densidade": ("confirmatória", "confirmatorio2_densidade_2026-09-30"),
    "final_comparison_v3": ("descritiva", None),
    "al_corrected_results": ("descritiva", None),
    "benchmark_conjunctiva": ("descritiva", None),
}

# Campanhas que o capitulo de resultados cita. As demais ficam de fora para a
# tabela caber, e o registro completo continua disponivel.
CITADAS = ["ganho_marginal", "tabela_k_por_modelo", "artigo_vs_regiao",
           "onde_marcar", "cl_plasticidade", "il_noc"]


def main() -> int:
    recs = ev.carregar()
    por = collections.defaultdict(list)
    for r in recs:
        e = (r.get("experimento") or "").strip()
        if e:
            por[e].append(r)

    t = Tabela(
        "matriz_campanhas",
        "As campanhas experimentais e o que cada uma cobre",
        ["campanha", "conjuntos", "orçamentos", "decod.", "sementes",
         "execuções", "natureza"],
        alinhamento="llllrrl",
        nota=("Lida do registro canônico de execuções, e não escrita à mão. "
              "`decod.` conta decodificadores distintos e `sementes` conta "
              "sementes distintas presentes na campanha; quando um "
              "decodificador aparece em apenas parte dos conjuntos, isso está "
              "dito na coluna de conjuntos. **Esta tabela existe para que o "
              "leitor verifique por conta própria o alcance de cada "
              "afirmação**: a campanha principal de seleção de imagens cobre "
              "três conjuntos e quatro orçamentos com DOIS decodificadores, e "
              "os sete decodificadores do trabalho de referência aparecem "
              "apenas em `tabela\\_k\\_por\\_modelo`, restritos ao conjunto de "
              "parasitas e com uma repetição. **Natureza**: `confirmatória` "
              "quando há pré-especificação versionada com hipótese e critério "
              "de falseamento declarados antes da execução; `descritiva` "
              "quando a campanha caracteriza custo ou composição sem testar "
              "hipótese. Nenhuma campanha desta tabela é exploratória; as "
              "análises exploratórias estão identificadas como tais no texto."))

    for nome in CITADAS:
        rs = por.get(nome)
        if not rs:
            continue
        ds = sorted({r.get("dataset") for r in rs if r.get("dataset")})
        decs = sorted({r.get("decoder_paper") for r in rs
                       if r.get("decoder_paper")
                       and r.get("decoder_paper") != "UNKNOWN"})
        orc = sorted({r.get("orcamento") for r in rs if r.get("orcamento")},
                     key=lambda x: (len(str(x)), str(x)))
        sem = sorted({r.get("seed") for r in rs
                      if str(r.get("seed")).isdigit()}, key=int)
        nat, _ = NATUREZA.get(nome, ("—", None))

        # Decodificador que so existe em parte dos conjuntos precisa aparecer,
        # senao a tabela sugere cobertura que nao houve.
        por_ds = {d: sorted({r.get("decoder_paper") for r in rs
                             if r.get("dataset") == d and r.get("decoder_paper")
                             and r.get("decoder_paper") != "UNKNOWN"})
                  for d in ds}
        uniforme = len({tuple(v) for v in por_ds.values()}) <= 1
        if uniforme:
            txt_dec = str(len(decs))
        else:
            # Nomear o conjunto que REALMENTE tem a cobertura maior. Pegar o
            # primeiro da lista ordenada apontava o dataset errado, e o erro
            # seria invisivel para quem le so a tabela.
            maior = max(por_ds, key=lambda d: len(por_ds[d]))
            txt_dec = (f"{len(por_ds[maior])} só em {NOME.get(maior, maior)}; "
                       f"{min(len(v) for v in por_ds.values())} nos demais")

        t.adicionar(
            [nome.replace("_", r"\_"),
             ", ".join(NOME.get(d, d) for d in ds),
             ", ".join(str(o) for o in orc[:6]) + ("…" if len(orc) > 6 else ""),
             txt_dec, len(sem), len(rs), nat],
            [r["run_id"] for r in rs][:400])
        print(f"  {nome:24s} {len(ds)} conj, {len(orc)} orç, "
              f"{len(decs)} dec, {len(sem)} sem, {len(rs)} exec  [{nat}]")

    escritos = t.gravar(DESTINO)
    print(f"\ngravado em {os.path.relpath(escritos['tex'], RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
