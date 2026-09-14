#!/usr/bin/env python3
"""
varrer_criterios.py — roda VÁRIAS técnicas de Active Learning na mesma campanha

Por que uma varredura, e não uma execução por vez
    Comparar técnicas rodando uma, olhando o resultado, e decidindo a próxima
    é p-hacking — mesmo sem intenção. A lista sai daqui declarada ANTES da
    primeira execução, todas rodam, e todas entram na tabela, inclusive as que
    perderam. `--criterios` existe para restringir a lista por custo, nunca
    para escolher depois de ver o resultado.

    Este script grava a lista e o momento em que ela foi fixada, em
    `evidencia/campanhas/<id>.json`. É o pré-registro mínimo: dá para provar
    depois que a lista não mudou no meio.

Técnicas disponíveis
    imagem   entropy · least_confidence · margin · coreset · badge · random
    região   region_entropy · region_margin · region_bald · random_region

    A distinção importa e é a pergunta central da dissertação: seleção por
    IMAGEM responde "quais imagens anotar"; por REGIÃO responde "onde dentro
    da imagem". Medimos que a geometria dos seeds pesa 2,1× a seleção de
    imagens, então uma varredura que só cobre o nível de imagem responde
    metade da pergunta.

Uso
    # o que roda por padrão: todas as de imagem + as de região
    python scripts/varrer_criterios.py --plano

    # campanha real (horas de GPU — confirme antes)
    python scripts/varrer_criterios.py --splits 1 2 3 --seeds 9 --budgets 3 5 8

    # só as baratas, para validar o encanamento
    python scripts/varrer_criterios.py --criterios random entropy \\
        --splits 1 --seeds 2 --budgets 3 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.stdout.reconfigure(encoding="utf-8")

from flim_al import evidencia as ev  # noqa: E402

FLIM_AD = os.path.join(RAIZ, "flim_ad")

# A lista completa. Ordem fixa, e o nível de cada uma explícito — é o nível
# que decide com o que ela pode ser comparada.
TECNICAS = [
    ("random",           "imagem", "o piso: sorteio, sem critério"),
    ("entropy",          "imagem", "incerteza pura, entropia do mapa"),
    ("least_confidence", "imagem", "incerteza linear, 1 - |p-0.5|*2"),
    ("margin",           "imagem", "margem entre classes"),
    ("coreset",          "imagem", "cobertura, k-center guloso"),
    ("badge",            "imagem", "incerteza x diversidade, k-means++"),
    ("random_region",    "região", "controle de região: seeds sorteados"),
    ("region_entropy",   "região", "incerteza espacialmente resolvida"),
    ("region_margin",    "região", "margem por região"),
    ("region_bald",      "região", "desacordo de comitê por região"),
]
NOMES = [t[0] for t in TECNICAS]


def registrar_campanha(criterios, args) -> str:
    """Grava a lista ANTES de rodar. É o pré-registro."""
    cid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = os.path.join(RAIZ, "evidencia", "campanhas")
    os.makedirs(destino, exist_ok=True)
    pre = {
        "campanha": cid,
        "declarada_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": ev.git_commit_atual(),
        "criterios": criterios,
        "splits": args.splits,
        "seeds": args.seeds,
        "budgets": args.budgets,
        "markers": args.markers,
        "observacao": (
            "Lista fixada antes da primeira execução. Alterar depois de ver "
            "resultado invalida a comparação."),
    }
    caminho = os.path.join(destino, f"{cid}.json")
    with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(pre, fh, ensure_ascii=False, indent=2)
    return caminho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterios", nargs="+", default=NOMES, choices=NOMES)
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--budgets", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--markers", default="schisto/user_A")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_dir", default="out/varredura")
    ap.add_argument("--plano", action="store_true",
                    help="mostra o que rodaria e sai, sem executar nada")
    a = ap.parse_args()

    print(f"{len(a.criterios)} técnica(s) · splits {a.splits} · "
          f"{a.seeds} seeds · orçamentos {a.budgets}\n")
    for nome, nivel, desc in TECNICAS:
        marca = "  ✓" if nome in a.criterios else "   "
        print(f"{marca} {nome:<18} [{nivel}]  {desc}")

    execucoes = len(a.criterios) * len(a.splits) * a.seeds * len(a.budgets)
    print(f"\n{execucoes} execuções de encoder. "
          f"A ~40 s cada, isso é ~{execucoes * 40 / 3600:.1f} h de GPU.")

    if a.plano:
        print("\n--plano: nada foi executado.")
        return 0

    if execucoes > 200:
        print("\nATENÇÃO: campanha longa. Confirme com o autor antes de "
              "disparar — horas de GPU não se recuperam.")

    caminho = registrar_campanha(a.criterios, a)
    print(f"\npré-registro: {os.path.relpath(caminho, RAIZ)}\n")

    falhas = []
    for i, criterio in enumerate(a.criterios, 1):
        t0 = time.time()
        cmd = [sys.executable, os.path.join(RAIZ, "flim_al",
                                            "al_encoder_experiment.py"),
               "--acquisition", criterio,
               "--splits", *map(str, a.splits),
               "--n_seeds", str(a.seeds),
               "--budgets", *map(str, a.budgets),
               "--markers", a.markers,
               "--device", a.device,
               "--save_dir", os.path.join(a.save_dir, criterio)]
        print(f"[{i}/{len(a.criterios)}] {criterio} …", flush=True)
        r = subprocess.run(cmd, cwd=FLIM_AD)
        dur = time.time() - t0
        if r.returncode != 0:
            # Uma técnica que falha não derruba as outras: o resultado das
            # que rodaram continua valendo, e a lista declarada registra que
            # esta ficou faltando.
            falhas.append(criterio)
            print(f"    FALHOU (código {r.returncode}) após {dur:.0f}s")
        else:
            print(f"    ok em {dur:.0f}s")

    print()
    if falhas:
        print(f"{len(falhas)} técnica(s) falharam: {', '.join(falhas)}")
        print("A campanha fica incompleta — diga isso ao reportar, em vez de "
              "apresentar as que rodaram como se fossem a comparação toda.\n")

    print("Agora, nesta ordem:")
    print("  bash scripts/arquivar_resultados.sh")
    print("  python scripts/migrar_evidencia.py")
    print("  python scripts/gerar_tabelas.py")
    print("  python -m pytest")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
