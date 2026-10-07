#!/usr/bin/env python3
"""
conferir_simbolos.py — letra grega e simbolo matematico em MODO TEXTO

A classe `ufscar.cls` carrega `[utf8]{inputenc}` com `[T1]{fontenc}`. Essa
combinacao resolve acentuacao latina mas NAO letra grega: um `beta` solto no
corpo do texto derruba o `pdflatex` com "Unicode character not set up for use
with LaTeX". Em modo matematico, `$\\beta$`, funciona.

O gerador de tabelas ja converte esses simbolos na geracao
(`flim_al.tabelas._SIMBOLOS_TEX`). O corpo dos capitulos nao passa por ele, e
e ai que o problema aparece: o texto foi escrito com `Fbeta` colado direto.

Esta checagem remove modo matematico antes de procurar, para nao acusar o que
esta correto.

Uso
    python scripts/conferir_simbolos.py
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISS = os.path.join(RAIZ, "DISSERTACAO")

GREGO = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
SIMBOLOS = "≠≤≥×÷−±∂√∈∉∑∏∞⊂⊆∩∪←→↔⇒⇔₀₁₂₃₄₅₆₇₈₉⁰¹²³"
SUSPEITOS = GREGO + SIMBOLOS


def sem_matematica(t: str) -> str:
    """Remove o que esta em modo matematico, onde o simbolo e valido."""
    t = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", t,
               flags=re.S)
    t = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", t, flags=re.S)
    # comentarios tambem nao compilam
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    return t


def main() -> int:
    alvos = (sorted(glob.glob(os.path.join(DISS, "*.tex")))
             + sorted(glob.glob(os.path.join(DISS, "*", "*.tex"))))
    total = 0
    for f in alvos:
        rel = os.path.relpath(f, DISS).replace(os.sep, "/")
        if rel.startswith(("tabelas/", "notas/")):
            continue        # tabelas sao geradas e ja convertem; notas nao compilam
        t = sem_matematica(io.open(f, encoding="utf-8",
                                   errors="replace").read())
        ocorr = []
        for n, linha in enumerate(t.split("\n"), 1):
            for c in linha:
                if c in SUSPEITOS:
                    ocorr.append((n, c, linha.strip()[:72]))
        if ocorr:
            total += len(ocorr)
            print(f"\n== {rel}: {len(ocorr)} ocorrencia(s)")
            vistos = set()
            for n, c, ctx in ocorr[:8]:
                if (n, c) in vistos:
                    continue
                vistos.add((n, c))
                print(f"   linha {n}: '{c}'  ...{ctx}")
            if len(ocorr) > 8:
                print(f"   ... e mais {len(ocorr) - 8}")

    if total:
        print(f"\nTOTAL: {total} simbolo(s) em modo texto. O pdflatex recusa "
              f"cada um deles.")
        print("Correcao: envolver em modo matematico, por exemplo "
              "$F_\\beta$ no lugar de Fbeta colado.")
        return 1
    print("ok: nenhuma letra grega ou simbolo matematico em modo texto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
