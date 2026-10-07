#!/usr/bin/env python3
"""
conferir_dissertacao.py — checagens mecânicas do texto em LaTeX

O que isto pega, e que revisão humana cansa de não pegar:

    1. citação sem entrada no .bib  (quebra a compilação ou sai como "?")
    2. entrada no .bib nunca citada (a ABNT pede que referência listada
       tenha sido usada no texto)
    3. \\ref ou \\label vazios, e \\ref para label inexistente
    4. sigla usada com \\ac sem estar declarada em abrev/
    5. travessão e meia-risca no corpo do texto

O item 5 existe porque é uma exigência de estilo deste trabalho: travessão
como recurso de pontuação deve dar lugar a vírgula, ponto, dois-pontos ou
parênteses. O hífen comum fica, porque é ortografia.

Uso
    python scripts/conferir_dissertacao.py
    python scripts/conferir_dissertacao.py --raiz DISSERTACAO
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

RAIZ_PADRAO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DISSERTACAO")


def sem_comentarios(t: str) -> str:
    """Remove linhas de comentário e o que vem depois de % não escapado."""
    saida = []
    for linha in t.split("\n"):
        pos = None
        for i, c in enumerate(linha):
            if c == "%" and (i == 0 or linha[i - 1] != "\\"):
                pos = i
                break
        saida.append(linha if pos is None else linha[:pos])
    return "\n".join(saida)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=RAIZ_PADRAO)
    a = ap.parse_args()

    texs = sorted(glob.glob(os.path.join(a.raiz, "*.tex"))
                  + glob.glob(os.path.join(a.raiz, "*", "*.tex")))
    bibs = sorted(glob.glob(os.path.join(a.raiz, "*", "*.bib")))
    if not texs:
        print(f"nenhum .tex em {a.raiz}")
        return 1

    corpo = {}
    for f in texs:
        with open(f, encoding="utf-8") as fh:
            corpo[f] = sem_comentarios(fh.read())

    chaves_bib = set()
    for b in bibs:
        with open(b, encoding="utf-8") as fh:
            chaves_bib |= set(re.findall(r"(?m)^@\w+\{\s*([^,\s]+)",
                                         fh.read()))

    citadas, labels, refs, siglas = set(), set(), [], []
    for f, t in corpo.items():
        for grupo in re.findall(r"\\cite[a-zA-Z]*\s*\{([^}]*)\}", t):
            for k in grupo.split(","):
                if k.strip():
                    citadas.add((k.strip(), f))
        labels |= set(re.findall(r"\\label\s*\{([^}]*)\}", t))
        refs += [(r, f) for r in re.findall(r"\\(?:page)?ref\s*\{([^}]*)\}", t)]
        siglas += [(s, f) for s in re.findall(r"\\ac\s*\{([^}]*)\}", t)]

    declaradas = set()
    for f in glob.glob(os.path.join(a.raiz, "pretextual", "*.tex")):
        with open(f, encoding="utf-8") as fh:
            declaradas |= set(re.findall(r"\\acro\s*\{([^}]*)\}", fh.read()))

    problemas = []

    sem_entrada = sorted({(k, os.path.basename(f))
                          for k, f in citadas if k not in chaves_bib})
    if sem_entrada:
        problemas.append(("citação sem entrada no .bib", sem_entrada))

    nunca_citadas = sorted(chaves_bib - {k for k, _ in citadas})
    if nunca_citadas:
        problemas.append(("entrada no .bib nunca citada no texto",
                          nunca_citadas))

    vazios = [(k, os.path.basename(f)) for k, f in refs if not k.strip()]
    if vazios:
        problemas.append(("\\ref vazio", vazios))

    quebradas = sorted({(k, os.path.basename(f)) for k, f in refs
                        if k.strip() and k not in labels})
    if quebradas:
        problemas.append(("\\ref para label inexistente", quebradas))

    sem_declarar = sorted({(s, os.path.basename(f)) for s, f in siglas
                           if s not in declaradas}) if declaradas else []
    if sem_declarar:
        problemas.append(("\\ac de sigla não declarada em abrev/",
                          sem_declarar))

    # ── erros que so aparecem na compilacao ────────────────────────────────
    #
    # Estes nao sao estilo: sao defeitos que o BibTeX ou o LaTeX rejeitam, e
    # que nenhuma leitura do .tex revela. O primeiro derrubou a primeira
    # tentativa de compilar esta dissertacao no Overleaf.
    autores = []
    for b_ in bibs:
        with open(b_, encoding="utf-8") as fh:
            texto = fh.read()
        for m in re.finditer(r"@\w+\{([^,]+),(.*?)(?=\n@|\Z)", texto, re.S):
            chave = m.group(1).strip()
            for campo in ("author", "editor"):
                c = re.search(campo + r"\s*=\s*\{(.*?)\}\s*,?\s*\n",
                              m.group(2), re.S)
                if not c:
                    continue
                v = c.group(1)
                # No BibTeX a virgula separa sobrenome de nome DENTRO de um
                # autor; quem separa autores e " and ". Uma lista escrita com
                # virgulas vira um nome so, e o BibTeX para com "Too many
                # commas in name".
                if " and " not in v and v.count(",") >= 2:
                    autores.append((chave, campo, v.strip()[:60]))
    if autores:
        problemas.append(
            ("lista de autores com virgula em vez de \" and \" (o BibTeX para)",
             autores))

    # Arquivo referenciado que nao existe, ou existe com outra caixa.
    #
    # O Overleaf compila em Linux, que diferencia maiuscula de minuscula; o
    # Windows nao. Uma figura gravada como `Oracle_Superpixel.PNG` e
    # referenciada como `.png` compila na maquina local e falha no Overleaf,
    # com um erro que fala de arquivo inexistente e nao de caixa. Conferir
    # isto e o que uma maquina sem pdflatex ainda consegue fazer.
    no_disco = set()
    for r_, _, fs in os.walk(a.raiz):
        for f in fs:
            no_disco.add(os.path.relpath(os.path.join(r_, f), a.raiz)
                         .replace(os.sep, "/"))
    por_minuscula = {d.lower(): d for d in no_disco}

    # onde o \includegraphics procura
    graf = re.search(r"\\graphicspath\s*\{\s*\{([^}]*)\}", "\n".join(
        corpo.values()))
    pasta_fig = (graf.group(1).lstrip("./").rstrip("/") + "/") if graf else ""

    faltando = []
    for f, t in corpo.items():
        alvos = []
        for cmd, suf in (("include", ".tex"), ("input", ""),
                         ("listasiglas", ".tex"), ("bibliography", ".bib")):
            for m in re.findall(r"\\" + cmd + r"\s*\{([^}]+)\}", t):
                alvos.append(m if m.endswith((".tex", ".bib")) else m + suf)
        for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            alvos.append(pasta_fig + m)
        for alvo in alvos:
            if alvo in no_disco:
                continue
            outra = por_minuscula.get(alvo.lower())
            faltando.append((os.path.basename(f), alvo,
                             f"no disco e {outra}" if outra else "nao existe"))
    if faltando:
        problemas.append(
            ("arquivo referenciado que falta, ou com a caixa trocada "
             "(o Overleaf compila em Linux)", faltando))

    # `\ref` para um label que existe mas em arquivo nao incluido no main
    principal = [f for f in texs
                 if re.search(r"\\begin\{document\}", corpo[f])]
    if len(principal) == 1:
        p = principal[0]
        # Procurar o CAMINHO no texto do principal, e nao so dentro de
        # `\include` ou `\input`: a lista de siglas entra por
        # `\listasiglas{abrev/Abreviaturas}`, e classes ABNT tem varios
        # comandos assim. Casar so os dois comandos acusava a lista de siglas
        # como orfa, que e falso.
        orfaos = []
        for f in texs:
            if f == p:
                continue
            rel = os.path.relpath(f, a.raiz).replace(os.sep, "/")
            if rel.startswith("tabelas/"):
                continue        # tabelas entram pelos capitulos, nao pelo main
            if rel.startswith("notas/"):
                continue        # anotacao de reuniao, nao faz parte do texto
            if rel[:-4] in corpo[p] or rel in corpo[p]:
                continue
            # tambem vale ser incluido por um capitulo
            if any(rel[:-4] in corpo[o] or rel in corpo[o]
                   for o in texs if o != f):
                continue
            orfaos.append(os.path.basename(f))
        if orfaos:
            problemas.append(
                ("arquivo .tex que nao e incluido pelo documento principal",
                 orfaos))

    # Letra grega e simbolo matematico em modo texto. A checagem vive em
    # `conferir_simbolos.py`, que tem a logica de remover modo matematico
    # antes de procurar; aqui ela e apenas acionada, para que uma unica
    # chamada cubra tudo o que impede a compilacao.
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable,
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "conferir_simbolos.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            problemas.append(
                ("letra grega ou simbolo matematico em modo texto "
                 "(o pdflatex recusa)",
                 [l for l in (r.stdout or "").split("\n")
                  if l.strip().startswith("linha ")
                  or l.strip().startswith("== ")]))
    except Exception as e:                      # nunca derrubar o verificador
        print(f"(aviso: checagem de simbolos nao rodou: {e})")

    travessoes = []
    for f, t in corpo.items():
        for n, linha in enumerate(t.split("\n"), 1):
            if "—" in linha or "–" in linha:
                travessoes.append((f"{os.path.basename(f)}:{n}",
                                   linha.strip()[:70]))
    if travessoes:
        problemas.append(("travessão ou meia-risca no corpo", travessoes))

    print(f"{len(texs)} arquivo(s) .tex · {len(chaves_bib)} entrada(s) no .bib "
          f"· {len({k for k, _ in citadas})} chave(s) citada(s)\n")
    if not problemas:
        print("nenhum problema mecânico encontrado.")
        return 0
    for titulo, itens in problemas:
        print(f"== {titulo}: {len(itens)}")
        for it in itens[:15]:
            print(f"   {it}")
        if len(itens) > 15:
            print(f"   ... e {len(itens) - 15} outro(s)")
        print()
    # Entrada nao citada e aviso, nao erro: o .bib do modelo vem povoado.
    graves = [t for t, _ in problemas
              if t != "entrada no .bib nunca citada no texto"]
    return 1 if graves else 0


if __name__ == "__main__":
    raise SystemExit(main())
