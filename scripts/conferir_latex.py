#!/usr/bin/env python3
"""
conferir_latex.py — o que a compilacao pegaria, sem ter pdflatex

Este ambiente nao tem distribuicao LaTeX. A compilacao, portanto, NAO foi
testada, e isso precisa estar dito em qualquer relato. O que esta checagem faz
e cobrir mecanicamente as classes de erro que o `pdflatex` acusaria e que uma
leitura humana deixa passar.

O que ela pega:

  1. comando indefinido: usado no texto, sem \\newcommand no projeto e fora da
     lista de comandos conhecidos do LaTeX, da classe e dos pacotes carregados.
     Foi assim que `\\fbeta` ficou no capitulo 3 sem ninguem notar: ele so
     apareceria como `Undefined control sequence` na primeira compilacao.
  2. ambiente aberto e nao fechado, ou fechado sem abrir
  3. pacote carregado duas vezes, que o LaTeX acusa quando as opcoes diferem
  4. chaves desbalanceadas na linha

O que ela NAO pega, e por isso compilacao continua sendo necessaria: overfull
boxes, colisao de floats, erros de fonte, e qualquer coisa que dependa do
estado do compilador.

Uso
    python scripts/conferir_latex.py
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

# Comandos que existem no LaTeX base, nos pacotes carregados pelo documento, ou
# na classe. Nao e exaustivo: e a lista do que este documento usa.
CONHECIDOS = set("""
documentclass usepackage input include includegraphics begin end
chapter section subsection subsubsection paragraph subparagraph
label ref pageref cite citeonline citeauthor citeyear bibliography
textbf textit texttt emph textsc underline textsuperscript textsubscript
footnote caption centering item itemize enumerate description
newpage clearpage cleardoublepage pagebreak linebreak newline
hline toprule midrule bottomrule cmidrule multicolumn multirow
frac sum prod int lim log exp sin cos tan max min arg sqrt
alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi
pi rho sigma tau upsilon phi chi psi omega varepsilon vartheta varphi
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
leq geq neq approx equiv sim propto times div pm mp cdot cdots ldots dots
in notin subset subseteq supset cup cap setminus emptyset forall exists
rightarrow leftarrow Rightarrow Leftarrow leftrightarrow mapsto
mathcal mathbb mathbf mathrm mathit text operatorname
left right big Big bigg Bigg langle rangle lVert rVert lvert rvert
quad qquad hspace vspace hfill vfill noindent indent par
ac acs acl acro acrodef listasiglas
titulo autor orientador coorientador maketitle
resumo abstract tableofcontents listailustracoes listatabelas
adjustbox resizebox scalebox rotatebox
small footnotesize scriptsize tiny normalsize large Large LARGE huge Huge
color textcolor colorbox fcolorbox definecolor
url href texorpdfstring
linewidth textwidth columnwidth textheight baselineskip
hat bar tilde vec dot ddot overline underline
ast star bullet circ diamond
newcommand renewcommand providecommand def let
bigcup bigcap bigoplus otimes oplus partial nabla infty
tfrac dfrac binom choose overbrace underbrace
checkmark textasciitilde textbackslash textasciicircum
rule raisebox makebox framebox parbox minipage
graphicspath DeclareGraphicsExtensions
mainmatter frontmatter backmatter appendix
makeindex printindex makenomenclature printnomenclature nomenclature
bookmarksetup hypersetup phantomsection addcontentsline
setlength setcounter addtolength renewenvironment newenvironment
arraystretch tabcolsep arrayrulewidth extrarowheight
turn sidewaystable landscape
""".split())

# Comandos do `abntex2`, que a classe `ufscar` herda via \LoadClass. Sem esta
# lista o verificador acusa `\data` e `\local` do proprio modelo da UFSCar, que
# sao legitimos, e o ruido faria a checagem ser ignorada.
CONHECIDOS |= set("""
data local instituicao tipotrabalho programa area orientador coorientador
preambulo fonte nota imprimirtitulo imprimirautor imprimirlocal imprimirdata
imprimirorientador imprimirinstituicao imprimirtipotrabalho imprimirprograma
anexo apendice errata folhadeaprovacao dedicatoria agradecimentos epigrafe
listadesiglas listadesimbolos autorrubrica
ABNTEXchapterfont ABNTEXsectionfont textual pretextual postextual
""".split())

ABRE = re.compile(r"\\begin\{([^}]+)\}")
FECHA = re.compile(r"\\end\{([^}]+)\}")
COMANDO = re.compile(r"\\([a-zA-Z]+)")
PACOTE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}")


def sem_comentarios(t: str) -> str:
    out = []
    for linha in t.split("\n"):
        p = None
        for i, c in enumerate(linha):
            if c == "%" and (i == 0 or linha[i - 1] != "\\"):
                p = i
                break
        out.append(linha if p is None else linha[:p])
    return "\n".join(out)


def main() -> int:
    arquivos = (sorted(glob.glob(os.path.join(DISS, "*.tex")))
                + sorted(glob.glob(os.path.join(DISS, "*", "*.tex"))))
    arquivos = [f for f in arquivos
                if "notas" + os.sep not in f]
    corpo = {f: sem_comentarios(io.open(f, encoding="utf-8",
                                        errors="replace").read())
             for f in arquivos}
    tudo = "\n".join(corpo.values())

    # comandos definidos pelo proprio projeto
    definidos = set(re.findall(
        r"\\(?:new|renew|provide)command\s*\{?\\([a-zA-Z]+)", tudo))
    cls = os.path.join(DISS, "ufscar.cls")
    if os.path.isfile(cls):
        tcls = io.open(cls, encoding="utf-8", errors="replace").read()
        definidos |= set(re.findall(
            r"\\(?:new|renew|provide)command\s*\{?\\([a-zA-Z]+)", tcls))
        definidos |= set(re.findall(r"\\def\s*\\([a-zA-Z]+)", tcls))
        definidos |= set(re.findall(
            r"\\DeclareRobustCommand\s*\{?\\([a-zA-Z]+)", tcls))
        # ambientes da classe tambem definem \nome
        definidos |= set(re.findall(r"\\newenvironment\s*\{([^}]+)\}", tcls))

    problemas = []

    # 1. comando indefinido
    indefinidos = {}
    for f, t in corpo.items():
        if f.endswith("ufscar.cls"):
            continue
        for n, linha in enumerate(t.split("\n"), 1):
            for cmd in COMANDO.findall(linha):
                if cmd in CONHECIDOS or cmd in definidos:
                    continue
                indefinidos.setdefault(cmd, (os.path.basename(f), n,
                                             linha.strip()[:70]))
    if indefinidos:
        problemas.append(("comando possivelmente indefinido",
                          [f"\\{c} em {v[0]}:{v[1]}"
                           for c, v in sorted(indefinidos.items())]))

    # 2. ambiente desbalanceado
    for f, t in corpo.items():
        pilha, erros = [], []
        for n, linha in enumerate(t.split("\n"), 1):
            for e in ABRE.findall(linha):
                pilha.append((e, n))
            for e in FECHA.findall(linha):
                if not pilha:
                    erros.append(f"\\end{{{e}}} sem abertura, linha {n}")
                elif pilha[-1][0] != e:
                    erros.append(f"\\end{{{e}}} fecha \\begin{{{pilha[-1][0]}}}"
                                 f" da linha {pilha[-1][1]}")
                    pilha.pop()
                else:
                    pilha.pop()
        for e, n in pilha:
            erros.append(f"\\begin{{{e}}} da linha {n} nunca fecha")
        if erros:
            problemas.append((f"ambiente desbalanceado em "
                              f"{os.path.basename(f)}", erros))

    # 3. pacote duplicado
    pacotes = {}
    for f, t in corpo.items():
        for grupo in PACOTE.findall(t):
            for p in grupo.split(","):
                p = p.strip()
                if p:
                    pacotes.setdefault(p, []).append(os.path.basename(f))
    dup = {p: v for p, v in pacotes.items() if len(v) > 1}
    if dup:
        problemas.append(("pacote carregado mais de uma vez",
                          [f"{p}: {', '.join(v)}" for p, v in sorted(dup.items())]))

    # 4. chaves desbalanceadas no ARQUIVO
    #
    # Por linha nao funciona: `\caption{...}` de varias linhas fecha numa linha
    # que nao abriu, e isso e correto. O que importa e o balanco do arquivo
    # inteiro, e, dentro dele, o ponto em que a contagem fica NEGATIVA, que e
    # onde ha uma chave fechando sem par.
    desbal = []
    for f, t in corpo.items():
        limpa = re.sub(r"\\[{}]", "", t)
        saldo, linha_neg = 0, None
        for n, linha in enumerate(limpa.split("\n"), 1):
            saldo += linha.count("{") - linha.count("}")
            if saldo < 0 and linha_neg is None:
                linha_neg = n
        if saldo != 0 or linha_neg is not None:
            onde = (f" (fica negativo na linha {linha_neg})"
                    if linha_neg else "")
            desbal.append(f"{os.path.basename(f)}: saldo {saldo:+d}{onde}")
    if desbal:
        problemas.append(("chaves desbalanceadas no arquivo", desbal))

    print(f"{len(arquivos)} arquivo(s) .tex · {len(definidos)} comando(s) "
          f"definido(s) no projeto\n")
    if not problemas:
        print("ok: nenhum erro estrutural de LaTeX encontrado.")
        print("\nATENCAO: este ambiente NAO tem pdflatex. A compilacao nao foi")
        print("testada. Esta checagem cobre comando indefinido, ambiente")
        print("desbalanceado, pacote duplicado e chave desbalanceada; ela NAO")
        print("cobre overfull box, colisao de float nem erro de fonte.")
        return 0
    for titulo, itens in problemas:
        print(f"== {titulo}: {len(itens)}")
        for i in itens[:20]:
            print(f"   {i}")
        if len(itens) > 20:
            print(f"   ... e mais {len(itens) - 20}")
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
