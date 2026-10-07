#!/usr/bin/env python3
"""
tabelas.py — resultados agregados viram tabela, com proveniência

Nenhum número desta saída é digitado. Cada célula vem de `agregacao.py`, que
vem de `evidencia/execucoes/`, que vem de uma execução com `run_id`. A cadeia
é navegável nos dois sentidos:

    célula da tabela → run_ids → execuções → config/seed/commit → hipótese
    hipótese         → execuções → run_ids → em que tabelas aparece

Ao lado de cada tabela sai um `.proveniencia.json` com essa lista. É ele que
responde "de onde veio este número?" sem depender de ninguém lembrar.

Formatos
    .csv    para inspeção e para outros scripts
    .md     para os documentos de estado do projeto
    .tex    para \\input{} no Overleaf — o número nunca é copiado à mão
    .proveniencia.json

Três regras de honestidade, impostas em código
    1. **Negrito só com suporte estatístico.** `melhor_numerico` e
       `suporte_estatistico` são colunas distintas, e só a segunda vira ênfase
       tipográfica. Marcar o maior número em negrito quando a diferença está
       dentro do ruído é afirmar, com a tipografia, algo que os dados não
       sustentam.
    2. **`n` sempre visível.** Média de 2 e média de 9 não se parecem numa
       tabela que esconde o n.
    3. **Sem filtro por sinal.** Nenhuma função aqui remove linha por ter dado
       resultado ruim.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys
from datetime import datetime, timezone

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from flim_al import agregacao as ag  # noqa: E402
from flim_al import evidencia as ev  # noqa: E402

DESTINO = os.path.join(RAIZ, "evidencia", "tabelas")

# Onde o Overleaf espera encontrar os .tex. O caminho vive aqui para que, ao
# mudar, mude num lugar só e não em cada \input{} da dissertação.
PREFIXO_LATEX = "generated/tables"


def _fmt(v, casas: int = 3) -> str:
    """
    Número ou travessão — nunca zero para representar ausência.

    Um "0.000" numa célula que na verdade não tem medida é indistinguível de
    uma medida que deu zero, e as duas coisas levam a leituras opostas.
    """
    if v is None:
        return "—"
    return f"{v:.{casas}f}"


# Símbolos que o texto usa e que o pdflatex NÃO aceita como caractere.
#
# A classe `ufscar.cls` carrega `[utf8]{inputenc}` com `[T1]{fontenc}`. Essa
# combinação resolve acentuação latina, mas não letras gregas nem sinais
# matemáticos: um α solto para a compilação com "Unicode character not set up
# for use with LaTeX". Como as tabelas vão para o Overleaf, que usa pdflatex
# por padrão, a conversão acontece aqui, na geração, e não no documento.
#
# A travessão vira `---`, que é a forma canônica em LaTeX e não depende de
# inputenc ter a entrada correspondente.
_SIMBOLOS_TEX = {
    "α": r"$\alpha$", "β": r"$\beta$", "ρ": r"$\rho$", "Δ": r"$\Delta$",
    "σ": r"$\sigma$", "μ": r"$\mu$", "χ": r"$\chi$",
    "≠": r"$\neq$", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
    "−": "-", "–": "-", "—": "---",
    "₀": r"$_0$", "₁": r"$_1$", "²": r"$^2$", "³": r"$^3$",
}


def _tex_escape(s: str) -> str:
    """
    Escapa para LaTeX e troca símbolos que o pdflatex não aceita.

    A ordem importa: a barra invertida sai primeiro, senão ela escaparia as
    barras que a própria substituição de símbolos acabou de inserir.
    """
    s = str(s)
    for de, para in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                     ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(de, para)
    for de, para in _SIMBOLOS_TEX.items():
        s = s.replace(de, para)
    return s


# ── construção de tabela ────────────────────────────────────────────────────

class Tabela:
    """
    Uma tabela com cabeçalho, linhas e a lista de run_ids de cada linha.

    A proveniência não é opcional: `adicionar` exige os ids. Uma linha sem
    origem rastreável não entra, porque é exatamente a linha que ninguém
    consegue conferir depois.
    """

    def __init__(self, nome: str, titulo: str, colunas: list,
                 nota: str = "", alinhamento: str = None):
        self.nome = nome
        self.titulo = titulo
        self.colunas = colunas
        self.nota = nota
        self.alinhamento = alinhamento or ("l" + "r" * (len(colunas) - 1))
        self.linhas = []
        self.proveniencia = []

    def adicionar(self, valores: list, run_ids: list, enfase: int = None):
        if len(valores) != len(self.colunas):
            raise ValueError(
                f"{len(valores)} valores para {len(self.colunas)} colunas")
        if not run_ids:
            raise ValueError(
                "linha sem run_ids: toda célula precisa ser rastreável até a "
                "execução que a produziu")
        self.linhas.append({"valores": valores, "enfase": enfase})
        self.proveniencia.append({"linha": len(self.linhas),
                                  "rotulo": str(valores[0]),
                                  "run_ids": sorted(set(run_ids))})

    # ── saídas ──────────────────────────────────────────────────────────────

    def csv(self) -> str:
        import io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.colunas)
        for l in self.linhas:
            w.writerow(l["valores"])
        return buf.getvalue()

    def markdown(self) -> str:
        sep = "|" + "|".join("---" for _ in self.colunas) + "|"
        out = [f"### {self.titulo}", "",
               "| " + " | ".join(self.colunas) + " |", sep]
        for l in self.linhas:
            cels = [str(v) for v in l["valores"]]
            if l["enfase"] is not None:
                i = l["enfase"]
                cels[i] = f"**{cels[i]}**"
            out.append("| " + " | ".join(cels) + " |")
        if self.nota:
            out += ["", self.nota]
        return "\n".join(out) + "\n"

    def latex(self) -> str:
        out = [
            f"% {self.titulo}",
            f"% GERADO por flim_al/tabelas.py em "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "% NAO EDITE A MAO: a proxima geracao sobrescreve, e o numero"
            " editado deixa de bater com evidencia/execucoes/.",
            f"% Proveniencia: {self.nome}.proveniencia.json",
            r"\begin{tabular}{" + self.alinhamento + "}",
            r"\hline",
            " & ".join(_tex_escape(c) for c in self.colunas) + r" \\",
            r"\hline",
        ]
        for l in self.linhas:
            cels = [_tex_escape(str(v)) for v in l["valores"]]
            if l["enfase"] is not None:
                i = l["enfase"]
                cels[i] = r"\textbf{" + cels[i] + "}"
            out.append(" & ".join(cels) + r" \\")
        out += [r"\hline", r"\end{tabular}"]
        return "\n".join(out) + "\n"

    def latex_nota(self) -> str:
        """
        A nota da tabela, pronta para entrar logo abaixo do float.

        O markdown da nota usa `**negrito**` e crase para código; aqui os dois
        viram `\\textbf` e `\\texttt`. A conversão é feita antes do escape de
        LaTeX, senão o escape transformaria a própria marcação.
        """
        if not self.nota:
            return "% (esta tabela nao tem nota)\n"
        t = self.nota
        partes, i = [], 0
        while True:
            a = t.find("**", i)
            if a < 0:
                partes.append(("txt", t[i:]))
                break
            b = t.find("**", a + 2)
            if b < 0:
                partes.append(("txt", t[i:]))
                break
            partes.append(("txt", t[i:a]))
            partes.append(("neg", t[a + 2:b]))
            i = b + 2
        saida = []
        for tipo, s in partes:
            s = _tex_escape(s)
            # crase vira \texttt, depois do escape para o `\_` sobreviver
            while s.count("`") >= 2:
                p = s.index("`")
                q = s.index("`", p + 1)
                s = s[:p] + r"\texttt{" + s[p + 1:q] + "}" + s[q + 1:]
            s = s.replace("`", "")
            saida.append(r"\textbf{" + s + "}" if tipo == "neg" else s)
        return (f"% Nota de {self.nome}. GERADA por flim_al/tabelas.py.\n"
                r"\footnotesize " + "".join(saida) + "\n\\normalsize\n")

    def gravar(self, destino: str = DESTINO) -> dict:
        os.makedirs(destino, exist_ok=True)
        escritos = {}
        # A nota sai em arquivo PRÓPRIO, e não dentro do `tabular`.
        #
        # Ela carrega as ressalvas que dão sentido aos números: o que é a
        # unidade de análise, o que não pode ser lido como método, qual
        # comparação é pareada. Sem ela a tabela vira um bloco de valores que
        # convida à leitura errada.
        #
        # Em arquivo separado porque o `.tex` da tabela é um `tabular` puro,
        # para caber dentro de qualquer float que o documento queira usar; a
        # nota vai depois do float, como texto corrido.
        for ext, conteudo in (("csv", self.csv()),
                              ("md", self.markdown()),
                              ("tex", self.latex()),
                              ("nota.tex", self.latex_nota())):
            caminho = os.path.join(destino, f"{self.nome}.{ext}")
            with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(conteudo)
            escritos[ext] = caminho

        prov = {
            "tabela": self.nome,
            "titulo": self.titulo,
            "gerado_em": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "gerado_por": "flim_al/tabelas.py",
            "git_commit": ev.git_commit_atual(),
            "registro": os.path.relpath(ev.ARQUIVO, RAIZ).replace(os.sep, "/"),
            "input_latex": f"\\input{{{PREFIXO_LATEX}/{self.nome}.tex}}",
            "linhas": self.proveniencia,
        }
        caminho = os.path.join(destino, f"{self.nome}.proveniencia.json")
        with open(caminho, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(prov, fh, ensure_ascii=False, indent=2)
        escritos["proveniencia"] = caminho
        return escritos


# ── tabelas do projeto ──────────────────────────────────────────────────────

def tabela_por_decoder(experimento: str, nome: str, titulo: str,
                       recs=None, referencia: str = None,
                       tratamento: str = "al") -> Tabela:
    """
    Fβ por decoder, com o teste contra o braço de referência.

    Escolhe sozinho o teste que a estrutura do dado permite:

      - se o braço de referência TEM seeds, pareia semente a semente;
      - se é determinístico (uma execução por célula, sem semente, como o
        braço do artigo), agrega as seeds do tratamento por célula e pareia as
        médias contra a referência ao longo das células.

    A segunda forma não é um atalho: parear as nove seeds contra o mesmo valor
    de referência multiplicaria por nove um n que é, de fato, o número de
    células — pseudo-replicação, e p artificialmente pequeno.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento]
    if not sub:
        raise ValueError(f"nenhuma execucao de {experimento}")

    bracos = sorted({r["braco"] for r in sub})
    if referencia is None:
        for cand in ("paper", "random", "oracle", "original"):
            if cand in bracos:
                referencia = cand
                break
        else:
            referencia = bracos[-1]
    if tratamento not in bracos:
        tratamento = next((b for b in bracos if b != referencia), referencia)

    ref_tem_seed = any(r["seed"] != ev.UNKNOWN
                       for r in sub if r["braco"] == referencia)

    # Nem toda familia registra o nome do decoder no artigo: al_corrected_results
    # so tem o nome no codigo. Agrupar por uma coluna toda UNKNOWN produziria
    # uma tabela vazia sem dizer por que.
    col_dec = ("decoder_paper"
               if any(r["decoder_paper"] != ev.UNKNOWN for r in sub)
               else "decoder")
    linhas = ag.agregar(sub, metrica="fb", condicao=[col_dec, "braco"])
    por_dec = {}
    for l in linhas:
        por_dec.setdefault(l[col_dec], {})[l["braco"]] = l

    modo = ("pareado por semente" if ref_tem_seed
            else "médias por célula (split × orçamento) contra a referência")
    t = Tabela(
        nome, titulo,
        ["decoder", f"{tratamento} (média)", f"{referencia} (média)",
         "n células", "Δ", "p", "veredito"],
        nota=(f"Δ = {tratamento} − {referencia}, teste **pareado**, {modo}. "
              f"Só entram células em que os dois braços existem no **mesmo "
              f"orçamento** — o braço do artigo para em 3–4 imagens e o de AL "
              f"vai a 9, e comparar orçamentos diferentes mediria duas coisas "
              f"ao mesmo tempo. Negrito só com p < 0,05. "
              f"Cada linha é um decoder; eles compartilham o encoder dentro de "
              f"cada célula, então **não são observações independentes entre "
              f"si** — não some os p."))

    for dec in sorted(d for d in por_dec if d != ev.UNKNOWN):
        cel = por_dec[dec]
        dsub = [r for r in sub if r[col_dec] == dec]
        if ref_tem_seed:
            comp = ag.comparar_bracos(dsub, base=referencia,
                                      tratamento=[tratamento])
            r = comp[0] if comp else {}
            n_cel = r.get("n_pares")
        else:
            r = ag.comparar_contra_referencia(dsub, referencia=referencia,
                                              tratamento=tratamento)
            n_cel = r.get("n_celulas")

        ids = [i for l in cel.values() for i in l["run_ids"]]
        m_trat = (cel.get(tratamento) or {}).get("media")
        m_ref = (cel.get(referencia) or {}).get("media")
        p = r.get("p")
        enfase = 4 if (p is not None and p < 0.05) else None

        t.adicionar(
            [dec, _fmt(m_trat), _fmt(m_ref), n_cel or 0,
             _fmt(r.get("delta")), _fmt(p, 4) if p is not None else "—",
             ag.classificar(p)],
            ids, enfase)

    # Linha de fechamento com TODOS os decoders juntos. Vai marcada como
    # agregada porque, com decoders correlacionados, ela nao e um teste
    # independente dos de cima — e sem a marca alguem vai le-la como se fosse.
    if not ref_tem_seed:
        g = ag.comparar_contra_referencia(sub, referencia=referencia,
                                          tratamento=tratamento)
        if g.get("delta") is not None:
            t.adicionar(
                ["todos (agregado)", _fmt(g["media_tratamento"]),
                 _fmt(g["media_referencia"]), g["n_celulas"],
                 _fmt(g["delta"]), _fmt(g["p"], 4),
                 ag.classificar(g["p"])],
                g["run_ids"], None)
    return t


def tabela_curva_orcamento(recs=None) -> Tabela:
    """Fβ por número de imagens anotadas — a curva que sustenta o argumento."""
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] in ("curva", "sel_cs_grande")]
    if not sub:
        raise ValueError("nenhuma execucao de curva")

    linhas = ag.agregar(sub, metrica="fb", condicao=["orcamento"])
    linhas = [l for l in linhas if l["orcamento"] != ev.UNKNOWN]
    linhas.sort(key=lambda l: int(l["orcamento"]))

    t = Tabela("curva_orcamento",
               "Fβ por número de imagens anotadas",
               ["imagens", "Fβ médio", "desvio", "IC95%", "n", "seeds"],
               nota=("Seleção sem consulta ao gabarito. `n` conta execuções; "
                     "`seeds` conta sementes distintas — nove execuções de uma "
                     "semente só não são nove repetições."))
    for l in linhas:
        ic = (f"[{_fmt(l['ic95'][0])}, {_fmt(l['ic95'][1])}]"
              if l["ic95"] else "—")
        t.adicionar([l["orcamento"], _fmt(l["media"]), _fmt(l["desvio"]),
                     ic, l["n"], l["n_seeds"]], l["run_ids"])
    return t


def tabela_resumo_registro(recs=None) -> Tabela:
    """
    Quanto da proveniência sobreviveu, por família de experimento.

    Esta tabela não entra na dissertação como resultado; ela existe para que a
    LACUNA seja visível. Uma família com 100% de seed desconhecida não pode
    sustentar afirmação sobre variância, e é melhor que isso esteja escrito.
    """
    recs = ev.carregar() if recs is None else recs
    fam = {}
    for r in recs:
        f = fam.setdefault(r["experimento"],
                           {"n": 0, "sem_seed": 0, "sem_commit": 0, "ids": []})
        f["n"] += 1
        f["ids"].append(r["run_id"])
        if r["seed"] == ev.UNKNOWN:
            f["sem_seed"] += 1
        if r["git_commit"] == ev.UNKNOWN:
            f["sem_commit"] += 1

    t = Tabela("proveniencia_registro",
               "Proveniência disponível por família de experimento",
               ["família", "execuções", "sem seed", "sem commit"],
               nota=("Diagnóstico, não resultado. Linha com 100% sem seed não "
                     "sustenta afirmação sobre variância entre execuções."))
    for k in sorted(fam, key=lambda k: -fam[k]["n"]):
        f = fam[k]
        t.adicionar([k, f["n"],
                     f"{100 * f['sem_seed'] / f['n']:.0f}%",
                     f"{100 * f['sem_commit'] / f['n']:.0f}%"], f["ids"])
    return t

# Nível de cada técnica. Importa para a leitura: uma técnica de REGIÃO muda
# duas coisas ao mesmo tempo — quais imagens entram e como os seeds são
# desenhados — então comparar região contra imagem sem o controle
# `random_region` credita à seleção um ganho que é de geometria.
# O braço do FLIM publicado: 3 imagens escolhidas pelos autores, sem seleção
# automática nenhuma. É a referência da pergunta central da dissertação.
FLIM_BASE = "flim_3img"

NIVEL_TECNICA = {
    FLIM_BASE: "linha de base",
    "random": "imagem", "entropy": "imagem", "least_confidence": "imagem",
    "margin": "imagem", "coreset": "imagem", "badge": "imagem",
    "random_region": "região", "region_entropy": "região",
    "region_margin": "região", "region_bald": "região",
    "oracle": "imagem (usa GT)",
}


def tabela_comparacao_criterios(recs=None, base: str = "random",
                                experimentos=None) -> Tabela:
    """
    Todas as técnicas de Active Learning lado a lado, contra o piso.

    O piso é o sorteio, e ele não é formalidade: neste projeto, no experimento
    com o encoder retreinado, os quatro critérios de imagem PERDERAM do
    aleatório. Sem a linha do sorteio na mesma tabela, uma coluna de Fβ alto
    parece boa e não é.

    Duas armadilhas que esta função evita, e a segunda quase passou
        1. **Rótulos que não são técnica.** A migração trouxe `none`
           (execuções do braço "original_3imgs", que não usa critério nenhum)
           e as versões em português `aleatorio` e `entropia`, da família
           `orcamento`. Listá-los como métodos inventaria quatro técnicas que
           não existem.

        2. **Média agregando protocolos diferentes.** O Δ e o p são pareados
           DENTRO de cada família — split, decoder, orçamento e semente têm de
           casar. Mas a média, se calculada sobre todas as execuções, mistura
           famílias com encoders, conjuntos de teste e orçamentos diferentes.
           Ficava uma linha em que a média dizia uma coisa e o Δ, outra.
           Por isso a média sai **das mesmas células que entraram no teste**:
           os dois números passam a falar do mesmo conjunto.

    Nível importa na leitura. Técnica de REGIÃO muda quais imagens entram E
    como os seeds são desenhados; o Δ dela contra `random` soma os dois
    efeitos. O controle certo é `random_region`, e a tabela traz a coluna
    "Δ vs controle do nível" para isso.
    """
    recs = ev.carregar() if recs is None else recs
    if experimentos:
        recs = [r for r in recs if r["experimento"] in experimentos]

    # Apelidos: a mesma técnica registrada com nomes diferentes ao longo do
    # tempo. Normalizar aqui, e não na migração, preserva o rótulo original
    # em `evidencia/execucoes/` — o registro guarda o que a origem dizia.
    APELIDOS = {"aleatorio": "random", "entropia": "entropy"}
    NAO_TECNICA = {ev.UNKNOWN, ""}

    limpos = []
    for r in recs:
        c = APELIDOS.get(r.get("criterio"), r.get("criterio"))
        # O braço `original` tem criterio="none" porque não seleciona nada: é
        # o FLIM do artigo, treinado com as 3 imagens que os autores usaram.
        # Numa primeira versão ele foi descartado junto com os rótulos de
        # migração — certo na definição (não é técnica de AL), errado na
        # consequência: é exatamente a referência contra a qual a pergunta
        # "AL melhora?" se responde. Entra com nome próprio.
        if r.get("braco") == "original":
            c = FLIM_BASE
        elif c in NAO_TECNICA or c == "none":
            continue
        limpos.append(dict(r, criterio=c))
    if not limpos:
        raise ValueError("nenhuma execução com técnica reconhecida")

    presentes = sorted({r["criterio"] for r in limpos})
    if base not in presentes:
        raise ValueError(f"critério de base `{base}` ausente — sem piso, a "
                         "tabela não diz se alguma técnica melhora nada")

    t = Tabela(
        "comparacao_criterios",
        "Técnicas de Active Learning contra o sorteio",
        ["técnica", "nível", "Fβ (células pareadas)", "n pares",
         f"Δ vs {base}", "p", "Δ vs FLIM (K=3)", "p (FLIM)",
         "Δ vs controle do nível", "veredito"],
        nota=(f"`{base}` é o piso — sem critério nenhum. O pareamento casa "
              "split, decoder, orçamento e semente **dentro da mesma família "
              "de experimento**; a média mostrada vem dessas mesmas células, "
              "para que média e Δ falem do mesmo conjunto. "
              "**Toda técnica que rodou está aqui, inclusive as que perderam "
              "do sorteio** — no experimento de encoder retreinado isso "
              "aconteceu com as quatro de imagem. Negrito só com p < 0,05. "
              "Para técnicas de REGIÃO, leia a coluna do controle de nível "
              "(`random_region`): o Δ contra `random` soma seleção e "
              "geometria, e a geometria pesa ~2× a seleção neste projeto. "
              f"**`{FLIM_BASE}` é o FLIM do artigo** — as 3 imagens que os "
              "autores escolheram, sem seleção automática. A coluna "
              "`Δ vs FLIM` responde à pergunta central: aplicar AL melhora "
              "sobre isso? Ela só aparece em orçamento casado (K=3), porque o "
              "braço do artigo só existe com 3 imagens; comparar AL com 10 "
              "imagens contra FLIM com 3 mediria orçamento, não seleção. "
              "**Coluna toda vazia significa comparação IMPOSSÍVEL com os "
              "dados existentes, não ausente por descuido**: o braço do "
              "artigo usa os 31 markers REAIS do especialista, e os braços "
              "de AL sobre o pool completo usam sintéticos, porque marker "
              "real só existe para 31 imagens. `marker_origem` entra no "
              "pareamento para que esse par não se forme — ele mediria "
              "quem desenhou o traço e apresentaria como efeito da "
              "seleção. Para responder à pergunta, rode "
              "`scripts/varrer_criterios.py --modo imagem`: marker real "
              "nos dois lados, com `oracle` (o passo 7 do Algoritmo 1) "
              "como referência."))

    def _mede(trat, ref):
        return _comparar_por_criterio(limpos, trat, ref)

    ordem = sorted(presentes, key=lambda c: (NIVEL_TECNICA.get(c, "zzz"), c))
    for crit in ordem:
        nivel = NIVEL_TECNICA.get(crit, "?")
        controle = "random_region" if nivel == "região" else None

        if crit == base:
            t.adicionar([crit, nivel, "—", "—", "—", "—", "—", "—", "—",
                         "piso"],
                        [r["run_id"] for r in limpos
                         if r["criterio"] == crit][:500])
            continue

        r = _mede(crit, base)
        if r["n_pares"] < 2:
            t.adicionar([crit, nivel, "—", r["n_pares"], "—", "—", "—", "—",
                         "—", "sem pares suficientes"],
                        [x["run_id"] for x in limpos
                         if x["criterio"] == crit][:500])
            continue

        dn = "—"
        if controle and controle in presentes and crit != controle:
            rc = _mede(crit, controle)
            if rc["p"] is not None:
                dn = _fmt(rc["delta"])

        # Contra o FLIM do artigo. O pareamento inclui `orcamento`, então só
        # casam as execuções em K=3 — que é o único orçamento em que o braço
        # do artigo existe. Sem essa restrição a coluna compararia AL com 10
        # imagens contra FLIM com 3.
        df, pf = "—", "—"
        if crit != FLIM_BASE and FLIM_BASE in presentes:
            rf = _mede(crit, FLIM_BASE)
            if rf["p"] is not None:
                df = _fmt(rf["delta"])
                pf = _fmt(rf["p"], 4)

        p = r["p"]
        t.adicionar(
            [crit, nivel, _fmt(r["media_a"]), r["n_pares"], _fmt(r["delta"]),
             _fmt(p, 4) if p is not None else "—", df, pf, dn,
             ag.classificar(p)],
            r["run_ids"],
            4 if (p is not None and p < 0.05) else None)
    return t


def _comparar_por_criterio(recs, tratamento: str, base: str) -> dict:
    """
    Teste pareado entre dois CRITÉRIOS (não entre braços).

    O pareamento casa família, split, decoder, orçamento e semente — tudo
    menos o critério, que é o tratamento. `experimento` entra na chave de
    propósito: famílias diferentes usaram encoders e conjuntos de teste
    diferentes, e parear entre elas compararia protocolos, não técnicas.

    Devolve também a média das células pareadas, para que a tabela não mostre
    uma média de um conjunto e um Δ de outro.
    """
    import collections as _c
    chave = ("dataset", "experimento", "decoder", "orcamento", "split",
             "seed", "marker_origem")
    mapa = _c.defaultdict(dict)
    for r in recs:
        v = ag._num(r.get("fb"))
        if v is None:
            continue
        mapa[tuple(r.get(c, ev.UNKNOWN) for c in chave)][r["criterio"]] = (
            v, r["run_id"])
    a, b, ids = [], [], []
    for d in mapa.values():
        if tratamento in d and base in d:
            a.append(d[tratamento][0])
            b.append(d[base][0])
            ids += [d[tratamento][1], d[base][1]]
    r = ag.teste_pareado(a, b)
    import statistics as _st
    r["media_a"] = _st.fmean(a) if a else None
    r["media_b"] = _st.fmean(b) if b else None
    r["run_ids"] = ids or ["sem-pares"]
    return r

def tabela_k_por_modelo(recs=None, experimento: str = "tabela_k_por_modelo",
                        base: str = "flim_paper", fonte=None) -> Tabela:
    """
    Fβ por (critério × K), com os sete decoders e o custo de cada configuração.

    Uma linha por (critério, K): o Fβ de cada decoder, o melhor deles, e o
    tempo. O tempo aparece porque é metade do argumento do FLIM — uma rede que
    estima kernels por k-means treina em segundos, e isso só vira vantagem se
    alguém puser o número na mesa.

    O Δ é contra `flim_paper`, o braço das imagens fixas do artigo, **no mesmo
    orçamento**. Todos os braços usam o mesmo gerador de marker sintético, o
    que torna a comparação internamente válida: o que varia é a seleção, não
    quem desenhou o traço.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}")

    # Uma tabela, uma campanha.
    #
    # Campanhas diferentes deste experimento usam conjuntos de teste
    # diferentes, e misturar as duas numa linha compara Fβ medido em
    # populações distintas. Sem `fonte`, escolhe-se a campanha mais completa —
    # a que cobre mais células (critério, K, decoder).
    if fonte is None:
        cobertura, ultimo = {}, {}
        for r in sub:
            cobertura.setdefault(r["fonte"], set()).add(
                (r["criterio"], r["orcamento"], r["decoder_paper"]))
            f = r["fonte"]
            ultimo[f] = max(ultimo.get(f, ""), r["registrado_em"])
        # Cobertura primeiro; empate vai para a campanha mais recente. Duas
        # campanhas com a mesma grade empatam em cobertura por construcao, e
        # sem o desempate a escolha dependeria da ordem das linhas no CSV.
        fonte = max(cobertura,
                    key=lambda f: (len(cobertura[f]), ultimo[f]))
    sub = [r for r in sub if r["fonte"] == fonte]

    # Depois do filtro, cada célula tem que ser única. Se não for, há duas
    # medidas para a mesma configuração e escolher uma em silêncio é inventar
    # resultado — melhor falhar dizendo qual.
    vistas = {}
    for r in sub:
        vistas.setdefault(
            (r["criterio"], r["orcamento"], r["decoder_paper"], r["seed"]),
            []).append(r)
    ambiguas = {k: v for k, v in vistas.items() if len(v) > 1}
    if ambiguas:
        k, v = next(iter(ambiguas.items()))
        raise ValueError(
            f"{len(ambiguas)} célula(s) com mais de um registro na fonte "
            f"{fonte!r}; a primeira é {k} com {len(v)} medidas "
            f"(fb={[x['fb'] for x in v]}). Passe `fonte=` para desambiguar.")

    decs = [d for d in dict.fromkeys(
        r["decoder_paper"] for r in sub if r["decoder_paper"] != ev.UNKNOWN)]

    # Fβ da referência por (K, decoder), média das sementes, para o Δ sair
    # casado no orçamento.
    acum = {}
    for r in sub:
        if r["criterio"] == base:
            v = ag._num(r.get("fb"))
            if v is not None:
                acum.setdefault((r["orcamento"], r["decoder_paper"]),
                                []).append(v)
    ref = {k: sum(v) / len(v) for k, v in acum.items()}

    t = Tabela(
        "k_por_modelo",
        "Fβ por critério e orçamento, nos sete decoders, com custo",
        ["critério", "K", "n"] + decs + ["melhor", "Δ vs artigo",
                                          "treino (s)", "teste (s)"],
        alinhamento="ll" + "r" * (len(decs) + 5),
        nota=("Uma linha por (critério, K); cada coluna de decoder é o Fβ "
              f"naquele decoder. `{base}` é o braço do artigo — as imagens "
              "fixas que os autores escolheram, sem seleção automática. "
              "**Todos os braços usam o mesmo gerador de marker sintético**, "
              "inclusive o do artigo: marker real só existe para 31 imagens, "
              "e misturar as duas origens faria o Δ medir quem desenhou o "
              "traço em vez da seleção. A comparação é internamente válida e "
              "**não** se compara com números produzidos a partir de markers "
              "reais. `treino` é estimar os kernels por k-means — o que o "
              "FLIM promete ser barato; `teste` é rodar os sete decoders no "
              "conjunto de avaliação. Ele cresce com K porque o encoder "
              "cresce: a arquitetura pede 200 kernels por camada, mas o "
              "k-means só produz tantos quantos os patches dos markers "
              "permitem — medido, 54/51/48/48 com uma imagem contra "
              "200/200/200/200 com oito. Com poucas anotações a rede não é "
              "só menos treinada, é menor. "
              "Tempo com `~` é estimado a partir da soma treino+avaliação, "
              "que era tudo que o registro guardava antes; sem `~` é medido. "
              "Δ é contra o artigo no MESMO K, e fica vazio em K acima de "
              "cinco porque o braço do artigo tem cinco imagens e não existe "
              "referência para comparar. Todas as linhas vêm de uma única "
              f"campanha (`{fonte}`): campanhas diferentes usam conjuntos de "
              "teste diferentes e não se misturam na mesma linha."))

    def _chave(r):
        return (r["criterio"], int(r["orcamento"])
                if r["orcamento"] != ev.UNKNOWN else 0)

    grupos = {}
    for r in sub:
        grupos.setdefault(_chave(r), []).append(r)

    for (crit, k) in sorted(grupos, key=lambda c: (c[0] != base, c[0], c[1])):
        linhas = grupos[(crit, k)]
        # Média das sementes por decoder. Com uma semente é o próprio valor.
        bruto = {}
        for r in linhas:
            v = ag._num(r.get("fb"))
            if v is not None:
                bruto.setdefault(r["decoder_paper"], []).append(v)
        por_dec = {d: sum(v) / len(v) for d, v in bruto.items()}
        n_sementes = len({r["seed"] for r in linhas})

        vals = [_fmt(por_dec.get(d)) for d in decs]
        melhor_dec = max(por_dec, key=por_dec.get) if por_dec else None
        melhor = (f"{por_dec[melhor_dec]:.3f} ({melhor_dec})"
                  if melhor_dec else "—")

        # Δ: média da diferença contra a referência, decoder a decoder, no
        # mesmo K. Só entram decoders em que os dois lados existem.
        difs = [por_dec[d] - ref[(str(k), d)] for d in por_dec
                if (str(k), d) in ref]
        delta = _fmt(sum(difs) / len(difs)) if difs else "—"

        # Treino e avaliação, medidos separados quando existem.
        #
        # O runner grava os dois desde que o schema foi estendido. Antes disso
        # só havia a soma por decoder, e o treino era estimado pelo menor dos
        # sete `segundos` — número que carrega a avaliação mais barata
        # embutida e erra o treino para cima. São sete equações e oito
        # incógnitas: a separação não se recupera do que já foi gravado.
        #
        # Por isso a estimativa sobrevive como fallback, mas marcada com "~".
        # Uma estimativa que se parece com medida é pior que nenhuma.
        tt = [x for x in (ag._num(r.get("segundos_treino")) for r in linhas)
              if x is not None]
        ta = [x for x in (ag._num(r.get("segundos_aval")) for r in linhas)
              if x is not None]
        segs = [x for x in (ag._num(r.get("segundos")) for r in linhas)
                if x is not None]
        if tt and ta:
            # O encoder é um só para os decoders de uma semente: o treino não
            # se soma entre eles, mas há um por semente — daí a média.
            treino = _fmt(sum(tt) / len(tt), 1)
            teste = _fmt(sum(ta) / max(1, n_sementes), 1)
        elif segs:
            treino = "~" + _fmt(min(segs), 1)
            teste = "~" + _fmt(sum(x - min(segs) for x in segs), 1)
        else:
            treino = teste = "—"

        t.adicionar([crit, k, n_sementes] + vals
                    + [melhor, delta, treino, teste],
                    [r["run_id"] for r in linhas])
    return t


def tabela_al_vs_flim(recs=None, experimento: str = "tabela_k_por_modelo",
                      base: str = "flim_paper", controle: str = "random",
                      fonte=None) -> Tabela:
    """
    Houve melhora? Diferença pareada por semente, com n, IC95% e p.

    Média não responde a pergunta: com uma amplitude de 0,10–0,22 no braço
    aleatório deste projeto, duas médias podem diferir por 0,05 sem que exista
    efeito. O que responde é a diferença PAREADA — dentro de cada semente os
    dois braços compartilham a partição, o conjunto de teste e o gerador de
    traços, então a diferença isola a seleção.

    As duas referências aparecem juntas de propósito:

    - contra o **artigo** (`flim_paper`), a pergunta é se a escolha automática
      bate as imagens que os autores escolheram a mão;
    - contra o **sorteio** (`random`), a pergunta é se o critério tem mérito
      próprio.

    Um critério que bate o artigo mas empata com o sorteio não demonstrou
    seleção — demonstrou que as imagens do artigo não eram especiais.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}")

    if fonte is None:
        # Aqui a campanha melhor e a com MAIS REPETICOES, nao a mais larga.
        # Uma campanha de cinco sementes e dois decoders responde "melhorou?";
        # uma de uma semente e sete decoders nao responde, por mais celulas
        # que tenha. Cobertura e recencia so desempatam.
        cob, sem, ult = {}, {}, {}
        for r in sub:
            f = r["fonte"]
            cob.setdefault(f, set()).add(
                (r["criterio"], r["orcamento"], r["decoder_paper"], r["seed"]))
            sem.setdefault(f, set()).add(r["seed"])
            ult[f] = max(ult.get(f, ""), r["registrado_em"])
        fonte = max(cob, key=lambda f: (len(sem[f]), len(cob[f]), ult[f]))
    sub = [r for r in sub if r["fonte"] == fonte]

    # (criterio, K, decoder, semente) -> Fbeta
    valor = {}
    for r in sub:
        v = ag._num(r.get("fb"))
        if v is not None:
            valor[(r["criterio"], r["orcamento"], r["decoder_paper"],
                   r["seed"])] = v
    sementes = sorted({r["seed"] for r in sub})
    criterios = [c for c in dict.fromkeys(r["criterio"] for r in sub)
                 if c != base]
    ks = sorted({r["orcamento"] for r in sub}, key=lambda x: int(x))
    decs = list(dict.fromkeys(r["decoder_paper"] for r in sub))

    t = Tabela(
        "al_vs_flim",
        "Melhora sobre o FLIM do artigo e sobre o sorteio, pareada por semente",
        ["critério", "K", "decoder", "n", "Fβ médio", "Δ vs artigo", "IC95%",
         "p", "Δ vs sorteio", "p "],
        alinhamento="lllr" + "r" * 6,
        nota=("Diferença pareada por semente: dentro de cada semente os braços "
              "compartilham partição, conjunto de teste e gerador de traços, "
              f"então a diferença isola a seleção. `{base}` é o braço do "
              f"artigo; `{controle}` é o sorteio. Bater o artigo sem bater o "
              "sorteio não demonstra seleção — demonstra que as imagens do "
              "artigo não eram especiais. p vem de t pareado bicaudal; com "
              "poucas sementes o teste tem pouco poder, e p alto significa "
              "**não decidido**, não 'igual'. Uma única campanha "
              f"(`{fonte}`)."))

    for crit in criterios:
        if crit == controle:
            continue
        for k in ks:
            for d in decs:
                a = [valor.get((crit, k, d, s)) for s in sementes]
                b = [valor.get((base, k, d, s)) for s in sementes]
                c = [valor.get((controle, k, d, s)) for s in sementes]
                vs = [x for x in a if x is not None]
                if not vs:
                    continue
                tb_ = ag.teste_pareado(a, b)
                tc_ = ag.teste_pareado(a, c)
                ic = (f"[{tb_['ic95'][0]:+.3f}, {tb_['ic95'][1]:+.3f}]"
                      if tb_["ic95"] else "—")
                ids = [r["run_id"] for r in sub
                       if r["criterio"] == crit and r["orcamento"] == k
                       and r["decoder_paper"] == d]
                t.adicionar(
                    [crit, k, d, len(vs), f"{sum(vs) / len(vs):.3f}",
                     _fmt(tb_["delta"]), ic, _fmt(tb_["p"]),
                     _fmt(tc_["delta"]), _fmt(tc_["p"])], ids)
    return t


def tabela_onde_marcar(recs=None, experimento: str = "onde_marcar",
                       base: str = "uniforme",
                       balanco: str = "uniforme_balanceado",
                       fonte=None) -> Tabela:
    """
    Onde anotar, sob orçamento fixo de pixels: houve melhora, e de quê?

    O orçamento é contado em pixels anotados por imagem, não em imagens,
    porque o que limita o FLIM é a quantidade de pixels marcados — a mesma
    imagem rende 54 kernels com traço esparso e 200 com traço denso. Todos os
    braços gastam exatamente o mesmo número de pixels; o que muda é onde eles
    caem.

    Duas referências, e a segunda é a que decide:

    - **`uniforme`** é o FLIM de hoje, traço espalhado. Bater ele diz que
      anotar por região ajuda.
    - **`uniforme_balanceado`** tem a mesma proporção objeto/fundo que a
      incerteza produz, mas com os pixels sorteados. Bater ele diz que o ganho
      veio de *onde* se anotou, e não de *quanto de cada classe* se anotou.

    Sem a segunda coluna o resultado é ambíguo: escolher regiões incertas
    escolhe, junto, muito mais foreground — ~73% contra ~16% do traço
    uniforme, medido antes de rodar.

    `regiao_borda` usa o gabarito para achar a fronteira. Não é método, é teto.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}")

    if fonte is None:
        cob, sem, ult = {}, {}, {}
        for r in sub:
            f = r["fonte"]
            cob.setdefault(f, set()).add(
                (r["criterio"], r["orcamento_px"], r["decoder_paper"],
                 r["seed"]))
            sem.setdefault(f, set()).add(r["seed"])
            ult[f] = max(ult.get(f, ""), r["registrado_em"])
        fonte = max(cob, key=lambda f: (len(sem[f]), len(cob[f]), ult[f]))
    sub = [r for r in sub if r["fonte"] == fonte]

    valor = {}
    for r in sub:
        v = ag._num(r.get("fb"))
        if v is not None:
            valor[(r["criterio"], r["orcamento_px"], r["decoder_paper"],
                   r["seed"])] = v
    sementes = sorted({r["seed"] for r in sub})
    bracos = [b for b in dict.fromkeys(r["criterio"] for r in sub)
              if b != base]
    pxs = sorted({r["orcamento_px"] for r in sub},
                 key=lambda x: int(x) if str(x).isdigit() else 0)
    decs = list(dict.fromkeys(r["decoder_paper"] for r in sub))

    t = Tabela(
        "onde_marcar",
        "Onde anotar, com orçamento fixo em pixels: Fβ e as duas diferenças",
        ["braço", "px/imagem", "decoder", "n", "Fβ médio", "Δ vs uniforme",
         "p", "Δ vs balanceado", "p "],
        alinhamento="lllr" + "r" * 5,
        nota=("Orçamento em **pixels anotados por imagem**, não em imagens: o "
              "que limita o FLIM é a quantidade de pixels marcados — a mesma "
              "imagem rende 54 kernels por camada com traço esparso e 200 com "
              "traço denso. Todos os braços gastam exatamente o mesmo número "
              "de pixels, nas MESMAS imagens (as do artigo); o que muda é "
              f"onde eles caem. `{base}` é o FLIM de hoje. "
              f"`{balanco}` tem a mesma proporção objeto/fundo que a "
              "incerteza produz, com os pixels sorteados — é ele que separa "
              "*onde se anotou* de *quanto de cada classe se anotou*, porque "
              "escolher região incerta escolhe junto muito mais foreground "
              "(~73% contra ~16%, medido antes de rodar). `regiao_borda` usa "
              "o gabarito e é teto, não método. Diferenças pareadas por "
              f"semente; p de t pareado bicaudal. Campanha `{fonte}`."))

    for b in bracos:
        for px in pxs:
            for d in decs:
                a = [valor.get((b, px, d, s)) for s in sementes]
                u = [valor.get((base, px, d, s)) for s in sementes]
                q = [valor.get((balanco, px, d, s)) for s in sementes]
                vs = [x for x in a if x is not None]
                if not vs:
                    continue
                tu = ag.teste_pareado(a, u)
                tq = (ag.teste_pareado(a, q) if b != balanco
                      else {"delta": None, "p": None})
                ids = [r["run_id"] for r in sub
                       if r["criterio"] == b and r["orcamento_px"] == px
                       and r["decoder_paper"] == d]
                t.adicionar(
                    [b, px, d, len(vs), f"{sum(vs) / len(vs):.3f}",
                     _fmt(tu["delta"]), _fmt(tu["p"]),
                     _fmt(tq["delta"]), _fmt(tq["p"])], ids)
    return t


NOME_DATASET = {"schisto": "Parasitas", "brats": "BraTS",
                "conjunctiva": "Conjuntivite"}


def tabela_dissertacao(recs=None, experimento: str = "tabela_k_por_modelo",
                       rotulo: str = "dissertacao", base: str = "random",
                       decoder: str = "FLIM_lm") -> Tabela:
    """
    FLIM puro contra cada critério de AL, nos três datasets e quatro orçamentos.

    A referência é `random` — o FLIM puro, que sorteia as imagens como o artigo
    faz na primeira rodada. Cada critério recebe o MESMO número de imagens.

    `regiao_confusa` é o braço diferente dos outros três: ele usa exatamente as
    imagens que o sorteio usaria naquela semente e muda só ONDE o traço cai,
    gastando o mesmo número de pixels. Por isso o Δ dele mede posição de
    anotação, enquanto o Δ dos outros mede escolha de imagem.

    Sobre a coluna de acurácia
        É `1 − MAE`, que para máscara binária é exatamente a fração de pixels
        certos. Ela fica acima de 0,97 em quase toda a tabela porque o objeto
        ocupa uma fração mínima da imagem: acertar todo o fundo já garante
        isso. Está aqui porque foi pedida, mas não separa os braços — quem
        separa é Fβ e IoU.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs
           if r["experimento"] == experimento and rotulo in r["fonte"]
           and r["decoder_paper"] == decoder]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento} com "
                         f"rotulo={rotulo!r} e decoder={decoder!r}")

    def _agr(rs, campo):
        vs = [ag._num(r.get(campo)) for r in rs]
        vs = [v for v in vs if v is not None]
        return sum(vs) / len(vs) if vs else None

    dss = sorted({r["dataset"] for r in sub},
                 key=lambda d: list(NOME_DATASET).index(d)
                 if d in NOME_DATASET else 99)
    ks = sorted({r["orcamento"] for r in sub},
                key=lambda x: int(x) if str(x).isdigit() else 0)
    crits = [c for c in ("random", "coreset", "entropy", "least_confidence",
                         "regiao_confusa", "oracle")
             if c in {r["criterio"] for r in sub}]

    t = Tabela(
        # O nome leva o decoder: as duas variantes gravam arquivos diferentes.
        # Sem isto a segunda sobrescrevia a primeira em dissertacao.tex.
        f"dissertacao_{decoder.replace('FLIM_', '').replace('*', 'x')}",
        f"FLIM puro × Active Learning nos três datasets ({decoder})",
        ["dataset", "K", "braço", "n", "Fβ", "acurácia", "IoU",
         "treino (s)", "teste (s)", "Δ Fβ", "p"],
        alinhamento="lll" + "r" * 8,
        nota=("`random` é o FLIM puro: sorteia as imagens, como o artigo faz "
              "na primeira rodada. Todos os braços recebem o MESMO número de "
              "imagens (K) e são medidos no mesmo conjunto de teste. "
              "**`regiao_confusa` é diferente dos outros três**: ele usa "
              "exatamente as imagens que o sorteio usaria naquela semente e "
              "muda só ONDE o traço cai, com o mesmo número de pixels — então "
              "o Δ dele mede posição de anotação, e o dos outros mede escolha "
              "de imagem. **Acurácia é `1 − MAE`**, a fração de pixels "
              "certos; ela passa de 0,97 em quase tudo porque o objeto ocupa "
              "uma fração mínima da imagem, e acertar o fundo já garante "
              "isso — quem separa os braços é Fβ e IoU. Δ e p são pareados "
              "por semente contra o FLIM puro, no mesmo dataset e no mesmo K; "
              "com poucas sementes o teste tem pouco poder, e p alto "
              f"significa **não decidido**, não 'igual'. Decoder {decoder}."))

    for ds in dss:
        for k in ks:
            ref = [r for r in sub if r["dataset"] == ds
                   and r["orcamento"] == k and r["criterio"] == base]
            ref_por_seed = {r["seed"]: ag._num(r.get("fb")) for r in ref}
            for c in crits:
                rs = [r for r in sub if r["dataset"] == ds
                      and r["orcamento"] == k and r["criterio"] == c]
                if not rs:
                    continue
                sementes = sorted({r["seed"] for r in rs})
                a = [ag._num(next((x for x in rs if x["seed"] == sd), {})
                             .get("fb")) for sd in sementes]
                b = [ref_por_seed.get(sd) for sd in sementes]
                tp = ({"delta": None, "p": None} if c == base
                      else ag.teste_pareado(a, b))
                mae = _agr(rs, "mae")
                t.adicionar(
                    [NOME_DATASET.get(ds, ds), k,
                     "FLIM puro" if c == base else c, len(sementes),
                     _fmt(_agr(rs, "fb")),
                     _fmt(None if mae is None else 1 - mae),
                     _fmt(_agr(rs, "iou")),
                     _fmt(_agr(rs, "segundos_treino"), 1),
                     _fmt(_agr(rs, "segundos_aval"), 1),
                     "—" if c == base else _fmt(tp["delta"]),
                     "—" if c == base else _fmt(tp["p"])],
                    [r["run_id"] for r in rs])
    return t

# Fbeta da Tabela III do artigo, usuario A, para a coluna de referencia.
# Soares et al., arXiv:2504.20872. Transcrito, nao recalculado.
TABELA_III_A = {"FLIM_lm": 0.860, "FLIM_pb": 0.857, "FLIM_mb": 0.843,
                "FLIM_at": 0.740, "FLIM_ts": 0.747, "FLIM_lt": 0.810}


def tabela_artigo_vs_regiao(recs=None, experimento: str = "artigo_vs_regiao",
                            base: str = "artigo") -> Tabela:
    """
    A tabela do artigo, e os mesmos cliques realocados — por AL e ao acaso.

    Os três braços usam as MESMAS imagens (as que os especialistas A e B
    anotaram), o MESMO número de cliques e o MESMO traço de fundo, copiado e
    não regerado. Só muda para onde vão os cliques de objeto.

    As três diferenças decompõem o efeito, e é por isso que o braço
    `aleatorio` não é opcional:

        AL − artigo        realocar o clique E escolher onde, juntos
        aleatorio − artigo **só** realocar (sai da borda, vai para o interior)
        AL − aleatorio     **só** a escolha do Active Learning

    Sem a terceira coluna, um Δ negativo na primeira seria atribuído ao AL
    quando pode ser inteiramente da segunda. Esta tabela existiu uma versão
    sem ela, e a leitura estava errada.

    A coluna `Fβ publicado (A)` é a Tabela III do artigo, transcrita. Ela
    **não** é comparável com a reproduzida: o artigo aplica Dynamic Trees e o
    binário é ELF de Linux, que não executa nesta máquina.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento
           and "smoke" not in r["fonte"]]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}")

    ordem = ["FLIM_lm", "FLIM_pb", "FLIM_mb", "FLIM_at", "FLIM_ts",
             "FLIM_lt", "FLIM_ts*"]
    decs = [d for d in ordem if d in {r["decoder_paper"] for r in sub}]

    t = Tabela(
        "artigo_vs_regiao",
        "Tabela do artigo × cliques realocados por AL × realocados ao acaso",
        ["decoder", "n", "Fβ artigo", "Fβ AL região", "Fβ aleatório",
         "Δ AL−artigo", "p", "Δ aleat−artigo", "p ", "Δ AL−aleat", "p  ",
         "Fβ publicado (A)"],
        alinhamento="lr" + "r" * 10,
        nota=("Mesmas imagens — as que os especialistas A e B anotaram —, "
              "mesmo número de cliques e mesmo traço de fundo (copiado, não "
              "regerado) nos três braços. Só muda para onde vão os cliques de "
              "objeto. **Não há seleção de imagem: é seleção de região.** As "
              "três diferenças decompõem o efeito — `aleat−artigo` isola o "
              "ato de tirar o clique da borda, e `AL−aleat` isola a escolha "
              "do Active Learning. Sem a terceira, um Δ negativo na primeira "
              "seria creditado ao AL quando pode ser todo da segunda. Cada "
              "célula é um par (usuário, split), n = 6, pareado. `Fβ "
              "publicado (A)` vem da Tabela III do artigo e **não** é "
              "comparável com a coluna reproduzida: o artigo aplica Dynamic "
              "Trees, cujo binário não executa nesta máquina. Avaliação em "
              "250 imagens de Z₁\\T, as mesmas em todos os braços."))

    for d in decs:
        def cel(c, campo="fb"):
            return {(r["usuario"], r["split"]): ag._num(r.get(campo))
                    for r in sub if r["criterio"] == c
                    and r["decoder_paper"] == d}
        a, al, ale = cel(base), cel("al_regiao"), cel("aleatorio")
        ks = sorted(set(a) & set(al) & set(ale))
        if not ks:
            continue
        t1 = ag.teste_pareado([al[k] for k in ks], [a[k] for k in ks])
        t2 = ag.teste_pareado([ale[k] for k in ks], [a[k] for k in ks])
        t3 = ag.teste_pareado([al[k] for k in ks], [ale[k] for k in ks])
        pub = TABELA_III_A.get(d)
        ids = [r["run_id"] for r in sub if r["decoder_paper"] == d]
        t.adicionar([d, len(ks),
                     _fmt(sum(a[k] for k in ks) / len(ks)),
                     _fmt(sum(al[k] for k in ks) / len(ks)),
                     _fmt(sum(ale[k] for k in ks) / len(ks)),
                     _fmt(t1["delta"]), _fmt(t1["p"], 4),
                     _fmt(t2["delta"]), _fmt(t2["p"], 4),
                     _fmt(t3["delta"]), _fmt(t3["p"], 4),
                     _fmt(pub) if pub else "—"], ids)
    return t

NOME_CRIT = {"random": "FLIM puro (sorteio)", "coreset": "CoreSet (imagem)",
             "entropy": "Entropia (imagem)",
             "least_confidence": "Least confidence (imagem)",
             "medoide": "Medoide (imagem)", "oracle": "Oracle (usa GT)",
             "regiao_confusa": "AL de REGIAO"}


def tabela_geral_al(recs=None, experimento: str = "tabela_k_por_modelo",
                    rotulo: str = "dissertacao", base: str = "random",
                    decoder: str = "FLIM_pb") -> Tabela:
    """
    Todos os critérios de AL, todos os K, os três datasets — uma linha cada.

    É a tabela de defesa: ela responde "o Active Learning ajuda o FLIM?" numa
    página, e a resposta está nas colunas de Δ.

    Os critérios de IMAGEM (CoreSet, entropia, least confidence, medoide)
    escolhem QUAIS imagens anotar. `AL de REGIAO` usa exatamente as imagens
    que o sorteio usaria e muda só ONDE o traço cai, com a mesma contagem de
    pixels — por isso o Δ dele mede posição de anotação, e o dos outros mede
    escolha de imagem.

    `Oracle` usa o gabarito para escolher a imagem em que o modelo atual vai
    pior: é o passo 7 do Algoritmo 1 do artigo. Não é método, é referência —
    e o fato de ele também não se destacar é o resultado mais informativo
    desta tabela.

    Todos os braços recebem o MESMO número de imagens e são medidos no mesmo
    conjunto de teste, dentro de cada dataset e cada K.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs
           if r["experimento"] == experimento and rotulo in r["fonte"]
           and r["decoder_paper"] == decoder]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}/{rotulo}")

    dss = [d for d in ("schisto", "brats", "conjunctiva")
           if d in {r["dataset"] for r in sub}]
    ks = sorted({r["orcamento"] for r in sub},
                key=lambda x: int(x) if str(x).isdigit() else 0)
    crits = [c for c in ("random", "coreset", "entropy", "least_confidence",
                         "medoide", "regiao_confusa", "oracle")
             if c in {r["criterio"] for r in sub}]

    cols = ["critério", "K"]
    for d in dss:
        cols += [NOME_DATASET.get(d, d)]
    cols += ["Δ médio vs sorteio", "p", "vence o sorteio em"]

    t = Tabela(
        f"geral_al_{decoder.replace('FLIM_','').replace('*','x')}",
        f"Active Learning no FLIM: todos os critérios, todos os K ({decoder})",
        cols, alinhamento="ll" + "r" * (len(dss) + 3),
        nota=("Uma linha por (critério, K). As colunas de dataset são o Fβ "
              "médio sobre as sementes. **`FLIM puro (sorteio)` é a "
              "referência**: todos os braços recebem o MESMO número de "
              "imagens e são medidos no mesmo conjunto de teste. Os critérios "
              "de IMAGEM escolhem quais imagens anotar; **`AL de REGIÃO` usa "
              "as mesmas imagens do sorteio e muda só ONDE o traço cai**, com "
              "a mesma contagem de pixels — o Δ dele mede posição de "
              "anotação, não escolha de imagem. **`Oracle` usa o gabarito** "
              "para escolher a imagem em que o modelo vai pior (passo 7 do "
              "Algoritmo 1 do artigo): não é método, é referência, e o fato "
              "de ele também não se destacar é o achado mais informativo "
              "daqui. Δ e p são pareados por semente dentro de cada dataset e "
              "K, depois agregados. `vence o sorteio em` conta as células "
              f"(dataset × semente) favoráveis. Decoder {decoder}."))

    for c in crits:
        if c == base:
            continue
        for k in ks:
            linha, difs, vit, tot = [NOME_CRIT.get(c, c), k], [], 0, 0
            for d in dss:
                def v(cc):
                    return {r["seed"]: ag._num(r.get("fb")) for r in sub
                            if r["dataset"] == d and r["orcamento"] == k
                            and r["criterio"] == cc}
                a, b = v(c), v(base)
                sem = sorted(set(a) & set(b))
                if not sem:
                    linha.append("—")
                    continue
                linha.append(_fmt(sum(a[s] for s in sem) / len(sem)))
                for s in sem:
                    difs.append(a[s] - b[s])
                    tot += 1
                    vit += 1 if a[s] > b[s] else 0
            if not difs:
                continue
            tp = ag.teste_pareado(difs, [0] * len(difs))
            ids = [r["run_id"] for r in sub
                   if r["criterio"] == c and r["orcamento"] == k]
            t.adicionar(linha + [_fmt(tp["delta"]), _fmt(tp["p"], 4),
                                 f"{vit} de {tot}"], ids)
    return t


# ── diagnóstico do ganho marginal ───────────────────────────────────────────

def _gm_campos(r: dict) -> dict:
    """
    Lê os escores que viajam na `variante` do experimento `ganho_marginal`.

    O registro canônico não tem coluna para escore de região, e acrescentar
    uma recalcularia o `run_id` das 10 mil execuções já gravadas — custo que
    já foi pago uma vez neste projeto e revertido (ver `orcamento_px` em
    `evidencia.CAMPOS_ID`). A `variante` carrega os escores como texto; esta
    função é o parser.
    """
    v = r.get("variante", "")
    out = {"papel": r.get("criterio", ""), "estrato": "", "ent": None,
           "lc": None, "fg": None, "kernels": None}
    for p in v.split("|"):
        if p.startswith("ent="):
            out["ent"] = float(p[4:])
        elif p.startswith("lc="):
            out["lc"] = float(p[3:])
        elif p.startswith("fg="):
            out["fg"] = float(p[3:])
        elif p.startswith("k="):
            try:
                out["kernels"] = sum(int(x) for x in p[2:].split("-") if x)
            except ValueError:
                pass
        elif p in ("objeto", "fundo"):
            out["estrato"] = p
    return out


def _spearman(xs: list, ys: list):
    """
    Correlação de postos.

    Implementada aqui para não depender de scipy no caminho que gera tabela —
    este ambiente já teve um pacote mudando resultado em silêncio (`faiss`,
    CLAUDE.md §2.2), e o caminho resultado→tabela é o que menos pode ter
    dependência opcional.
    """
    import math
    pares = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pares)
    if n < 3:
        return None

    def postos(vs):
        ordem = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(ordem):
            j = i
            while j + 1 < len(ordem) and vs[ordem[j + 1]] == vs[ordem[i]]:
                j += 1
            media = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[ordem[k]] = media
            i = j + 1
        return r

    rx, ry = postos([p[0] for p in pares]), postos([p[1] for p in pares])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def _gm_por_semente(recs, experimento, dataset, decoder):
    """
    Agrupa por semente, que é a unidade independente de análise.

    Os 24 candidatos dentro de uma semente compartilham encoder, imagens e
    partição — são correlacionados, e entram como n apenas depois de
    colapsados num número por semente.
    """
    sub = [r for r in recs if r["experimento"] == experimento
           and r["dataset"] == dataset
           and r["decoder_paper"] == decoder
           and "piloto" not in r.get("fonte", "")
           and "smoke" not in r.get("fonte", "")]
    por = {}
    for r in sub:
        fb = ag._num(r.get("fb"))
        if fb is None:
            continue
        s = str(r["seed"])
        por.setdefault(s, {"base": None, "cands": [], "imgs": []})
        linha = dict(_gm_campos(r), fb=fb, run_id=r["run_id"])
        if r["criterio"] == "base":
            por[s]["base"] = linha
        elif r["criterio"] == "imagem_nova":
            por[s]["imgs"].append(linha)
        else:
            por[s]["cands"].append(linha)
    # Sem base não há Δ: a semente inteira sai.
    return {s: v for s, v in por.items() if v["base"] and v["cands"]}


PESOS_ESTRATO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "evidencia", "bruto", "pesos_estrato.json")


def _pesos_estrato(dataset: str) -> dict:
    """
    Fração de superpixels que tocam o objeto, por semente.

    Medida por `scripts/pesos_estrato.py`, que roda SLIC e o ground truth sem
    treinar nada. Serve para reponderar o sorteio estratificado num sorteio
    uniforme — ver `_linha_uniforme`.
    """
    if not os.path.isfile(PESOS_ESTRATO):
        return {}
    with open(PESOS_ESTRATO, encoding="utf-8") as fh:
        d = json.load(fh)
    return (d.get(dataset) or {}).get("por_semente", {})


def _colapsou(base_fb: float) -> bool:
    """
    Base degenerada: Fβ exatamente zero.

    Não é limiar escolhido — é o piso aritmético. Com Fβ(sem) = 0, o Δ de
    qualquer candidato é ≥ 0 por construção, e a média dessa população não
    estima a mesma coisa que a média de uma população que pode piorar. Ver
    `evidencia/campanhas/adendo_ganho_marginal_2026-10-01.json`.
    """
    return base_fb == 0.0


def tabela_ganho_marginal(recs=None, experimento: str = "ganho_marginal",
                          dataset: str = "schisto",
                          decoder: str = "FLIM_lm",
                          regime: str = "funcional") -> Tabela:
    """
    Existe ganho a capturar na escolha da região, e o escore o encontra?

    Cada linha é um ponto na mesma escala de Δ = Fβ(validação) com o marcador
    extra, menos Fβ(validação) sem ele. Tudo o mais é idêntico dentro da
    semente: partição, imagens, marcadores base, encoder inicial, conjunto de
    validação. A única variável é qual região recebeu o marcador.

    As três linhas que importam, nesta ordem:

        melhor região (teto)   o máximo de Δ entre os candidatos sorteados —
                               quanto haveria para ganhar se soubéssemos qual
        argmax entropia        o que o Active Learning de fato escolhe
        candidato sorteado     o piso

    O teto usa o Fβ da validação para escolher. **Não é um método** — é a
    medida do que existe para ser capturado. Citá-lo como estratégia seria
    citar informação que não existe em produção.

    A linha `imagem nova` é o controle de escala, e é ela que separa "nenhuma
    região ajuda" de "o FLIM não incorpora marcador novo". Se ela se move e as
    regiões não, o mecanismo de atualização está intacto e o que falta é
    informação.

    n é o número de SEMENTES, nunca o de candidatos.
    """
    recs = ev.carregar() if recs is None else recs
    por = _gm_por_semente(recs, experimento, dataset, decoder)
    if not por:
        raise ValueError(f"nenhuma execução de {experimento}/{dataset}")

    # Estratificação por regime da base. POST-HOC e forçada pela aritmética:
    # Δ a partir de Fβ=0 exato não pode ser negativo, então as duas populações
    # não têm a mesma média estimável. Ver o adendo da campanha.
    if regime != "todos":
        quer = (regime == "colapsada")
        por = {k: v for k, v in por.items()
               if _colapsou(v["base"]["fb"]) == quer}
    if not por:
        raise ValueError(
            f"nenhuma semente no regime '{regime}' para {dataset}/{decoder}")

    sementes = sorted(por, key=lambda s: int(s))
    sufixo = {"funcional": "base funcional",
              "colapsada": "base colapsada (Fβ₀=0)",
              "todos": "todas as sementes"}[regime]
    t = Tabela(
        f"ganho_marginal_{dataset}_{decoder.replace('*', 'x')}_{regime}",
        "Ganho marginal de um marcador adicional — "
        f"{NOME_DATASET.get(dataset, dataset)}, {decoder}, {sufixo}",
        ["o que recebeu o marcador", "n (sementes)", "Δ Fβ médio", "IC95%",
         "pior", "melhor", "Δ vs sorteado", "p"],
        alinhamento="lrrrrrrr",
        nota=("Δ = Fβ na **validação** com o marcador extra menos Fβ sem ele. "
              "Dentro de cada semente, a partição, as imagens iniciais, os "
              "marcadores base, o encoder inicial e o conjunto de validação "
              "são idênticos — a única variável é qual região recebeu os "
              "~300 px adicionais. O treino é determinístico **dentro de um "
              "processo**, a partir dos mesmos arquivos de marcador "
              "(verificado: três repetições dão Fβ idêntico até a décima "
              "casa). **Entre** execuções há ruído: a reexecução completa do "
              "schisto deu Fβ idêntico em 538 de 540 registros, e nos 2 "
              "restantes o desvio máximo foi 0,0034 — duas ordens de "
              "grandeza abaixo dos efeitos reportados. "
              "**n é o número de sementes**: os "
              "candidatos de uma mesma semente compartilham encoder e são "
              "correlacionados. `melhor região (teto)` escolhe pelo Fβ da "
              "validação e **não é uma estratégia** — mede o que existe para "
              "capturar, **entre os 22 candidatos examinados** de ~380 "
              "superpixels; não é teto absoluto. "
              f"Linhas restritas a: **{sufixo}**. A separação por regime é "
              "post-hoc e forçada pela aritmética — Δ a partir de Fβ=0 exato "
              "não pode ser negativo, então as duas populações não têm média "
              "comparável. As sementes compartilham o pool e o conjunto de "
              "validação: o IC95% vale para **esta** validação sob sorteio "
              "das imagens de treino, e não generaliza para o dataset. "
              "O conjunto de teste não foi tocado por esta campanha."))

    def serie(f):
        vs, ids = [], []
        for s in sementes:
            b = por[s]["base"]
            r = f(por[s])
            if r is None:
                vs.append(None)
                continue
            vs.append(r["fb"] - b["fb"])
            ids += [r["run_id"], b["run_id"]]
        return vs, ids

    def sorteados(v):
        return [x for x in v["cands"] if x["papel"] == "sorteado"]

    def melhor(v):
        c = sorteados(v)
        return max(c, key=lambda x: x["fb"]) if c else None

    def pior(v):
        c = sorteados(v)
        return min(c, key=lambda x: x["fb"]) if c else None

    def papel(nome):
        return lambda v: next((x for x in v["cands"] if x["papel"] == nome),
                              None)

    def media_de(chave):
        def f(v):
            c = sorteados(v) if chave == "sorteado" else v["imgs"]
            if not c:
                return None
            return {"fb": sum(x["fb"] for x in c) / len(c),
                    "run_id": c[0]["run_id"]}
        return f

    ref, _ = serie(media_de("sorteado"))
    linhas = [
        ("melhor região (teto amostrado)", melhor),
        ("argmax entropia (o AL)", papel("argmax_entropia")),
        ("argmax least confidence", papel("argmax_lc")),
        ("candidato sorteado (média)", media_de("sorteado")),
        ("pior região", pior),
        ("imagem nova inteira (orçamento MAIOR)", media_de("imagem")),
    ]
    for rotulo, f in linhas:
        vs, ids = serie(f)
        reais = [v for v in vs if v is not None]
        d = ag.descrever(reais)
        if not d["n"]:
            continue
        eh_ref = rotulo.startswith("candidato")
        tp = ag.teste_pareado(vs, ref)
        ic = (f"[{_fmt(d['ic95'][0], 4)}, {_fmt(d['ic95'][1], 4)}]"
              if d["ic95"] else "—")
        t.adicionar(
            [rotulo, d["n"], _fmt(d["media"], 4), ic,
             _fmt(d["minimo"], 4), _fmt(d["maximo"], 4),
             "—" if eh_ref else _fmt(tp["delta"], 4),
             "—" if eh_ref else _fmt(tp["p"], 4)],
            ids or [por[sementes[0]]["base"]["run_id"]])

    # ── o sorteio UNIFORME, reponderado ────────────────────────────────────
    # O braço sorteado do pré-registro é estratificado por ground truth (12
    # objeto / 10 fundo), e no Schisto só ~7% dos superpixels tocam o objeto.
    # Ele recebe de graça um prior que o argmax de entropia não recebe, e
    # comparar os dois direto favorece o sorteio. A reponderação desfaz isso:
    # é o estimador estratificado padrão, e sai dos mesmos dados.
    pesos = _pesos_estrato(dataset)
    if pesos:
        vs, ids = [], []
        for s in sementes:
            p = (pesos.get(s) or {}).get("p_objeto")
            if p is None:
                vs.append(None)
                continue
            b = por[s]["base"]
            est = {}
            for e in ("objeto", "fundo"):
                cs = [c for c in por[s]["cands"]
                      if c["papel"] == "sorteado" and c["estrato"] == e]
                est[e] = (sum(c["fb"] for c in cs) / len(cs)) if cs else None
                ids += [c["run_id"] for c in cs]
            if est["objeto"] is None or est["fundo"] is None:
                vs.append(None)
                continue
            vs.append(p * est["objeto"] + (1 - p) * est["fundo"] - b["fb"])
        reais = [v for v in vs if v is not None]
        if reais:
            d = ag.descrever(reais)
            tp = ag.teste_pareado(vs, ref)
            ic = (f"[{_fmt(d['ic95'][0], 4)}, {_fmt(d['ic95'][1], 4)}]"
                  if d["ic95"] else "—")
            t.adicionar(
                ["sorteio uniforme (reponderado, sem GT)", d["n"],
                 _fmt(d["media"], 4), ic, _fmt(d["minimo"], 4),
                 _fmt(d["maximo"], 4), _fmt(tp["delta"], 4),
                 _fmt(tp["p"], 4)],
                ids or [por[sementes[0]]["base"]["run_id"]])
    return t


def tabela_ganho_mecanismo(recs=None, experimento: str = "ganho_marginal",
                           dataset: str = "schisto",
                           decoder: str = "FLIM_lm",
                           regime: str = "funcional") -> Tabela:
    """
    O escore de incerteza ranqueia as regiões úteis? E, se não, o que ranqueia?

    A correlação é calculada DENTRO de cada semente, sobre os candidatos
    sorteados daquela semente, e só então os valores por semente entram num
    teste com n = número de sementes. Um Spearman único sobre todos os
    candidatos de todas as sementes trataria observações correlacionadas como
    independentes e inflaria a significância — é o erro que o pré-registro
    desta campanha proíbe explicitamente.

    As variáveis candidatas a explicar Δ:

        entropia      o escore que o Active Learning usa para decidir
        fração fg     quanto da região é objeto — a geometria
        Δ kernels     quantos filtros a mais o encoder conseguiu extrair, que
                      é capacidade e não posição
    """
    recs = ev.carregar() if recs is None else recs
    por = _gm_por_semente(recs, experimento, dataset, decoder)
    if not por:
        raise ValueError(f"nenhuma execução de {experimento}/{dataset}")
    if regime != "todos":
        quer = (regime == "colapsada")
        por = {k: v for k, v in por.items()
               if _colapsou(v["base"]["fb"]) == quer}
    if not por:
        raise ValueError(
            f"nenhuma semente no regime '{regime}' para {dataset}/{decoder}")
    sementes = sorted(por, key=lambda s: int(s))

    t = Tabela(
        f"ganho_mecanismo_{dataset}_{decoder.replace('*', 'x')}_{regime}",
        "O que prevê o ganho de anotar uma região — "
        f"{NOME_DATASET.get(dataset, dataset)}, {decoder}",
        ["variável", "ρ de Spearman com Δ Fβ", "IC95%", "n (sementes)",
         "p (ρ ≠ 0)"],
        alinhamento="lrrrr",
        nota=("ρ calculado **dentro** de cada semente, sobre os candidatos "
              "sorteados daquela semente; são os ρ por semente que entram no "
              "teste, com n = sementes. Agregar candidatos de sementes "
              "diferentes num ρ único trataria observações correlacionadas "
              "como independentes. `entropia da região` é o escore que o "
              "Active Learning usa para decidir — um ρ indistinguível de "
              "zero significa que o escore não ordena as regiões por "
              "utilidade, e é isso que explicaria todos os resultados "
              "negativos das campanhas anteriores. `Δ kernels` mede "
              "capacidade, não posição: é quantos filtros a mais o encoder "
              "extraiu com o marcador extra."))

    base_k = {s: (por[s]["base"]["kernels"] or 0) for s in sementes}
    variaveis = [
        ("entropia da região", lambda c, s: c["ent"]),
        ("least confidence", lambda c, s: c["lc"]),
        ("fração de foreground", lambda c, s: c["fg"]),
        ("Δ kernels do encoder",
         lambda c, s: None if c["kernels"] is None
         else c["kernels"] - base_k[s]),
    ]
    for rotulo, f in variaveis:
        rhos, ids = [], []
        for s in sementes:
            b = por[s]["base"]
            cs = [c for c in por[s]["cands"] if c["papel"] == "sorteado"]
            if len(cs) < 3:
                continue
            rho = _spearman([f(c, s) for c in cs],
                            [c["fb"] - b["fb"] for c in cs])
            if rho is not None:
                rhos.append(rho)
                ids += [c["run_id"] for c in cs]
        if len(rhos) < 2:
            # Um ρ indefinido não é "sem dado": quando a variável é CONSTANTE
            # entre candidatos, isso é o achado. No BraTS a arquitetura fixa 8
            # kernels por camada e a anotação extra acrescenta zero — então o
            # Δ de kernels é constante e, ainda assim, o Fβ varia. Omitir a
            # linha esconderia exatamente isso.
            constante = all(
                len({f(c, s) for c in por[s]["cands"]
                     if c["papel"] == "sorteado"}) <= 1
                for s in sementes
                if [c for c in por[s]["cands"] if c["papel"] == "sorteado"])
            if constante:
                t.adicionar([rotulo, "constante", "—", len(sementes),
                             "não estimável"],
                            [por[s]["base"]["run_id"] for s in sementes])
            continue
        d = ag.descrever(rhos)
        tp = ag.teste_pareado(rhos, [0.0] * len(rhos))
        ic = (f"[{_fmt(d['ic95'][0], 3)}, {_fmt(d['ic95'][1], 3)}]"
              if d["ic95"] else "—")
        t.adicionar([rotulo, _fmt(d["media"], 3), ic, d["n"],
                     _fmt(tp["p"], 4)], ids)

    # Os estratos lado a lado: anotar objeto contra anotar fundo, na mesma
    # escala de Δ. É a leitura geométrica do mesmo dado.
    for estrato in ("objeto", "fundo"):
        vs, ids = [], []
        for s in sementes:
            b = por[s]["base"]
            cs = [c for c in por[s]["cands"]
                  if c["papel"] == "sorteado" and c["estrato"] == estrato]
            if not cs:
                continue
            vs.append(sum(c["fb"] for c in cs) / len(cs) - b["fb"])
            ids += [c["run_id"] for c in cs]
        if len(vs) < 2:
            continue
        d = ag.descrever(vs)
        ic = (f"[{_fmt(d['ic95'][0], 4)}, {_fmt(d['ic95'][1], 4)}]"
              if d["ic95"] else "—")
        t.adicionar([f"(Δ Fβ médio ao anotar região de {estrato})",
                     _fmt(d["media"], 4), ic, d["n"], "—"], ids)
    return t


# ── esforço de interação: NoC ────────────────────────────────────────────────

def _noc_campos(r: dict) -> dict:
    """
    Lê NoC, sucesso e alpha da `variante` do experimento `il_noc`.

    O registro canônico guarda uma métrica de qualidade por linha, não um par
    (esforço, sucesso). Acrescentar colunas recalcularia o `run_id` das 13 mil
    execuções já gravadas, custo que este projeto já pagou e reverteu. A
    `variante` carrega os três valores como texto, e esta função é o parser.
    """
    v = r.get("variante", "")
    out = {"alpha": None, "noc": None, "atingiu": None, "imagem": None}
    for p in v.split("|"):
        if p.startswith("alpha="):
            out["alpha"] = float(p[6:])
        elif p.startswith("noc="):
            out["noc"] = int(p[4:])
        elif p.startswith("atingiu="):
            out["atingiu"] = bool(int(p[8:]))
        elif not p.startswith("iou0="):
            out["imagem"] = p
    return out


def _mcnemar(a: list, b: list):
    """
    Exato bicaudal sobre os pares discordantes.

    Para taxa de sucesso, o par concordante não carrega informação: as duas
    condições acertaram, ou as duas erraram. O que separa os braços são os
    casos em que uma atinge o alvo e a outra não.
    """
    import math
    sa = sum(1 for x, y in zip(a, b) if x and not y)
    sb = sum(1 for x, y in zip(a, b) if y and not x)
    n = sa + sb
    if n == 0:
        return sa, sb, 1.0
    k = min(sa, sb)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return sa, sb, min(1.0, 2 * p)


def tabela_noc(recs=None, experimento: str = "il_noc",
               rotulo: str = "noc", base: float = 1.0) -> Tabela:
    """
    Cliques até a IoU alvo, por plasticidade do banco de filtros.

    O eixo é o da segmentação interativa, e não o Fβ a orçamento fixo: NoC@X,
    usado por RITM e SimpleClick. Um método pode não mudar o Fβ a orçamento
    fixo e ainda reduzir o esforço de interação, e é essa possibilidade que a
    tabela testa.

    A **taxa de sucesso** anda ao lado do NoC em todas as linhas, e isso não é
    decoração. Imagem que não atinge o alvo entra com o teto de cliques, então
    um braço pode exibir NoC baixo por desistir mais cedo. NoC sozinho
    premiaria esse comportamento.

    `α = 1` é o FLIM puro: o banco de filtros é refeito a cada clique. `α = 0`
    congela o banco, e os cliques só chegam ao modelo pelos rótulos que o
    decoder `labeled_marker` lê. Os valores intermediários interpolam.

    Os testes são SEPARADOS por dataset, de propósito. Agregar Schisto e
    BraTS, que têm taxas de sucesso de 7% e 30%, foi o que inflou um achado
    exploratório desta campanha a p = 0,0265 — o mesmo dado, separado, não
    passa de p = 0,10.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento
           and rotulo in r.get("fonte", "")
           and "piloto" not in r.get("fonte", "")]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}/{rotulo}")

    por = collections.defaultdict(dict)
    for r in sub:
        c = _noc_campos(r)
        if c["alpha"] is None or c["noc"] is None:
            continue
        por[(r["dataset"], c["imagem"])][c["alpha"]] = (
            c["noc"], c["atingiu"], ag._num(r.get("iou")), r["run_id"])

    alphas = sorted({a for v in por.values() for a in v})
    dss = [d for d in ("schisto", "brats", "conjunctiva")
           if d in {k[0] for k in por}]

    t = Tabela(
        f"noc_{rotulo}",
        "Cliques até a IoU alvo, por plasticidade do banco de filtros",
        ["dataset", "α", "n", "NoC médio", "NoC mediano", "taxa de sucesso",
         "IoU final", "Δ NoC vs α=1", "p", "McNemar"],
        alinhamento="lrrrrrrrrr",
        nota=("NoC@X é o número de cliques até a IoU passar do alvo, com teto "
              "de 12; é o eixo de RITM (arXiv:2102.06583) e SimpleClick "
              "(arXiv:2210.11006), e **não** se compara com o NoC publicado "
              "deles, que usa outros conjuntos e outra definição de alvo. "
              "Imagem que não atinge o alvo entra com o teto, e por isso a "
              "**taxa de sucesso** aparece em toda linha: um braço pode "
              "mostrar NoC baixo por desistir mais cedo. "
              "`α=1` é o FLIM puro, que refaz o banco de filtros a cada "
              "clique; `α=0` congela o banco, e o clique chega ao modelo só "
              "pelos rótulos que o decoder `labeled_marker` lê. "
              "Os testes são **separados por dataset**: agregar Schisto e "
              "BraTS, com 7% e 30% de sucesso, inflou um achado exploratório "
              "a p=0,0265 que separado não passa de p=0,10. "
              "O clique é simulado do ground truth pelo protocolo de Xu et "
              "al. (2016), com usuário que acerta sempre, então a tabela mede "
              "**número de interações e não tempo de especialista**."))

    for ds in dss:
        ims = [k for k in por if k[0] == ds]
        for al in alphas:
            pares = [k for k in ims if al in por[k] and base in por[k]]
            if not pares:
                continue
            nocs = [por[k][al][0] for k in pares]
            ats = [bool(por[k][al][1]) for k in pares]
            ious = [por[k][al][2] for k in pares
                    if por[k][al][2] is not None]
            ids = [por[k][al][3] for k in pares]
            if al == base:
                d_txt, p_txt, mc_txt = "—", "—", "—"
            else:
                tp = ag.teste_pareado(nocs, [por[k][base][0] for k in pares])
                sa, sb, pm = _mcnemar(
                    ats, [bool(por[k][base][1]) for k in pares])
                d_txt = _fmt(tp["delta"], 2)
                p_txt = _fmt(tp["p"], 4)
                mc_txt = f"{sa}/{sb} (p={_fmt(pm, 3)})"
                ids += [por[k][base][3] for k in pares]
            t.adicionar(
                [NOME_DATASET.get(ds, ds), f"{al:.2f}", len(pares),
                 _fmt(sum(nocs) / len(nocs), 2),
                 _fmt(sorted(nocs)[len(nocs) // 2], 1),
                 f"{sum(ats) / len(ats):.0%}",
                 _fmt(sum(ious) / len(ious), 4) if ious else "—",
                 d_txt, p_txt, mc_txt],
                ids)
    return t


# ── o gerador de candidatos ─────────────────────────────────────────────────

def tabela_custo_do_al(recs=None, experimento: str = "ganho_marginal",
                       criterio: str = "argmax_entropia") -> Tabela:
    """
    O custo do Active Learning: o teto, o que o critério captura, e a lacuna.

    As duas primeiras colunas de número já existem, espalhadas pelas seis
    tabelas de ganho marginal. O que esta tabela acrescenta é a SUBTRAÇÃO
    pareada entre elas, que é o resultado central da dissertação e que não
    pode ser lido das outras sem o leitor fazer a conta de cabeça, o que
    perderia o teste.

    Teto e critério saem da mesma semente, com a mesma base e a mesma
    validação, então a diferença é pareada e não mistura condições.

    O `teto amostrado` escolhe o candidato pelo Fβ da validação: NÃO é
    estratégia, e mede o que existe para capturar entre os candidatos
    examinados. A coluna `captura` é razão entre duas médias, fica instável
    quando o denominador é pequeno, e serve de ilustração e não de
    estatística; quem tem teste é a lacuna.
    """
    recs = ev.carregar() if recs is None else recs
    t = Tabela(
        "custo_do_al",
        "O custo do Active Learning: a margem que existe e a que se captura",
        ["conjunto", "decodificador", "n", "teto amostrado",
         "o critério", "lacuna", "p da lacuna", "captura"],
        alinhamento="llrrrrrr",
        nota=("Todas as colunas são diferenças de Fβ contra o **candidato "
              "sorteado**, medidas na validação, com a partição, as imagens, "
              "os marcadores base e o codificador inicial idênticos dentro de "
              "cada semente. `teto amostrado` é o melhor candidato entre os "
              "examinados, escolhido **pelo Fβ da validação**: não é "
              "estratégia, e mede o que existe para capturar, não um teto "
              "absoluto. `o critério` é o que o Active Learning de fato "
              "escolhe, sem olhar resultado. A **lacuna** entre os dois é o "
              "custo de o critério errar o lugar do clique, e é a coluna com "
              "teste: t pareado por semente, com n = sementes, porque os "
              "candidatos de uma mesma semente compartilham codificador. "
              "`captura` é razão entre duas médias, fica instável com "
              "denominador pequeno e serve de ilustração, não de "
              "estatística. Linhas restritas à **base funcional**; a "
              "estratificação por regime é post-hoc e forçada pela "
              "aritmética, porque Δ a partir de Fβ=0 exato não pode ser "
              "negativo. As sementes compartilham pool e validação, então o "
              "IC vale para esta validação sob sorteio das imagens de treino "
              "e não generaliza para o dataset. O conjunto de teste não foi "
              "tocado por esta campanha."))

    for ds in ("schisto", "brats", "conjunctiva"):
        for dec in ("FLIM_lm", "FLIM_pb"):
            try:
                por = _gm_por_semente(recs, experimento, ds, dec)
            except Exception:                                 # noqa: BLE001
                continue
            por = {k: v for k, v in por.items()
                   if not _colapsou(v["base"]["fb"])}
            if not por:
                continue

            tetos, crits, lacunas, ids = [], [], [], []
            for s in sorted(por, key=lambda x: int(x)):
                v = por[s]
                sort = [c for c in v["cands"] if c["papel"] == "sorteado"]
                alvo = next((c for c in v["cands"]
                             if c["papel"] == criterio), None)
                if not sort or alvo is None:
                    continue
                ref = sum(c["fb"] for c in sort) / len(sort)
                tetos.append(max(c["fb"] for c in sort) - ref)
                crits.append(alvo["fb"] - ref)
                lacunas.append(tetos[-1] - crits[-1])
                ids += [c["run_id"] for c in sort] + [alvo["run_id"]]
            if len(lacunas) < 2:
                continue

            dt, dc = ag.descrever(tetos), ag.descrever(crits)
            dl = ag.descrever(lacunas)
            tp = ag.teste_pareado(tetos, crits)
            cap = (dc["media"] / dt["media"] * 100.0) if dt["media"] else None
            t.adicionar(
                [NOME_DATASET.get(ds, ds), dec, dt["n"],
                 _fmt(dt["media"], 4), _fmt(dc["media"], 4),
                 _fmt(dl["media"], 4), _fmt(tp["p"], 4),
                 f"{cap:.0f}\\%" if cap is not None else "—"],
                ids, enfase=5 if tp["p"] is not None and tp["p"] < 0.05
                else None)
    return t


def tabela_gerador(recs=None, experimento: str = "ganho_marginal",
                   rotulo: str = "contorno") -> Tabela:
    """
    Trocar o gerador de candidatos, com o critério e o orçamento fixos.

    Os dois braços sorteiam o candidato e gastam os mesmos ~300 px. A única
    diferença é de onde vem a lista: superpixels SLIC num caso, discos
    centrados no contorno previsto no outro. Não há escore envolvido, de
    propósito, porque misturar gerador e critério mediria duas coisas.

    A unidade de análise é a SEMENTE. Os dois decodificadores de uma mesma
    semente compartilham o encoder e não são observações independentes, então
    entram colapsados por média antes do teste. Tratá-los como independentes
    dobraria o n artificialmente, que é o erro que este projeto já cometeu uma
    vez ao reportar n=42 onde o n real era 6.

    As colunas de pior caso e de dispersão não são exploração: foram
    declaradas no pré-registro com o mecanismo escrito antes de rodar. O
    candidato de contorno é pequeno e local, acrescenta menos filtros novos, e
    o diagnóstico de ganho marginal havia medido correlação negativa entre
    filtros acrescentados e ganho.
    """
    recs = ev.carregar() if recs is None else recs
    sub = [r for r in recs if r["experimento"] == experimento
           and rotulo in r.get("fonte", "")
           and "smoke" not in r.get("fonte", "")]
    if not sub:
        raise ValueError(f"nenhuma execução de {experimento}/{rotulo}")

    por = collections.defaultdict(lambda: collections.defaultdict(dict))
    ids = collections.defaultdict(list)
    for r in sub:
        v = ag._num(r.get("fb"))
        if v is None:
            continue
        d = por[r["dataset"]][(int(r["seed"]), r["decoder_paper"])]
        d.setdefault(r["criterio"], []).append(v)
        ids[r["dataset"]].append(r["run_id"])

    t = Tabela(
        f"gerador_{rotulo}",
        "Gerador de candidatos: contorno previsto contra superpixel",
        ["dataset", "n (sementes)", "Δ médio", "IC95%", "vitórias",
         "p", "Δ pior caso", "p ", "Δ dispersão", "p  "],
        alinhamento="lrrrrrrrrr",
        nota=("Δ é a variação de Fβ na validação ao acrescentar **uma** "
              "anotação, do braço de contorno menos o braço de superpixel. "
              "Os dois sorteiam o candidato e gastam os mesmos ~300 px: a "
              "**única** diferença é o gerador da lista, e por isso a "
              "comparação isola o gerador e não diz nada sobre qual escore "
              "usar. A unidade é a **semente**; os dois decodificadores de "
              "uma semente compartilham o encoder e entram colapsados por "
              "média. Pior caso e dispersão foram declarados no "
              "pré-registro, com o mecanismo escrito antes de rodar: o "
              "candidato de contorno é pequeno e local, acrescenta menos "
              "filtros, e o ganho marginal mede correlação negativa entre "
              "filtros acrescentados e ganho. O nulo do BraTS é **previsto**: "
              "a arquitetura dele fixa o banco em 8 filtros por camada, os "
              "dois braços acrescentam exatamente 24, e o deslocamento que a "
              "intervenção evita não pode ocorrer. Semente cuja base "
              "degenerou (Fβ=0) sai, porque Δ a partir de zero não pode ser "
              "negativo."))

    for ds in ("schisto", "brats", "conjunctiva"):
        if ds not in por:
            continue
        A, B, PA, PB, SA, SB = [], [], [], [], [], []
        for (s, dec), c in sorted(por[ds].items()):
            if not {"base", "sorteado", "contorno_sorteado"} <= set(c):
                continue
            bz = c["base"][0]
            if bz <= 0:
                continue
            ca = [x - bz for x in c["contorno_sorteado"]]
            cb = [x - bz for x in c["sorteado"]]
            A.append((s, sum(ca) / len(ca)))
            B.append((s, sum(cb) / len(cb)))
            PA.append((s, min(ca)))
            PB.append((s, min(cb)))
            if len(ca) > 1 and len(cb) > 1:
                import statistics as _st
                SA.append((s, _st.stdev(ca)))
                SB.append((s, _st.stdev(cb)))

        def _por_semente(pares):
            d = collections.defaultdict(list)
            for s, v in pares:
                d[s].append(v)
            return [sum(v) / len(v) for _, v in sorted(d.items())]

        a, b = _por_semente(A), _por_semente(B)
        if len(a) < 3:
            continue
        pa, pb = _por_semente(PA), _por_semente(PB)
        tp = ag.teste_pareado(a, b)
        tw = ag.teste_pareado(pa, pb)
        ic = (f"[{_fmt(tp['ic95'][0], 4)}, {_fmt(tp['ic95'][1], 4)}]"
              if tp["ic95"] else "—")
        if SA and SB:
            sa, sb = _por_semente(SA), _por_semente(SB)
            ts = ag.teste_pareado(sa, sb)
            d_txt, p_txt = _fmt(ts["delta"], 4), _fmt(ts["p"], 4)
        else:
            d_txt, p_txt = "—", "—"
        t.adicionar(
            [NOME_DATASET.get(ds, ds), len(a), _fmt(tp["delta"], 4), ic,
             f"{sum(1 for x, y in zip(a, b) if x > y)} de {len(a)}",
             _fmt(tp["p"], 4), _fmt(tw["delta"], 4), _fmt(tw["p"], 4),
             d_txt, p_txt],
            sorted(set(ids[ds])))
    return t
