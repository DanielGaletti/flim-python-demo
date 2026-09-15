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


def _tex_escape(s: str) -> str:
    for de, para in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                     ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
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

    def gravar(self, destino: str = DESTINO) -> dict:
        os.makedirs(destino, exist_ok=True)
        escritos = {}
        for ext, conteudo in (("csv", self.csv()),
                              ("md", self.markdown()),
                              ("tex", self.latex())):
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
