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
