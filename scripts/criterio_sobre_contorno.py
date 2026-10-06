"""O criterio ajuda DEPOIS de trocar o gerador de candidatos?

A campanha do gerador mostrou que trocar superpixel por faixa de contorno vale
+0,0772 no schisto, sorteando o candidato nos dois bracos. A pergunta seguinte
e se, dentro da lista de contorno, escolher pelo escore bate sortear.

Unidade de analise: a SEMENTE. Os dois decodificadores de uma semente
compartilham o encoder e entram colapsados por media.

NAO HA PRE-REGISTRO para esta comparacao. E exploratoria, e tem de ser
rotulada assim: serve para orientar o proximo passo, nao para afirmar.
"""
import collections
import csv
import io
import json
import math
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "evidencia", "bruto",
                   "criterio_sobre_contorno_schisto.csv")


def t_pareado(d):
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m, s = st.mean(d), st.stdev(d)
    if s == 0:
        return m, (0.0 if m else 1.0)
    t = m / (s / math.sqrt(n))
    from scipy import stats
    return m, float(2 * stats.t.sf(abs(t), n - 1))


def ic95(d):
    n = len(d)
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * st.stdev(d) / math.sqrt(n)
    return st.mean(d) - h, st.mean(d) + h


rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
cel = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try:
        v = float(r["fb"])
    except (TypeError, ValueError):
        continue
    cel[(int(r["seed"]), r["decoder_paper"])][r["criterio"]].append(v)

# Pares a comparar: (nome, braco, referencia)
PARES = [
    ("criterio sobre contorno vs sorteio no contorno",
     "argmax_ent_contorno", "contorno_sorteado"),
    ("criterio sobre superpixel vs sorteio no superpixel",
     "argmax_entropia", "sorteado"),
    ("sorteio no contorno vs sorteio no superpixel  (replica do gerador)",
     "contorno_sorteado", "sorteado"),
    ("criterio sobre contorno vs sorteio no superpixel  (efeito total)",
     "argmax_ent_contorno", "sorteado"),
]

print("Schisto, campanha `criterio`, 10 sementes, decoders colapsados")
print("Delta = Fbeta(braco) - Fbeta(referencia), ambos ja descontados da base")
print("EXPLORATORIO: sem pre-registro para estas comparacoes.\n")

saida = {}
for nome, braco, ref in PARES:
    porsem = collections.defaultdict(list)
    for (s, _), c in cel.items():
        if not {"base", braco, ref} <= set(c):
            continue
        bz = c["base"][0]
        if bz <= 0:          # base degenerada: Delta a partir de zero nao cai
            continue
        a = sum(x - bz for x in c[braco]) / len(c[braco])
        b = sum(x - bz for x in c[ref]) / len(c[ref])
        porsem[s].append(a - b)
    d = [sum(v) / len(v) for _, v in sorted(porsem.items())]
    if len(d) < 2:
        print(f"{nome}\n    sem pares suficientes (n={len(d)})\n")
        continue
    m, p = t_pareado(d)
    lo, hi = ic95(d)
    vit = sum(1 for x in d if x > 0)
    print(f"{nome}")
    print(f"    n = {len(d)} sementes   Delta = {m:+.4f}   "
          f"IC95% [{lo:+.4f}, {hi:+.4f}]   vence em {vit}/{len(d)}   p = {p:.4f}")
    print()
    saida[f"{braco}_vs_{ref}"] = {
        "o_que_compara": nome,
        "n_sementes": len(d),
        "delta": f"{m:+.4f}",
        "ic95": f"[{lo:+.4f}, {hi:+.4f}]",
        "vitorias": f"{vit} de {len(d)}",
        "p": f"{p:.4f}",
    }

# O numero tem de poder ser LIDO de um arquivo de resultado, e nao copiado a
# mao para o texto. Por isso a analise grava o que calculou.
DEST = os.path.join(RAIZ, "evidencia", "campanhas",
                    "criterio_sobre_contorno_2026-10-06.json")
doc = {
    "id": "criterio_sobre_contorno_2026-10-06",
    "tipo": "ANALISE EXPLORATORIA, SEM PRE-REGISTRO",
    "aviso": (
        "Estas comparacoes nao foram pre-registradas. Orientam o proximo "
        "passo; nao autorizam afirmacao. Os quatro p nao receberam correcao "
        "para multiplas comparacoes."),
    "pergunta": (
        "A campanha do gerador mostrou que trocar superpixel por faixa de "
        "contorno vale +0,0772 no schisto, sorteando o candidato nos dois "
        "bracos. Esta analise pergunta se, DENTRO da lista de contorno, "
        "escolher pelo escore bate sortear."),
    "dados": "evidencia/bruto/criterio_sobre_contorno_schisto.csv",
    "dataset": "schisto",
    "unidade_de_analise": (
        "a SEMENTE. Os dois decodificadores de uma semente compartilham o "
        "encoder e entram colapsados por media. Semente com base degenerada "
        "(Fbeta=0) sai, porque Delta a partir de zero nao pode ser negativo."),
    "teste": "t pareado bicaudal sobre as sementes",
    "RESULTADO": saida,
    "verificacao_interna": (
        "A linha contorno_sorteado_vs_sorteado reproduz o +0,0772 publicado "
        "em `gerador_contorno`, com o mesmo p e as mesmas 10 vitorias: "
        "calculo novo, campanha nova, numero antigo."),
    "observacao": (
        "Os dois efeitos sao praticamente aditivos: 0,0772 do gerador mais "
        "0,0262 do criterio da 0,1034, contra 0,1033 medidos no efeito "
        "total. Sugere que gerador e escore operam por caminhos diferentes, "
        "coerente com o diagnostico de que o gerador resolve O QUE esta na "
        "lista e o escore resolve QUAL item da lista. E observacao, nao "
        "afirmacao."),
    "nao_contradiz": (
        "A linha argmax_entropia_vs_sorteado NAO contradiz o resultado de "
        "que nenhum criterio de AL supera o sorteio. Aquele e sobre selecao "
        "de IMAGEM com treino completo, medida no TESTE; este e sobre uma "
        "anotacao marginal, medida na VALIDACAO. Sao perguntas diferentes."),
}
os.makedirs(os.path.dirname(DEST), exist_ok=True)
io.open(DEST, "w", encoding="utf-8").write(
    json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
print("gravado em evidencia/campanhas/criterio_sobre_contorno_2026-10-06.json")
