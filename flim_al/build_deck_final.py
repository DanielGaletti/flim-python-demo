#!/usr/bin/env python3
"""
build_deck_final.py — a apresentação completa da dissertação
============================================================
Tudo que foi feito: reprodução, active learning nos dois datasets, os modos de
falha medidos, o que foi testado e refutado, e a aplicação interativa.

Números vêm dos CSVs da rodada v3 (9 execuções independentes × 3 partições,
teste comum de 364 imagens). Onde um número está fixo no código, é porque veio
de uma medição pontual registrada no texto — e está marcado.

Uso (de dentro de flim_ad/):
    python ../flim_al/build_deck_final.py --out ../FLIM_AL_Dissertacao.pptx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from flim_al import deck_data as D  # noqa: E402
from flim_al.build_deck import (  # noqa: E402
    BLUE, CORAL, DARK, GRAY, GREEN, LIGHT, W, barra, caixa, figura, nota,
    slide, tabela, texto,
)

# ── resultados finais, teste Z₂ comum (364 imagens), média de 3 splits ───────
# 9 execuções independentes do Algoritmo 1 por split, exceto onde indicado.
RES = {
    "FLIM_lm":  [0.788, 0.729, 0.737, 0.738, 0.752],
    "FLIM_pb":  [0.786, 0.729, 0.735, 0.736, 0.768],
    "FLIM_mb":  [0.790, 0.700, 0.707, 0.721, 0.746],
    "FLIM_lt":  [0.730, 0.689, 0.698, 0.695, 0.679],
    "FLIM_ts":  [0.660, 0.647, 0.669, 0.633, 0.653],
    "FLIM_at":  [0.607, 0.599, 0.657, 0.589, 0.636],
    "FLIM_ts*": [0.584, 0.584, 0.585, 0.509, 0.593],
}
COLS = ["Alg. 1 reprod.", "aleatório", "entropia", "CoreSet",
        "CoreSet+orç."]
FORTES = ["FLIM_lm", "FLIM_pb", "FLIM_mb"]

# curva Fβ × número de imagens anotadas (CoreSet, sem ground truth)
CURVA = [(3, 0.638), (5, 0.715), (8, 0.767), (12, 0.767), (16, 0.781),
         (20, 0.772), (25, 0.759), (31, 0.711)]
ARTIGO_LM = 0.788

# ordem dos decoders na comparação região × reprodução
ORDEM_REG = ["labeled_marker", "decoder_2", "decoder_3",
             "hybrid_decoder", "vanilla_adaptive_decoder",
             "decoder_attention", "vanilla_adaptive_decoder_wt"]


def s_capa(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(7.5))
    bg.fill.solid(); bg.fill.fore_color.rgb = DARK; bg.line.fill.background()
    bg.shadow.inherit = False
    f = s.shapes.add_shape(1, Inches(0), Inches(3.15), Inches(W), Inches(0.05))
    f.fill.solid(); f.fill.fore_color.rgb = BLUE; f.line.fill.background()
    f.shadow.inherit = False
    texto(s, "Active Learning aplicado a FLIM", 1.0, 1.55, 11.3, 1.0,
          size=40, cor=RGBColor(0xFF, 0xFF, 0xFF), bold=True)
    texto(s, "Quanto vale a seleção supervisionada de imagens, e quanto dela "
             "se recupera sem anotação", 1.0, 2.45, 11.3, 0.7, size=18,
          cor=RGBColor(0xBF, 0xD4, 0xEC))
    texto(s, ["Daniel Galetti  ·  Mestrado",
              "Reprodução e extensão de Soares et al., arXiv:2504.20872",
              "Defesa: 14 de novembro de 2026"],
          1.0, 4.6, 11.3, 1.6, size=15,
          cor=RGBColor(0x9F, 0xB6, 0xD2), espaco=5)


def s_pergunta(prs):
    s = slide(prs, "O passo que motiva a dissertação",
              "O artigo treina com 3 imagens — mas para saber QUAIS 3, "
              "consulta a anotação de 848")
    caixa(s, 0.55, 1.5, 7.4, 2.5, LIGHT)
    texto(s, ["Algoritmo 1 do artigo, passos 5 a 7:",
              "",
              "5   avalia o modelo nas 848 imagens restantes",
              "6   x ← Fβ médio",
              "7   entre as de MENOR Fβ, seleciona a próxima"],
          0.85, 1.65, 6.9, 2.2, size=13.5, espaco=7)

    caixa(s, 8.25, 1.5, 4.53, 2.5, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "Calcular Fβ exige a resposta certa", 8.55, 1.65, 3.95, 0.45,
          size=15, bold=True, cor=CORAL)
    texto(s, "Os autores declaram a escolha e citam a alternativa sem ground "
             "truth. Não é ponto cego — é uma decisão assumida, porque o foco "
             "do artigo era o decoder.\n\nMas o custo dela nunca foi medido.",
          8.55, 2.2, 3.95, 1.7, size=12.5)

    caixa(s, 0.55, 4.3, 12.23, 1.15, LIGHT, BLUE)
    texto(s, "Quanto vale essa supervisão, e quanto se recupera sem ela?",
          0.9, 4.55, 11.5, 0.5, size=20, bold=True, align=PP_ALIGN.CENTER)
    texto(s, "Para responder é preciso régua com os dois extremos: o teto "
             "(seleção supervisionada) e o piso (sorteio, mesmo orçamento).",
          0.55, 5.65, 12.2, 0.5, size=14, cor=GRAY)
    nota(s, "Protocolo: 3 partições, 9 execuções independentes cada, teste "
            "comum de 364 imagens, bloco escolhido na validação como o artigo "
            "especifica.")


def s_controle(prs):
    s = slide(prs, "A reprodução, com controle experimental",
              "Os números ficaram 0.08 abaixo. A pergunta é se isso é erro "
              "meu ou componente ausente")
    linhas = [["dataset", "Δ médio vs artigo", "negativos", "p (média = 0)",
               "p (sinal)"],
              ["Parasites — o artigo aplica Dynamic Trees",
               ("−0.0797", "red"), ("6/6", "red"), "0.0004", "0.031"],
              ["BraTS — o artigo NÃO aplica Dynamic Trees",
               ("+0.0015", "green"), "3/6", "0.881", "1.000"]]
    tabela(s, linhas, 0.55, 1.6, 12.2, 1.4,
           larguras=[4.4, 2.2, 1.5, 2.0, 1.6], size=13, altura_linha=0.46)

    caixa(s, 0.55, 3.3, 12.23, 1.5, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "Onde o pipeline está completo, a reprodução é exata.",
          0.9, 3.45, 11.5, 0.45, size=19, bold=True, align=PP_ALIGN.CENTER)
    texto(s, "Um erro de implementação apareceria nos dois datasets, ou em "
             "direções imprevisíveis. Não sistemático em exatamente aquele que "
             "recebe uma etapa a mais — e nos seis decoders, o que sozinho tem "
             "p = 0.031.", 0.9, 3.95, 11.5, 0.75, size=13,
          align=PP_ALIGN.CENTER)

    caixa(s, 0.55, 5.05, 12.23, 1.35, LIGHT)
    texto(s, "Achado lateral: a escolha do bloco vale 0.08 de Fβ",
          0.85, 5.2, 6.5, 0.4, size=15, bold=True, cor=BLUE)
    texto(s, "Escolher o bloco do encoder na validação, como o artigo "
             "especifica, leva a reprodução de ρ = 0.60 para ρ = 0.81. Metade "
             "da lacuna que eu atribuía ao Dynamic Trees era, na verdade, "
             "avaliar cada decoder numa profundidade que não era a melhor "
             "para ele.", 0.85, 5.6, 11.6, 0.7, size=12.5)


def s_tabela_geral(prs):
    s = slide(prs, "Active learning: teto, piso e os critérios",
              "Teste Z₂ comum (364 imagens), 9 execuções × 3 partições, "
              "usuário A")
    linhas = [["modelo"] + COLS]
    for k, v in RES.items():
        melhor_al = max(v[2:])
        linhas.append([k] + [
            (f"{x:.3f}", "green" if i >= 2 and x == melhor_al and x > v[1]
             else None) for i, x in enumerate(v)])
    tabela(s, linhas, 0.55, 1.62, 8.5, 3.8,
           larguras=[1.5, 1.3, 1.5, 1.4, 1.4, 1.7], size=12.5,
           altura_linha=0.44)

    art = sum(RES[k][0] for k in FORTES) / 3
    alea = sum(RES[k][1] for k in FORTES) / 3
    cg = sum(RES[k][4] for k in FORTES) / 3
    caixa(s, 9.3, 1.62, 3.48, 1.8, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, f"{100 * (cg - alea) / (art - alea):.0f}%", 9.6, 1.75, 3.0, 0.7,
          size=38, bold=True, cor=GREEN)
    texto(s, "do benefício da supervisão, recuperado sem nenhuma consulta ao "
             "gabarito. p = 0.0001 nos três decoders fortes.",
          9.6, 2.4, 2.95, 0.95, size=12.5)

    caixa(s, 9.3, 3.55, 3.48, 1.85, LIGHT, BLUE)
    texto(s, "O que fez a diferença", 9.6, 3.68, 3.0, 0.4, size=14, bold=True,
          cor=BLUE)
    texto(s, "Não foi trocar o critério — entropia e CoreSet recuperam ~15%. "
             "Foi deixar o algoritmo anotar mais: o backtracking do artigo "
             "corta em 3.3 imagens, e o critério mal começou a trabalhar ali.",
          9.6, 4.1, 2.95, 1.2, size=12)
    nota(s, "A coluna 'Alg. 1 reprod.' é a NOSSA reprodução do Algoritmo 1 "
            "supervisionado neste mesmo teste comum — é o teto, e é contra ela "
            "que as demais colunas devem ser lidas. Não é o número da Tabela IV "
            "do artigo (0.857 para o lm), que usa Dynamic Trees e outro conjunto "
            "de teste. O braço 'CoreSet+orç.' relaxa o critério de parada e "
            "chega a |T| = 6.6 imagens. Verde marca o melhor critério sem "
            "gabarito de cada linha, quando supera o sorteio.")


def s_curva(prs):
    s = slide(prs, "A curva completa, até o pool inteiro",
              "Fβ × número de imagens anotadas, seleção por CoreSet, "
              "zero consultas ao gabarito")
    linhas = [["imagens"] + [str(k) for k, _ in CURVA]]
    linhas.append(["Fβ"] + [(f"{v:.3f}",
                             "green" if v == max(x for _, x in CURVA) else None)
                            for _, v in CURVA])
    tabela(s, linhas, 0.55, 1.7, 12.2, 0.95,
           larguras=[1.4] + [1.0] * len(CURVA), size=12.5, altura_linha=0.44)

    caixa(s, 0.55, 2.95, 6.0, 1.85, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "K = 16 empata com o artigo", 0.85, 3.1, 5.4, 0.45, size=17,
          bold=True, cor=GREEN)
    texto(s, f"0.781 contra {ARTIGO_LM:.3f}, pareado por partição, p = 0.33 — "
             "estatisticamente indistinguível. Numa das partições o AL até "
             "supera.", 0.85, 3.6, 5.4, 1.1, size=13)

    caixa(s, 6.8, 2.95, 5.98, 1.85, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "E depois de 16, cai", 7.1, 3.1, 5.4, 0.45, size=17, bold=True,
          cor=CORAL)
    texto(s, "Com o pool inteiro (31 imagens) o Fβ desaba para 0.711 — pior "
             "que com 8. Existe um número ótimo de imagens, e não é "
             "'quanto mais melhor'.", 7.1, 3.6, 5.4, 1.1, size=13)

    caixa(s, 0.55, 5.0, 12.23, 1.5, LIGHT, BLUE)
    texto(s, "A conta que fecha o argumento", 0.9, 5.12, 6.0, 0.4, size=15,
          bold=True, cor=BLUE)
    linhas2 = [["", "imagens anotadas", "consultas ao gabarito", "Fβ"],
               ["Artigo", "3", ("848", "red"), "0.788"],
               ["Proposto", "16", ("0", "green"), "0.781"]]
    tabela(s, linhas2, 6.4, 5.05, 6.2, 1.3, larguras=[1.3, 1.9, 2.2, 1.0],
           size=12, altura_linha=0.4)
    texto(s, "Rabiscar 16 imagens é uma tarde. Segmentar 848 com precisão é "
             "meses de um especialista.", 0.9, 5.6, 5.3, 0.8, size=12.5)


def s_variancia(prs):
    s = slide(prs, "A variância vem do sorteio, não dos dados",
              "O achado que mudou como todo o resto foi medido")
    linhas = [["fonte de variação", "desvio de Fβ"],
              [("imagem inicial sorteada — passo 1", "red"),
               ("0.10 a 0.22", "red")],
              ["partição dos dados (split)", "0.009"]]
    tabela(s, linhas, 0.55, 1.65, 7.0, 1.4, larguras=[4.5, 2.5], size=14,
           altura_linha=0.45)

    caixa(s, 7.9, 1.65, 4.88, 1.4, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "Uma ordem de grandeza", 8.2, 1.78, 4.3, 0.4, size=16, bold=True,
          cor=CORAL)
    texto(s, "O artigo roda uma execução e reporta o desvio sobre partições — "
             "mede a fonte menor e ignora a maior.",
          8.2, 2.2, 4.3, 0.75, size=12.5)

    caixa(s, 0.55, 3.35, 12.23, 1.5, LIGHT)
    texto(s, "Consequência prática", 0.85, 3.48, 5.0, 0.4, size=15, bold=True,
          cor=BLUE)
    texto(s, "Qualquer conclusão baseada numa execução é uma amostra de uma "
             "distribuição larga. Nesta investigação o “empate” apareceu e "
             "desapareceu três vezes conforme as execuções subiram de 1 para "
             "3 e depois para 9. Foi por isso que o protocolo final usa nove.",
          0.85, 3.9, 11.6, 0.85, size=13)

    caixa(s, 0.55, 5.1, 12.23, 1.35, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "Observável ao vivo na aplicação: três sorteios do braço sem AL "
             "deram 0.479, 0.678 e 0.700 — amplitude de 0.22, e um deles "
             "colapsou.", 0.9, 5.35, 11.5, 0.8, size=15, bold=True,
          align=PP_ALIGN.CENTER)


def s_colapso(prs):
    s = slide(prs, "Dois modos de falha, a mesma causa",
              "Concentrar anotação destrói o encoder — seja em poucas "
              "imagens, seja em poucos traços")
    texto(s, "Orçamento fixo de pinceladas, variando quantas imagens recebem",
          0.55, 1.5, 6.0, 0.4, size=13, bold=True, cor=BLUE)
    l1 = [["orçamento", "2 imagens", "4 imagens", "8 imagens"],
          ["48 pinceladas", ("12/12", "red"), "2/12", ("0/12", "green")],
          ["96 pinceladas", ("12/12", "red"), "9/12", ("0/12", "green")]]
    tabela(s, l1, 0.55, 1.9, 6.0, 1.3, larguras=[1.8, 1.4, 1.4, 1.4], size=12,
           altura_linha=0.42)
    texto(s, "colapsos do encoder em 12 execuções", 0.55, 3.25, 6.0, 0.3,
          size=11.5, cor=GRAY, italico=True)

    texto(s, "Volume de traço por imagem, nas MESMAS três imagens",
          6.85, 1.5, 6.0, 0.4, size=13, bold=True, cor=BLUE)
    l2 = [["estilo do marker", "px", "Fβ"],
          ["toques curtos (4 obj / 9 fundo)", "6.007", ("0.665", "green")],
          ["médio (12 / 27)", "18.419", "0.516"],
          ["traço longo (40 / 90)", "50.046", ("0.000", "red")]]
    tabela(s, l2, 6.85, 1.9, 5.93, 1.7, larguras=[3.1, 1.3, 1.2], size=12,
           altura_linha=0.42)
    texto(s, "medido durante o uso da aplicação", 6.85, 3.65, 6.0, 0.3,
          size=11.5, cor=GRAY, italico=True)

    caixa(s, 0.55, 4.15, 12.23, 1.65, LIGHT, BLUE)
    texto(s, "A causa é a mesma: diluição dos protótipos", 0.9, 4.28, 7.0, 0.4,
          size=16, bold=True, cor=BLUE)
    texto(s, "O FLIM estima os kernels por k-means sobre os patches ao redor "
             "de cada pixel marcado. Um toque curto fica dentro de uma textura "
             "só e vira um protótipo limpo. Um traço arrastado — ou muita "
             "marcação nas mesmas duas imagens — atravessa borda, detrito e "
             "fundo, e o centróide vira a média de tudo: um filtro que não "
             "discrimina nada.", 0.9, 4.72, 11.5, 1.0, size=13)
    nota(s, "Explicação coerente com a construção do método, não medição "
            "direta: isolar diversidade de volume exigiria variar os dois "
            "separadamente.")


def s_defeitos(prs):
    s = slide(prs, "Três defeitos do Algoritmo 1, medidos",
              "Nenhum estava documentado")
    itens = [
        ("O critério prioriza o insegmentável",
         "49% do pool do Schisto não tem objeto. Essas imagens têm Fβ = 0 "
         "porque o filtro de área zera a predição — não porque sejam "
         "informativas. O passo 7 manda anotar justamente elas.", CORAL),
        ("O backtracking pode entrar em ciclo",
         "O pseudocódigo devolve a imagem rejeitada aos candidatos. Como o "
         "critério é determinístico, ela volta a ser a de menor Fβ na rodada "
         "seguinte e o laço trava. Observado com a imagem 000013.", CORAL),
        ("A parada corta o processo cedo demais",
         "Retrocede na primeira piora, e a piora típica é menor que o ruído "
         "entre execuções. Termina com 3.3 imagens; relaxando, chega a 6.6 e "
         "a recuperação salta de 16% para 52%.", BLUE),
    ]
    for i, (t, d, c) in enumerate(itens):
        y = 1.6 + i * 1.55
        caixa(s, 0.55, y, 12.2, 1.35)
        barra(s, 0.55, y, 1.35, c)
        texto(s, t, 0.85, y + 0.12, 7.0, 0.4, size=16, bold=True, cor=c)
        texto(s, d, 0.85, y + 0.55, 11.6, 0.7, size=13)
    nota(s, "O terceiro é o único que rendeu melhoria mensurável — e foi o "
            "que produziu os 52% de recuperação.")


def s_refutados(prs):
    s = slide(prs, "O que testei e os dados refutaram",
              "Reporto porque uma dissertação que só mostra o que deu certo é "
              "menos confiável que uma que mostra o que caiu")
    linhas = [["hipótese", "resultado", "veredito"],
              ["Paciência no critério de parada",
               "dobra a anotação (|T| 2.96 → 5.56) por +0.003 de Fβ",
               ("refutada", "red")],
              ["Inicialização determinística por medoide",
               "p = 0.094; ajuda os fortes, prejudica os fracos",
               ("não sustentada", "red")],
              ["Filtro de candidato degenerado",
               "0.726 contra 0.719 do sorteio — sem ganho",
               ("refutada", "red")],
              ["Método combinado (medoide + filtro)",
               "0.716 — abaixo do próprio sorteio", ("refutada", "red")],
              ["CoreSet com orçamento pequeno",
               "16% de recuperação, pouco acima da entropia",
               ("insuficiente", "red")]]
    tabela(s, linhas, 0.55, 1.72, 12.2, 3.3,
           larguras=[3.6, 5.6, 2.2], size=12.5, altura_linha=0.52)

    caixa(s, 0.55, 5.3, 12.23, 1.15, LIGHT, BLUE)
    texto(s, "Sobrou uma: relaxar o critério de parada, com CoreSet. É a que "
             "produziu os 52%.", 0.9, 5.55, 11.5, 0.5, size=16, bold=True,
          align=PP_ALIGN.CENTER)


def s_app(prs):
    s = slide(prs, "A aplicação: o laço em tempo real",
              "localhost:8000 — o especialista anota, o modelo treina e "
              "escolhe a próxima, tudo ao vivo")
    cols = [
        ("Anotar e treinar", "O sistema escolhe a imagem e diz por quê: "
         "medoide na primeira rodada, CoreSet nas seguintes. Você rabisca, ele "
         "treina em 3 a 5 segundos e mostra a segmentação.", BLUE),
        ("Com AL vs sem AL", "Dois braços, mesmo orçamento, lado a lado. "
         "Sorteios extras expõem a amplitude de 0.22 do braço aleatório.",
         GREEN),
        ("Curva de aprendizado", "Fβ e tempo por K, com treino e teste "
         "medidos separadamente — é o argumento de leveza do FLIM com número.",
         BLUE),
        ("Validar e enviar dados", "O especialista julga as predições e sai a "
         "matriz de confusão. Upload de dataset novo, só com as imagens.",
         GRAY),
    ]
    for i, (t, d, c) in enumerate(cols):
        x = 0.55 + (i % 2) * 6.25
        y = 1.55 + (i // 2) * 1.75
        caixa(s, x, y, 6.0, 1.55)
        barra(s, x, y, 1.55, c)
        texto(s, t, x + 0.28, y + 0.12, 5.4, 0.4, size=16, bold=True, cor=c)
        texto(s, d, x + 0.28, y + 0.55, 5.5, 0.9, size=12.5)

    caixa(s, 0.55, 5.15, 12.23, 1.35, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "A demonstração é, ela própria, um argumento sobre o método.",
          0.9, 5.3, 11.5, 0.4, size=17, bold=True, align=PP_ALIGN.CENTER)
    texto(s, "Só cabe numa interface interativa porque o FLIM não usa "
             "backpropagation no encoder: estimar os kernels é k-means sobre "
             "os patches, 3 a 5 segundos. Um método com épocas de gradiente "
             "não caberia.", 0.9, 5.72, 11.5, 0.6, size=13,
          align=PP_ALIGN.CENTER)


def s_posicao(prs):
    s = slide(prs, "Posicionamento em relação ao estado da arte",
              "Seleção sem ground truth para FLIM já foi publicada pelo grupo "
              "do Falcão")
    linhas = [["trabalho", "seleção", "usa gabarito?", "tem piso aleatório?"],
              ["Soares et al. 2025 (reproduzido)", "Algoritmo 1",
               ("sim, de 848", "red"), ("não", "red")],
              ["Cerqueira et al. 2024 (EMBC)", "interativa, Dice",
               ("sim", "red"), ("não", "red")],
              ["Cerqueira et al. 2024 (SIBGRAPI)",
               "interativa, sem gabarito", ("não", "green"), ("não", "red")],
              ["Esta dissertação", "automática, sem usuário no laço",
               ("não", "green"), ("sim", "green")]]
    tabela(s, linhas, 0.55, 1.7, 12.2, 2.6,
           larguras=[3.6, 3.6, 2.5, 2.5], size=12.5, altura_linha=0.5)

    caixa(s, 0.55, 4.55, 12.23, 1.9, LIGHT, BLUE)
    texto(s, "A distinção", 0.9, 4.68, 5.0, 0.4, size=16, bold=True, cor=BLUE)
    texto(s, "O trabalho do Cerqueira é interativo — um usuário decide a cada "
             "rodada, guiado por explicação visual. O meu é automático, sem "
             "ninguém no laço.\n\nE nenhum dos três mede quanto a supervisão "
             "vale, porque nenhum tem braço aleatório como piso. É esse o "
             "número que esta dissertação acrescenta.",
          0.9, 5.1, 11.5, 1.25, size=13)


def s_contribuicoes(prs):
    s = slide(prs, "Contribuições", "")
    itens = [
        ("Reprodução com controle experimental",
         "A lacuna do Parasites tem causa identificada e demonstrada por um "
         "segundo dataset onde o componente não é aplicado."),
        ("O preço da supervisão: 0.060 de Fβ",
         "Número que o artigo assume mas nunca reporta."),
        ("52% recuperados sem nenhuma consulta",
         "p = 0.0001. E com 16 imagens o empate é estatístico (p = 0.33)."),
        ("A variância é dominada pelo sorteio inicial",
         "0.10 a 0.22 contra 0.009 entre partições — não reportado no artigo."),
        ("Dois modos de falha por concentração",
         "Poucas imagens muito anotadas, ou traços longos: o encoder colapsa."),
        ("Aplicação interativa",
         "O laço completo em tempo real, com validação por especialista."),
    ]
    for i, (t, d) in enumerate(itens):
        x = 0.55 + (i % 2) * 6.25
        y = 1.5 + (i // 2) * 1.6
        caixa(s, x, y, 6.0, 1.4)
        barra(s, x, y, 1.4, GREEN, w=0.07)
        texto(s, t, x + 0.28, y + 0.12, 5.5, 0.45, size=14.5, bold=True)
        texto(s, d, x + 0.28, y + 0.6, 5.5, 0.7, size=12)


def s_limites(prs):
    s = slide(prs, "Limitações — declaradas antes de perguntarem", "")
    itens = [
        "n = 3 partições. Todo intervalo de confiança sofre disso — mitigado "
        "com 9 execuções independentes por partição, que é onde está a "
        "variância dominante.",
        "Sem Dynamic Trees no Parasites. Os valores absolutos não são "
        "comparáveis à Tabela IV, só entre si. No BraTS, onde o artigo não "
        "aplica DT, a comparação é direta.",
        "A curva usa a mesma seleção nas três partições — o pool de markers "
        "reais não depende do split, então não são três observações "
        "independentes do conjunto de treino.",
        "Conclusões de AL com um usuário. O segundo usuário tem apenas o "
        "braço do artigo.",
        "A explicação do colapso é hipótese coerente com a construção do "
        "método, não medição: isolar diversidade de volume exigiria variar os "
        "dois separadamente.",
    ]
    for i, t in enumerate(itens):
        y = 1.5 + i * 1.05
        caixa(s, 0.55, y, 12.2, 0.9)
        barra(s, 0.55, y, 0.9, GRAY, w=0.07)
        texto(s, t, 0.85, y + 0.13, 11.7, 0.7, size=13)


def s_conclusao(prs):
    s = slide(prs, "Conclusão", "")
    caixa(s, 0.55, 1.6, 12.23, 2.0, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "O artigo demonstra que o FLIM aprende com 3 imagens. Esta "
             "dissertação mede que encontrar as 3 imagens certas custa 848 "
             "anotações — e mostra que um procedimento automático, sem "
             "nenhuma consulta, alcança o mesmo desempenho rabiscando 16.",
          0.95, 1.95, 11.4, 1.4, size=19, bold=True, align=PP_ALIGN.CENTER)

    caixa(s, 0.55, 3.85, 12.23, 1.5, LIGHT, BLUE)
    texto(s, "Não é melhoria de acurácia", 0.9, 3.98, 6.0, 0.4, size=16,
          bold=True, cor=BLUE)
    texto(s, "É a substituição de um custo oculto por um custo explícito e "
             "muito menor — 129× menos esforço de anotação — mais a "
             "caracterização de modos de falha do algoritmo publicado que não "
             "estavam documentados.", 0.9, 4.4, 11.5, 0.85, size=13.5)

    caixa(s, 0.55, 5.6, 12.23, 0.9, LIGHT)
    texto(s, "Próximos passos: recuperar Dynamic Trees · estender o AL ao "
             "BraTS com 9 execuções · validar o modo de falha por volume de "
             "traço com anotadores reais", 0.9, 5.78, 11.5, 0.6, size=13,
          cor=GRAY, align=PP_ALIGN.CENTER)


def s_regiao_vs_reproducao(prs, pm):
    """
    Item que faltava: o AL por região comparado com os dados de reprodução,
    e não apenas com os braços aleatórios do próprio experimento de região.
    """
    s = slide(prs, "AL por região vs a reprodução do artigo",
              "Nos dois casos o encoder é retreinado; muda quem desenha os "
              "marcadores — o especialista ou o critério de incerteza")
    REG = {"FLIM lm": (0.575, 0.605), "FLIM pb": (0.623, 0.586),
           "FLIM mb": (0.594, 0.560), "FLIM at": (0.350, 0.259),
           "FLIM lt": (0.503, 0.425), "FLIM ts": (0.466, 0.436),
           "FLIM ts*": (0.338, 0.226)}
    linhas = [["Decoder", "reprodução", "AL região", "random região",
               "Δ  região − reprod."]]
    deltas = []
    for cod in ORDEM_REG:
        nome = D.COD2PAPER[cod]
        if nome not in REG:
            continue
        p_rep = pm.get(("paper", cod))
        if not p_rep:
            continue
        al, rr = REG[nome]
        d = al - p_rep["fb"]
        deltas.append(d)
        linhas.append([nome, f"{p_rep['fb']:.3f}", f"{al:.3f}", f"{rr:.3f}",
                       (f"{d:+.3f}", "red" if d < 0 else "green")])
    med = sum(deltas) / len(deltas) if deltas else 0.0
    linhas.append(["média", "", "", "", (f"{med:+.3f}", "red")])
    tabela(s, linhas, 0.55, 1.62, 8.05, 3.9,
           larguras=[1.35, 1.75, 1.75, 1.5, 1.7], size=12, altura_linha=0.44)

    caixa(s, 8.85, 1.62, 3.93, 1.9, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, f"{med:+.3f}", 9.15, 1.74, 3.4, 0.7, size=38, bold=True,
          cor=CORAL)
    texto(s, "de Fβ é o que se perde ao trocar os marcadores do especialista "
             "por seeds desenhados pelo critério de incerteza. O AL por "
             "região NÃO alcança a reprodução.", 9.15, 2.4, 3.35, 1.05,
          size=12.5)

    caixa(s, 8.85, 3.65, 3.93, 1.85, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "O que ele ganha", 9.15, 3.78, 3.4, 0.4, size=14, bold=True,
          cor=GREEN)
    texto(s, "Contra o seu próprio piso, o AL por região ganha +0.050 de "
             "seleção e +0.108 de geometria. O nível de região melhora a "
             "anotação automática — não substitui o especialista.",
          9.15, 4.2, 3.35, 1.2, size=12)

    texto(s, "Conclusão desta comparação: o AL por região resolve ONDE anotar "
             "quando ninguém vai anotar; a reprodução mostra que, quando há "
             "especialista, o traço dele ainda vale mais que qualquer mapa de "
             "incerteza.", 0.55, 5.62, 8.05, 0.75, size=13.5, bold=True)
    nota(s, "Protocolos diferentes e declarados: a reprodução usa os 31 "
            "markers reais e o teste Z₂ do artigo; o experimento de região "
            "gera os seeds a partir do mapa de incerteza, com |T| pequeno. A "
            "coluna Δ é uma diferença de nível entre dois regimes de "
            "anotação, não um teste pareado.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../FLIM_AL_Dissertacao.pptx")
    a = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(7.5)

    from flim_al.build_deck import (
        s_algoritmo1, s_antes_depois, s_baselines, s_encoder, s_modelos,
        s_niveis, s_decomposicao, s_seg_modelos, s_tabela3, s_tabela4,
        s_reproducao as s_tabela_reproduzida,
    )
    rep = D.reproducao()
    pm = D.por_modelo()
    print(f"reprodução: {len(rep)} células · por modelo: {len(pm)} células")

    s_capa(prs)
    s_encoder(prs)
    s_modelos(prs)
    s_baselines(prs)
    s_tabela3(prs)
    s_tabela4(prs)
    s_tabela_reproduzida(prs, rep, pm)
    s_controle(prs)
    s_pergunta(prs)
    s_algoritmo1(prs)
    s_tabela_geral(prs)
    s_curva(prs)
    s_seg_modelos(prs)
    s_antes_depois(prs, "ganho")
    s_antes_depois(prs, "perda")
    s_variancia(prs)
    s_colapso(prs)
    s_niveis(prs)
    s_decomposicao(prs)
    s_regiao_vs_reproducao(prs, pm)
    s_defeitos(prs)
    s_refutados(prs)
    s_app(prs)
    s_posicao(prs)
    s_contribuicoes(prs)
    s_limites(prs)
    s_conclusao(prs)

    prs.save(a.out)
    print(f"[salvo] {a.out} · {len(prs.slides._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
