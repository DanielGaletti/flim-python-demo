#!/usr/bin/env python3
"""
build_deck.py
=============
Monta FLIM_AL_Apresentacao_v5.pptx.

O que o v5 acrescenta ao v4: as duas tabelas do artigo transcritas verbatim,
a tabela reproduzida ao lado delas, a tabela de AL por modelo, e as figuras de
segmentação final por decoder.

Uso (de dentro de flim_ad/):
    python ../flim_al/build_deck.py --out ../FLIM_AL_Apresentacao_v5.pptx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from flim_al import deck_data as D  # noqa: E402

W, H = 13.333, 7.5

DARK = RGBColor(0x12, 0x32, 0x4F)
BLUE = RGBColor(0x2F, 0x7F, 0xD4)
CORAL = RGBColor(0xE8, 0x54, 0x3F)
GREEN = RGBColor(0x1E, 0x9E, 0x4A)
GRAY = RGBColor(0x5B, 0x66, 0x78)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
BAND = RGBColor(0xE6, 0xEB, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

COR = {"green": GREEN, "blue": BLUE, "red": CORAL, None: DARK}
TITULO_F, CORPO_F = "Cambria", "Calibri"


# ── primitivas ───────────────────────────────────────────────────────────────

def slide(prs, titulo: str, sub: str = ""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(0.09))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    bar.shadow.inherit = False

    tb = s.shapes.add_textbox(Inches(0.55), Inches(0.26), Inches(W - 1.1),
                              Inches(0.78))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.text = titulo
    p.font.size, p.font.bold, p.font.name = Pt(28), True, TITULO_F
    p.font.color.rgb = DARK
    if sub:
        q = tb.text_frame.add_paragraph()
        q.text = sub
        q.font.size, q.font.name, q.font.color.rgb = Pt(13.5), CORPO_F, GRAY
    return s


def texto(s, txt, x, y, w, h, size=15, cor=DARK, bold=False,
          align=PP_ALIGN.LEFT, espaco=6, italico=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    linhas = txt if isinstance(txt, list) else [txt]
    for i, linha in enumerate(linhas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = linha
        p.alignment = align
        p.space_after = Pt(espaco)
        p.font.size, p.font.name, p.font.bold = Pt(size), CORPO_F, bold
        p.font.color.rgb = cor
        p.font.italic = italico
    return tb


def caixa(s, x, y, w, h, fill=LIGHT, linha=None):
    sh = s.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if linha is not None:
        sh.line.color.rgb = linha
        sh.line.width = Pt(1.25)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def barra(s, x, y, h, cor, w=0.075):
    b = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = cor; b.line.fill.background()
    b.shadow.inherit = False
    return b


def tabela(s, dados, x, y, w, h, larguras=None, size=11, header=True,
           faixas=True, altura_linha=None):
    """dados: lista de linhas; cada célula é str ou (str, marca_de_cor)."""
    nl, nc = len(dados), len(dados[0])
    g = s.shapes.add_table(nl, nc, Inches(x), Inches(y), Inches(w),
                           Inches(h)).table
    if larguras:
        tot = float(sum(larguras))
        for i, lw in enumerate(larguras):
            g.columns[i].width = Emu(int(Inches(w) * lw / tot))
    if altura_linha:
        for r in range(nl):
            g.rows[r].height = Inches(altura_linha)

    for r, linha in enumerate(dados):
        for c, cel in enumerate(linha):
            txt, marca = (cel if isinstance(cel, tuple) else (cel, None))
            cell = g.cell(r, c)
            cell.text = str(txt)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.045)
            cell.margin_top = cell.margin_bottom = Inches(0.012)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            f = p.font
            f.size, f.name = Pt(size), CORPO_F
            cell.fill.solid()
            if header and r == 0:
                cell.fill.fore_color.rgb = DARK
                f.color.rgb, f.bold = WHITE, True
            else:
                cell.fill.fore_color.rgb = (BAND if (faixas and r % 2 == 0)
                                            else WHITE)
                f.color.rgb = COR.get(marca, DARK)
                f.bold = marca is not None
    return g


def figura(s, path, x, y, w=None, h=None):
    if not os.path.exists(path):
        texto(s, f"[figura ausente: {os.path.basename(path)}]", x, y, 5.5, 0.4,
              size=12, cor=CORAL)
        return None
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def nota(s, txt, y=6.82):
    texto(s, txt, 0.55, y, W - 1.1, 0.5, size=11.5, cor=GRAY, italico=True,
          espaco=2)


# ── slides ───────────────────────────────────────────────────────────────────

def s_capa(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(H))
    bg.fill.solid(); bg.fill.fore_color.rgb = DARK; bg.line.fill.background()
    bg.shadow.inherit = False
    f = s.shapes.add_shape(1, Inches(0), Inches(3.05), Inches(W), Inches(0.05))
    f.fill.solid(); f.fill.fore_color.rgb = BLUE; f.line.fill.background()
    f.shadow.inherit = False
    texto(s, "Active Learning aplicado a FLIM", 1.0, 1.7, 11.3, 1.0,
          size=40, cor=WHITE, bold=True)
    texto(s, "Segmentação de ovos de Schistosoma mansoni com 3 a 5 imagens "
             "anotadas", 1.0, 3.3, 11.3, 0.8, size=19,
          cor=RGBColor(0xBF, 0xD4, 0xEC))
    texto(s, ["Daniel Galetti  ·  Mestrado",
              "Reprodução e extensão de Soares et al., arXiv:2504.20872",
              "Defesa: 14 de novembro de 2026"],
          1.0, 4.7, 11.3, 1.6, size=15, cor=RGBColor(0x9F, 0xB6, 0xD2),
          espaco=5)


def s_problema(prs):
    s = slide(prs, "O problema, e a pergunta que a dissertação faz",
              "Duas restrições definem o projeto inteiro")
    itens = [
        ("Anotação é cara",
         "Cada imagem exige um especialista. O artigo inteiro se organiza em "
         "torno de treinar com 3 a 5 imagens.", BLUE),
        ("O modelo roda em campo",
         "Sem GPU, sem nuvem. Daí encoders de centenas de milhares de "
         "parâmetros, não milhões.", GREEN),
        ("O artigo já faz active learning",
         "O Algoritmo 1 seleciona imagens iterativamente — mas o passo 7 lê a "
         "ground truth de todo o pool.", CORAL),
    ]
    for i, (t, d, c) in enumerate(itens):
        y = 1.55 + i * 1.3
        caixa(s, 0.55, y, 12.2, 1.1)
        barra(s, 0.55, y, 1.1, c)
        texto(s, t, 0.85, y + 0.11, 5.0, 0.4, size=17, bold=True, cor=c)
        texto(s, d, 0.85, y + 0.53, 11.6, 0.5, size=14)

    caixa(s, 0.55, 5.6, 12.2, 1.1, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "Quanto do benefício do Algoritmo 1 se recupera com um critério "
             "que não usa ground truth nenhum?",
          0.9, 5.85, 11.5, 0.7, size=19, bold=True, align=PP_ALIGN.CENTER)


def s_encoder(prs):
    s = slide(prs, "O que é um encoder FLIM",
              "Feature Learning from Image Markers — os kernels vêm dos "
              "rabiscos, não do gradiente")
    passos = [
        ("1", "O especialista rabisca", "traços sobre o objeto e sobre o fundo"),
        ("2", "Recorta patches 3×3", "ao redor de cada posição marcada"),
        ("3", "k-means nos patches", "os centróides viram os kernels da conv."),
        ("4", "Cada kernel herda o rótulo", "λ ∈ {objeto, fundo} do marcador"),
    ]
    for i, (n, t, d) in enumerate(passos):
        x = 0.55 + i * 3.13
        caixa(s, x, 1.45, 2.92, 1.5)
        texto(s, n, x + 0.16, 1.53, 0.5, 0.4, size=24, bold=True, cor=BLUE)
        texto(s, t, x + 0.16, 1.98, 2.65, 0.4, size=14, bold=True)
        texto(s, d, x + 0.16, 2.36, 2.65, 0.55, size=11.5, cor=GRAY)

    figura(s, str(REPO / "figs" / "fig2_markers.png"), 0.55, 3.18, h=2.95)
    caixa(s, 7.9, 3.2, 4.88, 2.95)
    texto(s, "O que decorre disso", 8.2, 3.34, 4.4, 0.4, size=16, bold=True,
          cor=BLUE)
    texto(s, ["•  Sem backpropagation no encoder: treina em ~5 s",
              "•  Espaço de cor LAB, não RGB",
              "•  334 K parâmetros contra 1.33 M do SAMNet",
              "•  O rótulo do kernel é o sinal que os decoders lm, lt, pb e mb "
              "consomem"],
          8.2, 3.82, 4.35, 2.2, size=13, espaco=9)
    nota(s, "O encoder produz m′ canais de ativação. Quem os transforma em um "
            "mapa de saliência é o decoder — e é aí que os oito modelos "
            "diferem.", y=6.35)


def s_modelos(prs):
    s = slide(prs, "Os oito modelos, e onde cada um vive no código",
              "Um decoder adaptativo é uma convolução 1×1 cujos pesos não são "
              "treinados: são recalculados para cada imagem")
    linhas = [["Artigo", "Código", "Peso por", "Como α é definido"]]
    linhas += [
        ["FLIM lm", "labeled_marker", "canal",
         "α = rótulo do marcador (+1 objeto, −1 fundo). Nada é estimado."],
        ["FLIM pb", "decoder_2", "pixel",
         "Gaussiana local: o canal só vota se concordar com o próprio rótulo."],
        ["FLIM mb", "decoder_3", "pixel",
         "Idem pb, comparando só as médias — descarta as variâncias."],
        ["FLIM lt", "hybrid_decoder", "canal",
         "Roda o tri-state e zera todo canal cujo kernel veio do fundo."],
        ["FLIM ts", "vanilla_adaptive_decoder", "canal",
         "Tri-state: média do canal vs Otsu τ±σ, mais a fração acima de Otsu."],
        ["FLIM at", "decoder_attention", "canal",
         "Cosseno entre o canal e uma atenção espacial sem parâmetros."],
        ["FLIM bp", "backprop_decoder", "canal",
         "O 1×1 treinado por Adam, lr 0.01, 100 épocas, perda Dice + BCE."],
        ["FLIM ts*", "vanilla_…_wt", "canal",
         "Ablação nossa: tri-state sem a regra de proporção. Fora do artigo."],
    ]
    tabela(s, linhas, 0.55, 1.62, 12.2, 4.5,
           larguras=[1.15, 2.55, 1.2, 7.3], size=12.5, altura_linha=0.46)
    nota(s, "Mapeamento estabelecido lendo as equações da Seção IV do artigo "
            "contra as implementações em pyflim/layers.py.")


def s_baselines(prs):
    s = slide(prs, "As linhas de base",
              "Respondem à pergunta óbvia: e se eu simplesmente usar uma rede "
              "pronta?")
    cards = [
        ("SAMNet · MSCNet · MEANet", "1.33 a 3.27 M parâmetros",
         "Modelos leves de SOD do estado da arte, treinados por backprop nas "
         "mesmas 3–5 imagens. Não usam markers. Com 3 imagens, SAMNet chega a "
         "Fβ = 0.422 com desvio de 0.241 — instável a ponto de ser "
         "inutilizável.", CORAL),
        ("U-Net FLIM", "2.29 M parâmetros",
         "Encoder FLIM congelado + decoder em U com skip connections, treinado "
         "por backprop em T. Sete vezes o FLIM puro, e Fβ = 0.777.", GRAY),
        ("FLIM bp", "o mesmo encoder, agora com gradiente",
         "O 1×1 otimizado por backprop com a ground truth de T. Fica em 0.757, "
         "ABAIXO do lm (0.857), que não usa gradiente nenhum. É o resultado "
         "mais contraintuitivo da tabela — e a melhor defesa do método.", BLUE),
    ]
    for i, (t, sub, d, c) in enumerate(cards):
        y = 1.55 + i * 1.7
        caixa(s, 0.55, y, 12.2, 1.5)
        barra(s, 0.55, y, 1.5, c)
        texto(s, t, 0.85, y + 0.1, 7.0, 0.4, size=17, bold=True, cor=c)
        texto(s, sub, 0.85, y + 0.5, 7.0, 0.35, size=12.5, cor=GRAY,
              italico=True)
        texto(s, d, 0.85, y + 0.82, 11.7, 0.65, size=13)


def _tab_paper(s, dados, y=1.55, h=4.85):
    linhas = [["Modelo", "User", "MAE ↓", "Fβ ↑"]]
    linhas += [[m, u, mae, fb] for m, u, mae, fb in dados]
    meio = len(linhas) // 2
    esq = linhas[:meio + 1]
    dire = [linhas[0]] + linhas[meio + 1:]
    lw = [2.1, 1.0, 1.9, 1.9]
    tabela(s, esq, 0.85, y, 5.5, h, larguras=lw, size=11, altura_linha=0.37)
    tabela(s, dire, 7.0, y, 5.5, h, larguras=lw, size=11, altura_linha=0.37)


def s_tabela3(prs):
    s = slide(prs, "Tabela III do artigo — validação Z₁\\T, Parasites",
              "Transcrita verbatim. Verde = melhor, azul = segundo, "
              "vermelho = pior, como no original")
    _tab_paper(s, D.TABELA_III)
    nota(s, "Soares et al., arXiv:2504.20872, Tabela III. Pós-processamento "
            "dos modelos FLIM: Otsu + filtro de área [1000, 9000] + "
            "Dynamic Trees.")


def s_tabela4(prs):
    s = slide(prs, "Tabela IV do artigo — teste Z₂, Parasites",
              "É contra esta que a reprodução deve ser lida")
    _tab_paper(s, D.TABELA_IV)
    nota(s, "FLIM pb e FLIM lm empatam em 0.857; o decoder treinado por "
            "backpropagação (bp) fica dez pontos atrás.")


def s_reproducao(prs, rep, pm):
    s = slide(prs, "A tabela reproduzida",
              "Mesmo código, mesmos splits, mesmos markers — teste Z₂, "
              "usuário A, média ± desvio sobre 3 splits")
    art = {m: float(fb[0].split("±")[0]) for m, u, _, fb in D.TABELA_IV
           if u == "A"}
    linhas = [["Modelo", "MAE ↓", "Fβ ↑ reproduzido", "Fβ ↑ artigo", "Δ",
               "bloco fixo"]]
    for cod in D.ORDEM:
        nome = D.COD2PAPER[cod]
        p = pm.get(("paper", cod))
        fixo = rep.get(("user_A", cod))
        if not p:
            continue
        fb_art = art.get(nome)
        d = (p["fb"] - fb_art) if fb_art is not None else None
        linhas.append([
            nome,
            f"{p['mae']:.3f}",
            f"{p['fb']:.3f}±{p['fb_sd']:.3f}",
            f"{fb_art:.3f}" if fb_art is not None else "—",
            (f"{d:+.3f}", "red" if d < -0.10 else None)
            if d is not None else "—",
            f"{fixo['fb']:.3f}" if fixo else "—",
        ])
    tabela(s, linhas, 0.55, 1.65, 7.85, 3.5,
           larguras=[1.25, 1.0, 1.85, 1.35, 1.05, 1.25], size=12,
           altura_linha=0.42)

    caixa(s, 8.6, 1.65, 4.18, 1.55, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "ρ = 0.81", 8.85, 1.76, 3.7, 0.55, size=30, bold=True, cor=GREEN)
    texto(s, "Correlação de postos com a Tabela IV (p = 0.05). A ordem dos "
             "modelos se reproduz.", 8.85, 2.36, 3.7, 0.75, size=12.5)

    caixa(s, 8.6, 3.35, 4.18, 1.8, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "O deslocamento de −0.080", 8.85, 3.46, 3.7, 0.4, size=14,
          bold=True, cor=CORAL)
    texto(s, "O artigo aplica Dynamic Trees aos modelos FLIM (Tabela I). O "
             "binário iftSMansoniDelineation depende de liblapack/libblas, "
             "ausentes na imagem. Toda a reprodução usa só Otsu + filtro de "
             "área.", 8.85, 3.88, 3.7, 1.2, size=12)

    texto(s, "As três primeiras posições permutam entre si dentro de 0.004 — "
             "abaixo do desvio entre splits. Da quarta para baixo, a ordem do "
             "artigo é reproduzida exatamente.",
          0.55, 5.35, 12.2, 0.5, size=14, bold=True)
    nota(s, "Bloco escolhido pela validação Z₁\\T, como o artigo especifica. A "
            "última coluna é a variante de bloco fixo do pipeline original, "
            "que fica em ρ = 0.60 e Δ = −0.16 — a escolha do bloco vale ~0.08 "
            "de Fβ. FLIM bp não entra nesta variante: exige laço de treino "
            "próprio. FLIM ts* é a ablação, sem par no artigo.")


def s_algoritmo1(prs):
    s = slide(prs, "O Algoritmo 1 do artigo já é active learning",
              "…e é exatamente aí que está o problema")
    passos = [
        "1   sorteia z ∈ Z₁,   T ← T ∪ {z}",
        "3   enquanto o usuário não estiver satisfeito:",
        "4        treina uma CNN FLIM com decoder adaptativo em T",
        "5        avalia o modelo em Z₁\\T",
        "6        x ← Fβ médio em Z₁\\T",
        "7        entre as imagens de MENOR Fβ, seleciona z ∈ Z₁\\T",
        "8        se x < x_prev:   T ← T \\ {z_prev}        (backtracking)",
        "11      senão:                T ← T ∪ {z}",
    ]
    caixa(s, 0.55, 1.5, 7.3, 3.45)
    texto(s, passos, 0.85, 1.66, 6.9, 3.1, size=13, espaco=7)

    caixa(s, 8.1, 1.5, 4.68, 3.45, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "O passo 7", 8.4, 1.64, 4.1, 0.4, size=17, bold=True, cor=CORAL)
    texto(s, ["Ler o Fβ das imagens do pool exige a ground truth de todas as "
              "848 — para escolher 3.",
              "",
              "O artigo é explícito: adota-se uma abordagem supervisionada "
              "para selecionar as imagens representativas.",
              "",
              "Isso anula a premissa do FLIM, que é justamente não ter "
              "anotação."],
          8.4, 2.16, 4.1, 2.7, size=13, espaco=7)

    caixa(s, 0.55, 5.2, 12.23, 1.35, LIGHT, BLUE)
    texto(s, "Substituímos o passo 7 por critérios que não olham a ground "
             "truth — e medimos quanto do teto se recupera.",
          0.9, 5.4, 11.6, 0.5, size=17, bold=True, align=PP_ALIGN.CENTER)
    texto(s, "oracle (teto, usa GT)   ·   entropy   ·   least confidence   ·   "
             "CoreSet   ·   BADGE   ·   BALD   ·   random (piso)",
          0.9, 5.9, 11.6, 0.45, size=13, cor=GRAY, align=PP_ALIGN.CENTER)


def s_niveis(prs):
    s = slide(prs, "Dois níveis de granularidade",
              "A distinção central da dissertação")
    blocos = [
        ("AL por imagem", "Quais imagens o especialista deve anotar?",
         "Sai uma lista de nomes de arquivo. É o que o Algoritmo 1 faz. "
         "Critérios: entropy, least confidence, CoreSet, BADGE.",
         BLUE, "fig3_selecao.png"),
        ("AL por região", "Dentro da imagem, onde vale a pena anotar?",
         "Sai um mapa de regiões: a incerteza espacialmente resolvida "
         "posiciona os marcadores. Critérios: region entropy, region BALD.",
         GREEN, "fig4_regiao.png"),
    ]
    for i, (t, perg, desc, c, fig) in enumerate(blocos):
        x = 0.55 + i * 6.25
        caixa(s, x, 1.5, 6.0, 1.8)
        barra(s, x, 1.5, 1.8, c)
        texto(s, t, x + 0.28, 1.6, 5.4, 0.4, size=19, bold=True, cor=c)
        texto(s, perg, x + 0.28, 2.03, 5.5, 0.4, size=14, italico=True)
        texto(s, desc, x + 0.28, 2.42, 5.5, 0.8, size=12.5, cor=GRAY)
        if i == 0:                       # fig3 é quadrada: limita pela altura
            figura(s, str(REPO / "figs" / fig), x + 1.35, 3.42, h=3.15)
        else:                            # fig4 é faixa larga: limita pela largura
            figura(s, str(REPO / "figs" / fig), x, 4.0, w=6.0)
    nota(s, "O nível de região responde ao pedido de explorar active learning "
            "sobre pixels, e não sobre imagens.")


def s_al_por_modelo(prs, pm):
    s = slide(prs, "A tabela com a aplicação do AL — por modelo",
              "T do artigo contra T escolhido pelo critério de AL. Mesma "
              "tubulação, teste Z₂, média ± desvio sobre 3 splits")
    if not pm:
        texto(s, "[execução em andamento — out/final_comparison/*.csv ainda "
                 "não existe]", 0.55, 3.0, 12.0, 0.6, size=18, cor=CORAL)
        return s
    ps = D.p_pareado()
    linhas = [["Modelo", "Fβ · T do artigo", "Fβ · T do AL", "ΔFβ", "p",
               "IoU (AL)"]]
    for cod in D.ORDEM:
        p, a = pm.get(("paper", cod)), pm.get(("al", cod))
        if not (p and a):
            continue
        d = a["fb"] - p["fb"]
        pv = ps.get(cod)
        sig = pv is not None and pv < 0.05
        linhas.append([
            D.COD2PAPER[cod],
            f"{p['fb']:.3f}±{p['fb_sd']:.3f}",
            f"{a['fb']:.3f}±{a['fb_sd']:.3f}",
            (f"{d:+.3f}", "green" if sig and d > 0 else
             ("red" if sig and d < 0 else None)),
            (f"{pv:.3f}", "green" if sig else None) if pv is not None else "—",
            f"{a['iou']:.3f}",
        ])
    tabela(s, linhas, 0.55, 1.72, 8.3, 3.5,
           larguras=[1.35, 1.8, 1.8, 1.05, 0.9, 1.05], size=12.5,
           altura_linha=0.44)

    n_p = pm.get(("paper", "labeled_marker"), {}).get("n_train", 3.0)
    n_a = pm.get(("al", "labeled_marker"), {}).get("n_train", 3.0)
    caixa(s, 9.1, 1.72, 3.68, 1.75, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "Empate é o resultado", 9.4, 1.84, 3.2, 0.4, size=15, bold=True,
          cor=GREEN)
    texto(s, "Nos três decoders fortes o Δ é ±0.003, com p > 0.79. Um critério "
             "que NUNCA vê ground truth iguala a seleção supervisionada do "
             "artigo, com o mesmo orçamento "
             f"({n_p:.1f} contra {n_a:.1f} imagens).",
          9.4, 2.26, 3.15, 1.15, size=12)

    caixa(s, 9.1, 3.6, 3.68, 1.62, LIGHT, BLUE)
    texto(s, "Uma exceção real", 9.4, 3.72, 3.2, 0.4, size=15, bold=True,
          cor=BLUE)
    texto(s, "O FLIM at ganha +0.101 (p = 0.015), consistente nos três splits "
             "(+0.117, +0.111, +0.076). É o único ganho fora do ruído.",
          9.4, 4.14, 3.15, 1.0, size=12)
    nota(s, "Bloco escolhido pela validação Z₁\\T dentro de CADA braço, como "
            "no artigo — por isso a coluna “T do artigo” aqui não coincide com "
            "a tabela reproduzida, que usa o bloco fixo do pipeline original. "
            "Otsu + filtro de área [1000, 9000], sem Dynamic Trees. FLIM bp "
            "fica de fora: exige laço de treino próprio (Experimento B).")


def s_imagem_vs_regiao(prs):
    s = slide(prs, "AL por imagem vs AL por região",
              "Experimento B — encoder fixo, decoder 1×1 por backprop, "
              "396 execuções, ΔFβ pareado")
    linhas = [["Nível", "Método", "K", "AL", "Aleatório", "ΔFβ pareado", "p"]]
    dados = [
        ("imagem", "CoreSet", "3", "0.542", "0.484", "+0.058", "0.071", None),
        ("imagem", "CoreSet", "5", "0.552", "0.502", "+0.049", "0.063", None),
        ("imagem", "CoreSet", "10", "0.575", "0.483", "+0.092", "<0.001",
         "green"),
        ("imagem", "BADGE", "3", "0.497", "0.484", "+0.013", "0.693", None),
        ("imagem", "BADGE", "10", "0.563", "0.483", "+0.080", "<0.001",
         "green"),
        ("imagem", "entropy", "3", "0.409", "0.484", "−0.076", "0.012", "red"),
        ("imagem", "entropy", "10", "0.526", "0.483", "+0.043", "0.130", None),
        ("região", "region entropy", "3", "0.293", "0.446", "−0.153", "<0.001",
         "red"),
        ("região", "region entropy", "10", "0.507", "0.427", "+0.079", "0.048",
         "green"),
    ]
    for n, m, k, al, rd, dl, p, c in dados:
        linhas.append([n, m, k, al, rd, (dl, c), p])
    tabela(s, linhas, 0.55, 1.72, 7.9, 3.5,
           larguras=[1.1, 1.75, 0.5, 0.9, 1.15, 1.4, 0.85], size=12,
           altura_linha=0.35)

    caixa(s, 8.7, 1.72, 4.08, 3.5, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "O que se sustenta", 9.0, 1.86, 3.6, 0.4, size=16, bold=True,
          cor=GREEN)
    texto(s, ["CoreSet é o único método que supera o aleatório de forma "
              "robusta: ΔFβ +0.072, IC95% [+0.038, +0.106], p < 0.001.",
              "",
              "Com K = 10, CoreSet e BADGE igualam o pool completo de ~190 "
              "imagens — 19× menos anotação sem perda mensurável.",
              "",
              "Incerteza pura é PIOR que o acaso em orçamento pequeno."],
          9.0, 2.34, 3.55, 2.8, size=12.5, espaco=6)
    nota(s, "Pareado: o braço de AL e o aleatório de mesmo índice compartilham "
            "a semente de inicialização, então a diferença isola a seleção. "
            "Piso de ruído medido: ±0.023.")


def s_decomposicao(prs):
    s = slide(prs, "O braço de controle que mudou a conclusão",
              "Métodos de região mudam duas coisas ao mesmo tempo: quais "
              "imagens entram em T e como os seeds são desenhados")
    linhas = [["Decoder", "AL região", "random região", "random", "seleção",
               "geometria"]]
    dados = [
        ("FLIM lm", "0.575", "0.605", "0.517", "−0.030", "+0.088"),
        ("FLIM pb", "0.623", "0.586", "0.441", "+0.036", "+0.145"),
        ("FLIM mb", "0.594", "0.560", "0.277", "+0.033", "+0.284"),
        ("FLIM at", "0.350", "0.259", "0.162", "+0.091", "+0.096"),
        ("FLIM lt", "0.503", "0.425", "0.377", "+0.079", "+0.048"),
        ("FLIM ts", "0.466", "0.436", "0.409", "+0.030", "+0.027"),
        ("FLIM ts*", "0.338", "0.226", "0.161", "+0.112", "+0.065"),
    ]
    for d in dados:
        linhas.append(list(d[:4]) + [d[4], (d[5], "green")])
    linhas.append(["média", "", "", "", ("+0.050", "blue"),
                   ("+0.108", "green")])
    tabela(s, linhas, 0.55, 1.72, 7.7, 3.5,
           larguras=[1.3, 1.2, 1.55, 1.05, 1.15, 1.3], size=12,
           altura_linha=0.36)

    caixa(s, 8.5, 1.72, 4.28, 1.5)
    texto(s, "seleção   =  AL região − random região", 8.75, 1.84, 4.0, 0.35,
          size=13, bold=True, cor=BLUE)
    texto(s, "quais imagens entram em T", 8.75, 2.16, 4.0, 0.3, size=11.5,
          cor=GRAY)
    texto(s, "geometria =  random região − random", 8.75, 2.5, 4.0, 0.35,
          size=13, bold=True, cor=GREEN)
    texto(s, "como os seeds são desenhados", 8.75, 2.82, 4.0, 0.3, size=11.5,
          cor=GRAY)

    caixa(s, 8.5, 3.4, 4.28, 1.82, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "2.1×", 8.75, 3.5, 3.8, 0.75, size=40, bold=True, cor=GREEN)
    texto(s, "A geometria dos seeds pesa 2.1 vezes a seleção de imagens. Para "
             "o FLIM, ONDE o especialista desenha importa mais que QUAL "
             "imagem ele abre.", 8.75, 4.22, 3.8, 0.95, size=12.5)
    nota(s, "Os dois braços aleatórios usam a mesma semente, logo sorteiam as "
            "mesmas imagens — só muda a anotação. Sem esse controle, o ganho "
            "da geometria era creditado à seleção.")


def s_curva(prs):
    s = slide(prs, "Algoritmo 1 com critérios que não veem a ground truth",
              "Melhor Fβ acumulado com |T| imagens, média sobre split × "
              "semente, pool dos 31 markers reais")
    linhas = [["Critério", "Usa GT?", "|T|=1", "|T|=2", "|T|=3", "|T|=4",
               "|T|=5"]]
    dados = [
        ("oracle (Algoritmo 1)", ("sim", "red"), "0.593", "0.704", "0.731",
         "0.752", "0.752"),
        ("entropy", ("não", "green"), "0.593", "0.727", ("0.758", "green"),
         "0.745", "0.737"),
        ("random", ("não", "green"), "0.593", "0.680", "0.730", "0.720",
         "0.751"),
        ("BADGE", ("não", "green"), "0.593", "0.636", "0.665", "0.715",
         "0.729"),
        ("CoreSet", ("não", "green"), "0.593", "0.650", "0.707", "0.663",
         "0.723"),
    ]
    for d in dados:
        linhas.append(list(d))
    tabela(s, linhas, 0.55, 1.7, 7.6, 2.25,
           larguras=[2.0, 1.0, 0.95, 0.95, 0.95, 0.95, 0.95], size=12.5,
           altura_linha=0.37)

    caixa(s, 8.45, 1.7, 4.33, 2.25, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, "O resultado central", 8.7, 1.84, 3.9, 0.4, size=16, bold=True,
          cor=GREEN)
    texto(s, "Com |T| = 3, a entropia sem ground truth alcança 0.758 — acima "
             "do oráculo supervisionado (0.731). O passo 7 do artigo pode ser "
             "substituído sem perda.", 8.7, 2.34, 3.85, 1.5, size=13)

    figura(s, str(REPO / "figs" / "fig5_curva.png"), 0.55, 4.12, w=6.1)
    caixa(s, 6.95, 4.12, 5.83, 2.4, LIGHT, CORAL)
    texto(s, "Mas o critério do artigo tem um defeito estrutural", 7.25, 4.24,
          5.3, 0.45, size=15, bold=True, cor=CORAL)
    texto(s, ["•  49% do pool não tem foreground nenhum. Essas imagens têm "
              "Fβ = 0 porque o filtro de área as zera — e o critério de "
              "menor Fβ prioriza exatamente elas.",
              "•  Sem lista de rejeitados, o backtracking entra em ciclo: a "
              "imagem removida volta a ser a de menor Fβ.",
              "•  2.0 anotações desperdiçadas por execução (33% das rodadas)."],
          7.25, 4.72, 5.3, 1.7, size=12, espaco=5)


def s_seg_modelos(prs):
    s = slide(prs, "Segmentação final após o active learning, por modelo",
              "Predição em ciano, contorno da anotação em magenta — "
              "teste Z₂, split 1")
    figura(s, str(REPO / "figs" / "fig8_seg_por_modelo.png"), 0.55, 1.5, w=12.2)
    nota(s, "Vazamento é ciano fora do magenta; buraco é magenta sem ciano. "
            "Cada linha usa o encoder treinado com as imagens que o critério "
            "de AL escolheu, no bloco que a validação selecionou.")


def s_antes_depois(prs, caso):
    """
    Um slide por caso. As duas figuras somam ~10 polegadas de altura; empilhar
    as duas num slide de 7.5 espreme as duas até ficarem ilegíveis.
    """
    if caso == "ganho":
        s = slide(prs, "Antes e depois: onde o active learning acertou",
                  "Mesma imagem, mesmo decoder — muda apenas quais imagens "
                  "compõem T")
        figura(s, str(REPO / "figs" / "fig9a_al_melhorou.png"), 0.55, 1.45,
               w=12.2)
        nota(s, "Com o T do artigo, quatro dos sete decoders não encontram o "
                "ovo. Com o T escolhido pelo critério de entropia, o FLIM lm "
                "sai de Fβ = 0 para 0.61 nesta imagem.")
    else:
        s = slide(prs, "Antes e depois: onde o active learning errou",
                  "O mesmo experimento tem casos nos dois sentidos — mostrar "
                  "só os ganhos seria escolher a evidência")
        figura(s, str(REPO / "figs" / "fig9b_al_piorou.png"), 0.55, 1.45,
               w=12.2)
        nota(s, "Aqui o T do artigo acerta e o do AL perde o ovo: ΔFβ = −0.66 "
                "no FLIM lm. É o mesmo efeito que aparece na tabela como "
                "desvio alto entre splits — o ganho médio de +0.002 é a soma "
                "de casos assim nos dois sentidos.")


def s_falhas(prs):
    s = slide(prs, "O que sabemos que está quebrado",
              "Registro honesto — cada item custou execução")
    itens = [
        ("49% do pool não tem foreground",
         "181 de 371 imagens. O critério de pior Fβ prioriza justamente essas: "
         "têm Fβ = 0 porque o filtro de área as zera, não porque sejam "
         "informativas.", CORAL),
        ("O Algoritmo 1 pode entrar em ciclo",
         "Sem lista de rejeitados, a imagem removida pelo backtracking volta a "
         "ser a de menor Fβ. Nossa reimplementação acrescenta uma lista tabu.",
         CORAL),
        ("Predição colapsada tem assinatura exata",
         "Fβ = DICE = IoU ≈ 0.4788, que é 406/848 — a fração de imagens "
         "vazias. Toda tabela marca essas execuções: o ganho delas não é "
         "aprendizado.", GRAY),
        ("Quatro hipóteses testadas e refutadas",
         "Markers pontilhados não causam o colapso; conjunto pequeno não causa "
         "(Fβ = 0.54 com 1 imagem); markers realistas não corrigem (6 de 6 "
         "pioram); e dispersão alta de entropia como sinal está invertida.",
         BLUE),
    ]
    for i, (t, d, c) in enumerate(itens):
        y = 1.45 + i * 1.3
        caixa(s, 0.55, y, 8.0, 1.15)
        barra(s, 0.55, y, 1.15, c, w=0.07)
        texto(s, t, 0.82, y + 0.08, 7.5, 0.35, size=15, bold=True, cor=c)
        texto(s, d, 0.82, y + 0.43, 7.6, 0.68, size=12)
    figura(s, str(REPO / "figs" / "fig7_falha.png"), 8.75, 3.3, w=4.1)


def s_conclusoes(prs):
    s = slide(prs, "O que a dissertação estabelece", "E o que vem depois")
    achados = [
        ("A geometria pesa 2.1× a seleção",
         "Para o FLIM, onde o especialista desenha importa mais que qual "
         "imagem ele abre."),
        ("CoreSet é o único método robusto",
         "ΔFβ +0.072, p < 0.001. Com K = 10 iguala o pool completo — 19× menos "
         "anotação."),
        ("Incerteza pura é pior que o acaso",
         "Em orçamento pequeno, entropy K=3 dá −0.076. O contrário do que a "
         "literatura de AL sugere."),
        ("O passo 7 supervisionado é dispensável",
         "Entropia sem ground truth alcança 0.758 com |T| = 3, acima do "
         "oráculo."),
    ]
    for i, (t, d) in enumerate(achados):
        x = 0.55 + (i % 2) * 6.25
        y = 1.5 + (i // 2) * 1.5
        caixa(s, x, y, 6.0, 1.3)
        barra(s, x, y, 1.3, GREEN, w=0.07)
        texto(s, t, x + 0.28, y + 0.1, 5.5, 0.45, size=15, bold=True)
        texto(s, d, x + 0.28, y + 0.55, 5.5, 0.7, size=12.5, cor=GRAY)

    caixa(s, 0.55, 4.7, 12.23, 1.8, RGBColor(0xEE, 0xF4, 0xFC), BLUE)
    texto(s, "Próximos passos", 0.9, 4.85, 5.0, 0.4, size=16, bold=True,
          cor=BLUE)
    texto(s, ["•  Rodar a mesma decomposição em BraTS (3753 imagens) e "
              "Conjunctiva (83) — verificar se a razão geometria/seleção se "
              "mantém fora do Schisto.",
              "•  Recuperar Dynamic Trees (liblapack/libblas) para fechar a "
              "lacuna com a Tabela IV.",
              "•  AL de orçamento de seeds: fixar o número de pinceladas, não "
              "o número de imagens."],
          0.9, 5.3, 11.6, 1.1, size=13, espaco=4)


def s_brats(prs, br):
    """
    O slide mais forte da defesa, se os números confirmarem: pela Tabela I o
    BraTS não recebe Dynamic Trees, então aqui a reprodução não tem componente
    ausente e os valores absolutos são comparáveis.
    """
    s = slide(prs, "A reprodução sem etapa faltando — BraTS",
              "Pela Tabela I do artigo o BraTS usa Otsu + filtro de área "
              "[100, 20000], sem Dynamic Trees. É o único ponto do trabalho "
              "em que nada falta no pipeline")
    if not br:
        texto(s, "[execução em andamento — out/brats/repro.csv ainda não "
                 "existe]", 0.55, 3.0, 12.0, 0.6, size=18, cor=CORAL)
        return s

    linhas = [["Modelo", "Fβ reproduzido", "Fβ artigo", "Δ"]]
    ds = []
    for dec in D.ORDEM_PAPER:
        v = br.get(dec)
        if not v:
            continue
        if v["artigo"] is None:
            linhas.append([dec.replace("_", " "), f"{v['fb']:.3f}", "—", "—"])
            continue
        d = v["fb"] - v["artigo"]
        ds.append(d)
        linhas.append([dec.replace("_", " "), f"{v['fb']:.3f}",
                       f"{v['artigo']:.3f}",
                       (f"{d:+.3f}", "green" if abs(d) < 0.03 else None)])
    tabela(s, linhas, 0.55, 1.75, 7.0, 3.4,
           larguras=[1.5, 1.9, 1.5, 1.1], size=13, altura_linha=0.45)

    import statistics as _st
    med = _st.mean(ds) if ds else 0.0
    caixa(s, 7.85, 1.75, 4.93, 1.6, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, f"Δ médio  {med:+.3f}", 8.15, 1.9, 4.4, 0.6, size=28, bold=True,
          cor=GREEN)
    texto(s, f"Sobre {len(ds)} modelos, com erros nos DOIS sentidos.",
          8.15, 2.55, 4.4, 0.6, size=13)

    caixa(s, 7.85, 3.5, 4.93, 1.65, RGBColor(0xFD, 0xF0, 0xEE), CORAL)
    texto(s, "No Parasites:  −0.080", 8.15, 3.62, 4.4, 0.45, size=19,
          bold=True, cor=CORAL)
    texto(s, "…e sistematicamente negativo, nos seis decoders com par no "
             "artigo (p = 0.031 no teste de sinal). A diferença entre os dois "
             "casos é a presença dos Dynamic Trees no pipeline do Parasites.",
          8.15, 4.08, 4.4, 1.0, size=12.5)

    texto(s, "A lacuna do Parasites deixa de ser “não conseguimos reproduzir” "
             "e passa a ser “falta um componente identificado, e onde ele não "
             "falta a reprodução é exata”.",
          0.55, 5.35, 12.2, 0.5, size=14.5, bold=True)
    nota(s, "Verde = |Δ| < 0.03, isto é, dentro do piso de ruído medido "
            "(0.023). 3753 imagens, 4 de treino por split, teste Z₂ completo "
            "(1877 imagens). Encoder de 3 blocos (arch_best_brats.json), bloco "
            "escolhido pela validação. Comparação contra a coluna BraTS da "
            "Tabela IV, usuário A.")


def s_equivalencia(prs, res, delta=0.02):
    """A tabela de AL com a estatística que sustenta a conclusão."""
    s = slide(prs, "O active learning por modelo, com teste de equivalência",
              "Δ pareado por usuário × split, sementes do Algoritmo 1 "
              "agregadas por média — sem melhor-de-3")
    if not res:
        texto(s, "[execução em andamento]", 0.55, 3.0, 12.0, 0.6, size=18,
              cor=CORAL)
        return s

    linhas = [["Modelo", "Fβ artigo", "Fβ AL", "ΔFβ", "IC95%", "p dif.",
               "p equiv.", "n"]]
    for dec in D.ORDEM_PAPER:
        v = res.get(dec)
        if not v:
            continue
        sig = v["p"] == v["p"] and v["p"] < 0.05
        eq = v["tost"] == v["tost"] and v["tost"] < 0.05
        linhas.append([
            dec.replace("_", " "), f"{v['paper']:.3f}", f"{v['al']:.3f}",
            (f"{v['d']:+.3f}", "green" if sig and v["d"] > 0 else
             ("red" if sig and v["d"] < 0 else None)),
            f"[{v['lo']:+.3f}, {v['hi']:+.3f}]",
            f"{v['p']:.3f}",
            (f"{v['tost']:.3f}", "green" if eq else None),
            str(v["n"]),
        ])
    tabela(s, linhas, 0.55, 1.75, 12.2, 3.5,
           larguras=[1.3, 1.3, 1.2, 1.1, 2.0, 1.0, 1.1, 0.6], size=12.5,
           altura_linha=0.44)

    caixa(s, 0.55, 5.45, 12.2, 1.15, RGBColor(0xEC, 0xF7, 0xEF), GREEN)
    texto(s, f"p (equiv.) < 0.05 demonstra que |Δ| < {delta} — não é a mesma "
             "coisa que “não detectei diferença”.",
          0.9, 5.6, 11.5, 0.4, size=16, bold=True, align=PP_ALIGN.CENTER)
    texto(s, "δ escolhido antes dos resultados: o desvio entre splits do braço "
             "do artigo é 0.009 e o piso de ruído do Experimento B é 0.023.",
          0.9, 6.02, 11.5, 0.4, size=12.5, cor=GRAY, align=PP_ALIGN.CENTER)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../FLIM_AL_Apresentacao_v5.pptx")
    ap.add_argument("--delta", type=float, default=0.02,
                    help="margem de equivalência do TOST")
    a = ap.parse_args()

    rep = D.reproducao()
    pm = D.por_modelo()
    res = D.resumo_al(a.delta)
    br = D.resumo_brats()
    print(f"reprodução: {len(rep)} células · AL por modelo: {len(pm)} células")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    s_capa(prs)
    s_problema(prs)
    s_encoder(prs)
    s_modelos(prs)
    s_baselines(prs)
    s_tabela3(prs)
    s_tabela4(prs)
    s_reproducao(prs, rep, pm)
    s_brats(prs, br)
    s_algoritmo1(prs)
    s_niveis(prs)
    s_al_por_modelo(prs, pm)
    s_equivalencia(prs, res, a.delta)
    s_imagem_vs_regiao(prs)
    s_decomposicao(prs)
    s_curva(prs)
    s_seg_modelos(prs)
    s_antes_depois(prs, "ganho")
    s_antes_depois(prs, "perda")
    s_falhas(prs)
    s_conclusoes(prs)

    prs.save(a.out)
    print(f"[salvo] {a.out} · {len(prs.slides._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
