#!/usr/bin/env python3
"""
server.py — FLIM + Active Learning ao vivo
==========================================
O laço inteiro da dissertação numa página: você desenha os markers, o encoder
FLIM é estimado na hora, a segmentação aparece, e o critério de aquisição diz
qual imagem anotar em seguida.

Isso só é viável porque o FLIM não usa backpropagation no encoder: estimar os
kernels é k-means sobre os patches dos markers, ~5 s. Um método que precisasse
de épocas de gradiente não caberia numa interface interativa — a demonstração
é, ela própria, um argumento sobre o método.

Sem dependência externa: usa `http.server` da biblioteca padrão. Numa aplicação
local de um usuário só, um framework acrescentaria instalação sem acrescentar
nada.

Rodar (dentro do container flim-app, que tem a porta 8000 publicada):
    docker exec -d flim-app bash -lc \\
      "cd /workspace/flim-python-demo/flim_ad && python ../flim_app/server.py"

Depois abrir http://localhost:8000
"""
from __future__ import annotations

import base64
import io as _io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    _official_filter_and_binarize, evaluate_decoder, load_input,
    retrain_encoder,
)
from flim_al.coreset_badge import (  # noqa: E402
    badge_select, coreset_select, extract_encoder_features,
)
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_al.paper_selection import imagem_medoide  # noqa: E402
from flim_app import datasets as DS  # noqa: E402
from flim_app import duplo as DUP  # noqa: E402
from flim_app import runner as RUN  # noqa: E402
from pyflim import layers  # noqa: E402

# Os caminhos vêm do dataset selecionado, não são mais fixos.
def _cfg() -> dict:
    return DS.resolver(S["ds"])


def _orig() -> str:
    return _cfg()["orig"]


def _label() -> str:
    return _cfg()["label"]


def _area() -> tuple:
    return tuple(_cfg()["area"])

DECODERS = [
    ("labeled_marker", "FLIM lm"), ("decoder_2", "FLIM pb"),
    ("decoder_3", "FLIM mb"), ("hybrid_decoder", "FLIM lt"),
    ("vanilla_adaptive_decoder", "FLIM ts"), ("decoder_attention", "FLIM at"),
]

# O tamanho da amostra é uma escolha de latência, não de rigor: 40 imagens dão
# um Fβ com erro padrão de ~0.05, suficiente para ver a curva subir ao vivo,
# e levam ~20 s em vez dos ~3 min do conjunto inteiro. Os números da
# dissertação vêm dos scripts em lote, não daqui.
# Identidade desta instancia do servidor. Como todo o estado da sessao vive em
# memoria, um reinicio descarta encoder, markers e a comparacao em andamento.
# O cliente compara este valor a cada consulta: se mudou, ele sabe que a sessao
# se perdeu e avisa, em vez de deixar a pessoa olhando um botao travado.
BOOT = f"{int(time.time())}-{os.getpid()}"

N_VAL = 24
N_POOL = 24
# quantas candidatas pontuar antes de estratificar a fila de validação.
# Cada pontuação é um forward; 120 leva poucos segundos em CPU e já dá
# cauda suficiente nos dois extremos para a fila não repetir vizinhos.
MAX_PONTUAR_VALID = 120

LOCK = threading.Lock()
S: dict = {
    "markers": {},      # img_id -> {"fg": [[col,row]], "bg": [[col,row]]}
    "encoder": None,
    "decoder": "labeled_marker",
    "block": 2,
    "history": [],
    "val": [],
    "pool": [],
    "work": tempfile.mkdtemp(prefix="flimapp_"),
    "busy": False,
    "alvo": None,        # imagem que o sistema mandou anotar nesta rodada
    "k_alvo": 3,         # quantas imagens o laço vai pedir
    "ds": "schisto",     # dataset corrente
    "criterio": "coreset",
}

# Execução em lote (critério × K). Roda em thread para a interface não
# congelar; `cancelar` permite abortar sem matar o servidor.
LOTE: dict = {"rodando": False, "feito": 0, "total": 0, "resultados": [],
              "erro": "", "cancelar": False, "n_test": 0, "n_pool": 0,
              "config": {}}

# Validação pelo especialista: o modelo prevê, a pessoa julga, sai a matriz.
VALID: dict = {"fila": [], "i": 0, "respostas": [], "pred": {}}

# Comparação de N braços com o mesmo orçamento: o FLIM puro (sorteio) contra um
# ou mais critérios de active learning. Os markers são globais por imagem — ver
# a seção "comparação de N braços" mais abaixo para o porquê.
D: dict = {"ativo": False, "k": 3, "rng": None, "semente": None,
           "markers": {}, "centros": {}, "confirmadas": set(), "bracos": {},
           "ordem": [], "fila": [], "fila_i": 0, "rodada": 0, "sugestor": None,
           "resultado": None}


# ── utilidades ───────────────────────────────────────────────────────────────

def _png(arr: np.ndarray) -> str:
    buf = _io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _listar() -> list[str]:
    """Nomes sem extensão, do dataset corrente."""
    return [os.path.splitext(f)[0] for f in DS.imagens(S["ds"])]


def _tem_objeto(img_id: str) -> bool:
    lp = DS.caminho_label(S["ds"], img_id + ".png") or         DS.caminho_label(S["ds"], img_id + ".jpg")
    if not lp:
        return True                      # sem anotação, não dá para filtrar
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


def _arquivo(img_id: str) -> str:
    """Nome com a extensão real (o dataset pode ser .jpg)."""
    for ext in (".png", ".jpg", ".jpeg"):
        if os.path.exists(os.path.join(_orig(), img_id + ext)):
            return img_id + ext
    return img_id + ".png"


def _init_conjuntos(forcar: bool = False) -> None:
    """
    Sorteia val e pool do dataset corrente, disjuntos, com semente fixa.

    Quando há anotação, ambos preferem imagens COM objeto. Não é conveniência:
    no Schisto 49% do pool não tem ovo, e nessas o Fβ é 1.0 por definição
    (vazio-vazio) — a métrica ficaria dominada por casos triviais e a curva
    não se moveria. Está declarado na interface.
    """
    if S["val"] and not forcar:
        return
    todas = _listar()

    # Validacao definida pelo especialista (upload em dois lotes): o que ele
    # marcou como validacao NAO entra no pool de anotacao, e o pool e todo o
    # resto. E o cenario "treino com estas, testo naquelas".
    fixa = [i for i in DS.validacao_fixa(S["ds"]) if i in set(todas)]
    if fixa:
        S["val"] = sorted(fixa)
        S["pool"] = sorted(i for i in todas if i not in set(fixa))[:N_POOL]
        S["markers"].clear()
        S["encoder"] = None
        S["history"].clear()
        S["alvo"] = None
        return

    rng = np.random.default_rng(0)
    tem_label = bool(_label())
    cand = [i for i in todas if not tem_label or _tem_objeto(i)] or todas
    cand = cand[:600]
    if len(cand) < 4:
        S["val"], S["pool"] = [], list(cand)
        return
    emb = rng.permutation(len(cand))
    S["val"] = sorted(cand[i] for i in emb[:min(N_VAL, len(cand) // 3)])
    resto = [cand[i] for i in emb[len(S["val"]):]]
    S["pool"] = sorted(resto[:N_POOL])
    S["markers"].clear()
    S["encoder"] = None
    S["history"].clear()
    S["alvo"] = None


def _marker_dir() -> str:
    """Materializa os markers desenhados no formato seeds.txt do FLIM."""
    d = os.path.join(S["work"], "markers")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for img_id, m in S["markers"].items():
        if not m["fg"] and not m["bg"]:
            continue
        with Image.open(os.path.join(_orig(), _arquivo(img_id))) as im:
            w, h = im.size
        # Mesmo recorte de duplo.escrever_markers: seed fora do quadro derruba
        # o treino com um erro que nao aponta para a causa.
        fg = [[c, r] for c, r in m["fg"] if 0 <= c < w and 0 <= r < h]
        bg = [[c, r] for c, r in m["bg"] if 0 <= c < w and 0 <= r < h]
        if not fg and not bg:
            continue
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": h,
                      "W": w}, os.path.join(d, f"{img_id}-seeds.txt"))
    return d


@torch.no_grad()
def _prever(img_id: str) -> dict:
    model = torch.load(S["encoder"], map_location="cpu", weights_only=False)
    model.device = "cpu"
    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            model.layers[l].marker_labels = ml.to("cpu")
    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type=S["decoder"],
        multi_layer=False)

    x, h, w = load_input(os.path.join(_orig(), _arquivo(img_id)))
    y_hat, _ = model.forward(x, decoder_layer=[S["block"] - 1])
    pred = y_hat[0].float()
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    sal = pred.squeeze().numpy().astype(np.uint8)
    if sal.shape[0] != h or sal.shape[1] != w:
        sal = np.array(Image.fromarray(sal).resize((w, h), Image.BILINEAR),
                       dtype=np.uint8)
    binm = _official_filter_and_binarize(sal, _area())

    # máscara RGBA: ciano onde prediz, transparente no resto
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[binm.astype(bool)] = (26, 217, 242, 140)

    fb = None
    lp = os.path.join(_label(), f"{img_id}.png")
    if os.path.exists(lp):
        gt = (np.array(Image.open(lp).convert("L")) > 0).astype(np.uint8)
        fb = _fb(binm, gt)
    return {"mask": _png(rgba), "saliency": _png(sal), "fb": fb,
            "area": int(binm.sum())}


def _fb(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    eps = 1e-8
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt.astype(bool)).sum())
    fn = float((~pred.astype(bool) & gt).sum())
    pr, rc = tp / (tp + fp + eps), tp / (tp + fn + eps)
    return float((1 + beta2) * pr * rc / (beta2 * pr + rc + eps))


@torch.no_grad()
def _montar_modelo(enc_path: str):
    """Encoder salvo + decoder adaptativo, prontos para prever em CPU."""
    model = torch.load(enc_path, map_location="cpu", weights_only=False)
    model.device = "cpu"
    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            model.layers[l].marker_labels = ml.to("cpu")
    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type=S["decoder"],
        multi_layer=False)
    return model


@torch.no_grad()
def _mapa_prob(model, img_id: str) -> np.ndarray:
    """Mapa de probabilidade [0,1] de uma imagem, no bloco corrente."""
    x, _, _ = load_input(os.path.join(_orig(), _arquivo(img_id)))
    y_hat, _ = model.forward(x, decoder_layer=[S["block"] - 1])
    pred = y_hat[0].float()
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    return np.clip(pred.squeeze().numpy() / 255.0, 1e-6, 1 - 1e-6)


def _pontuar_pool(ids: list[str], enc_path: str) -> list[dict]:
    """
    Entropia, least confidence e média da sigmoide por imagem, numa passada só.

    As três saem do MESMO mapa de probabilidade; calcular em passadas separadas
    custaria três forwards por imagem sem mudar nenhum número. A média serve de
    proxy de incerteza para o BADGE, que precisa dela para montar o gradient
    embedding.
    """
    model = _montar_modelo(enc_path)
    out = []
    for img_id in ids:
        p = _mapa_prob(model, img_id)
        ent = float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())
        lc = float((1 - np.abs(p - 0.5) * 2).mean())
        out.append({"id": img_id, "entropia": round(ent, 5),
                    "least_confidence": round(lc, 5), "media": float(p.mean())})
    return out


# ── escolha da proxima imagem: os criterios sem ground truth ─────────────────

CRITERIOS = [
    ("coreset", "CoreSet — cobertura",
     "escolhe a imagem mais DISTANTE de tudo que já foi anotado, no espaço de "
     "features do encoder. Foi o único que se sustentou no teste por decoder"),
    ("entropy", "Entropia — incerteza",
     "escolhe onde o modelo está mais indeciso (probabilidade perto de 0.5). "
     "Recupera ~15% da vantagem da supervisão; com pool de 49% de imagens "
     "vazias, persegue casos que não ensinam nada"),
    ("least_confidence", "Least confidence",
     "variante da incerteza: média de 1 − |p − 0.5| × 2. Ordena parecido com a "
     "entropia, com cauda diferente"),
    ("badge", "BADGE — incerteza + diversidade",
     "k-means++ sobre gradient embeddings (Ash et al., ICLR 2020): combina os "
     "dois sinais. Ficou entre o CoreSet e a entropia"),
    ("random", "Sorteio — o piso",
     "sem critério nenhum. É o controle correto, e é o que o artigo faz na "
     "primeira rodada"),
]
CRITERIO_IDS = [c[0] for c in CRITERIOS]


def _escolher_alvo(criterio: str, anotadas: list[str], restantes: list[str],
                   enc_path: str | None,
                   rng: "np.random.Generator | None" = None
                   ) -> tuple[str, str]:
    """
    A proxima imagem a anotar, e a frase que explica a escolha.

    NENHUM dos caminhos abaixo le a anotacao de referencia. Essa e a propriedade
    que separa esta funcao do passo 7 do Algoritmo 1 do artigo, que calcula o
    Fbeta das 848 imagens do pool -- e para isso precisa do gabarito de todas.
    Aqui o que se olha e: os pixels (medoide), a propria saida do modelo
    (entropia, least confidence) ou a geometria das features (CoreSet, BADGE).
    """
    if not restantes:
        return "", "pool esgotado"

    # Sem encoder ainda nao existe modelo a consultar. O artigo sorteia aqui, e
    # e dai que vem a variancia dominante do metodo; o medoide e deterministico.
    if enc_path is None or not anotadas:
        if criterio == "random":
            g = rng if rng is not None else np.random.default_rng()
            return (str(g.choice(restantes)),
                    "sorteada — é o que se faz sem critério de seleção")
        return (os.path.splitext(imagem_medoide(
                    [_arquivo(i) for i in restantes], _orig()))[0],
                "imagem mais típica do pool, medida direto nos pixels — "
                "ainda não há modelo para consultar")

    if criterio == "random":
        g = rng if rng is not None else np.random.default_rng()
        return (str(g.choice(restantes)),
                "sorteada — e o que se faz sem criterio de selecao")

    if criterio in ("entropy", "least_confidence"):
        # _pontuar_pool devolve as chaves em portugues; o id do criterio vem da
        # nomenclatura da literatura. O de-para fica aqui, num lugar so.
        chave = {"entropy": "entropia",
                 "least_confidence": "least_confidence"}[criterio]
        nome = {"entropy": "entropia",
                "least_confidence": "least confidence"}[criterio]
        pontos = _pontuar_pool(restantes, enc_path)
        pontos.sort(key=lambda r: -r[chave])
        return (pontos[0]["id"],
                f"onde o modelo está mais indeciso — {nome} = "
                f"{pontos[0][chave]:.5f}, a maior das {len(restantes)} "
                "imagens ainda não anotadas")

    enc = torch.load(enc_path, map_location="cpu", weights_only=False)
    todas = anotadas + restantes
    feats = extract_encoder_features(
        enc, [os.path.join(_orig(), _arquivo(i)) for i in todas],
        S["block"], "cpu")

    if criterio == "badge":
        # O BADGE seleciona um lote de uma vez; aqui o lote e 1 por rodada.
        # Ele nao recebe labeled_indices, entao as ja anotadas saem do conjunto
        # candidato antes -- senao poderia devolver uma imagem ja anotada.
        pontos = _pontuar_pool(restantes, enc_path)
        media = np.array([r["media"] for r in pontos], dtype=np.float32)
        escolha = badge_select(feats[len(anotadas):], media, budget=1)
        return (restantes[escolha[0]],
                "incerteza combinada com diversidade — k-means++ sobre os "
                "gradient embeddings (BADGE)")

    escolha = coreset_select(feats, budget=1,
                             labeled_indices=list(range(len(anotadas))))
    return (todas[escolha[0]],
            f"a mais distante das {len(anotadas)} já anotadas, no espaço de "
            "features do encoder (CoreSet)")


def _entropia_pool() -> list[dict]:
    """
    Entropia binária média do mapa de saliência, por imagem do pool.

    É o critério sem ground truth: mede o quanto o modelo está indeciso, e não
    o quanto ele está errado. Nenhuma anotação do pool é consultada.
    """
    anotadas = set(S["markers"])
    ids = [i for i in S["pool"] if i not in anotadas]
    out = _pontuar_pool(ids, S["encoder"])
    out.sort(key=lambda r: -r["entropia"])
    return out


# ── ações ────────────────────────────────────────────────────────────────────

def acao_treinar() -> dict:
    _init_conjuntos()
    mdir = _marker_dir()
    n = len([f for f in os.listdir(mdir) if f.endswith("-seeds.txt")])
    if n == 0:
        return {"erro": "nenhum marker desenhado ainda"}

    # Treinar e avaliar são endpoints separados de propósito. Estimar o
    # encoder leva ~2.5 s; medir Fβ em 24 imagens leva ~25 s. Juntos, o laço
    # interativo herdaria a latência da avaliação, que é a parte opcional.
    t0 = time.time()
    enc = os.path.join(S["work"], f"enc_{n}_{int(time.time())}.pth")
    retrain_encoder(_cfg()["arch"], mdir, _orig(), _label(), "cpu", enc)
    S["encoder"] = enc
    return {"n_imgs": n, "t_treino": round(time.time() - t0, 1),
            "history": S["history"]}


def acao_avaliar() -> dict:
    if not S["encoder"]:
        return {"erro": "treine primeiro"}
    t1 = time.time()
    m = evaluate_decoder(S["encoder"], S["decoder"], S["block"],
                         [_arquivo(i) for i in S["val"]], _orig(), _label(), "cpu",
                         area_range=_area())
    n = len(S["markers"])
    S["history"].append({
        "n_imgs": n, "fb": round(m["fb"], 4), "iou": round(m["iou"], 4),
        "T": sorted(S["markers"]), "decoder": S["decoder"],
    })
    return {"fb": round(m["fb"], 4), "iou": round(m["iou"], 4),
            "dice": round(m["dice"], 4), "n_imgs": n,
            "t_aval": round(time.time() - t1, 1), "history": S["history"]}


def acao_proxima() -> dict:
    """
    O sistema escolhe a próxima imagem — este é o laço de active learning.

    Rodada 1: não existe encoder ainda, então a escolha não pode depender de
    modelo. Usa-se o MEDOIDE do pool: a imagem mais típica, medida direto nos
    pixels (32x32 em LAB). Determinístico e sem ground truth. Medimos que a
    inicialização aleatória do artigo é a maior fonte de variância do método —
    desvio de 0.10 a 0.22, contra 0.009 entre partições dos dados.

    Rodadas seguintes: CoreSet (k-center guloso) sobre as features do encoder
    atual. Escolhe a imagem mais DISTANTE do que já foi anotado, cobrindo o
    espaço em vez de perseguir incerteza. Foi o critério que se sustentou no
    teste por decoder; a entropia não.
    """
    anotadas = [i for i in S["pool"] if i in S["markers"]]
    restantes = [i for i in S["pool"] if i not in S["markers"]]
    if not restantes:
        return {"erro": "pool esgotado", "rodada": len(anotadas)}

    alvo, motivo = _escolher_alvo(S["criterio"], anotadas, restantes,
                                  S["encoder"], S.setdefault(
                                      "rng", np.random.default_rng(11)))
    S["alvo"] = alvo
    return {"alvo": alvo, "motivo": motivo, "rodada": len(anotadas) + 1,
            "k_alvo": S["k_alvo"], "restantes": len(restantes),
            "criterio": S["criterio"]}


def acao_sugerir() -> dict:
    if not S["encoder"]:
        return {"erro": "treine ao menos uma vez antes de pedir sugestão"}
    r = _entropia_pool()
    if not r:
        return {"erro": "pool esgotado"}
    return {"ranking": r[:8], "escolhida": r[0]["id"],
            "motivo": f"maior entropia = {r[0]['entropia']:.5f}"}


# ── execução em lote: critério × K ───────────────────────────────────────────

def _lote_thread(cfg: dict) -> None:
    def prog(d):
        LOTE.update(d)

    LOTE.update({"rodando": True, "erro": "", "feito": 0, "resultados": [],
                 "cancelar": False, "config": cfg})
    try:
        RUN.executar(cfg["ds"], cfg["criterios"], cfg["ks"], cfg["decoder"],
                     cfg["n_test"], cfg["semente"], prog,
                     lambda: LOTE["cancelar"])
    except Exception as e:
        traceback.print_exc()
        LOTE["erro"] = str(e)[:300]
    finally:
        LOTE["rodando"] = False


# ── validação pelo especialista → matriz de confusão ─────────────────────────

def acao_validar_iniciar(n: int) -> dict:
    """
    Monta a fila de imagens para o especialista julgar.

    A matriz é de DETECÇÃO, não de pixels: o modelo diz "achei objeto" ou
    "não achei", e a pessoa confirma se havia. É o julgamento que um
    especialista consegue dar em um clique, e é a pergunta clínica real —
    esta lâmina precisa ser examinada?
    """
    if not S["encoder"]:
        return {"erro": "treine antes de validar"}
    usadas = set(S["markers"])
    cand = [i for i in _listar() if i not in usadas]
    if not cand:
        return {"erro": "sem imagens para validar"}

    # Metade das mais CONFIANTES, metade das mais CONFUSAS, alternadas.
    #
    # Sortear dilui o teste: a maior parte de um pool qualquer e trivial, e o
    # especialista gasta os cliques confirmando o que ja se sabia. Estratificar
    # pela incerteza do modelo poe metade do esforco exatamente onde o modelo
    # esta no muro -- e permite reportar a matriz separada por estrato, que diz
    # se o erro se concentra nos casos dificeis ou esta espalhado.
    #
    # A incerteza vem da saida do proprio modelo. Nenhuma anotacao e lida aqui:
    # o estrato e escolhido sem gabarito, como o resto do laco.
    rng = np.random.default_rng(3)
    amostra = sorted(rng.choice(
        cand, size=min(MAX_PONTUAR_VALID, len(cand)), replace=False).tolist())
    pontos = _pontuar_pool(amostra, S["encoder"])
    pontos.sort(key=lambda r: r["entropia"])

    metade = max(1, min(n, len(pontos)) // 2)
    faceis = pontos[:metade]
    dificeis = pontos[-(min(n, len(pontos)) - metade):]
    estrato = {}
    fila = []
    for a, b in zip(dificeis, faceis):          # comeca pela dificil
        for r, tag in ((a, "dificil"), (b, "facil")):
            if r["id"] in estrato:
                continue
            estrato[r["id"]] = {"classe": tag, "entropia": r["entropia"]}
            fila.append(r["id"])
    VALID.update({"fila": fila[:n], "i": 0, "respostas": [], "pred": {},
                  "estrato": estrato})
    return acao_validar_atual()


def acao_validar_atual() -> dict:
    if VALID["i"] >= len(VALID["fila"]):
        return {"fim": True, "matriz": _matriz(),
                "por_classe": {c: _matriz(c) for c in ("dificil", "facil")}}
    img = VALID["fila"][VALID["i"]]
    p = _prever(img)
    detectou = p["area"] > 0
    VALID["pred"][img] = detectou
    e = VALID.get("estrato", {}).get(img, {})
    return {"fim": False, "id": img, "mask": p["mask"],
            "detectou": detectou, "area": p["area"],
            "i": VALID["i"] + 1, "n": len(VALID["fila"]),
            "classe": e.get("classe"), "entropia": e.get("entropia"),
            "tem_gt": bool(_label()) and bool(DS.caminho_label(
                S["ds"], _arquivo(img)))}


def acao_validar_responder(tem_objeto: bool) -> dict:
    if VALID["i"] < len(VALID["fila"]):
        img = VALID["fila"][VALID["i"]]
        VALID["respostas"].append({
            "id": img, "modelo": VALID["pred"].get(img, False),
            "especialista": bool(tem_objeto),
            "classe": VALID.get("estrato", {}).get(img, {}).get("classe")})
        VALID["i"] += 1
    return acao_validar_atual()


def _matriz(classe: str | None = None) -> dict:
    vp = vn = fp = fn = 0
    for r in VALID["respostas"]:
        if classe is not None and r.get("classe") != classe:
            continue
        m, e = r["modelo"], r["especialista"]
        if m and e:
            vp += 1
        elif m and not e:
            fp += 1
        elif not m and e:
            fn += 1
        else:
            vn += 1
    tot = max(1, vp + vn + fp + fn)
    prec = vp / max(1, vp + fp)
    rec = vp / max(1, vp + fn)
    beta2 = 0.3
    fb = ((1 + beta2) * prec * rec / (beta2 * prec + rec)) if (prec + rec) else 0.0
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn, "n": tot,
            "acuracia": round((vp + vn) / tot, 3),
            "precisao": round(prec, 3), "revocacao": round(rec, 3),
            "fbeta": round(fb, 3)}


# ── AL por região: o especialista julga onde o modelo está confuso ───────────
#
# Este é o nível de granularidade que a medição diz ser o mais valioso. Ao
# decompor o ganho dos métodos de região (RESULTADOS_CONSOLIDADOS.md), a
# SELEÇÃO de imagens valeu +0.050 de Fβ e a GEOMETRIA dos seeds valeu +0.108 --
# 2.1x mais. Para o FLIM, onde o traço cai importa mais do que qual imagem se
# abre, porque os kernels são k-means sobre os patches ao redor dos pixels
# marcados.
#
# Mas o outro lado também foi medido: trocar os marcadores do especialista por
# seeds gerados inteiramente pela máquina custou -0.214 de Fβ. A máquina sozinha
# não chega perto.
#
# Daí o híbrido implementado aqui: a máquina PROPÕE as regiões (ela é boa em
# achar onde está confusa) e a pessoa DECIDE o rótulo de cada uma (ela é
# insubstituível nisso). Nenhuma anotação de referência é lida em nenhum passo.

REG: dict = {"img": None, "sp": None, "ordem": [], "i": 0, "respostas": {},
             "prob": None}
N_SEEDS_REGIAO = 12       # seeds sorteados dentro de cada região aceita
LIMIAR_ENTROPIA_UTIL = 0.02   # abaixo disso o ranking é ruído


def _regiao_overlay(sp: np.ndarray, rid: int) -> str:
    """RGBA destacando uma região: preenchimento suave + contorno forte."""
    m = sp == rid
    borda = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0)
                  & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
    rgba[m] = (255, 214, 0, 70)
    rgba[borda] = (255, 170, 0, 255)
    return _png(rgba)


def acao_regiao_iniciar(img_id: str | None, k: int) -> dict:
    """
    Monta a fila de regiões mais confusas de uma imagem.

    A imagem é a que o critério de AL escolheu (ou a informada). As regiões vêm
    de SLIC sobre a imagem e são ordenadas pela entropia média do mapa de
    saliência do modelo atual -- a mesma quantidade do critério por imagem,
    agora resolvida no espaço.
    """
    if not S["encoder"]:
        return {"erro": "treine ao menos uma vez antes de julgar regiões"}
    img_id = img_id or S["alvo"]
    if not img_id:
        return {"erro": "nenhuma imagem alvo — peça a próxima primeiro"}

    from flim_al.region_al import compute_superpixels, score_regions_by_entropy

    arr = np.array(Image.open(os.path.join(_orig(), _arquivo(img_id))))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    arr = arr[:, :, :3]
    h, w = arr.shape[:2]

    model = _montar_modelo(S["encoder"])
    prob = _mapa_prob(model, img_id)
    if prob.shape != (h, w):
        prob = np.array(Image.fromarray((prob * 255).astype(np.uint8))
                        .resize((w, h), Image.BILINEAR)) / 255.0

    sp = compute_superpixels(arr, n_segments=220)
    scores = score_regions_by_entropy(sp, prob)
    ordem = sorted(scores, key=lambda r: -scores[r])[:max(1, k)]

    REG.update({"img": img_id, "sp": sp, "ordem": ordem, "i": 0,
                "respostas": {}, "prob": prob,
                "scores": {int(r): round(scores[r], 5) for r in ordem}})
    r = acao_regiao_atual()

    # Quando o modelo está confiante em toda a imagem, a entropia é
    # praticamente uniforme e "as regiões mais confusas" viram ordenação de
    # ruído -- o ranking existe, mas não significa nada. Isso precisa aparecer
    # na tela: julgar regiões nessa situação gasta o tempo do especialista sem
    # informação. O limiar é conservador; a entropia binária vai a ln 2 ≈ 0.693.
    topo = max(REG["scores"].values()) if REG["scores"] else 0.0
    r["entropia_maxima"] = round(topo, 5)
    r["degenerado"] = topo < LIMIAR_ENTROPIA_UTIL
    return r


def acao_regiao_atual() -> dict:
    if REG["img"] is None:
        return {"erro": "nenhuma sessão de regiões aberta"}
    if REG["i"] >= len(REG["ordem"]):
        return _regiao_fechar()
    rid = REG["ordem"][REG["i"]]
    m = REG["sp"] == rid
    pm = float(REG["prob"][m].mean())
    return {
        "fim": False, "img": REG["img"], "regiao": int(rid),
        "i": REG["i"] + 1, "n": len(REG["ordem"]),
        "overlay": _regiao_overlay(REG["sp"], rid),
        "entropia": REG["scores"][int(rid)],
        "px": int(m.sum()),
        # o palpite do modelo, mostrado DEPOIS que a pessoa decide seria melhor;
        # mostrado antes, ancora a resposta. Vai como campo separado para a
        # interface poder esconder.
        "palpite": "objeto" if pm > 0.5 else "fundo",
        "prob_media": round(pm, 3),
    }


def acao_regiao_responder(resposta: str) -> dict:
    """resposta ∈ {fg, bg, pular}."""
    if REG["i"] < len(REG["ordem"]):
        rid = REG["ordem"][REG["i"]]
        if resposta in ("fg", "bg"):
            REG["respostas"][int(rid)] = resposta
        REG["i"] += 1
    return acao_regiao_atual()


def _regiao_fechar() -> dict:
    """
    Converte as regiões julgadas em seeds FLIM e junta aos markers da imagem.

    Os seeds são sorteados DENTRO da região, não no centroide: o k-means precisa
    de patches variados da mesma textura, e um ponto só daria um protótipo
    frágil. O número por região é pequeno de propósito -- medimos que concentrar
    traço colapsa o encoder (traço longo levou o Fβ a 0.000).
    """
    from flim_al.region_al import regions_to_flim_seeds

    img_id, sp = REG["img"], REG["sp"]
    aceitas = list(REG["respostas"])
    if not aceitas:
        REG["img"] = None
        return {"fim": True, "img": img_id, "fg": 0, "bg": 0,
                "aviso": "nenhuma região foi rotulada"}

    h, w = sp.shape
    d = regions_to_flim_seeds(sp, aceitas, REG["respostas"], h, w,
                              n_seeds_per_region=N_SEEDS_REGIAO)
    m = S["markers"].setdefault(img_id, {"fg": [], "bg": []})
    m["fg"] += [[int(c), int(r)] for c, r in d["fg_seeds"]]
    m["bg"] += [[int(c), int(r)] for c, r in d["bg_seeds"]]
    n_fg = int(len(d["fg_seeds"]))
    n_bg = int(len(d["bg_seeds"]))
    REG["img"] = None
    return {"fim": True, "img": img_id, "fg": n_fg, "bg": n_bg,
            "regioes": len(aceitas),
            "total_fg": len(m["fg"]), "total_bg": len(m["bg"]),
            "aviso": None if (n_fg and n_bg) else
                     "faltou rotular ao menos uma região de cada classe: sem "
                     "kernels de fundo o decoder não tem o que subtrair"}


# ── comparação de N braços: FLIM puro contra vários critérios de AL ──────────
#
# Estrutura importante: os markers são GLOBAIS por imagem, não por braço.
#
# Se dois critérios escolhem a mesma imagem numa rodada, o especialista anota
# uma vez só e os dois recebem exatamente o mesmo rabisco. Isso não é economia
# de esforço apenas: é o que faz a comparação ser pareada. Se cada braço tivesse
# a própria anotação da mesma imagem, parte da diferença medida no fim viria de
# dois rabiscos diferentes, e não do critério de seleção — e a geometria do
# traço pesa 2.1x a seleção de imagens (RESULTADOS_CONSOLIDADOS.md).
#
# Cada braço guarda apenas QUAIS imagens entraram no conjunto de treino dele.

N_FG_SUGERIDO = 4         # toques de objeto propostos por imagem
N_BG_SUGERIDO = 9         # toques de fundo — a proporção dos markers reais
RAIO_BG_SUGERIDO = 120    # distância mínima do objeto, em pixels
RAIO_PINCEL = 5           # o mesmo pincel da interface
MARGEM_BORDA = RAIO_PINCEL + 2   # para o disco caber inteiro na imagem


def _disco(cx: int, cy: int, h: int, w: int) -> list:
    """
    Um toque do pincel: disco de raio RAIO_PINCEL, recortado na imagem.

    A proposta precisa render os mesmos pixels que renderia se a pessoa tivesse
    clicado ali — senão a comparação entre anotar à mão e aceitar a proposta
    mediria também a diferença de volume de traço, que é grande: nas mesmas três
    imagens, 6.007 px de marcador deram Fβ 0.665 e 50.046 px deram 0.000.
    """
    pts = []
    for dy in range(-RAIO_PINCEL, RAIO_PINCEL + 1):
        for dx in range(-RAIO_PINCEL, RAIO_PINCEL + 1):
            if dx * dx + dy * dy > RAIO_PINCEL * RAIO_PINCEL:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                pts.append([int(x), int(y)])
    return pts


def _bracos_padrao(criterios: list) -> dict:
    nome = {c[0]: c[1] for c in CRITERIOS}
    b = {"puro": {"nome": "FLIM puro (sorteio)", "criterio": "random"}}
    for c in criterios:
        if c in CRITERIO_IDS and c != "random":
            b[c] = {"nome": nome.get(c, c), "criterio": c}
    return b


def acao_comp_iniciar(k: int, criterios: list) -> dict:
    _init_conjuntos()
    base = _bracos_padrao(criterios or ["coreset"])
    D.update({"ativo": True, "k": max(1, k), "rng": np.random.default_rng(7),
              "resultado": None, "semente": None, "markers": {},
              "centros": {}, "confirmadas": set(), "rodada": 0, "fila": [],
              "fila_i": 0, "sugestor": None,
              "ordem": list(base),
              "bracos": {i: dict(v, imgs=[], enc=None, alvo=None, motivo="",
                                 evento="") for i, v in base.items()}})
    return acao_comp_proxima()


def _fila_da_rodada() -> dict:
    """As imagens distintas que a rodada pede, e quem pediu cada uma."""
    pedidos = {}
    for bid in D["ordem"]:
        alvo = D["bracos"][bid]["alvo"]
        if alvo and alvo not in D["confirmadas"]:
            pedidos.setdefault(alvo, []).append(bid)
    D["fila"] = list(pedidos)
    D["fila_i"] = 0
    return pedidos


def _resumo_bracos() -> list:
    junto = {}
    for bid in D["ordem"]:
        a = D["bracos"][bid]["alvo"]
        if a:
            junto.setdefault(a, []).append(D["bracos"][bid]["nome"])
    return [{"id": b, "nome": D["bracos"][b]["nome"],
             "criterio": D["bracos"][b]["criterio"],
             "alvo": D["bracos"][b]["alvo"],
             "motivo": D["bracos"][b]["motivo"],
             "evento": D["bracos"][b].get("evento", ""),
             "compartilha_com": [n for n in junto.get(D["bracos"][b]["alvo"], [])
                                 if n != D["bracos"][b]["nome"]],
             "imgs": list(D["bracos"][b]["imgs"])} for b in D["ordem"]]


def _fila_detalhe() -> list:
    """A fila da rodada corrente no mesmo formato que _estado_rodada usa."""
    quem = {}
    for bid in D["ordem"]:
        a = D["bracos"][bid]["alvo"]
        if a in D["fila"]:
            quem.setdefault(a, []).append(bid)
    out = []
    for img in D["fila"]:
        bids = quem.get(img) or list(D["ordem"])
        out.append({
            "id": img,
            "pedida_por": [D["bracos"][b]["nome"] for b in bids],
            "motivo": D["bracos"][bids[0]]["motivo"],
            "anotada": img in D["confirmadas"],
            "proposto": D["markers"].get(img, {}).get("proposto", False),
            "centros": D["centros"].get(img),
            "px": {"fg": len(D["markers"].get(img, {}).get("fg", [])),
                   "bg": len(D["markers"].get(img, {}).get("bg", []))},
        })
    return out


def _estado_rodada(pedidos: dict, semente_comum: bool = False) -> dict:
    return {
        "fim": False, "rodada": D["rodada"], "k": D["k"],
        "semente_comum": semente_comum,
        "fila": [{"id": img,
                  "pedida_por": [D["bracos"][b]["nome"] for b in quem],
                  "motivo": D["bracos"][quem[0]]["motivo"],
                  # "anotada" = o especialista CONFIRMOU. Uma imagem
                  # pre-marcada tem markers mas ainda nao foi vista por
                  # ninguem; conta-la como anotada deixaria avancar de rodada
                  # sem revisao nenhuma, que e o oposto do que a pre-marcacao
                  # existe para fazer.
                  "anotada": img in D["confirmadas"],
                  "proposto": D["markers"].get(img, {}).get("proposto", False),
                  "centros": D["centros"].get(img),
                  "px": {"fg": len(D["markers"].get(img, {}).get("fg", [])),
                         "bg": len(D["markers"].get(img, {}).get("bg", []))}}
                 for img, quem in pedidos.items()],
        "bracos": _resumo_bracos(),
    }


def acao_comp_proxima() -> dict:
    """
    Retreina cada braço com as imagens dele e deixa cada critério escolher.

    O retreino entre rodadas não é otimização: CoreSet e BADGE medem distância
    no espaço de features do encoder ATUAL daquele braço. Sem retreinar, todos
    cairiam no medoide e deixariam de ser critérios diferentes.
    """
    cfg = _cfg()
    work = os.path.join(S["work"], "comp")
    os.makedirs(work, exist_ok=True)

    if D["bracos"] and min(len(b["imgs"]) for b in D["bracos"].values()) >= D["k"]:
        return {"fim": True, "k": D["k"], "rodada": D["rodada"],
                "bracos": _resumo_bracos()}

    # Avançar com imagens pendentes descartaria silenciosamente o que os
    # critérios escolheram nesta rodada — os braços ficariam com menos imagens
    # do que o orçamento, e a rodada seguinte partiria de encoders treinados com
    # conjuntos diferentes do que a tela mostrou. Melhor recusar e dizer o que
    # falta.
    pendentes = [i for i in D["fila"] if i not in D["confirmadas"]]
    if pendentes:
        return {"erro": "ainda faltam imagens desta rodada: "
                        + ", ".join(pendentes)
                        + ". Anote e confirme cada uma antes de avançar.",
                "pendentes": pendentes}

    # RODADA 1: todos os braços partem da MESMA imagem, com o MESMO rabisco.
    # Nenhum tem modelo ainda, então não há critério a exercer — e a imagem
    # inicial é a maior fonte de variância do método (0.10 a 0.22 de Fβ contra
    # 0.009 entre partições). Partindo do mesmo ponto, o que sobra é o critério.
    if not D["markers"]:
        semente, _ = _escolher_alvo("coreset", [], list(S["pool"]), None)
        D["semente"] = semente
        for bid in D["ordem"]:
            D["bracos"][bid]["alvo"] = semente
            D["bracos"][bid]["motivo"] = (
                "imagem inicial comum a todos os braços: a mais típica do "
                "pool, medida direto nos pixels — ainda não há modelo")
        D["rodada"] = 1
        return _estado_rodada(_fila_da_rodada(), semente_comum=True)

    for bid in D["ordem"]:
        b = D["bracos"][bid]
        if b["imgs"]:
            mdir = os.path.join(work, "m_" + bid)
            marc = {i: D["markers"][i] for i in b["imgs"] if i in D["markers"]}
            if DUP.escrever_markers(marc, cfg["orig"], mdir):
                enc = os.path.join(work, "e_" + bid + ".pth")
                retrain_encoder(cfg["arch"], mdir, cfg["orig"],
                                cfg["label"] or cfg["orig"], "cpu", enc)
                b["enc"] = enc
        # Se o criterio deste braco escolher uma imagem que OUTRO braco ja fez
        # o especialista anotar, ela entra sem custo nenhum: a anotacao ja
        # existe e e a mesma para todos. Sem isso o braco ficava com o alvo
        # marcado como "ja anotado", ninguem confirmava, e ele terminava com
        # menos imagens que os outros -- o que quebraria a premissa da
        # comparacao, que e orcamento igual em todos os bracos.
        b["alvo"], b["motivo"] = None, "orçamento concluído"
        b["evento"] = "orçamento concluído"
        reaproveitadas = []
        while len(b["imgs"]) < D["k"]:
            rest = [i for i in S["pool"] if i not in b["imgs"]]
            alvo, motivo = _escolher_alvo(b["criterio"], b["imgs"], rest,
                                          b["enc"], D["rng"])
            if not alvo:
                b["evento"] = "pool esgotado"
                break
            if alvo in D["confirmadas"]:
                b["imgs"].append(alvo)
                reaproveitadas.append(alvo)
                continue
            b["alvo"], b["motivo"] = alvo, motivo
            b["evento"] = f"pediu {alvo}"
            break
        if reaproveitadas:
            # Esta linha existe porque sem ela o braço parecia ter "sumido" da
            # rodada: o critério dele escolheu uma imagem que outro braço já
            # fez o especialista anotar, ela entrou sem custo nenhum, e nada
            # apareceu na fila. É comportamento correto — a anotação é a mesma
            # para todos — mas precisa estar escrito na tela.
            b["evento"] = ("reaproveitou " + ", ".join(reaproveitadas)
                           + " (já anotada, sem custo)"
                           + (f" · pediu {b['alvo']}" if b["alvo"] else ""))

    D["rodada"] += 1
    D["sugestor"] = None          # o sugestor é retreinado a cada rodada
    pedidos = _fila_da_rodada()

    # Da rodada 2 em diante a imagem já chega pré-marcada. É o ponto do
    # exercício: reduzir o esforço do especialista de "desenhar do zero" para
    # "conferir e, se preciso, corrigir". Uma passada só do sugestor serve todas
    # as imagens da rodada.
    for img in list(pedidos):
        if img not in D["confirmadas"] and img not in D["centros"]:
            _premarcar_silencioso(img)
    return _estado_rodada(pedidos)


def _premarcar_silencioso(img: str) -> None:
    """Pré-marca sem estourar: se falhar, a imagem simplesmente vem em branco."""
    try:
        enc = D.get("sugestor") or _treinar_sugestor()
        if not enc:
            return
        prop = _pontos_propostos(img, enc)
        if not prop["fg"] and not prop["bg"]:
            return
        D["markers"][img] = {"fg": prop["fg"], "bg": prop["bg"],
                             "proposto": True}
        D["centros"][img] = {"fg": prop["centros_fg"], "bg": prop["centros_bg"],
                             "encontrou_objeto": prop["encontrou_objeto"]}
    except Exception:
        traceback.print_exc()


def acao_comp_marker(img: str, fg: list, bg: list) -> dict:
    m = D["markers"].setdefault(img, {"fg": [], "bg": [], "proposto": False})
    m["fg"] += [[int(c), int(r)] for c, r in fg]
    m["bg"] += [[int(c), int(r)] for c, r in bg]
    return {"img": img, "fg": len(m["fg"]), "bg": len(m["bg"]),
            "proposto": m["proposto"]}


def acao_comp_limpar(img: str) -> dict:
    """O 'corrigir': apaga tudo daquela imagem e devolve a folha em branco."""
    D["markers"].pop(img, None)
    D["centros"].pop(img, None)
    D["confirmadas"].discard(img)
    for bid in D["ordem"]:
        b = D["bracos"][bid]
        if img in b["imgs"]:
            b["imgs"].remove(img)
    return {"img": img, "fg": 0, "bg": 0, "proposto": False,
            "aviso": "pontos apagados — anote a imagem do zero"}


def acao_comp_confirmar(img: str) -> dict:
    """
    Fecha a imagem: registra-a no conjunto de treino de quem a pediu.

    Só entra quem pediu. Uma imagem que o CoreSet escolheu não vai para o braço
    do sorteio, senão os conjuntos de treino convergiriam e a comparação
    perderia o sentido.
    """
    m = D["markers"].get(img)
    if not m or not m["fg"] or not m["bg"]:
        return {"erro": "a imagem precisa de pontos de objeto E de fundo: sem "
                        "kernels de fundo o decoder não tem o que subtrair das "
                        "ativações, e o Fβ despenca"}
    D["confirmadas"].add(img)
    for bid in D["ordem"]:
        b = D["bracos"][bid]
        if b["alvo"] == img:
            if img not in b["imgs"]:
                b["imgs"].append(img)
            # Sem zerar o alvo, a tabela seguiria mostrando como "próxima" uma
            # imagem que o braço acabou de receber.
            b["alvo"] = None
            b["evento"] = f"recebeu {img}"
    faltam = [i for i in D["fila"] if i not in D["confirmadas"]]
    return {"img": img, "faltam": faltam, "bracos": _resumo_bracos()}


# ── pré-marcação: o modelo propõe, o especialista corrige ────────────────────

def _treinar_sugestor():
    """
    Encoder auxiliar treinado com TODAS as imagens já anotadas na sessão.

    Ele não é nenhum dos braços — existe só para propor pontos. Usar a união
    garante que a proposta para uma imagem seja a mesma independentemente de
    qual braço a pediu, o que mantém a anotação idêntica entre os braços. Se
    cada braço propusesse com o próprio encoder, dois braços que escolhessem a
    mesma imagem receberiam rabiscos diferentes e a comparação deixaria de ser
    pareada — que é justamente o que ela precisa ser.
    """
    if not D["markers"]:
        return None
    cfg = _cfg()
    work = os.path.join(S["work"], "comp")
    os.makedirs(work, exist_ok=True)
    mdir = os.path.join(work, "m_sugestor")
    # So entra o que o especialista confirmou. Treinar o sugestor com as
    # proprias propostas dele realimentaria o erro: uma pre-marcacao errada
    # viraria evidencia para a proxima pre-marcacao.
    prontas = {i: m for i, m in D["markers"].items()
               if i in D["confirmadas"] and m["fg"] and m["bg"]}
    if not prontas or not DUP.escrever_markers(prontas, cfg["orig"], mdir):
        return None
    enc = os.path.join(work, "e_sugestor.pth")
    retrain_encoder(cfg["arch"], mdir, cfg["orig"], cfg["label"] or cfg["orig"],
                    "cpu", enc)
    D["sugestor"] = enc
    return enc


def _mascara_predita(img_id: str, enc_path: str, h: int, w: int):
    model = _montar_modelo(enc_path)
    prob = _mapa_prob(model, img_id)
    sal = (prob * 255).astype(np.uint8)
    if sal.shape != (h, w):
        sal = np.array(Image.fromarray(sal).resize((w, h), Image.BILINEAR),
                       dtype=np.uint8)
    return _official_filter_and_binarize(sal, _area()).astype(bool)


def _espalhar(rs, cs, n: int, rng):
    """
    n pontos espalhados pelo conjunto, por k-center guloso.

    Sortear puro juntaria toques vizinhos, e toque concentrado é o modo de falha
    medido: o mesmo orçamento de pinceladas concentrado em 2 imagens colapsou o
    encoder em 12 de 12 execuções; espalhado em 8, em 0 de 12.
    """
    total = len(rs)
    if total <= n:
        return list(range(total))
    pts = np.stack([cs.astype(np.float32), rs.astype(np.float32)], axis=1)
    escolhidos = [int(rng.integers(total))]
    d = np.linalg.norm(pts - pts[escolhidos[0]], axis=1)
    while len(escolhidos) < n:
        i = int(np.argmax(d))
        escolhidos.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return escolhidos


def _pontos_propostos(img_id: str, enc_path: str) -> dict:
    """
    Pontos que o modelo propõe: objeto na BORDA da predição, fundo bem longe.

    A geometria segue o que foi medido, não o que é conveniente. Nas mesmas três
    imagens, tocar na borda do ovo deu Fβ 0.665 e tocar no centro deu 0.000: no
    centro o patch 3x3 é textura uniforme, igual a qualquer região escura; na
    borda ele contém a transição casca-fundo, que é o que discrimina. Os toques
    de fundo saem a pelo menos 120 px do objeto — distância em que o Fβ medido
    sobe de 0.665 para 0.700, porque perto da borda o patch de fundo já contém
    parte da transição e fica parecido com o de objeto.
    """
    from scipy import ndimage

    arr = np.array(Image.open(os.path.join(_orig(), _arquivo(img_id))))
    h, w = arr.shape[0], arr.shape[1]
    obj = _mascara_predita(img_id, enc_path, h, w)
    rng = np.random.default_rng(abs(hash(img_id)) % (2 ** 31))

    # Nada de tocar na borda da imagem: o FLIM recorta um patch 3x3 ao redor de
    # cada pixel marcado, e um toque no canto produz patch truncado. O k-center
    # guloso, que busca o ponto mais distante, cai exatamente nos cantos se
    # deixado à vontade — foi o que aconteceu na primeira versão.
    dentro = np.zeros((h, w), dtype=bool)
    dentro[MARGEM_BORDA:h - MARGEM_BORDA, MARGEM_BORDA:w - MARGEM_BORDA] = True

    centros_fg = []
    if obj.any():
        borda = obj & ~ndimage.binary_erosion(obj, iterations=2) & dentro
        rs, cs = np.where(borda)
        if len(rs):
            idx = _espalhar(rs, cs, N_FG_SUGERIDO, rng)
            centros_fg = [(int(cs[i]), int(rs[i])) for i in idx]

    if obj.any():
        dist = ndimage.distance_transform_edt(~obj)
        longe = (dist > RAIO_BG_SUGERIDO) & dentro
        if longe.sum() < N_BG_SUGERIDO * 4:
            longe = (dist > max(20, RAIO_BG_SUGERIDO // 3)) & dentro
    else:
        longe = dentro.copy()
    rs, cs = np.where(longe)
    centros_bg = []
    if len(rs):
        idx = _espalhar(rs, cs, N_BG_SUGERIDO, rng)
        centros_bg = [(int(cs[i]), int(rs[i])) for i in idx]

    fg = [pt for cx, cy in centros_fg for pt in _disco(cx, cy, h, w)]
    bg = [pt for cx, cy in centros_bg for pt in _disco(cx, cy, h, w)]
    return {"fg": fg, "bg": bg, "area": int(obj.sum()),
            "encontrou_objeto": bool(obj.any()),
            "centros_fg": [list(c) for c in centros_fg],
            "centros_bg": [list(c) for c in centros_bg]}


def acao_comp_dica(img_id: str, k: int = 5) -> dict:
    """
    As regiões onde o modelo de AL está mais confuso nesta imagem.

    Devolve UM overlay com as `k` regiões mais incertas, a mais incerta com o
    preenchimento mais forte. Não grava marcador nenhum: a decisão de traçar
    ali continua sendo de quem anota.

    O modelo consultado é o do braço de AL que pediu esta imagem. Se mais de um
    pediu, o primeiro da ordem — os critérios divergem justamente em onde
    acham incerteza, e misturar dois mapas mostraria uma incerteza que nenhum
    modelo tem.
    """
    if not D["ativo"]:
        return {"erro": "a comparação não está em andamento"}

    candidatos = [b for bid, b in D["bracos"].items()
                  if bid != "puro" and b.get("enc")]
    if not candidatos:
        return {"erro": "nenhum braço de AL tem modelo ainda — na rodada 1 a "
                        "imagem é comum a todos e escolhida direto nos "
                        "pixels, então não há incerteza para apontar. "
                        "Anote esta rodada e a dica aparece na próxima."}
    dono = next((b for b in candidatos if b.get("alvo") == img_id),
                candidatos[0])

    from flim_al.region_al import compute_superpixels, score_regions_by_entropy

    arr = np.array(Image.open(os.path.join(_orig(), _arquivo(img_id))))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    arr = arr[:, :, :3]
    h, w = arr.shape[:2]

    modelo = _montar_modelo(dono["enc"])
    prob = _mapa_prob(modelo, img_id)
    if prob.shape != (h, w):
        prob = np.array(Image.fromarray((prob * 255).astype(np.uint8))
                        .resize((w, h), Image.BILINEAR)) / 255.0

    sp = compute_superpixels(arr, n_segments=220)
    scores = score_regions_by_entropy(sp, prob)
    ordem = sorted(scores.items(), key=lambda kv: -kv[1])[:max(1, k)]
    if not ordem:
        return {"erro": "nenhuma região calculada"}

    # Um modelo confiante em toda a imagem produz um ranking que ordena ruido.
    # Dizer isso e melhor que pintar cinco regioes quaisquer e deixar a pessoa
    # gastar o traco nelas.
    maxima = float(ordem[0][1])
    degenerado = maxima < 1e-4

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for pos, (rid, _) in enumerate(ordem):
        m = sp == rid
        # A mais incerta aparece mais forte: a ordem tem que ser visivel, senao
        # as cinco regioes parecem equivalentes.
        alfa = int(95 - pos * (55 / max(1, len(ordem) - 1))) if len(ordem) > 1 else 95
        rgba[m] = (255, 214, 0, max(28, alfa))
        borda = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0)
                      & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        rgba[borda] = (255, 150, 0, 255)

    return {
        "img": img_id, "overlay": _png(rgba),
        "braco": dono["nome"], "criterio": dono["criterio"],
        "n_regioes": len(ordem),
        "px": int(sum(int((sp == rid).sum()) for rid, _ in ordem)),
        "entropia_maxima": round(maxima, 5),
        "degenerado": bool(degenerado),
    }


def acao_comp_premarcar(img: str) -> dict:
    """
    Propõe os pontos e já os grava como marcadores da imagem.

    Ficam gravados de propósito: o caminho de menor esforço é aceitar e seguir.
    Quem quiser corrigir aperta 'corrigir', que apaga TUDO e devolve a imagem em
    branco — é uma decisão explícita, não um estado ambíguo com metade dos
    pontos da máquina e metade da pessoa.
    """
    if D["rodada"] < 2:
        return {"erro": "a pré-marcação começa na rodada 2, depois do primeiro "
                        "treinamento — na rodada 1 não há modelo para propor"}
    enc = D.get("sugestor") or _treinar_sugestor()
    if not enc:
        return {"erro": "ainda não há imagem anotada por completo para treinar "
                        "o modelo que propõe"}
    prop = _pontos_propostos(img, enc)
    if not prop["fg"] and not prop["bg"]:
        return {"erro": "o modelo não conseguiu propor pontos nesta imagem"}
    D["markers"][img] = {"fg": prop["fg"], "bg": prop["bg"], "proposto": True}
    D["centros"][img] = {"fg": prop["centros_fg"], "bg": prop["centros_bg"],
                         "encontrou_objeto": prop["encontrou_objeto"]}
    # A interface desenha os CENTROS (um círculo por toque); os pixels do disco
    # são o que vai para o encoder. Mandar os milhares de pixels para o canvas
    # só engordaria a resposta.
    return {"img": img, "fg": prop["centros_fg"], "bg": prop["centros_bg"],
            "n_fg": len(prop["centros_fg"]), "n_bg": len(prop["centros_bg"]),
            "px_fg": len(prop["fg"]), "px_bg": len(prop["bg"]),
            "encontrou_objeto": prop["encontrou_objeto"],
            "area": prop["area"], "proposto": True,
            "aviso": None if prop["encontrou_objeto"] else
                     "o modelo NÃO encontrou objeto aqui: propôs só pontos de "
                     "fundo. Marque o objeto você, ou corrija tudo"}


# ── treino e comparação final ────────────────────────────────────────────────

def acao_comp_treinar(sorteios: int) -> dict:
    # Treinar antes de fechar o orcamento compara bracos com quantidades
    # diferentes de imagens -- e "mesmo orcamento" e a premissa que faz a
    # comparacao significar alguma coisa. Com zero imagens o treino nem
    # acontece e a tabela sai toda de tracos, que foi o que apareceu na tela.
    if not D.get("bracos"):
        return {"erro": "nenhuma comparação iniciada"}
    faltando = {b["nome"]: D["k"] - len(b["imgs"])
                for b in D["bracos"].values() if len(b["imgs"]) < D["k"]}
    if faltando:
        det = ", ".join(f"{n} (faltam {q})" for n, q in faltando.items())
        return {"erro": f"o orçamento ainda não fechou: {det}. "
                        "Conclua as rodadas antes de treinar."}
    cfg = _cfg()
    val = [_arquivo(i) for i in S["val"]]
    work = os.path.join(S["work"], "comp")
    os.makedirs(work, exist_ok=True)

    linhas = []
    for bid in D["ordem"]:
        b = D["bracos"][bid]
        marc = {i: D["markers"][i] for i in b["imgs"] if i in D["markers"]}
        r = DUP.treinar_e_avaliar(cfg, marc, val, S["decoder"], S["block"],
                                  work, bid)
        if "encoder" in r:
            b["enc"] = r.pop("encoder")
        r.update({"id": bid, "nome": b["nome"], "criterio": b["criterio"],
                  "imagens": list(b["imgs"])})
        linhas.append(r)

    base = next((l for l in linhas if l["id"] == "puro"), None)
    for l in linhas:
        if base and l.get("fb") is not None and base.get("fb") is not None:
            l["delta"] = round(l["fb"] - base["fb"], 4)

    out = {"linhas": linhas, "n_val": len(val)}
    if sorteios > 1 and cfg["label"] and base and base.get("imagens"):
        pool = [_arquivo(i) for i in S["pool"]]
        m = DUP.sem_al_multiplos(cfg, pool, len(base["imagens"]), val,
                                 S["decoder"], S["block"], S["ds"], work,
                                 sorteios)
        for e in m.get("execucoes", []):
            e.pop("encoder", None)
        out["sorteios_extras"] = m
    D["resultado"] = out
    return out


def acao_comp_prever(img: str) -> dict:
    """A mesma imagem, por todos os encoders — é a comparação visual."""
    out = {"id": img, "bracos": {}, "ordem": list(D["ordem"])}
    for bid in D["ordem"]:
        b = D["bracos"][bid]
        if not b["enc"]:
            out["bracos"][bid] = {"erro": "não treinado", "nome": b["nome"]}
            continue
        antes = S["encoder"]
        S["encoder"] = b["enc"]
        try:
            r = _prever(img)
        finally:
            S["encoder"] = antes
        r["nome"] = b["nome"]
        out["bracos"][bid] = r
    return out


# ── HTTP ─────────────────────────────────────────────────────────────────────

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):        # silencia o log por requisição
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        try:
            if p in ("/", "/index.html"):
                html = (Path(__file__).parent / "index.html").read_text(
                    encoding="utf-8")
                return self._send(200, html, "text/html; charset=utf-8")
            if p == "/api/datasets":
                return self._send(200, {"datasets": DS.listar(),
                                        "atual": S["ds"]})
            if p == "/api/lote":
                return self._send(200, {k: v for k, v in LOTE.items()
                                        if k != "cancelar"})
            if p == "/api/estado":
                _init_conjuntos()
                return self._send(200, {
                    "T": sorted(S["markers"]), "pool": S["pool"],
                    "val": len(S["val"]), "history": S["history"],
                    "decoder": S["decoder"], "block": S["block"],
                    "decoders": [{"id": a, "nome": b} for a, b in DECODERS],
                    "treinado": S["encoder"] is not None,
                    "alvo": S["alvo"], "k_alvo": S["k_alvo"],
                    "ds": S["ds"], "ds_nome": _cfg()["nome"],
                    "tem_label": bool(_label()),
                    "boot": BOOT,
                    "criterio": S["criterio"],
                    "criterios": [{"id": a, "nome": b, "nota": c}
                                  for a, b, c in CRITERIOS],
                })
            if p.startswith("/fig/"):
                fp = REPO / "figs" / os.path.basename(p)
                if not fp.exists():
                    return self._send(404, {"erro": "figura ausente"})
                return self._send(200, fp.read_bytes(), "image/png")
            if p.startswith("/img/"):
                img_id = p.split("/")[-1].replace(".png", "")
                with open(os.path.join(_orig(), _arquivo(img_id)), "rb") as fh:
                    return self._send(200, fh.read(), "image/png")
            if p.startswith("/gt/"):
                img_id = p.split("/")[-1].replace(".png", "")
                lp = os.path.join(_label(), f"{img_id}.png")
                if not os.path.exists(lp):
                    return self._send(404, {"erro": "sem anotação"})
                gt = np.array(Image.open(lp).convert("L")) > 0
                rgba = np.zeros((*gt.shape, 4), dtype=np.uint8)
                borda = gt & ~(np.roll(gt, 1, 0) & np.roll(gt, -1, 0)
                               & np.roll(gt, 1, 1) & np.roll(gt, -1, 1))
                rgba[borda] = (255, 0, 217, 255)
                return self._send(200, json.dumps({"mask": _png(rgba)}))
            return self._send(404, {"erro": "rota desconhecida"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"erro": str(e)})

    def do_POST(self):
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if p == "/api/marker":
                img_id = body["id"]
                m = S["markers"].setdefault(img_id, {"fg": [], "bg": []})
                m["fg"] += [[int(c), int(r)] for c, r in body.get("fg", [])]
                m["bg"] += [[int(c), int(r)] for c, r in body.get("bg", [])]
                return self._send(200, {"fg": len(m["fg"]), "bg": len(m["bg"]),
                                        "T": sorted(S["markers"])})
            if p == "/api/comp/marker":
                return self._send(200, acao_comp_marker(
                    body["id"], body.get("fg", []), body.get("bg", [])))
            if p == "/api/comp/limpar":
                return self._send(200, acao_comp_limpar(body["id"]))
            if p == "/api/comp/confirmar":
                return self._send(200, acao_comp_confirmar(body["id"]))
            if p == "/api/comp/resultado":
                # A tabela vive so no cliente; recarregar a pagina a perdia,
                # e retreinar cinco bracos para reve-la custa minutos. O
                # resultado ja esta em memoria -- basta devolve-lo.
                return self._send(200, D.get("resultado") or {"vazio": True})
            if p == "/api/comp/estado":
                return self._send(200, {
                    "boot": BOOT,
                    "ativo": D["ativo"], "rodada": D["rodada"], "k": D["k"],
                    "fila": D["fila"],
                    # A fila detalhada e o que a interface precisa para RETOMAR
                    # uma rodada depois de um F5: sem "pedida_por", "motivo" e
                    # os centros propostos, a tela sabe que ha imagens
                    # pendentes mas nao consegue abrir nenhuma.
                    "fila_detalhe": _fila_detalhe(),
                    "bracos": _resumo_bracos(),
                    "confirmadas": sorted(D["confirmadas"]),
                    "sem_par": [i for i, m in D["markers"].items()
                                if not m["fg"] or not m["bg"]],
                    "markers": {i: {"fg": len(m["fg"]), "bg": len(m["bg"]),
                                    "proposto": m["proposto"]}
                                for i, m in D["markers"].items()}})
            if p == "/api/limpar":
                S["markers"].pop(body.get("id"), None)
                return self._send(200, {"T": sorted(S["markers"])})
            if p == "/api/config":
                if "decoder" in body:
                    S["decoder"] = body["decoder"]
                if "block" in body:
                    S["block"] = int(body["block"])
                if "criterio" in body and body["criterio"] in CRITERIO_IDS:
                    S["criterio"] = body["criterio"]
                return self._send(200, {"decoder": S["decoder"],
                                        "block": S["block"],
                                        "criterio": S["criterio"]})
            if p == "/api/dataset":
                novo = body["id"]
                DS.resolver(novo)                 # valida antes de trocar
                S["ds"] = novo
                S["val"] = []
                # A comparacao guarda NOMES de imagem. Mantida entre datasets,
                # ela passa a apontar para arquivos de outro conjunto -- e como
                # nomes se repetem (000008 existe em mais de um), o erro e
                # silencioso: treina com a imagem errada sem reclamar.
                D.update({"ativo": False, "bracos": {}, "ordem": [],
                          "markers": {}, "centros": {}, "confirmadas": set(),
                          "fila": [], "rodada": 0, "sugestor": None,
                          "semente": None, "resultado": None})
                _init_conjuntos(forcar=True)
                return self._send(200, {"ds": S["ds"], "nome": _cfg()["nome"],
                                        "pool": S["pool"], "val": len(S["val"]),
                                        "tem_label": bool(_label())})
            if p == "/api/upload/criar":
                return self._send(200, {"id": DS.criar_upload(
                    body.get("nome", "meu dataset"), body.get("existente"))})
            if p == "/api/upload/validacao":
                n = DS.marcar_validacao(body["id"], body.get("nomes", []))
                if body["id"] == S["ds"]:
                    _init_conjuntos(forcar=True)
                return self._send(200, {"id": body["id"], "n_validacao": n,
                                        "val": len(S["val"]),
                                        "pool": len(S["pool"])})
            if p == "/api/upload/arquivo":
                # uma imagem por requisição: mantém o corpo pequeno e permite
                # barra de progresso honesta no cliente
                d = DS.resolver(body["id"])
                nome = os.path.basename(body["nome"])
                dados = base64.b64decode(body["b64"].split(",")[-1])
                destino = os.path.join(
                    d["orig"] if body.get("tipo", "orig") == "orig"
                    else os.path.join(os.path.dirname(d["orig"]), "label"),
                    nome)
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                with open(destino, "wb") as fh:
                    fh.write(dados)
                return self._send(200, {"ok": True, "nome": nome})
            if p == "/api/upload/apagar":
                return self._send(200, {"ok": DS.apagar_upload(body["id"])})
            if p == "/api/lote/iniciar":
                if LOTE["rodando"]:
                    return self._send(200, {"erro": "já está rodando"})
                cfg = {"ds": body.get("ds", S["ds"]),
                       "criterios": ["coreset"],
                       "ks": [int(k) for k in body.get("ks", [3, 5, 8])],
                       "decoder": body.get("decoder", S["decoder"]),
                       "n_test": int(body.get("n_test", 60)),
                       "semente": int(body.get("semente", 0))}
                threading.Thread(target=_lote_thread, args=(cfg,),
                                 daemon=True).start()
                return self._send(200, {"iniciado": True, "config": cfg})
            if p == "/api/lote/cancelar":
                LOTE["cancelar"] = True
                return self._send(200, {"ok": True})
            if p == "/api/k_alvo":
                S["k_alvo"] = max(1, int(body.get("k", 3)))
                return self._send(200, {"k_alvo": S["k_alvo"]})
            if p in ("/api/treinar", "/api/sugerir", "/api/prever",
                     "/api/avaliar", "/api/proxima", "/api/validar/iniciar",
                     "/api/validar/responder", "/api/comp/iniciar",
                     "/api/comp/proxima", "/api/comp/treinar",
                     "/api/comp/prever", "/api/comp/premarcar",
                     "/api/regiao/iniciar", "/api/regiao/responder",
                     "/api/onde/comparar", "/api/comp/dica"):
                # A resposta é montada dentro do lock mas ENVIADA fora dele.
                # Enviando dentro, o cliente recebe e dispara a requisição
                # seguinte antes de o `finally` liberar a flag — e leva um 409
                # espúrio. O lock serializa; requisições concorrentes esperam
                # em vez de falhar, que é o certo para uma app de um usuário.
                with LOCK:
                    S["busy"] = True
                    try:
                        if p == "/api/comp/iniciar":
                            resultado = acao_comp_iniciar(
                                int(body.get("k", 3)),
                                body.get("criterios") or ["coreset"])
                        elif p == "/api/comp/proxima":
                            resultado = acao_comp_proxima()
                        elif p == "/api/comp/dica":
                            resultado = acao_comp_dica(
                                body["id"], int(body.get("k", 5)))
                        elif p == "/api/comp/premarcar":
                            resultado = acao_comp_premarcar(body["id"])
                        elif p == "/api/comp/treinar":
                            resultado = acao_comp_treinar(
                                int(body.get("sorteios", 3)))
                        elif p == "/api/comp/prever":
                            resultado = acao_comp_prever(body["id"])
                        elif p == "/api/onde/comparar":
                            # Compara ONDE anotar, nao QUAIS imagens anotar.
                            # Roda varios treinos seguidos, por isso entra no
                            # mesmo lock das acoes longas.
                            from flim_app import onde as ONDE
                            # `sys.modules[__name__]` e o modulo VIVO deste
                            # servidor, que roda como __main__. Passar o nome
                            # em vez de deixar o `onde` importar evita que ele
                            # receba uma copia nova, de estado vazio.
                            resultado = ONDE.comparar(
                                sys.modules[__name__],
                                body.get("bracos"),
                                int(body.get("n_segments", 220)),
                                int(body.get("semente", 0)))
                        elif p == "/api/regiao/iniciar":
                            resultado = acao_regiao_iniciar(
                                body.get("id"), int(body.get("k", 6)))
                        elif p == "/api/regiao/responder":
                            resultado = acao_regiao_responder(
                                body.get("resposta", "pular"))
                        elif p == "/api/validar/iniciar":
                            resultado = acao_validar_iniciar(
                                int(body.get("n", 12)))
                        elif p == "/api/validar/responder":
                            resultado = acao_validar_responder(
                                bool(body.get("tem_objeto")))
                        elif p == "/api/proxima":
                            resultado = acao_proxima()
                        elif p == "/api/treinar":
                            resultado = acao_treinar()
                        elif p == "/api/avaliar":
                            resultado = acao_avaliar()
                        elif p == "/api/sugerir":
                            resultado = acao_sugerir()
                        elif not S["encoder"]:
                            resultado = {"erro": "treine primeiro"}
                        else:
                            resultado = _prever(body["id"])
                    finally:
                        S["busy"] = False
                return self._send(200, resultado)
            return self._send(404, {"erro": "rota desconhecida"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"erro": str(e)})


def main() -> int:
    disponiveis = DS.listar()
    if not disponiveis:
        print("ERRO: nenhum dataset encontrado. Rode de dentro de flim_ad/.")
        return 1
    if not os.path.isdir(_orig()):
        S["ds"] = disponiveis[0]["id"]
    print("datasets:", ", ".join(f"{d['id']}({d['n_imagens']})"
                                 for d in disponiveis))
    _init_conjuntos()
    print(f"val={len(S['val'])} pool={len(S['pool'])} · "
          f"http://localhost:8000")
    ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
