#!/usr/bin/env python3
"""
onde.py — comparação de ONDE anotar, com o MESMO orçamento de pixels
====================================================================

A pergunta que esta tela responde é diferente da do `duplo.py`. Lá a disputa é
sobre **quais imagens** anotar; aqui as imagens são as mesmas em todos os
braços, e a disputa é sobre **onde traçar dentro delas**.

Os braços

    voce        o traço que você desenhou — o especialista escolhe o lugar
    modelo      o modelo propõe os pontos sozinho: objeto na borda da PREDIÇÃO
                dele, fundo bem longe. É a pré-marcação da outra tela
    al_regiao   Active Learning escolhe a região: superpixels onde a entropia
                do modelo é maior, e o rótulo vem do gabarito, simulando o
                especialista respondendo "objeto" ou "fundo"
    aleatorio   superpixels sorteados, mesmo rótulo simulado — CONTROLE

Por que o controle não é opcional
    Anotar por superpixel já muda a distribuição dos patches 3×3 que alimentam
    o k-means, e essa mudança sozinha move o Fβ. Sem `aleatorio` não dá para
    dizer se o ganho veio de *escolher bem a região* ou apenas de *anotar em
    blocos*. Comparar só `al_regiao` contra `voce` mediria as duas coisas
    somadas.

Por que o orçamento é contado em pixels
    Medido neste projeto: a arquitetura pede 200 kernels por camada, mas o
    k-means só produz tantos quantos os patches dos markers permitem. Traço
    esparso rende 54 kernels; traço denso rende 200. Um braço que marque mais
    pixels ganharia por isso, não por escolher melhor o lugar.

    Então todos os braços são cortados para o MESMO número de pixels por
    imagem — o menor entre eles, para que nenhum precise inventar pontos. O
    número usado aparece na resposta.

Uma advertência que a tela precisa repetir
    Isto é uma execução única, num conjunto de validação pequeno. Serve para
    VER onde cada estratégia marca e ter uma intuição do efeito. O número que
    entra na dissertação vem de `scripts/onde_marcar.py`, com sementes e teste
    pareado.
"""
from __future__ import annotations

import io
import os
import shutil
import time

import numpy as np
from PIL import Image

from flim_al.al_encoder_experiment import evaluate_decoder, retrain_encoder
from flim_al.marker_generator import save_markers

BRACOS = ["voce", "modelo", "al_regiao", "aleatorio"]

ROTULO = {
    "voce": "Você desenhou",
    "modelo": "O modelo propôs",
    "al_regiao": "AL escolheu a região",
    "aleatorio": "Região sorteada (controle)",
}


def _png(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay(h: int, w: int, fg, bg) -> str:
    """Onde cada braço marcou: objeto em verde, fundo em vermelho."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for pts, cor in ((bg, (232, 84, 63, 220)), (fg, (30, 158, 74, 220))):
        for c, r in pts:
            if 0 <= r < h and 0 <= c < w:
                rgba[r, c] = cor
    return _png(rgba)


def _corta(fg, bg, n_alvo, rng):
    """
    Corta para `n_alvo` pixels preservando a proporção objeto:fundo.

    Cortar sem preservar a proporção mudaria o balanço de classes junto com o
    orçamento, e aí duas coisas variariam de uma vez.
    """
    total = len(fg) + len(bg)
    if total <= n_alvo:
        return list(fg), list(bg)
    n_fg = max(1, round(n_alvo * len(fg) / total)) if len(fg) else 0
    n_bg = n_alvo - n_fg
    if n_bg > len(bg):
        n_bg, n_fg = len(bg), n_alvo - len(bg)
    fg = [fg[i] for i in rng.permutation(len(fg))[:n_fg]]
    bg = [bg[i] for i in rng.permutation(len(bg))[:n_bg]]
    return fg, bg


def _pixels_de_regioes(sp, ordem, gt, n_alvo, rng):
    """
    Gasta o orçamento percorrendo os superpixels na ordem dada.

    Todos os pixels de um superpixel entram antes de passar ao próximo: é o que
    um especialista faz ao responder "esta região é fundo" — ele rotula a
    região inteira, não um ponto dela. O rótulo de cada pixel sai do gabarito,
    que aqui faz o papel da resposta humana.
    """
    fg, bg = [], []
    for s in ordem:
        ys, xs = np.nonzero(sp == s)
        if not len(ys):
            continue
        for i in rng.permutation(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            (fg if gt[y, x] else bg).append((x, y))
            if len(fg) + len(bg) >= n_alvo:
                return fg, bg
    return fg, bg


def _gt(label_dir: str, img_id: str, h: int, w: int):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(label_dir, img_id + ext)
        if os.path.isfile(p):
            g = np.array(Image.open(p).convert("L")) > 127
            if g.shape != (h, w):
                g = np.array(Image.fromarray(g.astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            return g
    return None


def comparar(SV, bracos=None, n_segments: int = 220,
             semente: int = 0) -> dict:
    """
    Roda os braços pedidos sobre as MESMAS imagens e devolve Fβ de cada um.

    `SV` é o módulo do servidor, passado pelo chamador em vez de importado.

    Não é preciosismo: o servidor é iniciado como `python flim_app/server.py`,
    então ele roda com o nome `__main__`. Um `from flim_app import server`
    daqui criaria uma SEGUNDA cópia do módulo, com `S` vazio — e a comparação
    respondia "anote ao menos uma imagem" com a imagem anotada na tela. O
    chamador sabe qual módulo está vivo; este não tem como saber.
    """
    from flim_al.region_al import compute_superpixels, score_regions_by_entropy

    bracos = [b for b in (bracos or BRACOS) if b in BRACOS]
    if not SV.S["markers"]:
        return {"erro": "anote ao menos uma imagem antes de comparar"}
    if not SV.S["encoder"]:
        return {"erro": "treine ao menos uma vez — os braços `modelo` e "
                        "`al_regiao` precisam de um modelo para decidir onde"}
    if not SV.S["val"]:
        return {"erro": "nenhum conjunto de validação definido"}

    imagens = sorted(SV.S["markers"])
    rng = np.random.default_rng(semente)
    cfg, orig, label = SV._cfg(), SV._orig(), SV._label()

    # ── 1. pontos crus de cada braço, por imagem ────────────────────────────
    cru: dict[str, dict[str, tuple]] = {b: {} for b in bracos}
    formas: dict[str, tuple] = {}
    avisos: list[str] = []

    for img in imagens:
        with Image.open(os.path.join(orig, SV._arquivo(img))) as im:
            w, h = im.size
        formas[img] = (h, w)
        gt = _gt(label, img, h, w)

        m = SV.S["markers"][img]
        if "voce" in bracos:
            cru["voce"][img] = ([tuple(p) for p in m["fg"]],
                                [tuple(p) for p in m["bg"]])

        if "modelo" in bracos:
            p = SV._pontos_propostos(img, SV.S["encoder"])
            cru["modelo"][img] = ([tuple(x) for x in p["fg"]],
                                  [tuple(x) for x in p["bg"]])

        if gt is None:
            if any(b in bracos for b in ("al_regiao", "aleatorio")):
                avisos.append(f"{img}: sem gabarito, braços de região "
                              "ficaram de fora")
            for b in ("al_regiao", "aleatorio"):
                cru.get(b, {}).pop(img, None)
            continue

        arr = np.array(Image.open(os.path.join(orig, SV._arquivo(img))))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        sp = compute_superpixels(arr[:, :, :3], n_segments=n_segments)

        if "al_regiao" in bracos:
            modelo = SV._montar_modelo(SV.S["encoder"])
            prob = SV._mapa_prob(modelo, img)
            if prob.shape != (h, w):
                prob = np.array(
                    Image.fromarray((prob * 255).astype(np.uint8))
                    .resize((w, h), Image.BILINEAR)) / 255.0
            # `score_regions_by_entropy` devolve {id_da_regiao: entropia}.
            # A ordem é da maior entropia para a menor: primeiro onde o modelo
            # está mais em dúvida.
            scores = score_regions_by_entropy(sp, prob)
            ordem = [rid for rid, _ in sorted(scores.items(),
                                              key=lambda kv: -kv[1])]
            cru["al_regiao"][img] = (sp, ordem, gt)

        if "aleatorio" in bracos:
            ordem = list(rng.permutation(np.unique(sp)))
            cru["aleatorio"][img] = (sp, ordem, gt)

    # ── 2. orçamento igual: o menor total entre os braços, por imagem ───────
    #
    # Cortar para o menor evita que qualquer braço precise inventar pontos que
    # sua estratégia não produziria. O número fica na resposta porque ele é
    # metade da leitura: um Fβ maior com o dobro de pixels não é comparação.
    orcamento: dict[str, int] = {}
    for img in imagens:
        totais = []
        for b in bracos:
            v = cru[b].get(img)
            if v is None:
                continue
            if isinstance(v[0], np.ndarray):      # braço de região: ilimitado
                continue
            totais.append(len(v[0]) + len(v[1]))
        if totais:
            orcamento[img] = max(1, min(totais))

    # ── 3. materializa, treina e avalia cada braço ─────────────────────────
    val = [SV._arquivo(i) for i in SV.S["val"]]
    saida = []
    base = os.path.join(SV.S["work"], f"onde_{int(time.time())}")

    for b in bracos:
        d = os.path.join(base, b)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        overlays, gasto = {}, 0
        for img in imagens:
            v = cru[b].get(img)
            if v is None or img not in orcamento:
                continue
            alvo = orcamento[img]
            h, w = formas[img]
            if isinstance(v[0], np.ndarray):
                sp, ordem, gt = v
                fg, bg = _pixels_de_regioes(sp, ordem, gt, alvo, rng)
            else:
                fg, bg = _corta(v[0], v[1], alvo, rng)
            fg = [(int(c), int(r)) for c, r in fg if 0 <= c < w and 0 <= r < h]
            bg = [(int(c), int(r)) for c, r in bg if 0 <= c < w and 0 <= r < h]
            if not fg and not bg:
                continue
            save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": h, "W": w},
                         os.path.join(d, f"{img}-seeds.txt"))
            overlays[img] = _overlay(h, w, fg, bg)
            gasto += len(fg) + len(bg)

        item = {"braco": b, "rotulo": ROTULO[b], "px": gasto,
                "overlays": overlays}
        if gasto == 0:
            item["erro"] = "nenhum ponto gerado"
            saida.append(item)
            continue
        try:
            t0 = time.time()
            enc = os.path.join(base, f"{b}.pth")
            retrain_encoder(cfg["arch"], d, orig, label, "cpu", enc)
            t_treino = time.time() - t0
            m = evaluate_decoder(enc, SV.S["decoder"], SV.S["block"], val,
                                 orig, label, "cpu", area_range=SV._area())
            item.update({"fb": round(float(m["fb"]), 4),
                         "iou": round(float(m["iou"]), 4),
                         "t_treino": round(t_treino, 1)})
        except Exception as e:                                # noqa: BLE001
            item["erro"] = f"{type(e).__name__}: {e}"
        saida.append(item)

    medidos = [s for s in saida if "fb" in s]
    ref = next((s for s in saida if s["braco"] == "voce" and "fb" in s), None)
    for s in medidos:
        s["delta"] = (None if ref is None or s is ref
                      else round(s["fb"] - ref["fb"], 4))

    return {
        "imagens": imagens,
        "n_val": len(SV.S["val"]),
        "orcamento": orcamento,
        "decoder": SV.S["decoder"],
        "bracos": saida,
        "avisos": avisos,
        "melhor": (max(medidos, key=lambda s: s["fb"])["braco"]
                   if medidos else None),
    }
