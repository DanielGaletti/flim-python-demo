#!/usr/bin/env python3
"""
onde_marcar.py — o AL escolhe ONDE anotar, com orçamento contado em pixels

A pergunta
    O FLIM treina com poucas imagens. Se o Active Learning escolher QUAIS
    imagens anotar, o orçamento cresce em imagens e a premissa do FLIM se
    desfaz. A alternativa preserva a premissa inteira: mantém as imagens
    fixas — as do artigo — e escolhe ONDE traçar dentro delas.

Por que o orçamento é em pixels, e não em imagens
    Medido neste projeto (`flim_al/tabelas.py`, nota de `k_por_modelo`): a
    arquitetura pede 200 kernels por camada, mas o k-means só produz tantos
    quantos os patches dos markers permitem. Uma imagem com traço esparso
    (~4 mil pixels marcados) rende 54/54/48/48 kernels; a MESMA imagem com
    traço denso (~23 mil) rende 200/200/183/183. O que limita o FLIM não é o
    número de imagens, é o número de pixels anotados.

    A consequência para o desenho: um braço que anote mais densamente ganharia
    de graça se o orçamento fosse contado em imagens. Aqui todos os braços
    gastam EXATAMENTE o mesmo número de pixels, e é isso que torna a
    comparação uma comparação de *onde*, não de *quanto*.

Os braços
    uniforme          traços espalhados pela imagem — é o FLIM de hoje
    regiao_incerteza  um quarto do orçamento em traço uniforme, o resto nos
                      superpixels onde o modelo treinado nesse quarto está
                      mais incerto
    regiao_aleatoria  mesma geometria, mesmo quarto inicial, superpixels
                      sorteados — CONTROLE
    regiao_borda      superpixels que cruzam a borda do objeto no gabarito —
                      TETO, o que se ganharia sabendo onde está a fronteira
    uniforme_balanceado
                      TOQUES espalhados ao acaso, com a MESMA proporção
                      objeto/fundo que a incerteza produziu — CONTROLE do
                      balanço de classes. São toques, e não pixels soltos,
                      porque pixel solto gera patch 3×3 único e o encoder
                      colapsa: a primeira versão deu Fβ 0,000

O confundidor que este desenho precisa isolar
    Medido antes de rodar, com 3.000 px por imagem: o traço uniforme rende
    ~16% de pixels de objeto, e o braço de incerteza rende ~73%. Não é acaso —
    é sobre o objeto que o modelo fica incerto, então escolher regiões
    incertas escolhe, junto, muito mais foreground.

    Isso torna "regiao_incerteza vs regiao_aleatoria" insuficiente: os dois
    diferem em ONDE e em QUANTO de cada classe. `uniforme_balanceado` copia a
    proporção sem copiar a localização. Se a incerteza ganhar dele, a
    localização importa além do balanço; se empatar, o efeito era o balanço —
    o que continua sendo um achado útil, e bem mais barato de explorar.

    Sem `regiao_aleatoria` não dá para afirmar que a seleção funcionou:
    anotar por superpixel já muda a distribuição dos patches que alimentam o
    k-means, e essa mudança sozinha move o Fβ. O controle separa o efeito da
    geometria do efeito da escolha.

    `regiao_borda` usa o gabarito e por isso NÃO é um método — é referência.
    Corresponde ao papel que o `oracle` tem na seleção de imagens.

O especialista simulado
    Dentro de um superpixel escolhido, o rótulo de cada pixel vem do gabarito.
    É a simulação da pergunta que o professor descreveu: o sistema aponta a
    região confusa e o especialista responde "isto é fundo" ou "isto é
    objeto".

Uso
    python scripts/onde_marcar.py --plano
    python scripts/onde_marcar.py --px 2000 4000 8000 --sementes 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from flim_al import evidencia as ev  # noqa: E402
from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, retrain_encoder,
)
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

EXPERIMENTO = "onde_marcar"

DECODERS = [
    ("labeled_marker", "FLIM_lm"),
    ("decoder_2",      "FLIM_pb"),
]

# As imagens de treino do artigo. Nenhuma seleção de imagem acontece aqui —
# este experimento é sobre onde traçar, não sobre quais imagens usar.
IMAGENS_ARTIGO = ["000002", "000013", "000156", "000391", "000405"]

BRACOS = ["uniforme", "regiao_incerteza", "regiao_aleatoria",
          "regiao_borda", "uniforme_balanceado"]

# Fração do orçamento gasta antes de qualquer escolha. Os braços de região
# precisam de um modelo para pontuar, e o modelo precisa de anotação; esse
# quarto inicial é idêntico nos três, então a diferença entre eles vem só de
# onde os três quartos restantes foram colocados.
FRACAO_INICIAL = 0.25


def _arquivo(cfg, img: str) -> str:
    for ext in (".png", ".jpg", ".jpeg"):
        if os.path.exists(os.path.join(cfg["orig"], img + ext)):
            return img + ext
    return img + ".png"


def _semente_marker(img: str, semente: int) -> int:
    """Semente do traço, estável por imagem — ver `tabela_k_por_modelo`."""
    return int(hashlib.sha256(f"{semente}:{img}".encode()).hexdigest()[:8], 16)


def _tem_objeto(ds_id: str, img: str) -> bool:
    lp = DS.caminho_label(ds_id, img + ".png") or DS.caminho_label(ds_id, img + ".jpg")
    if not lp:
        return False
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


# ── contagem de pixels ──────────────────────────────────────────────────────

def _subamostra(fg, bg, n_alvo, rng):
    """
    Corta para exatamente `n_alvo` pixels, preservando a proporção fg:bg.

    Cortar sem preservar a proporção mudaria o balanço de classes junto com o
    orçamento, e aí duas coisas variariam ao mesmo tempo.
    """
    total = len(fg) + len(bg)
    if total <= n_alvo:
        return fg, bg
    n_fg = max(1, round(n_alvo * len(fg) / total)) if len(fg) else 0
    n_bg = n_alvo - n_fg
    if n_bg > len(bg):
        n_bg, n_fg = len(bg), n_alvo - len(bg)
    fg = [fg[i] for i in rng.permutation(len(fg))[:n_fg]]
    bg = [bg[i] for i in rng.permutation(len(bg))[:n_bg]]
    return fg, bg


def _uniforme(gt_path, n_alvo, semente):
    """
    Traço espalhado, escalado até cobrir o orçamento e cortado no alvo.

    O gerador trabalha em "dabs" (toques de pincel), não em pixels, e o número
    de pixels que um dab rende depende do tamanho do objeto. Então a escala
    sobe até passar do alvo e o corte final acerta o número exato.
    """
    rng = np.random.default_rng(semente)
    escala = 1.0
    d = None
    for _ in range(12):
        d = generate_realistic_markers(
            gt_path, n_fg_dabs=max(1, round(6 * escala)),
            n_bg_dabs=max(1, round(14 * escala)), seed=semente)
        if len(d["fg_seeds"]) + len(d["bg_seeds"]) >= n_alvo:
            break
        escala *= 1.6
    fg = [tuple(int(v) for v in p) for p in d["fg_seeds"]]
    bg = [tuple(int(v) for v in p) for p in d["bg_seeds"]]
    return _subamostra(fg, bg, n_alvo, rng)


def _superpixels(img_path, n_segments):
    from skimage.segmentation import slic
    im = np.array(Image.open(img_path).convert("RGB"))
    return slic(im, n_segments=n_segments, compactness=10.0, sigma=1.0,
                start_label=0, convert2lab=True).astype(np.int32)


def _ordem_incerteza(seg, prob):
    """Superpixels da maior para a menor entropia média do modelo."""
    ent = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
    if ent.shape != seg.shape:
        ent = np.array(Image.fromarray(ent).resize(
            (seg.shape[1], seg.shape[0]), Image.BILINEAR))
    ids = np.unique(seg)
    media = [(s, float(ent[seg == s].mean())) for s in ids]
    return [s for s, _ in sorted(media, key=lambda kv: -kv[1])]


def _ordem_borda(seg, gt):
    """
    Superpixels que contêm objeto E fundo, dos mais equilibrados aos menos.

    Um superpixel meio a meio está exatamente sobre a fronteira; um com 95% de
    fundo apenas encosta nela. Ordenar pelo equilíbrio põe a borda primeiro.
    """
    ids = np.unique(seg)
    pont = []
    for s in ids:
        m = seg == s
        f = float(gt[m].mean())
        if 0.0 < f < 1.0:
            pont.append((s, abs(f - 0.5)))
    return [s for s, _ in sorted(pont, key=lambda kv: kv[1])]


def _pixels_de_regioes(seg, ordem, gt, n_alvo, rng):
    """
    Gasta o orçamento percorrendo os superpixels na ordem dada.

    Todos os pixels de um superpixel entram antes de passar ao próximo: é o que
    um especialista faz ao pintar a região que o sistema apontou.
    """
    fg, bg = [], []
    for s in ordem:
        ys, xs = np.nonzero(seg == s)
        for i in rng.permutation(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            (fg if gt[y, x] else bg).append((x, y))
            if len(fg) + len(bg) >= n_alvo:
                return fg, bg
    return fg, bg


# ── um braço, uma semente, um orçamento ─────────────────────────────────────

def _escrever(dest, por_imagem):
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for img, (fg, bg, H, W) in por_imagem.items():
        save_markers({"fg_seeds": fg, "bg_seeds": bg, "H": H, "W": W},
                     os.path.join(dest, f"{img}-seeds.txt"))
        n += len(fg) + len(bg)
    return n


def _amostra_balanceada(gt, n_alvo, n_fg_alvo, rng, raio: int = 5):
    """
    `n_alvo` pixels com `n_fg_alvo` no objeto, em TOQUES e não em pixels soltos.

    A primeira versão sorteava pixels isolados pela imagem inteira, e colapsou:
    Fβ 0,000 nos dois decoders, com o treino levando 812 s contra 9 s dos
    braços de região. A causa é a mesma que explica todo este experimento —
    pixel isolado gera um patch 3×3 que não repete em lugar nenhum, então o
    k-means recebe milhares de patches únicos e nenhuma estrutura.

    Como controle aquilo era pior que inútil: mudava o balanço de classes E a
    coerência espacial ao mesmo tempo, então o Δ contra ele não media
    localização. Aqui os toques são discos de raio `raio`, a mesma geometria do
    pincel real — muda só ONDE caem, que é o que este braço existe para
    controlar.
    """
    disco = [(dx, dy) for dy in range(-raio, raio + 1)
             for dx in range(-raio, raio + 1) if dx * dx + dy * dy <= raio * raio]
    H, W = gt.shape

    def toques(mascara, alvo):
        ys, xs = np.nonzero(mascara)
        if not len(ys) or alvo <= 0:
            return []
        pontos, vistos = [], set()
        for i in rng.permutation(len(ys)):
            cy, cx = int(ys[i]), int(xs[i])
            for dx, dy in disco:
                x, y = cx + dx, cy + dy
                # O rótulo do toque é o da classe que ele veio representar; um
                # disco centrado no objeto pode encostar no fundo, e incluir
                # esse pixel poria rótulo errado no marker.
                if 0 <= x < W and 0 <= y < H and mascara[y, x] and (x, y) not in vistos:
                    vistos.add((x, y))
                    pontos.append((x, y))
            if len(pontos) >= alvo:
                break
        return pontos[:alvo]

    n_fg = min(n_fg_alvo, int(gt.sum()))
    fg = toques(gt, n_fg)
    bg = toques(~gt, n_alvo - len(fg))
    return fg, bg


def _markers_do_braco(braco, imagens, cfg, ds_id, px, semente, work,
                      device, n_segments):
    """
    Devolve (diretório de markers, pixels gastos).

    Todos os braços gastam `px` por imagem. O que muda é onde esses pixels
    caem — e, no caso de `uniforme_balanceado`, só a proporção entre classes.
    """
    if braco == "uniforme_balanceado":
        # Precisa saber quanto de objeto a incerteza produziu para copiar a
        # proporcao. Roda aquele braco primeiro e le os arquivos dele.
        d_i, _ = _markers_do_braco("regiao_incerteza", imagens, cfg, ds_id,
                                   px, semente, work, device, n_segments)
        alvo = {}
        for img in imagens:
            linhas = open(os.path.join(d_i, f"{img}-seeds.txt")).read(
                ).strip().split("\n")[1:]
            alvo[img] = sum(1 for l in linhas if l.split()[3] == "1")
        rng = np.random.default_rng(semente * 104729 + px)
        por_img = {}
        for img in imagens:
            lp = DS.caminho_label(ds_id, img + ".png")
            g = np.array(Image.open(lp).convert("L")) > 127
            fg, bg = _amostra_balanceada(g, px, alvo[img], rng)
            por_img[img] = (fg, bg, g.shape[0], g.shape[1])
        d = os.path.join(work, f"mk_{braco}_{px}_{semente}")
        return d, _escrever(d, por_img)

    gts, shapes = {}, {}
    for img in imagens:
        lp = DS.caminho_label(ds_id, img + ".png")
        g = np.array(Image.open(lp).convert("L")) > 127
        gts[img], shapes[img] = g, g.shape

    if braco == "uniforme":
        por_img = {}
        for img in imagens:
            fg, bg = _uniforme(DS.caminho_label(ds_id, img + ".png"), px,
                               _semente_marker(img, semente))
            H, W = shapes[img]
            por_img[img] = (fg, bg, H, W)
        d = os.path.join(work, f"mk_{braco}_{px}_{semente}")
        return d, _escrever(d, por_img)

    # Braços de região: um quarto do orçamento em traço uniforme, igual nos
    # três, e o resto onde cada um decidir.
    px0 = max(1, int(px * FRACAO_INICIAL))
    inicial = {}
    for img in imagens:
        fg, bg = _uniforme(DS.caminho_label(ds_id, img + ".png"), px0,
                           _semente_marker(img, semente))
        H, W = shapes[img]
        inicial[img] = (fg, bg, H, W)
    d0 = os.path.join(work, f"mk0_{px}_{semente}")
    _escrever(d0, inicial)

    prob = {}
    if braco == "regiao_incerteza":
        enc0 = os.path.join(work, f"e0_{px}_{semente}.pth")
        retrain_encoder(cfg["arch"], d0, cfg["orig"], cfg["label"], device,
                        enc0)
        from flim_app import server as SV
        SV.S["ds"] = ds_id
        SV.S["block"] = cfg["bloco"]
        SV.S["decoder"] = "labeled_marker"
        SV.S["encoder"] = enc0
        modelo = SV._montar_modelo(enc0)
        for img in imagens:
            prob[img] = SV._mapa_prob(modelo, img)

    rng = np.random.default_rng(semente * 7919 + px)
    por_img = {}
    for img in imagens:
        seg = _superpixels(os.path.join(cfg["orig"], _arquivo(cfg, img)),
                           n_segments)
        if braco == "regiao_incerteza":
            ordem = _ordem_incerteza(seg, prob[img])
        elif braco == "regiao_borda":
            ordem = _ordem_borda(seg, gts[img])
        else:
            ordem = list(rng.permutation(np.unique(seg)))
        fg0, bg0, H, W = inicial[img]
        fg1, bg1 = _pixels_de_regioes(seg, ordem, gts[img], px - px0, rng)
        por_img[img] = (list(fg0) + fg1, list(bg0) + bg1, H, W)

    d = os.path.join(work, f"mk_{braco}_{px}_{semente}")
    return d, _escrever(d, por_img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--k", type=int, default=3,
                    help="quantas imagens do artigo; NAO varia entre bracos")
    ap.add_argument("--px", nargs="+", type=int, default=[2000, 4000, 8000],
                    help="orcamento de pixels anotados POR IMAGEM")
    ap.add_argument("--bracos", nargs="+", default=BRACOS)
    ap.add_argument("--sementes", nargs="+", type=int, default=[0])
    ap.add_argument("--semente_particao", type=int, default=0)
    ap.add_argument("--n_pool", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--n_segments", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--registro", default=None,
                    help="CSV alternativo; use para rodar em paralelo a outra "
                         "campanha sem que as duas se atropelem no registro")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()
    arquivo = a.registro or ev.ARQUIVO

    if "orcamento_px" not in ev.CAMPOS:
        print("ERRO: o registro ainda nao tem o campo `orcamento_px`.\n"
              "Este experimento mede orcamento em PIXELS, e gravar sem o campo\n"
              "faria o numero entrar sem a condicao que o define.")
        return 2

    cfg = DS.resolver(a.ds)
    cfg["_id"] = a.ds
    cfg["bloco"] = cfg.get("bloco", 2)

    imagens = IMAGENS_ARTIGO[:a.k]

    # Mesma partição da campanha de seleção de imagens, para os dois
    # experimentos serem medidos no MESMO conjunto de teste.
    todas = [os.path.splitext(f)[0] for f in DS.imagens(a.ds)]
    rng = np.random.default_rng(a.semente_particao)
    com_obj = [i for i in todas[:600] if _tem_objeto(a.ds, i)]
    emb = rng.permutation(len(com_obj))
    test = sorted(com_obj[i] for i in emb[a.n_pool:a.n_pool + a.n_test])
    test = [t for t in test if t not in imagens]

    n_cel = len(a.bracos) * len(a.px) * len(a.sementes)
    print(f"dataset {a.ds} · {len(imagens)} imagens fixas {imagens}")
    print(f"teste {len(test)} imagens")
    print(f"{len(a.bracos)} braços × {len(a.px)} orçamentos × "
          f"{len(a.sementes)} sementes = {n_cel} células")
    print(f"{n_cel * len(DECODERS)} avaliações de decoder")
    print("orçamento em PIXELS anotados por imagem:", a.px)
    print("braços:", ", ".join(a.bracos))
    if a.plano:
        print("\n--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="onde_")
    fonte = (f"{EXPERIMENTO}/k{len(imagens)}/teste{len(test)}"
             f"/seg{a.n_segments}")
    print(f"fonte: {fonte}\n")
    print(f"registro: {arquivo}")

    registros, falhas = [], []
    for semente in a.sementes:
        for px in a.px:
            for braco in a.bracos:
                try:
                    mdir, gastos = _markers_do_braco(
                        braco, imagens, cfg, a.ds, px, semente, work,
                        a.device, a.n_segments)
                    enc = os.path.join(work, f"e_{braco}_{px}_{semente}.pth")
                    t0 = time.time()
                    retrain_encoder(cfg["arch"], mdir, cfg["orig"],
                                    cfg["label"], a.device, enc)
                    t_treino = time.time() - t0
                except Exception as e:                        # noqa: BLE001
                    falhas.append(f"{braco} px={px} s={semente}: "
                                  f"{type(e).__name__}: {e}")
                    print(f"  s{semente} {braco:<18} px={px}  FALHOU: {e}",
                          flush=True)
                    continue

                print(f"  s{semente} {braco:<18} px={px}  "
                      f"gastos {gastos:6d}  treino {t_treino:5.1f}s",
                      end="", flush=True)
                novos = []
                for dec, nome in DECODERS:
                    t1 = time.time()
                    try:
                        m = evaluate_decoder(
                            enc, dec, cfg["bloco"],
                            [_arquivo(cfg, i) for i in test],
                            cfg["orig"], cfg["label"], a.device,
                            area_range=tuple(cfg["area"]))
                    except Exception as e:                    # noqa: BLE001
                        falhas.append(f"{braco} px={px} {dec}: {e}")
                        continue
                    t_aval = time.time() - t1
                    novos.append(ev.execucao(
                        experimento=EXPERIMENTO, dataset=a.ds,
                        braco="flim" if braco == "uniforme" else "al",
                        # O orcamento entra na `variante` porque ela e campo
                        # identificador e `orcamento_px` nao e: dois
                        # orcamentos do mesmo braco e semente colidiriam no
                        # run_id sem isto.
                        variante=f"{braco}:px{px}", criterio=braco,
                        decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
                        orcamento=len(imagens), orcamento_px=px,
                        seed=semente, imagens=imagens,
                        marker_origem="sintetico",
                        fb=m["fb"], dice=m["dice"], iou=m.get("iou"),
                        mae=m["mae"], segundos=round(t_treino + t_aval, 2),
                        segundos_treino=round(t_treino, 2),
                        segundos_aval=round(t_aval, 2),
                        fonte=fonte,
                    ))
                if novos:
                    res = ev.registrar(novos, arquivo=arquivo)
                    registros.extend(novos)
                    if res["divergentes"]:
                        print(f"\n  DIVERGENCIA em "
                              f"{len(res['divergentes'])} run_id(s)",
                              flush=True)
                print(f"  ·  {len(novos)} decoders  "
                      f"[{len(registros)} registrados]", flush=True)

    if falhas:
        print(f"\n{len(falhas)} falha(s):")
        for f in falhas[:8]:
            print(f"  {f}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nregistrados: {len(registros)}")
    print("\nAgora: python scripts/gerar_tabelas.py && python -m pytest")
    return 1 if falhas else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrompido")
