#!/usr/bin/env python3
"""
tabela_k_por_modelo.py — Fβ e tempo por (critério × K × decoder)

O que esta campanha responde
    Para cada orçamento K de imagens anotadas, e para cada um dos sete
    decoders do artigo: quanto cada critério de seleção entrega, quanto custa
    treinar e quanto custa avaliar — contra o braço do FLIM do artigo, que usa
    as imagens fixas que os autores escolheram.

A otimização que torna isso viável
    O encoder FLIM **não depende do decoder**: os kernels saem do k-means
    sobre os patches dos markers, e o decoder é uma convolução 1×1 aplicada
    depois. Treinar uma vez por (critério, K) e avaliar os sete decoders
    naquele mesmo encoder é 7× mais barato que treinar por decoder — e dá
    exatamente o mesmo número.

    Por isso `t_treino` aparece uma vez por encoder e `t_aval` por decoder. Os
    dois são reportados separados porque medem coisas diferentes: o treino é o
    que o FLIM promete ser barato (k-means, sem backpropagação); a avaliação
    custa o que custaria em qualquer rede e cresce com o conjunto de teste.

Marker sintético em TODOS os braços, inclusive no do artigo
    Marker real só existe para 31 imagens do Schisto. Se o braço do artigo
    usasse traço real e os de AL usassem sintético, o Δ mediria quem desenhou
    o traço — foi assim que uma versão anterior deste projeto produziu
    "AL perde do FLIM por −0,29 com p<0,0001", número real e conclusão falsa.

    Aqui o gerador é o mesmo em todos os braços. A comparação é internamente
    válida; o que ela NÃO permite é comparar com números produzidos a partir
    de markers reais.

Uso
    python scripts/tabela_k_por_modelo.py --plano
    python scripts/tabela_k_por_modelo.py --ks 3 5 8 --criterios coreset random
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import traceback

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "flim_ad", "libs", "flim-python"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from flim_al import evidencia as ev  # noqa: E402
from flim_al.al_encoder_experiment import (  # noqa: E402
    evaluate_decoder, retrain_encoder,
)
from flim_al.marker_generator import save_markers  # noqa: E402
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402
from flim_al.paper_selection import imagem_medoide  # noqa: E402
from flim_app import datasets as DS  # noqa: E402

EXPERIMENTO = "tabela_k_por_modelo"

DECODERS = [
    ("labeled_marker",              "FLIM_lm"),
    ("decoder_2",                   "FLIM_pb"),
    ("decoder_3",                   "FLIM_mb"),
    ("decoder_attention",           "FLIM_at"),
    ("vanilla_adaptive_decoder",    "FLIM_ts"),
    ("hybrid_decoder",              "FLIM_lt"),
    ("vanilla_adaptive_decoder_wt", "FLIM_ts*"),
]

# As cinco imagens de treino do artigo, de split1-train.txt. O braço
# `flim_paper` usa exatamente estas, sem seleção automática nenhuma — é a
# referência contra a qual "AL melhora?" se responde.
IMAGENS_ARTIGO = ["000002", "000013", "000156", "000391", "000405"]

CRITERIOS = ["flim_paper", "random", "medoide", "coreset", "entropy",
             "least_confidence"]


def _tem_objeto(ds_id, img) -> bool:
    from PIL import Image
    lp = DS.caminho_label(ds_id, img + ".png") or DS.caminho_label(ds_id, img + ".jpg")
    if not lp:
        return False
    return bool((np.array(Image.open(lp).convert("L")) > 0).any())


def _arquivo(cfg, img) -> str:
    for ext in (".png", ".jpg", ".jpeg"):
        if os.path.exists(os.path.join(cfg["orig"], img + ext)):
            return img + ext
    return img + ".png"


def _semente_marker(img: str, semente: int) -> int:
    """
    Semente do traço, estável por imagem.

    A versão anterior sorteava com `rng.integers()` dentro do laço, avançando o
    gerador a cada arquivo: a mesma imagem recebia traços diferentes conforme
    quais outras estivessem no conjunto e em que ordem. Dois braços que
    escolhessem a mesma imagem anotavam ela de formas diferentes, e essa
    diferença entrava no Fβ como se fosse efeito da seleção.

    Derivar do id da imagem elimina isso: dentro de uma semente de campanha, a
    imagem 000123 recebe sempre o mesmo traço, em qualquer braço e em qualquer
    rodada. É o que `paper_selection.py` já fazia com `_stable_seed`.
    """
    h = hashlib.sha256(f"{semente}:{img}".encode()).hexdigest()
    return int(h[:8], 16)


def _markers(cfg, ds_id, sel, dest, semente) -> int:
    """Gera markers sintéticos — o MESMO processo em todos os braços."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in sel:
        lp = DS.caminho_label(ds_id, f + ".png")
        if not lp:
            continue
        d = generate_realistic_markers(lp, n_fg_dabs=6, n_bg_dabs=14,
                                       seed=_semente_marker(f, semente))
        save_markers(d, os.path.join(dest, f"{f}-seeds.txt"))
        n += 1
    return n


def _escolhe_uma(criterio, candidatos, sel, pool, cfg, enc) -> str:
    """A próxima imagem, segundo o encoder ATUAL."""
    if criterio == "coreset":
        from flim_al.coreset_badge import (coreset_select,
                                           extract_encoder_features)
        import torch
        e = torch.load(enc, map_location="cpu", weights_only=False)
        feats = extract_encoder_features(
            e, [os.path.join(cfg["orig"], _arquivo(cfg, i)) for i in pool],
            cfg["bloco"], "cpu")
        # `labeled_indices` informa o que já está anotado, para o k-center
        # medir distância ao conjunto existente em vez de recomeçar do zero.
        idx = coreset_select(feats, budget=1,
                             labeled_indices=[pool.index(x) for x in sel])
        return pool[idx[0]]

    from flim_app import server as SV
    SV.S["ds"] = cfg.get("_id", "schisto")
    SV.S["block"] = cfg["bloco"]
    SV.S["decoder"] = "labeled_marker"
    SV.S["encoder"] = enc
    chave = "entropia" if criterio == "entropy" else "least_confidence"
    pontos = SV._pontuar_pool(candidatos, enc)
    return max(pontos, key=lambda r: r[chave])["id"]


def _seleciona(criterio, pool, k, cfg, enc0, rng, work, ds_id, device,
               semente, modo) -> list:
    """
    As K imagens a anotar.

    `flim_paper`, `random` e `medoide` não consultam modelo nenhum, então não
    há laço: a escolha é a mesma feita de uma vez ou uma a uma.

    Os outros três são Active Learning de verdade só no modo `iterativa`:
    escolhe uma, retreina o encoder com o que já foi anotado, repontua o pool,
    escolhe a próxima. O modo `lote` pontua uma vez com o encoder inicial e
    pega o top-K — é o que este script fazia antes, mantido para que a
    comparação entre os dois seja ela mesma um resultado.
    """
    if criterio == "flim_paper":
        return IMAGENS_ARTIGO[:k]
    if criterio == "random":
        return [str(x) for x in rng.choice(pool, size=min(k, len(pool)),
                                           replace=False)]
    if criterio == "medoide":
        # Determinístico: a mais típica, depois as mais típicas restantes.
        sel, resto = [], list(pool)
        for _ in range(min(k, len(resto))):
            m = imagem_medoide([_arquivo(cfg, i) for i in resto], cfg["orig"])
            m = os.path.splitext(m)[0]
            sel.append(m)
            resto.remove(m)
        return sel

    if modo == "lote":
        from flim_al.coreset_badge import (coreset_select,
                                           extract_encoder_features)
        import torch
        if criterio == "coreset":
            e = torch.load(enc0, map_location="cpu", weights_only=False)
            feats = extract_encoder_features(
                e, [os.path.join(cfg["orig"], _arquivo(cfg, i)) for i in pool],
                cfg["bloco"], "cpu")
            return [pool[i] for i in coreset_select(feats,
                                                    budget=min(k, len(pool)))]
        from flim_app import server as SV
        SV.S["ds"] = cfg.get("_id", "schisto")
        SV.S["block"] = cfg["bloco"]
        SV.S["decoder"] = "labeled_marker"
        SV.S["encoder"] = enc0
        chave = "entropia" if criterio == "entropy" else "least_confidence"
        pontos = sorted(SV._pontuar_pool(pool, enc0),
                        key=lambda r: -r[chave])
        return [p["id"] for p in pontos[:k]]

    sel, enc = [], enc0
    for rodada in range(min(k, len(pool))):
        cand = [x for x in pool if x not in sel]
        sel.append(_escolhe_uma(criterio, cand, sel, pool, cfg, enc))
        if len(sel) >= k:
            break
        # Retreina com o que já está anotado. É este passo que torna o laço
        # ativo: a pontuação da próxima rodada sai de um modelo que já viu o
        # que foi escolhido até aqui.
        md = os.path.join(work, f"mit_{criterio}_{k}_{len(sel)}")
        if _markers(cfg, ds_id, sel, md, semente) == 0:
            break
        enc = os.path.join(work, f"eit_{criterio}_{k}_{len(sel)}.pth")
        retrain_encoder(cfg["arch"], md, cfg["orig"], cfg["label"], device, enc)
    return sel


def _uma_semente(a, semente, cfg, pool, test, work, enc0, fonte, decoders):
    """
    Uma repeticao completa: todos os criterios, todos os K, todos os decoders.

    Esta era a parte de dentro de `main`. Virou funcao para que a campanha
    ganhe varias sementes sem reindentar o corpo inteiro -- reindentacao em
    massa e como se introduz erro silencioso num script que ja produziu
    resultado.
    """
    registros, falhas = [], []
    for criterio in a.criterios:
        for k in a.ks:
            if criterio == "flim_paper" and k > len(IMAGENS_ARTIGO):
                continue
            r = np.random.default_rng(semente * 1000 + k)
            try:
                sel = _seleciona(criterio, pool, k, cfg, enc0, r,
                                 work, a.ds, a.device,
                                 semente, a.selecao)
                mdir = os.path.join(work, f"m_{criterio}_{k}_{semente}")
                if _markers(cfg, a.ds, sel, mdir, semente) == 0:
                    raise ValueError("nenhum marker gerado")
                enc = os.path.join(work, f"e_{criterio}_{k}_{semente}.pth")
                t0 = time.time()
                retrain_encoder(cfg["arch"], mdir, cfg["orig"], cfg["label"],
                                a.device, enc)
                t_treino = time.time() - t0
            except Exception as e:                            # noqa: BLE001
                falhas.append(f"{criterio} K={k}: {type(e).__name__}: {e}")
                print(f"  {criterio:<18} K={k}  FALHOU: {e}", flush=True)
                continue

            print(f"  s{semente} {criterio:<18} K={k}  treino {t_treino:5.1f}s", end="",
                  flush=True)
            novos_aqui = []
            for dec, nome in decoders:
                t1 = time.time()
                try:
                    m = evaluate_decoder(
                        enc, dec, cfg["bloco"],
                        [_arquivo(cfg, i) for i in test],
                        cfg["orig"], cfg["label"], a.device,
                        area_range=tuple(cfg["area"]))
                except Exception as e:                        # noqa: BLE001
                    falhas.append(f"{criterio} K={k} {dec}: {e}")
                    continue
                t_aval = time.time() - t1
                novos_aqui.append(ev.execucao(
                    experimento=EXPERIMENTO, dataset=a.ds,
                    braco="flim_paper" if criterio == "flim_paper" else "al",
                    variante=f"{criterio}:{a.selecao}", criterio=criterio,
                    decoder=dec, decoder_paper=nome, bloco=cfg["bloco"],
                    orcamento=k, seed=semente, imagens=sel,
                    marker_origem="sintetico",
                    fb=m["fb"], dice=m["dice"], iou=m.get("iou"),
                    mae=m["mae"], segundos=round(t_treino + t_aval, 2),
                    segundos_treino=round(t_treino, 2),
                    segundos_aval=round(t_aval, 2),
                    fonte=fonte,
                ))
            # Registra a cada encoder concluido, nao so no fim da campanha.
            #
            # A primeira versao acumulava tudo e gravava depois do ultimo
            # laco. Numa campanha de uma hora isso significa que qualquer
            # interrupcao -- Ctrl+C, falta de memoria, reinicio -- descarta
            # todo o trabalho ja medido. Gravar a cada passo custa
            # milissegundos e torna a campanha retomavel.
            if novos_aqui:
                res = ev.registrar(novos_aqui)
                registros.extend(novos_aqui)
                # Divergencia = mesmo run_id, metrica diferente. Significa que
                # algo mudou FORA do que o registro identifica. Silenciar isso
                # foi exatamente o que escondeu a colisao do smoke test.
                if res["divergentes"]:
                    print(f"\n  DIVERGENCIA em {len(res['divergentes'])} "
                          f"run_id(s): {res['divergentes'][:3]}", flush=True)
            print(f"  ·  {len(novos_aqui)} decoders", flush=True)

    return registros, falhas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="schisto")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 8])
    ap.add_argument("--criterios", nargs="+", default=CRITERIOS)
    ap.add_argument("--n_pool", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--sementes", nargs="+", type=int, default=[0],
                    help="repeticoes; cada uma varia tracos e braco aleatorio")
    ap.add_argument("--semente_particao", type=int, default=0,
                    help="fixa pool e teste; NAO varia entre repeticoes")
    ap.add_argument("--selecao", choices=["iterativa", "lote"],
                    default="iterativa",
                    help="iterativa retreina a cada escolha; lote pega top-K "
                         "de uma vez com o encoder inicial")
    ap.add_argument("--decoders", nargs="+", default=None,
                    help="nomes de codigo; padrao e os sete do artigo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plano", action="store_true")
    a = ap.parse_args()

    decoders = DECODERS if not a.decoders else [
        d for d in DECODERS if d[0] in a.decoders]
    if not decoders:
        print(f"nenhum decoder casa com {a.decoders}")
        return 2

    cfg = DS.resolver(a.ds)
    cfg["_id"] = a.ds
    cfg["bloco"] = cfg.get("bloco", 2)

    todas = [os.path.splitext(f)[0] for f in DS.imagens(a.ds)]
    # A particao nao depende da semente da campanha: todas as repeticoes
    # medem na mesma populacao de teste, e a diferenca pareada isola a selecao.
    rng = np.random.default_rng(a.semente_particao)

    # Pool e teste só com imagens que TÊM objeto. No Schisto 49% não têm, e
    # nelas o Fβ é 1.0 por definição (vazio-vazio): a métrica ficaria dominada
    # por acertos triviais e a curva não se moveria. Está declarado.
    com_obj = [i for i in todas[:600] if _tem_objeto(a.ds, i)]
    emb = rng.permutation(len(com_obj))
    pool = sorted(com_obj[i] for i in emb[:a.n_pool])
    test = sorted(com_obj[i] for i in emb[a.n_pool:a.n_pool + a.n_test])
    # As imagens do artigo não podem estar no teste.
    test = [t for t in test if t not in IMAGENS_ARTIGO]

    n_enc = len(a.criterios) * len(a.ks) * len(a.sementes)
    print(f"dataset {a.ds} · pool {len(pool)} · teste {len(test)}")
    print(f"{len(a.criterios)} critérios × {len(a.ks)} K × "
          f"{len(a.sementes)} sementes = {n_enc} encoders")
    print(f"{n_enc * len(decoders)} avaliações de decoder")
    print(f"seleção: {a.selecao}")
    print(f"marker: sintético em TODOS os braços (mesmo gerador)\n")
    print("critérios:", ", ".join(a.criterios))
    print("K:", a.ks)
    if a.plano:
        print(f"\nestimativa: ~{n_enc * 40 / 60:.0f} min de treino + "
              f"~{n_enc * len(DECODERS) * 8 / 60:.0f} min de avaliação")
        print("--plano: nada executado.")
        return 0

    work = tempfile.mkdtemp(prefix="tabk_")
    # Encoder inicial (medoide), ponto de partida dos critérios que precisam
    # de um modelo para pontuar o pool.
    mdir0 = os.path.join(work, "m0")
    _markers(cfg, a.ds, [os.path.splitext(
        imagem_medoide([_arquivo(cfg, i) for i in pool], cfg["orig"]))[0]],
        mdir0, a.semente_particao)
    enc0 = os.path.join(work, "e0.pth")
    retrain_encoder(cfg["arch"], mdir0, cfg["orig"], cfg["label"], a.device, enc0)

    # A identidade da campanha inclui em que conjunto se mediu. Sem isso, um
    # smoke test com teste menor colide com a campanha completa no run_id e a
    # substitui em silencio -- foi o que ocorreu na primeira execucao.
    fonte = (f"{EXPERIMENTO}/pool{len(pool)}/teste{len(test)}"
             f"/{a.selecao}")
    print(f"fonte: {fonte}\n")

    registros, falhas = [], []
    for semente in a.sementes:
        r_s, f_s = _uma_semente(a, semente, cfg, pool, test, work, enc0,
                                fonte, decoders)
        registros.extend(r_s)
        falhas.extend(f_s)
        print(f"  semente {semente}: {len(registros)} registrados no total",
              flush=True)

    if registros:
        print(f"\nregistrados: {len(registros)} no total")
    if falhas:
        print(f"\n{len(falhas)} falha(s):")
        for f in falhas[:8]:
            print(f"  {f}")
    shutil.rmtree(work, ignore_errors=True)

    print("\nAgora: python scripts/gerar_tabelas.py && python -m pytest")
    return 1 if falhas else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrompido")
        raise SystemExit(130)
