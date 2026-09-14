#!/usr/bin/env python3
"""
paper_selection.py
==================
Seleção ITERATIVA de imagens representativas para o encoder FLIM, no molde do
Algoritmo 1 de Soares et al. (arXiv:2504.20872), com o critério de escolha
trocável.

Por que este experimento
------------------------
O Algoritmo 1 do paper já é um laço de active learning — mas **supervisionado**:

    1  T ← {imagem aleatória}
    3  enquanto o usuário não estiver satisfeito:
    4      treina uma CNN FLIM sobre T
    5      avalia em Z1\\T
    6      x ← Fβ médio em Z1\\T
    7      entre as imagens de MENOR Fβ, escolhe a próxima z
    8      se x < x_anterior:  remove a última            (backtracking)
   11      senão:              T ← T ∪ {z}

O passo 7 exige ground truth de TODO o pool de validação. Isso derrota o
propósito do FLIM: se houvesse GT de 848 imagens, não haveria por que anotar
apenas 3 com markers.

A pergunta desta dissertação passa a ser bem-posta:

    quanto do benefício do Algoritmo 1 se recupera com um critério que NÃO
    usa ground truth nenhum para selecionar?

Papéis:
    oracle   — Algoritmo 1 do paper (usa GT do pool)  → TETO
    entropy / coreset / badge / bald  — sem GT        → proposta
    random                                            → PISO

O laço é honestamente iterativo: a cada rodada o encoder é retreinado e as
saliências do pool são REGERADAS com o encoder atual, então o critério
enxerga o estado corrente do modelo — diferente da seleção one-shot usada nos
experimentos anteriores.

Restrição de anotação
---------------------
Só existem markers REAIS para ~31 imagens do dataset. Com `--pool real`, a
seleção acontece apenas entre elas: pequeno, mas sem nenhum marker sintético,
que é o confundidor que degradou o Experimento A. Com `--pool full`, o pool é
o Z1 inteiro e as imagens escolhidas recebem markers sintéticos — mais amplo,
porém confundido. Os dois modos gravam a coluna `marker_source`.

Uso:
    cd flim_ad
    python ../flim_al/paper_selection.py --criterion oracle  --splits 1 2 3
    python ../flim_al/paper_selection.py --criterion entropy --splits 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

import torch  # noqa: E402

from flim_al.al_encoder_experiment import (
    image_to_lab,  # noqa: E402
    _official_filter_and_binarize,
    evaluate_decoder,
    generate_pool_saliencies,
    retrain_encoder,
)
from flim_al.marker_generator import (  # noqa: E402
    _stable_seed,
    generate_markers_from_gt,
    save_markers,
)
from flim_al.realistic_markers import generate_realistic_markers  # noqa: E402

CRITERIA = ["oracle", "entropy", "least_confidence", "coreset", "badge",
            "random"]


# ── fontes de markers reais ───────────────────────────────────────────────────

def real_marker_index(repo: Path) -> dict[str, str]:
    """
    Mapeia image_id → caminho do seeds.txt real, varrendo todas as fontes.
    Imagens anotadas de fato pelos usuários durante o Algoritmo 1 do paper.
    """
    idx: dict[str, str] = {}
    patterns = [
        repo / "data" / "markers" / "*-seeds.txt",
        repo / "data" / "markers_oracle" / "*-seeds.txt",
        repo / "flim_ad" / "data" / "schisto" / "user_*" / "split*" / "markers" / "*-seeds.txt",
    ]
    for pat in patterns:
        for p in glob.glob(str(pat)):
            img_id = os.path.basename(p).replace("-seeds.txt", "")
            idx.setdefault(img_id, p)
    return idx


# ── métricas por imagem ───────────────────────────────────────────────────────

def per_image_fb(sal_dir: str, fnames: list[str], label_folder: str,
                 beta2: float = 0.3,
                 area_range: tuple[int, int] = (1000, 9000)) -> dict[str, float]:
    """
    Fβ por imagem, a partir de saliências já geradas — replica o
    pós-processamento oficial (Otsu + filtro de área).

    É o insumo do passo 7 do Algoritmo 1: "entre as imagens de menor Fβ".
    """
    eps = 1e-8
    out: dict[str, float] = {}
    for fn in fnames:
        sp, lp = os.path.join(sal_dir, fn), os.path.join(label_folder, fn)
        if not (os.path.exists(sp) and os.path.exists(lp)):
            continue
        pred = _official_filter_and_binarize(np.array(Image.open(sp)), area_range)
        g = (np.array(Image.open(lp).convert("L")) > 0).astype(np.uint8)
        if g.sum() == 0 and pred.sum() == 0:
            out[fn] = 1.0
            continue
        tp = float((pred & g).sum())
        fp = float((pred & ~g.astype(bool)).sum())
        fn_ = float((~pred.astype(bool) & g).sum())
        pr, rc = tp / (tp + fp + eps), tp / (tp + fn_ + eps)
        out[fn] = (1 + beta2) * pr * rc / (beta2 * pr + rc + eps)
    return out


def pool_entropy(sal_dir: str, fnames: list[str]) -> dict[str, float]:
    """Entropia binária média do saliency map. Não usa ground truth."""
    eps = 1e-6
    out: dict[str, float] = {}
    for fn in fnames:
        p = os.path.join(sal_dir, fn)
        if not os.path.exists(p):
            continue
        a = np.clip(np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0,
                    eps, 1 - eps)
        out[fn] = float((-(a * np.log(a) + (1 - a) * np.log(1 - a))).mean())
    return out


def pool_least_confidence(sal_dir: str, fnames: list[str]) -> dict:
    """
    Least confidence médio do saliency map: 1 - |p - 0.5| * 2. Sem ground truth.

    Mede a mesma coisa que a entropia — o quanto o modelo está no muro — com
    uma curva diferente. A entropia pesa mais o que está perto de 0.5; o least
    confidence é linear na distância até 0.5. Em pools onde a saliência é quase
    toda saturada, os dois ordenam igual; onde há uma cauda de valores
    intermediários, divergem.

    Existe aqui porque estava implementado só na aplicação interativa
    (`flim_app/server.py`), e comparar critério em execução única de demo não
    é evidência. Para entrar numa tabela ele precisa do mesmo caminho dos
    outros: este script, com seeds e teste comum.
    """
    out: dict = {}
    for fn in fnames:
        p = os.path.join(sal_dir, fn)
        if not os.path.exists(p):
            continue
        a = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        out[fn] = float((1.0 - np.abs(a - 0.5) * 2.0).mean())
    return out


# ── critérios de escolha da próxima imagem ────────────────────────────────────

def pick_next(criterion: str, candidates: list[str], sal_dir: str,
              label_folder: str, orig_folder: str, enc_path: str,
              selected: list[str], device: str, rng: random.Random,
              proxy_layer: int = 3,
              area_range: tuple[int, int] = (1000, 9000)) -> tuple[str, str]:
    """
    Devolve (imagem escolhida, justificativa legível).
    Só `oracle` consulta ground truth.
    """
    if not candidates:
        raise ValueError("sem candidatos")

    if criterion == "random":
        z = rng.choice(candidates)
        return z, "sorteio"

    if criterion == "oracle":
        # Passo 7 do Algoritmo 1: entre as de MENOR Fβ
        fbs = per_image_fb(sal_dir, candidates, label_folder,
                           area_range=area_range)
        if not fbs:
            return rng.choice(candidates), "sem Fβ — sorteio"
        z = min(fbs, key=fbs.get)
        return z, f"pior Fβ = {fbs[z]:.4f}  [usa GT]"

    if criterion == "entropy":
        ents = pool_entropy(sal_dir, candidates)
        if not ents:
            return rng.choice(candidates), "sem saliência — sorteio"
        z = max(ents, key=ents.get)
        return z, f"maior entropia = {ents[z]:.4f}"

    if criterion == "least_confidence":
        lcs = pool_least_confidence(sal_dir, candidates)
        if not lcs:
            return rng.choice(candidates), "sem saliencia - sorteio"
        z = max(lcs, key=lcs.get)
        return z, f"maior least confidence = {lcs[z]:.4f}"

    if criterion in ("coreset", "badge"):
        from flim_al.coreset_badge import extract_encoder_features
        enc = torch.load(enc_path, map_location=device, weights_only=False)
        enc.eval()
        all_f = candidates + selected
        feats = extract_encoder_features(
            enc, [os.path.join(orig_folder, f) for f in all_f], proxy_layer, device)
        cand_f, sel_f = feats[:len(candidates)], feats[len(candidates):]

        if criterion == "coreset":
            # k-center: a mais distante do conjunto já selecionado
            d = np.full(len(candidates), np.inf, dtype=np.float64)
            for s in sel_f:
                d = np.minimum(d, np.sum((cand_f - s) ** 2, axis=1))
            i = int(np.argmax(d))
            return candidates[i], f"mais distante de T (d² = {d[i]:.1f})"

        # badge: distância × incerteza (saliência média como proxy da predição)
        ents = pool_entropy(sal_dir, candidates)
        d = np.full(len(candidates), np.inf, dtype=np.float64)
        for s in sel_f:
            d = np.minimum(d, np.sum((cand_f - s) ** 2, axis=1))
        d = d / (d.max() + 1e-12)
        u = np.array([ents.get(f, 0.0) for f in candidates])
        u = u / (u.max() + 1e-12)
        score = d * u
        i = int(np.argmax(score))
        return candidates[i], f"diversidade × incerteza = {score[i]:.4f}"

    raise ValueError(f"critério desconhecido: {criterion}")


# ── configuração por dataset ─────────────────────────────────────────────────
#
# O BraTS entra aqui por duas razões. Primeiro, a Tabela I do artigo não aplica
# Dynamic Trees a ele, então a comparação com as Tabelas III/IV não tem etapa
# faltando. Segundo, ele tem marker real para TODAS as 3753 imagens — no
# Schisto são 31 markers reais para 1220 imagens, e o pool completo só existia
# com markers sintéticos. O experimento de seleção sai sem essa muleta.

DATASETS = {
    "schisto": {
        "orig": "datasets/schistossoma-eggs/orig",
        "label": "datasets/schistossoma-eggs/label",
        "val": "datasets/schistossoma-eggs/Splits-5train-70_30/split{s}-val.txt",
        "test": "datasets/schistossoma-eggs/Splits-5train-70_30/test.txt",
        "arch": "data/{markers}/split{s}/arch2D.json",
        "area": (1000, 9000),
        "markers_glob": None,          # varre as três fontes de marker real
    },
    # ATENÇÃO: caminhos ABSOLUTOS, de propósito. O FLIMData decide se
    # acrescenta a extensão com `len(image_path.split(".")) == 1`, e o ".." de
    # um caminho relativo introduz dois pontos que quebram esse teste — o
    # ".png" deixa de ser acrescentado e o imread falha com o arquivo sem
    # extensão. Ver flim_ad/libs/flim-python/pyflim/data.py.
    "brats": {
        "orig": str(REPO / "data" / "brats" / "orig"),
        "label": str(REPO / "data" / "brats" / "label"),
        "val": str(REPO / "data" / "brats" / "Splits-50_50" / "split{s}-val.txt"),
        "test": str(REPO / "data" / "brats" / "Splits-50_50" / "split{s}-test.txt"),
        "arch": str(REPO / "arch_best_brats.json"),
        "area": (100, 20000),
        "markers_glob": str(REPO / "data" / "brats" / "markers" / "*-seeds.txt"),
    },
}



def imagem_medoide(pool: list[str], orig_folder: str) -> str:
    """
    A imagem mais "típica" do pool, escolhida SEM modelo nenhum.

    Por que isto existe: medimos que o desvio do Fβ final entre execuções do
    Algoritmo 1 é de 0.10 a 0.22, contra 0.009 entre partições dos dados. A
    fonte dominante de variância é o sorteio do passo 1 — e é a única entrada
    do algoritmo que não precisa ser aleatória.

    Não dá para usar features do encoder aqui: na primeira rodada não existe
    encoder. O descritor é portanto direto dos pixels — a imagem reduzida a
    32×32 em LAB — e o medoide é quem minimiza a distância média às demais.
    Determinístico, sem ground truth, custo desprezível.
    """
    from PIL import Image as _Im
    descs = []
    for f in pool:
        arr = np.array(_Im.open(os.path.join(orig_folder, f))
                       .convert("RGB").resize((32, 32)))
        descs.append(image_to_lab(arr.astype(np.uint8)).ravel())
    X = np.stack(descs)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return pool[int(d.sum(axis=1).argmin())]


def descarta_degenerados(cands: list[str], sal_dir: str,
                         area_range: tuple[int, int]) -> list[str]:
    """
    Remove candidatas cuja saliência prevista é vazia após o pós-processamento.

    Estas são as imagens que o critério "pior Fβ" do artigo prioriza: elas têm
    Fβ = 0 não por serem informativas, mas porque o filtro de área zera tudo.
    No Schisto são 49% do pool. O teste não usa ground truth — só olha se o
    próprio modelo enxerga alguma coisa.
    """
    vivas = []
    for f in cands:
        sp = os.path.join(sal_dir, f)
        if not os.path.exists(sp):
            vivas.append(f)
            continue
        if _official_filter_and_binarize(np.array(Image.open(sp)),
                                         area_range).sum() > 0:
            vivas.append(f)
    return vivas or cands          # nunca devolve lista vazia


# ── laço principal (Algoritmo 1 generalizado) ─────────────────────────────────

def run_selection(criterion: str, split: int, args, rng: random.Random) -> list[dict]:
    """
    Executa o laço iterativo para um split e devolve o histórico por rodada.
    """
    markers = args.markers
    cfg = DATASETS[args.dataset]
    area = cfg["area"]
    orig_folder = cfg["orig"]
    label_folder = cfg["label"]
    arch_file = cfg["arch"].format(markers=markers, s=split)

    val_list = cfg["val"].format(s=split)
    with open(val_list) as fh:
        val_fnames = [l.strip() for l in fh if l.strip()]

    if cfg["markers_glob"]:
        real_idx = {os.path.basename(q).replace("-seeds.txt", ""): q
                    for q in glob.glob(cfg["markers_glob"])}
    else:
        real_idx = real_marker_index(REPO)

    # Pool e conjunto de teste: onde cortar o vazamento.
    #
    # Vazamento é um problema de AVALIAÇÃO, não de seleção. Basta que o teste
    # não contenha nada que algum braço treinou. Tirar as mesmas imagens do
    # POOL é uma correção a mais que custa caro: no Schisto o T do artigo do
    # split 1 é {000479, 000675, 000917}, e 000675/000917 estão em test.txt.
    # Excluí-las do pool proíbe o critério de escolher exatamente o que o
    # usuário escolheu — o braço do artigo treina nelas e o de AL não pode.
    # A comparação deixa de medir o critério e passa a medir o handicap.
    #
    # Por isso a política é por dataset:
    #   schisto  pool com as 31 (mesmo conjunto que o usuário tinha); o
    #            vazamento morre no teste, via montar_teste_comum.py
    #   brats    pool restrito a Z₁; ali os markers cobrem o dataset inteiro
    #            (1877 dos 3753 estão no teste) e restringir é o que define
    #            o pool, não uma correção sobre ele
    test_path = cfg.get("test", "").format(s=split)
    test_set = set()
    if test_path and os.path.exists(test_path):
        with open(test_path) as fh:
            test_set = {l.strip() for l in fh if l.strip()}
    excluir = test_set if not args.pool_keeps_test else set()

    if args.pool == "real":
        pool = sorted(f"{k}.png" for k in real_idx
                      if f"{k}.png" not in excluir)
        marker_source = "real"
        # No BraTS o pool "real" tem 3753 imagens e cada rodada do laço
        # recalcula saliências para todas. Amostrar mantém o custo viável sem
        # mudar a natureza do experimento — o que importa é que o critério
        # escolha entre candidatos com marker de verdade.
        if args.pool_cap and len(pool) > args.pool_cap:
            pool = sorted(random.Random(11 + split).sample(pool, args.pool_cap))
    else:
        # Pool completo: tudo que NÃO está no val (o val é o conjunto de
        # avaliação do laço). As escolhidas recebem markers sintéticos.
        val_set = set(val_fnames)
        pool = sorted(os.path.basename(p) for p in
                      glob.glob(os.path.join(orig_folder, "*.png"))
                      if os.path.basename(p) not in val_set
                      and os.path.basename(p) not in excluir)
        marker_source = f"synthetic_{args.marker_style}"

    # pool ∩ val = ∅. A ordem importa: retira-se do val tudo que está no pool,
    # e NÃO o contrário. Filtrar o pool contra o val original descartaria as
    # imagens com marker real que por acaso caíram no val — no split 1 isso
    # derrubava o pool de 31 para 7, jogando fora a maior parte da anotação
    # disponível sem necessidade.
    pool_set = set(pool)
    val_fnames = [f for f in val_fnames if f not in pool_set]

    # Subamostra FIXA do conjunto de avaliação usado dentro do laço.
    # O passo 6 do Algoritmo 1 precisa do Fβ médio a cada rodada, e avaliar
    # 800+ imagens por rodada domina o custo (evaluate_decoder roda em CPU).
    # A subamostra é sorteada uma vez por split, com semente fixa, então todos
    # os critérios enxergam exatamente o mesmo conjunto — a comparação entre
    # eles continua pareada. O Fβ final deve ser recomputado no val completo.
    val_full = list(val_fnames)
    if args.val_subsample and args.val_subsample < len(val_fnames):
        val_fnames = random.Random(7 + split).sample(val_fnames, args.val_subsample)

    if len(pool) < 2:
        print(f"  split {split}: pool com {len(pool)} imagens — pulando")
        return []

    print(f"\n{'=' * 62}\nSplit {split} | critério: {criterion} | "
          f"pool: {len(pool)} ({marker_source}) | val: {len(val_fnames)}\n{'=' * 62}")

    work = tempfile.mkdtemp(prefix=f"sel_{criterion}_s{split}_")
    history: list[dict] = []
    try:
        # Passo 1. O artigo sorteia; com --init medoide escolhe-se a imagem
        # mais típica do pool, removendo a maior fonte de variância do método.
        if args.init == "medoide":
            inicial = imagem_medoide(pool, orig_folder)
            print(f"  inicial (medoide, determinística): {inicial[:-4]}")
        else:
            inicial = rng.choice(pool)
        selected = [inicial]
        best_fb, last_added = -1.0, selected[0]
        sem_melhora = 0          # pioras consecutivas, para o --patience
        rejected: set[str] = set()   # tabu: imagens já testadas e removidas

        for rnd in range(1, args.max_images + 1):
            mdir = os.path.join(work, f"r{rnd}_markers")
            os.makedirs(mdir, exist_ok=True)
            for f in selected:
                img_id = f.replace(".png", "")
                if img_id in real_idx:
                    shutil.copy2(real_idx[img_id], os.path.join(mdir, f"{img_id}-seeds.txt"))
            if marker_source == "synthetic":
                # FIX-SAMEFILE: create_combined_marker_dir copia o conteúdo de
                # original_marker_dir para output_dir. Passando o MESMO diretório
                # nos dois, ele tentava copiar cada arquivo sobre si mesmo e
                # estourava SameFileError assim que mdir já tinha algum marker.
                # Aqui só precisamos gerar os sintéticos que faltam.
                for f in selected:
                    img_id = f.replace(".png", "")
                    dst = os.path.join(mdir, f"{img_id}-seeds.txt")
                    if os.path.exists(dst):
                        continue
                    gt_path = os.path.join(label_folder, f"{img_id}.png")
                    if not os.path.exists(gt_path):
                        continue
                    if args.marker_style == "realistic":
                        seeds = generate_realistic_markers(
                            gt_path, seed=_stable_seed(img_id))
                    else:
                        seeds = generate_markers_from_gt(
                            gt_path, n_fg=args.n_fg, n_bg=args.n_bg,
                            seed=_stable_seed(img_id))
                    save_markers(seeds, dst)
            n_mk = len([x for x in os.listdir(mdir) if x.endswith("-seeds.txt")])
            if n_mk == 0:
                print("    nenhum marker disponível — abortando split")
                break

            enc_path = os.path.join(work, f"r{rnd}_encoder.pth")
            retrain_encoder(arch_file, mdir, orig_folder, label_folder,
                            args.device, enc_path)

            m = evaluate_decoder(enc_path, args.decoder, args.proxy_layer,
                                 val_fnames, orig_folder, label_folder,
                                 args.device, area_range=area)
            fb = m["fb"]

            # Saliências do pool com o encoder ATUAL — o que torna o laço iterativo
            sal_dir = os.path.join(work, f"r{rnd}_sal")
            generate_pool_saliencies(enc_path, pool, orig_folder, args.device,
                                     sal_dir, args.decoder, args.proxy_layer)

            improved = fb > best_fb
            print(f"  |T|={len(selected)}  Fβ={fb:.4f}  "
                  f"{'melhorou' if improved else 'PIOROU → backtrack'}  "
                  f"T={[s[:-4] for s in selected]}")

            history.append({
                "split": split, "criterion": criterion, "round": rnd,
                "n_images": len(selected), "fb": round(fb, 4),
                "dice": round(m["dice"], 4), "iou": round(m["iou"], 4),
                "mae": round(m["mae"], 4), "improved": int(improved),
                "selected": "|".join(s[:-4] for s in selected),
                "marker_source": marker_source, "pool_size": len(pool),
                "val_used": len(val_fnames),
                "rejected": "|".join(sorted(r[:-4] for r in rejected)),
            })

            # Passos 8-12: backtracking do Algoritmo 1.
            #
            # FIX-CICLO: o pseudocódigo do paper devolve a imagem rejeitada ao
            # conjunto de candidatos. Como o critério do passo 7 é determinístico
            # ("a de menor Fβ"), ela volta a ser a escolhida na rodada seguinte —
            # o laço trava propondo sempre a mesma imagem. Observado no split 1:
            # 000013 (Fβ=0, insegmentável pelo filtro de área) foi proposta,
            # rejeitada e proposta de novo.
            # O passo 7 diz "select a NEW image", o que sugere a intenção mas não
            # está formalizado. Mantemos aqui uma lista de rejeitados (tabu).
            # PACIÊNCIA (nosso acréscimo, `--patience`, padrão 0 = artigo).
            #
            # O passo 8 retrocede na PRIMEIRA piora. Isso pressupõe que uma
            # queda de Fβ é sinal de que a imagem não presta. Mas o desvio
            # entre execuções do próprio algoritmo é de 0.05 a 0.08 de Fβ —
            # muito maior que as quedas que disparam o retrocesso. O critério
            # está, na prática, lendo ruído como saturação: no BraTS ele
            # encerra com |T| = 1 ou 2 imagens, contra as 4 que os usuários
            # anotaram.
            #
            # Com paciência k, só se retrocede após k pioras consecutivas, e a
            # imagem devolvida é a última acrescentada. k=0 reproduz o artigo.
            if not improved:
                sem_melhora += 1
            else:
                sem_melhora = 0
                best_fb = fb

            if sem_melhora > args.patience and len(selected) > 1:
                selected.remove(last_added)
                rejected.add(last_added)
                sem_melhora = 0

            cands = [f for f in pool if f not in selected and f not in rejected]
            if args.filtra_degenerado and cands:
                antes = len(cands)
                cands = descarta_degenerados(cands, sal_dir, area)
                if len(cands) != antes:
                    print(f"      degeneradas descartadas: "
                          f"{antes - len(cands)} de {antes}")
            if not cands:
                print("      todos os candidatos rejeitados — encerrando split")
                break
            if not cands or len(selected) >= args.max_images:
                break
            z, why = pick_next(criterion, cands, sal_dir, label_folder,
                               orig_folder, enc_path, selected, args.device,
                               rng, args.proxy_layer, area)
            print(f"      próxima: {z[:-4]}  ({why})")
            selected.append(z)
            last_added = z
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--criterion", default="oracle", choices=CRITERIA)
    p.add_argument("--pool", default="real", choices=["real", "full"],
                   help="real: só imagens com markers reais (sem confundidor). "
                        "full: pool inteiro com markers sintéticos.")
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--seeds", type=int, default=3,
                   help="repetições (a imagem inicial de T é aleatória)")
    p.add_argument("--max_images", type=int, default=6)
    p.add_argument("--init", default="aleatorio",
                   choices=["aleatorio", "medoide"],
                   help="como escolher a primeira imagem. 'aleatorio' "
                        "reproduz o passo 1 do artigo.")
    p.add_argument("--filtra_degenerado", action="store_true",
                   help="descarta candidatas cuja saliência prevista é vazia")
    p.add_argument("--patience", type=int, default=0,
                   help="pioras consecutivas toleradas antes de retroceder. "
                        "0 reproduz o Algoritmo 1 do artigo.")
    p.add_argument("--decoder", default="labeled_marker")
    p.add_argument("--proxy_layer", type=int, default=3)
    p.add_argument("--markers", default="schisto/user_A")
    p.add_argument("--dataset_home", default="datasets")
    p.add_argument("--dataset", default="schisto", choices=list(DATASETS))
    p.add_argument("--pool_keeps_test", action="store_true",
                   help="mantém no pool as imagens que também estão no teste. "
                        "Use no Schisto, onde o vazamento é cortado no "
                        "conjunto de teste; NÃO use no BraTS, onde o pool "
                        "cobriria metade do teste.")
    p.add_argument("--pool_cap", type=int, default=0,
                   help="limita o pool amostrando; 0 = sem limite")
    p.add_argument("--marker_style", default="points",
                   choices=["points", "realistic"],
                   help="Como desenhar os markers sintéticos do pool full. "
                        "'points' sorteia pontos soltos (colapsa o encoder); "
                        "'realistic' usa pinceladas calibradas nos markers reais.")
    p.add_argument("--n_fg", type=int, default=100)
    p.add_argument("--n_bg", type=int, default=300)
    p.add_argument("--val_subsample", type=int, default=250,
                   help="Imagens de validação usadas DENTRO do laço (passo 6 "
                        "do Algoritmo 1). Subamostra fixa por split, igual "
                        "para todos os critérios. 0 = usar o val completo.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save_dir", default="out/paper_selection")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir,
                            f"selection_{args.criterion}_{args.pool}.csv")

    fields = ["split", "criterion", "seed", "round", "n_images", "fb", "dice",
              "iou", "mae", "improved", "selected", "marker_source", "pool_size",
              "val_used", "rejected"]
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["split"], r["seed"]))
        print(f"  [resume] {len(done)} (split, seed) já concluídos")

    for split in args.splits:
        for seed in range(args.seeds):
            if (str(split), str(seed)) in done:
                print(f"  [resume] split={split} seed={seed} — skip")
                continue
            rng = random.Random(1000 + seed)
            hist = run_selection(args.criterion, split, args, rng)
            if not hist:
                continue
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if write_header:
                    w.writeheader()
                for h in hist:
                    h["seed"] = seed
                    w.writerow(h)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
