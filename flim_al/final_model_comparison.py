#!/usr/bin/env python3
"""
final_model_comparison.py
=========================
A tabela que faltava: o efeito do active learning **por modelo**, não por
método de aquisição.

As tabelas anteriores (`TABELA_IMAGEM_VS_REGIAO.md`) comparam critérios de
aquisição entre si, fixando o decoder. Esta compara os oito decoders entre si,
fixando o critério — é a coluna que se encaixa ao lado da Tabela IV do artigo.

Protocolo idêntico ao do artigo (Seção VI):

    encoder FLIM treinado só com os markers de T
    bloco b ∈ {1..B} escolhido pelo melhor Fβ na validação Z₁\\T
    Fβ / MAE / DICE / IoU no teste Z₂, com Otsu + filtro de área [1000, 9000]

Dois braços, mesma tubulação:

    paper   T = os markers que os usuários desenharam no Algoritmo 1
    al      T = as imagens que o critério de AL escolheu, com os markers reais
            das mesmas imagens (nada sintético entra aqui)

Sem Dynamic Trees — o binário `iftSMansoniDelineation` depende de
liblapack/libblas que não existem na imagem. Os números ficam abaixo dos
publicados por esse motivo, e a comparação válida é sempre paper vs al dentro
desta tabela, nunca contra a Tabela IV.

Uso (de dentro de flim_ad/):
    python ../flim_al/final_model_comparison.py --split 1 --device cuda:0
    python ../flim_al/final_model_comparison.py --split all --examples
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "flim_ad" / "libs" / "flim-python"))

from flim_al.al_encoder_experiment import (  # noqa: E402
    _official_filter_and_binarize, evaluate_decoder, image_to_lab,
    retrain_encoder,
)
from pyflim import layers  # noqa: E402

ORIG = "datasets/schistossoma-eggs/orig"
LABEL = "datasets/schistossoma-eggs/label"
SPLITS = "datasets/schistossoma-eggs/Splits-5train-70_30"

# nome no artigo ← nome no código. Estabelecido lendo as equações da Seção IV
# contra as implementações em pyflim/layers.py.
#
# FLIM_bp fica de fora: o decoder por backpropagação precisa de um laço de
# treino próprio, não de pesos-função-do-encoder, e o Experimento B já mede
# exatamente esse caso em 396 execuções.
DECODERS = [
    ("labeled_marker",              "FLIM_lm"),
    ("decoder_2",                   "FLIM_pb"),
    ("decoder_3",                   "FLIM_mb"),
    ("decoder_attention",           "FLIM_at"),
    ("vanilla_adaptive_decoder",    "FLIM_ts"),
    ("hybrid_decoder",              "FLIM_lt"),
    ("vanilla_adaptive_decoder_wt", "FLIM_ts*"),   # ts sem a regra de proporção
]

FIELDS = ["user", "split", "arm", "decoder", "paper_name", "block", "n_train",
          "train_ids", "fb", "dice", "mae", "iou", "fb_val"]


# ── seleção de T ──────────────────────────────────────────────────────────────

def real_marker_index() -> dict[str, str]:
    """image_id → caminho do seeds.txt real. Só markers desenhados por humano."""
    idx: dict[str, str] = {}
    for pat in (REPO / "data" / "markers" / "*-seeds.txt",
                REPO / "data" / "markers_oracle" / "*-seeds.txt",
                REPO / "flim_ad" / "data" / "schisto" / "user_*" / "split*"
                / "markers" / "*-seeds.txt"):
        for p in glob.glob(str(pat)):
            idx.setdefault(os.path.basename(p).replace("-seeds.txt", ""), p)
    return idx


def al_selection(split: int, criterion: str = "entropy",
                 seed: int | None = None,
                 path: str | None = None) -> tuple[list[str], float]:
    """
    O T a que o Algoritmo 1 chegou.

    `seed=None` devolve o melhor Fβ entre TODAS as sementes — e isso é um viés.
    O algoritmo é executado uma vez, com uma imagem inicial sorteada; o usuário
    para na melhor *rodada* ("enquanto não estiver satisfeito"), não na melhor
    *semente*. Escolher entre sementes dá ao braço de AL uma vantagem que o
    braço do artigo não tem: no split 1 as três sementes rendem 0.804 / 0.789 /
    0.774 na validação, então o best-of-3 infla ~0.015 num Δ reportado como
    +0.002.

    Passando `seed`, a busca fica restrita àquela execução — é assim que a
    comparação deve ser feita, com uma linha por semente.

    Empate resolve pelo menor |T|: anotar menos pelo mesmo resultado é
    estritamente melhor.
    """
    path = path or f"out/paper_selection/selection_{criterion}_real.csv"
    rows = [r for r in csv.DictReader(open(path))
            if int(r["split"]) == split
            and (seed is None or int(r["seed"]) == seed)]
    if not rows:
        raise SystemExit(f"sem linhas para split {split} seed {seed} em {path}")
    best = max(rows, key=lambda r: (float(r["fb"]), -int(r["n_images"])))
    return best["selected"].split("|"), float(best["fb"])


def build_marker_dir(img_ids: list[str], dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    idx = real_marker_index()
    faltando = [i for i in img_ids if i not in idx]
    if faltando:
        raise SystemExit(f"markers reais ausentes para {faltando}")
    for i in img_ids:
        shutil.copy(idx[i], os.path.join(dest, f"{i}-seeds.txt"))
    return dest


# ── segmentação final para as figuras ────────────────────────────────────────

@torch.no_grad()
def save_segmentations(encoder_path: str, decoder_type: str, block: int,
                       fnames: list[str], out_dir: str,
                       area_range=(1000, 9000)) -> None:
    """
    Mesma cadeia da avaliação — forward → uint8 → Otsu + filtro de área — mas
    guardando a máscara binária em vez de só a métrica. É o que vai para o
    slide de exemplos.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = torch.load(encoder_path, map_location="cpu", weights_only=False)
    model.device = "cpu"
    for l in range(model.architecture.nlayers):
        ml = getattr(model.layers[l], "marker_labels", None)
        if ml is not None and hasattr(ml, "to"):
            model.layers[l].marker_labels = ml.to("cpu")
    model.decoder = layers.FLIMAdaptiveDecoderLayer(
        1, adaptation_function="robust_weights", filter_by_size=False,
        device="cpu", adj_radius=1.5, decoder_type=decoder_type,
        multi_layer=False)

    for fname in fnames:
        p = os.path.join(ORIG, fname)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        h, w = img.size[1], img.size[0]
        x = torch.tensor(image_to_lab(np.array(img, dtype=np.uint8))
                         .transpose(2, 0, 1)).unsqueeze(0)
        y_hat, _ = model.forward(x, decoder_layer=[block - 1])
        pred = y_hat[0].float()
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        sal = pred.squeeze().numpy().astype(np.uint8)
        if sal.shape[0] != h or sal.shape[1] != w:
            sal = np.array(Image.fromarray(sal).resize((w, h), Image.BILINEAR),
                           dtype=np.uint8)
        binm = _official_filter_and_binarize(sal, area_range)
        Image.fromarray((binm * 255).astype(np.uint8)).save(
            os.path.join(out_dir, fname))


# ── uma condição = um encoder ────────────────────────────────────────────────

def run_arm(split: int, arm: str, img_ids: list[str], marker_dir: str,
            val: list[str], test: list[str], device: str, blocks: list[int],
            work: str, examples: list[str], ex_root: str,
            user: str = "A") -> list[dict]:
    arch = f"data/schisto/user_{user}/split{split}/arch2D.json"
    enc = os.path.join(work, f"enc_s{split}_{arm}.pth")
    print(f"\n=== split {split} · {arm} · T={img_ids} ===", flush=True)
    retrain_encoder(arch, marker_dir, ORIG, LABEL, device, enc)

    out = []
    for dec, paper_name in DECODERS:
        t0 = time.time()
        # bloco escolhido na validação, como no artigo
        por_bloco = {}
        for b in blocks:
            try:
                por_bloco[b] = evaluate_decoder(enc, dec, b, val, ORIG, LABEL,
                                                device)["fb"]
            except Exception as e:
                print(f"    bloco {b} falhou: {e}", flush=True)
        if not por_bloco:
            continue
        best_b = max(por_bloco, key=por_bloco.get)
        m = evaluate_decoder(enc, dec, best_b, test, ORIG, LABEL, device)
        out.append({
            "user": user, "split": split, "arm": arm, "decoder": dec,
            "paper_name": paper_name, "block": best_b,
            "n_train": len(img_ids), "train_ids": "|".join(img_ids),
            "fb": round(m["fb"], 4), "dice": round(m["dice"], 4),
            "mae": round(m["mae"], 4), "iou": round(m["iou"], 4),
            "fb_val": round(por_bloco[best_b], 4),
        })
        print(f"  {paper_name:9s} b={best_b}  val={por_bloco[best_b]:.3f}  "
              f"teste Fβ={m['fb']:.3f} MAE={m['mae']:.4f} IoU={m['iou']:.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

        if examples:
            save_segmentations(enc, dec, best_b, examples,
                               os.path.join(ex_root, f"split{split}", arm, dec))
    return out


def append_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    novo = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if novo:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="1", help="1, 2, 3 ou all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--criterion", default="entropy")
    ap.add_argument("--blocks", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--n_val", type=int, default=120)
    ap.add_argument("--examples", action="store_true",
                    help="salva as máscaras binárias das imagens de exemplo")
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--user", default="A", choices=["A", "B"])
    ap.add_argument("--al_seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="uma execução do Algoritmo 1 por semente. Passar uma "
                         "só reproduz o comportamento antigo, enviesado.")
    ap.add_argument("--arms", nargs="+", default=["paper", "al"],
                    choices=["paper", "al"],
                    help="permite recomputar só o braço que falta")
    ap.add_argument("--selection_csv", default="",
                    help="por padrão out/paper_selection/selection_"
                         "{criterion}_real.csv")
    ap.add_argument("--test_list", default="",
                    help="conjunto de teste comum a TODOS os braços e usuários. "
                         "Sem ele, cada execução remove do teste só o que os "
                         "seus próprios braços treinaram, e aí o usuário A é "
                         "avaliado em 366 imagens e o B em 365 — a comparação "
                         "entre eles deixa de ser sobre o mesmo conjunto.")
    ap.add_argument("--out", default="out/final_comparison/per_model.csv")
    ap.add_argument("--ex_root", default="out/final_comparison/seg")
    a = ap.parse_args()

    splits = [1, 2, 3] if a.split == "all" else [int(a.split)]
    test = [l.strip() for l in open(a.test_list or f"{SPLITS}/test.txt")
            if l.strip()]
    if a.test_list:
        print(f"teste comum: {len(test)} imagens ({a.test_list})")

    # imagens de exemplo: as primeiras do teste com ovo grande o bastante para
    # sobreviver ao filtro de área — mostrar um caso insegmentável por
    # construção não diz nada sobre o decoder.
    examples = []
    if a.examples:
        for f in test:
            lp = os.path.join(LABEL, f)
            if not os.path.exists(lp):
                continue
            n = (np.array(Image.open(lp).convert("L")) > 0).sum()
            if 1500 < n < 9000:
                examples.append(f)
            if len(examples) >= a.n_examples:
                break
        print(f"exemplos: {examples}")

    work = tempfile.mkdtemp(prefix="finalcmp_")
    try:
        for s in splits:
            val_all = [l.strip() for l in open(f"{SPLITS}/split{s}-val.txt")
                       if l.strip()]
            mk_dir = f"data/schisto/user_{a.user}/split{s}/markers"
            paper_ids = sorted(os.path.basename(p).replace("-seeds.txt", "")
                               for p in glob.glob(f"{mk_dir}/*-seeds.txt"))

            # uma seleção por semente: cada uma é uma execução independente do
            # Algoritmo 1, e é a variação entre elas que a tabela precisa medir
            sel = {sd: al_selection(s, a.criterion, sd, a.selection_csv or None)
                   for sd in a.al_seeds}
            for sd, (ids, fbv) in sel.items():
                print(f"split {s} · semente {sd}: T={ids} (Fβ val = {fbv:.4f})")
            print(f"split {s}: paper T={paper_ids}")

            # Validação E teste comuns a TODOS os braços: tira de ambos tudo
            # que qualquer braço viu no treino.
            #
            # No teste isso não é zelo excessivo. Dois dos 31 markers reais do
            # Schisto (000675 e 000917) estão em test.txt: 000917 no T do
            # artigo do split 1, e 000675 no T do artigo do split 1 E na
            # semente 1 dos três splits. Sem o filtro, alguns braços seriam
            # avaliados em imagens que treinaram e outros não — a comparação
            # deixaria de ser pareada exatamente onde precisa ser.
            treinadas = {i + ".png" for i in paper_ids}
            for ids, _ in sel.values():
                treinadas |= {i + ".png" for i in ids}
            # Com --test_list o conjunto já vem depurado e é comum a todos;
            # filtrar de novo por split reintroduziria a assimetria que o
            # arquivo existe para eliminar.
            # Sem ele, filtra-se aqui — variável nova por split, porque
            # reatribuir `test` acumularia o filtro de um split no seguinte.
            if a.test_list:
                test_s = test
            else:
                test_s = [f for f in test if f not in treinadas]
                if len(test_s) != len(test):
                    print(f"  teste: {len(test)} → {len(test_s)} "
                          f"(removidas as que algum braço treinou)")
            pool = [f for f in val_all if f not in treinadas]
            rng = np.random.default_rng(0)
            val = list(rng.choice(pool, size=min(a.n_val, len(pool)),
                                  replace=False))

            tarefas = []
            if "paper" in a.arms:
                tarefas.append(("paper", paper_ids, mk_dir))
            if "al" in a.arms:
                for sd, (ids, _) in sel.items():
                    tarefas.append((f"al_seed{sd}", ids,
                                    build_marker_dir(
                                        ids, os.path.join(work,
                                                          f"m_s{s}_sd{sd}"))))
            for arm, ids, mdir in tarefas:
                rows = run_arm(s, arm, ids, mdir, val, test_s, a.device,
                               a.blocks, work, examples, a.ex_root, a.user)
                append_csv(a.out, rows)
                print(f"  [salvo] {a.out}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
