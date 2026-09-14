"""
Experimento FLIM — Comparação User A vs User B (decoder_3 = FLIMpb)
Reproduz o setup do paper: 3 splits, avaliados no test set.

Fixes aplicados:
  - decoder_type="decoder_3"  (FLIMpb — melhor resultado no paper)
  - batch_size=1 na inferência (adaptive_decoder3 só processa feature[0])
  - saliency salva como (H,W) garantido (squeeze)
"""

import os
import time
import torch
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
from pyflim import flim, arch, data, metrics, util

ARCH_FILE     = "arch_best.json"   # gerado por arch_search.py / arch_search2.py
ORIG_FOLDER   = "data/orig/"
MARKER_FOLDER = "data/markers/"
LABEL_FOLDER  = "data/label/"
ORIG_EXT      = ".png"
MARKER_EXT    = "-seeds.txt"
TEST_LIST     = "test.txt"

ORACLE_FOLDER = "data/markers_oracle/"

EXPERIMENTS = [
    # Example markers (repositório)
    ("User_A_split1",      "split1-train.txt", MARKER_FOLDER, "out_A_split1/"),
    ("User_A_split2",      "split2-train.txt", MARKER_FOLDER, "out_A_split2/"),
    ("User_B_split3",      "split3-train.txt", MARKER_FOLDER, "out_B_split3/"),
    # Oracle markers (gerados do ground-truth — upper bound teórico)
    ("Oracle_split1",      "split1-train.txt", ORACLE_FOLDER, "out_oracle_split1/"),
    ("Oracle_split2",      "split2-train.txt", ORACLE_FOLDER, "out_oracle_split2/"),
    ("Oracle_split3",      "split3-train.txt", ORACLE_FOLDER, "out_oracle_split3/"),
]


def build_train_dataset(train_list, marker_folder):
    ds = data.FLIMData(
        ORIG_FOLDER,
        images_list=train_list,
        marker_folder=marker_folder,
        orig_ext=ORIG_EXT,
        marker_ext=MARKER_EXT,
        transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()])
    )
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds),
        batch_size=5, drop_last=False
    )
    return DataLoader(ds, batch_sampler=sampler)


def build_test_dataset_batched():
    """Batch para treino — não usado na inferência com decoder_3."""
    ds = data.FLIMData(
        ORIG_FOLDER, images_list=TEST_LIST, orig_ext=ORIG_EXT,
        transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()])
    )
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(ds), batch_size=5, drop_last=False
    )
    return DataLoader(ds, batch_sampler=sampler)


def build_test_dataset_single():
    """batch_size=1 — obrigatório para decoder_3 (FLIMpb)."""
    ds = data.FLIMData(
        ORIG_FOLDER, images_list=TEST_LIST, orig_ext=ORIG_EXT,
        transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()])
    )
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(ds), batch_size=1, drop_last=False
    )
    return DataLoader(ds, batch_sampler=sampler)


def compute_metrics_manual(output_folder, label_folder, file_list):
    """
    Recomputa MAE e Fβ manualmente para contornar bug do metrics.py.
    Retorna dict com arrays de métricas por imagem.
    """
    from skimage.filters import threshold_otsu
    from sklearn.metrics import mean_absolute_error

    result_ext = "." + util.get_files_extension(output_folder)
    label_ext  = "." + util.get_files_extension(label_folder)

    maes, wfscores, dices = [], [], []
    skipped = 0

    for fname in file_list:
        fname = str(fname).strip()
        if "." in fname:
            fname = fname.split(".")[0]

        rpath = output_folder + fname + result_ext
        lpath = label_folder  + fname + label_ext

        if not os.path.isfile(rpath) or not os.path.isfile(lpath):
            skipped += 1
            continue

        sal = io.imread(rpath)
        if sal.ndim == 3:
            sal = sal[:, :, 0]
        sal = sal.astype(np.uint8)

        lbl = io.imread(lpath).astype(np.uint8)
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        lbl[lbl > 0] = 1

        thr = threshold_otsu(sal)
        bin_sal = (sal > thr).astype(np.uint8)

        mae = mean_absolute_error(bin_sal.flatten().astype(np.float32), lbl.flatten().astype(np.float32))
        maes.append(mae)

        tp = np.sum(bin_sal * lbl)
        fp = np.sum(bin_sal * (1 - lbl))
        fn = np.sum((1 - bin_sal) * lbl)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        beta2 = 0.3
        fb = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-8)
        wfscores.append(fb)

        denom = np.sum(bin_sal) + np.sum(lbl)
        dice = (2 * tp / denom) if denom > 0 else 1.0
        dices.append(dice)

    print(f"  Avaliados: {len(maes)} | Pulados (sem arquivo): {skipped}")
    return {
        "MAE":  np.mean(maes)     if maes     else float("nan"),
        "Fb":   np.mean(wfscores) if wfscores else float("nan"),
        "DICE": np.mean(dices)    if dices    else float("nan"),
    }


def run_experiment(name, train_list, marker_folder, output_folder):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    architecture = arch.FLIMArchitecture(ARCH_FILE)
    model = flim.FLIMModel(
        architecture,
        decoder_type="decoder_3",       # FLIMpb
        adaptation_function="robust_weights",
        device="cpu",
        filter_by_size=False,
        track_gpu_stats=False
    )

    # Treino (batch OK para fit)
    train_ds = build_train_dataset(train_list, marker_folder)
    t0 = time.time()
    model.fit(train_ds)
    print(f"  Treino: {time.time()-t0:.1f}s")

    # Inferência com batch_size=1 (requerido pelo decoder_3)
    test_ds = build_test_dataset_single()
    t0 = time.time()
    model.run(test_ds, output_folder)
    print(f"  Inferência: {time.time()-t0:.1f}s")

    # Métricas manuais (evita bug do metrics.py)
    file_list = util.readFileList(TEST_LIST)
    result = compute_metrics_manual(output_folder, LABEL_FOLDER, file_list)
    print(f"  MAE={result['MAE']:.4f}  Fβ={result['Fb']:.4f}  DICE={result['DICE']:.4f}")
    return result


if __name__ == "__main__":
    results = {}
    for name, train_list, marker_folder, output_folder in EXPERIMENTS:
        results[name] = run_experiment(name, train_list, marker_folder, output_folder)

    print(f"\n{'='*65}")
    print(f"{'Experimento':<22} {'MAE':>8} {'Fβ':>8} {'DICE':>8}")
    print(f"{'-'*65}")
    for name, r in results.items():
        print(f"{name:<22} {r['MAE']:>8.4f} {r['Fb']:>8.4f} {r['DICE']:>8.4f}")
    print(f"{'='*65}")
    print("\nReferência paper TABLE IV (FLIMpb — Parasites):")
    print(f"  User A: MAE=0.005  Fβ=0.857")
    print(f"  User B: MAE=0.005  Fβ=0.843")
