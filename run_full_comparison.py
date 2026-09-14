"""
Experimento completo — FLIMts, FLIMat, FLIMpb × Parasites × BraTS × 3 splits
Gera tabela comparativa vs TABLE III do paper (2504.20872v1)
"""
import os, time, json
import torch
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader
from pyflim import flim, arch, data, util
from sklearn.metrics import mean_absolute_error

# ─── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────

DATASETS = {
    "Parasites": {
        "orig":    "data/orig/",
        "markers": "data/markers/",
        "label":   "data/label/",
        "ext":     ".png",
        "mext":    "-seeds.txt",
        "splits": [
            ("split1", "split1-train.txt", "test.txt"),
            ("split2", "split2-train.txt", "test.txt"),
            ("split3", "split3-train.txt", "test.txt"),
        ],
        "arch": "arch_best.json",
        "batch_train": 5,
    },
    "BraTS": {
        "orig":    "data/brats/orig/",
        "markers": "data/brats/markers/",
        "label":   "data/brats/label/",
        "ext":     ".png",
        "mext":    "-seeds.txt",
        "splits": [
            ("split1", "data/brats/Splits-50_50/split1-train.txt", "data/brats/Splits-50_50/split1-test.txt"),
            ("split2", "data/brats/Splits-50_50/split2-train.txt", "data/brats/Splits-50_50/split2-test.txt"),
            ("split3", "data/brats/Splits-50_50/split3-train.txt", "data/brats/Splits-50_50/split3-test.txt"),
        ],
        "arch": "arch_best_brats.json",
        "batch_train": 4,
    },
}

DECODERS = {
    "FLIMts": "vanilla_adaptive_decoder",
    "FLIMat": "decoder_2",
    "FLIMpb": "decoder_3",
}

# Valores de referência do paper TABLE III (validation set)
PAPER_TABLE3 = {
    "Parasites": {
        "FLIMts": {"A": (0.010, 0.747), "B": (0.013, 0.687)},
        "FLIMat": {"A": (0.011, 0.740), "B": (0.014, 0.660)},
        "FLIMpb": {"A": (0.006, 0.857), "B": (0.006, 0.847)},
    },
    "BraTS": {
        "FLIMts": {"A": (0.017, 0.709), "B": (0.020, 0.697)},
        "FLIMat": {"A": (0.022, 0.679), "B": (0.024, 0.694)},
        "FLIMpb": {"A": (0.019, 0.703), "B": (0.022, 0.691)},
    },
}

# ─── FUNÇÕES ───────────────────────────────────────────────────────────────────

def build_train_loader(cfg, train_list):
    ds = data.FLIMData(
        cfg["orig"], images_list=train_list,
        marker_folder=cfg["markers"], orig_ext=cfg["ext"], marker_ext=cfg["mext"],
        transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()])
    )
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds),
        batch_size=cfg["batch_train"], drop_last=False)
    return DataLoader(ds, batch_sampler=sampler)


def build_test_loader(cfg, test_list):
    ds = data.FLIMData(
        cfg["orig"], images_list=test_list, orig_ext=cfg["ext"],
        transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()])
    )
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(ds),
        batch_size=1, drop_last=False)
    return DataLoader(ds, batch_sampler=sampler)


def compute_metrics(output_folder, label_folder, file_list):
    try:
        result_ext = "." + util.get_files_extension(output_folder)
    except:
        return {"MAE": float("nan"), "Fb": float("nan"), "DICE": float("nan")}
    label_ext = "." + util.get_files_extension(label_folder)
    maes, fbs, dices = [], [], []
    for fname in file_list:
        fname = str(fname).strip().split(".")[0]
        rp = output_folder + fname + result_ext
        lp = label_folder  + fname + label_ext
        if not os.path.isfile(rp) or not os.path.isfile(lp):
            continue
        sal = io.imread(rp); sal = sal[:,:,0] if sal.ndim==3 else sal
        lbl = io.imread(lp).astype(np.uint8); lbl = lbl[:,:,0] if lbl.ndim==3 else lbl
        lbl[lbl > 0] = 1
        thr = threshold_otsu(sal)
        b = (sal > thr).astype(np.uint8)
        maes.append(mean_absolute_error(b.flatten().astype(float), lbl.flatten().astype(float)))
        tp = np.sum(b*lbl); fp = np.sum(b*(1-lbl)); fn = np.sum((1-b)*lbl)
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
        fbs.append(1.3*p*r/(0.3*p+r+1e-8))
        denom = np.sum(b)+np.sum(lbl)
        dices.append((2*tp/denom) if denom > 0 else 1.0)
    return {
        "MAE":  np.mean(maes)  if maes  else float("nan"),
        "Fb":   np.mean(fbs)   if fbs   else float("nan"),
        "DICE": np.mean(dices) if dices else float("nan"),
    }


def run_single(dataset_name, cfg, decoder_name, decoder_type, split_name, train_list, test_list):
    out = f"results/{dataset_name}_{decoder_name}_{split_name}/"
    os.makedirs(out, exist_ok=True)

    arch_file = cfg["arch"]
    if not os.path.isfile(arch_file):
        arch_file = "arch.json"  # fallback

    architecture = arch.FLIMArchitecture(arch_file)
    model = flim.FLIMModel(
        architecture, decoder_type=decoder_type,
        adaptation_function="robust_weights", device="cpu",
        filter_by_size=False, track_gpu_stats=False
    )
    model.fit(build_train_loader(cfg, train_list))
    model.run(build_test_loader(cfg, test_list), out)

    file_list = util.readFileList(test_list)
    return compute_metrics(out, cfg["label"], file_list)


# ─── EXECUÇÃO ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_results = {}  # [dataset][decoder] = list of (MAE, Fb) per split

    total = len(DATASETS) * len(DECODERS) * 3
    done  = 0

    for ds_name, cfg in DATASETS.items():
        all_results[ds_name] = {}
        for dec_name, dec_type in DECODERS.items():
            split_metrics = []
            for split_name, train_list, test_list in cfg["splits"]:
                done += 1
                print(f"\n[{done}/{total}] {ds_name} | {dec_name} | {split_name}")
                t0 = time.time()
                m = run_single(ds_name, cfg, dec_name, dec_type, split_name, train_list, test_list)
                print(f"  MAE={m['MAE']:.4f}  Fβ={m['Fb']:.4f}  DICE={m['DICE']:.4f}  ({time.time()-t0:.0f}s)")
                split_metrics.append(m)

            maes  = [m["MAE"] for m in split_metrics if not np.isnan(m["MAE"])]
            fbs   = [m["Fb"]  for m in split_metrics if not np.isnan(m["Fb"])]
            all_results[ds_name][dec_name] = {
                "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
                "Fb_mean":  np.mean(fbs),  "Fb_std":  np.std(fbs),
            }

    # Salvar JSON com resultados brutos
    with open("results/full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ─── GERAR TABELA COMPARATIVA ─────────────────────────────────────────────
    print("\n" + "="*90)
    print("  TABELA COMPARATIVA — Nossos Resultados vs Paper TABLE III (val set)")
    print("="*90)
    print(f"{'Modelo':<10} {'Dataset':<10} {'Ours MAE':>10} {'Paper MAE':>10} {'Ours Fβ':>10} {'Paper Fβ':>10}")
    print("-"*90)

    for ds_name in DATASETS:
        for dec_name in DECODERS:
            r    = all_results[ds_name][dec_name]
            pA   = PAPER_TABLE3[ds_name][dec_name]["A"]
            pB   = PAPER_TABLE3[ds_name][dec_name]["B"]
            p_mae_str = f"{pA[0]:.3f}/{pB[0]:.3f}"
            p_fb_str  = f"{pA[1]:.3f}/{pB[1]:.3f}"
            print(f"{dec_name:<10} {ds_name:<10} "
                  f"{r['MAE_mean']:.4f}±{r['MAE_std']:.4f}  "
                  f"{p_mae_str:>10}  "
                  f"{r['Fb_mean']:.4f}±{r['Fb_std']:.4f}  "
                  f"{p_fb_str:>10}")

    print("="*90)
    print("Nota: Paper = validation set (Z1\\T) | Ours = test set")
    print("Resultados salvos em results/full_results.json")

    # ─── GERAR HTML ────────────────────────────────────────────────────────────
    generate_html_table(all_results)
    print("Tabela HTML salva em results/comparison_table.html")


def generate_html_table(all_results):
    rows = ""
    for ds_name in DATASETS:
        first_ds = True
        for dec_name in DECODERS:
            r  = all_results[ds_name][dec_name]
            pA = PAPER_TABLE3[ds_name][dec_name]["A"]
            pB = PAPER_TABLE3[ds_name][dec_name]["B"]

            def color(ours, paper):
                diff = ours - paper
                if abs(diff) < 0.02:  return "#2ecc71"   # verde — próximo
                elif diff > 0:        return "#2ecc71"   # verde — melhor
                elif diff > -0.10:    return "#f39c12"   # laranja — gap moderado
                else:                 return "#e74c3c"   # vermelho — gap grande

            mae_color = color(-r['MAE_mean'], -(pA[0]+pB[0])/2)
            fb_color  = color(r['Fb_mean'],   (pA[1]+pB[1])/2)

            ds_cell = f'<td rowspan="3" style="font-weight:bold;vertical-align:middle">{ds_name}</td>' if first_ds else ""
            first_ds = False

            rows += f"""
            <tr>
              {ds_cell}
              <td>{dec_name}</td>
              <td style="color:{mae_color};font-weight:bold">{r['MAE_mean']:.4f} ± {r['MAE_std']:.4f}</td>
              <td>{pA[0]:.3f} / {pB[0]:.3f}</td>
              <td style="color:{fb_color};font-weight:bold">{r['Fb_mean']:.4f} ± {r['Fb_std']:.4f}</td>
              <td>{pA[1]:.3f} / {pB[1]:.3f}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FLIM Comparison Table</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 20px; background:#1a1a2e; color:#eee; }}
  h2   {{ color:#00d4ff; }}
  table{{ border-collapse:collapse; width:100%; font-size:14px; }}
  th   {{ background:#16213e; color:#00d4ff; padding:10px; border:1px solid #333; }}
  td   {{ padding:8px 12px; border:1px solid #333; text-align:center; }}
  tr:hover td {{ background:#16213e; }}
  .note {{ font-size:12px; color:#aaa; margin-top:10px; }}
  .legend {{ display:flex; gap:20px; margin:10px 0; font-size:13px; }}
  .leg-item {{ display:flex; align-items:center; gap:6px; }}
  .dot {{ width:12px;height:12px;border-radius:50%; }}
</style>
</head><body>
<h2>FLIM Comparison — Our Results vs Paper TABLE III</h2>
<div class="legend">
  <div class="leg-item"><div class="dot" style="background:#2ecc71"></div> Close to / better than paper</div>
  <div class="leg-item"><div class="dot" style="background:#f39c12"></div> Moderate gap (&lt;10%)</div>
  <div class="leg-item"><div class="dot" style="background:#e74c3c"></div> Large gap (&gt;10%)</div>
</div>
<table>
  <thead>
    <tr>
      <th>Dataset</th><th>Model</th>
      <th>Our MAE ↓</th><th>Paper MAE (A/B) ↓</th>
      <th>Our Fβ ↑</th><th>Paper Fβ (A/B) ↑</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
<p class="note">Our results: test set (3 splits, mean ± std) | Paper: validation set Z1\\T, User A / User B</p>
<p class="note">Architecture: arch_best.json (optimized for FLIMpb/split1-val Parasites) | No post-processing (Otsu only)</p>
</body></html>"""

    with open("results/comparison_table.html", "w") as f:
        f.write(html)
