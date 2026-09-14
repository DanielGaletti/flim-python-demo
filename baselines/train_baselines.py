"""
Treino e avaliação dos modelos baseline: SAMNet, MSCNet, MEANet, UNet, U-NetFLIM
Protocolo do paper 2504.20872v1:
  - Fine-tuning com GT masks, 100 epochs
  - SAMNet lr=0.005, MSCNet lr=0.03, MEANet lr=0.01, UNet/UNetFLIM lr=0.01
  - 3 splits × 2 datasets = 6 runs por modelo
"""
import os, time, json, argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage import io
from skimage.filters import threshold_otsu
from sklearn.metrics import mean_absolute_error

from baselines.dataset import SODDataset
from baselines.losses  import multi_scale_loss

# ─── CONFIGURAÇÕES ─────────────────────────────────────────────────────────────

DATASETS = {
    "Parasites": {
        "orig":   "data/orig/",
        "label":  "data/label/",
        "ext":    ".png",
        "splits": [
            ("split1", "split1-train.txt",  "test.txt"),
            ("split2", "split2-train.txt",  "test.txt"),
            ("split3", "split3-train.txt",  "test.txt"),
        ],
        "arch":    "arch_best.json",
        "markers": "data/markers/",
        "mext":    "-seeds.txt",
    },
    "BraTS": {
        "orig":   "data/brats/orig/",
        "label":  "data/brats/label/",
        "ext":    ".png",
        "splits": [
            ("split1", "data/brats/Splits-50_50/split1-train.txt", "data/brats/Splits-50_50/split1-test.txt"),
            ("split2", "data/brats/Splits-50_50/split2-train.txt", "data/brats/Splits-50_50/split2-test.txt"),
            ("split3", "data/brats/Splits-50_50/split3-train.txt", "data/brats/Splits-50_50/split3-test.txt"),
        ],
        "arch":    "arch_best_brats.json",
        "markers": "data/brats/markers/",
        "mext":    "-seeds.txt",
    },
}

MODEL_CONFIGS = {
    "SAMNet":    {"module": "baselines.models.samnet",  "class": "SAMNet",  "lr": 0.005, "wd": 5e-4},
    "MSCNet":    {"module": "baselines.models.mscnet",  "class": "MSCNet",  "lr": 0.030, "wd": 5e-4},
    "MEANet":    {"module": "baselines.models.meanet",  "class": "MEANet",  "lr": 0.010, "wd": 5e-4},
    "UNet":      {"module": "baselines.models.unet",    "class": "UNet",    "lr": 0.010, "wd": 5e-4},
    "UNetFLIM":  None,  # Tratamento especial — ver train_unet_flim()
}

EPOCHS      = 100
BATCH_SIZE  = 4
REPEAT      = 8   # 5 imgs × 8 = 40 samples/epoch — suficiente com augmentation forte
INFER_BATCH = 8   # inferência em batch para acelerar
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "results/baselines"


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_model(name, pretrained=True):
    import importlib
    cfg = MODEL_CONFIGS[name]
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    return cls(pretrained=pretrained)


# ─── U-NetFLIM: treino em 2 fases ──────────────────────────────────────────────

def train_unet_flim(ds_name, cfg, split_name, train_file, test_file):
    """
    Fase 1: treina encoder FLIM com markers (k-means, sem backprop)
    Fase 2: congela encoder, treina decoder U-Net com GT (100 epochs)
    """
    from baselines.models.unet_flim import build_unet_flim
    import sys; sys.path.insert(0, ".")
    from pyflim import data as flim_data

    print(f"\n{'='*60}")
    print(f"  UNetFLIM | {ds_name} | {split_name}")
    print(f"{'='*60}")

    arch_file = cfg.get("arch", "arch_best.json")
    if not os.path.isfile(arch_file):
        arch_file = "arch_best.json"

    # ── Fase 1: fit encoder FLIM ──
    print("  [1/2] Treinando encoder FLIM com markers...")
    from pyflim import arch as flim_arch, flim as flim_module
    architecture = flim_arch.FLIMArchitecture(arch_file)
    flim_model   = flim_module.FLIMModel(
        architecture, decoder_type="vanilla_adaptive_decoder",
        adaptation_function="robust_weights", device="cpu",
        filter_by_size=False, track_gpu_stats=False
    )
    train_ds_flim = flim_data.FLIMData(
        cfg["orig"], images_list=train_file,
        marker_folder=cfg["markers"], orig_ext=cfg["ext"], marker_ext=cfg["mext"],
        transform=flim_data.transforms.Compose([flim_data.Rescale(256), flim_data.ToTensor()])
    )
    import torch.utils.data as tud
    sampler = tud.BatchSampler(tud.RandomSampler(train_ds_flim), batch_size=4, drop_last=False)
    flim_dl = DataLoader(train_ds_flim, batch_sampler=sampler)
    flim_model.fit(flim_dl)
    torch.set_grad_enabled(True)   # pyflim.fit pode deixar grad desabilitado
    print("  Encoder FLIM treinado.")

    # ── Fase 2: treina decoder U-Net ──
    print("  [2/2] Treinando decoder U-Net com GT (100 epochs)...")
    model = build_unet_flim(arch_file, flim_model)
    model = model.to(DEVICE)
    opt   = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01, weight_decay=5e-4
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_sod = SODDataset(cfg["orig"], cfg["label"], train_file,
                           orig_ext=cfg["ext"], augment=True)
    from torch.utils.data import ConcatDataset
    rep_ds  = ConcatDataset([train_sod] * REPEAT)
    train_dl = DataLoader(rep_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=False)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, lbls, _ in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = multi_scale_loss(out, lbls)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={epoch_loss/len(train_dl):.4f} | {time.time()-t0:.0f}s")

    # Salvar predições e avaliar
    out_dir = os.path.join(SAVE_DIR, f"UNetFLIM_{ds_name}_{split_name}")
    with open(test_file) as f:
        test_list = [l.strip() for l in f if l.strip()]
    save_predictions(model, cfg["orig"], test_list, out_dir, cfg["ext"])
    mae, fb = compute_metrics(out_dir, cfg["label"], test_list)
    print(f"  RESULT → MAE={mae:.4f}  Fβ={fb:.4f}")

    torch.save({k: v for k, v in model.state_dict().items()
                if "flim_model" not in k},
               os.path.join(SAVE_DIR, f"UNetFLIM_{ds_name}_{split_name}.pth"))
    return {"MAE": mae, "Fb": fb}


def compute_metrics(pred_dir, label_dir, file_list):
    """Compute MAE e Fβ (β²=0.3)."""
    try:
        from pyflim import util
        ext = "." + util.get_files_extension(pred_dir)
    except Exception:
        ext = ".png"
    lext = ".png"
    maes, fbs = [], []
    for fname in file_list:
        fname = str(fname).strip().split(".")[0]
        rp = os.path.join(pred_dir,   fname + ext)
        lp = os.path.join(label_dir,  fname + lext)
        if not os.path.isfile(rp) or not os.path.isfile(lp):
            continue
        sal = io.imread(rp); sal = sal[:, :, 0] if sal.ndim == 3 else sal
        lbl = io.imread(lp).astype(np.uint8); lbl = lbl[:, :, 0] if lbl.ndim == 3 else lbl
        lbl[lbl > 0] = 1
        thr = threshold_otsu(sal)
        b   = (sal > thr).astype(np.uint8)
        maes.append(mean_absolute_error(b.flatten().astype(float), lbl.flatten().astype(float)))
        tp = np.sum(b * lbl); fp = np.sum(b * (1 - lbl)); fn = np.sum((1 - b) * lbl)
        p  = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
        fbs.append(1.3 * p * r / (0.3 * p + r + 1e-8))
    return (np.mean(maes) if maes else float("nan"),
            np.mean(fbs)  if fbs  else float("nan"))


def save_predictions(model, orig_folder, file_list, out_dir, ext=".png"):
    """Inferência em batch para acelerar (INFER_BATCH imagens por vez)."""
    import torchvision.transforms.functional as TF
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

    # Filtra arquivos existentes e guarda tamanhos originais
    valid, orig_sizes = [], []
    for fname in file_list:
        base = str(fname).strip().split(".")[0]
        p    = os.path.join(orig_folder, base + ext)
        if os.path.isfile(p):
            valid.append(base)
            with Image.open(p) as im:
                orig_sizes.append(im.size)  # (W, H)

    with torch.inference_mode():
        for i in range(0, len(valid), INFER_BATCH):
            batch_names = valid[i:i + INFER_BATCH]
            batch_sizes = orig_sizes[i:i + INFER_BATCH]
            tensors = []
            for base in batch_names:
                img = Image.open(os.path.join(orig_folder, base + ext)).convert("RGB")
                t   = TF.normalize(TF.to_tensor(img.resize((256, 256))), MEAN, STD)
                tensors.append(t)
            batch  = torch.stack(tensors).to(DEVICE)
            preds  = model(batch).squeeze(1).sigmoid().cpu().numpy()  # (B, H, W)
            for pred, base, (W0, H0) in zip(preds, batch_names, batch_sizes):
                sal = Image.fromarray((pred * 255).astype(np.uint8))
                sal = sal.resize((W0, H0), Image.BILINEAR)
                sal.save(os.path.join(out_dir, base + ".png"))


# ─── TREINO ────────────────────────────────────────────────────────────────────

def train_one(model_name, ds_name, cfg, split_name, train_file, test_file):
    print(f"\n{'='*60}")
    print(f"  {model_name} | {ds_name} | {split_name}")
    print(f"{'='*60}")

    mc     = MODEL_CONFIGS[model_name]
    model  = load_model(model_name, pretrained=True).to(DEVICE)
    opt    = optim.SGD(model.parameters(), lr=mc["lr"],
                       momentum=0.9, weight_decay=mc["wd"])
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Dataset com repetição para datasets pequenos
    train_ds  = SODDataset(cfg["orig"], cfg["label"], train_file,
                           orig_ext=cfg["ext"], augment=True)
    # Repeat dataset
    from torch.utils.data import ConcatDataset
    rep_ds    = ConcatDataset([train_ds] * REPEAT)
    train_dl  = DataLoader(rep_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, drop_last=False)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, lbls, _ in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = multi_scale_loss(out, lbls)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={epoch_loss/len(train_dl):.4f} | "
                  f"lr={sched.get_last_lr()[0]:.5f} | {time.time()-t0:.0f}s")

    # Salvar predições e avaliar
    out_dir   = os.path.join(SAVE_DIR, f"{model_name}_{ds_name}_{split_name}")
    with open(test_file) as f:
        test_list = [l.strip() for l in f if l.strip()]
    save_predictions(model, cfg["orig"], test_list, out_dir, cfg["ext"])
    mae, fb = compute_metrics(out_dir, cfg["label"], test_list)
    print(f"  RESULT → MAE={mae:.4f}  Fβ={fb:.4f}")

    # Salvar modelo
    ckpt = os.path.join(SAVE_DIR, f"{model_name}_{ds_name}_{split_name}.pth")
    torch.save(model.state_dict(), ckpt)
    return {"MAE": mae, "Fb": fb}


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def run_all(models_to_run=None, datasets_to_run=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Carregar resultados anteriores (para não repetir runs já feitos)
    out_json = os.path.join(SAVE_DIR, "baseline_results.json")
    results  = {}
    if os.path.isfile(out_json):
        with open(out_json) as f:
            results = json.load(f)

    models_to_run   = models_to_run   or list(MODEL_CONFIGS.keys())
    datasets_to_run = datasets_to_run or list(DATASETS.keys())

    for mname in models_to_run:
        if mname not in results:
            results[mname] = {}
        for dname in datasets_to_run:
            if dname in results[mname]:
                print(f"  [SKIP] {mname} | {dname} — já calculado")
                continue
            cfg = DATASETS[dname]
            splits_metrics = []
            for split_name, train_file, test_file in cfg["splits"]:
                if mname == "UNetFLIM":
                    m = train_unet_flim(dname, cfg, split_name, train_file, test_file)
                else:
                    m = train_one(mname, dname, cfg, split_name, train_file, test_file)
                splits_metrics.append(m)

            maes = [m["MAE"] for m in splits_metrics if not np.isnan(m["MAE"])]
            fbs  = [m["Fb"]  for m in splits_metrics if not np.isnan(m["Fb"])]
            results[mname][dname] = {
                "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
                "Fb_mean":  float(np.mean(fbs)),  "Fb_std":  float(np.std(fbs)),
            }
            print(f"\n  {mname} | {dname} → "
                  f"MAE={results[mname][dname]['MAE_mean']:.4f}±{results[mname][dname]['MAE_std']:.4f}  "
                  f"Fβ={results[mname][dname]['Fb_mean']:.4f}±{results[mname][dname]['Fb_std']:.4f}")

            # Salvar progressivamente
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nResultados salvos em {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=None,
                        choices=["SAMNet", "MSCNet", "MEANet", "UNet", "UNetFLIM"],
                        help="Modelos a rodar (default: todos)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=["Parasites", "BraTS"],
                        help="Datasets a rodar (default: todos)")
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    args = parser.parse_args()
    EPOCHS = args.epochs
    run_all(args.models, args.datasets)
