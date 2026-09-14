"""
Experimento Conjunctiva Segmentation Dataset (Kaggle: mdraselsarker)
Modelos: SAMNet, MSCNet, MEANet, UNet, UNetFLIM, FLIMts, FLIMat, FLIMpb
Métricas: MAE, Fβ (β²=0.3), IoU
Split único: train.txt / valid.txt / test.txt
"""
import os, time, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from skimage import io
from skimage.filters import threshold_otsu
from sklearn.metrics import mean_absolute_error
from PIL import Image
import torchvision.transforms.functional as TF

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DS_CFG = {
    "orig":    "data/conjunctiva/images/",
    "label":   "data/conjunctiva/labels/",
    "ext":     ".jpg",
    "lext":    ".png",
    "train":   "data/conjunctiva/splits/train.txt",
    "val":     "data/conjunctiva/splits/valid.txt",
    "test":    "data/conjunctiva/splits/test.txt",
    "markers": "data/conjunctiva/markers/",
    "mext":    "-seeds.txt",
    "arch":    "arch_conjunctiva.json",
}

MODEL_CONFIGS = {
    "SAMNet":  {"module": "baselines.models.samnet", "class": "SAMNet", "lr": 0.005, "wd": 5e-4},
    "MSCNet":  {"module": "baselines.models.mscnet", "class": "MSCNet", "lr": 0.030, "wd": 5e-4},
    "MEANet":  {"module": "baselines.models.meanet", "class": "MEANet", "lr": 0.010, "wd": 5e-4},
    "UNet":    {"module": "baselines.models.unet",   "class": "UNet",   "lr": 0.010, "wd": 5e-4},
}

FLIM_DECODERS = {
    "FLIMts": "vanilla_adaptive_decoder",
    "FLIMat": "decoder_2",
    "FLIMpb": "decoder_3",
}

EPOCHS      = 100
BATCH_SIZE  = 4
REPEAT      = 8
INFER_BATCH = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "results/conjunctiva"
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# ─── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_dir, file_list):
    """MAE, Fβ (β²=0.3), IoU — threshold Otsu."""
    # Detecta extensão dos arquivos de predição no diretório
    pred_ext = ".png"
    for f in os.listdir(pred_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp"):
            pred_ext = ext
            break

    maes, fbs, ious = [], [], []
    for fname in file_list:
        base = str(fname).strip().split(".")[0]
        rp = os.path.join(pred_dir, base + pred_ext)
        lp = os.path.join(DS_CFG["label"], base + DS_CFG["lext"])
        if not os.path.isfile(rp) or not os.path.isfile(lp):
            continue
        sal = io.imread(rp)
        sal = sal[:, :, 0] if sal.ndim == 3 else sal
        lbl = io.imread(lp).astype(np.uint8)
        lbl = lbl[:, :, 0] if lbl.ndim == 3 else lbl
        lbl[lbl > 0] = 1

        thr = threshold_otsu(sal)
        b   = (sal > thr).astype(np.uint8)

        # MAE
        maes.append(mean_absolute_error(b.flatten().astype(float), lbl.flatten().astype(float)))

        # Fβ (β²=0.3)
        tp = np.sum(b * lbl); fp = np.sum(b * (1 - lbl)); fn = np.sum((1 - b) * lbl)
        p  = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
        fbs.append(1.3 * p * r / (0.3 * p + r + 1e-8))

        # IoU
        intersection = np.sum(b * lbl)
        union        = np.sum(np.clip(b + lbl, 0, 1))
        ious.append(intersection / (union + 1e-8))

    return (
        np.mean(maes) if maes else float("nan"),
        np.mean(fbs)  if fbs  else float("nan"),
        np.mean(ious) if ious else float("nan"),
    )


def save_predictions(model, file_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    valid, sizes = [], []
    for fname in file_list:
        base = str(fname).strip().split(".")[0]
        p    = os.path.join(DS_CFG["orig"], base + DS_CFG["ext"])
        if os.path.isfile(p):
            valid.append(base)
            with Image.open(p) as im:
                sizes.append(im.size)

    with torch.inference_mode():
        for i in range(0, len(valid), INFER_BATCH):
            names  = valid[i:i + INFER_BATCH]
            szs    = sizes[i:i + INFER_BATCH]
            tensors = []
            for base in names:
                img = Image.open(os.path.join(DS_CFG["orig"], base + DS_CFG["ext"])).convert("RGB")
                t   = TF.normalize(TF.to_tensor(img.resize((256, 256))), MEAN, STD)
                tensors.append(t)
            batch = torch.stack(tensors).to(DEVICE)
            preds = model(batch).squeeze(1).sigmoid().cpu().numpy()
            for pred, base, (W0, H0) in zip(preds, names, szs):
                sal = Image.fromarray((pred * 255).astype(np.uint8))
                sal = sal.resize((W0, H0), Image.BILINEAR)
                sal.save(os.path.join(out_dir, base + ".png"))


def load_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# ─── SUPERVISED BASELINES ──────────────────────────────────────────────────────

def train_supervised(model_name):
    import importlib
    from baselines.dataset import SODDataset
    from baselines.losses  import multi_scale_loss

    print(f"\n{'='*55}\n  {model_name} | Conjunctiva\n{'='*55}")

    mc     = MODEL_CONFIGS[model_name]
    mod    = importlib.import_module(mc["module"])
    model  = getattr(mod, mc["class"])(pretrained=True).to(DEVICE)
    opt    = optim.SGD(model.parameters(), lr=mc["lr"], momentum=0.9, weight_decay=mc["wd"])
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_ds  = SODDataset(DS_CFG["orig"], DS_CFG["label"], DS_CFG["train"],
                           orig_ext=DS_CFG["ext"], augment=True)
    train_dl  = DataLoader(ConcatDataset([train_ds] * REPEAT),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for imgs, lbls, _ in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            loss = multi_scale_loss(model(imgs), lbls)
            loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={total/len(train_dl):.4f} | {time.time()-t0:.0f}s")

    out_dir = os.path.join(SAVE_DIR, model_name)
    test_list = load_list(DS_CFG["test"])
    save_predictions(model, test_list, out_dir)
    mae, fb, iou = compute_metrics(out_dir, test_list)
    print(f"  TEST → MAE={mae:.4f}  Fβ={fb:.4f}  IoU={iou:.4f}")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}.pth"))
    return {"MAE": mae, "Fb": fb, "IoU": iou}


# ─── U-NETFLIM ─────────────────────────────────────────────────────────────────

def train_unet_flim():
    from baselines.models.unet_flim import build_unet_flim
    from baselines.dataset import SODDataset
    from baselines.losses  import multi_scale_loss
    from pyflim import data as flim_data, arch as flim_arch, flim as flim_module
    import torch.utils.data as tud

    print(f"\n{'='*55}\n  UNetFLIM | Conjunctiva\n{'='*55}")

    architecture = flim_arch.FLIMArchitecture(DS_CFG["arch"])
    flim_model   = flim_module.FLIMModel(
        architecture, decoder_type="vanilla_adaptive_decoder",
        adaptation_function="robust_weights", device="cpu",
        filter_by_size=False, track_gpu_stats=False
    )
    train_flim = flim_data.FLIMData(
        DS_CFG["orig"], images_list=DS_CFG["train"],
        marker_folder=DS_CFG["markers"], orig_ext=DS_CFG["ext"],
        marker_ext=DS_CFG["mext"],
        transform=flim_data.transforms.Compose([flim_data.Rescale(256), flim_data.ToTensor()])
    )
    sampler = tud.BatchSampler(tud.RandomSampler(train_flim), batch_size=4, drop_last=False)
    flim_dl = DataLoader(train_flim, batch_sampler=sampler)

    print("  [1/2] Treinando encoder FLIM...")
    flim_model.fit(flim_dl)
    torch.set_grad_enabled(True)
    print("  [2/2] Treinando decoder U-Net...")

    model  = build_unet_flim(DS_CFG["arch"], flim_model).to(DEVICE)
    opt    = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=0.01, weight_decay=5e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_ds = SODDataset(DS_CFG["orig"], DS_CFG["label"], DS_CFG["train"],
                          orig_ext=DS_CFG["ext"], augment=True)
    train_dl = DataLoader(ConcatDataset([train_ds] * REPEAT),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for imgs, lbls, _ in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            from baselines.losses import multi_scale_loss
            loss = multi_scale_loss(model(imgs), lbls)
            loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={total/len(train_dl):.4f} | {time.time()-t0:.0f}s")

    out_dir   = os.path.join(SAVE_DIR, "UNetFLIM")
    test_list = load_list(DS_CFG["test"])
    save_predictions(model, test_list, out_dir)
    mae, fb, iou = compute_metrics(out_dir, test_list)
    print(f"  TEST → MAE={mae:.4f}  Fβ={fb:.4f}  IoU={iou:.4f}")
    return {"MAE": mae, "Fb": fb, "IoU": iou}


# ─── FLIM DECODERS ─────────────────────────────────────────────────────────────

def train_flim_decoder(decoder_name, decoder_type):
    from pyflim import data as flim_data, arch as flim_arch, flim as flim_module, util
    import torch.utils.data as tud

    print(f"\n{'='*55}\n  {decoder_name} | Conjunctiva\n{'='*55}")

    architecture = flim_arch.FLIMArchitecture(DS_CFG["arch"])
    flim_model   = flim_module.FLIMModel(
        architecture, decoder_type=decoder_type,
        adaptation_function="robust_weights", device="cpu",
        filter_by_size=False, track_gpu_stats=False
    )
    train_ds = flim_data.FLIMData(
        DS_CFG["orig"], images_list=DS_CFG["train"],
        marker_folder=DS_CFG["markers"], orig_ext=DS_CFG["ext"],
        marker_ext=DS_CFG["mext"],
        transform=flim_data.transforms.Compose([flim_data.Rescale(256), flim_data.ToTensor()])
    )
    sampler = tud.BatchSampler(tud.RandomSampler(train_ds), batch_size=4, drop_last=False)
    loader  = DataLoader(train_ds, batch_sampler=sampler)

    print("  Treinando FLIM (fit)...")
    flim_model.fit(loader)
    torch.set_grad_enabled(True)

    # Inferência
    out_dir   = os.path.join(SAVE_DIR, decoder_name)
    os.makedirs(out_dir, exist_ok=True)
    test_list = load_list(DS_CFG["test"])

    test_ds = flim_data.FLIMData(
        DS_CFG["orig"], images_list=DS_CFG["test"], orig_ext=DS_CFG["ext"],
        transform=flim_data.transforms.Compose([flim_data.Rescale(256), flim_data.ToTensor()])
    )
    test_sampler = tud.BatchSampler(tud.SequentialSampler(test_ds), batch_size=1, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_sampler=test_sampler)

    flim_model.run(test_loader, out_dir, decoder_layer=-1)
    torch.set_grad_enabled(True)

    mae, fb, iou = compute_metrics(out_dir, test_list)
    print(f"  TEST → MAE={mae:.4f}  Fβ={fb:.4f}  IoU={iou:.4f}")
    return {"MAE": mae, "Fb": fb, "IoU": iou}


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_json = os.path.join(SAVE_DIR, "conjunctiva_results.json")
    results  = {}
    if os.path.isfile(out_json):
        with open(out_json) as f:
            results = json.load(f)

    # Supervised baselines
    for name in MODEL_CONFIGS:
        if name in results:
            print(f"  [SKIP] {name} — já calculado")
            continue
        results[name] = train_supervised(name)
        with open(out_json, "w") as f: json.dump(results, f, indent=2)

    # UNetFLIM
    if "UNetFLIM" not in results:
        results["UNetFLIM"] = train_unet_flim()
        with open(out_json, "w") as f: json.dump(results, f, indent=2)
    else:
        print("  [SKIP] UNetFLIM — já calculado")

    # FLIM decoders
    for dname, dtype in FLIM_DECODERS.items():
        if dname in results:
            print(f"  [SKIP] {dname} — já calculado")
            continue
        results[dname] = train_flim_decoder(dname, dtype)
        with open(out_json, "w") as f: json.dump(results, f, indent=2)

    # Tabela final
    print("\n" + "="*65)
    print(f"{'Model':<12} {'MAE':>8} {'Fβ':>8} {'IoU':>8}")
    print("-"*40)
    for m, v in results.items():
        print(f"{m:<12} {v['MAE']:>8.4f} {v['Fb']:>8.4f} {v['IoU']:>8.4f}")
    print("="*65)
    print(f"\nResultados salvos em {out_json}")


if __name__ == "__main__":
    main()
