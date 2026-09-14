"""
Experimento FLIM — BRATS (BraTS2021, tumor segmentation)
3 splits × 4 imagens de treino, avaliados no test set (1877 imagens).
"""
import os, time
import torch
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader
from pyflim import flim, arch, data, util
from sklearn.metrics import mean_absolute_error

ARCH_FILE     = "arch_best_brats.json"
BRATS         = "data/brats/"
ORIG_FOLDER   = BRATS + "orig/"
MARKER_FOLDER = BRATS + "markers/"
LABEL_FOLDER  = BRATS + "label/"
ORIG_EXT      = ".png"
MARKER_EXT    = "-seeds.txt"

SPLITS = [
    ("split1", BRATS+"Splits-50_50/split1-train.txt", BRATS+"Splits-50_50/split1-test.txt", "out_brats_split1/"),
    ("split2", BRATS+"Splits-50_50/split2-train.txt", BRATS+"Splits-50_50/split2-test.txt", "out_brats_split2/"),
    ("split3", BRATS+"Splits-50_50/split3-train.txt", BRATS+"Splits-50_50/split3-test.txt", "out_brats_split3/"),
]


def build_train_loader(train_list):
    ds = data.FLIMData(ORIG_FOLDER, images_list=train_list, marker_folder=MARKER_FOLDER,
                       orig_ext=ORIG_EXT, marker_ext=MARKER_EXT,
                       transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()]))
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds), batch_size=4, drop_last=False)
    return DataLoader(ds, batch_sampler=sampler)


def build_test_loader(test_list):
    ds = data.FLIMData(ORIG_FOLDER, images_list=test_list, orig_ext=ORIG_EXT,
                       transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()]))
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(ds), batch_size=1, drop_last=False)
    return DataLoader(ds, batch_sampler=sampler)


def compute_metrics(output_folder, label_folder, file_list):
    result_ext = "." + util.get_files_extension(output_folder)
    label_ext  = "." + util.get_files_extension(label_folder)
    maes, fbs, dices = [], [], []
    skipped = 0
    for fname in file_list:
        fname = str(fname).strip().split(".")[0]
        rp = output_folder + fname + result_ext
        lp = label_folder  + fname + label_ext
        if not os.path.isfile(rp) or not os.path.isfile(lp):
            skipped += 1; continue
        sal = io.imread(rp); sal = sal[:,:,0] if sal.ndim==3 else sal
        lbl = io.imread(lp).astype(np.uint8); lbl = lbl[:,:,0] if lbl.ndim==3 else lbl
        lbl[lbl > 0] = 1
        thr = threshold_otsu(sal)
        b   = (sal > thr).astype(np.uint8)
        mae = mean_absolute_error(b.flatten().astype(np.float32), lbl.flatten().astype(np.float32))
        maes.append(mae)
        tp  = np.sum(b*lbl); fp = np.sum(b*(1-lbl)); fn = np.sum((1-b)*lbl)
        p   = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
        fbs.append(1.3*p*r/(0.3*p+r+1e-8))
        denom = np.sum(b) + np.sum(lbl)
        dices.append((2*tp/denom) if denom > 0 else 1.0)
    print(f"  Avaliados: {len(maes)} | Pulados: {skipped}")
    return {
        "MAE":  np.mean(maes)  if maes  else float("nan"),
        "Fb":   np.mean(fbs)   if fbs   else float("nan"),
        "DICE": np.mean(dices) if dices else float("nan"),
    }


def run_split(name, train_list, test_list, output_folder):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    architecture = arch.FLIMArchitecture(ARCH_FILE)
    model = flim.FLIMModel(architecture, decoder_type="decoder_3",
                           adaptation_function="robust_weights", device="cpu",
                           filter_by_size=False, track_gpu_stats=False)
    t0 = time.time()
    model.fit(build_train_loader(train_list))
    print(f"  Treino: {time.time()-t0:.1f}s")
    t0 = time.time()
    model.run(build_test_loader(test_list), output_folder)
    print(f"  Inferência: {time.time()-t0:.1f}s")
    file_list = util.readFileList(test_list)
    result = compute_metrics(output_folder, LABEL_FOLDER, file_list)
    print(f"  MAE={result['MAE']:.4f}  Fβ={result['Fb']:.4f}  DICE={result['DICE']:.4f}")
    return result


if __name__ == "__main__":
    results = {}
    for name, train_list, test_list, out in SPLITS:
        results[name] = run_split(name, train_list, test_list, out)

    print(f"\n{'='*65}")
    print(f"{'Split':<12} {'MAE':>8} {'Fβ':>8} {'DICE':>8}")
    print(f"{'-'*65}")
    for name, r in results.items():
        print(f"{name:<12} {r['MAE']:>8.4f} {r['Fb']:>8.4f} {r['DICE']:>8.4f}")

    maes  = [r['MAE']  for r in results.values()]
    fbs   = [r['Fb']   for r in results.values()]
    dices = [r['DICE'] for r in results.values()]
    print(f"{'MEAN':<12} {np.mean(maes):>8.4f} {np.mean(fbs):>8.4f} {np.mean(dices):>8.4f}")
    print(f"{'='*65}")
    print("\nReferência paper TABLE IV (FLIMpb — BRATS):")
    print(f"  MAE≈0.xx  Fβ≈0.xxx  (consultar tabela do artigo)")
