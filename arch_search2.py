"""
Architecture search round 2 — foco nos padrões promissores do round 1.
Parte do best: dilations=[1,3,5,7], channels=[12,10,8,6], kpm=5 → Fβ=0.4704
"""

import os, json, itertools
import torch
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader
from pyflim import flim, arch, data, util

ORIG_FOLDER   = "data/orig/"
MARKER_FOLDER = "data/markers/"
ORIG_EXT      = ".png"
MARKER_EXT    = "-seeds.txt"
LABEL_FOLDER  = "data/label/"
TRAIN_LIST    = "split1-train.txt"
VAL_LIST      = "split1-val.txt"
TMP_OUT       = "/tmp/arch_search2_out/"
os.makedirs(TMP_OUT, exist_ok=True)

# Expandir ao redor do melhor resultado
DILATION_CONFIGS = [
    [1, 2, 4, 7],
    [1, 3, 5, 7],
    [1, 3, 6, 9],
    [1, 2, 5, 9],
    [1, 3, 5, 7, 9],   # 5 camadas
    [1, 2, 4, 6, 9],
]

CHANNEL_CONFIGS = [
    [12, 10, 8, 6],
    [16, 12, 8, 6],
    [14, 12, 10, 8],
    [20, 16, 12, 8],
    [12, 10, 8, 6, 4],  # 5 camadas
    [16, 12, 10, 8, 6],
]

KERNELS_PER_MARKER = [5, 8, 10, 15]


def make_arch_json(dilations, channels, kpm, path):
    cfg = {"nlayers": len(dilations), "stdev_factor": 0.01}
    pool_types = ["max_pool", "avg_pool", "max_pool", "avg_pool", "max_pool"]
    pool_sizes = [3, 3, 5, 5, 5]
    for i, (d, c) in enumerate(zip(dilations, channels)):
        ps = pool_sizes[min(i, len(pool_sizes)-1)]
        cfg[f"layer{i+1}"] = {
            "conv": {
                "dilation_rate": [d, d, 0],
                "kernel_size":   [3, 3, 0],
                "nkernels_per_image":  100000,
                "nkernels_per_marker": kpm,
                "noutput_channels":    c
            },
            "pooling": {
                "size":   [ps, ps, 0],
                "stride": 1,
                "type":   pool_types[min(i, len(pool_types)-1)]
            },
            "relu": True
        }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)


def build_loader(list_file, with_markers=True, bs=5):
    kwargs = dict(
        orig_folder=ORIG_FOLDER,
        images_list=list_file,
        orig_ext=ORIG_EXT,
        transform=data.transforms.Compose([data.Rescale(256), data.ToTensor()])
    )
    if with_markers:
        kwargs["marker_folder"] = MARKER_FOLDER
        kwargs["marker_ext"]    = MARKER_EXT
    ds = data.FLIMData(**kwargs)
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(ds) if not with_markers
        else torch.utils.data.sampler.RandomSampler(ds),
        batch_size=bs if with_markers else 1, drop_last=False
    )
    return DataLoader(ds, batch_sampler=sampler)


def eval_fb(output_folder, file_list):
    ext = "." + util.get_files_extension(output_folder)
    lext = "." + util.get_files_extension(LABEL_FOLDER)
    fbs = []
    for fname in file_list:
        fname = str(fname).strip().split(".")[0]
        rp = output_folder + fname + ext
        lp = LABEL_FOLDER  + fname + lext
        if not os.path.isfile(rp) or not os.path.isfile(lp):
            continue
        sal = io.imread(rp)
        if sal.ndim == 3: sal = sal[:,:,0]
        lbl = io.imread(lp).astype(np.uint8)
        if lbl.ndim == 3: lbl = lbl[:,:,0]
        lbl[lbl > 0] = 1
        thr = threshold_otsu(sal)
        b = (sal > thr).astype(np.uint8)
        tp = np.sum(b * lbl)
        fp = np.sum(b * (1-lbl))
        fn = np.sum((1-b) * lbl)
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        fbs.append(1.3*p*r / (0.3*p + r + 1e-8))
    return np.mean(fbs) if fbs else 0.0


if __name__ == "__main__":
    val_files = util.readFileList(VAL_LIST)[:100]
    train_ds  = build_loader(TRAIN_LIST, with_markers=True, bs=5)

    # Carregar best atual
    best_fb  = 0.4704
    best_cfg = ([1,3,5,7], [12,10,8,6], 5)

    configs = [(d,c,k) for d,c,k in itertools.product(DILATION_CONFIGS, CHANNEL_CONFIGS, KERNELS_PER_MARKER)
               if len(d) == len(c)]

    print(f"Round 2: {len(configs)} configurações | best atual = {best_fb:.4f}\n")

    for i, (dilations, channels, kpm) in enumerate(configs):
        tag = f"d{dilations}_c{channels}_k{kpm}"
        out = TMP_OUT + tag.replace(" ","").replace(",","") + "/"
        os.makedirs(out, exist_ok=True)

        make_arch_json(dilations, channels, kpm, "/tmp/tmp_arch2.json")
        architecture = arch.FLIMArchitecture("/tmp/tmp_arch2.json")
        model = flim.FLIMModel(architecture, decoder_type="decoder_3",
                               adaptation_function="robust_weights",
                               device="cpu", filter_by_size=False, track_gpu_stats=False)
        model.fit(train_ds)

        val_ds = build_loader(VAL_LIST, with_markers=False, bs=1)
        model.run(val_ds, out)

        fb = eval_fb(out, val_files)

        marker = " ← NOVO BEST" if fb > best_fb else ""
        print(f"[{i+1:3d}/{len(configs)}] Fβ={fb:.4f}  {tag}{marker}")

        if fb > best_fb:
            best_fb  = fb
            best_cfg = (dilations, channels, kpm)
            make_arch_json(dilations, channels, kpm, "arch_best.json")

    print(f"\n{'='*65}")
    print(f"MELHOR FINAL: Fβ={best_fb:.4f}")
    print(f"  dilations={best_cfg[0]}  channels={best_cfg[1]}  kpm={best_cfg[2]}")
    print(f"  Salvo em arch_best.json")
