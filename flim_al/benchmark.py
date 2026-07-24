"""
benchmark.py — Benchmark multi-dataset para FLIM AL
=====================================================
Roda todos os métodos AL em qualquer dataset compatível:
  dataset_path/
    orig/    ← imagens PNG
    label/   ← máscaras binárias PNG (mesmo nome que orig)
    markers/ ← opcional: seeds.txt iniciais (auto-gerado se ausente)

Uso (dentro de flim_ad/):
  python3 ../flim_al/benchmark.py \\
      --dataset_path /path/to/dataset \\
      --save_dir out/benchmark \\
      --budgets 3 5 10 \\
      --dt_bin libs/ift/bin/iftSMansoniDelineation

  # Sem DT (mais rápido):
  python3 ../flim_al/benchmark.py \\
      --dataset_path /path/to/dataset \\
      --methods no_al entropy region_entropy \\
      --budgets 3 5 10

  # Só alguns métodos:
  python3 ../flim_al/benchmark.py \\
      --dataset_path /path/to/dataset \\
      --methods no_al entropy region_entropy region_bald \\
      --n_committee 3

Métodos disponíveis:
  no_al            — seleção aleatória (baseline)
  entropy          — entropia da saliency map
  least_confidence — 1 - max(p, 1-p)
  margin           — diferença entre as duas classes mais prováveis
  region_entropy   — entropy por regiões SLIC (marker-level)
  region_bald      — BALD com comitê de N encoders por regiões
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parent.parent
FLIMPY = REPO / "flim_ad" / "libs" / "flim-python"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FLIMPY))

from flim_al.al_encoder_experiment import (
    EVAL_DECODERS,
    _ensure_dt_symlinks,
    build_committee_saliencies,
    evaluate_all_decoders,
    generate_pool_saliencies,
    retrain_encoder,
    score_saliencies,
    score_saliencies_bald,
)
from flim_al.marker_generator import (
    create_combined_marker_dir,
    generate_markers_from_gt,
    save_markers,
)
from flim_al.region_al import (
    create_combined_region_marker_dir,
    create_combined_region_marker_dir_bald,
)

# ── Arch padrão (fallback quando arch2D.json não encontrado) ──────────────────
DEFAULT_ARCH = {
    "stdev_factor": 0.01,
    "nlayers": 4,
    "apply_intrinsic_atrous": False,
    **{
        f"layer{i}": {
            "conv": {
                "kernel_size": [3, 3, 0],
                "nkernels_per_marker": 3,
                "dilation_rate": [1, 1, 0],
                "nkernels_per_image": 10000,
                "noutput_channels": 200,
            },
            "relu": True,
            "pooling": {"type": "avg_pool", "size": [3, 3, 0], "stride": 2},
        }
        for i in range(1, 5)
    },
}

# ── CSV ───────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "dataset", "split", "budget", "method", "acquisition",
    "decoder", "eval_mode", "fb", "dice", "mae", "iou",
]


def _load_csv(csv_path: str) -> tuple[list[dict], set]:
    rows, done = [], set()
    if not os.path.exists(csv_path):
        return rows, done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            done.add((
                row["split"], row["budget"], row["method"],
                row["decoder"], row.get("eval_mode", "otsu"),
            ))
    print(f"  [resume] {len(rows)} linhas no CSV")
    return rows, done


def _append_csv(csv_path: str, new_rows: list[dict]) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(new_rows)


# ── Dataset ───────────────────────────────────────────────────────────────────

def validate_dataset(dataset_path: str) -> tuple[list[str], str, str]:
    """Valida estrutura e retorna (all_fnames, orig_dir, label_dir)."""
    orig_dir  = os.path.join(dataset_path, "orig")
    label_dir = os.path.join(dataset_path, "label")

    for d in [orig_dir, label_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Diretório não encontrado: {d}")

    orig_fnames  = sorted(f for f in os.listdir(orig_dir)  if f.endswith(".png"))
    label_fnames = set(f for f in os.listdir(label_dir) if f.endswith(".png"))

    if not orig_fnames:
        raise ValueError(f"Nenhuma imagem PNG em {orig_dir}")

    missing = [f for f in orig_fnames if f not in label_fnames]
    if missing:
        print(f"  [aviso] {len(missing)} imagens sem label ignoradas")
        orig_fnames = [f for f in orig_fnames if f in label_fnames]

    print(f"  Dataset: {len(orig_fnames)} imagens com label")
    return orig_fnames, orig_dir, label_dir


def create_splits(
    all_fnames: list[str],
    n_splits: int = 1,
    val_ratio: float = 0.3,
    n_init: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Retorna lista de dicts com train/val/init/pool para cada split."""
    rng = random.Random(seed)
    splits = []
    for s in range(n_splits):
        shuffled = all_fnames.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val   = shuffled[:n_val]
        train = shuffled[n_val:]
        n_init_real = min(n_init, len(train))
        if n_init_real < n_init:
            print(f"  [aviso] Split {s+1}: só {len(train)} imgs de treino, n_init={n_init_real}")
        splits.append({
            "split": s + 1,
            "train": train,
            "val":   val,
            "init":  train[:n_init_real],
            "pool":  train[n_init_real:],
        })
    return splits


def setup_initial_markers(
    init_fnames: list[str],
    label_dir: str,
    markers_dir: str,
    existing_markers_dir: str | None = None,
    n_fg: int = 100,
    n_bg: int = 300,
) -> str:
    """
    Cria markers iniciais para as imagens seed.
    Usa existing_markers_dir se disponível, senão gera do GT.
    """
    os.makedirs(markers_dir, exist_ok=True)

    for fname in init_fnames:
        name     = os.path.splitext(fname)[0]
        out_path = os.path.join(markers_dir, f"{name}-seeds.txt")
        if os.path.exists(out_path):
            continue

        # 1. Tenta copiar de markers existentes
        if existing_markers_dir:
            src = os.path.join(existing_markers_dir, f"{name}-seeds.txt")
            if os.path.exists(src):
                shutil.copy2(src, out_path)
                continue

        # 2. Gera sintético do GT
        gt_path = os.path.join(label_dir, fname)
        if not os.path.exists(gt_path):
            print(f"  [aviso] GT não encontrado para {fname}")
            continue
        seeds = generate_markers_from_gt(
            gt_path, n_fg=n_fg, n_bg=n_bg, seed=hash(name) % 2**31
        )
        save_markers(seeds, out_path)

    n = len([f for f in os.listdir(markers_dir) if f.endswith("-seeds.txt")])
    print(f"  [markers] {n} markers iniciais em {markers_dir}")
    return markers_dir


def resolve_arch_file(arch_file_arg: str | None, save_dir: str) -> str:
    """Retorna path para arch2D.json válido."""
    if arch_file_arg and os.path.exists(arch_file_arg):
        return arch_file_arg

    candidates = [
        "data/schisto/user_A/split1/arch2D.json",
        str(REPO / "flim_ad" / "data" / "schisto" / "user_A" / "split1" / "arch2D.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"  [arch] Usando {c}")
            return c

    arch_path = os.path.join(save_dir, "arch2D.json")
    with open(arch_path, "w") as f:
        json.dump(DEFAULT_ARCH, f, indent=2)
    print(f"  [arch] Default criado em {arch_path}")
    return arch_path


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_method_scores(
    method: str,
    sal_dir: str,
    pool_fnames: list[str],
    arch_file: str,
    markers_dir: str,
    orig_dir: str,
    label_dir: str,
    device: str,
    n_committee: int,
    split_dir: str,
    split: int,
    proxy_layer: int,
    committee_cache: dict,
) -> tuple[list[str], list[float], list[int], list | None]:
    """
    Retorna (fnames, scores, ranking, committee_sal_dirs).
    Reutiliza committee já treinado via committee_cache.
    """
    if method == "region_bald":
        if "dirs" not in committee_cache:
            print(f"    Construindo comitê ({n_committee} encoders)...")
            committee_cache["dirs"] = build_committee_saliencies(
                arch_file, markers_dir, orig_dir, label_dir,
                device, n_committee, split_dir, split,
                pool_fnames, proxy_layer,
            )
        comm_dirs = committee_cache["dirs"]
        fnames_s, scores_s = score_saliencies_bald(comm_dirs, pool_fnames)
    else:
        # region_* e image-level usam o mesmo sal_dir
        fnames_s, scores_s = score_saliencies(sal_dir, device, method)

    ranking = sorted(range(len(fnames_s)), key=lambda i: scores_s[i], reverse=True)
    comm_dirs = committee_cache.get("dirs")
    return fnames_s, scores_s, ranking, comm_dirs


# ── Geração de markers por método ─────────────────────────────────────────────

def generate_al_markers(
    method: str,
    sel_fnames: list[str],
    markers_dir: str,
    label_dir: str,
    orig_dir: str,
    sal_dir: str,
    committee_sal_dirs: list | None,
    work_dir: str,
    n_fg: int,
    n_bg: int,
) -> str:
    """Gera diretório de markers para o método e imagens selecionadas."""
    out_dir = os.path.join(work_dir, f"{method}_markers")

    if method == "region_bald":
        assert committee_sal_dirs is not None
        create_combined_region_marker_dir_bald(
            markers_dir, sel_fnames, label_dir, orig_dir,
            committee_sal_dirs, out_dir,
            budget_per_image=20, n_superpixels=300,
            n_seeds_per_region=15, fg_threshold=0.15,
        )
    elif method.startswith("region_"):
        region_method = method.replace("region_", "")
        create_combined_region_marker_dir(
            markers_dir, sel_fnames, label_dir, orig_dir,
            sal_dir, out_dir,
            budget_per_image=20, n_superpixels=300,
            n_seeds_per_region=15, fg_threshold=0.15,
            method=region_method,
        )
    else:
        create_combined_marker_dir(
            markers_dir, sel_fnames, label_dir, out_dir,
            n_fg=n_fg, n_bg=n_bg,
        )

    return out_dir


# ── Tabela de resultados ──────────────────────────────────────────────────────

def print_summary(
    all_rows: list[dict],
    dataset_name: str,
    methods: list[str],
    budgets: list[int],
    eval_mode: str,
    target_decoder: str = "labeled_marker",
) -> None:
    """Imprime tabela comparativa Fβ × método × budget."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in all_rows:
        if (
            row.get("dataset") == dataset_name
            and row.get("decoder") == target_decoder
            and row.get("eval_mode", "otsu") == eval_mode
        ):
            grouped[row["method"]][row["budget"]].append(float(row.get("fb", 0)))

    budget_strs = sorted(
        set(b for m in grouped.values() for b in m.keys()),
        key=lambda x: int(x),
    )

    eval_label = eval_mode.upper()
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {dataset_name} | decoder={target_decoder} | eval={eval_label}")
    print(f"{'='*80}")

    col_w = 9
    header = f"{'Método':25s}" + "".join(f"{'K='+b:>{col_w}}" for b in budget_strs)
    print(header)
    print("─" * len(header))

    method_order = ["baseline", "no_al"] + [m for m in methods if m not in ("no_al",)]

    for method in method_order:
        if method not in grouped:
            continue
        vals    = grouped[method]
        row_str = f"{method:25s}"
        for b in budget_strs:
            vs = vals.get(b, [])
            v  = float(np.mean(vs)) if vs else float("nan")
            row_str += f"{v:>{col_w}.3f}"
        print(row_str)

    print("─" * len(header))

    # Melhor AL vs no_al por budget
    print("\n  Melhor AL por budget (vs no_al):")
    al_methods = [m for m in methods if m != "no_al"]
    for b in budget_strs:
        no_al_vs = grouped.get("no_al", {}).get(b, [])
        no_al_v  = float(np.mean(no_al_vs)) if no_al_vs else float("nan")

        best_m, best_v = None, -1.0
        for m in al_methods:
            vs = grouped.get(m, {}).get(b, [])
            v  = float(np.mean(vs)) if vs else -1.0
            if v > best_v:
                best_v, best_m = v, m

        if best_m and best_v >= 0:
            delta = best_v - no_al_v if not np.isnan(no_al_v) else float("nan")
            print(f"    K={b}: {best_m:20s}  Fβ={best_v:.3f}  "
                  f"no_al={no_al_v:.3f}  Δ={delta:+.3f}")


# ── Benchmark principal ───────────────────────────────────────────────────────

def run_benchmark(args: argparse.Namespace) -> str:
    device = args.device if torch.cuda.is_available() else "cpu"
    dataset_name = os.path.basename(os.path.abspath(args.dataset_path))

    print(f"\n{'='*70}")
    print(f"FLIM Benchmark: {dataset_name}")
    print(f"Device  : {device}")
    print(f"Methods : {args.methods}")
    print(f"Budgets : {args.budgets} | Splits: {args.n_splits} | n_init: {args.n_init}")
    print(f"{'='*70}\n")

    # ── Validação ────────────────────────────────────────────────────────────
    all_fnames, orig_dir, label_dir = validate_dataset(args.dataset_path)

    # ── Diretório de saída ────────────────────────────────────────────────────
    save_dir = os.path.join(args.save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── Arch file ────────────────────────────────────────────────────────────
    arch_file = resolve_arch_file(args.arch_file, save_dir)

    # ── Modos de avaliação ────────────────────────────────────────────────────
    eval_modes: list[str] = []
    use_dt_global = (
        not args.no_dt
        and args.dt_bin is not None
        and os.path.isfile(args.dt_bin)
    )
    if use_dt_global:
        eval_modes.append("dt")
        print(f"  [DT] {args.dt_bin}")
    if not args.dt_only:
        eval_modes.append("otsu")
    if not eval_modes:
        eval_modes = ["otsu"]

    primary_eval = eval_modes[0]  # para tabela final

    # ── CSV incremental ───────────────────────────────────────────────────────
    csv_path = os.path.join(save_dir, "benchmark_results.csv")
    all_rows, done_keys = _load_csv(csv_path)

    def _done(split: int, budget: int, method: str, decoder: str, eval_mode: str) -> bool:
        return (str(split), str(budget), method, decoder, eval_mode) in done_keys

    def _save(new_rows: list[dict]) -> None:
        all_rows.extend(new_rows)
        _append_csv(csv_path, new_rows)
        for r in new_rows:
            done_keys.add((
                str(r["split"]), str(r["budget"]), r["method"],
                r["decoder"], r.get("eval_mode", "otsu"),
            ))

    def _make_rows(
        split: int, budget: int, method: str, acquisition: str,
        decoder_results: dict[str, dict], eval_mode: str,
    ) -> list[dict]:
        rows = []
        for dec, m in decoder_results.items():
            rows.append({
                "dataset":     dataset_name,
                "split":       split,
                "budget":      budget,
                "method":      method,
                "acquisition": acquisition,
                "decoder":     dec,
                "eval_mode":   eval_mode,
                **{k: round(v, 4) for k, v in m.items()},
            })
        return rows

    # ── Splits ───────────────────────────────────────────────────────────────
    splits = create_splits(
        all_fnames, n_splits=args.n_splits,
        val_ratio=args.val_ratio, n_init=args.n_init, seed=args.seed,
    )

    for split_info in splits:
        split     = split_info["split"]
        val       = split_info["val"]
        init_imgs = split_info["init"]
        pool      = split_info["pool"]

        print(f"\n{'='*70}")
        print(f"Split {split}: {len(init_imgs)} init | {len(pool)} pool | {len(val)} val")
        print(f"{'='*70}")

        if not pool:
            print("  Pool vazio — aumenta o dataset ou reduz --n_init")
            continue

        split_dir = os.path.join(save_dir, f"split{split}")
        os.makedirs(split_dir, exist_ok=True)

        # ── Markers iniciais ─────────────────────────────────────────────────
        markers_dir = os.path.join(split_dir, "markers")
        setup_initial_markers(
            init_imgs, label_dir, markers_dir,
            existing_markers_dir=args.existing_markers,
            n_fg=args.n_fg, n_bg=args.n_bg,
        )

        # ── Encoder baseline ─────────────────────────────────────────────────
        base_enc_path = os.path.join(split_dir, "base_encoder.pth")
        if not os.path.exists(base_enc_path):
            print(f"\n  [baseline] Treinando encoder com {len(init_imgs)} imgs...")
            retrain_encoder(
                arch_file, markers_dir, orig_dir, label_dir,
                device, base_enc_path,
            )
        else:
            print(f"  [baseline] Encoder existe — skip")

        # ── Saliency maps do pool ────────────────────────────────────────────
        sal_dir = os.path.join(split_dir, "saliencies")
        n_sal   = (
            len([f for f in os.listdir(sal_dir) if f.endswith(".png")])
            if os.path.isdir(sal_dir) else 0
        )
        if n_sal < len(pool):
            print(f"  [saliency] Gerando para {len(pool)} imgs do pool...")
            generate_pool_saliencies(
                base_enc_path, pool, orig_dir, device, sal_dir,
                proxy_layer=args.proxy_layer,
            )
        else:
            print(f"  [saliency] {n_sal} saliencies existem — skip")

        # ── Avaliação do baseline (encoder inicial com n_init imgs) ──────────
        if use_dt_global:
            try:
                _ensure_dt_symlinks(args.dataset_path)
            except Exception as e:
                print(f"  [aviso] DT symlinks: {e}")

        for eval_mode in eval_modes:
            use_dt = eval_mode == "dt"
            first_dec = "labeled_marker"
            if not _done(split, args.n_init, "baseline", first_dec, eval_mode):
                print(f"\n  [baseline eval/{eval_mode.upper()}]...")
                try:
                    res = evaluate_all_decoders(
                        base_enc_path, val, orig_dir, label_dir, device,
                        use_dt=use_dt, dt_bin=args.dt_bin,
                        dataset_folder=args.dataset_path if use_dt else None,
                    )
                    _save(_make_rows(split, args.n_init, "baseline", "none", res, eval_mode))
                except Exception as e:
                    print(f"  [erro] baseline eval: {e}")
                    traceback.print_exc()

        # ── Pré-computa scores para cada método ──────────────────────────────
        method_data: dict[str, tuple] = {}  # method → (fnames, scores, ranking)
        committee_cache: dict = {}           # {"dirs": [...]} reutilizado entre métodos

        al_methods = [m for m in args.methods if m != "no_al"]
        for method in al_methods:
            print(f"\n  [scoring/{method}]...")
            try:
                fnames_s, scores_s, ranking, comm_dirs = compute_method_scores(
                    method=method,
                    sal_dir=sal_dir,
                    pool_fnames=pool,
                    arch_file=arch_file,
                    markers_dir=markers_dir,
                    orig_dir=orig_dir,
                    label_dir=label_dir,
                    device=device,
                    n_committee=args.n_committee,
                    split_dir=split_dir,
                    split=split,
                    proxy_layer=args.proxy_layer,
                    committee_cache=committee_cache,
                )
                method_data[method] = (fnames_s, scores_s, ranking)
                if comm_dirs:
                    committee_cache["dirs"] = comm_dirs
                print(f"    top-3: {[fnames_s[i] for i in ranking[:3]]}")
            except Exception as e:
                print(f"    [erro] scoring {method}: {e}")
                traceback.print_exc()

        # ── Loop por budget ───────────────────────────────────────────────────
        budgets_valid = sorted(b for b in args.budgets if b <= len(pool))
        if not budgets_valid:
            budgets_valid = [len(pool)]

        for budget in budgets_valid:
            print(f"\n  {'─'*60}\n  Budget = {budget}")
            work_dir = tempfile.mkdtemp(prefix=f"flim_bench_s{split}_K{budget}_")

            try:
                # ── no_al (random) ───────────────────────────────────────────
                if "no_al" in args.methods:
                    _run_no_al(
                        split=split, budget=budget,
                        pool=pool, val=val,
                        markers_dir=markers_dir, label_dir=label_dir,
                        orig_dir=orig_dir, arch_file=arch_file,
                        device=device, work_dir=work_dir, split_dir=split_dir,
                        eval_modes=eval_modes, args=args,
                        dataset_name=dataset_name,
                        done_fn=_done, save_fn=_save, make_rows_fn=_make_rows,
                        all_rows=all_rows,
                    )

                # ── AL methods ───────────────────────────────────────────────
                for method in al_methods:
                    if method not in method_data:
                        print(f"\n    [{method}] sem scores — skip")
                        continue

                    fnames_s, _, ranking = method_data[method]
                    sel_fnames = [fnames_s[i] for i in ranking[:budget]]
                    print(f"\n    [{method}] selecionadas: {sel_fnames[:3]}...")

                    enc_path = os.path.join(split_dir, method, f"K{budget}", "encoder.pth")
                    os.makedirs(os.path.dirname(enc_path), exist_ok=True)

                    # Gera markers
                    try:
                        al_marker_dir = generate_al_markers(
                            method=method,
                            sel_fnames=sel_fnames,
                            markers_dir=markers_dir,
                            label_dir=label_dir,
                            orig_dir=orig_dir,
                            sal_dir=sal_dir,
                            committee_sal_dirs=committee_cache.get("dirs"),
                            work_dir=work_dir,
                            n_fg=args.n_fg,
                            n_bg=args.n_bg,
                        )
                    except Exception as e:
                        print(f"    [erro] markers {method}: {e}")
                        traceback.print_exc()
                        continue

                    # Treina encoder
                    if not os.path.exists(enc_path):
                        try:
                            retrain_encoder(
                                arch_file, al_marker_dir,
                                orig_dir, label_dir, device, enc_path,
                            )
                        except Exception as e:
                            print(f"    [erro] treino {method}: {e}")
                            traceback.print_exc()
                            continue
                    else:
                        print(f"    [resume] encoder existe")

                    # Avalia
                    for eval_mode in eval_modes:
                        first_dec = "labeled_marker"
                        if _done(split, budget, method, first_dec, eval_mode):
                            print(f"    [resume] {method}/{eval_mode} skip")
                            continue
                        use_dt = eval_mode == "dt"
                        try:
                            res = evaluate_all_decoders(
                                enc_path, val, orig_dir, label_dir, device,
                                use_dt=use_dt, dt_bin=args.dt_bin,
                                dataset_folder=args.dataset_path if use_dt else None,
                            )
                            _save(_make_rows(split, budget, method, method, res, eval_mode))
                        except Exception as e:
                            print(f"    [erro] eval {method}/{eval_mode}: {e}")
                            traceback.print_exc()

            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

    # ── Tabela final ──────────────────────────────────────────────────────────
    print_summary(
        all_rows=all_rows,
        dataset_name=dataset_name,
        methods=args.methods,
        budgets=[str(b) for b in args.budgets],
        eval_mode=primary_eval,
    )
    print(f"\nCSV: {csv_path}")
    return csv_path


def _run_no_al(
    split: int,
    budget: int,
    pool: list[str],
    val: list[str],
    markers_dir: str,
    label_dir: str,
    orig_dir: str,
    arch_file: str,
    device: str,
    work_dir: str,
    split_dir: str,
    eval_modes: list[str],
    args: argparse.Namespace,
    dataset_name: str,
    done_fn,
    save_fn,
    make_rows_fn,
    all_rows: list[dict],
) -> None:
    """Roda seleção aleatória com n_rand_seeds e salva média."""
    print(f"\n    [no_al] {args.n_rand_seeds} seeds...")

    # Acumulador separado por eval_mode
    acc_by_mode: dict[str, dict[str, dict[str, list]]] = {
        em: defaultdict(lambda: defaultdict(list)) for em in eval_modes
    }

    for rs in range(args.n_rand_seeds):
        seed_method = f"no_al_seed{rs}"
        rng         = random.Random(args.seed + rs)
        sel_fnames  = rng.sample(pool, min(budget, len(pool)))

        enc_path    = os.path.join(split_dir, "no_al", f"K{budget}", f"seed{rs}", "encoder.pth")
        os.makedirs(os.path.dirname(enc_path), exist_ok=True)
        marker_dir  = os.path.join(work_dir, f"no_al_seed{rs}_markers")

        encoder_ready = os.path.exists(enc_path)

        for eval_mode in eval_modes:
            first_dec = "labeled_marker"
            if done_fn(split, budget, seed_method, first_dec, eval_mode):
                # Carrega do CSV para acumular
                for row in all_rows:
                    if (
                        str(row["split"]) == str(split)
                        and str(row["budget"]) == str(budget)
                        and row["method"] == seed_method
                        and row.get("eval_mode", "otsu") == eval_mode
                    ):
                        for k in ("fb", "dice", "mae", "iou"):
                            if k in row:
                                acc_by_mode[eval_mode][row["decoder"]][k].append(float(row[k]))
                continue

            # Treina uma vez (compartilhado entre eval_modes)
            if not encoder_ready:
                create_combined_marker_dir(
                    markers_dir, sel_fnames, label_dir, marker_dir,
                    n_fg=args.n_fg, n_bg=args.n_bg,
                )
                try:
                    retrain_encoder(arch_file, marker_dir, orig_dir, label_dir, device, enc_path)
                    encoder_ready = True
                except Exception as e:
                    print(f"      [erro] treino seed{rs}: {e}")
                    break  # skip todas as eval_modes para este seed

            use_dt = eval_mode == "dt"
            try:
                res = evaluate_all_decoders(
                    enc_path, val, orig_dir, label_dir, device,
                    use_dt=use_dt, dt_bin=args.dt_bin,
                    dataset_folder=args.dataset_path if use_dt else None,
                )
                save_fn(make_rows_fn(split, budget, seed_method, "random", res, eval_mode))
                for dec, m in res.items():
                    for k, v in m.items():
                        acc_by_mode[eval_mode][dec][k].append(v)
            except Exception as e:
                print(f"      [erro] eval seed{rs}/{eval_mode}: {e}")
                traceback.print_exc()

    # Salva médias
    for eval_mode in eval_modes:
        first_dec = "labeled_marker"
        if done_fn(split, budget, "no_al", first_dec, eval_mode):
            continue
        avg_rows = []
        for dec, acc in acc_by_mode[eval_mode].items():
            if acc.get("fb"):
                m = {k: float(np.mean(v)) for k, v in acc.items()}
                avg_rows.append({
                    "dataset":     dataset_name,
                    "split":       split,
                    "budget":      budget,
                    "method":      "no_al",
                    "acquisition": "random",
                    "decoder":     dec,
                    "eval_mode":   eval_mode,
                    **{k: round(v, 4) for k, v in m.items()},
                })
        if avg_rows:
            save_fn(avg_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark multi-dataset para FLIM Active Learning"
    )

    # Dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset_path",     required=True,
                   help="Dir com orig/ e label/")
    g.add_argument("--existing_markers", default=None,
                   help="Dir com seeds.txt existentes para as imgs iniciais")
    g.add_argument("--arch_file",        default=None,
                   help="arch2D.json (auto-detectado se omitido)")

    # Splits
    g = p.add_argument_group("Splits")
    g.add_argument("--n_splits",   type=int,   default=1)
    g.add_argument("--val_ratio",  type=float, default=0.3)
    g.add_argument("--n_init",     type=int,   default=3,
                   help="Imagens anotadas inicialmente (seed)")
    g.add_argument("--seed",       type=int,   default=42)

    # AL
    g = p.add_argument_group("Active Learning")
    g.add_argument("--methods", nargs="+",
                   default=["no_al", "entropy", "least_confidence", "margin",
                            "region_entropy", "region_bald"],
                   choices=["no_al", "entropy", "least_confidence", "margin",
                            "region_entropy", "region_bald", "coreset", "badge"],
                   help="Métodos a comparar")
    g.add_argument("--budgets",      nargs="+", type=int, default=[3, 5, 10],
                   help="Budgets K (imgs adicionais a anotar)")
    g.add_argument("--n_rand_seeds", type=int,  default=3,
                   help="Seeds para média do no_al/random")
    g.add_argument("--n_committee",  type=int,  default=3,
                   help="Tamanho do comitê para region_bald")
    g.add_argument("--proxy_layer",  type=int,  default=3)

    # Markers
    g = p.add_argument_group("Markers sintéticos")
    g.add_argument("--n_fg", type=int, default=100, help="Seeds de foreground por img")
    g.add_argument("--n_bg", type=int, default=300, help="Seeds de background por img")

    # Avaliação
    g = p.add_argument_group("Avaliação")
    g.add_argument("--device",   default="cpu")
    g.add_argument("--save_dir", default="out/benchmark",
                   help="Dir base de saída (dataset_name/ criado dentro)")
    g.add_argument("--dt_bin",   default=None,
                   help="Path para iftSMansoniDelineation (opcional)")
    g.add_argument("--no_dt",    action="store_true",
                   help="Ignora DT mesmo com --dt_bin")
    g.add_argument("--dt_only",  action="store_true",
                   help="Só avalia com DT (omite Otsu+AF)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
