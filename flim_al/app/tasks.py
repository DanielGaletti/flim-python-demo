"""
tasks.py — Gerenciamento de tarefas de treino em background
============================================================
Usa asyncio + threading para rodar treinos FLIM sem bloquear a API.
O estado de progresso é mantido em memória e pode ser consultado via SSE.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from config import (
    DEFAULT_ARCH, DEFAULT_N_BG, DEFAULT_N_FG,
    DEFAULT_N_COMMITTEE, DEFAULT_PROXY_LAYER,
    DT_BIN, DT_AVAILABLE, REPO_ROOT, FLIMPY_PATH,
)
from models import MethodProgress, TrainResult, TrainStatus
from session import (
    get_session, markers_dir, models_dir, orig_dir,
    label_dir, saliencies_dir, results_dir,
)

# Adiciona paths do FLIM ao sys.path
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLIMPY_PATH))


# ── Store de progresso em memória ─────────────────────────────────────────────
# session_id → dict com estado do treino atual
_TRAIN_STATE: dict[str, dict] = {}
_TRAIN_STATE_LOCK = threading.Lock()


def get_train_state(session_id: str) -> dict | None:
    return _TRAIN_STATE.get(session_id)


def _set_state(session_id: str, **kwargs) -> None:
    with _TRAIN_STATE_LOCK:
        if session_id not in _TRAIN_STATE:
            _TRAIN_STATE[session_id] = {
                "overall_status":  TrainStatus.PENDING,
                "current_method":  None,
                "current_budget":  None,
                "results":         [],
                "started_at":      time.time(),
                "error":           None,
            }
        _TRAIN_STATE[session_id].update(kwargs)


# ── Worker principal ──────────────────────────────────────────────────────────

def run_training_worker(
    session_id: str,
    methods:     list[str],
    budgets:     list[int],
    n_init:      int,
    n_committee: int,
    use_dt:      bool,
    on_progress: Callable | None = None,
) -> None:
    """
    Roda em thread separada. Importa e chama o pipeline FLIM diretamente.
    Atualiza _TRAIN_STATE conforme progride.

    TODO: implementar corpo da função usando:
      - flim_al.benchmark.run_benchmark() (já implementado)
      - OU chamar as funções core diretamente:
          retrain_encoder, evaluate_all_decoders, score_saliencies,
          create_combined_marker_dir, create_combined_region_marker_dir, etc.
    """
    _set_state(session_id, overall_status=TrainStatus.RUNNING)
    started = time.time()

    try:
        _, session_dir = get_session(session_id)

        # Importa pipeline FLIM
        from flim_al.al_encoder_experiment import (
            retrain_encoder, evaluate_all_decoders,
            generate_pool_saliencies, score_saliencies,
            score_saliencies_bald, build_committee_saliencies,
            EVAL_DECODERS,
        )
        from flim_al.marker_generator import (
            setup_initial_markers_from_gt, create_combined_marker_dir,
        )
        from flim_al.region_al import (
            create_combined_region_marker_dir,
            create_combined_region_marker_dir_bald,
        )

        images_list = sorted(
            f for f in orig_dir(session_dir).iterdir() if f.suffix == ".png"
        )
        pool     = [f.name for f in images_list[n_init:]]
        init_imgs = [f.name for f in images_list[:n_init]]
        val      = pool  # sem split separado: usa pool como validação também
                         # TODO: separar val set explicitamente

        # ── Arch file ─────────────────────────────────────────────────────────
        arch_file = str(DEFAULT_ARCH) if DEFAULT_ARCH.exists() else _create_default_arch(session_dir)

        # ── Markers iniciais ──────────────────────────────────────────────────
        _set_state(session_id, current_method="setup", current_budget=None)
        # TODO: chamar setup_initial_markers

        # ── Encoder baseline ──────────────────────────────────────────────────
        base_enc = models_dir(session_dir) / "base_encoder.pth"
        if not base_enc.exists():
            retrain_encoder(
                arch_file, str(markers_dir(session_dir)),
                str(orig_dir(session_dir)), str(label_dir(session_dir)),
                "cpu", str(base_enc),
            )

        # ── Saliency maps ─────────────────────────────────────────────────────
        sal_dir = str(saliencies_dir(session_dir))
        generate_pool_saliencies(str(base_enc), pool, str(orig_dir(session_dir)),
                                  "cpu", sal_dir, proxy_layer=DEFAULT_PROXY_LAYER)

        # ── Loop por método × budget ──────────────────────────────────────────
        results = []
        committee_cache: dict = {}

        for method in methods:
            for budget in sorted(budgets):
                _set_state(session_id,
                           current_method=method,
                           current_budget=budget)
                t0 = time.time()

                try:
                    # TODO: implementar seleção + marker generation + retrain + eval
                    # (usar as funções do benchmark.py como referência)
                    pass

                    r = TrainResult(
                        method=method, budget=budget,
                        fb=0.0, iou=0.0, dice=0.0,
                        elapsed_s=time.time() - t0,
                    )
                    results.append(r)
                    _set_state(session_id, results=[x.dict() for x in results])

                    if on_progress:
                        on_progress(session_id, method, budget, r)

                except Exception as e:
                    traceback.print_exc()
                    _set_state(session_id, error=str(e))

        _set_state(session_id,
                   overall_status=TrainStatus.DONE,
                   current_method=None,
                   current_budget=None,
                   elapsed_total_s=time.time() - started)

    except Exception as e:
        traceback.print_exc()
        _set_state(session_id,
                   overall_status=TrainStatus.ERROR,
                   error=str(e))


def _create_default_arch(session_dir: Path) -> str:
    """Cria arch2D.json padrão se o schisto não estiver disponível."""
    import json
    arch = {
        "stdev_factor": 0.01, "nlayers": 4, "apply_intrinsic_atrous": False,
        **{f"layer{i}": {
            "conv": {"kernel_size": [3,3,0], "nkernels_per_marker": 3,
                     "dilation_rate": [1,1,0], "nkernels_per_image": 10000,
                     "noutput_channels": 200},
            "relu": True,
            "pooling": {"type": "avg_pool", "size": [3,3,0], "stride": 2},
        } for i in range(1, 5)},
    }
    p = session_dir / "arch2D.json"
    with open(p, "w") as f:
        json.dump(arch, f, indent=2)
    return str(p)


# ── Lançamento do treino ──────────────────────────────────────────────────────

def start_training(
    session_id: str,
    methods:     list[str],
    budgets:     list[int],
    n_init:      int       = 3,
    n_committee: int       = DEFAULT_N_COMMITTEE,
    use_dt:      bool      = False,
) -> None:
    """Lança treino em thread daemon."""
    t = threading.Thread(
        target=run_training_worker,
        args=(session_id, methods, budgets, n_init, n_committee, use_dt),
        daemon=True,
    )
    t.start()
