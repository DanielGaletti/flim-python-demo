"""
session.py — Gerenciamento de sessões (dataset + estado)
=========================================================
Cada sessão é um diretório em WORKSPACE_DIR/<session_id>/ com:
  images/       ← imagens originais upadas (aceito pelo usuário como "images/")
  orig/         ← symlink ou cópia de images/ (formato esperado pelo FLIM)
  label/        ← máscaras binárias geradas pelo usuário
  markers/      ← seeds FLIM gerados do label
  saliencies/   ← saliency maps do pool
  models/       ← encoders treinados por método/budget
  results/      ← CSVs de métricas
  meta.json     ← estado da sessão
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from config import WORKSPACE_DIR, SessionStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _meta_path(session_dir: Path) -> Path:
    return session_dir / "meta.json"


def _load_meta(session_dir: Path) -> dict:
    p = _meta_path(session_dir)
    if not p.exists():
        raise FileNotFoundError(f"Sessão não encontrada: {session_dir}")
    with open(p) as f:
        return json.load(f)


def _save_meta(session_dir: Path, meta: dict) -> None:
    meta["updated_at"] = _now()
    with open(_meta_path(session_dir), "w") as f:
        json.dump(meta, f, indent=2)


# ── Criação / carregamento ────────────────────────────────────────────────────

def create_session() -> tuple[str, Path]:
    """Cria nova sessão. Retorna (session_id, session_dir)."""
    session_id  = str(uuid.uuid4())[:8]
    session_dir = WORKSPACE_DIR / session_id
    session_dir.mkdir(parents=True)

    for d in ["images", "orig", "label", "markers", "saliencies", "models", "results"]:
        (session_dir / d).mkdir(exist_ok=True)

    meta = {
        "session_id":  session_id,
        "status":      SessionStatus.CREATED,
        "n_images":    0,
        "n_labeled":   0,
        "has_models":  False,
        "created_at":  _now(),
        "updated_at":  _now(),
    }
    _save_meta(session_dir, meta)
    return session_id, session_dir


def get_session(session_id: str) -> tuple[dict, Path]:
    """Carrega sessão existente. Retorna (meta, session_dir)."""
    session_dir = WORKSPACE_DIR / session_id
    meta = _load_meta(session_dir)
    return meta, session_dir


def list_sessions() -> list[dict]:
    """Lista todas as sessões."""
    sessions = []
    for d in sorted(WORKSPACE_DIR.iterdir()):
        try:
            meta, _ = get_session(d.name)
            sessions.append(meta)
        except Exception:
            pass
    return sessions


def delete_session(session_id: str) -> None:
    session_dir = WORKSPACE_DIR / session_id
    shutil.rmtree(session_dir, ignore_errors=True)


# ── Images ────────────────────────────────────────────────────────────────────

def get_images(session_dir: Path) -> list[str]:
    """Retorna lista de fnames em images/ (ordenados)."""
    img_dir = session_dir / "images"
    return sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))


def get_unlabeled(session_dir: Path) -> list[str]:
    """Imagens sem label correspondente em label/."""
    images   = set(get_images(session_dir))
    labeled  = set(os.listdir(session_dir / "label"))
    return sorted(images - labeled)


def get_labeled(session_dir: Path) -> list[str]:
    images  = set(get_images(session_dir))
    labeled = set(os.listdir(session_dir / "label"))
    return sorted(images & labeled)


# ── Status update ─────────────────────────────────────────────────────────────

def update_status(session_dir: Path, **kwargs) -> dict:
    """Atualiza campos do meta.json."""
    meta = _load_meta(session_dir)
    meta.update(kwargs)

    # Re-computa status automaticamente
    images  = get_images(session_dir)
    labeled = get_labeled(session_dir)
    meta["n_images"]  = len(images)
    meta["n_labeled"] = len(labeled)

    if not meta.get("status") in (SessionStatus.TRAINING, SessionStatus.DONE):
        if len(labeled) == 0:
            meta["status"] = SessionStatus.CREATED
        elif len(labeled) < len(images):
            meta["status"] = SessionStatus.LABELING
        else:
            meta["status"] = SessionStatus.LABELED

    _save_meta(session_dir, meta)
    return meta


# ── Paths utilitários ─────────────────────────────────────────────────────────

def images_dir(session_dir: Path) -> Path:
    return session_dir / "images"


def orig_dir(session_dir: Path) -> Path:
    """FLIM espera 'orig/' — aponta para o mesmo conteúdo de images/."""
    return session_dir / "orig"


def label_dir(session_dir: Path) -> Path:
    return session_dir / "label"


def markers_dir(session_dir: Path) -> Path:
    return session_dir / "markers"


def saliencies_dir(session_dir: Path) -> Path:
    return session_dir / "saliencies"


def models_dir(session_dir: Path) -> Path:
    return session_dir / "models"


def results_dir(session_dir: Path) -> Path:
    return session_dir / "results"
