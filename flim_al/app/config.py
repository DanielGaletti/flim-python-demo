"""
config.py — Configurações globais do app FLIM Local
"""
from pathlib import Path
import os

# ── Workspace ─────────────────────────────────────────────────────────────────
# Diretório onde as sessões ficam armazenadas
WORKSPACE_DIR = Path(os.environ.get("FLIM_WORKSPACE", Path.home() / "flim_workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ── FLIM paths ────────────────────────────────────────────────────────────────
# Raiz do repositório (dois níveis acima de flim_al/app/)
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
FLIMPY_PATH = REPO_ROOT / "flim_ad" / "libs" / "flim-python"

# Arch file padrão (4 camadas, schisto)
DEFAULT_ARCH = REPO_ROOT / "flim_ad" / "data" / "schisto" / "user_A" / "split1" / "arch2D.json"

# DT binary (opcional — avaliação com Dynamic Trees)
DT_BIN = REPO_ROOT / "flim_ad" / "libs" / "ift" / "bin" / "iftSMansoniDelineation"
DT_AVAILABLE = DT_BIN.exists()

# ── AL params padrão ──────────────────────────────────────────────────────────
DEFAULT_N_INIT      = 3    # imagens anotadas inicialmente
DEFAULT_N_FG        = 100  # seeds foreground por imagem
DEFAULT_N_BG        = 300  # seeds background por imagem
DEFAULT_N_COMMITTEE = 3    # encoders no comitê (region_bald)
DEFAULT_PROXY_LAYER = 3

# ── Métodos disponíveis ───────────────────────────────────────────────────────
AL_METHODS = [
    {"id": "no_al",            "label": "Sem AL (aleatório)",      "needs_committee": False},
    {"id": "entropy",          "label": "Entropia",                 "needs_committee": False},
    {"id": "least_confidence", "label": "Least Confidence",         "needs_committee": False},
    {"id": "margin",           "label": "Margin Sampling",          "needs_committee": False},
    {"id": "region_entropy",   "label": "Region Entropy (RIPU)",    "needs_committee": False},
    {"id": "region_bald",      "label": "Region BALD (Comitê)",     "needs_committee": True},
]

# ── Servidor ──────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
