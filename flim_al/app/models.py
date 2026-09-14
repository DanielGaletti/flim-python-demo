"""
models.py — Pydantic models para requests/responses da API
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel


# ── Session ───────────────────────────────────────────────────────────────────

class SessionStatus(str, Enum):
    CREATED   = "created"    # pasta upada, sem labels
    LABELING  = "labeling"   # algumas labels feitas, não todas
    LABELED   = "labeled"    # todas as imagens têm label
    TRAINING  = "training"   # treino em andamento
    DONE      = "done"       # treino concluído, pronto para inferência


class SessionInfo(BaseModel):
    session_id:    str
    status:        SessionStatus
    n_images:      int
    n_labeled:     int
    has_models:    bool
    created_at:    str
    updated_at:    str


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id:    str
    n_images:      int
    n_labels:      int          # 0 se nenhuma label subida
    needs_labeling: bool
    message:       str


# ── Labeling ──────────────────────────────────────────────────────────────────

class LabelSaveRequest(BaseModel):
    """Recebe máscara binária como base64 PNG."""
    image_name: str
    mask_b64:   str             # data:image/png;base64,...


class LabelSaveResponse(BaseModel):
    image_name: str
    saved:      bool
    n_labeled:  int
    n_total:    int


class LabelingProgress(BaseModel):
    n_labeled:  int
    n_total:    int
    remaining:  list[str]       # fnames ainda sem label
    labeled:    list[str]


# ── Training ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    session_id:  str
    methods:     list[str]      # ["entropy", "region_entropy", ...]
    budgets:     list[int]      # [3, 5, 10]
    n_init:      int = 3
    n_committee: int = 3
    use_dt:      bool = False


class TrainStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    ERROR    = "error"


class MethodProgress(BaseModel):
    method:    str
    budget:    Optional[int]
    status:    TrainStatus
    fb:        Optional[float]
    iou:       Optional[float]
    elapsed_s: Optional[float]
    message:   str = ""


class TrainProgressResponse(BaseModel):
    session_id:      str
    overall_status:  TrainStatus
    current_method:  Optional[str]
    current_budget:  Optional[int]
    results:         list[MethodProgress]
    elapsed_total_s: float
    eta_s:           Optional[float]


class TrainResult(BaseModel):
    method:    str
    budget:    int
    fb:        float
    iou:       float
    dice:      float
    elapsed_s: float


class TrainSummary(BaseModel):
    session_id:   str
    results:      list[TrainResult]
    best_method:  str
    best_budget:  int
    best_fb:      float
    total_time_s: float


# ── Inference ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id:    str
    method:        str          # método treinado a usar
    budget:        int
    return_overlay: bool = True  # retorna imagem com overlay colorido


class PredictResponse(BaseModel):
    image_name:    str
    overlay_b64:   str          # PNG com segmentação sobreposta (vermelho)
    mask_b64:      str          # máscara binária
    fb:            Optional[float]   # se GT disponível
    iou:           Optional[float]
    has_gt:        bool


# ── Results ───────────────────────────────────────────────────────────────────

class ResultsResponse(BaseModel):
    session_id: str
    results:    list[TrainResult]
    best:       TrainResult
