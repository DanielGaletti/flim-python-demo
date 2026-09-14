"""
main.py — FastAPI app — FLIM Local Training Tool
=================================================
Rotas:
  GET  /                          → index.html
  GET  /api/sessions              → lista sessões
  POST /api/sessions              → cria sessão
  GET  /api/sessions/{id}         → info da sessão

  POST /api/sessions/{id}/upload  → sobe imagens (+ opcional label/)
  GET  /api/sessions/{id}/images  → lista imagens
  GET  /api/sessions/{id}/image/{fname}  → serve imagem original
  GET  /api/sessions/{id}/label/{fname}  → serve label (se existir)
  POST /api/sessions/{id}/label   → salva label desenhada

  GET  /api/sessions/{id}/label/progress → quantas labels faltam
  POST /api/sessions/{id}/train   → inicia treino (background)
  GET  /api/sessions/{id}/train/status   → progresso (SSE ou polling)
  GET  /api/sessions/{id}/results → tabela de resultados

  POST /api/sessions/{id}/predict → inferência em imagem nova
  DELETE /api/sessions/{id}       → deleta sessão

  GET  /api/methods               → lista métodos AL disponíveis
  GET  /api/dt_available          → DT binário está disponível?

Uso:
  cd flim_al/app
  uvicorn main:app --reload --port 8000
  # abre http://localhost:8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from fastapi import (
    BackgroundTasks, FastAPI, File, HTTPException,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import AL_METHODS, DT_AVAILABLE, HOST, PORT
from models import (
    LabelSaveRequest, LabelSaveResponse, LabelingProgress,
    PredictRequest, PredictResponse, ResultsResponse,
    SessionInfo, SessionStatus, TrainProgressResponse,
    TrainRequest, TrainStatus, UploadResponse,
)
from session import (
    create_session, delete_session, get_images, get_labeled,
    get_session, get_unlabeled, label_dir, list_sessions,
    models_dir, orig_dir, results_dir, saliencies_dir,
    update_status,
)
from tasks import get_train_state, start_training

app = FastAPI(title="FLIM Local Training Tool", version="0.1.0")

# Serve arquivos estáticos (HTML/CSS/JS)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/label", response_class=HTMLResponse)
async def label_page():
    return FileResponse(STATIC_DIR / "label.html")


@app.get("/train", response_class=HTMLResponse)
async def train_page():
    return FileResponse(STATIC_DIR / "train.html")


@app.get("/results", response_class=HTMLResponse)
async def results_page():
    return FileResponse(STATIC_DIR / "results.html")


@app.get("/predict", response_class=HTMLResponse)
async def predict_page():
    return FileResponse(STATIC_DIR / "predict.html")


# ── Info ──────────────────────────────────────────────────────────────────────

@app.get("/api/methods")
async def get_methods():
    """Lista métodos AL disponíveis."""
    return AL_METHODS


@app.get("/api/dt_available")
async def check_dt():
    return {"available": DT_AVAILABLE}


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_all_sessions():
    return list_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str):
    try:
        meta, _ = get_session(session_id)
        return meta
    except FileNotFoundError:
        raise HTTPException(404, "Sessão não encontrada")


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    return {"deleted": session_id}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/sessions/upload", response_model=UploadResponse)
async def upload_dataset(
    images: list[UploadFile] = File(..., description="Imagens da pasta images/"),
    labels: list[UploadFile] | None = File(None, description="Labels (opcional)"),
):
    """
    Sobe imagens PNG (e opcionalmente labels).
    O frontend deve enviar TODOS os arquivos da pasta images/ (+ label/ se existir).

    Aceita dois formatos:
      1. multipart com campo 'images' (arquivos individuais)
      2. TODO: upload de ZIP com estrutura images/ e label/

    Retorna session_id da nova sessão criada.
    """
    session_id, session_dir = create_session()

    # Salva imagens
    n_images = 0
    img_dir  = orig_dir(session_dir)  # salva em orig/ (FLIM compat)
    img_link = session_dir / "images"  # e em images/ (nome do usuário)

    for f in images:
        if not f.filename.endswith(".png"):
            continue
        fname = Path(f.filename).name
        data  = await f.read()
        dest  = img_dir / fname
        dest.write_bytes(data)
        # symlink images/ → orig/ (mesmo conteúdo)
        link = img_link / fname
        if not link.exists():
            os.symlink(dest, link)
        n_images += 1

    # Salva labels (se fornecidas)
    n_labels = 0
    if labels:
        lbl_dir = label_dir(session_dir)
        for f in labels:
            if not f.filename.endswith(".png"):
                continue
            fname = Path(f.filename).name
            data  = await f.read()
            (lbl_dir / fname).write_bytes(data)
            n_labels += 1

    meta = update_status(session_dir)
    needs_labeling = n_labels < n_images

    return UploadResponse(
        session_id=session_id,
        n_images=n_images,
        n_labels=n_labels,
        needs_labeling=needs_labeling,
        message=(
            f"{n_images} imagens carregadas. "
            + (f"Nenhuma label — vá para Labeling." if needs_labeling
               else f"{n_labels} labels carregadas. Pronto para treinar.")
        ),
    )


# ── Images ────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/images")
async def list_images(session_id: str):
    _, session_dir = get_session(session_id)
    labeled   = set(get_labeled(session_dir))
    unlabeled = set(get_unlabeled(session_dir))
    return {
        "labeled":   sorted(labeled),
        "unlabeled": sorted(unlabeled),
        "total":     len(labeled) + len(unlabeled),
    }


@app.get("/api/sessions/{session_id}/image/{fname}")
async def serve_image(session_id: str, fname: str):
    _, session_dir = get_session(session_id)
    p = orig_dir(session_dir) / fname
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/sessions/{session_id}/label/{fname}")
async def serve_label(session_id: str, fname: str):
    _, session_dir = get_session(session_id)
    p = label_dir(session_dir) / fname
    if not p.exists():
        raise HTTPException(404, "Label não encontrada")
    return FileResponse(str(p), media_type="image/png")


# ── Labeling ──────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/label/progress", response_model=LabelingProgress)
async def labeling_progress(session_id: str):
    _, session_dir = get_session(session_id)
    labeled   = get_labeled(session_dir)
    unlabeled = get_unlabeled(session_dir)
    return LabelingProgress(
        n_labeled=len(labeled),
        n_total=len(labeled) + len(unlabeled),
        remaining=unlabeled,
        labeled=labeled,
    )


@app.post("/api/sessions/{session_id}/label", response_model=LabelSaveResponse)
async def save_label(session_id: str, req: LabelSaveRequest):
    """
    Recebe máscara binária como base64 PNG (do canvas do frontend).
    Salva como label/<image_name>.png em escala de cinza (0 ou 255).
    """
    _, session_dir = get_session(session_id)

    # Decodifica base64 → PIL → converte para binário (0/255)
    try:
        b64_data = req.mask_b64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        mask_img = Image.open(io.BytesIO(raw)).convert("L")
        # Binariza: qualquer pixel > 0 vira 255
        mask_arr = (lambda a: (a > 0).astype("uint8") * 255)(
            __import__("numpy").array(mask_img)
        )
        out_img = Image.fromarray(mask_arr, mode="L")
    except Exception as e:
        raise HTTPException(400, f"Máscara inválida: {e}")

    out_path = label_dir(session_dir) / req.image_name
    out_img.save(str(out_path))

    meta = update_status(session_dir)
    return LabelSaveResponse(
        image_name=req.image_name,
        saved=True,
        n_labeled=meta["n_labeled"],
        n_total=meta["n_images"],
    )


# ── Training ──────────────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/train")
async def start_train(session_id: str, req: TrainRequest):
    """Valida request e dispara treino em thread background."""
    meta, session_dir = get_session(session_id)

    # Verifica se tem labels suficientes
    labeled = get_labeled(session_dir)
    if len(labeled) < req.n_init:
        raise HTTPException(400,
            f"Precisa de pelo menos {req.n_init} imagens com label "
            f"para começar (tem {len(labeled)})")

    # Verifica se não está em treino
    state = get_train_state(session_id)
    if state and state.get("overall_status") == TrainStatus.RUNNING:
        raise HTTPException(409, "Treino já em andamento")

    # Dispara
    start_training(
        session_id=session_id,
        methods=req.methods,
        budgets=req.budgets,
        n_init=req.n_init,
        n_committee=req.n_committee,
        use_dt=req.use_dt and DT_AVAILABLE,
    )

    update_status(session_dir, status=SessionStatus.TRAINING)
    return {"message": "Treino iniciado", "session_id": session_id}


@app.get("/api/sessions/{session_id}/train/status", response_model=TrainProgressResponse)
async def train_status(session_id: str):
    """Polling de progresso do treino."""
    state = get_train_state(session_id)
    if not state:
        raise HTTPException(404, "Nenhum treino para esta sessão")

    elapsed = time.time() - state.get("started_at", time.time())

    from models import MethodProgress
    results_raw = state.get("results", [])
    results = [MethodProgress(**r) if isinstance(r, dict) else r for r in results_raw]

    return TrainProgressResponse(
        session_id=session_id,
        overall_status=state["overall_status"],
        current_method=state.get("current_method"),
        current_budget=state.get("current_budget"),
        results=results,
        elapsed_total_s=round(elapsed, 1),
        eta_s=None,  # TODO: estimar ETA
    )


@app.get("/api/sessions/{session_id}/train/stream")
async def train_status_stream(session_id: str):
    """
    Server-Sent Events — frontend faz EventSource('/api/.../train/stream')
    e recebe atualizações em tempo real sem precisar fazer polling.
    """
    async def event_generator():
        while True:
            state = get_train_state(session_id)
            if state:
                data = json.dumps({
                    "overall_status": state.get("overall_status"),
                    "current_method": state.get("current_method"),
                    "current_budget": state.get("current_budget"),
                    "n_results":      len(state.get("results", [])),
                })
                yield f"data: {data}\n\n"

                if state.get("overall_status") in (TrainStatus.DONE, TrainStatus.ERROR):
                    break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Results ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/results")
async def get_results(session_id: str):
    """
    Retorna todos os resultados de treino da sessão.
    Lê do CSV salvo em results/ ou do estado em memória.
    """
    import csv
    _, session_dir = get_session(session_id)
    res_dir = results_dir(session_dir)

    rows = []
    for csv_file in res_dir.glob("*.csv"):
        with open(csv_file, newline="") as f:
            rows.extend(csv.DictReader(f))

    # Fallback: estado em memória
    if not rows:
        state = get_train_state(session_id)
        if state:
            rows = state.get("results", [])

    if not rows:
        return {"results": [], "best": None}

    # Melhor por Fβ
    best = max(rows, key=lambda r: float(r.get("fb", 0)), default=None)
    return {"results": rows, "best": best}


@app.get("/api/sessions/{session_id}/results/csv")
async def download_results_csv(session_id: str):
    """Download do CSV consolidado de resultados."""
    _, session_dir = get_session(session_id)
    res_dir = results_dir(session_dir)
    csvs    = list(res_dir.glob("*.csv"))
    if not csvs:
        raise HTTPException(404, "Nenhum resultado ainda")
    return FileResponse(str(csvs[0]), filename="flim_results.csv",
                        media_type="text/csv")


# ── Inference ──────────────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/predict", response_model=PredictResponse)
async def predict(
    session_id: str,
    req: PredictRequest,
    image: UploadFile = File(...),
):
    """
    Roda inferência em imagem nova (não precisa estar no dataset).
    Retorna overlay PNG (base64) + métricas se GT disponível.
    """
    _, session_dir = get_session(session_id)

    enc_path = models_dir(session_dir) / req.method / f"K{req.budget}" / "encoder.pth"
    if not enc_path.exists():
        # Tenta o encoder base
        enc_path = models_dir(session_dir) / "base_encoder.pth"
    if not enc_path.exists():
        raise HTTPException(404, f"Encoder não encontrado para {req.method} K={req.budget}")

    # Salva imagem temporária
    img_data = await image.read()
    tmp_path = session_dir / "tmp_predict.png"
    tmp_path.write_bytes(img_data)

    try:
        from inference import run_inference
        result = run_inference(
            image_path=str(tmp_path),
            encoder_path=str(enc_path),
            use_dt=DT_AVAILABLE,
        )
        return PredictResponse(
            image_name=image.filename,
            overlay_b64=result["overlay_b64"],
            mask_b64=result["mask_b64"],
            fb=result.get("fb"),
            iou=result.get("iou"),
            has_gt=result.get("has_gt", False),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    def _open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
