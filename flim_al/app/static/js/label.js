/**
 * label.js — Canvas de labeling com pincel, polígono e eraser
 *
 * Três camadas de canvas sobrepostas (z-index crescente):
 *   1. canvas-image  → exibe a imagem original (somente leitura)
 *   2. canvas-label  → exibe label existente carregada do servidor (semitransparente)
 *   3. canvas-draw   → onde o usuário desenha (acumula pixels pintados)
 *
 * Ao salvar, o canvas-draw é convertido para PNG binário (base64) e enviado via POST.
 *
 * Ferramentas:
 *   brush   → pincel circular (hold + drag)
 *   eraser  → apagador circular
 *   polygon → clica pontos → Enter/duplo-clique fecha e preenche
 *   fill    → flood fill no ponto clicado
 *
 * Keyboard shortcuts:
 *   B → brush | E → eraser | P → polygon | F → fill
 *   Ctrl+Z → undo | Ctrl+S → save
 */

// ── Estado global ────────────────────────────────────────────────────────────

const state = {
  sessionId:    null,
  images:       [],      // todos os fnames do dataset
  labeled:      new Set(),
  currentIdx:   0,

  // Ferramentas
  tool:         "brush",
  brushSize:    15,
  isDrawing:    false,
  lastX:        0,
  lastY:        0,

  // Polígono
  polyPoints:   [],

  // Undo stack (snapshots do canvas-draw como ImageData)
  undoStack:    [],
  MAX_UNDO:     30,
};

// ── Canvas refs ───────────────────────────────────────────────────────────────

let cImage, cLabel, cDraw;
let ctxImage, ctxLabel, ctxDraw;

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  state.sessionId = getSessionId();
  if (!state.sessionId) {
    window.location.href = "/";
    return;
  }

  // Canvas refs
  cImage    = document.getElementById("canvas-image");
  cLabel    = document.getElementById("canvas-label");
  cDraw     = document.getElementById("canvas-draw");
  ctxImage  = cImage.getContext("2d");
  ctxLabel  = cLabel.getContext("2d");
  ctxDraw   = cDraw.getContext("2d");

  await loadImageList();
  setupToolbar();
  setupCanvasEvents();
  setupKeyboard();

  // Carrega primeira imagem não-labeled (ou primeira da lista)
  const firstUnlabeled = state.images.findIndex(f => !state.labeled.has(f));
  goToImage(firstUnlabeled >= 0 ? firstUnlabeled : 0);
});

// ── Lista de imagens ──────────────────────────────────────────────────────────

async function loadImageList() {
  const data = await apiGet(`/api/sessions/${state.sessionId}/images`);
  state.labeled = new Set(data.labeled);
  state.images  = [...data.labeled, ...data.unlabeled].sort();

  updateSidebar();
  updateProgress();
}

function updateSidebar() {
  const ul = document.getElementById("image-list");
  ul.innerHTML = state.images.map((fname, i) => {
    const cls = state.labeled.has(fname) ? "labeled" : "unlabeled";
    return `<li class="${cls}${i === state.currentIdx ? " active" : ""}"
               onclick="goToImage(${i})" title="${fname}">
              ${fname}
            </li>`;
  }).join("");
}

function updateProgress() {
  const n = state.images.length;
  const l = state.labeled.size;
  document.getElementById("label-count").textContent = `${l} / ${n} imagens com label`;
  setProgress(document.getElementById("label-progress-bar"), (l / n) * 100);
  document.getElementById("img-counter").textContent =
    `${state.currentIdx + 1} / ${n}`;
}

// ── Navegação ─────────────────────────────────────────────────────────────────

async function goToImage(idx) {
  if (idx < 0 || idx >= state.images.length) return;
  state.currentIdx = idx;
  state.undoStack  = [];
  state.polyPoints = [];

  const fname = state.images[idx];
  updateSidebar();
  updateProgress();

  await Promise.all([
    loadImageCanvas(fname),
    loadLabelCanvas(fname),
  ]);
  clearDraw();
}

async function loadImageCanvas(fname) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const W = img.naturalWidth, H = img.naturalHeight;
      [cImage, cLabel, cDraw].forEach(c => { c.width = W; c.height = H; });
      ctxImage.drawImage(img, 0, 0);
      resolve();
    };
    img.src = `/api/sessions/${state.sessionId}/image/${fname}`;
  });
}

async function loadLabelCanvas(fname) {
  // Carrega label existente (se houver) no canvas-label
  ctxLabel.clearRect(0, 0, cLabel.width, cLabel.height);
  if (!state.labeled.has(fname)) return;

  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      ctxLabel.globalCompositeOperation = "source-over";
      ctxLabel.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = resolve;
    img.src = `/api/sessions/${state.sessionId}/label/${fname}`;
  });
}

function clearDraw() {
  ctxDraw.clearRect(0, 0, cDraw.width, cDraw.height);
}

// ── Toolbar ────────────────────────────────────────────────────────────────────

function setupToolbar() {
  const tools = ["brush", "eraser", "polygon", "fill"];
  tools.forEach(t => {
    document.getElementById(`tool-${t}`)?.addEventListener("click", () => setTool(t));
  });

  const brushSlider = document.getElementById("brush-size");
  brushSlider.addEventListener("input", () => {
    state.brushSize = parseInt(brushSlider.value);
    document.getElementById("brush-size-label").textContent = `${state.brushSize}px`;
  });

  document.getElementById("btn-undo")?.addEventListener("click", undo);
  document.getElementById("btn-clear")?.addEventListener("click", () => {
    if (confirm("Limpar tudo?")) clearDraw();
  });
  document.getElementById("btn-save-label")?.addEventListener("click", saveLabel);

  document.getElementById("btn-prev")?.addEventListener("click",
    () => goToImage(state.currentIdx - 1));
  document.getElementById("btn-next")?.addEventListener("click",
    () => goToImage(state.currentIdx + 1));

  document.getElementById("chk-show-label")?.addEventListener("change", e => {
    cLabel.style.display = e.target.checked ? "" : "none";
  });
}

function setTool(tool) {
  state.tool       = tool;
  state.polyPoints = [];
  document.querySelectorAll(".tool-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tool-${tool}`)?.classList.add("active");
}

// ── Canvas events ─────────────────────────────────────────────────────────────

function setupCanvasEvents() {
  cDraw.addEventListener("mousedown",  onMouseDown);
  cDraw.addEventListener("mousemove",  onMouseMove);
  cDraw.addEventListener("mouseup",    onMouseUp);
  cDraw.addEventListener("mouseleave", onMouseUp);
  cDraw.addEventListener("dblclick",   onDblClick);
}

function getPos(e) {
  const rect  = cDraw.getBoundingClientRect();
  const scaleX = cDraw.width  / rect.width;
  const scaleY = cDraw.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  };
}

function onMouseDown(e) {
  const { x, y } = getPos(e);
  if (state.tool === "polygon") {
    state.polyPoints.push({ x, y });
    drawPolyPreview();
    return;
  }
  pushUndo();
  state.isDrawing = true;
  state.lastX = x;
  state.lastY = y;
  drawAt(x, y);
}

function onMouseMove(e) {
  if (!state.isDrawing) return;
  const { x, y } = getPos(e);
  drawLine(state.lastX, state.lastY, x, y);
  state.lastX = x;
  state.lastY = y;
}

function onMouseUp() {
  state.isDrawing = false;
}

function onDblClick(e) {
  if (state.tool === "polygon" && state.polyPoints.length >= 3) {
    closePoly();
  }
}

// ── Desenho ────────────────────────────────────────────────────────────────────

function drawAt(x, y) {
  const r = state.brushSize / 2;
  ctxDraw.globalCompositeOperation =
    state.tool === "eraser" ? "destination-out" : "source-over";
  ctxDraw.fillStyle = "rgba(255, 50, 50, 0.85)";
  ctxDraw.beginPath();
  ctxDraw.arc(x, y, r, 0, Math.PI * 2);
  ctxDraw.fill();
}

function drawLine(x0, y0, x1, y1) {
  // Interpola pontos para pinceladas suaves
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const steps = Math.max(1, Math.ceil(dist / (state.brushSize / 4)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    drawAt(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
  }
}

function drawPolyPreview() {
  // Redesenha o preview do polígono em andamento
  // (usa o canvas-label temporariamente ou um overlay separado)
  // TODO: implementar preview suave do polígono em andamento
}

function closePoly() {
  if (state.polyPoints.length < 3) return;
  pushUndo();
  ctxDraw.globalCompositeOperation = "source-over";
  ctxDraw.fillStyle = "rgba(255, 50, 50, 0.85)";
  ctxDraw.beginPath();
  ctxDraw.moveTo(state.polyPoints[0].x, state.polyPoints[0].y);
  state.polyPoints.forEach(p => ctxDraw.lineTo(p.x, p.y));
  ctxDraw.closePath();
  ctxDraw.fill();
  state.polyPoints = [];
}

// ── Flood fill ────────────────────────────────────────────────────────────────

function floodFill(x0, y0) {
  // TODO: implementar flood fill no ctxDraw usando BFS pixel a pixel
  // Referência: https://en.wikipedia.org/wiki/Flood_fill
  // Cuidado: para imagens grandes, usar Web Worker para não bloquear UI
  console.warn("TODO: flood fill não implementado");
}

// ── Undo ──────────────────────────────────────────────────────────────────────

function pushUndo() {
  if (cDraw.width === 0) return;
  const snapshot = ctxDraw.getImageData(0, 0, cDraw.width, cDraw.height);
  state.undoStack.push(snapshot);
  if (state.undoStack.length > state.MAX_UNDO) state.undoStack.shift();
}

function undo() {
  if (!state.undoStack.length) return;
  const snap = state.undoStack.pop();
  ctxDraw.putImageData(snap, 0, 0);
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

function setupKeyboard() {
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if (e.ctrlKey && e.key === "z") { e.preventDefault(); undo(); return; }
    if (e.ctrlKey && e.key === "s") { e.preventDefault(); saveLabel(); return; }
    if (e.key === "Enter" && state.tool === "polygon") { closePoly(); return; }
    if (e.key === "b") setTool("brush");
    if (e.key === "e") setTool("eraser");
    if (e.key === "p") setTool("polygon");
    if (e.key === "f") setTool("fill");
    if (e.key === "ArrowRight") goToImage(state.currentIdx + 1);
    if (e.key === "ArrowLeft")  goToImage(state.currentIdx - 1);
  });
}

// ── Save ──────────────────────────────────────────────────────────────────────

async function saveLabel() {
  const fname = state.images[state.currentIdx];
  if (!fname) return;

  // Combina canvas-label (label existente) + canvas-draw (novo desenho)
  // em um único canvas binário para salvar
  const mergeCanvas = document.createElement("canvas");
  mergeCanvas.width  = cDraw.width;
  mergeCanvas.height = cDraw.height;
  const ctx = mergeCanvas.getContext("2d");

  // Pinta existente
  ctx.drawImage(cLabel, 0, 0);
  // Adiciona novo desenho
  ctx.drawImage(cDraw, 0, 0);

  // Converte para máscara binária (canal alpha > 0 → branco)
  const imgData = ctx.getImageData(0, 0, mergeCanvas.width, mergeCanvas.height);
  const bin     = ctx.createImageData(mergeCanvas.width, mergeCanvas.height);
  for (let i = 0; i < imgData.data.length; i += 4) {
    const fg = imgData.data[i + 3] > 10;  // alpha > 10 = foreground
    bin.data[i]     = fg ? 255 : 0;
    bin.data[i + 1] = fg ? 255 : 0;
    bin.data[i + 2] = fg ? 255 : 0;
    bin.data[i + 3] = 255;
  }
  ctx.putImageData(bin, 0, 0);
  const maskB64 = mergeCanvas.toDataURL("image/png");

  const statusEl = document.getElementById("label-status");
  try {
    const res = await apiPost(`/api/sessions/${state.sessionId}/label`, {
      image_name: fname,
      mask_b64:   maskB64,
    });
    state.labeled.add(fname);
    updateSidebar();
    updateProgress();
    showStatus(statusEl, `✅ Salvo! ${res.n_labeled}/${res.n_total} com label`, "success");
    setTimeout(() => hideEl(statusEl), 2500);

    // Auto-avança para próxima imagem não-labeled
    setTimeout(() => {
      const next = state.images.findIndex((f, i) => i > state.currentIdx && !state.labeled.has(f));
      if (next >= 0) goToImage(next);
    }, 800);

  } catch (err) {
    showStatus(statusEl, `❌ Erro ao salvar: ${err.message}`, "error");
  }
}
