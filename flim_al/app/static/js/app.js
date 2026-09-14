/**
 * app.js — Utilitários globais compartilhados entre todas as páginas
 */

// ── Session management ─────────────────────────────────────────────────────

const SESSION_KEY = "flim_session_id";

function getSessionId() {
  return localStorage.getItem(SESSION_KEY);
}

function setSessionId(id) {
  localStorage.setItem(SESSION_KEY, id);
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

// ── API helpers ────────────────────────────────────────────────────────────

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}: ${await r.text()}`);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}: ${await r.text()}`);
  return r.json();
}

async function apiDelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) throw new Error(`DELETE ${path} → ${r.status}`);
  return r.json();
}

async function apiUpload(path, formData) {
  const r = await fetch(path, { method: "POST", body: formData });
  if (!r.ok) throw new Error(`UPLOAD ${path} → ${r.status}: ${await r.text()}`);
  return r.json();
}

// ── UI helpers ─────────────────────────────────────────────────────────────

function showStatus(el, msg, type = "info") {
  el.textContent = msg;
  el.className   = `status ${type}`;
  el.classList.remove("hidden");
}

function hideEl(el) { el?.classList.add("hidden"); }
function showEl(el) { el?.classList.remove("hidden"); }

function setProgress(bar, pct) {
  bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

// ── Status tag ─────────────────────────────────────────────────────────────

function statusTag(status) {
  return `<span class="tag ${status}">${status}</span>`;
}

// ── Formatadores ───────────────────────────────────────────────────────────

function fmtFloat(v) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(3);
}

function fmtTime(seconds) {
  if (seconds < 60)   return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds/60)}m ${Math.round(seconds%60)}s`;
  return `${Math.floor(seconds/3600)}h ${Math.floor((seconds%3600)/60)}m`;
}

// ── Drag-and-drop helpers ──────────────────────────────────────────────────

function setupDropZone(zone, input, onFiles) {
  zone.addEventListener("click",      () => input.click());
  zone.addEventListener("dragover",   e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave",  () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop",       e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    onFiles(Array.from(e.dataTransfer.files));
  });
  input.addEventListener("change",    e => onFiles(Array.from(e.target.files)));
}
