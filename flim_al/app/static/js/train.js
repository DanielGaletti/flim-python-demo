/**
 * train.js — Configuração de treino + polling de progresso
 */

let sessionId  = null;
let pollTimer  = null;
let startedAt  = null;

document.addEventListener("DOMContentLoaded", async () => {
  sessionId = getSessionId();
  if (!sessionId) { window.location.href = "/"; return; }

  await loadMethods();
  await checkLabels();
  await checkDT();
  setupBudgets();
  setupActions();
});

// ── Carrega métodos do servidor ───────────────────────────────────────────────

async function loadMethods() {
  const methods = await apiGet("/api/methods");
  const container = document.getElementById("methods-checkboxes");
  container.innerHTML = methods.map(m => `
    <label class="checkbox-item ${m.needs_committee ? "needs-committee" : ""}">
      <input type="checkbox" name="method" value="${m.id}" checked>
      <span>
        <strong>${m.label}</strong>
        ${m.needs_committee ? '<span class="tag training" style="font-size:0.7rem">comitê</span>' : ""}
      </span>
    </label>
  `).join("");

  document.getElementById("btn-select-all").onclick = () =>
    container.querySelectorAll("input").forEach(i => i.checked = true);
  document.getElementById("btn-select-none").onclick = () =>
    container.querySelectorAll("input").forEach(i => i.checked = false);
}

// ── Verifica labels ───────────────────────────────────────────────────────────

async function checkLabels() {
  const progress = await apiGet(`/api/sessions/${sessionId}/label/progress`);
  const alert    = document.getElementById("alert-no-labels");
  if (progress.n_labeled < progress.n_total) {
    showEl(alert);
  } else {
    hideEl(alert);
  }
}

// ── DT disponível? ────────────────────────────────────────────────────────────

async function checkDT() {
  const { available } = await apiGet("/api/dt_available");
  const dtLabel  = document.getElementById("dt-label");
  const dtStatus = document.getElementById("dt-status");
  const dtCheck  = document.getElementById("use-dt");

  if (!available) {
    dtCheck.disabled = true;
    dtStatus.textContent = " (não disponível)";
    dtStatus.style.color = "var(--text-muted)";
  } else {
    dtStatus.textContent = " ✓ disponível";
    dtStatus.style.color = "var(--success)";
  }
}

// ── Budgets ───────────────────────────────────────────────────────────────────

function setupBudgets() {
  document.getElementById("btn-add-budget").onclick = () => {
    const val = parseInt(document.getElementById("custom-budget").value);
    if (!val || val < 1) return;
    const container = document.querySelector(".budget-checkboxes");
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" name="budget" value="${val}" checked> K=${val}`;
    container.appendChild(label);
    document.getElementById("custom-budget").value = "";
  };
}

// ── Ações ─────────────────────────────────────────────────────────────────────

function setupActions() {
  document.getElementById("btn-start-train").onclick = startTrain;
}

function getSelectedMethods() {
  return Array.from(
    document.querySelectorAll('input[name="method"]:checked')
  ).map(i => i.value);
}

function getSelectedBudgets() {
  return Array.from(
    document.querySelectorAll('input[name="budget"]:checked')
  ).map(i => parseInt(i.value)).filter(Boolean);
}

async function startTrain() {
  const methods = getSelectedMethods();
  const budgets = getSelectedBudgets();

  if (!methods.length) { alert("Selecione pelo menos um método."); return; }
  if (!budgets.length) { alert("Selecione pelo menos um budget."); return; }

  const btn = document.getElementById("btn-start-train");
  btn.disabled = true;
  btn.textContent = "Iniciando...";

  try {
    await apiPost(`/api/sessions/${sessionId}/train`, {
      session_id:  sessionId,
      methods,
      budgets,
      n_init:      parseInt(document.getElementById("n-init").value) || 3,
      n_committee: parseInt(document.getElementById("n-committee").value) || 3,
      use_dt:      document.getElementById("use-dt").checked,
    });

    hideEl(document.getElementById("train-config"));
    showEl(document.getElementById("train-progress"));

    startedAt = Date.now();
    initProgressTable(methods, budgets);
    startPolling();

  } catch (err) {
    btn.disabled = false;
    btn.textContent = "▶ Iniciar Treino";
    showEl(document.getElementById("train-error"));
    document.getElementById("train-error").textContent = `Erro: ${err.message}`;
  }
}

// ── Tabela de progresso ───────────────────────────────────────────────────────

function initProgressTable(methods, budgets) {
  const tbody = document.getElementById("progress-rows");
  tbody.innerHTML = methods.flatMap(m =>
    budgets.map(b => `
      <tr id="row-${m}-${b}">
        <td>${m}</td>
        <td>K=${b}</td>
        <td><span class="tag training">aguardando</span></td>
        <td>—</td>
        <td>—</td>
        <td>—</td>
      </tr>
    `)
  ).join("");
}

function updateProgressTable(results) {
  results.forEach(r => {
    const row = document.getElementById(`row-${r.method}-${r.budget}`);
    if (!row) return;
    const cells = row.querySelectorAll("td");
    cells[2].innerHTML = statusTag(r.status);
    cells[3].textContent = fmtFloat(r.fb);
    cells[4].textContent = fmtFloat(r.iou);
    cells[5].textContent = r.elapsed_s ? fmtTime(r.elapsed_s) : "—";
  });
}

// ── Polling ───────────────────────────────────────────────────────────────────

function startPolling() {
  pollTimer = setInterval(pollStatus, 2000);
  pollStatus();  // imediato
}

async function pollStatus() {
  try {
    const status = await apiGet(`/api/sessions/${sessionId}/train/status`);

    // Atualiza header
    const elapsed = (Date.now() - startedAt) / 1000;
    document.getElementById("elapsed-label").textContent = fmtTime(elapsed);
    document.getElementById("progress-label").textContent =
      status.current_method
        ? `${status.current_method} K=${status.current_budget}...`
        : status.overall_status;

    // Progresso global (% de resultados concluídos)
    const totalExpected =
      document.querySelectorAll("#progress-rows tr").length;
    const done = status.results.filter(r => r.status === "done").length;
    setProgress(document.getElementById("global-progress-bar"),
      totalExpected > 0 ? (done / totalExpected) * 100 : 0);

    updateProgressTable(status.results);

    // Fim
    if (status.overall_status === "done" || status.overall_status === "error") {
      clearInterval(pollTimer);
      if (status.overall_status === "done") {
        showEl(document.getElementById("train-done-actions"));
      } else {
        showEl(document.getElementById("train-error"));
        document.getElementById("train-error").textContent =
          "Erro durante o treino. Veja o console para detalhes.";
      }
    }
  } catch (err) {
    console.error("Erro no polling:", err);
  }
}
