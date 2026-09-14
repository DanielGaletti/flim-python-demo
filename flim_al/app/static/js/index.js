/**
 * index.js — Upload de dataset + listagem de sessões
 */

let imageFiles = [];
let labelFiles = [];

document.addEventListener("DOMContentLoaded", async () => {
  setupDropZone(
    document.getElementById("drop-zone"),
    document.getElementById("images-input"),
    files => {
      imageFiles = files.filter(f => f.name.endsWith(".png"));
      updateUploadBtn();
      document.querySelector("#drop-zone p").innerHTML =
        `✅ ${imageFiles.length} imagens selecionadas<br><small>Clique para trocar</small>`;
    }
  );

  setupDropZone(
    document.getElementById("drop-zone-label"),
    document.getElementById("labels-input"),
    files => {
      labelFiles = files.filter(f => f.name.endsWith(".png"));
      document.querySelector("#drop-zone-label p").innerHTML =
        `✅ ${labelFiles.length} labels selecionadas<br><small>Clique para trocar</small>`;
    }
  );

  document.getElementById("btn-upload").onclick = uploadDataset;

  await loadSessions();
});

function updateUploadBtn() {
  document.getElementById("btn-upload").disabled = imageFiles.length === 0;
}

async function uploadDataset() {
  const btn    = document.getElementById("btn-upload");
  const status = document.getElementById("upload-status");
  btn.disabled = true;
  btn.textContent = "Carregando...";

  const form = new FormData();
  imageFiles.forEach(f => form.append("images", f));
  labelFiles.forEach(f => form.append("labels", f));

  try {
    const res = await apiUpload("/api/sessions/upload", form);
    setSessionId(res.session_id);
    showStatus(status,
      `✅ ${res.message} (sessão: ${res.session_id})`, "success");

    setTimeout(() => {
      window.location.href = res.needs_labeling ? "/label" : "/train";
    }, 1500);
  } catch (err) {
    showStatus(status, `❌ Erro: ${err.message}`, "error");
    btn.disabled = false;
    btn.textContent = "Carregar →";
  }
}

async function loadSessions() {
  const sessions  = await apiGet("/api/sessions");
  const container = document.getElementById("sessions-list");

  if (!sessions.length) {
    container.innerHTML = '<p class="loading">Nenhuma sessão ainda.</p>';
    return;
  }

  container.innerHTML = sessions.map(s => `
    <div class="session-card" onclick="selectSession('${s.session_id}')">
      <div>
        <strong>${s.session_id}</strong> &nbsp;
        ${statusTag(s.status)}
      </div>
      <div style="color:var(--text-muted);font-size:0.85rem">
        ${s.n_images} imagens · ${s.n_labeled} labels
        ${s.has_models ? "· ✅ modelos treinados" : ""}
      </div>
      <div style="margin-left:auto">
        <button onclick="event.stopPropagation(); deleteSession('${s.session_id}')"
                class="btn-danger" style="padding:0.25rem 0.6rem;font-size:0.8rem">
          🗑️
        </button>
      </div>
    </div>
  `).join("");
}

function selectSession(id) {
  setSessionId(id);
  window.location.href = "/train";
}

async function deleteSession(id) {
  if (!confirm(`Deletar sessão ${id}?`)) return;
  await apiDelete(`/api/sessions/${id}`);
  if (getSessionId() === id) clearSession();
  await loadSessions();
}
