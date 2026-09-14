/**
 * predict.js — Inferência em imagem nova
 */

let selectedFile = null;

document.addEventListener("DOMContentLoaded", async () => {
  const sid = getSessionId();
  if (!sid) { window.location.href = "/"; return; }

  await loadModelOptions(sid);
  setupUpload();
  document.getElementById("btn-predict").onclick = () => runPredict(sid);
});

async function loadModelOptions(sid) {
  const select = document.getElementById("model-select");
  try {
    const data = await apiGet(`/api/sessions/${sid}/results`);
    if (!data.results?.length) {
      select.innerHTML = '<option disabled>Nenhum modelo treinado ainda</option>';
      return;
    }

    const sorted = [...data.results].sort((a, b) =>
      parseFloat(b.fb || 0) - parseFloat(a.fb || 0));

    select.innerHTML = sorted.map((r, i) => `
      <option value="${r.method}|${r.budget}" ${i === 0 ? "selected" : ""}>
        ${i === 0 ? "🏆 " : ""}${r.method}  K=${r.budget}  (Fβ=${fmtFloat(r.fb)})
      </option>
    `).join("");
  } catch {
    select.innerHTML = '<option>Erro ao carregar modelos</option>';
  }
}

function setupUpload() {
  const zone  = document.getElementById("predict-drop-zone");
  const input = document.getElementById("predict-image-input");
  const btn   = document.getElementById("btn-predict");

  setupDropZone(zone, input, files => {
    selectedFile = files[0];
    if (selectedFile) {
      zone.querySelector("p").innerHTML =
        `✅ ${selectedFile.name}<br><small>Clique para trocar</small>`;
      btn.disabled = false;

      // Preview
      const reader = new FileReader();
      reader.onload = e => {
        document.getElementById("original-img").src = e.target.result;
        showEl(document.getElementById("predict-result"));
      };
      reader.readAsDataURL(selectedFile);
    }
  });
}

async function runPredict(sid) {
  if (!selectedFile) return;

  const [method, budget] = document.getElementById("model-select").value.split("|");
  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.textContent = "Segmentando...";

  const form = new FormData();
  form.append("image", selectedFile);

  // Adiciona JSON como campo separado (FastAPI não suporta JSON + File diretamente)
  // → envia como query params
  const url = `/api/sessions/${sid}/predict?method=${method}&budget=${budget}`;

  try {
    const r = await fetch(url, { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();

    document.getElementById("overlay-img").src = `data:image/png;base64,${data.overlay_b64}`;
    document.getElementById("mask-img").src    = `data:image/png;base64,${data.mask_b64}`;

    // Download links
    document.getElementById("btn-download-overlay").href =
      `data:image/png;base64,${data.overlay_b64}`;
    document.getElementById("btn-download-mask").href =
      `data:image/png;base64,${data.mask_b64}`;

    // Métricas
    if (data.has_gt) {
      document.getElementById("metric-fb").textContent  = fmtFloat(data.fb);
      document.getElementById("metric-iou").textContent = fmtFloat(data.iou);
      showEl(document.getElementById("predict-metrics"));
    }
    document.getElementById("metric-model").textContent = `${method} K=${budget}`;
    showEl(document.getElementById("predict-metrics"));

  } catch (err) {
    alert(`Erro na inferência: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 Segmentar";
  }
}
