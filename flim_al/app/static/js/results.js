/**
 * results.js — Tabela comparativa de resultados + gráfico simples
 */

document.addEventListener("DOMContentLoaded", async () => {
  const sid = getSessionId();
  if (!sid) { window.location.href = "/"; return; }

  document.getElementById("btn-download-csv").href =
    `/api/sessions/${sid}/results/csv`;

  try {
    const data = await apiGet(`/api/sessions/${sid}/results`);
    if (!data.results?.length) {
      showEl(document.getElementById("no-results"));
      return;
    }
    renderTable(data.results, data.best);
    renderBestCard(data.best);
    renderChart(data.results);
  } catch (err) {
    showEl(document.getElementById("no-results"));
  }
});

function renderTable(results, best) {
  const tbody = document.getElementById("results-rows");

  // Ordena por Fβ desc
  const sorted = [...results].sort((a, b) =>
    parseFloat(b.fb || 0) - parseFloat(a.fb || 0));

  tbody.innerHTML = sorted.map(r => {
    const isBest = best && r.method === best.method && r.budget == best.budget;
    return `
      <tr>
        <td class="${isBest ? "best" : ""}">${r.method} ${isBest ? "🏆" : ""}</td>
        <td>K=${r.budget}</td>
        <td class="${isBest ? "best" : ""}">${fmtFloat(r.fb)}</td>
        <td>${fmtFloat(r.iou)}</td>
        <td>${fmtFloat(r.dice)}</td>
        <td>${r.elapsed_s ? fmtTime(parseFloat(r.elapsed_s)) : "—"}</td>
      </tr>
    `;
  }).join("");
}

function renderBestCard(best) {
  if (!best) return;
  const card = document.getElementById("best-model-card");
  document.getElementById("best-model-info").innerHTML = `
    <p><strong>Método:</strong> ${best.method}</p>
    <p><strong>Budget:</strong> K=${best.budget}</p>
    <p><strong>Fβ:</strong> <span style="color:var(--success);font-size:1.3rem;font-weight:700">${fmtFloat(best.fb)}</span></p>
    <p><strong>IoU:</strong> ${fmtFloat(best.iou)}</p>
  `;
  showEl(card);
}

function renderChart(results) {
  /**
   * TODO: renderizar gráfico de linhas Fβ × Budget para cada método.
   * Opções:
   *   1. Chart.js (CDN)  → <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
   *   2. SVG gerado manualmente (sem dependências)
   *
   * Por ora, exibe tabela de grupos.
   */
  const canvas  = document.getElementById("fb-chart");
  const section = document.getElementById("chart-section");

  // Agrupa por método
  const byMethod = {};
  results.forEach(r => {
    if (!byMethod[r.method]) byMethod[r.method] = [];
    byMethod[r.method].push({ budget: parseInt(r.budget), fb: parseFloat(r.fb || 0) });
  });

  // Placeholder visual (implementar Chart.js na fase de implementação)
  section.innerHTML = `
    <h3>Fβ por método e budget</h3>
    <p style="color:var(--text-muted);font-size:0.85rem">
      [Gráfico será renderizado com Chart.js na implementação final]
    </p>
    <pre style="background:var(--bg3);padding:1rem;border-radius:8px;font-size:0.8rem;overflow:auto">
${Object.entries(byMethod).map(([m, pts]) =>
  `${m.padEnd(25)} ${pts.sort((a,b)=>a.budget-b.budget).map(p => `K=${p.budget}:${fmtFloat(p.fb)}`).join("  ")}`
).join("\n")}
    </pre>
  `;
}
