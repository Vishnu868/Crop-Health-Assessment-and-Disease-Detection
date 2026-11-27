const $ = (id) => document.getElementById(id);

$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) { $("preview").src = URL.createObjectURL(f); $("preview").hidden = false; }
});

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = $("file").files[0];
  if (!f) return;
  const body = new FormData();
  body.append("image", f);
  const r = await fetch("/predict", { method: "POST", body });
  const d = await r.json();
  if (!r.ok) { alert(d.error || "prediction failed"); return; }
  $("result").hidden = false;
  $("disease").textContent = d.disease;
  $("conf").textContent = (100 * d.confidence).toFixed(1) + "%";
  $("warn").hidden = !d.low_confidence;
  $("top3").innerHTML = d.top3.map((t) =>
    `<div class="row"><span>${t.class.replace("_", " ")}</span><div class="track"><div class="fill" style="width:${100 * t.prob}%"></div></div><span>${(100 * t.prob).toFixed(1)}%</span></div>`).join("");
  const hs = d.health.health_score;
  $("healthbar").style.width = `${Math.max(0, Math.min(100, hs))}%`;
  $("healthtxt").textContent = `Health score ${hs.toFixed(0)}/100 · lesion area ${(100 * d.health.lesion_fraction).toFixed(1)}% · VARI ${d.health.vari.toFixed(2)} · ExG ${d.health.exg.toFixed(2)}`;
  for (const k of ["symptoms", "fertilizer", "pesticide", "precautions"]) $(k).textContent = d[k];
});

async function pollSensors() {
  try {
    const d = await (await fetch("/api/sensors")).json();
    if (d.moisture !== undefined) {
      $("sensors").innerHTML = `Soil moisture <b>${d.moisture.toFixed(0)}%</b> · Temp <b>${d.temperature.toFixed(1)} °C</b> · Humidity <b>${d.humidity.toFixed(0)}%</b><br>Pump <b>${d.pump ? "ON" : "OFF"}</b> · Disease risk <b>${d.disease_risk}</b>`;
    }
  } catch (_) { /* server offline */ }
}
pollSensors();
setInterval(pollSensors, 5000);
