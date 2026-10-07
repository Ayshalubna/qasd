/* Qasd demo: the fine-tuned model runs in the browser with transformers.js (ONNX Runtime Web). No server. */
const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";
const MODEL_HOST = "https://raw.githubusercontent.com/Ayshalubna/qasd/";   // the CI job publishes the model to the "model" branch
const MODEL_PATH = "model/export/";
const MODEL_BASE = MODEL_HOST + MODEL_PATH;
const $ = (s, el = document) => el.querySelector(s);
const main = $("#main");
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const pretty = (s) => { const t = s.replace(/_/g, " ").replace("?", "").trim(); return t.charAt(0).toUpperCase() + t.slice(1); };

let ROUTE = {};
let results = null;
let classifier = null, loading = null, lastText = "";
const loadState = { pct: 0, msg: "" };

const EXAMPLES = [
  "I think someone has stolen my card and is using it",
  "My transfer to my brother still hasn't arrived after 3 days",
  "Why was I charged a fee for withdrawing cash?",
  "The exchange rate on my card payment in Dubai looks wrong",
  "How do I top up with Apple Pay?",
  "I was charged twice for the same coffee",
  "Why do you need to verify my identity again?",
  "My card got stuck in the ATM",
];

// ------------------------------------------------------------------ model
function loadModel(onProgress) {
  if (loading) return loading;
  loading = (async () => {
    const { pipeline, env } = await import(TJS);
    env.allowLocalModels = false;
    env.remoteHost = MODEL_HOST;
    env.remotePathTemplate = MODEL_PATH;
    const files = {};
    classifier = await pipeline("text-classification", "qasd", {
      dtype: "q8", device: "wasm",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          files[p.file] = [p.loaded, p.total];
          const [l, t] = Object.values(files).reduce((a, b) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
          onProgress(l / t, `Downloading the model once (${(t / 1e6).toFixed(0)} MB, then cached)… ${Math.round((100 * l) / t)}%`);
        }
      },
    });
    onProgress(1, "Model ready. It runs in this tab; your message is never sent anywhere.");
    return classifier;
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

// ------------------------------------------------------------------ pages
const PAGES = { "": pageRoute, results: pageResults, how: pageHow };
function route() {
  const name = location.hash.replace(/^#\/?/, "");
  const page = PAGES[name] ? name : "";
  document.querySelectorAll(".nav a").forEach((a) => {
    if (a.getAttribute("href").replace(/^#\/?/, "") === page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  PAGES[page]();
}
window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });

function pageRoute() {
  const lora = results && results.methods && results.methods.lora;
  main.innerHTML = `
    <h1>Send every customer message to the right team, instantly</h1>
    <p class="lede">A DistilBERT model fine-tuned with LoRA reads a bank customer's message, recognises which of 77 banking requests it is${lora ? ` (${pct(lora.test_accuracy)} accurate on 3,080 unseen messages)` : ""}, and routes it to the team that handles it. It runs entirely in your browser.</p>
    <div class="grid">
      <div class="card">
        <label for="msg" class="small muted">Customer message</label>
        <textarea id="msg" maxlength="500" placeholder="Type what a customer might write…">${esc(lastText || EXAMPLES[0])}</textarea>
        <div class="row"><button class="btn" id="go" type="button">Route it</button><span class="status" id="status" role="status"></span></div>
        <div class="progress" id="prog" hidden><i></i></div>
        <div class="chips">${EXAMPLES.map((e) => `<button class="chip" type="button">${esc(e)}</button>`).join("")}</div>
      </div>
      <div class="card" id="out" aria-live="polite"><p class="muted">The routing decision appears here.</p></div>
    </div>
    <p class="note">The model only knows Banking77's 77 English intents (cards, transfers, top-ups, fees, identity checks…). Anything outside that will still be forced into one of them, which is why low-confidence results go to a human triage queue.</p>`;
  const status = $("#status"), prog = $("#prog");
  const progress = (f, msg) => { prog.hidden = f >= 1; $("i", prog).style.width = (100 * f).toFixed(0) + "%"; status.textContent = msg; };
  const run = async () => {
    const text = $("#msg").value.trim();
    if (!text) return;
    lastText = text;
    $("#go").disabled = true;
    try {
      if (!classifier) { progress(0, "Loading the model…"); prog.hidden = false; }
      const clf = await loadModel(progress);
      const t0 = performance.now();
      const top = await clf(text, { top_k: 3 });
      show(top, performance.now() - t0);
    } catch (e) {
      console.error(e);
      status.textContent = "The model could not be loaded in this browser. Try a recent Chrome, Edge, Firefox or Safari.";
    } finally { $("#go").disabled = false; }
  };
  $("#go").onclick = run;
  $("#msg").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
  main.querySelectorAll(".chip").forEach((c) => (c.onclick = () => { $("#msg").value = c.textContent; run(); }));
  loadModel(progress).catch(() => progress(0, "The model could not be loaded."));
  prog.hidden = false;
}

function show(top, ms) {
  const best = top[0];
  const [team, urgency] = ROUTE[best.label] || ["General enquiries", "normal"];
  const confident = best.score >= 0.5;
  const t = confident ? team : "Human triage queue";
  const u = confident ? urgency : "normal";
  $("#out").innerHTML = `
    <div class="route ${u}">
      <div class="small muted">Route to</div>
      <div class="team">${esc(t)}${u !== "normal" ? `<span class="badge ${u}">${u}</span>` : ""}</div>
      <div class="why">${confident ? `Recognised as “${esc(pretty(best.label))}” with ${pct(best.score, 0)} confidence.` : `Not sure (top guess ${pct(best.score, 0)}), so a person should look at it.`}</div>
    </div>
    <h2>Top 3 intents</h2>
    <div class="intents">${top.map((r) => `<div class="intent"><span>${esc(pretty(r.label))} <small>· ${esc((ROUTE[r.label] || ["?"])[0])}</small></span><b>${pct(r.score)}</b><div class="bar"><i style="width:${(100 * r.score).toFixed(1)}%"></i></div></div>`).join("")}</div>
    <p class="small muted" style="margin-top:14px">Answered in ${ms.toFixed(0)} ms inside this browser tab.</p>`;
}

// ------------------------------------------------------------------ results
function curves(m) {
  const W = 1000, H = 280, L = 50, B = 28, T = 12, R = 14;
  const series = [["full", "#0F2D3A"], ["lora", "#12857A"], ["head", "#B97A12"]].filter(([k]) => m[k]);
  const maxE = Math.max(...series.map(([k]) => m[k].history.length));
  const vals = series.flatMap(([k]) => m[k].history.map((h) => h.valid_accuracy));
  const lo = Math.max(0, Math.floor(Math.min(...vals) * 10) / 10), hi = 1;
  const X = (e) => L + ((e - 1) / Math.max(1, maxE - 1)) * (W - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const grid = [lo, (lo + hi) / 2, hi].map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#E8EEEC"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${pct(v, 0)}</text>`).join("");
  const lines = series.map(([k, c]) => `<path d="${m[k].history.map((h, i) => `${i ? "L" : "M"}${X(h.epoch)},${Y(h.valid_accuracy)}`).join("")}" fill="none" stroke="${c}" stroke-width="2.4"/>` +
    m[k].history.map((h) => `<circle cx="${X(h.epoch)}" cy="${Y(h.valid_accuracy)}" r="3" fill="${c}"/>`).join("")).join("");
  const xt = Array.from({ length: maxE }, (_, i) => `<text x="${X(i + 1)}" y="${H - 8}" text-anchor="middle">${i + 1}</text>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation accuracy by epoch">${grid}${lines}${xt}</svg>
    <div class="legend">${series.map(([k, c]) => `<span><i style="background:${c}"></i>${esc(m[k].label)}</span>`).join("")}<span class="muted">x: epoch · y: validation accuracy</span></div>`;
}

function pageResults() {
  if (!results) {
    main.innerHTML = `<h1>Fine-tuning results</h1><p class="lede">The training run's results are not published yet. They are produced by the CI training job on GitHub and appear here automatically.</p>`;
    return;
  }
  const m = results.methods, full = m.full, lora = m.lora, head = m.head, ex = results.export || {};
  const rows = ["full", "lora", "head"].filter((k) => m[k]).map((k) => {
    const r = m[k];
    return `<tr class="${k === results.deployed ? "pick" : ""}"><td>${esc(r.label)}</td><td>${pct(r.test_accuracy)}</td><td>${pct(r.test_macro_f1)}</td>
      <td>${(r.trainable_params / 1e6).toFixed(2)} M <span class="muted">(${r.trainable_pct.toFixed(2)}%)</span></td>
      <td>${(r.train_seconds / 60).toFixed(1)} min</td><td>${(r.peak_memory_mb / 1024).toFixed(2)} GB</td><td>${r.task_checkpoint_mb.toFixed(1)} MB</td></tr>`;
  }).join("");
  const gap = full && lora ? (full.test_accuracy - lora.test_accuracy) * 100 : null;
  main.innerHTML = `
    <h1>Fine-tuning results</h1>
    <p class="lede">The same pretrained model (${esc(results.base_model)}, ${(full ? full.total_params / 1e6 : 66).toFixed(0)}M parameters) adapted three ways on ${results.data.train.toLocaleString()} messages, scored on the official ${results.data.test.toLocaleString()}-message test set. Trained on a ${esc(results.hardware || "CPU")} by the CI job; every run is tracked in MLflow.</p>
    <div class="figs">
      <div class="fig dark"><div class="v">${pct(lora.test_accuracy)}</div><p>LoRA test accuracy${gap !== null ? ` (${gap >= 0 ? gap.toFixed(1) + " points below" : (-gap).toFixed(1) + " points above"} full fine-tuning)` : ""}</p></div>
      <div class="fig"><div class="v">${lora.trainable_pct.toFixed(1)}%</div><p>of the weights trained (${(lora.trainable_params / 1e6).toFixed(2)}M of ${(lora.total_params / 1e6).toFixed(0)}M)</p></div>
      <div class="fig"><div class="v">${full ? (full.task_checkpoint_mb / lora.task_checkpoint_mb).toFixed(0) + "×" : "–"}</div><p>smaller per-task checkpoint than full fine-tuning (${lora.task_checkpoint_mb.toFixed(1)} MB vs ${full ? full.task_checkpoint_mb.toFixed(0) : "–"} MB)</p></div>
      <div class="fig"><div class="v">${ex.onnx_int8 ? ex.onnx_int8.size_mb.toFixed(0) + " MB" : "–"}</div><p>deployed int8 ONNX model${ex.onnx_int8 ? `, ${pct(ex.onnx_int8.test_accuracy)} accurate, ${pct(ex.onnx_int8.agreement_with_pytorch)} agreement with PyTorch` : ""}</p></div>
    </div>
    <div class="card"><h2>Comparison</h2><div class="scroll"><table>
      <thead><tr><th>Method</th><th>Test accuracy</th><th>Macro-F1</th><th>Trained weights</th><th>Training time</th><th>Peak memory</th><th>Per-task checkpoint</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="note">Highlighted: the deployed model. "Per-task checkpoint" is what has to be stored for each new task on top of the shared base model: the whole model for full fine-tuning, only the adapter (plus classifier) for LoRA, only the classifier for the frozen-encoder baseline.</p></div>
    <div class="card" style="margin-top:16px"><h2>Learning curves</h2>${curves(m)}</div>
    ${results.findings ? `<div class="card prose" style="margin-top:16px"><h2>What this shows</h2><ul>${results.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></div>` : ""}`;
}

function pageHow() {
  main.innerHTML = `
    <h1>How it works</h1>
    <div class="flow">
      <div><b>1 · Data</b>Banking77: 10,003 training and 3,080 test messages, 77 intents. 10 per intent held out for validation.</div>
      <div><b>2 · Adapt</b>DistilBERT + LoRA (rank 8 on attention query/value) and a new classifier head. Compared with full fine-tuning and a frozen-encoder baseline.</div>
      <div><b>3 · Track</b>Each run logs parameters, curves, time, memory and checkpoint size to MLflow; the best epoch on validation is kept.</div>
      <div><b>4 · Ship</b>Merge the adapter, export to ONNX, quantise to int8; check accuracy and agreement with PyTorch before publishing.</div>
      <div><b>5 · Serve</b>FastAPI + ONNX Runtime for systems; the same file runs in this page with transformers.js.</div>
    </div>
    <div class="prose">
      <h2>Why LoRA</h2>
      <p>Full fine-tuning updates all 66M weights and needs a full copy of the model for every task. LoRA freezes them and learns small low-rank updates to the attention layers, so each new task (another product line, another language) adds a few megabytes instead of a whole model, and training needs less memory. The results page shows what that costs in accuracy here.</p>
      <h2>Routing</h2>
      <p>Each intent belongs to one of eight teams (fraud & security, cards, payments & transfers, top-ups, cash & ATM, FX & fees, disputes & refunds, account & identity). Fraud intents are flagged urgent. If the model's confidence is under 50%, the message goes to a human triage queue instead of a guess.</p>
      <h2>Reproduce</h2>
      <p>Every number on the results page comes from the CI training job (<code>.github/workflows/train.yml</code>), which trains all three methods from scratch on a GitHub runner's CPU, exports the model and publishes it with its results. Locally: <code>python -m scripts.train</code>.</p>
      <h2>Limits</h2>
      <ul><li>English only, and only these 77 intents; a real deployment needs the bank's own labelled messages (and Arabic).</li><li>Banking77 messages are short, clean single requests; real messages mix several requests.</li></ul>
    </div>`;
}

// ------------------------------------------------------------------ boot
Promise.allSettled([
  fetch("routing.json").then((r) => r.json()).then((j) => { ROUTE = j; }),
  fetch(MODEL_BASE + "results.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).then((j) => { results = j; }),
]).then(route);
