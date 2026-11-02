// Proportio – lokale Gesichtsanalyse mit MediaPipe Face Landmarker
// Pfade lassen sich über window.PROPORTIO_CONFIG überschreiben (z. B. für selbst gehostete Modelle)
const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const CFG = Object.assign({
  bundleUrl: `${MP_BASE}/vision_bundle.mjs`,
  wasmLoaderUrl: `${MP_BASE}/wasm/vision_wasm_internal.js`,
  wasmBinaryUrl: `${MP_BASE}/wasm/vision_wasm_internal.wasm`,
  modelUrl: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}, window.PROPORTIO_CONFIG || {});
const abs = (u) => new URL(u, document.baseURI).href;
let MP = null; // { FaceLandmarker, DrawingUtils }

// ---------- Landmark-Indizes (MediaPipe Face Mesh, 478 Punkte) ----------
const L = {
  foreheadTop: 10, glabella: 9, noseBridge: 168, noseTip: 1, subnasale: 2, menton: 152,
  upperLipTop: 0, stomionUp: 13, stomionLow: 14, lowerLipBottom: 17,
  rEyeOuter: 33, rEyeInner: 133, lEyeOuter: 263, lEyeInner: 362,
  rEyeTop: 159, rEyeBottom: 145, lEyeTop: 386, lEyeBottom: 374,
  rIris: 468, lIris: 473,
  rZygo: 234, lZygo: 454, rGonion: 172, lGonion: 397,
  rMouth: 61, lMouth: 291, rAla: 64, lAla: 294,
  rBrow: 105, lBrow: 334,
};
const SYM_PAIRS = [[33,263],[133,362],[159,386],[145,374],[105,334],[70,300],[61,291],[64,294],[234,454],[172,397],[58,288],[132,361],[50,280],[205,425]];
const MIDLINE = [10, 9, 168, 6, 197, 195, 5, 4, 1, 2, 0, 17, 152];

// ---------- Metrik-Definitionen ----------
// ideal: [min, max] je Geschlecht; tol = Abstand außerhalb des Bereichs, bei dem der Score ~6/10 beträgt
const METRICS = [
  { id: "symmetry", name: "Gesichtssymmetrie", cat: "harmony", unit: "%", dec: 1, weight: 2,
    ideal: { m: [93, 100], f: [93, 100] }, tol: 4, range: [75, 100],
    info: "Wie spiegelgleich linke und rechte Gesichtshälfte sind (gemessen an 14 Punktpaaren).",
    tipLow: "Leichte Asymmetrien sind normal. Achte auf gerade Kopfhaltung beim Foto, gleichmäßiges Kauen und Schlafen nicht immer auf derselben Seite.",
  },
  { id: "thirds", name: "Mittel- zu Untergesicht", cat: "harmony", unit: "", dec: 2, weight: 1.5,
    ideal: { m: [0.92, 1.05], f: [0.95, 1.1] }, tol: 0.12, range: [0.7, 1.4],
    info: "Verhältnis Glabella→Nasenansatz zu Nasenansatz→Kinn. Klassisch ideal sind gleich große Gesichtsdrittel (≈ 1,0).",
    tipLow: "Ein längeres Untergesicht lässt sich optisch durch Bartkontur (Männer) oder Konturing balancieren.",
    tipHigh: "Ein kürzeres Untergesicht wirkt durch mehr Kinnprojektion (z. B. Bart am Kinn, Haltung: Kinn leicht nach vorn) ausgeglichener.",
  },
  { id: "fwhr", name: "FWHR (Breite/Höhe)", cat: "harmony", unit: "", dec: 2, weight: 1.5,
    ideal: { m: [1.85, 2.05], f: [1.75, 1.95] }, tol: 0.15, range: [1.4, 2.4],
    info: "Facial Width-to-Height Ratio: Jochbeinbreite geteilt durch Höhe Augenbrauen→Oberlippe. Höhere Werte wirken markanter.",
    tipLow: "Ein niedrigerer Körperfettanteil und Frisuren mit Volumen an den Seiten betonen die Gesichtsbreite.",
    tipHigh: "Frisuren mit mehr Höhe strecken das Gesicht optisch.",
  },
  { id: "canthal", name: "Canthal Tilt", cat: "eyes", unit: "°", dec: 1, weight: 1.5,
    ideal: { m: [4, 8], f: [5, 9] }, tol: 3, range: [-6, 14],
    info: "Neigung der Augenachse vom inneren zum äußeren Augenwinkel. Ein positiver Tilt gilt als attraktiv.",
    tipLow: "Gepflegte, leicht angehobene Augenbrauen-Enden und ausreichend Schlaf (weniger Schwellung) lassen die Augen offener und angehobener wirken.",
    tipHigh: "Ein stark positiver Tilt ist meist unproblematisch – Augen-Make-up kann ihn bei Bedarf horizontal ausgleichen.",
  },
  { id: "esr", name: "Augenabstand (ESR)", cat: "eyes", unit: "", dec: 3, weight: 1,
    ideal: { m: [0.45, 0.47], f: [0.45, 0.48] }, tol: 0.025, range: [0.38, 0.54],
    info: "Eye Separation Ratio: Pupillenabstand geteilt durch Jochbeinbreite. Ideal liegt bei ca. 0,46.",
    tipLow: "Augenbrauen mit etwas mehr Abstand zur Nasenwurzel lassen eng stehende Augen weiter wirken.",
    tipHigh: "Augenbrauen, die näher an der Nasenwurzel beginnen, balancieren weit stehende Augen.",
  },
  { id: "fifths", name: "Augen-Fünftel", cat: "eyes", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.9, 1.1], f: [0.9, 1.1] }, tol: 0.15, range: [0.6, 1.5],
    info: "Abstand der inneren Augenwinkel geteilt durch Augenbreite. Nach der Fünftel-Regel ideal ≈ 1,0.",
    tipLow: "Eng stehende Augen: Augenbrauen innen leicht ausdünnen.",
    tipHigh: "Weit stehende Augen: Brauen innen etwas voller tragen.",
  },
  { id: "eyeOpen", name: "Augenöffnung", cat: "eyes", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [0.26, 0.34], f: [0.3, 0.4] }, tol: 0.06, range: [0.12, 0.55],
    info: "Lidspalten-Höhe zu Augenbreite. Etwas geringere Werte bei Männern („hunter eyes“), höhere bei Frauen.",
    tipLow: "Ausreichend Schlaf, Kühlung und weniger Salz reduzieren Schwellungen der Lider.",
    tipHigh: "Ein entspannter, leicht fokussierter Blick wirkt auf Fotos markanter als weit aufgerissene Augen.",
  },
  { id: "midface", name: "Mittelgesichts-Ratio", cat: "midface", unit: "", dec: 2, weight: 1.25,
    ideal: { m: [0.95, 1.05], f: [0.97, 1.08] }, tol: 0.1, range: [0.7, 1.3],
    info: "Pupillenabstand geteilt durch Höhe Pupillenlinie→Oberlippe. Ein kompaktes Mittelgesicht (≈ 1,0) wirkt jugendlich.",
    tipLow: "Ein längeres Mittelgesicht kann durch einen gepflegten Schnurrbart oder Pony optisch verkürzt werden.",
  },
  { id: "noseWidth", name: "Nasenbreite", cat: "midface", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.95, 1.15], f: [0.9, 1.05] }, tol: 0.15, range: [0.6, 1.6],
    info: "Nasenflügelbreite geteilt durch den Abstand der inneren Augenwinkel. Ideal ≈ 1,0.",
    tipHigh: "Konturing an den Nasenflanken oder eine Frisur mit Mittelscheitel-Variation lenken den Fokus.",
    tipLow: "Eine schmale Nase ist ästhetisch meist unproblematisch.",
  },
  { id: "mouthNose", name: "Mund- zu Nasenbreite", cat: "lower", unit: "", dec: 2, weight: 1,
    ideal: { m: [1.5, 1.65], f: [1.5, 1.65] }, tol: 0.15, range: [1.0, 2.2],
    info: "Mundbreite geteilt durch Nasenbreite. Häufig zitiertes Ideal: ≈ 1,6 (Goldener Schnitt).",
    tipLow: "Lippenpflege und ein natürliches Lächeln lassen den Mund breiter wirken.",
  },
  { id: "lipRatio", name: "Lippenverhältnis", cat: "lower", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [1.4, 2.0], f: [1.3, 1.8] }, tol: 0.35, range: [0.6, 3.0],
    info: "Unterlippen- zu Oberlippenhöhe. Ideal ist eine Unterlippe, die ca. 1,6× so voll ist wie die Oberlippe.",
    tipLow: "Feuchtigkeitspflege und Peeling lassen die Lippen voller wirken.",
    tipHigh: "Lippenkontur an der Oberlippe betonen gleicht das Verhältnis optisch aus.",
  },
  { id: "chinPhiltrum", name: "Kinn zu Philtrum", cat: "lower", unit: "", dec: 2, weight: 1,
    ideal: { m: [2.0, 2.5], f: [1.8, 2.25] }, tol: 0.35, range: [1.0, 3.6],
    info: "Kinnhöhe (Unterlippe→Kinnspitze) geteilt durch Philtrumlänge (Nase→Oberlippe).",
    tipLow: "Ein kurzes Kinn wirkt mit Bart am Kinn oder korrekter Zungenhaltung („Mewing“-Haltung) prominenter.",
    tipHigh: "Ein langes Kinn lässt sich durch Bart an den Seiten (mehr Breite) ausgleichen.",
  },
  { id: "jaw", name: "Kieferbreite", cat: "lower", unit: "", dec: 2, weight: 1.25,
    ideal: { m: [0.88, 0.96], f: [0.8, 0.88] }, tol: 0.06, range: [0.65, 1.05],
    info: "Kieferwinkel-Breite geteilt durch Jochbeinbreite. Breitere Kiefer wirken maskuliner, schmalere femininer.",
    tipLow: "Weniger Körperfett, Kaumuskeltraining und ein seitlich voller Bart betonen die Kieferlinie.",
    tipHigh: "Ein breiter Kiefer ist markant – Frisuren mit Höhe balancieren die Proportionen.",
  },
];
const CATS = { harmony: "Harmonie", eyes: "Augen", midface: "Mittelgesicht", lower: "Untergesicht" };

// ---------- State ----------
let landmarker = null;
let sex = "m";
let current = null; // { img, lms, results }
let stream = null;
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// ---------- Modell ----------
async function getLandmarker() {
  if (landmarker) return landmarker;
  try { MP = await import(CFG.bundleUrl); }
  catch { throw new Error("Die KI-Bibliothek konnte nicht geladen werden. Bitte Internetverbindung prüfen und Werbe-/Skriptblocker für diese Seite deaktivieren."); }
  const fileset = { wasmLoaderPath: abs(CFG.wasmLoaderUrl), wasmBinaryPath: abs(CFG.wasmBinaryUrl) };
  let model;
  try {
    const r = await fetch(abs(CFG.modelUrl));
    if (!r.ok) throw new Error(r.status);
    model = new Uint8Array(await r.arrayBuffer());
  } catch { throw new Error("Das KI-Modell konnte nicht geladen werden. Bitte Seite neu laden."); }
  const opts = (delegate) => ({ baseOptions: { modelAssetBuffer: model, delegate }, runningMode: "IMAGE", numFaces: 1 });
  try { landmarker = await MP.FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { landmarker = await MP.FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
  return landmarker;
}

// ---------- Geometrie ----------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

// Pixelkoordinaten + Roll-Korrektur (Augenlinie horizontal)
function normalize(raw, w, h) {
  const px = raw.map((p) => ({ x: p.x * w, y: p.y * h }));
  const a = px[L.rEyeOuter], b = px[L.lEyeOuter];
  const ang = Math.atan2(b.y - a.y, b.x - a.x);
  const c = mid(a, b), cos = Math.cos(-ang), sin = Math.sin(-ang);
  const pts = px.map((p) => ({ x: c.x + (p.x - c.x) * cos - (p.y - c.y) * sin, y: c.y + (p.x - c.x) * sin + (p.y - c.y) * cos }));
  return { pts, rollDeg: (ang * 180) / Math.PI };
}

// Gierwinkel-Schätzung: Nasenspitze relativ zu den Jochbeinen
function yawEstimate(p) {
  const r = p[L.noseTip].x - p[L.rZygo].x, l = p[L.lZygo].x - p[L.noseTip].x;
  return (r - l) / (r + l); // 0 = frontal
}

function computeRaw(p) {
  const faceW = dist(p[L.rZygo], p[L.lZygo]);

  // Symmetrie: Spiegelung an der Mittellinie (x = Mittelwert der Mittellinienpunkte)
  const mx = MIDLINE.reduce((s, i) => s + p[i].x, 0) / MIDLINE.length;
  let err = 0;
  for (const [r, l] of SYM_PAIRS) {
    const mirrored = { x: 2 * mx - p[r].x, y: p[r].y };
    err += dist(mirrored, p[l]);
  }
  const symmetry = Math.max(0, 100 - (err / SYM_PAIRS.length / faceW) * 100 * 4);

  const browY = (p[L.rBrow].y + p[L.lBrow].y) / 2;
  const midH = p[L.subnasale].y - p[L.glabella].y;
  const lowH = p[L.menton].y - p[L.subnasale].y;

  const fwhr = faceW / (p[L.upperLipTop].y - browY);

  // Canthal Tilt (Bild bereits rollkorrigiert; y wächst nach unten)
  const tilt = (inner, outer, sign) => (Math.atan2(p[inner].y - p[outer].y, sign * (p[outer].x - p[inner].x)) * 180) / Math.PI;
  const canthal = (tilt(L.rEyeInner, L.rEyeOuter, -1) + tilt(L.lEyeInner, L.lEyeOuter, 1)) / 2;

  const ipd = dist(p[L.rIris], p[L.lIris]);
  const esr = ipd / faceW;

  const eyeWR = dist(p[L.rEyeInner], p[L.rEyeOuter]), eyeWL = dist(p[L.lEyeInner], p[L.lEyeOuter]);
  const eyeW = (eyeWR + eyeWL) / 2;
  const intercanthal = dist(p[L.rEyeInner], p[L.lEyeInner]);
  const fifths = intercanthal / eyeW;
  const eyeOpen = ((dist(p[L.rEyeTop], p[L.rEyeBottom]) / eyeWR) + (dist(p[L.lEyeTop], p[L.lEyeBottom]) / eyeWL)) / 2;

  const pupilY = (p[L.rIris].y + p[L.lIris].y) / 2;
  const midface = ipd / (p[L.upperLipTop].y - pupilY);

  const noseW = dist(p[L.rAla], p[L.lAla]);
  const noseWidth = noseW / intercanthal;
  const mouthNose = dist(p[L.rMouth], p[L.lMouth]) / noseW;

  const upperLip = p[L.stomionUp].y - p[L.upperLipTop].y;
  const lowerLip = p[L.lowerLipBottom].y - p[L.stomionLow].y;
  const lipRatio = lowerLip / Math.max(upperLip, 0.5);

  const philtrum = p[L.upperLipTop].y - p[L.subnasale].y;
  const chin = p[L.menton].y - p[L.lowerLipBottom].y;
  const chinPhiltrum = chin / Math.max(philtrum, 0.5);

  const jaw = dist(p[L.rGonion], p[L.lGonion]) / faceW;

  return { symmetry, thirds: midH / lowH, fwhr, canthal, esr, fifths, eyeOpen, midface, noseWidth, mouthNose, lipRatio, chinPhiltrum, jaw };
}

function scoreMetric(v, [lo, hi], tol) {
  if (v >= lo && v <= hi) {
    // innerhalb: 9.0–10 (Mitte = 10)
    const c = (lo + hi) / 2, half = (hi - lo) / 2 || 1;
    return 10 - Math.abs(v - c) / half;
  }
  const d = v < lo ? lo - v : v - hi;
  return Math.max(0, 9 * Math.exp(-0.4 * (d / tol) ** 2));
}

function evaluate(raw) {
  return METRICS.map((m) => {
    const v = raw[m.id];
    const ideal = m.ideal[sex];
    const score = scoreMetric(v, ideal, m.tol);
    const status = v < ideal[0] ? "low" : v > ideal[1] ? "high" : "ok";
    return { ...m, value: v, idealR: ideal, score, status };
  });
}

function overall(results) {
  const tw = results.reduce((s, r) => s + r.weight, 0);
  const raw = results.reduce((s, r) => s + r.score * r.weight, 0) / tw;
  // Kompression auf eine realistischere Skala (wenige Gesichter erreichen 10)
  return Math.min(10, Math.max(1, 0.3 + raw * 0.92));
}

const TIERS = [
  [8.5, "Herausragende Harmonie", "Deine Proportionen liegen fast überall im Idealbereich."],
  [7.5, "Sehr harmonisch", "Die meisten Metriken liegen im oder nahe am Idealbereich."],
  [6.5, "Überdurchschnittlich", "Solide Proportionen mit einzelnen Ausreißern."],
  [5.5, "Durchschnittlich", "Ausgewogen, mit einigen Bereichen außerhalb der Ideale."],
  [0, "Ausbaufähig", "Mehrere Werte liegen außerhalb der Idealbereiche – Foto-Bedingungen prüfen und Tipps ansehen."],
];

// ---------- Zeichnen ----------
function drawStage() {
  if (!current) return;
  const { img, rawLms } = current;
  canvas.width = img.width; canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  if (!rawLms || !MP) return;
  const { FaceLandmarker, DrawingUtils } = MP;
  const s = Math.max(1, img.width / 700);
  if ($("tglMesh").checked) {
    const du = new DrawingUtils(ctx);
    du.drawConnectors(rawLms, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "rgba(255,255,255,0.12)", lineWidth: 0.6 * s });
    du.drawConnectors(rawLms, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "rgba(124,92,255,0.9)", lineWidth: 1.5 * s });
    for (const k of ["FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE", "FACE_LANDMARKS_LIPS", "FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW", "FACE_LANDMARKS_LEFT_IRIS", "FACE_LANDMARKS_RIGHT_IRIS"])
      du.drawConnectors(rawLms, FaceLandmarker[k], { color: "rgba(34,211,238,0.9)", lineWidth: 1.2 * s });
  }
  if ($("tglLines").checked) {
    const P = (i) => ({ x: rawLms[i].x * img.width, y: rawLms[i].y * img.height });
    const line = (a, b, color, dash = []) => { ctx.strokeStyle = color; ctx.lineWidth = 2 * s; ctx.setLineDash(dash.map((d) => d * s)); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.setLineDash([]); };
    // Augenlinie als Referenz, Drittel-Linien senkrecht dazu
    const eyeA = P(L.rEyeOuter), eyeB = P(L.lEyeOuter);
    const ang = Math.atan2(eyeB.y - eyeA.y, eyeB.x - eyeA.x);
    const w = dist(P(L.rZygo), P(L.lZygo)) * 0.65;
    const hline = (pt, color) => line({ x: pt.x - Math.cos(ang) * w, y: pt.y - Math.sin(ang) * w }, { x: pt.x + Math.cos(ang) * w, y: pt.y + Math.sin(ang) * w }, color, [6, 6]);
    hline(P(L.glabella), "rgba(250,204,21,.9)");
    hline(P(L.subnasale), "rgba(250,204,21,.9)");
    hline(P(L.menton), "rgba(250,204,21,.9)");
    line(P(L.foreheadTop), P(L.menton), "rgba(255,255,255,.5)", [4, 6]);
    line(P(L.rZygo), P(L.lZygo), "rgba(124,92,255,.9)");
    line(P(L.rGonion), P(L.lGonion), "rgba(236,72,153,.9)");
    line(P(L.rEyeInner), P(L.rEyeOuter), "rgba(34,197,94,1)");
    line(P(L.lEyeInner), P(L.lEyeOuter), "rgba(34,197,94,1)");
  }
}

// ---------- UI ----------
function fmt(v, dec) { return v.toLocaleString("de-DE", { minimumFractionDigits: dec, maximumFractionDigits: dec }); }
function cls(score) { return score >= 8 ? "good" : score >= 5 ? "mid" : "bad"; }
function scoreColor(s) { return s >= 7.5 ? "#22c55e" : s >= 6 ? "#7c5cff" : s >= 5 ? "#eab308" : "#ef4444"; }

function renderResults(results) {
  const total = overall(results);
  $("results").hidden = false;
  $("overallScore").textContent = fmt(total, 1);
  const ring = $("ringFg");
  ring.style.stroke = scoreColor(total);
  requestAnimationFrame(() => { ring.style.strokeDashoffset = String(326.7 * (1 - total / 10)); });
  const tier = TIERS.find(([t]) => total >= t);
  $("tierLabel").textContent = tier[1];
  $("summaryText").textContent = tier[2] + ` ${results.filter((r) => r.status === "ok").length} von ${results.length} Metriken im Idealbereich.`;

  $("subScores").innerHTML = Object.entries(CATS).map(([k, label]) => {
    const rs = results.filter((r) => r.cat === k);
    const s = rs.reduce((a, r) => a + r.score * r.weight, 0) / rs.reduce((a, r) => a + r.weight, 0);
    return `<div class="sub"><span>${label}</span><strong style="color:${scoreColor(s)}">${fmt(s, 1)}</strong></div>`;
  }).join("");

  $("metricsGrid").innerHTML = results.map((r) => {
    const [a, b] = r.range;
    const pct = (v) => Math.min(100, Math.max(0, ((v - a) / (b - a)) * 100));
    const statusText = r.status === "ok" ? "Ideal" : r.status === "low" ? "Unter Ideal" : "Über Ideal";
    return `<article class="card metric" data-cat="${r.cat}">
      <div class="metric-top"><h4>${r.name}</h4><span class="badge ${cls(r.score)}">${fmt(r.score, 1)} · ${statusText}</span></div>
      <div class="val">${fmt(r.value, r.dec)}${r.unit}</div>
      <div class="scale"><div class="ideal" style="left:${pct(r.idealR[0])}%;width:${pct(r.idealR[1]) - pct(r.idealR[0])}%"></div><div class="marker" style="left:${pct(r.value)}%"></div></div>
      <div class="scale-legend"><span>${fmt(a, r.dec)}</span><span>Ideal ${fmt(r.idealR[0], r.dec)}–${fmt(r.idealR[1], r.dec)}${r.unit}</span><span>${fmt(b, r.dec)}</span></div>
      <p>${r.info}</p>
    </article>`;
  }).join("");
  applyFilter(document.querySelector(".tab.active").dataset.filter);

  const weak = results.filter((r) => r.status !== "ok").sort((x, y) => x.score * x.weight - y.score * y.weight).slice(0, 4);
  $("tipsList").innerHTML = weak.length
    ? weak.map((r) => `<li><strong>${r.name}:</strong> ${(r.status === "low" ? r.tipLow : r.tipHigh) || r.tipLow || r.tipHigh || "Liegt nahe am Ideal."}</li>`).join("")
    : "<li>Alle Metriken liegen im Idealbereich – stark! Fokussiere dich auf Hautpflege, Frisur und Fitness, um das Beste herauszuholen.</li>";

  current.results = results; current.total = total;
}

function applyFilter(f) {
  document.querySelectorAll(".metric").forEach((el) => { el.hidden = f !== "all" && el.dataset.cat !== f; });
}

function showError(msg) { const e = $("errorMsg"); e.textContent = msg; e.hidden = !msg; }
function setLoading(on, text) { $("loader").hidden = !on; if (text) $("loaderText").textContent = text; }

async function analyzeImage(img) {
  showError("");
  $("dzEmpty").hidden = true; $("stage").hidden = false;
  current = { img, rawLms: null };
  drawStage();
  setLoading(true, landmarker ? "Gesicht wird analysiert…" : "KI-Modell wird geladen (einmalig ~4 MB)…");
  try {
    const lm = await getLandmarker();
    setLoading(true, "Gesicht wird analysiert…");
    await new Promise((r) => setTimeout(r, 30));
    const res = lm.detect(img);
    if (!res.faceLandmarks || !res.faceLandmarks.length) throw new Error("Kein Gesicht erkannt. Bitte ein gut beleuchtetes Frontalfoto verwenden.");
    const rawLms = res.faceLandmarks[0];
    current.rawLms = rawLms;
    const { pts, rollDeg } = normalize(rawLms, img.width, img.height);
    const yaw = yawEstimate(pts);
    if (Math.abs(yaw) > 0.18) showError(`Hinweis: Der Kopf ist deutlich zur Seite gedreht (${Math.round(Math.abs(yaw) * 100)} %). Für genaue Werte frontal in die Kamera schauen.`);
    else if (Math.abs(rollDeg) > 8) showError(`Hinweis: Kopf um ${fmt(Math.abs(rollDeg), 0)}° geneigt – wurde automatisch korrigiert.`);
    current.lms = pts;
    drawStage();
    $("toggles").hidden = false;
    renderResults(evaluate(computeRaw(pts)));
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error(err);
    $("results").hidden = true;
    showError(err.message || "Analyse fehlgeschlagen.");
  } finally {
    setLoading(false);
  }
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return showError("Bitte eine Bilddatei wählen.");
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    // Große Bilder herunterskalieren
    const max = 1400, sc = Math.min(1, max / Math.max(img.width, img.height));
    const c = document.createElement("canvas");
    c.width = Math.round(img.width * sc); c.height = Math.round(img.height * sc);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    analyzeImage(c);
  };
  img.onerror = () => showError("Bild konnte nicht geladen werden.");
  img.src = url;
}

// Kamera
async function startCam() {
  showError("");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } } });
  } catch { return showError("Kein Kamerazugriff möglich."); }
  const v = $("video");
  v.srcObject = stream; await v.play();
  $("dzEmpty").hidden = true; $("stage").hidden = false;
  canvas.hidden = true; v.hidden = false; $("camBar").hidden = false;
}
function stopCam() {
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  $("video").hidden = true; canvas.hidden = false; $("camBar").hidden = true;
  if (!current) { $("stage").hidden = true; $("dzEmpty").hidden = false; }
}
function snap() {
  const v = $("video");
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  const cx = c.getContext("2d");
  cx.translate(c.width, 0); cx.scale(-1, 1); cx.drawImage(v, 0, 0);
  stopCam();
  analyzeImage(c);
}

// Verlauf (lokal)
const HKEY = "proportio-history";
function getHistory() { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } }
function setHistory(h) { try { localStorage.setItem(HKEY, JSON.stringify(h)); } catch { showError("Verlauf konnte nicht gespeichert werden (Speicher voll oder blockiert)."); } }
function thumb() {
  const c = document.createElement("canvas"), s = 160; c.width = c.height = s;
  const img = current.img, side = Math.min(img.width, img.height);
  c.getContext("2d").drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, s, s);
  return c.toDataURL("image/jpeg", 0.7);
}
function renderHistory() {
  const h = getHistory();
  $("history").hidden = !h.length;
  $("historyList").innerHTML = h.map((e) => `<div class="hist-item"><img src="${e.thumb}" alt="" /><strong style="color:${scoreColor(e.total)}">${fmt(e.total, 1)}</strong><span>${new Date(e.date).toLocaleDateString("de-DE")}</span></div>`).join("");
}
function saveToHistory() {
  if (!current?.results) return;
  const h = getHistory();
  h.unshift({ date: Date.now(), total: current.total, thumb: thumb(), values: Object.fromEntries(current.results.map((r) => [r.id, r.value])) });
  setHistory(h.slice(0, 12));
renderHistory();
  $("saveBtn").textContent = "Gespeichert ✓";
  setTimeout(() => ($("saveBtn").textContent = "Im Verlauf speichern"), 1800);
}

// Export
function exportImage() {
  if (!current?.results) return;
  const img = canvas, W = 1080, pad = 40;
  const imgH = Math.round((img.height / img.width) * (W - pad * 2));
  const rowH = 46, H = pad + 90 + imgH + 30 + current.results.length * rowH + pad;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d");
  x.fillStyle = "#0b0b12"; x.fillRect(0, 0, W, H);
  x.fillStyle = "#ececf4"; x.font = "800 40px Inter, sans-serif"; x.fillText("Proportio", pad, pad + 40);
  x.font = "700 40px Inter, sans-serif"; x.fillStyle = scoreColor(current.total); x.textAlign = "right";
  x.fillText(`${fmt(current.total, 1)} / 10`, W - pad, pad + 40); x.textAlign = "left";
  x.drawImage(img, pad, pad + 80, W - pad * 2, imgH);
  let y = pad + 80 + imgH + 50;
  x.font = "500 24px Inter, sans-serif";
  for (const r of current.results) {
    x.fillStyle = "#9a9ab3"; x.fillText(r.name, pad, y);
    x.fillStyle = "#ececf4"; x.textAlign = "right"; x.fillText(`${fmt(r.value, r.dec)}${r.unit}`, W - pad - 120, y);
    x.fillStyle = scoreColor(r.score); x.fillText(fmt(r.score, 1), W - pad, y); x.textAlign = "left";
    y += rowH;
  }
  const url = c.toDataURL("image/png");
  $("exportImg").src = url;
  const a = $("exportLink");
  a.href = url; a.download = `proportio-analyse-${new Date().toISOString().slice(0, 10)}.png`;
  $("exportModal").hidden = false;
}

// Metrik-Infokarten
$("metricInfo").innerHTML = METRICS.map((m) => `<article class="card"><h3>${m.name}</h3><p class="muted">${m.info}</p></article>`).join("");
$("year").textContent = new Date().getFullYear();

// Events
$("fileInput").addEventListener("change", (e) => { loadFile(e.target.files[0]); e.target.value = ""; });
const dz = $("dropZone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
$("camBtn").addEventListener("click", startCam);
$("camStop").addEventListener("click", stopCam);
$("snapBtn").addEventListener("click", snap);
$("tglMesh").addEventListener("change", drawStage);
$("tglLines").addEventListener("change", drawStage);
$("saveBtn").addEventListener("click", saveToHistory);
$("exportBtn").addEventListener("click", exportImage);
$("clearHistory").addEventListener("click", () => { setHistory([]); renderHistory(); });
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  t.classList.add("active"); applyFilter(t.dataset.filter);
}));
document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".seg-btn").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-checked", x === b); });
  sex = b.dataset.sex;
  if (current?.lms) renderResults(evaluate(computeRaw(current.lms)));
}));
if (CFG.noCamera) $("camBtn").hidden = true;
$("exportClose").addEventListener("click", () => ($("exportModal").hidden = true));
renderHistory();
