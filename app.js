// Proportio – lokale Gesichtsanalyse (frontal + Seitenprofil) mit MediaPipe Face Landmarker
// Pfade lassen sich über window.PROPORTIO_CONFIG überschreiben (z. B. für selbst gehostete Modelle)
const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const CFG = Object.assign({
  bundleUrl: `${MP_BASE}/vision_bundle.mjs`,
  wasmLoaderUrl: `${MP_BASE}/wasm/vision_wasm_internal.js`,
  wasmBinaryUrl: `${MP_BASE}/wasm/vision_wasm_internal.wasm`,
  modelUrl: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}, window.PROPORTIO_CONFIG || {});
const abs = (u) => new URL(u, document.baseURI).href;
let MP = null; // { FaceLandmarker, ... }

// ======================= Geometrie-Helfer =======================
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const rot = (p, c, th) => { const co = Math.cos(th), si = Math.sin(th), dx = p.x - c.x, dy = p.y - c.y; return { x: c.x + dx * co - dy * si, y: c.y + dx * si + dy * co }; };
const deg = (r) => (r * 180) / Math.PI;
const angleAt = (a, v, b) => { const a1 = Math.atan2(a.y - v.y, a.x - v.x), a2 = Math.atan2(b.y - v.y, b.x - v.x); let d = Math.abs(a1 - a2); if (d > Math.PI) d = 2 * Math.PI - d; return deg(d); };
const lineAngle = (a, b, c, d) => { let x = Math.abs(Math.atan2(b.y - a.y, b.x - a.x) - Math.atan2(d.y - c.y, d.x - c.x)) % Math.PI; if (x > Math.PI / 2) x = Math.PI - x; return deg(x); };
const ext = (a, b, f) => ({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f });
const foot = (p, a, b) => { const dx = b.x - a.x, dy = b.y - a.y, t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy); return { x: a.x + t * dx, y: a.y + t * dy }; };
const sideOf = (p, a, b) => Math.sign((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));

// ======================= Farben & Primitive =======================
const C = { cyan: "#5eead4", yellow: "#facc15", pink: "#f472b6", violet: "#a78bfa", green: "#4ade80", red: "#f87171", white: "rgba(255,255,255,.75)" };
const seg = (a, b, color = C.cyan, label, dash) => ({ t: "seg", a, b, color, label, dash });
const vseg = (x, y1, y2, color, label) => seg({ x, y: y1 }, { x, y: y2 }, color, label);
const hl = (y, x1, x2, color = "rgba(250,204,21,.8)", label) => seg({ x: x1, y }, { x: x2, y }, color, label, [10, 8]);
const arc = (a, v, b, color, label) => ({ t: "arc", a, v, b, color, label });
const ghost = (p, color = C.red) => ({ t: "ghost", p, color });
const tag = (p, text, color = C.white) => ({ t: "tag", p, text, color });

// ======================= FRONTAL =======================
const L = {
  foreheadTop: 10, glabella: 9, nasion: 168, noseTip: 1, subnasale: 2, menton: 152,
  upperLipTop: 0, stomionUp: 13, stomionLow: 14, lowerLipBottom: 17,
  rEyeOuter: 33, rEyeInner: 133, lEyeOuter: 263, lEyeInner: 362,
  rEyeTop: 159, rEyeBottom: 145, lEyeTop: 386, lEyeBottom: 374, rIris: 468, lIris: 473,
  rZygo: 234, lZygo: 454, rGonion: 172, lGonion: 397, rMouth: 61, lMouth: 291, rAla: 64, lAla: 294, rBrow: 105, lBrow: 334,
};
const SYM_PAIRS = [[33,263],[133,362],[159,386],[145,374],[105,334],[70,300],[61,291],[64,294],[234,454],[172,397],[58,288],[132,361],[50,280],[205,425]];
const MIDLINE = [10, 9, 168, 6, 197, 195, 5, 4, 1, 2, 0, 17, 152];
// R/L = rechte/linke Gesichtshälfte der Person
const FRONT_KP = [
  [9, "Glabella"], [168, "Nasion (Nasenwurzel)"], [1, "Nasenspitze"], [2, "Subnasale"], [0, "Oberlippe"], [13, "Stomion oben"], [14, "Stomion unten"], [17, "Unterlippe"], [152, "Menton (Kinn)"],
  [33, "Äußerer Augenwinkel R"], [133, "Innerer Augenwinkel R"], [362, "Innerer Augenwinkel L"], [263, "Äußerer Augenwinkel L"],
  [159, "Oberlid R"], [145, "Unterlid R"], [386, "Oberlid L"], [374, "Unterlid L"], [468, "Pupille R"], [473, "Pupille L"],
  [105, "Braue R"], [334, "Braue L"], [234, "Jochbein R"], [454, "Jochbein L"], [172, "Kieferwinkel R"], [397, "Kieferwinkel L"],
  [61, "Mundwinkel R"], [291, "Mundwinkel L"], [64, "Nasenflügel R"], [294, "Nasenflügel L"],
  [55, "Brauenkopf R"], [285, "Brauenkopf L"], [46, "Brauenende R"], [276, "Brauenende L"], [52, "Braue unten R"], [282, "Braue unten L"],
].map(([id, label]) => ({ id, label }));
const midX = (P) => MIDLINE.reduce((s, i) => s + P[i].x, 0) / MIDLINE.length;

function computeFront(p) {
  const faceW = dist(p[L.rZygo], p[L.lZygo]);
  const mx = midX(p);
  let err = 0;
  for (const [r, l] of SYM_PAIRS) err += dist({ x: 2 * mx - p[r].x, y: p[r].y }, p[l]);
  const symmetry = Math.max(0, 100 - (err / SYM_PAIRS.length / faceW) * 400);
  const browY = (p[L.rBrow].y + p[L.lBrow].y) / 2;
  const tilt = (inner, outer, sign) => deg(Math.atan2(p[inner].y - p[outer].y, sign * (p[outer].x - p[inner].x)));
  const tiltR = tilt(L.rEyeInner, L.rEyeOuter, -1), tiltL = tilt(L.lEyeInner, L.lEyeOuter, 1);
  const ipd = dist(p[L.rIris], p[L.lIris]);
  const eyeWR = dist(p[L.rEyeInner], p[L.rEyeOuter]), eyeWL = dist(p[L.lEyeInner], p[L.lEyeOuter]);
  const intercanthal = dist(p[L.rEyeInner], p[L.lEyeInner]);
  const pupilY = (p[L.rIris].y + p[L.lIris].y) / 2;
  const noseW = dist(p[L.rAla], p[L.lAla]);
  const upperLip = p[L.stomionUp].y - p[L.upperLipTop].y, lowerLip = p[L.lowerLipBottom].y - p[L.stomionLow].y;
  const philtrum = p[L.upperLipTop].y - p[L.subnasale].y, chin = p[L.menton].y - p[L.lowerLipBottom].y;
  const mouthW = dist(p[L.rMouth], p[L.lMouth]);
  const btR = tilt(55, 46, -1), btL = tilt(285, 276, 1);
  return {
    faceIndex: (p[152].y - p[168].y) / faceW,
    noseLength: (p[2].y - p[168].y) / (p[152].y - p[168].y),
    eyeWidth: (eyeWR + eyeWL) / 2 / faceW,
    browTilt: (btR + btL) / 2, _btR: btR, _btL: btL,
    browEye: ((p[159].y - p[52].y) / eyeWR + (p[386].y - p[282].y) / eyeWL) / 2,
    lipFull: (p[17].y - p[0].y) / mouthW,
    mouthIpd: mouthW / ipd,
    _foreheadW: dist(p[71], p[301]) / faceW,
    symmetry,
    thirds: (p[L.subnasale].y - p[L.glabella].y) / (p[L.menton].y - p[L.subnasale].y),
    fwhr: faceW / (p[L.upperLipTop].y - browY),
    canthal: (tiltR + tiltL) / 2, _tiltR: tiltR, _tiltL: tiltL,
    esr: ipd / faceW,
    fifths: intercanthal / ((eyeWR + eyeWL) / 2),
    eyeOpen: (dist(p[L.rEyeTop], p[L.rEyeBottom]) / eyeWR + dist(p[L.lEyeTop], p[L.lEyeBottom]) / eyeWL) / 2,
    midface: ipd / (p[L.upperLipTop].y - pupilY),
    noseWidth: noseW / intercanthal,
    mouthNose: dist(p[L.rMouth], p[L.lMouth]) / noseW,
    lipRatio: lowerLip / Math.max(upperLip, 0.5),
    chinPhiltrum: chin / Math.max(philtrum, 0.5),
    jaw: dist(p[L.rGonion], p[L.lGonion]) / faceW,
  };
}

const FRONT_CATS = { harmony: "Harmonie", eyes: "Augen", midface: "Mittelgesicht", lower: "Untergesicht" };
const FRONT_METRICS = [
  { id: "symmetry", name: "Gesichtssymmetrie", cat: "harmony", unit: "%", dec: 1, weight: 2,
    ideal: { m: [93, 100], f: [93, 100] }, tol: 4, range: [75, 100],
    info: "Wie spiegelgleich linke und rechte Gesichtshälfte sind (gemessen an 14 Punktpaaren).",
    tipLow: "Leichte Asymmetrien sind normal. Achte auf gerade Kopfhaltung beim Foto, gleichmäßiges Kauen und Schlafen nicht immer auf derselben Seite.",
    kp: [9, 152, 33, 263, 133, 362, 61, 291, 234, 454, 172, 397, 64, 294, 105, 334],
    viz: (P) => { const mx = midX(P); const out = [seg({ x: mx, y: P[10].y - 30 }, { x: mx, y: P[152].y + 30 }, C.white, "Mittellinie", [10, 8])];
      for (const [r, l] of SYM_PAIRS) { const m = { x: 2 * mx - P[r].x, y: P[r].y }; out.push(ghost(m, C.pink), seg(m, P[l], C.pink)); }
      return out; } },
  { id: "thirds", name: "Mittel- zu Untergesicht", cat: "harmony", unit: "", dec: 2, weight: 1.5,
    ideal: { m: [0.92, 1.05], f: [0.95, 1.1] }, tol: 0.12, range: [0.7, 1.4],
    info: "Verhältnis Glabella→Nasenansatz zu Nasenansatz→Kinn. Klassisch ideal sind gleich große Gesichtsdrittel (≈ 1,0).",
    tipLow: "Ein längeres Untergesicht lässt sich optisch durch Bartkontur (Männer) oder Konturing balancieren.",
    tipHigh: "Ein kürzeres Untergesicht wirkt durch mehr Kinnprojektion (z. B. Bart am Kinn, Kinn leicht nach vorn) ausgeglichener.",
    kp: [9, 2, 152],
    viz: (P) => { const x1 = P[234].x - 20, x2 = P[454].x + 20, bx = P[454].x - 10;
      return [hl(P[9].y, x1, x2), hl(P[2].y, x1, x2), hl(P[152].y, x1, x2), vseg(bx, P[9].y, P[2].y, C.cyan, "Mitte"), vseg(bx, P[2].y, P[152].y, C.pink, "Unten")]; } },
  { id: "fwhr", name: "FWHR (Breite/Höhe)", cat: "harmony", unit: "", dec: 2, weight: 1.5,
    ideal: { m: [1.85, 2.05], f: [1.75, 1.95] }, tol: 0.15, range: [1.4, 2.4],
    info: "Facial Width-to-Height Ratio: Jochbeinbreite geteilt durch Höhe Augenbrauen→Oberlippe. Höhere Werte wirken markanter.",
    tipLow: "Ein niedrigerer Körperfettanteil und Frisuren mit Volumen an den Seiten betonen die Gesichtsbreite.",
    tipHigh: "Frisuren mit mehr Höhe strecken das Gesicht optisch.",
    kp: [234, 454, 105, 334, 0],
    viz: (P) => { const by = (P[105].y + P[334].y) / 2;
      return [hl(by, P[234].x, P[454].x), hl(P[0].y, P[234].x, P[454].x), seg(P[234], P[454], C.cyan, "Breite"), vseg(P[168].x, by, P[0].y, C.pink, "Höhe")]; } },
  { id: "canthal", name: "Canthal Tilt", cat: "eyes", unit: "°", dec: 1, weight: 1.5,
    ideal: { m: [4, 8], f: [5, 9] }, tol: 3, range: [-6, 14],
    info: "Neigung der Augenachse vom inneren zum äußeren Augenwinkel. Ein positiver Tilt gilt als attraktiv.",
    tipLow: "Gepflegte, leicht angehobene Augenbrauen-Enden und ausreichend Schlaf lassen die Augen offener und angehobener wirken.",
    tipHigh: "Ein stark positiver Tilt ist meist unproblematisch – Augen-Make-up kann ihn bei Bedarf horizontal ausgleichen.",
    kp: [33, 133, 362, 263],
    viz: (P, r) => { const out = [];
      for (const [i, o, v] of [[133, 33, r._tiltR], [362, 263, r._tiltL]]) {
        const h = { x: P[o].x + (P[o].x - P[i].x) * 0.35, y: P[i].y };
        out.push(seg(P[i], h, C.white, null, [6, 6]), seg(P[i], ext(P[i], P[o], 1.35), C.green), arc(h, P[i], P[o], C.yellow, `${fmt(v, 1)}°`));
      } return out; } },
  { id: "esr", name: "Augenabstand (ESR)", cat: "eyes", unit: "", dec: 3, weight: 1,
    ideal: { m: [0.45, 0.47], f: [0.45, 0.48] }, tol: 0.025, range: [0.38, 0.54],
    info: "Eye Separation Ratio: Pupillenabstand geteilt durch Jochbeinbreite. Ideal liegt bei ca. 0,46.",
    tipLow: "Augenbrauen mit etwas mehr Abstand zur Nasenwurzel lassen eng stehende Augen weiter wirken.",
    tipHigh: "Augenbrauen, die näher an der Nasenwurzel beginnen, balancieren weit stehende Augen.",
    kp: [468, 473, 234, 454],
    viz: (P) => [seg(P[468], P[473], C.cyan, "Pupillen"), seg(P[234], P[454], C.pink, "Jochbein")] },
  { id: "fifths", name: "Augen-Fünftel", cat: "eyes", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.9, 1.1], f: [0.9, 1.1] }, tol: 0.15, range: [0.6, 1.5],
    info: "Abstand der inneren Augenwinkel geteilt durch Augenbreite. Nach der Fünftel-Regel ideal ≈ 1,0.",
    tipLow: "Eng stehende Augen: Augenbrauen innen leicht ausdünnen.",
    tipHigh: "Weit stehende Augen: Brauen innen etwas voller tragen.",
    kp: [33, 133, 362, 263],
    viz: (P) => { const y1 = Math.min(P[105].y, P[334].y), y2 = P[2].y, out = [];
      for (const i of [33, 133, 362, 263]) out.push(seg({ x: P[i].x, y: y1 }, { x: P[i].x, y: y2 }, C.white, null, [6, 6]));
      return [...out, seg(P[33], P[133], C.cyan, "Auge"), seg(P[133], P[362], C.yellow, "Abstand"), seg(P[362], P[263], C.cyan, "Auge")]; } },
  { id: "eyeOpen", name: "Augenöffnung", cat: "eyes", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [0.26, 0.34], f: [0.3, 0.4] }, tol: 0.06, range: [0.12, 0.55],
    info: "Lidspalten-Höhe zu Augenbreite. Etwas geringere Werte bei Männern („hunter eyes“), höhere bei Frauen.",
    tipLow: "Ausreichend Schlaf, Kühlung und weniger Salz reduzieren Schwellungen der Lider.",
    tipHigh: "Ein entspannter, leicht fokussierter Blick wirkt auf Fotos markanter als weit aufgerissene Augen.",
    kp: [159, 145, 386, 374, 33, 133, 362, 263],
    viz: (P) => [seg(P[33], P[133], C.cyan), seg(P[362], P[263], C.cyan), seg(P[159], P[145], C.yellow, "Höhe"), seg(P[386], P[374], C.yellow, "Höhe")] },
  { id: "midface", name: "Mittelgesichts-Ratio", cat: "midface", unit: "", dec: 2, weight: 1.25,
    ideal: { m: [0.95, 1.05], f: [0.97, 1.08] }, tol: 0.1, range: [0.7, 1.3],
    info: "Pupillenabstand geteilt durch Höhe Pupillenlinie→Oberlippe. Ein kompaktes Mittelgesicht (≈ 1,0) wirkt jugendlich.",
    tipLow: "Ein längeres Mittelgesicht kann durch einen gepflegten Schnurrbart oder Pony optisch verkürzt werden.",
    kp: [468, 473, 0],
    viz: (P) => { const py = (P[468].y + P[473].y) / 2;
      return [hl(py, P[468].x - 30, P[473].x + 30), hl(P[0].y, P[468].x - 30, P[473].x + 30), seg(P[468], P[473], C.cyan, "Pupillenabstand"), vseg(P[473].x + 14, py, P[0].y, C.pink, "Höhe")]; } },
  { id: "noseWidth", name: "Nasenbreite", cat: "midface", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.95, 1.15], f: [0.9, 1.05] }, tol: 0.15, range: [0.6, 1.6],
    info: "Nasenflügelbreite geteilt durch den Abstand der inneren Augenwinkel. Ideal ≈ 1,0.",
    tipHigh: "Konturing an den Nasenflanken oder ein Bart, der das Untergesicht betont, lenkt den Fokus.",
    tipLow: "Eine schmale Nase ist ästhetisch meist unproblematisch.",
    kp: [64, 294, 133, 362],
    viz: (P) => [seg(P[133], { x: P[133].x, y: P[64].y }, C.white, null, [6, 6]), seg(P[362], { x: P[362].x, y: P[294].y }, C.white, null, [6, 6]),
      seg(P[133], P[362], C.yellow, "Augenabstand"), seg(P[64], P[294], C.cyan, "Nase")] },
  { id: "mouthNose", name: "Mund- zu Nasenbreite", cat: "lower", unit: "", dec: 2, weight: 1,
    ideal: { m: [1.5, 1.65], f: [1.5, 1.65] }, tol: 0.15, range: [1.0, 2.2],
    info: "Mundbreite geteilt durch Nasenbreite. Häufig zitiertes Ideal: ≈ 1,6 (Goldener Schnitt).",
    tipLow: "Lippenpflege und ein natürliches Lächeln lassen den Mund breiter wirken.",
    kp: [61, 291, 64, 294],
    viz: (P) => [seg(P[64], { x: P[64].x, y: P[61].y }, C.white, null, [6, 6]), seg(P[294], { x: P[294].x, y: P[291].y }, C.white, null, [6, 6]),
      seg(P[64], P[294], C.cyan, "Nase"), seg(P[61], P[291], C.pink, "Mund")] },
  { id: "lipRatio", name: "Lippenverhältnis", cat: "lower", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [1.4, 2.0], f: [1.3, 1.8] }, tol: 0.35, range: [0.6, 3.0],
    info: "Unterlippen- zu Oberlippenhöhe. Ideal ist eine Unterlippe, die ca. 1,6× so voll ist wie die Oberlippe.",
    tipLow: "Feuchtigkeitspflege und Peeling lassen die Lippen voller wirken.",
    tipHigh: "Lippenkontur an der Oberlippe betonen gleicht das Verhältnis optisch aus.",
    kp: [0, 13, 14, 17],
    viz: (P) => { const x = P[291].x + 18;
      return [hl(P[0].y, P[61].x, x), hl(P[13].y, P[61].x, x), hl(P[17].y, P[61].x, x), vseg(x, P[0].y, P[13].y, C.cyan, "Ober"), vseg(x, P[14].y, P[17].y, C.pink, "Unter")]; } },
  { id: "chinPhiltrum", name: "Kinn zu Philtrum", cat: "lower", unit: "", dec: 2, weight: 1,
    ideal: { m: [2.0, 2.5], f: [1.8, 2.25] }, tol: 0.35, range: [1.0, 3.6],
    info: "Kinnhöhe (Unterlippe→Kinnspitze) geteilt durch Philtrumlänge (Nase→Oberlippe).",
    tipLow: "Ein kurzes Kinn wirkt mit Bart am Kinn oder korrekter Zungenhaltung prominenter.",
    tipHigh: "Ein langes Kinn lässt sich durch Bart an den Seiten (mehr Breite) ausgleichen.",
    kp: [2, 0, 17, 152],
    viz: (P) => { const x = P[291].x + 18;
      return [hl(P[2].y, P[61].x, x), hl(P[0].y, P[61].x, x), hl(P[17].y, P[61].x, x), hl(P[152].y, P[61].x, x), vseg(x, P[2].y, P[0].y, C.cyan, "Philtrum"), vseg(x, P[17].y, P[152].y, C.pink, "Kinn")]; } },
  { id: "jaw", name: "Kieferbreite", cat: "lower", unit: "", dec: 2, weight: 1.25,
    ideal: { m: [0.88, 0.96], f: [0.8, 0.88] }, tol: 0.06, range: [0.65, 1.05],
    info: "Kieferwinkel-Breite geteilt durch Jochbeinbreite. Breitere Kiefer wirken maskuliner, schmalere femininer.",
    tipLow: "Weniger Körperfett, Kaumuskeltraining und ein seitlich voller Bart betonen die Kieferlinie.",
    tipHigh: "Ein breiter Kiefer ist markant – Frisuren mit Höhe balancieren die Proportionen.",
    kp: [172, 397, 234, 454],
    viz: (P) => [seg(P[234], P[454], C.violet, "Jochbein"), seg(P[172], P[397], C.pink, "Kiefer")] },
  { id: "faceIndex", name: "Gesichtsindex", cat: "harmony", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.85, 0.93], f: [0.83, 0.91] }, tol: 0.06, range: [0.7, 1.1],
    info: "Gesichtshöhe (Nasenwurzel→Kinn) geteilt durch Jochbeinbreite. Zeigt, ob ein Gesicht eher lang oder breit gebaut ist.",
    tipLow: "Ein breites Gesicht wirkt mit Volumen am Oberkopf und kürzeren Seiten länger.",
    tipHigh: "Ein langes Gesicht wirkt mit Pony oder etwas seitlichem Volumen ausgeglichener.",
    kp: [168, 152, 234, 454],
    viz: (P) => [hl(P[168].y, P[234].x, P[454].x), hl(P[152].y, P[234].x, P[454].x), seg(P[234], P[454], C.cyan, "Breite"), vseg(P[168].x, P[168].y, P[152].y, C.pink, "Höhe")] },
  { id: "noseLength", name: "Nasenlänge", cat: "midface", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.43, 0.47], f: [0.42, 0.46] }, tol: 0.03, range: [0.35, 0.55],
    info: "Nasenwurzel→Nasenansatz im Verhältnis zur Gesichtshöhe (Nasenwurzel→Kinn).",
    tipHigh: "Eine längere Nase wirkt mit kräftigerem Kinn und gepflegtem Bart ausgewogener.",
    tipLow: "Eine kurze Nase wirkt jugendlich – meist kein Handlungsbedarf.",
    kp: [168, 2, 152],
    viz: (P) => { const x = P[454].x - 10, x1 = P[234].x + 20; return [hl(P[168].y, x1, x), hl(P[2].y, x1, x), hl(P[152].y, x1, x), vseg(x, P[168].y, P[2].y, C.cyan, "Nase"), vseg(x - 34, P[168].y, P[152].y, C.pink, "Gesicht")]; } },
  { id: "eyeWidth", name: "Augenbreite", cat: "eyes", unit: "", dec: 3, weight: 0.75,
    ideal: { m: [0.21, 0.24], f: [0.215, 0.245] }, tol: 0.015, range: [0.16, 0.29],
    info: "Breite eines Auges im Verhältnis zur Gesichtsbreite. Nach der Fünftel-Regel ist ein Auge ≈ 1/5 der Gesichtsbreite.",
    tipLow: "Kleinere Augen wirken mit gepflegten, nicht zu dichten Brauen und ausgeruhtem Blick größer.",
    tipHigh: "Große Augen sind ein attraktives Merkmal – kein Handlungsbedarf.",
    kp: [33, 133, 362, 263, 234, 454],
    viz: (P) => [seg(P[234], P[454], C.violet, "Gesichtsbreite"), seg(P[33], P[133], C.cyan, "Auge"), seg(P[362], P[263], C.cyan, "Auge")] },
  { id: "browTilt", name: "Brauenneigung", cat: "eyes", unit: "°", dec: 1, weight: 0.75,
    ideal: { m: [0, 8], f: [6, 16] }, tol: 4, range: [-15, 25],
    info: "Anstieg der Augenbraue vom Brauenkopf zum Brauenende. Flachere Brauen wirken maskuliner, ansteigende femininer.",
    tipLow: "Abfallende Brauen wirken müde – das Brauenende leicht anheben (zupfen/formen) öffnet den Blick.",
    tipHigh: "Stark ansteigende Brauen wirken überrascht – ein flacherer Bogen wirkt ruhiger.",
    kp: [55, 46, 285, 276],
    viz: (P, r) => { const out = [];
      for (const [i, o, v] of [[55, 46, r._btR], [285, 276, r._btL]]) {
        const h = { x: P[o].x + (P[o].x - P[i].x) * 0.3, y: P[i].y };
        out.push(seg(P[i], h, C.white, null, [6, 6]), seg(P[i], ext(P[i], P[o], 1.3), C.green), arc(h, P[i], P[o], C.yellow, `${fmt(v, 1)}°`));
      } return out; } },
  { id: "browEye", name: "Brauen-Lid-Abstand", cat: "eyes", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [0.28, 0.42], f: [0.38, 0.55] }, tol: 0.08, range: [0.1, 0.8],
    info: "Abstand Braue→Oberlid im Verhältnis zur Augenbreite. Tief sitzende Brauen geben einen intensiven Blick.",
    tipHigh: "Hoch sitzende Brauen: Brauen unten nicht zu stark zupfen, damit sie tiefer wirken.",
    tipLow: "Sehr tief sitzende Brauen: Unterkante leicht formen, damit das Lid mehr Raum bekommt.",
    kp: [52, 159, 282, 386],
    viz: (P) => [vseg(P[159].x, P[52].y, P[159].y, C.pink, "Abstand"), vseg(P[386].x, P[282].y, P[386].y, C.pink, "Abstand"), seg(P[33], P[133], C.cyan), seg(P[362], P[263], C.cyan)] },
  { id: "lipFull", name: "Lippenfülle", cat: "lower", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [0.28, 0.38], f: [0.33, 0.45] }, tol: 0.06, range: [0.15, 0.6],
    info: "Gesamthöhe der Lippen geteilt durch Mundbreite.",
    tipLow: "Schmale Lippen wirken mit Pflege (Peeling, Feuchtigkeit) und entspannter Lippenhaltung voller.",
    tipHigh: "Volle Lippen sind ein attraktives Merkmal – kein Handlungsbedarf.",
    kp: [0, 17, 61, 291],
    viz: (P) => [seg(P[61], P[291], C.cyan, "Breite"), vseg(P[291].x + 18, P[0].y, P[17].y, C.pink, "Höhe")] },
  { id: "mouthIpd", name: "Mund zu Pupillen", cat: "lower", unit: "", dec: 2, weight: 0.75,
    ideal: { m: [0.8, 0.95], f: [0.8, 0.95] }, tol: 0.08, range: [0.6, 1.2],
    info: "Mundbreite geteilt durch den Pupillenabstand.",
    tipLow: "Ein schmaler Mund wirkt mit natürlichem Lächeln breiter.",
    tipHigh: "Ein breiter Mund wirkt ausdrucksstark – kein Handlungsbedarf.",
    kp: [61, 291, 468, 473],
    viz: (P) => [seg(P[468], { x: P[468].x, y: P[61].y }, C.white, null, [6, 6]), seg(P[473], { x: P[473].x, y: P[291].y }, C.white, null, [6, 6]), seg(P[468], P[473], C.cyan, "Pupillen"), seg(P[61], P[291], C.pink, "Mund")] },
];

// Gesichtsform aus den Proportionen
const SHAPES = {
  oval: ["Oval", "Ausgewogene Länge und Breite, sanft gerundeter Kiefer.", "Fast jede Frisur passt – du kannst frei experimentieren."],
  round: ["Rund", "Breite und Höhe ähnlich, weiche Konturen.", "Volumen am Oberkopf und kürzere Seiten strecken das Gesicht."],
  square: ["Eckig", "Breiter, kantiger Kiefer bei kompakter Gesichtshöhe.", "Markante Kieferlinie betonen – kurze Seiten, Struktur oben."],
  rect: ["Rechteckig", "Kantiger Kiefer bei eher langem Gesicht.", "Seitliches Volumen und kein zu hoher Oberkopf gleichen die Länge aus."],
  long: ["Länglich", "Gesichtshöhe deutlich größer als die Breite.", "Pony oder Volumen an den Seiten verkürzen optisch."],
  heart: ["Herzförmig", "Breite Stirn, schmaler zulaufendes Kinn.", "Volumen auf Kinnhöhe (z. B. Bart, längere Haare) balanciert die Stirn."],
  diamond: ["Raute", "Breite Wangenknochen, schmale Stirn und schmales Kinn.", "Volumen an Stirn und Kinn betont die Wangenknochen harmonisch."],
};
function faceShape(r) {
  const fi = r.faceIndex, jw = r.jaw, fw = r._foreheadW;
  if (fi > 0.98) return jw >= 0.9 ? SHAPES.rect : SHAPES.long;
  if (jw >= 0.92) return fi >= 0.9 ? SHAPES.rect : SHAPES.square;
  if (fw > 0.9 && jw < 0.82) return SHAPES.heart;
  if (fw < 0.84 && jw < 0.84) return SHAPES.diamond;
  if (fi < 0.84) return SHAPES.round;
  return SHAPES.oval;
}
function profileType(r) {
  if (r.convexity < 163) return ["Konvexes Profil", "Kinn liegt hinter der Gesichtsebene – typisch für ein fliehendes Kinn oder eine prominente Mittelgesichtspartie.", "Kinnbart und aufrechte Kopfhaltung lassen das Profil gerader wirken."];
  if (r.convexity > 177) return ["Konkaves Profil", "Kinn steht vor der Gesichtsebene – kräftige Kinn- und Unterkieferpartie.", "Ein Bart an den Seiten statt am Kinn mildert die Kinnprojektion."];
  return ["Gerades Profil", "Stirn, Nasenansatz und Kinn liegen nahezu auf einer Linie – gilt als harmonisch.", "Klare Konturen beibehalten – eine definierte Kieferlinie unterstreicht das Profil."];
}

// ======================= SEITENPROFIL =======================
const SIDE_KP = [
  { id: "G", label: "Glabella", hint: "Vorderster Punkt der Stirn zwischen den Augenbrauen", mp: 9 },
  { id: "N", label: "Nasion", hint: "Tiefster Punkt der Nasenwurzel zwischen Stirn und Nase", mp: 168 },
  { id: "Prn", label: "Pronasale", hint: "Nasenspitze – vorderster Punkt der Nase", mp: 1 },
  { id: "Sn", label: "Subnasale", hint: "Übergang vom Nasensteg zur Oberlippe", mp: 2 },
  { id: "Ac", label: "Nasenflügelansatz", hint: "Hinterster Punkt der Nasenflügelfalte, wo der Nasenflügel in die Wange übergeht", mp: [129, 358] },
  { id: "Ls", label: "Labrale superius", hint: "Vorderster Punkt der Oberlippe", mp: 0 },
  { id: "St", label: "Stomion", hint: "Treffpunkt der Lippen", mp: 13 },
  { id: "Li", label: "Labrale inferius", hint: "Vorderster Punkt der Unterlippe", mp: 17 },
  { id: "Sm", label: "Supramentale", hint: "Tiefster Punkt der Falte zwischen Unterlippe und Kinn", mp: 18 },
  { id: "Pog", label: "Pogonion", hint: "Vorderster Punkt des Kinns", mp: 199 },
  { id: "Me", label: "Menton", hint: "Unterster Punkt des Kinns", mp: 152 },
  { id: "Go", label: "Gonion", hint: "Kieferwinkel – Ecke des Unterkiefers", mp: [172, 397] },
  { id: "Ar", label: "Ramus oben", hint: "Hinterkante des Kiefers direkt unter dem Ohrläppchen", mp: [93, 323] },
];
const PROFILE_LINE = ["G", "N", "Prn", "Sn", "Ls", "St", "Li", "Sm", "Pog", "Me"];

function computeSide(P) {
  const faceH = dist(P.N, P.Me), mm = (sex === "m" ? 123 : 112) / faceH;
  // vorzeichenbehafteter Abstand zur Linie a–b in mm (>0 = vor der Linie, weg vom Ohr)
  const fwd = (p, a, b) => -sideOf(P.Go, a, b) * sideOf(p, a, b) * dist(p, foot(p, a, b)) * mm;
  const eDist = (p) => fwd(p, P.Prn, P.Pog);
  return {
    goode: Math.abs(P.Prn.x - P.Ac.x) / dist(P.N, P.Prn),
    burstoneU: fwd(P.Ls, P.Sn, P.Pog), burstoneL: fwd(P.Li, P.Sn, P.Pog),
    mandPlane: deg(Math.atan2(Math.abs(P.Me.y - P.Go.y), Math.abs(P.Me.x - P.Go.x))),
    ramus: dist(P.Ar, P.Go) / dist(P.Go, P.Me),
    nasofrontal: angleAt(P.G, P.N, P.Prn),
    nasolabial: angleAt(P.Prn, P.Sn, P.Ls),
    nasofacial: lineAngle(P.G, P.Pog, P.N, P.Prn),
    nasomental: angleAt(P.N, P.Prn, P.Pog),
    mentolabial: angleAt(P.Li, P.Sm, P.Pog),
    convexity: angleAt(P.G, P.Sn, P.Pog),
    eUpper: eDist(P.Ls), eLower: eDist(P.Li),
    gonial: angleAt(P.Ar, P.Go, P.Me),
    lowerThird: (P.St.y - P.Sn.y) / Math.max(P.Me.y - P.St.y, 0.5),
  };
}

const SIDE_CATS = { nose: "Nase", lips: "Lippen & Kinn", jaw: "Kiefer & Harmonie" };
const SIDE_METRICS = [
  { id: "nasofrontal", name: "Nasofrontaler Winkel", cat: "nose", unit: "°", dec: 0, weight: 1,
    ideal: { m: [115, 130], f: [120, 135] }, tol: 8, range: [90, 160], kp: ["G", "N", "Prn"],
    info: "Winkel zwischen Stirn und Nasenrücken an der Nasenwurzel.",
    tipLow: "Ein tiefer Nasenansatz wirkt mit kräftigeren Augenbrauen harmonischer.", tipHigh: "Ein flacher Übergang Stirn–Nase ist typisch für ein gerades, klassisches Profil.",
    viz: (P, r) => [arc(P.G, P.N, P.Prn, C.yellow, `${fmt(r.nasofrontal, 0)}°`)] },
  { id: "nasolabial", name: "Nasolabialwinkel", cat: "nose", unit: "°", dec: 0, weight: 1.25,
    ideal: { m: [90, 105], f: [95, 115] }, tol: 8, range: [70, 140], kp: ["Prn", "Sn", "Ls"],
    info: "Winkel zwischen Nasensteg und Oberlippe. Beschreibt, wie stark die Nasenspitze nach oben zeigt.",
    tipLow: "Eine hängende Nasenspitze wirkt beim Lächeln oft stärker – ein leicht angehobenes Kinn beim Foto gleicht aus.", tipHigh: "Eine stark nach oben zeigende Nasenspitze wirkt femininer.",
    viz: (P, r) => [arc(P.Prn, P.Sn, P.Ls, C.yellow, `${fmt(r.nasolabial, 0)}°`)] },
  { id: "nasofacial", name: "Nasofazialer Winkel", cat: "nose", unit: "°", dec: 0, weight: 1,
    ideal: { m: [30, 36], f: [32, 38] }, tol: 4, range: [20, 50], kp: ["G", "Pog", "N", "Prn"],
    info: "Winkel zwischen Gesichtsebene (Glabella–Kinn) und Nasenrücken – ein Maß für die Nasenprojektion.",
    tipLow: "Eine wenig projizierte Nase wirkt im Profil mit kürzerer Frisur und klarer Kinnlinie ausgewogener.", tipHigh: "Eine stark projizierte Nase wird durch ein kräftiges Kinn und Bart optisch ausgeglichen.",
    viz: (P, r) => [seg(ext(P.Pog, P.G, 1.08), ext(P.G, P.Pog, 1.08), C.white, "Gesichtsebene", [8, 6]), seg(P.N, ext(P.N, P.Prn, 1.15), C.cyan), tag(mid(P.N, P.Prn), `${fmt(r.nasofacial, 0)}°`, C.yellow)] },
  { id: "nasomental", name: "Nasomentaler Winkel", cat: "nose", unit: "°", dec: 0, weight: 1,
    ideal: { m: [120, 132], f: [120, 132] }, tol: 5, range: [105, 145], kp: ["N", "Prn", "Pog"],
    info: "Winkel an der Nasenspitze zwischen Nasenrücken und Linie zum Kinn – zeigt das Verhältnis von Nase zu Kinn.",
    tipLow: "Nase dominiert gegenüber dem Kinn: Kinnbart oder Kinnbetonung bringt Balance.", tipHigh: "Kinn dominiert: meist ein Zeichen für ein starkes Kinn.",
    viz: (P, r) => [arc(P.N, P.Prn, P.Pog, C.yellow, `${fmt(r.nasomental, 0)}°`)] },
  { id: "mentolabial", name: "Mentolabialwinkel", cat: "lips", unit: "°", dec: 0, weight: 1,
    ideal: { m: [110, 130], f: [115, 135] }, tol: 10, range: [80, 160], kp: ["Li", "Sm", "Pog"],
    info: "Winkel der Falte zwischen Unterlippe und Kinn. Zu spitz wirkt das Kinn eingedrückt, zu flach wirkt es fliehend.",
    tipLow: "Eine tiefe Kinnfalte entspannt sich oft bei lockerer Lippenhaltung.", tipHigh: "Ein flacher Übergang deutet auf ein fliehendes Kinn hin – ein Kinnbart schafft Kontur.",
    viz: (P, r) => [arc(P.Li, P.Sm, P.Pog, C.yellow, `${fmt(r.mentolabial, 0)}°`)] },
  { id: "eUpper", name: "E-Linie Oberlippe", cat: "lips", unit: " mm", dec: 1, weight: 1,
    ideal: { m: [-6, -3], f: [-5, -2] }, tol: 2, range: [-12, 6], kp: ["Prn", "Pog", "Ls"],
    info: "Abstand der Oberlippe zur Ricketts-E-Linie (Nasenspitze–Kinn). Negativ = hinter der Linie. Geschätzt in mm.",
    tipLow: "Weit zurückliegende Lippen wirken durch Lippenpflege und entspannte Lippenhaltung voller.", tipHigh: "Vorstehende Lippen werden durch ein kräftiges Kinn im Profil ausgeglichen.",
    viz: (P, r) => [seg(ext(P.Pog, P.Prn, 1.15), ext(P.Prn, P.Pog, 1.15), C.cyan, "E-Linie"), seg(P.Ls, foot(P.Ls, P.Prn, P.Pog), C.pink, `${fmt(r.eUpper, 1)} mm`)] },
  { id: "eLower", name: "E-Linie Unterlippe", cat: "lips", unit: " mm", dec: 1, weight: 1,
    ideal: { m: [-4, -1], f: [-3, 0] }, tol: 2, range: [-10, 8], kp: ["Prn", "Pog", "Li"],
    info: "Abstand der Unterlippe zur E-Linie. Ideal liegt sie knapp dahinter.",
    tipLow: "Zurückliegende Unterlippe: Kinnhaltung prüfen – oft hilft ein leicht nach vorn genommenes Kinn.", tipHigh: "Vorstehende Unterlippe wirkt mit mehr Kinnprojektion harmonischer.",
    viz: (P, r) => [seg(ext(P.Pog, P.Prn, 1.15), ext(P.Prn, P.Pog, 1.15), C.cyan, "E-Linie"), seg(P.Li, foot(P.Li, P.Prn, P.Pog), C.pink, `${fmt(r.eLower, 1)} mm`)] },
  { id: "convexity", name: "Profilkonvexität", cat: "jaw", unit: "°", dec: 0, weight: 1.25,
    ideal: { m: [165, 175], f: [163, 173] }, tol: 4, range: [145, 180], kp: ["G", "Sn", "Pog"],
    info: "Winkel Glabella–Subnasale–Kinn. Werte nahe 170° stehen für ein gerades Profil, kleinere für ein konvexes (fliehendes Kinn).",
    tipLow: "Ein konvexes Profil wirkt mit Kinnbart, guter Haltung (Kopf aufrecht, nicht nach vorn geschoben) und niedrigerem Körperfett gerader.", tipHigh: "Ein sehr gerades bis konkaves Profil wirkt markant.",
    viz: (P, r) => [arc(P.G, P.Sn, P.Pog, C.yellow, `${fmt(r.convexity, 0)}°`)] },
  { id: "gonial", name: "Kieferwinkel (Gonial)", cat: "jaw", unit: "°", dec: 0, weight: 1.25,
    ideal: { m: [115, 125], f: [120, 130] }, tol: 6, range: [100, 145], kp: ["Ar", "Go", "Me"],
    info: "Winkel zwischen hinterem Kieferast und Unterkieferrand. Kleinere Winkel stehen für eine kantige Kieferlinie.",
    tipHigh: "Ein steiler Kieferwinkel wird durch niedrigeren Körperfettanteil, Kaumuskeltraining und Bart entlang der Kieferlinie definierter.", tipLow: "Ein sehr kantiger Kieferwinkel ist ein markantes Merkmal.",
    viz: (P, r) => [arc(P.Ar, P.Go, P.Me, C.yellow, `${fmt(r.gonial, 0)}°`)] },
  { id: "lowerThird", name: "Lippe zu Kinn (Profil)", cat: "jaw", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.45, 0.55], f: [0.45, 0.55] }, tol: 0.06, range: [0.3, 0.8], kp: ["Sn", "St", "Me"],
    info: "Oberlippenhöhe (Subnasale→Stomion) geteilt durch Unterlippe+Kinn (Stomion→Menton). Ideal ≈ 1:2.",
    tipLow: "Eine kurze Oberlippe ist meist unproblematisch.", tipHigh: "Ein kurzes Kinn wirkt mit Kinnbart oder mehr Kinnprojektion länger.",
    viz: (P) => { const faceRight = P.Prn.x > P.Go.x;
      const X = faceRight ? Math.max(P.Prn.x, P.Pog.x, P.Ls.x) + 30 : Math.min(P.Prn.x, P.Pog.x, P.Ls.x) - 30, X2 = X + (faceRight ? 26 : -26), x0 = faceRight ? Math.min(P.Sn.x, P.Me.x) - 10 : Math.max(P.Sn.x, P.Me.x) + 10;
      return [hl(P.Sn.y, x0, X), hl(P.St.y, x0, X2), hl(P.Me.y, x0, X2), vseg(X, P.Sn.y, P.St.y, C.cyan, "Oberlippe"), vseg(X2, P.St.y, P.Me.y, C.pink, "Unterlippe+Kinn")]; } },
  { id: "goode", name: "Nasenprojektion (Goode)", cat: "nose", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.55, 0.6], f: [0.55, 0.6] }, tol: 0.05, range: [0.4, 0.75], kp: ["Ac", "Prn", "N"],
    info: "Abstand Nasenflügelansatz→Nasenspitze geteilt durch Nasenlänge (Nasenwurzel→Spitze). Ideal nach Goode: 0,55–0,60.",
    tipLow: "Eine wenig projizierte Nase wirkt im Profil mit klarer Kinnlinie ausgewogener.", tipHigh: "Eine stark projizierte Nase wird durch ein kräftiges Kinn und Bart ausgeglichen.",
    viz: (P) => [seg(P.N, P.Prn, C.cyan, "Nasenlänge"), seg(P.Ac, { x: P.Prn.x, y: P.Ac.y }, C.pink, "Projektion")] },
  { id: "burstoneU", name: "Burstone-Linie Oberlippe", cat: "lips", unit: " mm", dec: 1, weight: 0.75,
    ideal: { m: [2, 5], f: [2.5, 5.5] }, tol: 1.5, range: [-4, 10], kp: ["Sn", "Pog", "Ls"],
    info: "Abstand der Oberlippe vor der Linie Subnasale–Kinn (Burstone). Positiv = vor der Linie. Geschätzt in mm.",
    tipLow: "Zurückliegende Oberlippe wirkt mit Lippenpflege und entspannter Lippenhaltung voller.", tipHigh: "Stark vorstehende Lippen werden durch mehr Kinnprojektion ausgeglichen.",
    viz: (P, r) => [seg(ext(P.Pog, P.Sn, 1.12), ext(P.Sn, P.Pog, 1.12), C.cyan, "Sn–Pog"), seg(P.Ls, foot(P.Ls, P.Sn, P.Pog), C.pink, `${fmt(r.burstoneU, 1)} mm`)] },
  { id: "burstoneL", name: "Burstone-Linie Unterlippe", cat: "lips", unit: " mm", dec: 1, weight: 0.75,
    ideal: { m: [1, 4], f: [1.5, 4.5] }, tol: 1.5, range: [-5, 9], kp: ["Sn", "Pog", "Li"],
    info: "Abstand der Unterlippe vor der Linie Subnasale–Kinn. Positiv = vor der Linie.",
    tipLow: "Zurückliegende Unterlippe: oft ein Hinweis auf ein kräftiges Kinn.", tipHigh: "Vorstehende Unterlippe wirkt mit mehr Kinnprojektion harmonischer.",
    viz: (P, r) => [seg(ext(P.Pog, P.Sn, 1.12), ext(P.Sn, P.Pog, 1.12), C.cyan, "Sn–Pog"), seg(P.Li, foot(P.Li, P.Sn, P.Pog), C.pink, `${fmt(r.burstoneL, 1)} mm`)] },
  { id: "mandPlane", name: "Unterkieferneigung", cat: "jaw", unit: "°", dec: 0, weight: 1,
    ideal: { m: [18, 27], f: [20, 29] }, tol: 4, range: [5, 45], kp: ["Go", "Me"],
    info: "Neigung des Unterkieferrands (Kieferwinkel→Kinn) gegenüber der Horizontalen. Flachere Werte stehen für eine kräftige, horizontale Kieferlinie.",
    tipHigh: "Eine steile Kieferlinie wirkt mit niedrigerem Körperfett, guter Kopfhaltung (Kinn nicht nach vorn schieben) und Bart entlang des Kiefers definierter.", tipLow: "Eine sehr flache Kieferlinie ist markant.",
    viz: (P, r) => { const h = { x: P.Go.x, y: P.Me.y }; return [seg(P.Me, h, C.white, null, [6, 6]), arc(h, P.Me, P.Go, C.yellow, `${fmt(r.mandPlane, 0)}°`)]; } },
  { id: "ramus", name: "Kieferast zu Kieferkörper", cat: "jaw", unit: "", dec: 2, weight: 1,
    ideal: { m: [0.65, 0.8], f: [0.6, 0.75] }, tol: 0.08, range: [0.4, 1.1], kp: ["Ar", "Go", "Me"],
    info: "Länge des aufsteigenden Kieferasts (unter dem Ohr→Kieferwinkel) im Verhältnis zum Unterkieferkörper (Kieferwinkel→Kinn).",
    tipLow: "Ein kurzer Kieferast wirkt mit Bart entlang der hinteren Kieferlinie markanter.", tipHigh: "Ein langer Kieferast ist ein markantes Merkmal.",
    viz: (P) => [seg(P.Ar, P.Go, C.cyan, "Ast"), seg(P.Go, P.Me, C.pink, "Körper")] },
];

const MODES = {
  front: { metrics: FRONT_METRICS, cats: FRONT_CATS, kp: () => FRONT_KP,
    hint: "Frontalfoto: gerade in die Kamera schauen, neutraler Ausdruck, Haare aus dem Gesicht. Das Gesicht wird automatisch erkannt, gedreht und zugeschnitten.",
    dz: "JPG, PNG oder WebP · Frontalaufnahme · ein Gesicht" },
  side: { metrics: SIDE_METRICS, cats: SIDE_CATS, kp: () => SIDE_KP,
    hint: "Seitenfoto: Kopf exakt 90° zur Seite, Blick geradeaus, Ohr und Kinnlinie sichtbar. Punkte werden automatisch gesetzt (wenn möglich) – sonst klickst du sie nacheinander an.",
    dz: "JPG, PNG oder WebP · Seitenansicht (Profil) · Ohr und Kinn sichtbar" },
};

// ======================= State =======================
let landmarker = null;
let sex = "m";
let mode = "front";
const sessions = { front: null, side: null };
let active = null, pinned = null, hoverKp = null, drag = null, stream = null;
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const cur = () => sessions[mode];

// ======================= Modell =======================
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
  const opts = (delegate) => ({ baseOptions: { modelAssetBuffer: model, delegate }, runningMode: "IMAGE", numFaces: 1, minFaceDetectionConfidence: 0.3, minFacePresenceConfidence: 0.3 });
  try { landmarker = await MP.FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { landmarker = await MP.FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
  return landmarker;
}

function rotateCanvas(img, d) {
  const c = document.createElement("canvas"), t = (d * Math.PI) / 180;
  const co = Math.abs(Math.round(Math.cos(t) * 1e6) / 1e6), si = Math.abs(Math.round(Math.sin(t) * 1e6) / 1e6);
  c.width = Math.round(img.width * co + img.height * si); c.height = Math.round(img.width * si + img.height * co);
  const x = c.getContext("2d");
  x.fillStyle = "#07070c"; x.fillRect(0, 0, c.width, c.height);
  x.translate(c.width / 2, c.height / 2); x.rotate(t); x.drawImage(img, -img.width / 2, -img.height / 2);
  return c;
}

// Probiert 0°, 90°, 270°, 180°; ist das Gesicht danach noch deutlich geneigt, wird das Bild
// exakt aufrecht gedreht und erneut erkannt – das Modell arbeitet auf aufrechten Gesichtern am genauesten.
async function detectFace(img) {
  const lm = await getLandmarker();
  for (const d of [0, 90, 270, 180]) {
    const c = d ? rotateCanvas(img, d) : img;
    const r = lm.detect(c);
    if (!r.faceLandmarks?.length) continue;
    let res = { img: c, lms: r.faceLandmarks[0], rotated: d };
    const P = (i) => ({ x: res.lms[i].x * c.width, y: res.lms[i].y * c.height });
    const roll = deg(Math.atan2(P(263).y - P(33).y, P(263).x - P(33).x));
    if (Math.abs(roll) > 12 && Math.abs(roll) < 168) {
      const c2 = rotateCanvas(c, -roll), r2 = lm.detect(c2);
      if (r2.faceLandmarks?.length) res = { img: c2, lms: r2.faceLandmarks[0], rotated: Math.round((d - roll + 360) % 360) };
    }
    return res;
  }
  return null;
}

// ======================= Koordinaten =======================
// Bildpixel p --(Ausrichtung s.roll um s.center)--> ausgerichtet a --(Ansicht)--> Canvas v
const align = (s, p) => rot(p, s.center, -s.roll);
const unalign = (s, a) => rot(a, s.center, s.roll);
const toView = (s, p) => { const v = s.view, q = rot(p, { x: v.cx, y: v.cy }, -v.roll); return { x: (q.x - v.ox) * v.k, y: (q.y - v.oy) * v.k }; };
const fromView = (s, q) => { const v = s.view; return rot({ x: q.x / v.k + v.ox, y: q.y / v.k + v.oy }, { x: v.cx, y: v.cy }, v.roll); };
const A2V = (s, a) => toView(s, unalign(s, a));

function alignedPts(s) {
  if (s.mode === "front") return s.pts.map((p) => align(s, p));
  const o = {}; for (const k of SIDE_KP) if (s.pts[k.id]) o[k.id] = align(s, s.pts[k.id]); return o;
}

function viewForBox(x0, y0, x1, y1, roll, c) {
  let bw = x1 - x0, bh = y1 - y0;
  const ar = W / H;
  if (bw / bh > ar) { const nh = bw / ar; y0 -= (nh - bh) / 2; bh = nh; } else { const nw = bh * ar; x0 -= (nw - bw) / 2; bw = nw; }
  return { k: W / bw, ox: x0, oy: y0, roll, cx: c.x, cy: c.y };
}
const bbox = (pts) => ({ x0: Math.min(...pts.map((p) => p.x)), x1: Math.max(...pts.map((p) => p.x)), y0: Math.min(...pts.map((p) => p.y)), y1: Math.max(...pts.map((p) => p.y)) });
const useAlign = (s) => $("tglAlign").checked && s.ready;

function baseView(s) {
  if (!useAlign(s)) return viewForBox(0, 0, s.img.width, s.img.height, 0, { x: 0, y: 0 });
  let { x0, x1, y0, y1 } = bbox(Object.values(alignedPts(s)));
  const w = x1 - x0, h = y1 - y0;
  if (s.mode === "front") { x0 -= w * 0.22; x1 += w * 0.22; y0 -= h * 0.32; y1 += h * 0.1; }
  else { x0 -= w * 0.3; x1 += w * 0.3; y0 -= h * 0.35; y1 += h * 0.25; }
  return viewForBox(x0, y0, x1, y1, s.roll, s.center);
}

// Ausschnitt, der genau die Messung einer Metrik zeigt (für den Auto-Zoom)
function metricView(s, metric) {
  const base = baseView(s);
  if (!useAlign(s) || !$("tglZoom").checked) return base;
  const P = alignedPts(s), pts = metric.kp.map((id) => P[id]).filter(Boolean);
  for (const pr of metric.viz(P, s.raw)) for (const k of ["a", "b", "v", "p"]) if (pr[k]) pts.push(pr[k]);
  const face = bbox(Object.values(P)), fw = face.x1 - face.x0;
  let { x0, x1, y0, y1 } = bbox(pts);
  const pad = Math.max(x1 - x0, y1 - y0) * 0.3 + fw * 0.12;
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const minW = fw * 0.75, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (x1 - x0 < minW) { x0 = cx - minW / 2; x1 = cx + minW / 2; }
  if (y1 - y0 < minW * (H / W) * 0.7) { const mh = minW * (H / W) * 0.7; y0 = cy - mh / 2; y1 = cy + mh / 2; }
  const v = viewForBox(x0, y0, x1, y1, s.roll, s.center);
  return v.k < base.k * 1.12 ? base : v;
}

function fitView(s, animate = false) { const v = baseView(s); if (animate && s.view) animateView(s, v); else { s.view = v; if (anim.view?.s === s) anim.view = null; } }

// ---------- Animationen ----------
const anim = { view: null, drawT0: 0, scan: null };
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function lerpView(a, b, t) {
  const ca = { x: a.ox + W / (2 * a.k), y: a.oy + H / (2 * a.k) }, cb = { x: b.ox + W / (2 * b.k), y: b.oy + H / (2 * b.k) };
  const k = Math.exp(Math.log(a.k) + (Math.log(b.k) - Math.log(a.k)) * t), L = (u, v) => u + (v - u) * t;
  return { k, ox: L(ca.x, cb.x) - W / (2 * k), oy: L(ca.y, cb.y) - H / (2 * k), roll: L(a.roll, b.roll), cx: L(a.cx, b.cx), cy: L(a.cy, b.cy) };
}
function animateView(s, target) {
  if (reduceMotion) { s.view = target; draw(); return; }
  anim.view = { s, from: { ...s.view }, to: target, t0: performance.now(), dur: 520 };
  kick();
}
let looping = false;
function kick() { if (!looping) { looping = true; requestAnimationFrame(loop); } }
function loop(now) {
  let more = false;
  const a = anim.view;
  if (a) { const t = Math.min(1, (now - a.t0) / a.dur); a.s.view = lerpView(a.from, a.to, ease(t)); if (t >= 1) anim.view = null; else more = true; }
  if (now - anim.drawT0 < 420) more = true;
  if (anim.scan) { if (now - anim.scan.t0 < anim.scan.dur) more = true; else { const done = anim.scan.done; anim.scan = null; done?.(); } }
  draw(now);
  if (more) requestAnimationFrame(loop); else looping = false;
}

// ======================= Zeichnen =======================
// Schrift- und Punktgröße passen sich der angezeigten Canvas-Größe an (Handy = größer im Canvas)
let FS = 22, U = 1;
function updateScale() { const shown = canvas.clientWidth || 450; FS = Math.round(Math.min(44, Math.max(20, (12.5 * W) / shown))); U = FS / 22; }
function pill(x, y, text, color) {
  ctx.font = `600 ${FS}px "JetBrains Mono", ui-monospace, monospace`;
  const w = ctx.measureText(text).width + 16, h = FS + 12;
  x = Math.min(Math.max(x, w / 2 + 4), W - w / 2 - 4); y = Math.min(Math.max(y, h / 2 + 4), H - h / 2 - 4);
  ctx.fillStyle = "rgba(10,10,18,.88)"; ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(x - w / 2, y - h / 2, w, h, h / 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y + 1);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
}

function drawPrim(s, pr, t = 1) {
  ctx.setLineDash([]);
  ctx.globalAlpha = Math.min(1, t * 1.6);
  if (pr.t === "seg") {
    const a = A2V(s, pr.a), b0 = A2V(s, pr.b), b = { x: a.x + (b0.x - a.x) * t, y: a.y + (b0.y - a.y) * t };
    ctx.strokeStyle = pr.color; ctx.lineWidth = 3 * U; ctx.setLineDash((pr.dash || []).map((d) => d * U));
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.setLineDash([]);
    if (!pr.dash) for (const p of [a, b]) { ctx.fillStyle = pr.color; ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, 7); ctx.fill(); }
    if (pr.label && t > 0.55) { const m = mid(a, b0), len = dist(a, b) || 1, nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;
      pill(m.x + nx * 22 * U, m.y + ny * 22 * U, pr.label, pr.color); }
    ctx.globalAlpha = 1;
  } else if (pr.t === "arc") {
    const a = A2V(s, pr.a), v = A2V(s, pr.v), b = A2V(s, pr.b);
    ctx.strokeStyle = C.cyan; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(v.x, v.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    const t1 = Math.atan2(a.y - v.y, a.x - v.x); let d = Math.atan2(b.y - v.y, b.x - v.x) - t1;
    while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
    const dFull = d; d *= t;
    const r = Math.max(26, Math.min(60, dist(a, v) * 0.45, dist(b, v) * 0.45));
    ctx.fillStyle = "rgba(250,204,21,.18)"; ctx.strokeStyle = pr.color; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(v.x, v.y); ctx.arc(v.x, v.y, r, t1, t1 + d, d < 0); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.arc(v.x, v.y, r, t1, t1 + d, d < 0); ctx.stroke();
    const bis = t1 + dFull / 2;
    if (pr.label && t > 0.55) pill(v.x + Math.cos(bis) * (r + 34), v.y + Math.sin(bis) * (r + 34), pr.label, pr.color);
  } else if (pr.t === "ghost") {
    const p = A2V(s, pr.p); ctx.strokeStyle = pr.color; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.stroke();
  } else if (pr.t === "tag") {
    const p = A2V(s, pr.p); if (t > 0.55) pill(p.x + 40 * U, p.y, pr.text, pr.color);
  }
  ctx.globalAlpha = 1;
}

function draw(now = performance.now()) {
  updateScale();
  const s = cur();
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#07070c"; ctx.fillRect(0, 0, W, H);
  if (!s) return;
  const v = s.view;
  ctx.save();
  ctx.translate(-v.ox * v.k, -v.oy * v.k); ctx.scale(v.k, v.k);
  ctx.translate(v.cx, v.cy); ctx.rotate(-v.roll); ctx.translate(-v.cx, -v.cy);
  ctx.drawImage(s.img, 0, 0);
  ctx.restore();

  const metric = active && s.ready ? MODES[s.mode].metrics.find((m) => m.id === active) : null;
  if (metric) { ctx.fillStyle = "rgba(7,7,12,.45)"; ctx.fillRect(0, 0, W, H); }

  // Scan-Animation nach der Erkennung: Linie läuft über das Gesicht, Netz und Punkte erscheinen
  const scan = anim.scan && anim.scan.s === s ? anim.scan : null;
  let scanY = Infinity;
  if (scan) {
    const kps = MODES[s.mode].kp().map((k) => s.pts[k.id]).filter(Boolean).map((p) => toView(s, p));
    const bb = bbox(kps), t = Math.min(1, (now - scan.t0) / (scan.dur * 0.85)), top = bb.y0 - 60, bot = bb.y1 + 60;
    scanY = top + (bot - top) * ease(t);
    const g = ctx.createLinearGradient(0, scanY - 90, 0, scanY);
    g.addColorStop(0, "rgba(45,212,191,0)"); g.addColorStop(1, "rgba(45,212,191,.28)");
    ctx.fillStyle = g; ctx.fillRect(0, scanY - 90, W, 90);
    ctx.fillStyle = "rgba(94,234,212,.95)"; ctx.fillRect(0, scanY - 1.5, W, 3);
  }
  const visible = (q) => q.y <= scanY;

  // Gesichtsnetz (frontal)
  if (s.mode === "front" && s.ready && MP && $("tglMesh").checked && !metric) {
    ctx.strokeStyle = "rgba(255,255,255,.13)"; ctx.lineWidth = 1; ctx.beginPath();
    const V = s.pts.map((p) => toView(s, p));
    for (const { start, end } of MP.FaceLandmarker.FACE_LANDMARKS_TESSELATION) { if (!visible(V[start]) || !visible(V[end])) continue; ctx.moveTo(V[start].x, V[start].y); ctx.lineTo(V[end].x, V[end].y); }
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,138,61,.85)"; ctx.lineWidth = 2; ctx.beginPath();
    for (const { start, end } of MP.FaceLandmarker.FACE_LANDMARKS_FACE_OVAL) { if (!visible(V[start]) || !visible(V[end])) continue; ctx.moveTo(V[start].x, V[start].y); ctx.lineTo(V[end].x, V[end].y); }
    ctx.stroke();
  }
  // Profillinie
  if (s.mode === "side" && !metric) {
    const chain = (ids) => { const pts = ids.map((k) => s.pts[k]).filter(Boolean).map((p) => toView(s, p)); if (pts.length < 2) return;
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke(); };
    ctx.strokeStyle = "rgba(255,138,61,.85)"; ctx.lineWidth = 2; chain(PROFILE_LINE); chain(["Ar", "Go", "Me"]); chain(["Sn", "Ac"]);
  }

  // Messung der aktiven Metrik
  if (metric) { const P = alignedPts(s), t = reduceMotion ? 1 : ease(Math.min(1, Math.max(0, (now - anim.drawT0) / 380))); for (const pr of metric.viz(P, s.raw)) drawPrim(s, pr, t); }

  // Punkte
  if ($("tglPts").checked || metric || s.placing != null) {
    const involved = metric ? new Set(metric.kp) : null;
    for (const kp of MODES[s.mode].kp()) {
      const p = s.pts[kp.id]; if (!p) continue;
      if (involved && !involved.has(kp.id)) continue;
      const q = toView(s, p), hov = hoverKp === kp.id || drag?.id === kp.id;
      if (!visible(q)) continue;
      ctx.beginPath(); ctx.arc(q.x, q.y, (hov ? 9 : metric ? 7 : 5.5) * U, 0, 7);
      ctx.fillStyle = hov ? C.yellow : metric ? "#fff" : C.cyan; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = "rgba(7,7,12,.9)"; ctx.stroke();
      if (s.mode === "side" && !metric) { ctx.font = `700 ${FS - 4}px "JetBrains Mono", ui-monospace, monospace`; ctx.fillStyle = "#fff"; ctx.strokeStyle = "rgba(7,7,12,.9)"; ctx.lineWidth = 4;
        ctx.strokeText(kp.id, q.x + 10, q.y - 10); ctx.fillText(kp.id, q.x + 10, q.y - 10); }
    }
    const hk = hoverKp ?? drag?.id;
    if (hk != null && s.pts[hk]) { const kp = MODES[s.mode].kp().find((k) => k.id === hk); const q = toView(s, s.pts[hk]); pill(q.x, q.y - 34 * U, kp.label, C.yellow); }
  }

  // Lupe beim Verschieben – zeigt die Umgebung des Punkts vergrößert, damit der Finger nichts verdeckt
  if (drag && s.pts[drag.id]) {
    const q = toView(s, s.pts[drag.id]), R = 105 * U, Z = 2.6, lx = q.x < W / 2 ? W - R - 18 : R + 18, ly = q.y < H * 0.4 ? H - R - 18 : R + 18;
    ctx.save(); ctx.beginPath(); ctx.arc(lx, ly, R, 0, 7); ctx.clip();
    ctx.fillStyle = "#0c0e10"; ctx.fillRect(lx - R, ly - R, 2 * R, 2 * R);
    ctx.drawImage(canvas, q.x - R / Z, q.y - R / Z, (2 * R) / Z, (2 * R) / Z, lx - R, ly - R, 2 * R, 2 * R);
    ctx.strokeStyle = C.yellow; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(lx - 16, ly); ctx.lineTo(lx + 16, ly); ctx.moveTo(lx, ly - 16); ctx.lineTo(lx, ly + 16); ctx.stroke();
    ctx.restore();
    ctx.strokeStyle = "#ff8a3d"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(lx, ly, R, 0, 7); ctx.stroke();
  }
}

// ======================= Bewertung =======================
function scoreMetric(v, [lo, hi], tol) {
  if (v >= lo && v <= hi) { const c = (lo + hi) / 2, half = (hi - lo) / 2 || 1; return 10 - Math.abs(v - c) / half; }
  const d = v < lo ? lo - v : v - hi;
  return Math.max(0, 9 * Math.exp(-0.4 * (d / tol) ** 2));
}
function evaluate(s) {
  s.raw = s.mode === "front" ? computeFront(alignedPts(s)) : computeSide(alignedPts(s));
  s.results = MODES[s.mode].metrics.map((m) => {
    const v = s.raw[m.id], ideal = m.ideal[sex];
    return { ...m, value: v, idealR: ideal, score: scoreMetric(v, ideal, m.tol), status: v < ideal[0] ? "low" : v > ideal[1] ? "high" : "ok" };
  });
  const tw = s.results.reduce((a, r) => a + r.weight, 0);
  s.total = Math.min(10, Math.max(1, 0.3 + (s.results.reduce((a, r) => a + r.score * r.weight, 0) / tw) * 0.92));
}

const TIERS = [
  [8.5, "Herausragende Harmonie", "Deine Proportionen liegen fast überall im Idealbereich."],
  [7.5, "Sehr harmonisch", "Die meisten Metriken liegen im oder nahe am Idealbereich."],
  [6.5, "Überdurchschnittlich", "Solide Proportionen mit einzelnen Ausreißern."],
  [5.5, "Durchschnittlich", "Ausgewogen, mit einigen Bereichen außerhalb der Ideale."],
  [0, "Ausbaufähig", "Mehrere Werte liegen außerhalb der Idealbereiche – Foto-Bedingungen und Punkte prüfen."],
];

// ======================= UI =======================
function fmt(v, dec) { return v.toLocaleString("de-DE", { minimumFractionDigits: dec, maximumFractionDigits: dec }); }
const cls = (score) => (score >= 8 ? "good" : score >= 5 ? "mid" : "bad");
const scoreColor = (s) => (s >= 8 ? "#22c55e" : s >= 6.5 ? "#2dd4bf" : s >= 5 ? "#f5a524" : "#ef4444");
const rangeTxt = (r) => `${fmt(r.idealR[0], r.dec)}${r.idealR[0] < 0 ? " bis " : "–"}${fmt(r.idealR[1], r.dec)}${r.unit}`;
const statusText = (r) => (r.status === "ok" ? "Ideal" : r.status === "low" ? "Unter Ideal" : "Über Ideal");
let tabFilter = "all";

function renderTabs() {
  $("tabs").innerHTML = [["all", "Alle"], ...Object.entries(MODES[mode].cats)].map(([k, l]) => `<button class="tab${k === tabFilter ? " active" : ""}" data-filter="${k}">${l}</button>`).join("");
}

function renderResults() {
  const s = cur();
  if (!s?.ready) { $("results").hidden = true; return; }
  const { results, total } = s;
  $("results").hidden = false;
  if (s.countedUp) $("overallScore").textContent = fmt(total, 1); else { s.countedUp = true; countUp($("overallScore"), total, 1); }
  const shape = s.mode === "front" ? faceShape(s.raw) : profileType(s.raw);
  $("shapeChip").innerHTML = `<span class="shape-label">${s.mode === "front" ? "Gesichtsform" : "Profiltyp"}</span><strong>${shape[0]}</strong><span class="muted">${shape[1]}</span><span class="shape-tip">${shape[2]}</span>`;
  const ring = $("ringFg");
  ring.style.stroke = scoreColor(total);
  requestAnimationFrame(() => { ring.style.strokeDashoffset = String(326.7 * (1 - total / 10)); });
  const tier = TIERS.find(([t]) => total >= t);
  $("tierLabel").textContent = `${tier[1]} · ${mode === "front" ? "Frontal" : "Profil"}`;
  $("summaryText").textContent = `${tier[2]} ${results.filter((r) => r.status === "ok").length} von ${results.length} Metriken im Idealbereich.`;
  $("subScores").innerHTML = Object.entries(MODES[mode].cats).map(([k, label]) => {
    const rs = results.filter((r) => r.cat === k);
    const sc = rs.reduce((a, r) => a + r.score * r.weight, 0) / rs.reduce((a, r) => a + r.weight, 0);
    return `<div class="sub"><span>${label}</span><strong style="color:${scoreColor(sc)}">${fmt(sc, 1)}</strong></div>`;
  }).join("");
  renderTabs();
  $("metricsGrid").innerHTML = results.map((r) => {
    const [a, b] = r.range, pct = (v) => Math.min(100, Math.max(0, ((v - a) / (b - a)) * 100));
    return `<article class="card metric${pinned === r.id ? " active" : ""}" data-id="${r.id}" data-cat="${r.cat}" tabindex="0" ${tabFilter !== "all" && tabFilter !== r.cat ? "hidden" : ""}>
      <div class="metric-top"><h4>${r.name}</h4><span class="badge ${cls(r.score)}">${fmt(r.score, 1)} · ${statusText(r)}</span></div>
      <div class="val">${fmt(r.value, r.dec)}${r.unit}</div>
      <div class="scale"><div class="ideal" style="left:${pct(r.idealR[0])}%;width:${pct(r.idealR[1]) - pct(r.idealR[0])}%"></div><div class="marker" style="left:${pct(r.value)}%"></div></div>
      <div class="scale-legend"><span>${fmt(a, r.dec)}</span><span>Ideal ${rangeTxt(r)}</span><span>${fmt(b, r.dec)}</span></div>
      <p>${r.info}</p>
    </article>`;
  }).join("");
  const weak = results.filter((r) => r.status !== "ok").sort((x, y) => x.score * x.weight - y.score * y.weight).slice(0, 4);
  const strong = [...results].sort((x, y) => y.score - x.score || y.weight - x.weight).slice(0, 4);
  const item = (r, text) => `<li data-id="${r.id}" tabindex="0"><span class="ins-top"><strong>${r.name}</strong><span class="badge ${cls(r.score)}">${fmt(r.score, 1)}</span></span><span>${text}</span></li>`;
  $("strengthList").innerHTML = strong.map((r) => item(r, `${fmt(r.value, r.dec)}${r.unit} – Ideal ${rangeTxt(r)}`)).join("");
  $("tipsList").innerHTML = weak.length
    ? weak.map((r) => item(r, (r.status === "low" ? r.tipLow : r.tipHigh) || r.tipLow || r.tipHigh)).join("")
    : "<li>Alle Metriken liegen im Idealbereich – stark! Fokussiere dich auf Hautpflege, Frisur und Fitness, um das Beste herauszuholen.</li>";
  renderCaption();
}

function renderCaption() {
  const s = cur(), el = $("caption");
  if (!s) { el.innerHTML = ""; return; }
  if (s.placing != null) { el.innerHTML = "Tippe/klicke ins Bild, um den markierten Punkt zu setzen."; return; }
  if (!s.ready) { el.innerHTML = ""; return; }
  if (anim.scan) { el.innerHTML = `<span class="scan-dot"></span>Analysiere ${s.mode === "front" ? "478 Gesichtspunkte" : "Profilpunkte"}…`; return; }
  const r = active && s.results.find((x) => x.id === active);
  const ids = visibleIds(), i = r ? ids.indexOf(r.id) : -1;
  const nav = `<span class="stepper"><button class="step-btn" data-step="-1" aria-label="Vorheriger Wert">‹</button><span class="step-count">${i >= 0 ? i + 1 : "–"}/${ids.length}</span><button class="step-btn" data-step="1" aria-label="Nächster Wert">›</button></span>`;
  el.innerHTML = r
    ? `<span class="cap-main"><strong>${r.name}</strong><span class="cap-val">${fmt(r.value, r.dec)}${r.unit}</span><span class="badge ${cls(r.score)}">${fmt(r.score, 1)} · ${statusText(r)}</span><span class="cap-ideal">Ideal ${rangeTxt(r)}</span></span>${nav}`
    : `<span class="cap-main">Wähle einen Wert – die Messung erscheint im Bild. Punkte lassen sich verschieben.</span>${nav}`;
}

let zoomTimer = null;
function setActive(id) {
  if (active === id) return;
  active = id; anim.drawT0 = performance.now();
  const s = cur();
  clearTimeout(zoomTimer);
  if (s?.ready && !drag) {
    // kurze Verzögerung, damit schnelles Überfahren mehrerer Karten nicht hin und her zoomt
    zoomTimer = setTimeout(() => { const m = active && MODES[s.mode].metrics.find((x) => x.id === active); animateView(s, m ? metricView(s, m) : baseView(s)); }, id ? 140 : 260);
  }
  kick(); renderCaption();
}
function visibleIds() { return [...grid.querySelectorAll(".metric")].filter((el) => !el.hidden).map((el) => el.dataset.id); }
function pin(id, scroll) {
  pinned = id;
  grid.querySelectorAll(".metric").forEach((m) => m.classList.toggle("active", m.dataset.id === pinned));
  setActive(id);
  if (scroll && id) grid.querySelector(`[data-id="${id}"]`)?.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
}
function stepMetric(dir) {
  const ids = visibleIds(); if (!ids.length) return;
  const i = ids.indexOf(pinned ?? active);
  pin(ids[i < 0 ? (dir > 0 ? 0 : ids.length - 1) : (i + dir + ids.length) % ids.length], true);
}
function countUp(el, to, dec, dur = 900) {
  if (reduceMotion) { el.textContent = fmt(to, dec); return; }
  const t0 = performance.now();
  const f = (now) => { const t = Math.min(1, (now - t0) / dur); el.textContent = fmt(to * ease(t), dec); if (t < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}

function showError(msg) { const e = $("errorMsg"); e.textContent = msg || ""; e.hidden = !msg; }
function setLoading(on, text) { $("loader").hidden = !on; if (text) $("loaderText").textContent = text; }

function showWorkspace() {
  const s = cur();
  $("workspace").hidden = !s;
  $("dropZone").hidden = !!s && !stream;
  $("meshLabel").hidden = mode !== "front";
  $("replaceBtn").hidden = mode !== "side" || !s;
  updatePlaceBar();
  if (s) { draw(); if (!anim.scan) renderResults(); }
}

function updatePlaceBar() {
  const s = cur(), on = !!s && s.placing != null;
  $("placeBar").hidden = !on; canvas.classList.toggle("placing", on);
  if (!on) return;
  const kp = SIDE_KP[s.placing];
  $("placeStep").textContent = `${s.placing + 1}/${SIDE_KP.length}`;
  $("placeTitle").textContent = `${kp.label} (${kp.id})`;
  $("placeHint").textContent = kp.hint;
  $("placeBack").disabled = s.placing === 0;
  renderCaption();
}

// ======================= Analyse-Abläufe =======================
function yawEstimate(p) { // erwartet ausgerichtete Punkte
  const r = p[L.noseTip].x - p[L.rZygo].x, l = p[L.lZygo].x - p[L.noseTip].x; return (r - l) / (Math.abs(r) + Math.abs(l) || 1); }

async function analyzeImage(img) {
  showError("");
  active = pinned = null;
  const s = { mode, img, pts: mode === "front" ? [] : {}, ready: false, placing: null };
  sessions[mode] = s;
  fitView(s);
  $("results").hidden = true;
  showWorkspace();
  $("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  setLoading(true, landmarker ? "Gesicht wird erkannt…" : "KI-Modell wird geladen (einmalig ~4 MB)…");
  try {
    let det = null;
    try { await getLandmarker(); setLoading(true, "Gesicht wird erkannt…"); await new Promise((r) => setTimeout(r, 30)); det = await detectFace(img); }
    catch (e) { if (s.mode === "front") throw e; console.warn(e); }
    if (sessions[s.mode] !== s) return;
    if (s.mode === "front") {
      if (!det) throw new Error("Kein Gesicht erkannt. Bitte ein gut beleuchtetes Frontalfoto verwenden.");
      s.img = det.img;
      s.pts = det.lms.map((p) => ({ x: p.x * s.img.width, y: p.y * s.img.height }));
      const a = s.pts[L.rEyeOuter], b = s.pts[L.lEyeOuter];
      s.roll = Math.atan2(b.y - a.y, b.x - a.x); s.center = mid(a, b);
      const yaw = yawEstimate(alignedPts(s)), notes = [];
      if (det.rotated) notes.push(`Gesicht war um ${det.rotated > 180 ? 360 - det.rotated : det.rotated}° geneigt – automatisch aufgerichtet.`);
      if (Math.abs(yaw) > 0.18) notes.push(`Der Kopf ist zur Seite gedreht (${Math.round(Math.abs(yaw) * 100)} %). Für genaue Werte frontal in die Kamera schauen.`);
      finish(s, true);
      showError(notes.join(" "));
    } else if (det) {
      s.img = det.img;
      const px = det.lms.map((p) => ({ x: p.x * s.img.width, y: p.y * s.img.height }));
      const ea = px[33], eb = px[263], rr = Math.atan2(eb.y - ea.y, eb.x - ea.x);
      if (Math.abs(yawEstimate(px.map((p) => rot(p, ea, -rr)))) < 0.35) {
        startPlacing(s);
        showError("Das sieht nach einem Frontalfoto aus. Für die Profilanalyse bitte ein Seitenfoto verwenden – oder die Punkte jetzt manuell setzen.");
      } else {
        const rightVisible = det.lms[234].z < det.lms[454].z;
        for (const k of SIDE_KP) { const i = Array.isArray(k.mp) ? k.mp[rightVisible ? 0 : 1] : k.mp; s.pts[k.id] = px[i]; }
        finishSide(s, true);
        showError(`${det.rotated ? `Gesicht war geneigt – automatisch aufgerichtet. ` : ""}Punkte wurden automatisch gesetzt – prüfe sie und ziehe sie bei Bedarf an die richtige Stelle.`);
      }
    } else {
      startPlacing(s);
      showError("Kein Gesicht automatisch erkannt (bei reinen Seitenansichten normal). Setze die Punkte bitte nacheinander per Klick.");
    }
  } catch (err) {
    console.error(err);
    showError(err.message || "Analyse fehlgeschlagen.");
  } finally {
    setLoading(false);
  }
}

function finish(s, withScan = false) {
  s.ready = true; s.placing = null; evaluate(s); fitView(s);
  if (sessions[mode] !== s) return;
  if (withScan && !reduceMotion) {
    $("results").hidden = true; draw(); updatePlaceBar();
    $("workspace").hidden = false; $("dropZone").hidden = true;
    anim.scan = { s, t0: performance.now(), dur: 1700, done: () => { showWorkspace(); } };
    kick(); renderCaption();
  } else showWorkspace();
}
function finishSide(s, withScan = false) {
  const v = { x: s.pts.Me.x - s.pts.G.x, y: s.pts.Me.y - s.pts.G.y };
  s.roll = Math.atan2(v.y, v.x) - Math.PI / 2; s.center = mid(s.pts.G, s.pts.Me);
  finish(s, withScan);
}
function startPlacing(s) { s.pts = {}; s.ready = false; s.placing = 0; active = pinned = null; fitView(s); $("results").hidden = true; showWorkspace(); }

// ======================= Canvas-Interaktion =======================
function canvasPoint(e) { const r = canvas.getBoundingClientRect(); return { x: ((e.clientX - r.left) * W) / r.width, y: ((e.clientY - r.top) * H) / r.height }; }
function hitKp(q) {
  const s = cur(); if (!s || s.placing != null) return null;
  const metric = active && s.ready && MODES[s.mode].metrics.find((m) => m.id === active);
  if (!metric && !$("tglPts").checked) return null;
  let best = null, bd = 30 * U;
  for (const kp of MODES[s.mode].kp()) {
    if (!s.pts[kp.id] || (metric && !metric.kp.includes(kp.id))) continue;
    const d = dist(q, toView(s, s.pts[kp.id])); if (d < bd) { bd = d; best = kp.id; }
  }
  return best;
}
let rafPending = false;
function liveUpdate() {
  if (rafPending) return; rafPending = true;
  requestAnimationFrame(() => { rafPending = false; const s = cur(); if (s?.ready) evaluate(s); draw(); renderCaption(); });
}
canvas.addEventListener("pointerdown", (e) => {
  const s = cur(); if (!s) return;
  const q = canvasPoint(e);
  if (s.placing != null) {
    s.pts[SIDE_KP[s.placing].id] = fromView(s, q);
    s.placing++;
    if (s.placing >= SIDE_KP.length) { finishSide(s); showError(""); }
    else { updatePlaceBar(); draw(); }
    return;
  }
  const id = hitKp(q);
  if (id != null) { drag = { id }; canvas.setPointerCapture(e.pointerId); e.preventDefault(); draw(); }
});
canvas.addEventListener("pointermove", (e) => {
  const s = cur(); if (!s) return;
  const q = canvasPoint(e);
  if (drag) { s.pts[drag.id] = fromView(s, q); liveUpdate(); return; }
  if (e.pointerType === "mouse") { const id = hitKp(q); if (id !== hoverKp) { hoverKp = id; canvas.style.cursor = id != null ? "grab" : ""; draw(); } }
});
const endDrag = () => { if (!drag) return; drag = null; const s = cur(); if (s?.ready) { evaluate(s); renderResults(); } draw(); };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("pointerleave", () => { if (hoverKp != null && !drag) { hoverKp = null; draw(); } });
// Auf Touch-Geräten Scrollen nur verhindern, wenn ein Punkt gegriffen oder gesetzt wird
canvas.addEventListener("touchstart", (e) => { const s = cur(); if (!s) return; if (s.placing != null || hitKp(canvasPoint(e.touches[0])) != null) e.preventDefault(); }, { passive: false });

// Metrik-Karten ↔ Bild
const grid = $("metricsGrid");
grid.addEventListener("mouseover", (e) => { const c = e.target.closest(".metric"); if (c) setActive(c.dataset.id); });
grid.addEventListener("mouseleave", () => setActive(pinned));
grid.addEventListener("focusin", (e) => { const c = e.target.closest(".metric"); if (c) setActive(c.dataset.id); });
grid.addEventListener("click", (e) => {
  const c = e.target.closest(".metric"); if (!c) return;
  pin(pinned === c.dataset.id ? null : c.dataset.id);
});
$("caption").addEventListener("click", (e) => { const b = e.target.closest(".step-btn"); if (b) stepMetric(+b.dataset.step); });
for (const id of ["strengthList", "tipsList"]) $(id).addEventListener("click", (e) => { const li = e.target.closest("li[data-id]"); if (!li) return; pin(li.dataset.id, true); $("workspace").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" }); });
document.addEventListener("keydown", (e) => {
  if ($("workspace").hidden || !cur()?.ready || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const r = $("workspace").getBoundingClientRect(); if (r.bottom < 0 || r.top > innerHeight) return; e.preventDefault(); stepMetric(e.key === "ArrowRight" ? 1 : -1); }
  if (e.key === "Escape") pin(null);
});
$("tabs").addEventListener("click", (e) => {
  const t = e.target.closest(".tab"); if (!t) return;
  tabFilter = t.dataset.filter;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  grid.querySelectorAll(".metric").forEach((el) => { el.hidden = tabFilter !== "all" && el.dataset.cat !== tabFilter; });
  renderCaption();
});

// ======================= Laden / Kamera =======================
function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return showError("Bitte eine Bilddatei wählen.");
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const max = 1600, sc = Math.min(1, max / Math.max(img.width, img.height));
    const c = document.createElement("canvas");
    c.width = Math.round(img.width * sc); c.height = Math.round(img.height * sc);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    analyzeImage(c);
  };
  img.onerror = () => showError("Bild konnte nicht geladen werden.");
  img.src = url;
}

async function startCam() {
  showError("");
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } } }); }
  catch { return showError("Kein Kamerazugriff möglich."); }
  const v = $("video");
  v.srcObject = stream; await v.play();
  $("dropZone").hidden = false; $("dzEmpty").hidden = true; v.hidden = false; $("camBar").hidden = false;
}
function stopCam() {
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  $("video").hidden = true; $("dzEmpty").hidden = false; $("camBar").hidden = true;
  showWorkspace();
}
function snap() {
  const v = $("video"), c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  const cx = c.getContext("2d"); cx.translate(c.width, 0); cx.scale(-1, 1); cx.drawImage(v, 0, 0);
  stopCam(); analyzeImage(c);
}

// ======================= Verlauf & Export =======================
const HKEY = "proportio-history";
function getHistory() { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } }
function setHistory(h) { try { localStorage.setItem(HKEY, JSON.stringify(h)); return true; } catch { showError("Verlauf konnte nicht gespeichert werden (Speicher voll oder blockiert)."); return false; } }
function thumb() {
  const c = document.createElement("canvas"); c.width = 160; c.height = 200;
  const keep = active; active = null; draw(); c.getContext("2d").drawImage(canvas, 0, 0, 160, 200); active = keep; draw();
  return c.toDataURL("image/jpeg", 0.7);
}
function renderHistory() {
  const h = getHistory();
  $("history").hidden = !h.length;
  $("historyList").innerHTML = h.map((e) => `<div class="hist-item"><img src="${e.thumb}" alt="" /><strong style="color:${scoreColor(e.total)}">${fmt(e.total, 1)}</strong><span>${e.mode === "side" ? "Profil" : "Frontal"} · ${new Date(e.date).toLocaleDateString("de-DE")}</span></div>`).join("");
}
function saveToHistory() {
  const s = cur(); if (!s?.ready) return;
  const h = getHistory();
  h.unshift({ date: Date.now(), mode: s.mode, total: s.total, thumb: thumb(), values: Object.fromEntries(s.results.map((r) => [r.id, r.value])) });
  if (!setHistory(h.slice(0, 12))) return;
  renderHistory();
  $("saveBtn").textContent = "Gespeichert ✓";
  setTimeout(() => ($("saveBtn").textContent = "Im Verlauf speichern"), 1800);
}
function exportImage() {
  const s = cur(); if (!s?.ready) return;
  const Wd = 1080, pad = 40, imgW = Wd - pad * 2, imgH = Math.round((H / W) * imgW);
  const rowH = 46, Hd = pad + 90 + imgH + 30 + s.results.length * rowH + pad;
  const c = document.createElement("canvas"); c.width = Wd; c.height = Hd;
  const x = c.getContext("2d");
  x.fillStyle = "#0c0e10"; x.fillRect(0, 0, Wd, Hd);
  x.fillStyle = "#ececf4"; x.font = "800 40px Sora, sans-serif"; x.fillText(`Proportio · ${s.mode === "side" ? "Profil" : "Frontal"}`, pad, pad + 40);
  x.font = "700 40px 'JetBrains Mono', monospace"; x.fillStyle = scoreColor(s.total); x.textAlign = "right";
  x.fillText(`${fmt(s.total, 1)} / 10`, Wd - pad, pad + 40); x.textAlign = "left";
  x.drawImage(canvas, pad, pad + 80, imgW, imgH);
  let y = pad + 80 + imgH + 50;
  x.font = "500 24px Manrope, sans-serif";
  for (const r of s.results) {
    x.fillStyle = "#9a9ab3"; x.fillText(r.name, pad, y);
    x.fillStyle = "#ececf4"; x.textAlign = "right"; x.fillText(`${fmt(r.value, r.dec)}${r.unit}`, Wd - pad - 120, y);
    x.fillStyle = scoreColor(r.score); x.fillText(fmt(r.score, 1), Wd - pad, y); x.textAlign = "left";
    y += rowH;
  }
  const url = c.toDataURL("image/png");
  $("exportImg").src = url;
  const a = $("exportLink");
  a.href = url; a.download = `proportio-${s.mode}-${new Date().toISOString().slice(0, 10)}.png`;
  $("exportModal").hidden = false;
}

// ======================= Init & Events =======================
const infoCard = (m) => `<article class="card"><h3>${m.name}</h3><p class="muted">${m.info}</p></article>`;
$("metricInfo").innerHTML = FRONT_METRICS.map(infoCard).join("");
$("metricInfoSide").innerHTML = SIDE_METRICS.map(infoCard).join("");
$("year").textContent = new Date().getFullYear();

function setMode(m) {
  mode = m; active = pinned = null; hoverKp = null; tabFilter = "all";
  document.querySelectorAll("#modeSeg .seg-btn").forEach((x) => { x.classList.toggle("active", x.dataset.mode === m); x.setAttribute("aria-checked", x.dataset.mode === m); });
  $("modeHint").textContent = MODES[m].hint; $("dzSub").textContent = MODES[m].dz;
  $("sampleRow").hidden = m !== "front";
  showError("");
  showWorkspace();
}

$("fileInput").addEventListener("change", (e) => { loadFile(e.target.files[0]); e.target.value = ""; });
const dropTarget = $("analyze");
["dragenter", "dragover"].forEach((t) => dropTarget.addEventListener(t, (e) => { e.preventDefault(); $("dropZone").classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dropTarget.addEventListener(t, (e) => { e.preventDefault(); $("dropZone").classList.remove("drag"); }));
dropTarget.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
$("camBtn").addEventListener("click", startCam);
$("camStop").addEventListener("click", stopCam);
$("snapBtn").addEventListener("click", snap);
$("tglMesh").addEventListener("change", draw);
$("tglPts").addEventListener("change", draw);
$("tglAlign").addEventListener("change", () => { const s = cur(); if (s) fitView(s, true); });
$("tglZoom").addEventListener("change", () => { const s = cur(); if (!s?.ready) return; const m = active && MODES[s.mode].metrics.find((x) => x.id === active); animateView(s, m ? metricView(s, m) : baseView(s)); });
$("replaceBtn").addEventListener("click", () => { const s = cur(); if (s) { showError(""); startPlacing(s); } });
$("placeBack").addEventListener("click", () => { const s = cur(); if (s?.placing > 0) { s.placing--; delete s.pts[SIDE_KP[s.placing].id]; updatePlaceBar(); draw(); } });
$("saveBtn").addEventListener("click", saveToHistory);
$("exportBtn").addEventListener("click", exportImage);
$("clearHistory").addEventListener("click", () => { setHistory([]); renderHistory(); });
document.querySelectorAll("#modeSeg .seg-btn").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
document.querySelectorAll("#sexSeg .seg-btn").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#sexSeg .seg-btn").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-checked", x === b); });
  sex = b.dataset.sex;
  for (const s of Object.values(sessions)) if (s?.ready) evaluate(s);
  renderResults(); draw();
}));
if (CFG.noCamera) $("camBtn").hidden = true;
$("sampleBtn").addEventListener("click", () => {
  const img = new Image();
  img.onload = () => { const c = document.createElement("canvas"); c.width = img.width; c.height = img.height; c.getContext("2d").drawImage(img, 0, 0); analyzeImage(c); };
  img.onerror = () => showError("Beispielfoto konnte nicht geladen werden.");
  img.src = CFG.sampleUrl || "assets/sample-front.jpg";
});
$("exportClose").addEventListener("click", () => ($("exportModal").hidden = true));
let resizeT = null;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => draw(), 120); });
setMode("front");
renderHistory();
