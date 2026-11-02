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
const C = { cyan: "#22d3ee", yellow: "#facc15", pink: "#f472b6", violet: "#a78bfa", green: "#4ade80", red: "#f87171", white: "rgba(255,255,255,.75)" };
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
  return {
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
];

// ======================= SEITENPROFIL =======================
const SIDE_KP = [
  { id: "G", label: "Glabella", hint: "Vorderster Punkt der Stirn zwischen den Augenbrauen", mp: 9 },
  { id: "N", label: "Nasion", hint: "Tiefster Punkt der Nasenwurzel zwischen Stirn und Nase", mp: 168 },
  { id: "Prn", label: "Pronasale", hint: "Nasenspitze – vorderster Punkt der Nase", mp: 1 },
  { id: "Sn", label: "Subnasale", hint: "Übergang vom Nasensteg zur Oberlippe", mp: 2 },
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
  const front = (p) => -sideOf(P.Go, P.Prn, P.Pog) * sideOf(p, P.Prn, P.Pog); // >0 = vor der E-Linie
  const eDist = (p) => front(p) * dist(p, foot(p, P.Prn, P.Pog)) * mm;
  return {
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

function fitView(s) {
  const useAlign = $("tglAlign").checked && s.ready;
  let x0 = 0, y0 = 0, x1 = s.img.width, y1 = s.img.height, roll = 0, c = { x: 0, y: 0 };
  if (useAlign) {
    roll = s.roll; c = s.center;
    const P = Object.values(alignedPts(s));
    x0 = Math.min(...P.map((p) => p.x)); x1 = Math.max(...P.map((p) => p.x));
    y0 = Math.min(...P.map((p) => p.y)); y1 = Math.max(...P.map((p) => p.y));
    const w = x1 - x0, h = y1 - y0;
    if (s.mode === "front") { x0 -= w * 0.22; x1 += w * 0.22; y0 -= h * 0.32; y1 += h * 0.1; }
    else { x0 -= w * 0.3; x1 += w * 0.3; y0 -= h * 0.35; y1 += h * 0.25; }
  }
  let bw = x1 - x0, bh = y1 - y0;
  const ar = W / H;
  if (bw / bh > ar) { const nh = bw / ar; y0 -= (nh - bh) / 2; bh = nh; } else { const nw = bh * ar; x0 -= (nw - bw) / 2; bw = nw; }
  s.view = { k: W / bw, ox: x0, oy: y0, roll, cx: c.x, cy: c.y };
}

// ======================= Zeichnen =======================
// Schrift- und Punktgröße passen sich der angezeigten Canvas-Größe an (Handy = größer im Canvas)
let FS = 22, U = 1;
function updateScale() { const shown = canvas.clientWidth || 450; FS = Math.round(Math.min(44, Math.max(20, (12.5 * W) / shown))); U = FS / 22; }
function pill(x, y, text, color) {
  ctx.font = `700 ${FS}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 16, h = FS + 12;
  x = Math.min(Math.max(x, w / 2 + 4), W - w / 2 - 4); y = Math.min(Math.max(y, h / 2 + 4), H - h / 2 - 4);
  ctx.fillStyle = "rgba(10,10,18,.88)"; ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(x - w / 2, y - h / 2, w, h, h / 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y + 1);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
}

function drawPrim(s, pr) {
  ctx.setLineDash([]);
  if (pr.t === "seg") {
    const a = A2V(s, pr.a), b = A2V(s, pr.b);
    ctx.strokeStyle = pr.color; ctx.lineWidth = 3 * U; ctx.setLineDash((pr.dash || []).map((d) => d * U));
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.setLineDash([]);
    if (!pr.dash) for (const p of [a, b]) { ctx.fillStyle = pr.color; ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, 7); ctx.fill(); }
    if (pr.label) { const m = mid(a, b), len = dist(a, b) || 1, nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;
      pill(m.x + nx * 22 * U, m.y + ny * 22 * U, pr.label, pr.color); }
  } else if (pr.t === "arc") {
    const a = A2V(s, pr.a), v = A2V(s, pr.v), b = A2V(s, pr.b);
    ctx.strokeStyle = C.cyan; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(v.x, v.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    const t1 = Math.atan2(a.y - v.y, a.x - v.x); let d = Math.atan2(b.y - v.y, b.x - v.x) - t1;
    while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
    const r = Math.max(26, Math.min(60, dist(a, v) * 0.45, dist(b, v) * 0.45));
    ctx.fillStyle = "rgba(250,204,21,.18)"; ctx.strokeStyle = pr.color; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(v.x, v.y); ctx.arc(v.x, v.y, r, t1, t1 + d, d < 0); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.arc(v.x, v.y, r, t1, t1 + d, d < 0); ctx.stroke();
    const bis = t1 + d / 2;
    if (pr.label) pill(v.x + Math.cos(bis) * (r + 34), v.y + Math.sin(bis) * (r + 34), pr.label, pr.color);
  } else if (pr.t === "ghost") {
    const p = A2V(s, pr.p); ctx.strokeStyle = pr.color; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.stroke();
  } else if (pr.t === "tag") {
    const p = A2V(s, pr.p); pill(p.x + 40, p.y, pr.text, pr.color);
  }
}

function draw() {
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

  // Gesichtsnetz (frontal)
  if (s.mode === "front" && s.ready && MP && $("tglMesh").checked && !metric) {
    ctx.strokeStyle = "rgba(255,255,255,.13)"; ctx.lineWidth = 1; ctx.beginPath();
    const V = s.pts.map((p) => toView(s, p));
    for (const { start, end } of MP.FaceLandmarker.FACE_LANDMARKS_TESSELATION) { ctx.moveTo(V[start].x, V[start].y); ctx.lineTo(V[end].x, V[end].y); }
    ctx.stroke();
    ctx.strokeStyle = "rgba(124,92,255,.85)"; ctx.lineWidth = 2; ctx.beginPath();
    for (const { start, end } of MP.FaceLandmarker.FACE_LANDMARKS_FACE_OVAL) { ctx.moveTo(V[start].x, V[start].y); ctx.lineTo(V[end].x, V[end].y); }
    ctx.stroke();
  }
  // Profillinie
  if (s.mode === "side" && !metric) {
    const chain = (ids) => { const pts = ids.map((k) => s.pts[k]).filter(Boolean).map((p) => toView(s, p)); if (pts.length < 2) return;
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke(); };
    ctx.strokeStyle = "rgba(124,92,255,.8)"; ctx.lineWidth = 2; chain(PROFILE_LINE); chain(["Ar", "Go", "Me"]);
  }

  // Messung der aktiven Metrik
  if (metric) { const P = alignedPts(s); for (const pr of metric.viz(P, s.raw)) drawPrim(s, pr); }

  // Punkte
  if ($("tglPts").checked || metric || s.placing != null) {
    const involved = metric ? new Set(metric.kp) : null;
    for (const kp of MODES[s.mode].kp()) {
      const p = s.pts[kp.id]; if (!p) continue;
      if (involved && !involved.has(kp.id)) continue;
      const q = toView(s, p), hov = hoverKp === kp.id || drag?.id === kp.id;
      ctx.beginPath(); ctx.arc(q.x, q.y, (hov ? 9 : metric ? 7 : 5.5) * U, 0, 7);
      ctx.fillStyle = hov ? C.yellow : metric ? "#fff" : C.cyan; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = "rgba(7,7,12,.9)"; ctx.stroke();
      if (s.mode === "side" && !metric) { ctx.font = `700 ${FS - 4}px Inter, sans-serif`; ctx.fillStyle = "#fff"; ctx.strokeStyle = "rgba(7,7,12,.9)"; ctx.lineWidth = 4;
        ctx.strokeText(kp.id, q.x + 10, q.y - 10); ctx.fillText(kp.id, q.x + 10, q.y - 10); }
    }
    const hk = hoverKp ?? drag?.id;
    if (hk != null && s.pts[hk]) { const kp = MODES[s.mode].kp().find((k) => k.id === hk); const q = toView(s, s.pts[hk]); pill(q.x, q.y - 34 * U, kp.label, C.yellow); }
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
const scoreColor = (s) => (s >= 7.5 ? "#22c55e" : s >= 6 ? "#7c5cff" : s >= 5 ? "#eab308" : "#ef4444");
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
  $("overallScore").textContent = fmt(total, 1);
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
  $("tipsList").innerHTML = weak.length
    ? weak.map((r) => `<li><strong>${r.name}:</strong> ${(r.status === "low" ? r.tipLow : r.tipHigh) || r.tipLow || r.tipHigh}</li>`).join("")
    : "<li>Alle Metriken liegen im Idealbereich – stark! Fokussiere dich auf Hautpflege, Frisur und Fitness, um das Beste herauszuholen.</li>";
  renderCaption();
}

function renderCaption() {
  const s = cur(), el = $("caption");
  if (!s) { el.innerHTML = ""; return; }
  if (s.placing != null) { el.innerHTML = "Tippe/klicke ins Bild, um den markierten Punkt zu setzen."; return; }
  if (!s.ready) { el.innerHTML = ""; return; }
  const r = active && s.results.find((x) => x.id === active);
  el.innerHTML = r
    ? `<strong>${r.name}</strong><span>${fmt(r.value, r.dec)}${r.unit}</span><span>· Ideal ${rangeTxt(r)}</span><span class="badge ${cls(r.score)}">${fmt(r.score, 1)} · ${statusText(r)}</span>`
    : "Fahre über einen Wert (oder tippe ihn an), um die Messung im Bild zu sehen. Punkte lassen sich verschieben.";
}

function setActive(id) { if (active === id) return; active = id; draw(); renderCaption(); }

function showError(msg) { const e = $("errorMsg"); e.textContent = msg || ""; e.hidden = !msg; }
function setLoading(on, text) { $("loader").hidden = !on; if (text) $("loaderText").textContent = text; }

function showWorkspace() {
  const s = cur();
  $("workspace").hidden = !s;
  $("dropZone").hidden = !!s && !stream;
  $("meshLabel").hidden = mode !== "front";
  $("replaceBtn").hidden = mode !== "side" || !s;
  updatePlaceBar();
  if (s) { draw(); renderResults(); }
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
      finish(s);
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
        finishSide(s);
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

function finish(s) { s.ready = true; s.placing = null; evaluate(s); fitView(s); if (sessions[mode] === s) showWorkspace(); }
function finishSide(s) {
  const v = { x: s.pts.Me.x - s.pts.G.x, y: s.pts.Me.y - s.pts.G.y };
  s.roll = Math.atan2(v.y, v.x) - Math.PI / 2; s.center = mid(s.pts.G, s.pts.Me);
  finish(s);
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
  pinned = pinned === c.dataset.id ? null : c.dataset.id;
  grid.querySelectorAll(".metric").forEach((m) => m.classList.toggle("active", m.dataset.id === pinned));
  active = null; setActive(pinned ?? c.dataset.id);
});
$("tabs").addEventListener("click", (e) => {
  const t = e.target.closest(".tab"); if (!t) return;
  tabFilter = t.dataset.filter;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  grid.querySelectorAll(".metric").forEach((el) => { el.hidden = tabFilter !== "all" && el.dataset.cat !== tabFilter; });
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
  x.fillStyle = "#0b0b12"; x.fillRect(0, 0, Wd, Hd);
  x.fillStyle = "#ececf4"; x.font = "800 40px Inter, sans-serif"; x.fillText(`Proportio · ${s.mode === "side" ? "Profil" : "Frontal"}`, pad, pad + 40);
  x.font = "700 40px Inter, sans-serif"; x.fillStyle = scoreColor(s.total); x.textAlign = "right";
  x.fillText(`${fmt(s.total, 1)} / 10`, Wd - pad, pad + 40); x.textAlign = "left";
  x.drawImage(canvas, pad, pad + 80, imgW, imgH);
  let y = pad + 80 + imgH + 50;
  x.font = "500 24px Inter, sans-serif";
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
$("tglAlign").addEventListener("change", () => { const s = cur(); if (s) { fitView(s); draw(); } });
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
$("exportClose").addEventListener("click", () => ($("exportModal").hidden = true));
setMode("front");
renderHistory();
