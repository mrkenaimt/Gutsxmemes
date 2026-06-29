// ============================================================
//  Meme Detector — منطق مترجم من كود بايثون (mediapipe) إلى JS
//  باستخدام @mediapipe/tasks-vision (FaceLandmarker + HandLandmarker)
// ============================================================

import {
  FilesetResolver,
  FaceLandmarker,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15";

// ---------- روابط صور الميمات (غيّرها لمسارات صورك) ----------
const MEME_PATHS = {
  sonic: "./memes/Sonic.jpeg",
  cara: "./memes/cara.jpeg",
  cristiano: "./memes/cristiano.png",
  lengua: "./memes/gato1.png",
  ceja: "./memes/perro.jpeg",
  rata: "./memes/rata.jpeg",
};

// ---------- عناصر DOM ----------
const startScreen = document.getElementById("startScreen");
const camScreen   = document.getElementById("camScreen");
const startBtn    = document.getElementById("startBtn");
const closeBtn    = document.getElementById("closeBtn");
const errBox      = document.getElementById("errBox");
const video       = document.getElementById("video");
const overlay     = document.getElementById("overlay");
const ctx         = overlay.getContext("2d");
const calOverlay  = document.getElementById("calOverlay");
const calBarFill  = document.getElementById("calBarFill");
const calPct      = document.getElementById("calPct");
const hudName     = document.getElementById("hudName");
const memeImg     = document.getElementById("memeImg");
const memePlaceholder = document.getElementById("memePlaceholder");

// ============================================================
//  دوال هندسية أساسية (تقابل d() و esc() و px() بالبايثون)
// ============================================================

// المسافة الإقليدية بين نقطتين بثلاث أبعاد (تقابل دالة d في بايثون)
function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

// مقياس الوجه: المسافة بين الذقن وأعلى الجبهة (تقابل esc)
function faceScale(lm) {
  return dist(lm[152], lm[10]) + 1e-6;
}

// ============================================================
//  حالة الأصابع (تقابل dedos_estado بالبايثون)
//  isLeft هنا تُمرَّر بعد تصحيح handedness (راجع renderLoop بالأسفل)
//  بحيث تطابق تماماً معنى "izq" بكود البايثون الأصلي.
// ============================================================
function fingersState(lm, isLeft) {
  const tips = [8, 12, 16, 20];
  const mids = [6, 10, 14, 18];
  const out = [];
  // الإبهام: نفس شرط بايثون (lm[4].x > lm[3].x if izq else lm[4].x < lm[3].x)
  out.push(isLeft ? (lm[4].x > lm[3].x ? 1 : 0) : (lm[4].x < lm[3].x ? 1 : 0));
  for (let i = 0; i < tips.length; i++) {
    out.push(lm[tips[i]].y < lm[mids[i]].y ? 1 : 0);
  }
  return out;
}

// ============================================================
//  كلاس المعايرة (يقابل class Cal بالبايثون)
// ============================================================
class Calibrator {
  constructor() {
    this.N = 45;
    this.buf = { ci: [], cd: [], cen: [], lap: [], llb: [], bi_y: [], bd_y: [], gap: [] };
    this.done = false;
    this.thr = {
      ci: 0.180, cd: 0.180, cen_lo: 0.185,
      lap: 0.055, llb: 0.145,
      bi_y_lo: 0.30, bd_y_lo: 0.30,
      gap_lo: 0.10,
    };
  }

  feed(lm) {
    if (this.done) return;
    const e = faceScale(lm);
    this.buf.ci.push(dist(lm[52], lm[159]) / e);
    this.buf.cd.push(dist(lm[282], lm[386]) / e);
    this.buf.cen.push(dist(lm[55], lm[285]) / e);
    this.buf.lap.push(dist(lm[13], lm[14]) / e);
    this.buf.llb.push(dist(lm[17], lm[152]) / e);
    this.buf.bi_y.push(lm[55].y - lm[9].y);
    this.buf.bd_y.push(lm[285].y - lm[9].y);
    this.buf.gap.push(Math.abs(lm[55].x - lm[285].x));
    if (this.buf.ci.length >= this.N) this._calc();
  }

  _median(arr) {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }

  _std(arr) {
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
    return Math.sqrt(variance);
  }

  _calc() {
    const m = (k) => this._median(this.buf[k]);
    const s = (k) => this._std(this.buf[k]);
    const mgC = (k) => Math.max(1.5 * s(k), 0.015);
    const mgB = (k, mn) => Math.max(3 * s(k), mn);

    this.thr.ci      = m("ci")  + mgC("ci");
    this.thr.cd      = m("cd")  + mgC("cd");
    this.thr.cen_lo  = m("cen") - mgC("cen");
    this.thr.lap     = m("lap") + mgB("lap", 0.032);
    this.thr.llb     = m("llb") - mgB("llb", 0.018);
    this.thr.bi_y_lo = m("bi_y") + mgC("bi_y");
    this.thr.bd_y_lo = m("bd_y") + mgC("bd_y");
    this.thr.gap_lo  = m("gap")  - mgC("gap");
    this.done = true;
  }

  get progress() {
    return Math.min(this.buf.ci.length / this.N, 1.0);
  }
}

// ============================================================
//  دوال الاكتشاف (تقابل det_lengua / det_ceja / ... بالبايثون)
// ============================================================

function detLengua(lm, cal) {
  const e = faceScale(lm);
  const bocaAbierta = dist(lm[13], lm[14]) / e > cal.thr.lap;
  const linguaBaja  = dist(lm[17], lm[152]) / e < cal.thr.llb;
  const puntaFuera  = lm[17].y > lm[14].y + 0.012;
  return bocaAbierta && linguaBaja && puntaFuera;
}

function detCeja(lm, cal) {
  const e    = faceScale(lm);
  const ci   = dist(lm[52],  lm[159]) / e;
  const cd   = dist(lm[282], lm[386]) / e;
  const cen  = dist(lm[55],  lm[285]) / e;
  const bi_y = lm[55].y  - lm[9].y;
  const bd_y = lm[285].y - lm[9].y;
  const gap  = Math.abs(lm[55].x - lm[285].x);
  return (
    ci > cal.thr.ci ||
    cd > cal.thr.cd ||
    cen < cal.thr.cen_lo ||
    bi_y > cal.thr.bi_y_lo ||
    bd_y > cal.thr.bd_y_lo ||
    gap < cal.thr.gap_lo
  );
}

// manos: مصفوفة عناصر {fingers, lm}
function detCristiano(manos, lmCara) {
  const boca = lmCara[13];
  return manos.some(({ lm }) => dist(lm[8], boca) < 0.09 || dist(lm[12], boca) < 0.09);
}

function detRata(fingers) {
  return fingers.length === 5 &&
    fingers[0] === 0 && fingers[1] === 1 && fingers[2] === 1 &&
    fingers[3] === 0 && fingers[4] === 0;
}

function detSonic(manos, lmCara) {
  if (manos.length !== 2) return false;
  const narizY = lmCara[1].y;
  return manos.every(({ lm }) => lm[9].y < narizY);
}

function detCara(manos) {
  if (manos.length !== 2) return false;
  for (const { fingers, lm } of manos) {
    const restAbiertos = fingers[1] === 1 && fingers[2] === 1 && fingers[3] === 1 && fingers[4] === 1;
    if (!restAbiertos || lm[0].y < 0.50) return false;
  }
  return Math.abs(manos[0].lm[0].x - manos[1].lm[0].x) >= 0.20;
}

// ============================================================
//  نقاط الرسم (تقابل FACE_OVAL / EYE_L / ... بالبايثون)
// ============================================================
const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10];
const EYE_L  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33];
const EYE_R  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362];
const BROW_L = [70,63,105,66,107,55,65,52,53,46];
const BROW_R = [300,293,334,296,336,285,295,282,283,276];
const LIPS_OUT = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61];
const LIPS_IN  = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78];
const NOSE = [168,6,197,195,5,4,1,19,94,2];

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const COL_BASE = "rgb(140,200,140)";
const COL_ACT  = "rgb(80,240,80)";

// نقطة نسبية (0..1) إلى بكسل على الكانفاس.
// landmarks القادمة من mediapipe غير معكوسة (تطابق الصورة الأصلية من الكاميرا)،
// لكن الفيديو يُعرض معكوساً بصرياً بالـ CSS (transform: scaleX(-1)) ليبدو كمرآة،
// فنعكس إحداثي x هنا أيضاً (mirror=true) ليتطابق الرسم تماماً مع ما يراه المستخدم.
function px(pt, W, H, mirror = true) {
  const x = mirror ? (1 - pt.x) * W : pt.x * W;
  return [x, pt.y * H];
}

function drawPath(ctx, lm, indices, W, H, color, close = false) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  const pts = indices.map((i) => px(lm[i], W, H));
  pts.forEach(([x, y], j) => (j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  if (close) ctx.closePath();
  ctx.stroke();
  pts.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 1.3, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function drawFaceMinimal(ctx, lm, W, H, cal) {
  const e = faceScale(lm);
  const ci = dist(lm[52], lm[159]) / e;
  const cd = dist(lm[282], lm[386]) / e;
  const cen = dist(lm[55], lm[285]) / e;
  const bocaAct = dist(lm[13], lm[14]) / e > cal.thr.lap && dist(lm[17], lm[152]) / e < cal.thr.llb;
  const cejaAct = ci > cal.thr.ci || cd > cal.thr.cd || cen < cal.thr.cen_lo;

  drawPath(ctx, lm, FACE_OVAL, W, H, COL_BASE, false);
  drawPath(ctx, lm, EYE_L, W, H, COL_BASE, true);
  drawPath(ctx, lm, EYE_R, W, H, COL_BASE, true);
  drawPath(ctx, lm, BROW_L, W, H, cejaAct ? COL_ACT : COL_BASE, false);
  drawPath(ctx, lm, BROW_R, W, H, cejaAct ? COL_ACT : COL_BASE, false);
  drawPath(ctx, lm, NOSE, W, H, COL_BASE, false);
  drawPath(ctx, lm, LIPS_OUT, W, H, bocaAct ? COL_ACT : COL_BASE, true);
  drawPath(ctx, lm, LIPS_IN, W, H, bocaAct ? COL_ACT : COL_BASE, true);
}

function drawHandMinimal(ctx, lm, W, H, fingers) {
  ctx.strokeStyle = COL_BASE;
  ctx.lineWidth = 1.2;
  for (const [a, b] of HAND_CONNECTIONS) {
    const [x1, y1] = px(lm[a], W, H);
    const [x2, y2] = px(lm[b], W, H);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.fillStyle = COL_BASE;
  for (let i = 0; i < 21; i++) {
    const [x, y] = px(lm[i], W, H);
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
  const tipsIdx = [4, 8, 12, 16, 20];
  ctx.fillStyle = COL_ACT;
  tipsIdx.forEach((tip, i) => {
    if (fingers[i]) {
      const [x, y] = px(lm[tip], W, H);
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

// ============================================================
//  "تصويت" الاستقرار على آخر 10 إطارات (تقابل deque + Counter)
// ============================================================
class VoteBuffer {
  constructor(size = 10, minVotos = 6) {
    this.size = size;
    this.minVotos = minVotos;
    this.buf = [];
  }
  push(val) {
    this.buf.push(val);
    if (this.buf.length > this.size) this.buf.shift();
  }
  // يرجع المفتاح الأكثر تكراراً إذا تجاوز الحد الأدنى، وإلا null
  topIfStable() {
    const counts = new Map();
    for (const v of this.buf) counts.set(v, (counts.get(v) || 0) + 1);
    let top = null, max = 0;
    for (const [k, c] of counts) {
      if (c > max) { max = c; top = k; }
    }
    return max >= this.minVotos ? top : undefined; // undefined = لا تغيّر القيمة الحالية
  }
}

// ============================================================
//  التشغيل الرئيسي
// ============================================================
let faceLandmarker, handLandmarker;
let cal = new Calibrator();
let voteBuf = new VoteBuffer(10, 6);
let imgActual = null;
let running = false;

async function initModels() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
}

function setMeme(key) {
  if (key === imgActual) return;
  imgActual = key;
  if (key && MEME_PATHS[key]) {
    memeImg.src = MEME_PATHS[key];
    memeImg.style.display = "block";
    memePlaceholder.style.display = "none";
    hudName.textContent = labelFor(key);
    hudName.classList.remove("neutral");
  } else {
    memeImg.style.display = "none";
    memePlaceholder.style.display = "block";
    hudName.textContent = "محايد";
    hudName.classList.add("neutral");
  }
}

function labelFor(key) {
  const map = {
    sonic: "Sonic", cara: "تأطير الوجه", cristiano: "سيوو!",
    lengua: "لسان طالع 😛", ceja: "حاجب مرفوع 🤨", rata: "إشارة فأر",
  };
  return map[key] || key;
}

function renderLoop() {
  if (!running) return;
  const W = overlay.width, H = overlay.height;
  ctx.clearRect(0, 0, W, H);

  const now = performance.now();
  const faceRes = faceLandmarker.detectForVideo(video, now);
  const handRes = handLandmarker.detectForVideo(video, now);

  let lmCara = null;
  if (faceRes.faceLandmarks && faceRes.faceLandmarks.length > 0) {
    lmCara = faceRes.faceLandmarks[0];
  }

  const manos = [];
  if (handRes.landmarks && handRes.landmarks.length > 0) {
    for (let i = 0; i < handRes.landmarks.length; i++) {
      const lm = handRes.landmarks[i];
      // ملاحظة مهمة: كود البايثون الأصلي يعمل flip(frame,1) قبل تمرير الصورة لـ mediapipe،
      // فالـ landmarks هناك كانت تُحسب من صورة "معكوسة أصلاً" (مثل كاميرا سيلفي).
      // هنا بالمتصفح، عنصر <video> يحتوي الصورة الأصلية (غير معكوسة)، والعكس البصري (mirror)
      // يتم فقط بـ CSS للعرض. هذا يجعل categoryName القادم من النموذج معكوساً عن
      // ما يقابل منطق البايثون، فنبدّله هنا لمطابقة نفس السلوك تماماً.
      const rawHandedness = handRes.handedness[i][0].categoryName; // "Left" | "Right"
      const handedness = rawHandedness === "Left" ? "Right" : "Left";
      const isLeft = handedness === "Left";
      const fingers = fingersState(lm, isLeft);
      drawHandMinimal(ctx, lm, W, H, fingers);
      manos.push({ fingers, lm, side: isLeft ? "ي" : "د" });
    }
  }

  if (!cal.done) {
    // شاشة المعايرة
    if (lmCara) cal.feed(lmCara);
    const pct = Math.round(cal.progress * 100);
    calBarFill.style.width = pct + "%";
    calPct.textContent = pct + "%";
    if (cal.done) {
      calOverlay.style.display = "none";
    }
  } else {
    if (lmCara) drawFaceMinimal(ctx, lmCara, W, H, cal);

    let det = null;
    if (lmCara && manos.length === 2 && detSonic(manos, lmCara)) {
      det = "sonic";
    } else if (manos.length === 2 && detCara(manos)) {
      det = "cara";
    } else if (lmCara && manos.length > 0 && detCristiano(manos, lmCara)) {
      det = "cristiano";
    } else if (lmCara && detLengua(lmCara, cal)) {
      det = "lengua";
    } else if (lmCara && detCeja(lmCara, cal)) {
      det = "ceja";
    } else if (manos.length === 1 && detRata(manos[0].fingers)) {
      det = "rata";
    }

    voteBuf.push(det);
    const stable = voteBuf.topIfStable();
    if (stable !== undefined) setMeme(stable);
  }

  requestAnimationFrame(renderLoop);
}

function resizeOverlay() {
  const rect = camScreen.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}

async function startApp() {
  errBox.style.display = "none";
  startBtn.disabled = true;
  startBtn.textContent = "...جاري التحميل";
  try {
    await startCamera();
    await initModels();
    startScreen.style.display = "none";
    camScreen.style.display = "block";
    resizeOverlay();
    window.addEventListener("resize", resizeOverlay);
    running = true;
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error(err);
    errBox.textContent = "تعذر تشغيل الكاميرا أو تحميل الموديل: " + err.message;
    errBox.style.display = "block";
    startBtn.disabled = false;
    startBtn.textContent = "تشغيل الكاميرا";
  }
}

function stopApp() {
  running = false;
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (window.Telegram && window.Telegram.WebApp) {
    window.Telegram.WebApp.close();
  } else {
    camScreen.style.display = "none";
    startScreen.style.display = "flex";
  }
}

startBtn.addEventListener("click", startApp);
closeBtn.addEventListener("click", stopApp);

// تهيئة واجهة تيليجرام (إن وُجدت)
if (window.Telegram && window.Telegram.WebApp) {
  window.Telegram.WebApp.ready();
  window.Telegram.WebApp.expand();
}
