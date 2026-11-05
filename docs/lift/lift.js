// Subject Lift: tap a person (or object) in a photo to lift it out, then
// compose lifted subjects into a new image. Everything runs in the browser:
// the photo never leaves the device. Two MediaPipe models do the work:
//
//   * magic_touch: tap-prompted segmentation. Gives the crisp outline of the
//     thing under the finger.
//   * deeplab_v3: 21-class scene segmentation, used only for its "person"
//     class, always on a crop around the subject (it sees a 257 px square, so
//     a crop keeps people big and undistorted). It decides whether a tap hit a
//     person, grows a partial lift (just the face) to the whole person, and
//     trims stray background patches Magic Touch sometimes adds.
//
// Models and runtime (@mediapipe/tasks-vision, pinned below) are Apache-2.0.

const MP_VERSION = '1.1.1';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODELS = {
  person: {
    url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite',
    size: 2780176,
  },
  touch: {
    url: 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite',
    size: 6227884,
  },
};
const MAX_SIDE = 2048;          // longest side we process; keeps masks fast and memory sane
const MIN_PIXELS = 64;          // a selection smaller than this is treated as "nothing found"

const $ = (id) => document.getElementById(id);
const el = {
  open: $('btn-open'), file: $('file'), drop: $('drop'), stage: $('stage'), photo: $('photo'),
  base: $('base'), overlay: $('overlay'), busy: $('busy'), busyText: $('busy-text'),
  tools: $('lift-tools'), status: $('status'), all: $('btn-all'), actions: $('sel-actions'),
  use: $('btn-use'), copy: $('btn-copy'), save: $('btn-save'), clear: $('btn-clear'),
  feather: $('feather'), grow: $('grow'), tray: $('tray'),
  compose: $('compose'), composeWrap: $('compose-wrap'), composeEmpty: $('compose-empty'),
  size: $('size'), bg: $('bg'), bgColor: $('bg-color'), bgColor2: $('bg-color2'),
  bgColorL: $('bg-color-l'), bgColor2L: $('bg-color2-l'), bgFile: $('bg-file'), fmt: $('fmt'),
  exp: $('btn-export'), expCopy: $('btn-export-copy'),
  layerProps: $('layer-props'), lScale: $('l-scale'), lRot: $('l-rot'), lShadow: $('l-shadow'),
  lSticker: $('l-sticker'), lFlip: $('l-flip'), lUp: $('l-up'), lDown: $('l-down'),
  lDup: $('l-dup'), lDel: $('l-del'), toast: $('toast'),
};

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── Small utilities ──────────────────────────────────────────────────────

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

let toastTimer = 0;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove('on'), 2200);
}

function setStatus(msg, isErr = false) {
  el.status.textContent = msg;
  el.status.classList.toggle('err', isErr);
}

function setBusy(msg) {
  if (msg) {
    el.busyText.textContent = msg;
    el.busy.classList.add('on');
  } else {
    el.busy.classList.remove('on');
  }
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

function canvasToBlob(canvas, type = 'image/png', quality) {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode image'))), type, quality));
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function copyCanvas(canvas) {
  try {
    if (!navigator.clipboard || typeof ClipboardItem === 'undefined') throw new Error('unsupported');
    // Passing the promise keeps Safari's user-gesture requirement satisfied.
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': canvasToBlob(canvas) })]);
    toast('Copied to clipboard');
  } catch {
    toast('This browser blocked copying images. Use Download instead.');
  }
}

const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');

// ── Mask maths (Float32Array, one value 0..1 per pixel) ──────────────────

// Separable box blur, run twice (close to a gaussian). Pure JS so it behaves
// the same in every browser, including ones without canvas filters.
function boxBlur(src, w, h, r) {
  r = Math.round(r);
  if (r < 1) return src.slice();
  let a = src.slice();
  let b = new Float32Array(a.length);
  const inv = 1 / (2 * r + 1);
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 0; y < h; y++) {               // horizontal
      const row = y * w;
      let acc = 0;
      for (let k = -r; k <= r; k++) acc += a[row + Math.min(w - 1, Math.max(0, k))];
      for (let x = 0; x < w; x++) {
        b[row + x] = acc * inv;
        acc += a[row + Math.min(w - 1, x + r + 1)] - a[row + Math.max(0, x - r)];
      }
    }
    for (let x = 0; x < w; x++) {               // vertical
      let acc = 0;
      for (let k = -r; k <= r; k++) acc += b[Math.min(h - 1, Math.max(0, k)) * w + x];
      for (let y = 0; y < h; y++) {
        a[y * w + x] = acc * inv;
        acc += b[Math.min(h - 1, y + r + 1) * w + x] - b[Math.max(0, y - r) * w + x];
      }
    }
  }
  return a;
}

// Flood fill from (sx, sy) over pixels where bin[i] is 1. Returns a 0/1 mask.
function floodFill(bin, w, h, sx, sy) {
  const out = new Uint8Array(w * h);
  const start = sy * w + sx;
  if (!bin[start]) return out;
  const stack = new Int32Array(w * h);
  let sp = 0;
  stack[sp++] = start;
  out[start] = 1;
  while (sp) {
    const i = stack[--sp];
    const x = i % w;
    if (x > 0 && bin[i - 1] && !out[i - 1]) { out[i - 1] = 1; stack[sp++] = i - 1; }
    if (x < w - 1 && bin[i + 1] && !out[i + 1]) { out[i + 1] = 1; stack[sp++] = i + 1; }
    if (i >= w && bin[i - w] && !out[i - w]) { out[i - w] = 1; stack[sp++] = i - w; }
    if (i < w * (h - 1) && bin[i + w] && !out[i + w]) { out[i + w] = 1; stack[sp++] = i + w; }
  }
  return out;
}

// Keep only the soft region connected to the tap. Small gaps (a thin dark
// strap, a sliver of background between arm and body) are bridged first so a
// person doesn't lose an arm, while separate people stay separate.
function isolateAt(conf, w, h, px, py) {
  const n = w * h;
  const hard = new Float32Array(n);
  for (let i = 0; i < n; i++) hard[i] = conf[i] > 0.5 ? 1 : 0;
  const bridge = Math.max(2, Math.round(Math.max(w, h) / 500));
  const spread = boxBlur(hard, w, h, bridge);
  const bin = new Uint8Array(n);
  for (let i = 0; i < n; i++) bin[i] = spread[i] > 0.12 ? 1 : 0;
  const comp = floodFill(bin, w, h, px, py);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = comp[i] ? conf[i] : 0;
  return out;
}

// Fill enclosed holes smaller than `share` of the subject (a missed patch of
// shirt between crossed arms). Bigger holes, like the gap between an arm on a
// hip and the body, are real background and stay open.
function fillSmallHoles(mask, w, h, share = 0.015) {
  const n = w * h;
  const limit = maskArea(mask) * share;
  const label = new Int32Array(n);               // 0 = unvisited background
  const stack = new Int32Array(n);
  const out = mask.slice();
  let next = 1;
  for (let s = 0; s < n; s++) {
    if (mask[s] > 0.5 || label[s]) continue;
    let sp = 0, area = 0, edge = false;
    const members = [];
    stack[sp++] = s;
    label[s] = next;
    while (sp) {
      const i = stack[--sp];
      area++;
      if (area <= limit) members.push(i);
      const x = i % w;
      if (x === 0 || x === w - 1 || i < w || i >= n - w) edge = true;
      if (x > 0 && mask[i - 1] <= 0.5 && !label[i - 1]) { label[i - 1] = next; stack[sp++] = i - 1; }
      if (x < w - 1 && mask[i + 1] <= 0.5 && !label[i + 1]) { label[i + 1] = next; stack[sp++] = i + 1; }
      if (i >= w && mask[i - w] <= 0.5 && !label[i - w]) { label[i - w] = next; stack[sp++] = i - w; }
      if (i < n - w && mask[i + w] <= 0.5 && !label[i + w]) { label[i + w] = next; stack[sp++] = i + w; }
    }
    if (!edge && area <= limit) for (const i of members) out[i] = 1;
    next++;
  }
  return out;
}

function maskArea(m) {
  let s = 0;
  for (let i = 0; i < m.length; i++) if (m[i] > 0.5) s++;
  return s;
}

// Edge softness = feather radius in px; grow shifts the edge in or out.
function refineMask(m, w, h, feather, grow) {
  let a = m;
  if (grow !== 0) {
    const r = Math.max(2, Math.round(Math.max(w, h) / 600));
    const blurred = boxBlur(a, w, h, r);
    const lo = 0.5 - grow - 0.2;
    const hi = 0.5 - grow + 0.2;
    a = new Float32Array(blurred.length);
    for (let i = 0; i < a.length; i++) {
      const t = clamp01((blurred[i] - lo) / (hi - lo));
      a[i] = t * t * (3 - 2 * t);
    }
  }
  if (feather > 0) a = boxBlur(a, w, h, feather * Math.max(1, Math.max(w, h) / 1600));
  return a;
}

// ── Model loading ────────────────────────────────────────────────────────

let models = null;      // { person, touch, personIndex }
let modelsPromise = null;

async function fetchModel(url, onBytes) {
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`Model download failed (${res.status})`);
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onBytes(buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const parts = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.length;
    onBytes(value.length);
  }
  const buf = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  return buf;
}

function loadModels(onProgress) {
  if (modelsPromise) return modelsPromise;
  modelsPromise = (async () => {
    const expected = MODELS.person.size + MODELS.touch.size;
    let got = 0;
    const tick = (n) => { got += n; onProgress(Math.min(99, Math.round((got / expected) * 100))); };
    const [vision, personBuf, touchBuf] = await Promise.all([
      import(`${MP_BASE}/vision_bundle.mjs`),
      fetchModel(MODELS.person.url, tick),
      fetchModel(MODELS.touch.url, tick),
    ]);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const person = await vision.ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: personBuf, delegate: 'CPU' },
      runningMode: 'IMAGE',
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
    const touch = await vision.InteractiveSegmenterLegacy.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: touchBuf, delegate: 'CPU' },
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
    const personIndex = person.getLabels().indexOf('person');
    if (personIndex < 0) throw new Error('Person class missing from the segmentation model');
    models = { person, touch, personIndex };
    return models;
  })();
  modelsPromise.catch(() => { modelsPromise = null; });
  return modelsPromise;
}

// ── Source photo state ───────────────────────────────────────────────────

const src = { canvas: null, ctx: null, w: 0, h: 0, pixels: null, id: 0 };
let sel = null;              // current raw selection mask
let selVersion = 0;
let refined = null;          // { key, cut } cached refined cutout of `sel`
let tapMode = 'new';
let working = false;
let animId = 0;

async function decodeImage(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    // Some formats (HEIC in Safari) decode through <img> but not createImageBitmap.
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

async function openFile(file) {
  if (!file || working) return;
  if (file.type && !file.type.startsWith('image/')) { toast('That file is not an image.'); return; }
  let img;
  try {
    img = await decodeImage(file);
  } catch {
    toast("This browser can't open that image. Try a JPG or PNG.");
    return;
  }
  const iw = img.width || img.naturalWidth;
  const ih = img.height || img.naturalHeight;
  const k = Math.min(1, MAX_SIDE / Math.max(iw, ih));
  const w = Math.round(iw * k);
  const h = Math.round(ih * k);
  const c = makeCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  if (img.close) img.close();

  Object.assign(src, { canvas: c, ctx, w, h, pixels: ctx.getImageData(0, 0, w, h).data, id: src.id + 1 });
  el.base.width = w; el.base.height = h;
  el.base.getContext('2d').drawImage(c, 0, 0);
  el.overlay.width = w; el.overlay.height = h;
  clearSelection();

  el.drop.hidden = true;
  el.stage.hidden = false;
  el.tools.hidden = false;

  await prepare();
}

// Load the models the first time a photo is opened.
async function prepare() {
  if (models) { setStatus('Ready. Tap a person to lift them.'); return; }
  working = true;
  try {
    setBusy('Downloading AI models… 0%');
    await loadModels((p) => setBusy(`Downloading AI models… ${p}%`));
    setStatus('Ready. Tap a person to lift them.');
  } catch (err) {
    console.error(err);
    setStatus("Couldn't load the AI models. Check your connection and reopen the photo.", true);
  } finally {
    setBusy('');
    working = false;
  }
}

function bboxOf(mask, w) {
  let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0.5) {
      const x = i % w;
      const y = (i - x) / w;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

// A square crop centred on a box, `scale` times its longer side and at least
// `minSide`, shifted to stay inside the photo.
function squareAround(b, scale, minSide) {
  const { w, h } = src;
  let side = Math.max(b.x1 - b.x0 + 1, b.y1 - b.y0 + 1) * scale;
  side = Math.max(side, minSide);
  const cw = Math.round(Math.min(side, w));
  const ch = Math.round(Math.min(side, h));
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const x0 = Math.round(Math.min(Math.max(0, cx - cw / 2), w - cw));
  const y0 = Math.round(Math.min(Math.max(0, cy - ch / 2), h - ch));
  return { x0, y0, cw, ch };
}

// DeepLab on a rectangle of the photo, as full-size masks (0 outside the
// rectangle): `person` is the person class, `thing` is anything that isn't
// background (people, pets, cars, furniture, ...). The model sees a 257 px
// square, so a long rectangle is covered with overlapping square tiles rather
// than squashed.
function sceneInRect(rect) {
  const n = src.w * src.h;
  const person = new Float32Array(n);
  const thing = new Float32Array(n);
  const side = Math.min(rect.cw, rect.ch);
  const tiles = Math.ceil(Math.max(rect.cw, rect.ch) / side - 0.15);
  for (let t = 0; t < tiles; t++) {
    const off = tiles === 1 ? 0 : Math.round((t * (Math.max(rect.cw, rect.ch) - side)) / (tiles - 1));
    const tile = rect.cw >= rect.ch
      ? { x0: rect.x0 + off, y0: rect.y0, cw: tiles === 1 ? rect.cw : side, ch: rect.ch }
      : { x0: rect.x0, y0: rect.y0 + off, cw: rect.cw, ch: tiles === 1 ? rect.ch : side };
    sceneTile(tile, person, thing);
  }
  return { person, thing };
}

function sceneTile({ x0, y0, cw, ch }, person, thing) {
  const c = makeCanvas(cw, ch);
  c.getContext('2d').drawImage(src.canvas, x0, y0, cw, ch, 0, 0, cw, ch);
  const res = models.person.segment(c);
  const pm = res.confidenceMasks[models.personIndex].getAsFloat32Array();
  const bg = res.confidenceMasks[0].getAsFloat32Array();
  for (let y = 0; y < ch; y++) {
    const row = (y + y0) * src.w + x0;
    for (let x = 0; x < cw; x++) {
      const j = y * cw + x;
      if (pm[j] > person[row + x]) person[row + x] = pm[j];
      if (1 - bg[j] > thing[row + x]) thing[row + x] = 1 - bg[j];
    }
  }
  res.close();
}

// The most solid pixel of a soft mask: a good place to tap it again.
function solidPoint(mask, w, h) {
  const solid = boxBlur(mask, w, h, Math.max(2, Math.round(Math.max(w, h) / 150)));
  let best = -1, bestV = -1;
  for (let i = 0; i < solid.length; i++) if (mask[i] > 0.5 && solid[i] > bestV) { bestV = solid[i]; best = i; }
  return best < 0 ? null : { x: best % w, y: (best / w) | 0 };
}

function touchMask(x, y) {
  const res = models.touch.segment(src.canvas, { keypoint: { x: (x + 0.5) / src.w, y: (y + 0.5) / src.h } });
  const m = res.confidenceMasks[0].getAsFloat32Array().slice();
  res.close();
  return m;
}

function overlapShare(part, whole) {
  let both = 0, all = 0;
  for (let i = 0; i < whole.length; i++) {
    if (whole[i] > 0.5) { all++; if (part[i] > 0.5) both++; }
  }
  return all ? both / all : 1;
}

// Keep the region only near pixels the scene model agrees are subject,
// keeping Magic Touch's crisp edges. Trimming more than a quarter means the
// two models disagree about the subject itself, so then trust Magic Touch.
function trimTo(region, support, px, py, fallback) {
  const { w, h } = src;
  const hard = new Float32Array(support.length);
  for (let i = 0; i < support.length; i++) hard[i] = support[i] > 0.35 ? 1 : 0;
  const zone = boxBlur(hard, w, h, Math.max(3, Math.round(Math.max(w, h) / 60)));
  const gated = new Float32Array(region.length);
  for (let i = 0; i < gated.length; i++) gated[i] = region[i] * clamp01(zone[i] * 4);
  if (maskArea(gated) < 0.75 * maskArea(region)) return region;
  const at = gated[py * w + px] > 0.5 ? { x: px, y: py } : fallback;
  return at ? isolateAt(gated, w, h, at.x, at.y) : gated;
}

// Lift whatever is under (px, py).
function regionAtPx(px, py) {
  const { w, h } = src;
  const big = Math.max(w, h);
  const raw = touchMask(px, py);
  if (raw[py * w + px] <= 0.5) return raw;
  let region = isolateAt(raw, w, h, px, py);

  // What did we hit? Look at a generous crop, so a face still comes with a body.
  const wide = sceneInRect(squareAround(bboxOf(region, w), 2.2, big * 0.4));
  // Thin limbs are below the scene model's resolution, so a tap on an arm
  // counts as a person tap if there is a person right next to the finger.
  const isPerson = overlapShare(wide.person, region) >= 0.4 || nearMax(wide.person, px, py, Math.round(big / 60)) > 0.5;

  if (isPerson) {
    // Tapped part of a person (face, shirt): tap again on the person's most
    // solid point and merge, so the whole person lifts.
    const at = solidPoint(maskAnd(wide.person, region), w, h) || nearestAbove(wide.person, px, py, Math.round(big / 60));
    const person = at ? isolateAt(wide.person, w, h, at.x, at.y) : null;
    const seed = person && solidPoint(person, w, h);
    if (seed && overlapShare(region, person) < 0.7) {
      const more = touchMask(seed.x, seed.y);
      if (more[seed.y * w + seed.x] > 0.5) {
        const extra = isolateAt(more, w, h, seed.x, seed.y);
        for (let i = 0; i < region.length; i++) if (extra[i] > region[i]) region[i] = extra[i];
      }
    }
    const tight = sceneInRect(squareAround(bboxOf(region, w), 1.25, big * 0.25));
    return trimTo(region, tight.person, px, py, seed);
  }

  if (overlapShare(wide.thing, region) >= 0.4) {
    // A pet, a car, a chair...: the scene model knows it, so it can trim too.
    const tight = sceneInRect(squareAround(bboxOf(region, w), 1.25, big * 0.25));
    return trimTo(region, tight.thing, px, py, solidPoint(maskAnd(tight.thing, region), w, h));
  }
  return region;                                   // anything else: Magic Touch alone
}

function nearestAbove(mask, px, py, r) {
  const { w, h } = src;
  let best = null, bestD = Infinity;
  for (let y = Math.max(0, py - r); y <= Math.min(h - 1, py + r); y++) {
    for (let x = Math.max(0, px - r); x <= Math.min(w - 1, px + r); x++) {
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (mask[y * w + x] > 0.5 && d < bestD) { bestD = d; best = { x, y }; }
    }
  }
  return best;
}

function nearMax(mask, px, py, r) {
  const { w, h } = src;
  let m = 0;
  for (let y = Math.max(0, py - r); y <= Math.min(h - 1, py + r); y++) {
    for (let x = Math.max(0, px - r); x <= Math.min(w - 1, px + r); x++) {
      if (mask[y * w + x] > m) m = mask[y * w + x];
    }
  }
  return m;
}

function maskAnd(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.min(a[i], b[i]);
  return out;
}

function regionAt(nx, ny) {
  const px = Math.min(src.w - 1, Math.max(0, Math.floor(nx * src.w)));
  const py = Math.min(src.h - 1, Math.max(0, Math.floor(ny * src.h)));
  return solidify(fillSmallHoles(regionAtPx(px, py), src.w, src.h));
}

// Magic Touch is sometimes only half sure inside the subject (hands, dark
// fabric), which reads as see-through. Push confident-ish values to solid and
// leave only real edges soft; the Edge softness slider adds softness back.
function solidify(m) {
  const out = new Float32Array(m.length);
  for (let i = 0; i < m.length; i++) out[i] = clamp01((m[i] - 0.25) / 0.5);
  return out;
}

// Find everyone: the person model over square tiles covering the photo (so
// nobody is squashed), one seed point per person, each lifted like a tap.
function peopleSeeds() {
  const { w, h } = src;
  const p = sceneInRect({ x0: 0, y0: 0, cw: w, ch: h }).person;
  const bin = new Uint8Array(p.length);
  for (let i = 0; i < p.length; i++) bin[i] = p[i] > 0.5 ? 1 : 0;
  const seen = new Uint8Array(p.length);
  const minArea = p.length * 0.002;
  const seeds = [];
  for (let i = 0; i < p.length; i++) {
    if (!bin[i] || seen[i]) continue;
    const comp = floodFill(bin, w, h, i % w, (i / w) | 0);
    let area = 0;
    const soft = new Float32Array(p.length);
    for (let j = 0; j < comp.length; j++) if (comp[j]) { seen[j] = 1; area++; soft[j] = p[j]; }
    if (area < minArea) continue;
    const s = solidPoint(soft, w, h);
    if (s) seeds.push({ ...s, area, soft });
  }
  return seeds.sort((a, b) => b.area - a.area).slice(0, 8);
}

async function onTap(e) {
  if (!models || !src.canvas || working) return;
  const r = el.base.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width;
  const ny = (e.clientY - r.top) / r.height;
  if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return;
  const mode = e.shiftKey ? 'add' : e.altKey ? 'sub' : tapMode;
  if (mode === 'sub' && !sel) return;

  working = true;
  setBusy('Lifting…');
  await nextFrame();
  try {
    const region = regionAt(nx, ny);
    if (maskArea(region) < MIN_PIXELS) {
      setStatus('Nothing to lift there. Try tapping closer to the middle of the subject.');
      return;
    }
    let next;
    if (mode === 'add' && sel) {
      next = new Float32Array(sel.length);
      for (let i = 0; i < sel.length; i++) next[i] = Math.max(sel[i], region[i]);
    } else if (mode === 'sub') {
      next = new Float32Array(sel.length);
      for (let i = 0; i < sel.length; i++) next[i] = sel[i] * (1 - region[i]);
    } else {
      next = region;
    }
    if (maskArea(next) < MIN_PIXELS) { clearSelection(); setStatus('Selection cleared.'); return; }
    setSelection(next, mode !== 'sub');
    setStatus(mode === 'add' ? 'Added to the selection.' : mode === 'sub' ? 'Removed from the selection.' : 'Lifted.');
  } catch (err) {
    console.error(err);
    setStatus('Something went wrong while lifting. Try again.', true);
  } finally {
    setBusy('');
    working = false;
  }
}

async function selectAllPeople() {
  if (!models || !src.canvas || working) return;
  working = true;
  setBusy('Finding everyone…');
  await nextFrame();
  try {
    const seeds = peopleSeeds();
    if (!seeds.length) { setStatus('No people found. Tap a person to lift them.'); return; }
    const { w, h } = src;
    const all = new Float32Array(w * h);
    let count = 0;
    for (const s of seeds) {
      if (all[s.y * w + s.x] > 0.5) continue;      // already covered by an earlier lift
      let r = solidify(fillSmallHoles(regionAtPx(s.x, s.y), w, h));
      if (maskArea(r) < MIN_PIXELS) r = s.soft;
      for (let i = 0; i < all.length; i++) if (r[i] > all[i]) all[i] = r[i];
      count++;
    }
    setSelection(all, true);
    setStatus(count === 1 ? 'Lifted the person.' : `Lifted ${count} people.`);
  } catch (err) {
    console.error(err);
    setStatus('Something went wrong while lifting. Try again.', true);
  } finally {
    setBusy('');
    working = false;
  }
}

function setSelection(mask, shimmer) {
  sel = mask;
  selVersion++;
  refined = null;
  el.photo.classList.add('has-sel');
  el.actions.classList.add('on');
  drawSelection(shimmer);
}

function clearSelection() {
  sel = null;
  selVersion++;
  refined = null;
  cancelAnimationFrame(animId);
  el.photo.classList.remove('has-sel');
  el.actions.classList.remove('on');
  if (el.overlay.width) el.overlay.getContext('2d').clearRect(0, 0, el.overlay.width, el.overlay.height);
}

// Cut the selection out of the photo: RGBA canvas cropped to the subject.
function currentCutout() {
  const feather = +el.feather.value;
  const grow = +el.grow.value;
  const key = `${selVersion}|${feather}|${grow}`;
  if (refined && refined.key === key) return refined.cut;
  const { w, h, pixels } = src;
  const a = refineMask(sel, w, h, feather, grow);
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (a[row + x] > 0.02) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;
  const pad = 2;
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
  x1 = Math.min(w - 1, x1 + pad); y1 = Math.min(h - 1, y1 + pad);
  const cw = x1 - x0 + 1;
  const ch = y1 - y0 + 1;
  const c = makeCanvas(cw, ch);
  const ctx = c.getContext('2d');
  const out = ctx.createImageData(cw, ch);
  const d = out.data;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const si = (y + y0) * w + (x + x0);
      const di = (y * cw + x) * 4;
      d[di] = pixels[si * 4];
      d[di + 1] = pixels[si * 4 + 1];
      d[di + 2] = pixels[si * 4 + 2];
      d[di + 3] = Math.round(pixels[si * 4 + 3] * clamp01(a[si]));
    }
  }
  ctx.putImageData(out, 0, 0);
  const cut = { canvas: c, x0, y0, w: cw, h: ch, srcW: w, srcH: h };
  refined = { key, cut };
  return cut;
}

// iOS-style lift: dim the photo, glow the subject, sweep a light across it.
function drawSelection(shimmer) {
  const cut = currentCutout();
  const ctx = el.overlay.getContext('2d');
  cancelAnimationFrame(animId);
  if (!cut) { ctx.clearRect(0, 0, el.overlay.width, el.overlay.height); return; }
  const { w, h } = src;
  const plain = () => { ctx.clearRect(0, 0, w, h); ctx.drawImage(cut.canvas, cut.x0, cut.y0); };
  if (!shimmer || reducedMotion) { plain(); return; }
  const t0 = performance.now();
  const dur = 950;
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    plain();
    if (p < 1) {
      const c = -0.2 + p * 1.4;   // band centre travels past both ends
      const band = 0.14;
      const g = ctx.createLinearGradient(cut.x0, cut.y0, cut.x0 + cut.w, cut.y0 + cut.h);
      g.addColorStop(clamp01(c - band), 'rgba(255,255,255,0)');
      g.addColorStop(clamp01(c), 'rgba(255,255,255,0.55)');
      g.addColorStop(clamp01(c + band), 'rgba(255,255,255,0)');
      ctx.save();
      ctx.globalCompositeOperation = 'source-atop';
      ctx.fillStyle = g;
      ctx.fillRect(cut.x0, cut.y0, cut.w, cut.h);
      ctx.restore();
      animId = requestAnimationFrame(step);
    }
  };
  animId = requestAnimationFrame(step);
}

// ── Tray of lifted subjects ──────────────────────────────────────────────

const tray = [];   // { id, cut, url }
let trayId = 0;

async function addToTray(cut) {
  const blob = await canvasToBlob(cut.canvas);
  const item = { id: ++trayId, cut, url: URL.createObjectURL(blob) };
  tray.push(item);
  const chip = document.createElement('div');
  chip.className = 'lift-chip checker';
  chip.dataset.id = item.id;
  chip.title = 'Add to the new image again';
  chip.tabIndex = 0;
  chip.setAttribute('role', 'button');
  chip.innerHTML = `<img alt="Lifted subject ${item.id}" src="${item.url}"><button type="button" class="x" aria-label="Remove from tray">&times;</button>`;
  el.tray.appendChild(chip);
  return item;
}

el.tray.addEventListener('click', (e) => {
  const chip = e.target.closest('.lift-chip');
  if (!chip) return;
  const idx = tray.findIndex((t) => t.id === +chip.dataset.id);
  if (idx < 0) return;
  if (e.target.closest('.x')) {
    URL.revokeObjectURL(tray[idx].url);
    tray.splice(idx, 1);
    chip.remove();
    return;
  }
  addLayer(tray[idx].cut);
});
el.tray.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('lift-chip')) {
    e.preventDefault();
    e.target.click();
  }
});

// ── Composer ─────────────────────────────────────────────────────────────

const comp = {
  W: 1080, H: 1080,
  sized: false,                 // becomes true once the canvas has a real size
  bgMode: 'gradient',
  bgImage: null,                // canvas used by the photo backgrounds
  bgBlurCache: null,
  layers: [],
  selected: null,
};
let layerId = 0;
const cctx = el.compose.getContext('2d');

function photoSize() {
  const p = comp.bgImage || src.canvas;
  return p ? { W: p.width, H: p.height } : { W: 1080, H: 1080 };
}

function resizeCompose(W, H) {
  const sx = W / comp.W;
  const sy = H / comp.H;
  for (const L of comp.layers) {
    const grounded = Math.abs(L.y + (L.cut.h * L.scale) / 2 - comp.H) < 2 && L.rot === 0;
    L.x *= sx; L.y *= sy;
    L.scale *= Math.min(sx, sy);
    // A person cut off by the bottom of their photo stays standing on the bottom edge.
    if (grounded) L.y = H - (L.cut.h * L.scale) / 2;
  }
  comp.W = W; comp.H = H;
  el.compose.width = W; el.compose.height = H;
  comp.bgBlurCache = null;
  comp.sized = true;
  renderCompose();
}

function applySizeSelect() {
  const v = el.size.value;
  const s = v === 'photo' ? photoSize() : (([W, H]) => ({ W, H }))(v.split('x').map(Number));
  resizeCompose(s.W, s.H);
}

function itemAlpha(cut) {
  if (!cut.alpha) {
    const d = cut.canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, cut.w, cut.h).data;
    const a = new Uint8Array(cut.w * cut.h);
    for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + 3];
    cut.alpha = a;
  }
  return cut.alpha;
}

function stickerFor(cut) {
  if (cut.sticker) return cut.sticker;
  const r = Math.max(6, Math.round(Math.max(cut.w, cut.h) * 0.022));
  const c = makeCanvas(cut.w + 2 * r, cut.h + 2 * r);
  const ctx = c.getContext('2d');
  const steps = 24;
  for (let i = 0; i < steps; i++) {
    const t = (i / steps) * Math.PI * 2;
    ctx.drawImage(cut.canvas, r + Math.cos(t) * r, r + Math.sin(t) * r);
  }
  ctx.drawImage(cut.canvas, r, r);
  ctx.globalCompositeOperation = 'source-in';
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  cut.sticker = { canvas: c, r };
  return cut.sticker;
}

function addLayer(cut) {
  if (!comp.sized || comp.layers.length === 0) {
    if (el.size.value === 'photo') {
      const s = { W: cut.srcW, H: cut.srcH };
      if (!comp.sized || s.W !== comp.W || s.H !== comp.H) resizeCompose(s.W, s.H);
    } else if (!comp.sized) {
      applySizeSelect();
    }
  }
  let x, y, scale;
  if (cut.srcW === comp.W && cut.srcH === comp.H) {
    // Same frame as the original photo: drop it exactly where it was.
    x = cut.x0 + cut.w / 2;
    y = cut.y0 + cut.h / 2;
    scale = 1;
  } else {
    scale = Math.min(1.5, (comp.H * 0.8) / cut.h, (comp.W * 0.8) / cut.w);
    x = comp.W / 2;
    // Cut off by the bottom of the photo? Stand it on the bottom edge.
    const grounded = cut.y0 + cut.h >= cut.srcH - 2;
    y = comp.H - (cut.h * scale) / 2 - (grounded ? 0 : comp.H * 0.04);
  }
  // Nudge copies so a second add is visibly a second layer.
  const dupes = comp.layers.filter((L) => L.cut === cut).length;
  x += dupes * comp.W * 0.06;
  const L = { id: ++layerId, cut, x, y, scale, rot: 0, flip: false, shadow: false, sticker: false };
  comp.layers.push(L);
  selectLayer(L);
  renderCompose();
  el.compose.scrollIntoView?.({ block: 'nearest', behavior: reducedMotion ? 'auto' : 'smooth' });
  return L;
}

function drawCover(ctx, img, W, H) {
  const k = Math.max(W / img.width, H / img.height);
  const w = img.width * k;
  const h = img.height * k;
  ctx.drawImage(img, (W - w) / 2, (H - h) / 2, w, h);
}

function blurredBackground() {
  const { W, H } = comp;
  if (comp.bgBlurCache && comp.bgBlurCache.src === comp.bgImage) return comp.bgBlurCache.canvas;
  // Downscale then upscale with smoothing: a soft blur that works everywhere.
  const small = makeCanvas(Math.max(8, W / 24), Math.max(8, H / 24));
  const sctx = small.getContext('2d');
  sctx.imageSmoothingQuality = 'high';
  drawCover(sctx, comp.bgImage, small.width, small.height);
  const mid = makeCanvas(Math.max(16, W / 6), Math.max(16, H / 6));
  const mctx = mid.getContext('2d');
  mctx.imageSmoothingQuality = 'high';
  mctx.drawImage(small, 0, 0, mid.width, mid.height);
  const big = makeCanvas(W, H);
  const bctx = big.getContext('2d');
  bctx.imageSmoothingQuality = 'high';
  bctx.drawImage(mid, 0, 0, W, H);
  comp.bgBlurCache = { src: comp.bgImage, canvas: big };
  return big;
}

function drawBackground(ctx, forJpeg) {
  const { W, H } = comp;
  const mode = comp.bgMode;
  if ((mode === 'photo' || mode === 'photo-blur' || mode === 'upload') && comp.bgImage) {
    if (mode === 'photo-blur') ctx.drawImage(blurredBackground(), 0, 0);
    else drawCover(ctx, comp.bgImage, W, H);
  } else if (mode === 'color') {
    ctx.fillStyle = el.bgColor.value;
    ctx.fillRect(0, 0, W, H);
  } else if (mode === 'gradient') {
    const g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, el.bgColor.value);
    g.addColorStop(1, el.bgColor2.value);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  } else if (forJpeg) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);
  }
}

function drawLayer(ctx, L) {
  const { cut } = L;
  ctx.save();
  ctx.translate(L.x, L.y);
  ctx.rotate(L.rot);
  ctx.scale(L.flip ? -L.scale : L.scale, L.scale);
  if (L.shadow) {
    const size = Math.max(cut.w, cut.h) * L.scale;
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = size * 0.05;
    ctx.shadowOffsetY = size * 0.025;
  }
  if (L.sticker) {
    const s = stickerFor(cut);
    ctx.drawImage(s.canvas, -cut.w / 2 - s.r, -cut.h / 2 - s.r);
    ctx.shadowColor = 'transparent';
  }
  ctx.drawImage(cut.canvas, -cut.w / 2, -cut.h / 2);
  ctx.restore();
}

// Canvas pixels per CSS pixel, so handles stay finger-sized at any zoom.
function pxRatio() {
  const r = el.compose.getBoundingClientRect();
  return r.width ? comp.W / r.width : 1;
}

function handlePos(L) {
  const hx = (L.cut.w / 2) * L.scale;
  const hy = (-L.cut.h / 2) * L.scale;
  const c = Math.cos(L.rot);
  const s = Math.sin(L.rot);
  // Keep it on the canvas so it can always be grabbed.
  const m = 14 * pxRatio();
  return {
    x: Math.min(comp.W - m, Math.max(m, L.x + hx * c - hy * s)),
    y: Math.min(comp.H - m, Math.max(m, L.y + hx * s + hy * c)),
  };
}

function drawSelectionUI(ctx, L) {
  const k = pxRatio();
  ctx.save();
  ctx.translate(L.x, L.y);
  ctx.rotate(L.rot);
  const w = L.cut.w * L.scale;
  const h = L.cut.h * L.scale;
  ctx.lineWidth = 1.5 * k;
  ctx.setLineDash([6 * k, 5 * k]);
  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.strokeRect(-w / 2, -h / 2, w, h);
  ctx.strokeStyle = 'rgba(124,58,237,0.95)';
  ctx.lineDashOffset = 5.5 * k;
  ctx.strokeRect(-w / 2, -h / 2, w, h);
  ctx.restore();
  const hp = handlePos(L);
  ctx.save();
  ctx.beginPath();
  ctx.arc(hp.x, hp.y, 11 * k, 0, Math.PI * 2);
  ctx.fillStyle = '#7c3aed';
  ctx.fill();
  ctx.lineWidth = 2.5 * k;
  ctx.strokeStyle = '#fff';
  ctx.stroke();
  // little rotate/resize glyph
  ctx.beginPath();
  ctx.arc(hp.x, hp.y, 5 * k, -Math.PI * 0.9, Math.PI * 0.4);
  ctx.lineWidth = 1.6 * k;
  ctx.stroke();
  ctx.restore();
}

function render(ctx, { ui = false, forJpeg = false } = {}) {
  ctx.clearRect(0, 0, comp.W, comp.H);
  drawBackground(ctx, forJpeg);
  for (const L of comp.layers) drawLayer(ctx, L);
  if (ui && comp.selected) drawSelectionUI(ctx, comp.selected);
}

let renderQueued = false;
function renderCompose() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    render(cctx, { ui: true });
    const has = comp.layers.length > 0;
    el.composeEmpty.hidden = has;
    el.exp.disabled = !has;
    el.expCopy.disabled = !has;
    el.composeWrap.classList.toggle('checker', comp.bgMode === 'none');
  });
}

function selectLayer(L) {
  comp.selected = L;
  el.layerProps.classList.toggle('on', !!L);
  if (L) {
    el.lScale.value = L.scale;
    el.lRot.value = Math.round((L.rot * 180) / Math.PI);
    el.lShadow.checked = L.shadow;
    el.lSticker.checked = L.sticker;
  }
  renderCompose();
}

function toCanvasPoint(e) {
  const r = el.compose.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * comp.W, y: ((e.clientY - r.top) / r.height) * comp.H };
}

function hitLayer(p) {
  for (let i = comp.layers.length - 1; i >= 0; i--) {
    const L = comp.layers[i];
    const dx = p.x - L.x;
    const dy = p.y - L.y;
    const c = Math.cos(-L.rot);
    const s = Math.sin(-L.rot);
    let lx = (dx * c - dy * s) / L.scale;
    const ly = (dx * s + dy * c) / L.scale;
    if (L.flip) lx = -lx;
    const ix = Math.floor(lx + L.cut.w / 2);
    const iy = Math.floor(ly + L.cut.h / 2);
    if (ix < 0 || iy < 0 || ix >= L.cut.w || iy >= L.cut.h) continue;
    // The selected layer can be grabbed anywhere in its box; others need a hit
    // on an opaque pixel so you can reach a layer that sits behind.
    if (L === comp.selected || itemAlpha(L.cut)[iy * L.cut.w + ix] > 24) return L;
  }
  return null;
}

const pointers = new Map();
let gesture = null;

function beginGesture() {
  const L = comp.selected;
  if (!L) { gesture = null; return; }
  const pts = [...pointers.values()];
  if (pts.length >= 2) {
    const [a, b] = pts;
    gesture = {
      type: 'pinch', L, scale: L.scale, rot: L.rot, x: L.x, y: L.y,
      d: Math.hypot(b.x - a.x, b.y - a.y) || 1,
      ang: Math.atan2(b.y - a.y, b.x - a.x),
      mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
    };
  }
}

el.compose.addEventListener('pointerdown', (e) => {
  const p = toCanvasPoint(e);
  pointers.set(e.pointerId, p);
  el.compose.setPointerCapture(e.pointerId);
  if (pointers.size >= 2) { beginGesture(); return; }

  const L = comp.selected;
  if (L) {
    const hp = handlePos(L);
    if (Math.hypot(p.x - hp.x, p.y - hp.y) <= 18 * pxRatio()) {
      gesture = {
        type: 'handle', L, scale: L.scale, rot: L.rot,
        d: Math.hypot(p.x - L.x, p.y - L.y) || 1,
        ang: Math.atan2(p.y - L.y, p.x - L.x),
      };
      return;
    }
  }
  const hit = hitLayer(p);
  selectLayer(hit);
  gesture = hit ? { type: 'move', L: hit, dx: p.x - hit.x, dy: p.y - hit.y } : null;
});

el.compose.addEventListener('pointermove', (e) => {
  if (!pointers.has(e.pointerId)) {
    const p = toCanvasPoint(e);
    const L = comp.selected;
    const onHandle = L && Math.hypot(p.x - handlePos(L).x, p.y - handlePos(L).y) <= 18 * pxRatio();
    el.compose.style.cursor = onHandle ? 'nwse-resize' : hitLayer(p) ? 'move' : 'default';
    return;
  }
  const p = toCanvasPoint(e);
  pointers.set(e.pointerId, p);
  if (!gesture) return;
  const { L } = gesture;
  if (gesture.type === 'move') {
    L.x = p.x - gesture.dx;
    L.y = p.y - gesture.dy;
  } else if (gesture.type === 'handle') {
    L.scale = Math.max(0.03, gesture.scale * (Math.hypot(p.x - L.x, p.y - L.y) / gesture.d));
    L.rot = gesture.rot + Math.atan2(p.y - L.y, p.x - L.x) - gesture.ang;
  } else if (gesture.type === 'pinch' && pointers.size >= 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(b.x - a.x, b.y - a.y);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    L.scale = Math.max(0.03, gesture.scale * (d / gesture.d));
    L.rot = gesture.rot + Math.atan2(b.y - a.y, b.x - a.x) - gesture.ang;
    L.x = gesture.x + (mid.x - gesture.mid.x);
    L.y = gesture.y + (mid.y - gesture.mid.y);
  }
  el.lScale.value = L.scale;
  el.lRot.value = Math.round((((L.rot * 180) / Math.PI + 540) % 360) - 180);
  renderCompose();
});

function endPointer(e) {
  pointers.delete(e.pointerId);
  if (pointers.size === 1 && gesture?.type === 'pinch') {
    // Lifting one finger of a pinch continues as a drag with the other.
    const [p] = [...pointers.values()];
    gesture = { type: 'move', L: gesture.L, dx: p.x - gesture.L.x, dy: p.y - gesture.L.y };
  } else if (pointers.size === 0) {
    gesture = null;
  }
}
el.compose.addEventListener('pointerup', endPointer);
el.compose.addEventListener('pointercancel', endPointer);

el.compose.addEventListener('wheel', (e) => {
  const L = comp.selected;
  if (!L) return;
  e.preventDefault();
  L.scale = Math.max(0.03, L.scale * Math.exp(-e.deltaY * 0.0015));
  el.lScale.value = L.scale;
  renderCompose();
}, { passive: false });

document.addEventListener('keydown', (e) => {
  const L = comp.selected;
  if (!L) return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  const step = (e.shiftKey ? 10 : 1) * pxRatio();
  const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  if (moves[e.key]) {
    e.preventDefault();
    L.x += moves[e.key][0];
    L.y += moves[e.key][1];
    renderCompose();
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    e.preventDefault();
    deleteLayer(L);
  } else if (e.key === 'Escape') {
    selectLayer(null);
  }
});

function deleteLayer(L) {
  comp.layers = comp.layers.filter((x) => x !== L);
  selectLayer(null);
}

el.lScale.addEventListener('input', () => { if (comp.selected) { comp.selected.scale = +el.lScale.value; renderCompose(); } });
el.lRot.addEventListener('input', () => { if (comp.selected) { comp.selected.rot = (+el.lRot.value * Math.PI) / 180; renderCompose(); } });
el.lShadow.addEventListener('change', () => { if (comp.selected) { comp.selected.shadow = el.lShadow.checked; renderCompose(); } });
el.lSticker.addEventListener('change', () => { if (comp.selected) { comp.selected.sticker = el.lSticker.checked; renderCompose(); } });
el.lFlip.addEventListener('click', () => { if (comp.selected) { comp.selected.flip = !comp.selected.flip; renderCompose(); } });
el.lDel.addEventListener('click', () => { if (comp.selected) deleteLayer(comp.selected); });
el.lDup.addEventListener('click', () => {
  const L = comp.selected;
  if (!L) return;
  const C = { ...L, id: ++layerId, x: L.x + comp.W * 0.05, y: L.y + comp.H * 0.03 };
  comp.layers.push(C);
  selectLayer(C);
});
el.lUp.addEventListener('click', () => moveZ(1));
el.lDown.addEventListener('click', () => moveZ(-1));
function moveZ(dir) {
  const L = comp.selected;
  if (!L) return;
  const i = comp.layers.indexOf(L);
  const j = Math.min(comp.layers.length - 1, Math.max(0, i + dir));
  if (i === j) return;
  comp.layers.splice(i, 1);
  comp.layers.splice(j, 0, L);
  renderCompose();
}

// Background + size controls
function syncBgControls() {
  const m = comp.bgMode;
  el.bgColorL.hidden = !(m === 'color' || m === 'gradient');
  el.bgColor2L.hidden = m !== 'gradient';
}

el.bg.addEventListener('change', () => {
  const v = el.bg.value;
  if (v === 'upload') {
    el.bgFile.click();
    el.bg.value = comp.bgMode;          // stay on the old choice until a file arrives
    return;
  }
  if ((v === 'photo' || v === 'photo-blur') && !src.canvas) {
    toast('Open a photo first.');
    el.bg.value = comp.bgMode;
    return;
  }
  if (v === 'photo' || v === 'photo-blur') {
    if (comp.bgImage !== src.canvas) { comp.bgImage = src.canvas; comp.bgBlurCache = null; }
  }
  comp.bgMode = v;
  syncBgControls();
  if (el.size.value === 'photo' && comp.bgImage && (v === 'photo' || v === 'photo-blur')) applySizeSelect();
  renderCompose();
});

el.bgFile.addEventListener('change', async () => {
  const f = el.bgFile.files[0];
  el.bgFile.value = '';
  if (!f) return;
  try {
    const img = await decodeImage(f);
    const iw = img.width || img.naturalWidth;
    const ih = img.height || img.naturalHeight;
    const k = Math.min(1, 4096 / Math.max(iw, ih));
    const c = makeCanvas(iw * k, ih * k);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    comp.bgImage = c;
    comp.bgBlurCache = null;
    comp.bgMode = 'upload';
    el.bg.value = 'upload';
    syncBgControls();
    if (el.size.value === 'photo') applySizeSelect();
    renderCompose();
  } catch {
    toast("This browser can't open that image.");
  }
});

[el.bgColor, el.bgColor2].forEach((i) => i.addEventListener('input', renderCompose));
el.size.addEventListener('change', applySizeSelect);

async function exportImage() {
  const jpeg = el.fmt.value === 'jpeg';
  const c = makeCanvas(comp.W, comp.H);
  render(c.getContext('2d'), { forJpeg: jpeg });
  return { canvas: c, jpeg };
}

el.exp.addEventListener('click', async () => {
  const { canvas, jpeg } = await exportImage();
  const blob = await canvasToBlob(canvas, jpeg ? 'image/jpeg' : 'image/png', 0.92);
  downloadBlob(blob, `subject-lift-${stamp()}.${jpeg ? 'jpg' : 'png'}`);
});
el.expCopy.addEventListener('click', async () => {
  const c = makeCanvas(comp.W, comp.H);
  render(c.getContext('2d'));
  copyCanvas(c);
});

// ── Lift panel wiring ────────────────────────────────────────────────────

el.open.addEventListener('click', () => el.file.click());
el.drop.addEventListener('click', () => el.file.click());
el.drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); }
});
el.file.addEventListener('change', () => {
  const f = el.file.files[0];
  el.file.value = '';
  openFile(f);
});

for (const zone of [el.drop, el.stage]) {
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('over');
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) openFile(f);
  });
}

document.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (!item) return;
  e.preventDefault();
  openFile(item.getAsFile());
});

el.base.addEventListener('click', onTap);
el.all.addEventListener('click', selectAllPeople);
el.clear.addEventListener('click', () => { clearSelection(); setStatus('Tap a person to lift them.'); });

document.querySelectorAll('.lift-seg button').forEach((b) => b.addEventListener('click', () => {
  tapMode = b.dataset.mode;
  document.querySelectorAll('.lift-seg button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
}));

let refineTimer = 0;
for (const r of [el.feather, el.grow]) {
  r.addEventListener('input', () => {
    if (!sel) return;
    clearTimeout(refineTimer);
    refineTimer = setTimeout(() => drawSelection(false), 60);
  });
}

el.use.addEventListener('click', async () => {
  if (!sel) return;
  const cut = currentCutout();
  if (!cut) return;
  // Freeze this cutout so later edge tweaks on the selection don't change it.
  const frozen = { ...cut };
  await addToTray(frozen);
  addLayer(frozen);
  toast('Added to the new image');
});
el.copy.addEventListener('click', () => { const c = sel && currentCutout(); if (c) copyCanvas(c.canvas); });
el.save.addEventListener('click', async () => {
  const c = sel && currentCutout();
  if (!c) return;
  downloadBlob(await canvasToBlob(c.canvas), `lifted-${stamp()}.png`);
});

syncBgControls();
renderCompose();

// Exposed for automated tests only.
window.__lift = { comp, src, get sel() { return sel; }, get models() { return models; } };
