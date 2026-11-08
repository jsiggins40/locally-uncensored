// ✂️ Lift for the Photo Edit page: tap a person to lift them out of a photo
// (like on an iPhone), arrange lifted people from one or more photos into a
// new picture, then hand it back to the edit page as the photo for Seedream
// (or any other mode), or as an extra photo.
//
// It runs in the browser, so the GPU stays free for the edit models. Two
// MediaPipe models (Apache-2.0) do the work:
//   * magic_touch: tap-prompted segmentation, the crisp outline under the finger.
//   * deeplab_v3: scene segmentation, used on crops around the subject to tell
//     whether a tap hit a person, to grow a partial lift (a face) to the whole
//     person, and to trim stray background patches.
// edit-restart copies both, and the MediaPipe runtime, into web/lift/ on the
// VM; until it has, they load from the public CDN and Google's storage.
//
// edit.html opens it with: (await import('./lift.js')).openLift({ photo, photoKey, readable, onUse })

const MP_VERSION = '1.1.1';
const HERE = new URL('./lift/', import.meta.url).href;
const LOCAL = {
  bundle: HERE + 'vision_bundle.js',
  wasm: HERE + 'wasm',
  person: HERE + 'deeplab_v3.tflite',
  touch: HERE + 'magic_touch.tflite',
};
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const REMOTE = {
  bundle: `${CDN}/vision_bundle.mjs`,
  wasm: `${CDN}/wasm`,
  person: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite',
  touch: 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite',
};
const MODELS = { person: { size: 2780176 }, touch: { size: 6227884 } };
const MAX_SIDE = 2048;          // same as the edit page's photo
const MIN_PIXELS = 64;          // a selection smaller than this is treated as "nothing found"

let opts = {};                  // from the edit page, see openLift()
let srcFromMain = false;        // the photo being lifted from is the edit page's photo
let mainKey = null;             // ...and which version of it

const STYLE = `
#lf-sheet { position:fixed; inset:0; z-index:1000; background:var(--bg,#111); color:var(--text,#eee); overflow:auto;
  -webkit-overflow-scrolling:touch; padding:12px 16px 40px; }
#lf-sheet [hidden] { display:none !important; }
#lf-sheet .lf-top { display:flex; align-items:center; gap:8px; margin:4px 0 12px; }
#lf-sheet .lf-top h2 { font-size:18px; margin:0; flex:1; }
#lf-sheet .lf-cols { display:block; }
@media (min-width:1100px) { #lf-sheet .lf-cols { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; align-items:start; } }
#lf-sheet .lf-h { font-weight:600; margin:0 0 8px; }
#lf-sheet .lf-row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:8px; }
#lf-sheet .lf-row > button, #lf-sheet .lf-row > .btn { flex:1 1 auto; }
#lf-sheet .lf-seg { display:flex; gap:0; flex:1 1 220px; }
#lf-sheet .lf-seg button { flex:1; border-radius:0; padding:10px 6px; }
#lf-sheet .lf-seg button:first-child { border-radius:10px 0 0 10px; }
#lf-sheet .lf-seg button:last-child { border-radius:0 10px 10px 0; }
#lf-sheet button.on { background:var(--accent,#4f8cff); color:#fff; }
#lf-sheet .lf-drop { border:2px dashed #444; border-radius:12px; padding:40px 10px; text-align:center; color:var(--muted,#999); }
#lf-sheet .lf-stage { position:relative; display:grid; place-items:center; touch-action:manipulation; }
#lf-sheet .lf-photo { position:relative; line-height:0; }
#lf-sheet .lf-photo canvas { display:block; max-width:100%; max-height:62vh; width:auto; height:auto; border-radius:8px; }
#lf-base { cursor:crosshair; transition:filter .35s ease; }
#lf-sheet .lf-photo.has-sel #lf-base { filter:brightness(.42) saturate(.6); }
#lf-overlay { position:absolute; inset:0; width:100% !important; height:100% !important; max-height:none !important; pointer-events:none;
  filter:drop-shadow(0 0 2px rgba(255,255,255,.95)) drop-shadow(0 0 14px rgba(255,255,255,.45)); transition:transform .35s ease; }
#lf-sheet .lf-photo.has-sel #lf-overlay { transform:translateY(-3px) scale(1.006); }
#lf-sheet .lf-busy { position:absolute; inset:0; display:none; place-items:center; background:rgba(0,0,0,.45); color:#fff; font-size:15px; z-index:2; border-radius:8px; }
#lf-sheet .lf-busy.on { display:grid; }
#lf-sheet .lf-spin { width:28px; height:28px; border-radius:50%; border:3px solid rgba(255,255,255,.25); border-top-color:#fff; animation:lfspin .8s linear infinite; margin:0 auto 10px; }
@keyframes lfspin { to { transform:rotate(360deg); } }
#lf-status { min-height:1.4em; color:var(--muted,#999); margin-top:8px; font-size:14px; }
#lf-status.err { color:var(--danger,#ff5a5a); }
#lf-sel-actions { display:none; }
#lf-sel-actions.on { display:block; }
#lf-sheet .lf-sl { display:flex; align-items:center; gap:10px; color:var(--muted,#999); font-size:14px; margin-top:6px; }
#lf-sheet .lf-sl span { flex:none; width:110px; }
#lf-sheet .lf-sl input { flex:1; min-width:0; }
#lf-tray { display:flex; gap:8px; overflow-x:auto; padding:8px 2px 4px; min-height:80px; }
#lf-tray:empty::before { content:"Lifted people collect here. Tap one to add it again."; color:var(--muted,#999); font-size:13px; align-self:center; }
#lf-sheet .lift-chip { position:relative; flex:0 0 auto; width:72px; height:72px; border-radius:8px; border:1px solid #3a3a3c; padding:3px; cursor:pointer; background-color:#2a2a2c; }
#lf-sheet .lift-chip img { width:100%; height:100%; object-fit:contain; display:block; }
#lf-sheet .lift-chip .x { position:absolute; top:-6px; right:-6px; width:22px; height:22px; border-radius:50%; padding:0; font-size:12px; line-height:22px; background:#444; }
#lf-sheet .checker { background-image:conic-gradient(#3a3a3c 25%, #2a2a2c 0 50%, #3a3a3c 0 75%, #2a2a2c 0); background-size:16px 16px; }
#lf-compose-wrap { position:relative; display:grid; place-items:center; border-radius:10px; overflow:hidden; background:#2a2a2c; padding:6px; }
#lf-compose-wrap.over { outline:3px dashed var(--accent,#4f8cff); outline-offset:-3px; }
#lf-compose { display:block; max-width:100%; max-height:58vh; width:auto; height:auto; touch-action:none; }
#lf-compose-empty { position:absolute; inset:0; display:grid; place-items:center; pointer-events:none; padding:16px; text-align:center; }
#lf-compose-empty span { background:rgba(0,0,0,.6); color:#fff; font-size:14px; border-radius:10px; padding:10px 12px; }
#lf-sheet select, #lf-sheet input[type=color] { font:inherit; color:var(--text,#eee); background:#2a2a2c; border:1px solid #3a3a3c; border-radius:10px; padding:8px; }
#lf-sheet input[type=color] { width:52px; height:42px; padding:2px; }
#lf-sheet .lf-field { display:flex; flex-direction:column; gap:4px; color:var(--muted,#999); font-size:13px; flex:1 1 140px; }
#lf-layer-props { display:none; }
#lf-layer-props.on { display:block; }
#lf-sheet .lf-checks { display:flex; gap:16px; margin-top:8px; color:var(--text,#eee); font-size:15px; }
#lf-sheet .lf-checks input { width:20px; height:20px; vertical-align:-4px; margin-right:6px; }
#lf-items { position:fixed; inset:0; z-index:1002; background:rgba(0,0,0,.6); display:flex; align-items:flex-end; justify-content:center; }
#lf-items[hidden] { display:none !important; }
#lf-sheet .lf-items-box { background:var(--panel,#1c1c1e); width:100%; max-width:900px; max-height:88vh; overflow:auto; border-radius:14px 14px 0 0; padding:12px 14px 20px; }
#lf-sheet .lf-chips { display:flex; gap:6px; overflow-x:auto; padding-bottom:6px; }
#lf-sheet .lf-chips button { flex:0 0 auto; padding:8px 12px; border-radius:18px; font-size:14px; }
#lf-items-search { width:100%; font:inherit; color:var(--text,#eee); background:#2a2a2c; border:1px solid #3a3a3c; border-radius:10px; padding:10px; margin-top:6px; }
#lf-items-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(96px,1fr)); gap:8px; }
#lf-items-grid button { padding:4px; border-radius:10px; background:#2a2a2c; display:flex; flex-direction:column; align-items:center; gap:4px; font-size:11px; color:var(--muted,#999); }
#lf-items-grid img { width:100%; aspect-ratio:1; object-fit:contain; display:block; }
#lf-toast { position:fixed; left:50%; bottom:24px; transform:translateX(-50%); background:#333; color:#fff; border-radius:10px; padding:10px 14px; font-size:14px;
  opacity:0; pointer-events:none; transition:opacity .25s; z-index:1001; max-width:90vw; text-align:center; }
#lf-toast.on { opacity:1; }
`;

const MARKUP = `
<div class="lf-top"><h2>✂️ Lift people</h2><button type="button" id="lf-close">✕ Close</button></div>
<div class="lf-cols">
<div>
  <div class="panel">
    <div class="lf-h">1 · Tap a person to lift them</div>
    <div class="lf-row" style="margin-top:0">
      <button type="button" id="lf-main" hidden>🖼️ The edit photo</button>
      <label class="btn" style="text-align:center;margin:0;color:var(--text,#eee);flex:1 1 auto">📷 Other photo
        <input type="file" id="lf-file" accept="image/*,.heic,.heif" style="display:none"></label>
    </div>
    <div class="lf-drop" id="lf-drop" style="margin-top:10px">Choose a photo to lift people from.<br><span style="font-size:13px">Lift people from several photos, one after another: they all collect in the new picture below.</span></div>
    <div class="lf-stage" id="lf-stage" hidden style="margin-top:10px">
      <div class="lf-photo" id="lf-photo">
        <canvas id="lf-base" aria-label="Photo: tap a person to lift them"></canvas>
        <canvas id="lf-overlay" aria-hidden="true"></canvas>
      </div>
      <div class="lf-busy" id="lf-busy"><div><div class="lf-spin"></div><span id="lf-busy-text">Working…</span></div></div>
    </div>
    <div id="lf-lift-tools" hidden>
      <div class="lf-row">
        <div class="lf-seg" role="group" aria-label="What a tap does">
          <button type="button" data-mode="new" class="on">New</button>
          <button type="button" data-mode="add">➕ Add</button>
          <button type="button" data-mode="sub">➖ Remove</button>
        </div>
        <button type="button" id="lf-btn-all" style="flex:0 1 auto">👥 Everyone</button>
      </div>
      <div id="lf-status" role="status" aria-live="polite"></div>
      <div class="hint" style="margin-top:4px">Tap a person: the whole person lifts. On a computer you can then drag them straight onto the new picture. <b>➕ Add</b> then tap more people to lift them together; <b>➖ Remove</b> then tap a stray bit to drop it. Tapping anything else (a dog, a bag) lifts that.</div>
      <div id="lf-sel-actions">
        <div class="lf-row"><button type="button" class="primary" id="lf-btn-use" style="margin:0">✂️ Add to the new picture</button></div>
        <div class="lf-row" style="margin-top:6px"><button type="button" id="lf-btn-save">⬇️ Save cut-out</button><button type="button" id="lf-btn-clear">Clear</button></div>
        <div class="lf-sl"><span>Edge softness</span><input type="range" id="lf-feather" min="0" max="8" step="0.5" value="1"></div>
        <div class="lf-sl"><span>Shrink ↔ grow</span><input type="range" id="lf-grow" min="-0.3" max="0.3" step="0.02" value="0"></div>
      </div>
      <div id="lf-tray" aria-label="Lifted people"></div>
    </div>
  </div>
</div>
<div>
  <div class="panel">
    <div class="lf-h">2 · New picture</div>
    <div id="lf-compose-wrap">
      <canvas id="lf-compose" width="1080" height="1080" aria-label="New picture: drag people to move them; the round handle resizes and turns them"></canvas>
      <div id="lf-compose-empty"><span>Lifted people appear here.<br>Drag to move · round handle or two fingers to resize and turn.</span></div>
    </div>
    <div class="lf-row">
      <label class="lf-field">Size
        <select id="lf-size">
          <option value="photo">Same as the photo</option>
          <option value="1536x2048">Portrait 3:4</option>
          <option value="2048x2048">Square</option>
          <option value="1152x2048">Story 9:16</option>
          <option value="2048x1536">Landscape 4:3</option>
          <option value="2048x1152">Wide 16:9</option>
        </select></label>
      <label class="lf-field">Background
        <select id="lf-bg">
          <option value="color">Plain colour</option>
          <option value="gradient">Gradient</option>
          <option value="photo">The photo</option>
          <option value="photo-blur">The photo, blurred</option>
          <option value="upload">Another photo…</option>
          <option value="none">See-through (download only)</option>
        </select></label>
      <label class="lf-field" id="lf-bg-color-l" style="flex:0 0 auto">Colour <input type="color" id="lf-bg-color" value="#ffffff"></label>
      <label class="lf-field" id="lf-bg-color2-l" style="flex:0 0 auto">to <input type="color" id="lf-bg-color2" value="#9ca3af"></label>
      <input type="file" id="lf-bg-file" accept="image/*,.heic,.heif" style="display:none">
    </div>
    <div id="lf-layer-props">
      <div class="lf-sl"><span>Size</span><input type="range" id="lf-l-scale" min="0.05" max="4" step="0.01"></div>
      <div class="lf-sl"><span>Turn</span><input type="range" id="lf-l-rot" min="-180" max="180" step="1"></div>
      <div class="lf-checks"><label><input type="checkbox" id="lf-l-shadow">Shadow</label><label><input type="checkbox" id="lf-l-sticker">Sticker outline</label></div>
      <div class="lf-row">
        <button type="button" id="lf-l-flip">↔️ Flip</button>
        <button type="button" id="lf-l-up">Forward</button>
        <button type="button" id="lf-l-down">Back</button>
        <button type="button" id="lf-l-dup">Copy</button>
        <button type="button" id="lf-l-del">🗑️</button>
      </div>
      <div class="lf-h" style="margin:12px 0 0;font-size:14px">Depth: paint which parts are in front</div>
      <div class="lf-seg" data-group="depth" style="margin-top:6px">
        <button type="button" data-depth="" class="on">✋ Move</button>
        <button type="button" data-depth="front">⬆️ In front</button>
        <button type="button" data-depth="back">⬇️ Behind</button>
        <button type="button" data-depth="clear">🧽 Normal</button>
      </div>
      <div class="lf-sl" id="lf-brush-row" hidden><span>Brush size</span><input type="range" id="lf-brush" min="6" max="120" step="1" value="36"></div>
      <div class="hint" id="lf-depth-hint" hidden style="margin-top:4px"></div>
    </div>
    <div class="hint" style="margin-top:8px">Tap a person in the new picture to move, resize, flip or delete them.</div>
    <div class="lf-row"><button type="button" id="lf-items-open">🛋️ Add furniture &amp; decor</button></div>
    <div class="lf-row" style="margin-top:12px">
      <button type="button" class="primary" id="lf-use-main" disabled style="margin:0">☁️ Use as the photo</button>
    </div>
    <div class="lf-row" style="margin-top:6px">
      <button type="button" id="lf-use-more" disabled>➕ Add as an extra photo</button>
      <button type="button" id="lf-download" disabled>⬇️ Download</button>
    </div>
    <div class="hint" id="lf-use-hint" style="margin-top:6px"></div>
  </div>
</div>
</div>
<div id="lf-items" hidden>
  <div class="lf-items-box">
    <div class="lf-top" style="margin:0 0 8px"><h2 style="font-size:17px">🛋️ Furniture &amp; decor</h2><button type="button" id="lf-items-close">✕ Close</button></div>
    <div class="lf-chips" id="lf-items-cats"></div>
    <input id="lf-items-search" type="search" placeholder="Search, e.g. sofa, lamp, round table" autocomplete="off" autocapitalize="off" spellcheck="false">
    <div id="lf-items-status" class="hint" style="margin:6px 0"></div>
    <div id="lf-items-grid"></div>
    <div class="hint" style="margin-top:8px">Free 3D-scanned models from Poly Haven (CC0: free for any use). Tap one to add it; it goes behind your people, sized to them.</div>
  </div>
</div>
<div id="lf-toast" role="status" aria-live="polite"></div>
`;

const sheet = document.createElement('div');
sheet.id = 'lf-sheet';
sheet.hidden = true;
sheet.innerHTML = MARKUP;
const styleEl = document.createElement('style');
styleEl.textContent = STYLE;
document.head.appendChild(styleEl);
document.body.appendChild(sheet);

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (id) => document.getElementById('lf-' + id);
const el = {
  file: $('file'), drop: $('drop'), stage: $('stage'), photo: $('photo'),
  base: $('base'), overlay: $('overlay'), busy: $('busy'), busyText: $('busy-text'),
  tools: $('lift-tools'), status: $('status'), all: $('btn-all'), actions: $('sel-actions'),
  use: $('btn-use'), save: $('btn-save'), clear: $('btn-clear'),
  feather: $('feather'), grow: $('grow'), tray: $('tray'),
  compose: $('compose'), composeWrap: $('compose-wrap'), composeEmpty: $('compose-empty'),
  size: $('size'), bg: $('bg'), bgColor: $('bg-color'), bgColor2: $('bg-color2'),
  bgColorL: $('bg-color-l'), bgColor2L: $('bg-color2-l'), bgFile: $('bg-file'),
  layerProps: $('layer-props'), lScale: $('l-scale'), lRot: $('l-rot'), lShadow: $('l-shadow'),
  lSticker: $('l-sticker'), lFlip: $('l-flip'), lUp: $('l-up'), lDown: $('l-down'),
  lDup: $('l-dup'), lDel: $('l-del'), toast: $('toast'),
  main: $('main'), close: $('close'), useMain: $('use-main'), useMore: $('use-more'),
  download: $('download'), useHint: $('use-hint'),
  brushRow: $('brush-row'), brush: $('brush'), depthHint: $('depth-hint'),
  itemsOpen: $('items-open'), items: $('items'), itemsClose: $('items-close'), itemsCats: $('items-cats'),
  itemsSearch: $('items-search'), itemsStatus: $('items-status'), itemsGrid: $('items-grid'),
};

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

// Models and runtime come from the VM (edit-restart copies them next to this
// file); if they aren't there yet, from the public CDN and Google's storage.
async function assetBase() {
  try {
    const r = await fetch(LOCAL.bundle, { method: 'HEAD', cache: 'no-store' });
    if (r.ok) return LOCAL;
  } catch {}
  return REMOTE;
}

function loadModels(onProgress) {
  if (modelsPromise) return modelsPromise;
  modelsPromise = (async () => {
    const from = await assetBase();
    const expected = MODELS.person.size + MODELS.touch.size;
    let got = 0;
    const tick = (n) => { got += n; onProgress(Math.min(99, Math.round((got / expected) * 100))); };
    const [vision, personBuf, touchBuf] = await Promise.all([
      import(from.bundle),
      fetchModel(from.person, tick),
      fetchModel(from.touch, tick),
    ]);
    const fileset = await vision.FilesetResolver.forVisionTasks(from.wasm);
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
    models = { person, touch, personIndex, from: from === LOCAL ? 'VM' : 'internet' };
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
  try {
    if (opts.readable) file = await opts.readable(file);    // HEIC: converted on the VM
  } catch (err) {
    toast(err.message);
    return;
  }
  let img;
  try {
    img = await decodeImage(file);
  } catch {
    toast("This browser can't open that image. Try a JPG or PNG.");
    return;
  }
  srcFromMain = false;
  await openImage(img);
}

// Show an image (a decoded file, or the edit page's photo canvas) for lifting.
async function openImage(img) {
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

  if (sam.ok) samImage().catch(() => {});   // send it to the VM now: the first tap won't wait
  await prepare();
}

// Load the models the first time a photo is opened.
async function prepare() {
  if (models) { setStatus('Tap a person to lift them.'); return; }
  working = true;
  try {
    setBusy('Loading the lift models… 0%');
    await loadModels((p) => setBusy(`Loading the lift models… ${p}%`));
    setStatus('Tap a person to lift them.');
  } catch (err) {
    console.error(err);
    setStatus("Couldn't load the lift models. Run edit-restart on the VM, then reopen ✂️ Lift.", true);
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

let lastWasPerson = false;

// Lift whatever is under (px, py).
function regionAtPx(px, py) {
  const { w, h } = src;
  const big = Math.max(w, h);
  lastWasPerson = false;
  const raw = touchMask(px, py);
  if (raw[py * w + px] <= 0.5) return raw;
  let region = isolateAt(raw, w, h, px, py);

  // What did we hit? Look at a generous crop, so a face still comes with a body.
  const wide = sceneInRect(squareAround(bboxOf(region, w), 2.2, big * 0.4));
  // Thin limbs are below the scene model's resolution, so a tap on an arm
  // counts as a person tap if there is a person right next to the finger.
  const isPerson = overlapShare(wide.person, region) >= 0.4 || nearMax(wide.person, px, py, Math.round(big / 60)) > 0.5;

  lastWasPerson = isPerson;
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

// A tap's lift: the outline from the browser models, then SAM on the VM keeps
// only the tapped person when people touch or overlap.
async function liftAt(px, py) {
  const r = fillSmallHoles(regionAtPx(px, py), src.w, src.h);
  return solidify(await separate(r, px, py, lastWasPerson));
}

// ── SAM on the VM: tells touching people apart ───────────────────────────
// Magic Touch and DeepLab see "person here", not "which person": people who
// touch or overlap come out as one. SAM (Segment Anything, already on the VM
// for Selected area) outlines the one object under the finger; its largest
// answer is the whole tapped person. The photo goes to the VM once per photo.
const sam = { ok: true, forId: -1, name: null, pending: null };

function samImage() {
  if (sam.forId === src.id) return sam.name ? Promise.resolve(sam.name) : sam.pending;
  const id = src.id;
  sam.forId = id;
  sam.name = null;
  sam.pending = (async () => {
    const fd = new FormData();
    fd.append('image', await canvasToBlob(src.canvas, 'image/jpeg', 0.92), `rv6m_lift_${Date.now().toString(36)}.jpg`);
    fd.append('overwrite', 'true');
    const r = await fetch('/api/upload/image', { method: 'POST', body: fd });
    if (!r.ok) throw Object.assign(new Error(`upload failed (${r.status})`), { permanent: r.status === 404 });
    const j = await r.json();
    const name = j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
    if (sam.forId === id) sam.name = name;
    return name;
  })();
  sam.pending.catch(() => { if (sam.forId === id) sam.forId = -1; });
  return sam.pending;
}

async function samMask(px, py) {
  const name = await samImage();
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 120000);   // the first use loads SAM on the VM
  try {
    const r = await fetch('/rv6m/segment', {
      method: 'POST', signal: ctl.signal, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: name, points: [[px + 0.5, py + 0.5]], labels: [1], grow: 0, pick: 'largest' }),
    });
    if (!r.ok) throw Object.assign(new Error(`SAM failed (${r.status})`), { permanent: r.status === 404 });
    const bmp = await createImageBitmap(await r.blob());
    const c = makeCanvas(src.w, src.h);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0, src.w, src.h);
    if (bmp.close) bmp.close();
    const d = ctx.getImageData(0, 0, src.w, src.h).data;
    const m = new Float32Array(src.w * src.h);
    for (let i = 0; i < m.length; i++) m[i] = d[i * 4 + 3] > 127 ? 1 : 0;
    return m;
  } finally {
    clearTimeout(timer);
  }
}

// Keep the browser outline only where SAM agrees it's the tapped one (soft
// edges stay), and fill in what SAM is sure of but the outline missed. If SAM
// grabbed something much bigger than the tap (a table under a cup), only trim.
async function separate(r, px, py, isPerson) {
  if (!sam.ok) return r;
  let S;
  try {
    S = await samMask(px, py);
  } catch (err) {
    if (err.permanent) sam.ok = false;      // this VM has no SAM: don't keep asking
    console.warn('[lift] SAM unavailable, keeping the browser outline:', err.message);
    return r;
  }
  const { w, h } = src;
  const sArea = maskArea(S);
  if (S[py * w + px] < 0.5 || sArea < MIN_PIXELS) return r;
  const fill = isPerson || sArea <= 1.6 * maskArea(r);
  const b = boxBlur(S, w, h, Math.max(2, Math.round(Math.max(w, h) / 300)));
  const out = new Float32Array(r.length);
  for (let i = 0; i < out.length; i++) {
    const kept = Math.min(r[i], clamp01(b[i] * 3));
    out[i] = fill ? Math.max(kept, clamp01((b[i] - 0.85) / 0.15)) : kept;
  }
  return out;
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
    const px = Math.min(src.w - 1, Math.max(0, Math.floor(nx * src.w)));
    const py = Math.min(src.h - 1, Math.max(0, Math.floor(ny * src.h)));
    const region = await liftAt(px, py);
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
      let r = await liftAt(s.x, s.y);
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
  chip.title = 'Click, or drag it onto the new picture, to add it again';
  chip.tabIndex = 0;
  chip.draggable = true;
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
  bgMode: 'color',
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

// ── Depth: parts of a layer painted "in front" are drawn over every layer,
// parts painted "behind" under every layer; the rest keeps the layer order.
// Masks live in the layer's own pixels, so a rough stroke keeps exact edges.
function layerPart(L, part) {
  if (!L.front && !L.back) return part === 'base' ? L.cut.canvas : null;
  if (!L.parts || L.parts.v !== L.maskV) {
    const { cut } = L;
    // Brush edges are soft; a soft split would leave a faint seam where the
    // two parts meet. Hard-edged masks split the person's own pixels cleanly
    // (the person keeps their own soft outline).
    const hard = (m) => {
      if (!m) return null;
      const c = makeCanvas(m.width, m.height);
      const x = c.getContext('2d');
      x.drawImage(m, 0, 0);
      const id = x.getImageData(0, 0, c.width, c.height);
      for (let i = 3; i < id.data.length; i += 4) id.data[i] = id.data[i] > 127 ? 255 : 0;
      x.putImageData(id, 0, 0);
      return c;
    };
    const fm = hard(L.front), bm = hard(L.back);
    const piece = (keep, drop) => {
      const c = makeCanvas(cut.w, cut.h);
      const x = c.getContext('2d');
      x.drawImage(cut.canvas, 0, 0);
      x.globalCompositeOperation = keep ? 'destination-in' : 'destination-out';
      if (keep) x.drawImage(keep, 0, 0);
      else for (const m of drop) if (m) x.drawImage(m, 0, 0);
      return c;
    };
    const tint = (part, color) => {
      if (!part) return null;
      const c = makeCanvas(cut.w, cut.h);
      const x = c.getContext('2d');
      x.drawImage(part, 0, 0);
      x.globalCompositeOperation = 'source-in';
      x.fillStyle = color;
      x.fillRect(0, 0, cut.w, cut.h);
      return c;
    };
    const front = fm ? piece(fm) : null;
    const back = bm ? piece(bm) : null;
    L.parts = { v: L.maskV, base: piece(null, [fm, bm]), front, back,
                frontTint: tint(front, 'rgba(52,211,153,0.55)'), backTint: tint(back, 'rgba(96,165,250,0.6)') };
  }
  return L.parts[part];
}

function drawLayer(ctx, L, part = 'base') {
  const { cut } = L;
  const img = layerPart(L, part);
  if (!img) return;
  ctx.save();
  ctx.translate(L.x, L.y);
  ctx.rotate(L.rot);
  ctx.scale(L.flip ? -L.scale : L.scale, L.scale);
  if (L.shadow && part === 'base') {
    const size = Math.max(cut.w, cut.h) * L.scale;
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = size * 0.05;
    ctx.shadowOffsetY = size * 0.025;
  }
  if (L.sticker && part === 'base') {
    const s = stickerFor(cut);
    ctx.drawImage(s.canvas, -cut.w / 2 - s.r, -cut.h / 2 - s.r);
    ctx.shadowColor = 'transparent';
  }
  ctx.drawImage(img, -cut.w / 2, -cut.h / 2);
  ctx.restore();
}

// While painting depth, show the painted parts of the selected layer tinted
function drawDepthTint(ctx, L) {
  for (const k of ['backTint', 'frontTint']) {
    const t = layerPart(L, k);
    if (!t) continue;
    ctx.save();
    ctx.translate(L.x, L.y);
    ctx.rotate(L.rot);
    ctx.scale(L.flip ? -L.scale : L.scale, L.scale);
    ctx.drawImage(t, -L.cut.w / 2, -L.cut.h / 2);
    ctx.restore();
  }
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
  for (const part of ['back', 'base', 'front']) for (const L of comp.layers) drawLayer(ctx, L, part);
  if (ui && comp.selected && depthMode) drawDepthTint(ctx, comp.selected);
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
    for (const b of [el.useMain, el.useMore, el.download]) b.disabled = !has;
    el.composeWrap.classList.toggle('checker', comp.bgMode === 'none');
  });
}

function selectLayer(L) {
  comp.selected = L;
  if (!L) setDepthMode('');
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

// Opaque pixels first (so you can reach someone standing behind), then
// anywhere in a person's box: thin arms and legs are hard to hit with a finger.
function hitLayer(p) {
  return hitLayerPass(p, true) || hitLayerPass(p, false);
}

function hitLayerPass(p, opaqueOnly) {
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
    if (!opaqueOnly || L === comp.selected || itemAlpha(L.cut)[iy * L.cut.w + ix] > 24) return L;
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
  if (L && depthMode) {
    gesture = { type: 'paint', L, last: p };
    paintDepth(L, p, p);
    return;
  }
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
  if (hit) {
    selectLayer(hit);
    gesture = { type: 'move', L: hit, dx: p.x - hit.x, dy: p.y - hit.y };
  } else if (L) {
    // Empty space: a drag still moves the selected person (on a small phone
    // screen they're hard to hit); a plain tap deselects.
    gesture = { type: 'move', L, dx: p.x - L.x, dy: p.y - L.y, start: p, tapOff: true };
  } else {
    gesture = null;
  }
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
  if (gesture.type === 'paint') {
    paintDepth(L, gesture.last, p);
    gesture.last = p;
    renderCompose();
    return;
  }
  if (gesture.type === 'move') {
    if (gesture.tapOff && Math.hypot(p.x - gesture.start.x, p.y - gesture.start.y) > 8 * pxRatio()) gesture.tapOff = false;
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

// iOS Safari can start scrolling the sheet under a finger on the picture and
// then cancel the drag; blocking the touch's default keeps the drag going.
for (const t of ['touchstart', 'touchmove']) {
  el.compose.addEventListener(t, (e) => { if (e.cancelable) e.preventDefault(); }, { passive: false });
}

function endPointer(e) {
  pointers.delete(e.pointerId);
  if (e.type === 'pointerup' && pointers.size === 0 && gesture?.tapOff) {
    // Undo the tiny wobble of a tap, then deselect
    gesture.L.x = gesture.start.x - gesture.dx;
    gesture.L.y = gesture.start.y - gesture.dy;
    selectLayer(null);
  }
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
  if (!L || sheet.hidden) return;
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
  const copy = (m) => { if (!m) return null; const c = makeCanvas(m.width, m.height); c.getContext('2d').drawImage(m, 0, 0); return c; };
  const C = { ...L, id: ++layerId, x: L.x + comp.W * 0.05, y: L.y + comp.H * 0.03,
              front: copy(L.front), back: copy(L.back), parts: null, maskV: 0 };
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

// The finished picture. For the edit page it never has see-through parts:
// Seedream and the edit models would read them as black.
function exportCanvas(opaque) {
  const c = makeCanvas(comp.W, comp.H);
  const ctx = c.getContext('2d');
  render(ctx, { forJpeg: opaque });
  return c;
}


// ── Depth brush ──────────────────────────────────────────────────────────
let depthMode = '';            // '' (move) | 'front' | 'back' | 'clear'
const DEPTH_HINTS = {
  front: 'Paint roughly over the part that should be in front (e.g. a leg). Only this person\'s own pixels change. Green = in front.',
  back: 'Paint roughly over the part that should go behind everyone else (e.g. an arm). Blue = behind.',
  clear: 'Paint over a green or blue part to put it back to normal.',
};
const depthButtons = sheet.querySelectorAll('.lf-seg[data-group=depth] button');
function setDepthMode(m) {
  depthMode = m;
  depthButtons.forEach((b) => b.classList.toggle('on', b.dataset.depth === m));
  el.brushRow.hidden = !m;
  el.depthHint.hidden = !m;
  el.depthHint.textContent = DEPTH_HINTS[m] || '';
  el.compose.style.cursor = m ? 'crosshair' : 'default';
  renderCompose();
}
depthButtons.forEach((b) => b.addEventListener('click', () => setDepthMode(b.dataset.depth)));

// Canvas point -> the layer's own pixel coordinates
function toLayer(L, p) {
  const dx = p.x - L.x, dy = p.y - L.y;
  const c = Math.cos(-L.rot), s = Math.sin(-L.rot);
  let lx = (dx * c - dy * s) / L.scale;
  const ly = (dx * s + dy * c) / L.scale;
  if (L.flip) lx = -lx;
  return { x: lx + L.cut.w / 2, y: ly + L.cut.h / 2 };
}

function paintDepth(L, a, b) {
  const { w, h } = L.cut;
  if (depthMode !== 'clear') {
    if (depthMode === 'front' && !L.front) L.front = makeCanvas(w, h);
    if (depthMode === 'back' && !L.back) L.back = makeCanvas(w, h);
  }
  const r = (+el.brush.value * pxRatio()) / L.scale;   // brush is in screen pixels
  const A = toLayer(L, a), B = toLayer(L, b);
  const stroke = (m, erase) => {
    if (!m) return;
    const x = m.getContext('2d');
    x.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
    x.strokeStyle = x.fillStyle = '#fff';
    x.lineCap = x.lineJoin = 'round';
    x.lineWidth = r * 2;
    x.beginPath(); x.moveTo(A.x, A.y); x.lineTo(B.x, B.y); x.stroke();
    x.beginPath(); x.arc(B.x, B.y, r, 0, Math.PI * 2); x.fill();
  };
  // A pixel is in front, behind or normal, never two at once
  stroke(L.front, depthMode !== 'front');
  stroke(L.back, depthMode !== 'back');
  L.maskV = (L.maskV || 0) + 1;
}

// ── Furniture & decor (Poly Haven, CC0) ─────────────────────────────────
// Free 3D-scanned models; their previews are photo-real renders on a
// transparent background, served with CORS, so they load straight into the
// picture. Each model lists its real size, used to size it next to people.
const PH_API = 'https://api.polyhaven.com/assets?t=models&categories=';
const ITEM_CATS = [
  ['Sofas & chairs', 'seating'], ['Tables', 'table'], ['Beds', 'bed'], ['Storage', 'shelves'],
  ['Lamps', 'lighting'], ['Plants', 'plants'], ['Decor', 'decorative'],
];
const itemCache = {};           // category -> [{ id, name, tags, thumb, full, heightMm }]
let itemCat = ITEM_CATS[0][1];

async function loadItems(cat) {
  if (itemCache[cat]) return itemCache[cat];
  const r = await fetch(PH_API + encodeURIComponent(cat));
  if (!r.ok) throw new Error(`Poly Haven answered ${r.status}`);
  const d = await r.json();
  itemCache[cat] = Object.entries(d)
    .map(([id, v]) => {
      const ver = (String(v.thumbnail_url || '').match(/[?&]v=([^&]+)/) || [])[1] || '';
      return {
        id, name: v.name || id, tags: (v.tags || []).join(' ').toLowerCase(),
        thumb: `https://cdn.polyhaven.com/asset_img/thumbs/${id}.png?width=256&height=256&format=webp${ver ? '&v=' + ver : ''}`,
        full: `https://cdn.polyhaven.com/asset_img/thumbs/${id}.png?width=1024&height=1024${ver ? '&v=' + ver : ''}`,
        heightMm: Array.isArray(v.dimensions) && v.dimensions[2] > 0 ? v.dimensions[2] : 0,
        downloads: v.download_count || 0,
      };
    })
    .sort((a, b) => b.downloads - a.downloads);
  return itemCache[cat];
}

function renderItemCats() {
  el.itemsCats.innerHTML = '';
  for (const [label, cat] of ITEM_CATS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.classList.toggle('on', cat === itemCat);
    b.addEventListener('click', () => { itemCat = cat; el.itemsSearch.value = ''; renderItemCats(); showItems(); });
    el.itemsCats.appendChild(b);
  }
}

async function showItems() {
  const cat = itemCat;
  el.itemsGrid.innerHTML = '';
  el.itemsStatus.textContent = 'Loading…';
  let list;
  try {
    list = await loadItems(cat);
  } catch (err) {
    el.itemsStatus.textContent = `Couldn't reach Poly Haven (${err.message}). Check the internet connection and try again.`;
    return;
  }
  if (cat !== itemCat) return;
  const q = el.itemsSearch.value.trim().toLowerCase();
  const shown = q ? list.filter((it) => it.name.toLowerCase().includes(q) || it.tags.includes(q)) : list;
  el.itemsStatus.textContent = shown.length ? `${shown.length} item${shown.length > 1 ? 's' : ''}` : 'Nothing matches that here: try another tab.';
  for (const it of shown) {
    const b = document.createElement('button');
    b.type = 'button';
    b.title = it.name;
    b.innerHTML = `<img alt="" loading="lazy" src="${it.thumb}"><span></span>`;
    b.querySelector('span').textContent = it.name;
    b.addEventListener('click', () => addItem(it));
    el.itemsGrid.appendChild(b);
  }
}

async function addItem(it) {
  el.itemsStatus.textContent = `Adding ${it.name}…`;
  try {
    const img = new Image();
    img.crossOrigin = 'anonymous';           // keeps the picture exportable
    img.src = it.full;
    await img.decode();
    const c = makeCanvas(img.naturalWidth, img.naturalHeight);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    // Trim the transparent margin so the box hugs the item
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let x0 = c.width, y0 = c.height, x1 = -1, y1 = -1;
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      if (d[(y * c.width + x) * 4 + 3] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    if (x1 < 0) throw new Error('empty picture');
    const cw = x1 - x0 + 1, ch = y1 - y0 + 1;
    const t = makeCanvas(cw, ch);
    t.getContext('2d').drawImage(c, x0, y0, cw, ch, 0, 0, cw, ch);
    const cut = { canvas: t, x0: 0, y0: 0, w: cw, h: ch, srcW: cw, srcH: ch, item: it };
    addItemLayer(cut, it);
    el.items.hidden = true;
    toast(`🛋️ Added ${it.name}`);
  } catch (err) {
    console.error(err);
    el.itemsStatus.textContent = `Couldn't load ${it.name}. Try again or pick another.`;
  }
}

// Sized next to the people already in the picture (a person ≈ 1.7 m), else by
// its own real height; it goes behind the people.
function addItemLayer(cut, it) {
  if (!comp.sized) applySizeSelect();
  const people = comp.layers.filter((L) => !L.cut.item);
  const personPx = Math.max(0, ...people.map((L) => L.cut.h * L.scale));
  const hMm = it.heightMm || 900;
  let targetH = personPx ? personPx * (hMm / 1700) : comp.H * Math.min(0.85, Math.max(0.15, hMm / 2200));
  targetH = Math.min(targetH, comp.H * 0.95);
  const scale = Math.max(0.03, targetH / cut.h);
  const x = comp.W / 2 + (comp.layers.length % 3 - 1) * comp.W * 0.12;
  const y = comp.H * 0.96 - (cut.h * scale) / 2;
  const L = { id: ++layerId, cut, x, y, scale, rot: 0, flip: false, shadow: false, sticker: false };
  comp.layers.unshift(L);                   // behind everyone; Forward brings it in front
  selectLayer(L);
  renderCompose();
  return L;
}

el.itemsOpen.addEventListener('click', () => {
  el.items.hidden = false;
  renderItemCats();
  showItems();
});
el.itemsClose.addEventListener('click', () => { el.items.hidden = true; });
el.items.addEventListener('click', (e) => { if (e.target === el.items) el.items.hidden = true; });
let itemSearchTimer = 0;
el.itemsSearch.addEventListener('input', () => { clearTimeout(itemSearchTimer); itemSearchTimer = setTimeout(showItems, 150); });

// ── Drag and drop onto the new picture (laptop) ──────────────────────────
// Drag the glowing person straight off the photo, or a cut-out from the row
// under it, and let go on the new picture: it lands where you let go.
let dragging = null;                       // { cut, fromSel }
const dragImg = makeCanvas(1, 1);
dragImg.style.cssText = 'position:fixed;left:-10000px;top:0;pointer-events:none';
sheet.appendChild(dragImg);

function startDrag(e, cut, fromSel) {
  dragging = { cut, fromSel };
  e.dataTransfer.effectAllowed = 'copy';
  e.dataTransfer.setData('text/plain', 'lifted person');
  const k = Math.min(1, 180 / Math.max(cut.w, cut.h));
  dragImg.width = Math.max(1, Math.round(cut.w * k));
  dragImg.height = Math.max(1, Math.round(cut.h * k));
  dragImg.getContext('2d').drawImage(cut.canvas, 0, 0, dragImg.width, dragImg.height);
  try { e.dataTransfer.setDragImage(dragImg, dragImg.width / 2, dragImg.height / 2); } catch {}
}

el.base.draggable = true;
el.base.addEventListener('dragstart', (e) => {
  // Only a drag that starts on the lifted person; anything else isn't a drag
  const r = el.base.getBoundingClientRect();
  const x = Math.floor(((e.clientX - r.left) / r.width) * src.w);
  const y = Math.floor(((e.clientY - r.top) / r.height) * src.h);
  const cut = sel && !working && x >= 0 && y >= 0 && x < src.w && y < src.h && sel[y * src.w + x] > 0.5 && currentCutout();
  if (!cut) { e.preventDefault(); return; }
  startDrag(e, { ...cut }, true);
});
el.tray.addEventListener('dragstart', (e) => {
  const chip = e.target.closest?.('.lift-chip');
  const item = chip && tray.find((t) => t.id === +chip.dataset.id);
  if (!item) { e.preventDefault(); return; }
  startDrag(e, item.cut, false);
});
document.addEventListener('dragend', () => { dragging = null; el.composeWrap.classList.remove('over'); });
el.composeWrap.addEventListener('dragover', (e) => {
  if (!dragging) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
  el.composeWrap.classList.add('over');
});
el.composeWrap.addEventListener('dragleave', (e) => {
  if (!el.composeWrap.contains(e.relatedTarget)) el.composeWrap.classList.remove('over');
});
el.composeWrap.addEventListener('drop', async (e) => {
  if (!dragging) return;
  e.preventDefault();
  e.stopPropagation();                     // not a photo to lift from
  el.composeWrap.classList.remove('over');
  const { cut, fromSel } = dragging;
  dragging = null;
  const r = el.compose.getBoundingClientRect();
  const L = addLayer(cut);                 // sizes the canvas on the first person
  L.x = Math.min(comp.W, Math.max(0, ((e.clientX - r.left) / r.width) * comp.W));
  L.y = Math.min(comp.H, Math.max(0, ((e.clientY - r.top) / r.height) * comp.H));
  renderCompose();
  if (fromSel) {
    clearSelection();
    setStatus('Added to the new picture. Tap another person, or open another photo to add people from it.');
    await addToTray(cut);
  }
});

// ── Wiring ───────────────────────────────────────────────────────────────

// The edit page's range inputs colour their track from --pct
function paintRange(r) {
  const pct = ((+r.value - +r.min) / (+r.max - +r.min)) * 100;
  r.style.setProperty('--pct', pct + '%');
}
sheet.querySelectorAll('input[type=range]').forEach((r) => {
  paintRange(r);
  r.addEventListener('input', () => paintRange(r));
});
const syncRanges = () => sheet.querySelectorAll('input[type=range]').forEach(paintRange);
for (const r of [el.lScale, el.lRot]) r.addEventListener('change', syncRanges);
el.compose.addEventListener('pointerup', syncRanges);

el.file.addEventListener('change', () => {
  const f = el.file.files[0];
  el.file.value = '';
  openFile(f);
});
el.main.addEventListener('click', () => { if (opts.photo) useMainPhoto(); });

el.base.addEventListener('click', onTap);
el.all.addEventListener('click', selectAllPeople);
el.clear.addEventListener('click', () => { clearSelection(); setStatus('Tap a person to lift them.'); });

const tapButtons = sheet.querySelectorAll('.lf-seg[role=group] button');
tapButtons.forEach((b) => b.addEventListener('click', () => {
  tapMode = b.dataset.mode;
  tapButtons.forEach((x) => x.classList.toggle('on', x === b));
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
  const frozen = { ...cut };          // later edge tweaks don't change this one
  // Clear and place it right away: a quick tap on the next person must not be
  // wiped by a clear that lands after the tray thumbnail is saved
  clearSelection();
  addLayer(frozen);
  await addToTray(frozen);
  setStatus('Added to the new picture (below). Tap another person, or open another photo to add people from it.');
  toast('✂️ Added to the new picture');
});
el.save.addEventListener('click', async () => {
  const c = sel && currentCutout();
  if (!c) return;
  downloadBlob(await canvasToBlob(c.canvas), `lifted-${stamp()}.png`);
});

el.download.addEventListener('click', async () => {
  const see = comp.bgMode === 'none';
  const c = exportCanvas(!see);
  downloadBlob(await canvasToBlob(c, see ? 'image/png' : 'image/jpeg', 0.95), `lift-${stamp()}.${see ? 'png' : 'jpg'}`);
});

async function handBack(as) {
  if (!opts.onUse) return;
  const c = exportCanvas(true);
  const blob = await canvasToBlob(c, 'image/jpeg', 0.95);
  close();
  await opts.onUse(blob, as);
}
el.useMain.addEventListener('click', () => handBack('main'));
el.useMore.addEventListener('click', () => handBack('more'));

el.close.addEventListener('click', close);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !sheet.hidden && !comp.selected) close();
});

// A photo pasted or dropped while the sheet is open is lifted from (the edit
// page leaves those to the sheet while it's open)
document.addEventListener('paste', (e) => {
  if (sheet.hidden) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (!item) return;
  e.preventDefault();
  openFile(item.getAsFile());
});
sheet.addEventListener('drop', (e) => {
  const f = [...(e.dataTransfer?.files || [])].find((x) => /^image\//.test(x.type) || /\.(heic|heif)$/i.test(x.name));
  if (!f) return;
  e.preventDefault();
  openFile(f);
});

function useMainPhoto() {
  srcFromMain = true;
  mainKey = opts.photoKey;
  openImage(opts.photo);
}

let scrollY0 = 0;
function close() {
  sheet.hidden = true;
  document.documentElement.style.overflow = '';
  document.body.style.overflow = '';
  window.scrollTo(0, scrollY0);
}

syncBgControls();
renderCompose();

// Opens the sheet. opts:
//   photo:    the edit page's photo (a canvas), or null; lifted from first
//   photoKey: changes whenever the edit page's photo changes
//   readable: (file) => file the browser can open (HEIC conversion on the VM)
//   onUse:    (jpegBlob, 'main' | 'more') => the new picture goes back to the page
//   useHint:  a line under the hand-back buttons
export function openLift(o = {}) {
  opts = o;
  scrollY0 = window.scrollY;
  sheet.hidden = false;
  sheet.scrollTop = 0;
  document.documentElement.style.overflow = 'hidden';
  document.body.style.overflow = 'hidden';
  el.main.hidden = !opts.photo;
  el.useHint.textContent = opts.useHint || '';
  // Start on the edit page's photo, and follow it when it has changed since
  if (opts.photo && (!src.canvas || (srcFromMain && mainKey !== opts.photoKey))) useMainPhoto();
  renderCompose();
}

// For automated tests
window.__lift = { comp, src, get sel() { return sel; }, get models() { return models; } };
