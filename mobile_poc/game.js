// SpaceAce Mobile — proof-of-concept HTML5 port.
// Physics is a faithful JS port of src/real_physics.rs / real_game.rs:
// gravity 100, thrust 400, rotation 4.363323 rad/s, fixed 60 Hz ticks,
// 5-segment ship hull vs wall lines, pickup radius 46.5 (36.5 + 10).
// Wall collision is checked EVERY frame (strict mode — the original
// engine's every-other-frame skip above ~316 px/s is a banned exploit).

'use strict';

// ---------------------------------------------------------------- constants
const GRAVITY = 100.0;
const THRUST_POWER = 400.0;
const ROTATION_SPEED = 4.363323;
const SHIP_COLLISION_RADIUS = 36.5;
const PICKUP_RADIUS = 10.0;
const PICKUP_COLLECT_R2 = (SHIP_COLLISION_RADIUS + PICKUP_RADIUS) ** 2;
const TICK = 1 / 60;

// Exact shipVerts from the original engine (ship-local, 0 rotation = up).
const SHIP_VERTS = [
  [0, -36.5], [-19, 23.5], [-24, 23.5], [-15.675, 13], [19, 23.5],
  [24, 23.5], [15.675, 13], [0, 67.45], [-14.1075, 13], [14.1075, 13],
];
// Collision segments: pairs of indices into SHIP_VERTS.
const SHIP_SEGS = [[3, 6], [2, 1], [1, 0], [0, 4], [4, 5]];
const RENDER_TRI = [0, 3, 6];       // nose, left wing, right wing
const FLAME_TRI = [8, 7, 9];

// ------------------------------------------------------------- level parser
// Same algorithm as src/real_map_parser.rs / docs/LEVEL_FORMAT.md.
function parseLevel(D) {
  const n = D[0] | 0;
  const verts = [];
  for (let i = 0; i < n; i++) verts.push([D[1 + i * 2], D[2 + i * 2]]);
  const ol = 1 + n * 2 + 1;
  const lineCount = D[ol - 1] | 0;
  const walls = [];
  for (let i = 0; i < lineCount; i++) {
    const a = D[ol + i * 2] | 0, b = D[ol + i * 2 + 1] | 0;
    if (a < n && b < n) walls.push([verts[a], verts[b]]);
  }
  const startIdx = D[ol + lineCount * 2] | 0;
  const ql = ol + lineCount * 2 + 4;
  const pickupCount = D[ql - 1] | 0;
  const pickups = [];
  for (let i = 0; i < pickupCount; i++) {
    const p = D[ql + i] | 0;
    if (p < n) pickups.push([verts[p][0], verts[p][1]]);
  }
  const sv = startIdx < n ? verts[startIdx] : verts[0];
  // Bounds from geometry (like real_game.rs), with margin.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of verts) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  return {
    verts, walls, pickups,
    spawn: [sv[0], sv[1] - 100],     // engine spawns 100 px above start vertex
    bounds: [minX - 50, minY - 50, maxX + 50, maxY + 50],
  };
}

// ------------------------------------------------------------ geometry util
function segsIntersect(ax, ay, bx, by, cx, cy, dx, dy) {
  const d1x = bx - ax, d1y = by - ay, d2x = dx - cx, d2y = dy - cy;
  const denom = d1x * d2y - d1y * d2x;
  if (denom === 0) return false;
  const t = ((cx - ax) * d2y - (cy - ay) * d2x) / denom;
  const u = ((cx - ax) * d1y - (cy - ay) * d1x) / denom;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

// ---------------------------------------------------------------- game sim
class Sim {
  constructor(level) {
    this.level = level;
    this.reset();
  }
  reset() {
    const [sx, sy] = this.level.spawn;
    this.x = sx; this.y = sy;
    this.vx = 0; this.vy = 0;
    this.rot = 0;
    this.exploded = false;
    this.time = 0;
    this.ticks = 0;
    this.collected = this.level.pickups.map(() => false);
    this.remaining = this.level.pickups.length;
    this.completed = false;
    this.justCollected = null;
  }
  shipSegments(x = this.x, y = this.y, rot = this.rot) {
    const c = Math.cos(rot), s = Math.sin(rot);
    const tv = SHIP_VERTS.map(([vx, vy]) => [vx * c - vy * s + x, vx * s + vy * c + y]);
    return SHIP_SEGS.map(([a, b]) => [tv[a], tv[b]]);
  }
  step(left, right, thrust) {
    this.justCollected = null;
    if (this.exploded || this.completed) return;
    const dt = TICK;
    if (left) this.rot -= ROTATION_SPEED * dt;
    if (right) this.rot += ROTATION_SPEED * dt;
    if (thrust) {
      const a = this.rot - Math.PI / 2;
      this.vx += THRUST_POWER * Math.cos(a) * dt;
      this.vy += THRUST_POWER * Math.sin(a) * dt;
    }
    this.vy += GRAVITY * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.time += dt;
    this.ticks++;

    // Strict collision: every frame, no high-speed skip.
    for (const [[ax, ay], [bx, by]] of this.shipSegments()) {
      for (const [[cx, cy], [dx, dy]] of this.level.walls) {
        if (segsIntersect(ax, ay, bx, by, cx, cy, dx, dy)) {
          this.exploded = true;
          return;
        }
      }
    }
    for (let i = 0; i < this.level.pickups.length; i++) {
      if (this.collected[i]) continue;
      const [px, py] = this.level.pickups[i];
      const ddx = this.x - px, ddy = this.y - py;
      if (ddx * ddx + ddy * ddy <= PICKUP_COLLECT_R2) {
        this.collected[i] = true;
        this.remaining--;
        this.justCollected = i;
      }
    }
    if (this.remaining === 0) this.completed = true;
  }
}

// ---------------------------------------------------------------- app state
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
let W = 0, H = 0, DPR = 1;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W * DPR; canvas.height = H * DPR;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
}
window.addEventListener('resize', resize);
resize();

const state = {
  screen: 'menu',            // menu | playing | dead | won
  levelKey: null,
  sim: null,
  controlMode: localStorage.getItem('sa_mode') || 'swoop',   // swoop | classic
  cam: { x: 0, y: 0, zoom: 0.5 },
  particles: [],
  ghost: null,               // best-run tape being replayed: [[x,y,rot],...]
  tape: [],                  // current run recording
  newBest: false,
  deathTime: 0,
  frame: 0,
};

const bestKey = k => 'sa_best_' + k;
function loadBest(k) {
  try { return JSON.parse(localStorage.getItem(bestKey(k))); } catch { return null; }
}
function saveBest(k, time, tape) {
  try {
    localStorage.setItem(bestKey(k), JSON.stringify({ time, tape }));
  } catch { /* tape too large for quota — save time only */
    try { localStorage.setItem(bestKey(k), JSON.stringify({ time, tape: [] })); } catch {}
  }
}

// ------------------------------------------------------------------- input
// touches: map id -> {x, y, zone}
const touches = new Map();
const keys = new Set();

function zoneFor(x, y) {
  if (state.controlMode !== 'classic') return 'steer';
  if (y < H * 0.62) return 'steer';                 // upper area: nothing in classic
  if (x < W * 0.25) return 'left';
  if (x < W * 0.5) return 'right';
  return 'thrust';
}

function onTouch(e) {
  e.preventDefault();
  for (const t of e.changedTouches) {
    if (e.type === 'touchstart') {
      touches.set(t.identifier, { x: t.clientX, y: t.clientY, zone: zoneFor(t.clientX, t.clientY) });
      handleTap(t.clientX, t.clientY);
    } else if (e.type === 'touchmove') {
      const rec = touches.get(t.identifier);
      if (rec) { rec.x = t.clientX; rec.y = t.clientY; }
    } else {
      touches.delete(t.identifier);
    }
  }
}
for (const ev of ['touchstart', 'touchmove', 'touchend', 'touchcancel'])
  canvas.addEventListener(ev, onTouch, { passive: false });

// Mouse fallback for desktop testing.
let mouseDown = false;
canvas.addEventListener('mousedown', e => {
  mouseDown = true;
  touches.set('m', { x: e.clientX, y: e.clientY, zone: zoneFor(e.clientX, e.clientY) });
  handleTap(e.clientX, e.clientY);
});
canvas.addEventListener('mousemove', e => {
  if (mouseDown) { const r = touches.get('m'); if (r) { r.x = e.clientX; r.y = e.clientY; } }
});
window.addEventListener('mouseup', () => { mouseDown = false; touches.delete('m'); });
window.addEventListener('keydown', e => keys.add(e.code));
window.addEventListener('keyup', e => keys.delete(e.code));

// UI hit regions computed each frame during render.
let uiHits = [];
function handleTap(x, y) {
  for (const h of uiHits) {
    if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) { h.fn(); return; }
  }
  if (state.screen === 'dead' && performance.now() - state.deathTime > 400) startLevel(state.levelKey);
  else if (state.screen === 'won') { state.screen = 'menu'; }
}

function currentControls() {
  // keyboard
  let left = keys.has('ArrowLeft') || keys.has('KeyA');
  let right = keys.has('ArrowRight') || keys.has('KeyD');
  let thrust = keys.has('ArrowUp') || keys.has('KeyW') || keys.has('Space');

  if (state.controlMode === 'classic') {
    for (const t of touches.values()) {
      if (t.zone === 'left') left = true;
      else if (t.zone === 'right') right = true;
      else if (t.zone === 'thrust') thrust = true;
    }
  } else {
    // Swoop: hold anywhere — ship steers toward your finger (rotation still
    // rate-limited by the engine's 4.363 rad/s) and thrusts while held.
    let steer = null;
    for (const t of touches.values()) { steer = t; break; }
    if (steer && state.sim) {
      const [sx, sy] = worldToScreen(state.sim.x, state.sim.y);
      const dx = steer.x - sx, dy = steer.y - sy;
      if (dx * dx + dy * dy > 24 * 24) {          // deadzone right on the ship
        const want = Math.atan2(dy, dx) + Math.PI / 2;   // heading with 0 = up
        let diff = want - state.sim.rot;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        if (diff > 0.02) right = true;
        else if (diff < -0.02) left = true;
      }
      thrust = true;
    }
  }
  return [left, right, thrust];
}

// ------------------------------------------------------------------ camera
function worldToScreen(wx, wy) {
  return [(wx - state.cam.x) * state.cam.zoom + W / 2,
          (wy - state.cam.y) * state.cam.zoom + H / 2];
}

function updateCamera(snap = false) {
  const sim = state.sim;
  if (!sim) return;
  // Lookahead by velocity so you see where you're going at speed.
  const tx = sim.x + sim.vx * 0.4;
  const ty = sim.y + sim.vy * 0.4;
  const speed = Math.hypot(sim.vx, sim.vy);
  // Zoom out as you go faster; base zoom fits ~750 world px on short axis.
  const base = Math.min(W, H) / 750;
  const tz = base / (1 + speed / 900);
  if (snap) {
    state.cam.x = tx; state.cam.y = ty; state.cam.zoom = tz;
  } else {
    state.cam.x += (tx - state.cam.x) * 0.08;
    state.cam.y += (ty - state.cam.y) * 0.08;
    state.cam.zoom += (tz - state.cam.zoom) * 0.05;
  }
}

// ------------------------------------------------------------------- flow
const levels = {};
for (const k of LEVEL_META.order) levels[k] = parseLevel(LEVEL_DATA[k]);

function startLevel(key) {
  state.levelKey = key;
  state.sim = new Sim(levels[key]);
  state.screen = 'playing';
  state.particles = [];
  state.tape = [];
  state.newBest = false;
  const best = loadBest(key);
  state.ghost = best && best.tape && best.tape.length ? best.tape : null;
  updateCamera(true);
}

function vibrate(ms) { if (navigator.vibrate) navigator.vibrate(ms); }

function explodeFx(x, y) {
  for (let i = 0; i < 26; i++) {
    const a = Math.random() * Math.PI * 2, sp = 60 + Math.random() * 360;
    state.particles.push({
      x, y, vx: Math.cos(a) * sp + state.sim.vx * 0.3, vy: Math.sin(a) * sp + state.sim.vy * 0.3,
      life: 0.6 + Math.random() * 0.7, c: Math.random() < 0.5 ? '#ff3030' : '#ffa030',
    });
  }
  vibrate(120);
}
function pickupFx(x, y) {
  for (let i = 0; i < 14; i++) {
    const a = Math.random() * Math.PI * 2, sp = 40 + Math.random() * 160;
    state.particles.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, life: 0.5, c: '#00ffff' });
  }
  vibrate(25);
}

// -------------------------------------------------------------- simulation
let acc = 0, lastT = performance.now();
function tickSim() {
  const sim = state.sim;
  const [l, r, th] = currentControls();
  sim.step(l, r, th);
  state.tape.push([Math.round(sim.x * 10) / 10, Math.round(sim.y * 10) / 10, Math.round(sim.rot * 1000) / 1000]);
  if (sim.justCollected !== null) {
    const [px, py] = sim.level.pickups[sim.justCollected];
    pickupFx(px, py);
  }
  if (sim.exploded) {
    explodeFx(sim.x, sim.y);
    state.screen = 'dead';
    state.deathTime = performance.now();
  } else if (sim.completed) {
    state.screen = 'won';
    const best = loadBest(state.levelKey);
    if (!best || sim.time < best.time) {
      saveBest(state.levelKey, sim.time, state.tape);
      state.newBest = true;
    }
    vibrate([30, 40, 30]);
  }
}

// --------------------------------------------------------------- rendering
function fmtTime(t) {
  return t.toFixed(2) + 's';
}

function drawWorld() {
  const sim = state.sim, cam = state.cam, z = cam.zoom;
  // grid
  const grid = 100 * z >= 18 ? 100 : 500;
  ctx.strokeStyle = 'rgba(0,80,30,0.35)';
  ctx.lineWidth = 1;
  const x0 = cam.x - W / 2 / z, x1 = cam.x + W / 2 / z;
  const y0 = cam.y - H / 2 / z, y1 = cam.y + H / 2 / z;
  ctx.beginPath();
  for (let gx = Math.floor(x0 / grid) * grid; gx <= x1; gx += grid) {
    const [sx] = worldToScreen(gx, 0); ctx.moveTo(sx, 0); ctx.lineTo(sx, H);
  }
  for (let gy = Math.floor(y0 / grid) * grid; gy <= y1; gy += grid) {
    const [, sy] = worldToScreen(0, gy); ctx.moveTo(0, sy); ctx.lineTo(W, sy);
  }
  ctx.stroke();

  // walls (neon green, glow)
  ctx.save();
  ctx.shadowColor = '#00ff41';
  ctx.shadowBlur = 8;
  ctx.strokeStyle = '#00ff41';
  ctx.lineWidth = Math.max(1.5, 2.5 * z);
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [[ax, ay], [bx, by]] of sim.level.walls) {
    const [sax, say] = worldToScreen(ax, ay);
    const [sbx, sby] = worldToScreen(bx, by);
    if ((sax < -80 && sbx < -80) || (sax > W + 80 && sbx > W + 80) ||
        (say < -80 && sby < -80) || (say > H + 80 && sby > H + 80)) continue;
    ctx.moveTo(sax, say); ctx.lineTo(sbx, sby);
  }
  ctx.stroke();
  ctx.restore();

  // pickups (pulsing cyan)
  const pulse = 0.75 + 0.25 * Math.sin(state.frame * 0.09);
  for (let i = 0; i < sim.level.pickups.length; i++) {
    if (sim.collected[i]) continue;
    const [px, py] = sim.level.pickups[i];
    const [sx, sy] = worldToScreen(px, py);
    if (sx < -60 || sx > W + 60 || sy < -60 || sy > H + 60) continue;
    ctx.save();
    ctx.shadowColor = '#00ffff'; ctx.shadowBlur = 14;
    ctx.strokeStyle = 'rgba(0,255,255,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 16 * z * 3 * pulse, 0, 7); ctx.stroke();
    ctx.fillStyle = '#00ffff';
    ctx.beginPath(); ctx.arc(sx, sy, Math.max(3, 8 * z * 3 * pulse), 0, 7); ctx.fill();
    ctx.restore();
  }

  // off-screen pickup arrows (guidance — the maps are much bigger than a phone)
  for (let i = 0; i < sim.level.pickups.length; i++) {
    if (sim.collected[i]) continue;
    const [px, py] = sim.level.pickups[i];
    const [sx, sy] = worldToScreen(px, py);
    if (sx >= 0 && sx <= W && sy >= 0 && sy <= H) continue;
    const cx = W / 2, cy = H / 2;
    const dx = sx - cx, dy = sy - cy;
    const m = 28;
    const t = Math.min(
      Math.abs(dx) > 1 ? (W / 2 - m) / Math.abs(dx) : Infinity,
      Math.abs(dy) > 1 ? (H / 2 - m) / Math.abs(dy) : Infinity);
    const ex = cx + dx * t, ey = cy + dy * t;
    const a = Math.atan2(dy, dx);
    ctx.save();
    ctx.translate(ex, ey); ctx.rotate(a);
    ctx.fillStyle = 'rgba(0,255,255,0.85)';
    ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-6, -7); ctx.lineTo(-6, 7); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
}

function drawShipAt(x, y, rot, color, alpha, thrusting) {
  const c = Math.cos(rot), s = Math.sin(rot), z = state.cam.zoom;
  const tv = SHIP_VERTS.map(([vx, vy]) => worldToScreen(vx * c - vy * s + x, vx * s + vy * c + y));
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.shadowColor = color; ctx.shadowBlur = 10;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.5, 2.5 * z);
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(...tv[RENDER_TRI[0]]);
  ctx.lineTo(...tv[RENDER_TRI[1]]);
  ctx.lineTo(...tv[RENDER_TRI[2]]);
  ctx.closePath();
  ctx.stroke();
  if (thrusting) {
    const flick = 0.7 + 0.3 * Math.random();
    ctx.strokeStyle = '#ffb300';
    ctx.shadowColor = '#ff8800';
    ctx.beginPath();
    ctx.moveTo(...tv[FLAME_TRI[0]]);
    const [fx, fy] = tv[FLAME_TRI[1]];
    const [mx, my] = worldToScreen(x, y);
    ctx.lineTo(mx + (fx - mx) * flick, my + (fy - my) * flick);
    ctx.lineTo(...tv[FLAME_TRI[2]]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawParticles(dt) {
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const p = state.particles[i];
    p.life -= dt;
    if (p.life <= 0) { state.particles.splice(i, 1); continue; }
    p.x += p.vx * dt; p.y += p.vy * dt;
    p.vy += GRAVITY * dt;
    const [sx, sy] = worldToScreen(p.x, p.y);
    ctx.globalAlpha = Math.min(1, p.life * 2);
    ctx.fillStyle = p.c;
    ctx.fillRect(sx - 2, sy - 2, 4, 4);
  }
  ctx.globalAlpha = 1;
}

function button(x, y, w, h, label, fn, opts = {}) {
  uiHits.push({ x, y, w, h, fn });
  ctx.save();
  ctx.fillStyle = opts.fill || 'rgba(0,255,65,0.08)';
  ctx.strokeStyle = opts.stroke || '#00ff41';
  ctx.lineWidth = 1.5;
  roundRect(x, y, w, h, 12);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = opts.color || '#00ff41';
  ctx.font = (opts.font || '600 17px') + ' -apple-system, system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, y + h / 2 + 1);
  ctx.restore();
}
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawHUD() {
  const sim = state.sim;
  ctx.save();
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.font = '700 26px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(fmtTime(sim.time), W / 2, 14);
  ctx.font = '600 14px -apple-system, system-ui, sans-serif';
  ctx.fillStyle = '#00ffff';
  const total = sim.level.pickups.length;
  ctx.fillText('◈ ' + (total - sim.remaining) + ' / ' + total, W / 2, 46);
  const best = loadBest(state.levelKey);
  if (best) {
    ctx.fillStyle = 'rgba(255,0,255,0.9)';
    ctx.fillText('best ' + fmtTime(best.time), W / 2, 66);
  }
  ctx.restore();
  button(12, 12, 44, 44, '✕', () => { state.screen = 'menu'; }, { stroke: 'rgba(0,255,65,0.5)' });
  button(W - 56, 12, 44, 44, '↺', () => startLevel(state.levelKey), { stroke: 'rgba(0,255,65,0.5)' });

  if (state.controlMode === 'classic') {
    // control zone hints
    ctx.save();
    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = '#00ff41';
    ctx.setLineDash([6, 8]);
    const y = H * 0.62;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W / 2, y); ctx.moveTo(W * 0.25, y); ctx.lineTo(W * 0.25, H);
    ctx.moveTo(W / 2, y); ctx.lineTo(W / 2, H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = '#00ff41';
    ctx.font = '700 30px -apple-system, system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('⟲', W * 0.125, H * 0.81);
    ctx.fillText('⟳', W * 0.375, H * 0.81);
    ctx.fillText('▲', W * 0.75, H * 0.81);
    ctx.restore();
  }
}

function drawMenu() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.textAlign = 'center';
  ctx.shadowColor = '#00ff41'; ctx.shadowBlur = 18;
  ctx.fillStyle = '#00ff41';
  ctx.font = '800 44px -apple-system, system-ui, sans-serif';
  ctx.fillText('SPACE ACE', W / 2, H * 0.14);
  ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '500 15px -apple-system, system-ui, sans-serif';
  ctx.fillText('mobile proof of concept', W / 2, H * 0.14 + 30);
  ctx.restore();

  const bw = Math.min(W - 48, 360), bx = (W - bw) / 2;
  let by = H * 0.24;
  for (const k of LEVEL_META.order) {
    const lvl = levels[k];
    const best = loadBest(k);
    const name = LEVEL_META.names[k] || ('Level ' + k);
    button(bx, by, bw, 64, '', () => startLevel(k));
    ctx.save();
    ctx.fillStyle = '#00ff41';
    ctx.font = '700 18px -apple-system, system-ui, sans-serif';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(name, bx + 18, by + 24);
    ctx.font = '500 13px -apple-system, system-ui, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText(lvl.pickups.length + ' pickups', bx + 18, by + 45);
    if (best) {
      ctx.textAlign = 'right';
      ctx.fillStyle = '#ff00ff';
      ctx.font = '600 15px ui-monospace, Menlo, monospace';
      ctx.fillText(fmtTime(best.time), bx + bw - 18, by + 32);
    }
    ctx.restore();
    by += 76;
  }

  by += 8;
  const modeLabel = state.controlMode === 'swoop' ? 'Controls: SWOOP (one thumb)' : 'Controls: CLASSIC (buttons)';
  button(bx, by, bw, 50, modeLabel, () => {
    state.controlMode = state.controlMode === 'swoop' ? 'classic' : 'swoop';
    localStorage.setItem('sa_mode', state.controlMode);
  }, { stroke: 'rgba(0,255,255,0.7)', color: '#00ffff' });

  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = '400 12px -apple-system, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(state.controlMode === 'swoop'
    ? 'Hold anywhere: ship steers toward your finger and thrusts. Release to coast.'
    : 'Bottom corners: rotate left / right · right side: thrust.', W / 2, by + 74);
  ctx.fillText('Collect every ◈ without touching a wall. Gravity is real. Momentum is king.', W / 2, by + 92);
  ctx.restore();
}

function drawOverlay() {
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.textAlign = 'center';
  if (state.screen === 'dead') {
    ctx.fillStyle = '#ff3030';
    ctx.shadowColor = '#ff0000'; ctx.shadowBlur = 16;
    ctx.font = '800 38px -apple-system, system-ui, sans-serif';
    ctx.fillText('CRASHED', W / 2, H * 0.4);
    ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.font = '500 16px -apple-system, system-ui, sans-serif';
    ctx.fillText('tap to retry', W / 2, H * 0.4 + 36);
  } else {
    ctx.fillStyle = '#00ff41';
    ctx.shadowColor = '#00ff41'; ctx.shadowBlur = 16;
    ctx.font = '800 34px -apple-system, system-ui, sans-serif';
    ctx.fillText('LEVEL CLEAR', W / 2, H * 0.36);
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 30px ui-monospace, Menlo, monospace';
    ctx.fillText(fmtTime(state.sim.time), W / 2, H * 0.36 + 46);
    if (state.newBest) {
      ctx.fillStyle = '#ff00ff';
      ctx.font = '700 17px -apple-system, system-ui, sans-serif';
      ctx.fillText('★ NEW BEST — your ghost is saved', W / 2, H * 0.36 + 78);
    }
    button(W / 2 - 130, H * 0.55, 125, 52, 'RETRY', () => startLevel(state.levelKey));
    button(W / 2 + 5, H * 0.55, 125, 52, 'LEVELS', () => { state.screen = 'menu'; });
  }
  ctx.restore();
}

// --------------------------------------------------------------- main loop
function frame(now) {
  requestAnimationFrame(frame);
  const dtReal = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
  state.frame++;
  uiHits = [];

  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);

  if (state.screen === 'menu') { drawMenu(); return; }

  if (state.screen === 'playing') {
    acc += dtReal;
    while (acc >= TICK && state.screen === 'playing') {
      acc -= TICK;
      tickSim();
    }
  }
  updateCamera();
  drawWorld();

  // ghost (best run) — magenta, like the dashboard's AI ghost
  if (state.ghost && state.sim.ticks < state.ghost.length && state.screen === 'playing') {
    const [gx, gy, gr] = state.ghost[state.sim.ticks];
    drawShipAt(gx, gy, gr, '#ff00ff', 0.35, false);
  }

  if (!state.sim.exploded) {
    const [, , th] = state.screen === 'playing' ? currentControls() : [0, 0, false];
    drawShipAt(state.sim.x, state.sim.y, state.sim.rot, '#00ff41', 1, th);
  }
  drawParticles(dtReal);
  drawHUD();
  if (state.screen === 'dead' || state.screen === 'won') drawOverlay();
}
requestAnimationFrame(frame);

// Deep link for testing: index.html?level=7 jumps straight into a level.
try {
  const lk = new URLSearchParams(location.search).get('level');
  if (lk && levels[lk]) startLevel(lk);
} catch { /* non-browser harness */ }
