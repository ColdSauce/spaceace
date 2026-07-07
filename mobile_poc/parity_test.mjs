// Physics parity test: replays the same deterministic action tape through
// the JS Sim (game.js) and the real Rust engine (PyGameInstance via uv),
// then diffs the trajectories tick by tick.
//
// Usage:  node mobile_poc/parity_test.mjs   (from the repo root)
//
// Expected result: max position delta well under 1 px over 600 ticks
// (the Rust engine is f32, JS is f64, so tiny drift is inherent), and
// identical pickup/crash event ticks. Note the engine skips every other
// wall check above ~316 px/s and the JS sim deliberately does not (strict
// mode) — the tape below stays in open space at moderate speed so wall
// events are comparable.

import { readFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { createContext, runInContext } from 'node:vm';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
// The Rust side needs a checkout with a built venv; override with
// SPACEACE_ROOT if running from a worktree that hasn't built one.
const repoRoot = process.env.SPACEACE_ROOT || join(here, '..');

// Deterministic tape: thrust up, arc right, coast, arc left — 600 ticks
// inside level 5000's box (523x455 interior) without crashing.
const LEVEL = 5000;
// Duty-cycled hover (thrust 1-in-4 ticks: 400/4 = gravity 100) with
// symmetric rotation wiggles, so the ship stays airborne for the full
// tape and every control path (left/right/thrust/coast) gets exercised.
const tape = [];
for (let i = 0; i < 600; i++) {
  const thrust = i % 4 === 0 ? 1 : 0;
  const ph = i % 80;
  const left = ph < 10 ? 1 : 0;
  const right = ph >= 10 && ph < 20 ? 1 : 0;
  tape.push([left, right, thrust]);
}

// ---- JS side: load game.js with DOM shims and drive its Sim ------------
const noop = () => {};
const ctxStub = new Proxy({}, { get: (o, k) => (k in o ? o[k] : noop), set: (o, k, v) => (o[k] = v, true) });
const shim = {
  document: { getElementById: () => ({ getContext: () => ctxStub, addEventListener: noop, style: {}, width: 0, height: 0 }) },
  window: { innerWidth: 400, innerHeight: 800, devicePixelRatio: 1, addEventListener: noop },
  localStorage: { getItem: () => null, setItem: noop },
  navigator: {},
  performance: { now: () => 0 },
  requestAnimationFrame: noop,
  Math, JSON, console,
};
shim.globalThis = shim;
const vmCtx = createContext(shim);
runInContext(readFileSync(join(here, 'levels.js'), 'utf8'), vmCtx);
runInContext(readFileSync(join(here, 'game.js'), 'utf8'), vmCtx);
runInContext(`
  __sim = new Sim(parseLevel(LEVEL_DATA["${LEVEL}"]));
  __trace = [];
  for (const [l, r, t] of ${JSON.stringify(tape)}) {
    __sim.step(l, r, t);
    __trace.push([__sim.x, __sim.y, __sim.exploded ? 1 : 0, __sim.remaining]);
    if (__sim.exploded || __sim.completed) break;
  }
`, vmCtx);
const jsTrace = runInContext('__trace', vmCtx);

// ---- Rust side: same tape through PyGameInstance ------------------------
const py = `
import json, sys
import numpy as np
from spaceace.core.env import SpaceAceDirectEnv
tape = json.load(open(sys.argv[1]))
env = SpaceAceDirectEnv(level=${LEVEL}, max_steps=10000)
env.reset()
trace = []
for l, r, t in tape:
    _, _, term, trunc, step_info = env.step(np.array([l, r, t], dtype=np.float32))
    info = env.get_info()
    pos = info["ship_position"]
    trace.append([pos["x"], pos["y"], int(info["ship_exploded"]), int(step_info["pickups_remaining"])])
    if term or trunc:
        break
print(json.dumps(trace))
`;
const tmp = join(here, '.parity_tape.json');
const { writeFileSync, unlinkSync } = await import('node:fs');
writeFileSync(tmp, JSON.stringify(tape));
let rustTrace;
try {
  const out = execSync(`uv run python -c '${py.replace(/'/g, "'\\''")}' ${JSON.stringify(tmp)}`, {
    cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'],
  });
  rustTrace = JSON.parse(out.trim().split('\n').pop());
} finally { unlinkSync(tmp); }

// ---- diff ----------------------------------------------------------------
const n = Math.min(jsTrace.length, rustTrace.length);
let maxD = 0, maxTick = -1, eventMismatch = null;
for (let i = 0; i < n; i++) {
  const [jx, jy, je, jr] = jsTrace[i];
  const [rx, ry, re, rr] = rustTrace[i];
  const d = Math.hypot(jx - rx, jy - ry);
  if (d > maxD) { maxD = d; maxTick = i; }
  if ((je !== re || jr !== rr) && eventMismatch === null) eventMismatch = i;
}
console.log(`ticks compared: ${n} (js ran ${jsTrace.length}, rust ran ${rustTrace.length})`);
console.log(`max position delta: ${maxD.toFixed(6)} px at tick ${maxTick}`);
console.log(`event mismatch (exploded/pickups): ${eventMismatch === null ? 'none' : 'tick ' + eventMismatch}`);
const pass = n > 0 && jsTrace.length === rustTrace.length && maxD < 1.0 && eventMismatch === null;
console.log(pass ? 'PARITY: PASS' : 'PARITY: FAIL');
process.exit(pass ? 0 : 1);
