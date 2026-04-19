/**
 * SpaceAce game renderer — extracted from web/index.html.
 * Same ship vertices, green wireframe style, camera system, minimap.
 *
 * Usage:
 *   import { renderFrame } from './renderer.js';
 *   renderFrame(canvas, replay, frameIndex);
 */

// Ship vertices — exact match of web/index.html
const SHIP_VERTS = [
  {x: 0,      y:-36.5},  // 0 nose
  {x:-19,     y: 23.5},  // 1
  {x:-24,     y: 23.5},  // 2
  {x:-15.675, y: 13},    // 3
  {x: 19,     y: 23.5},  // 4
  {x: 24,     y: 23.5},  // 5
  {x: 15.675, y: 13},    // 6
  {x: 0,      y: 67.45}, // 7 thruster tip
  {x:-14.1075,y: 13},    // 8
  {x: 14.1075,y: 13},    // 9
];

const PICKUP_RADIUS = 10;

function worldToScreen(wx, wy, camX, camY, zoom, cw, ch) {
  return {
    x: (wx - camX) * zoom + cw * 0.5,
    y: (wy - camY) * zoom + ch * 0.5,
  };
}

function drawMap(ctx, walls, camX, camY, zoom, cw, ch) {
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [x1, y1, x2, y2] of walls) {
    const s1 = worldToScreen(x1, y1, camX, camY, zoom, cw, ch);
    const s2 = worldToScreen(x2, y2, camX, camY, zoom, cw, ch);
    // off-screen culling
    if ((s1.x < -50 && s2.x < -50) || (s1.x > cw+50 && s2.x > cw+50) ||
        (s1.y < -50 && s2.y < -50) || (s1.y > ch+50 && s2.y > ch+50)) continue;
    ctx.moveTo(s1.x, s1.y);
    ctx.lineTo(s2.x, s2.y);
  }
  ctx.stroke();
}

function drawPickups(ctx, pickups, collectedFlags, camX, camY, zoom, cw, ch) {
  const scaledR = PICKUP_RADIUS * zoom;

  // uncollected
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < pickups.length; i++) {
    if (collectedFlags && collectedFlags[i]) continue;
    const [px, py] = pickups[i];
    const s = worldToScreen(px, py, camX, camY, zoom, cw, ch);
    if (s.x < -50 || s.x > cw+50 || s.y < -50 || s.y > ch+50) continue;
    ctx.moveTo(s.x + scaledR, s.y);
    ctx.arc(s.x, s.y, scaledR, 0, Math.PI * 2);
  }
  ctx.stroke();

  // collected — dim
  ctx.strokeStyle = 'rgba(100,100,100,0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const smallR = scaledR * 0.5;
  for (let i = 0; i < pickups.length; i++) {
    if (!collectedFlags || !collectedFlags[i]) continue;
    const [px, py] = pickups[i];
    const s = worldToScreen(px, py, camX, camY, zoom, cw, ch);
    if (s.x < -50 || s.x > cw+50 || s.y < -50 || s.y > ch+50) continue;
    ctx.moveTo(s.x + smallR, s.y);
    ctx.arc(s.x, s.y, smallR, 0, Math.PI * 2);
  }
  ctx.stroke();
}

function drawShip(ctx, f, camX, camY, zoom, cw, ch) {
  const cos = Math.cos(f.rotation);
  const sin = Math.sin(f.rotation);

  function vert(i) {
    const v = SHIP_VERTS[i];
    const rx = v.x * cos - v.y * sin;
    const ry = v.x * sin + v.y * cos;
    return worldToScreen(f.x + rx, f.y + ry, camX, camY, zoom, cw, ch);
  }

  // Body: 3→6, then 2→1→0→4→5
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  let s = vert(3); ctx.moveTo(s.x, s.y);
  s = vert(6); ctx.lineTo(s.x, s.y);
  s = vert(2); ctx.moveTo(s.x, s.y);
  for (const i of [1, 0, 4, 5]) { s = vert(i); ctx.lineTo(s.x, s.y); }
  ctx.stroke();

  // Thruster flame: 8→7→9
  if (f.action[2] > 0) {
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    s = vert(8); ctx.moveTo(s.x, s.y);
    s = vert(7); ctx.lineTo(s.x, s.y);
    s = vert(9); ctx.lineTo(s.x, s.y);
    ctx.stroke();
  }
}

function drawPath(ctx, path, camX, camY, zoom, cw, ch) {
  if (path.length < 2) return;
  ctx.strokeStyle = 'rgba(255, 0, 255, 0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  const s0 = worldToScreen(path[0][0], path[0][1], camX, camY, zoom, cw, ch);
  ctx.moveTo(s0.x, s0.y);
  for (let i = 1; i < path.length; i++) {
    const s = worldToScreen(path[i][0], path[i][1], camX, camY, zoom, cw, ch);
    ctx.lineTo(s.x, s.y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw a small diamond at the path target (last point)
  const end = worldToScreen(path[path.length-1][0], path[path.length-1][1], camX, camY, zoom, cw, ch);
  const d = 5;
  ctx.fillStyle = 'rgba(255, 0, 255, 0.7)';
  ctx.beginPath();
  ctx.moveTo(end.x, end.y - d);
  ctx.lineTo(end.x + d, end.y);
  ctx.lineTo(end.x, end.y + d);
  ctx.lineTo(end.x - d, end.y);
  ctx.closePath();
  ctx.fill();
}

function drawRaycasts(ctx, f, camX, camY, zoom, cw, ch) {
  if (!f.wall8) return;
  const cos = Math.cos(f.rotation);
  const sin = Math.sin(f.rotation);

  // 8 coarse directions in ship-local space (forward, fwd-right, right, ...)
  const dirs8 = [
    [0, -1], [0.707, -0.707], [1, 0], [0.707, 0.707],
    [0, 1], [-0.707, 0.707], [-1, 0], [-0.707, -0.707],
  ];

  // 16 fine directions (15° spacing, interleaved with the 8 coarse)
  const fineAngles = [15, 30, 60, 75, 105, 120, 150, 165, 195, 210, 240, 255, 285, 300, 330, 345];
  const dirs16 = fineAngles.map(deg => {
    const rad = deg * Math.PI / 180;
    return [Math.sin(rad), -Math.cos(rad)];
  });

  function drawRay(localDx, localDy, dist, color, width) {
    // Rotate to world space
    const wdx = localDx * cos - localDy * sin;
    const wdy = localDx * sin + localDy * cos;
    const endX = f.x + wdx * dist;
    const endY = f.y + wdy * dist;
    const s1 = worldToScreen(f.x, f.y, camX, camY, zoom, cw, ch);
    const s2 = worldToScreen(endX, endY, camX, camY, zoom, cw, ch);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(s1.x, s1.y);
    ctx.lineTo(s2.x, s2.y);
    ctx.stroke();
  }

  // Draw fine rays (dimmer)
  if (f.wall16) {
    for (let i = 0; i < 16; i++) {
      drawRay(dirs16[i][0], dirs16[i][1], f.wall16[i], 'rgba(0,150,255,0.15)', 1);
    }
  }

  // Draw coarse rays (brighter)
  for (let i = 0; i < 8; i++) {
    const dist = f.wall8[i];
    const danger = dist < 80;
    const color = danger ? 'rgba(255,80,80,0.5)' : 'rgba(0,150,255,0.3)';
    drawRay(dirs8[i][0], dirs8[i][1], dist, color, danger ? 1.5 : 1);
  }

  // Draw pathfinder direction arrow (from path data)
  if (f.path && f.path.length >= 2) {
    const target = f.path[f.path.length - 1];
    const dx = target[0] - f.x, dy = target[1] - f.y;
    const mag = Math.sqrt(dx * dx + dy * dy);
    if (mag > 1) {
      const arrowLen = Math.min(80, mag * 0.3);
      const endX = f.x + (dx / mag) * arrowLen;
      const endY = f.y + (dy / mag) * arrowLen;
      const s1 = worldToScreen(f.x, f.y, camX, camY, zoom, cw, ch);
      const s2 = worldToScreen(endX, endY, camX, camY, zoom, cw, ch);
      ctx.strokeStyle = 'rgba(255,0,255,0.8)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(s1.x, s1.y);
      ctx.lineTo(s2.x, s2.y);
      ctx.stroke();
      // Arrowhead
      const angle = Math.atan2(s2.y - s1.y, s2.x - s1.x);
      const headLen = 8;
      ctx.beginPath();
      ctx.moveTo(s2.x, s2.y);
      ctx.lineTo(s2.x - headLen * Math.cos(angle - 0.4), s2.y - headLen * Math.sin(angle - 0.4));
      ctx.moveTo(s2.x, s2.y);
      ctx.lineTo(s2.x - headLen * Math.cos(angle + 0.4), s2.y - headLen * Math.sin(angle + 0.4));
      ctx.stroke();
    }
  }
}

function drawVelocityVector(ctx, f, camX, camY, zoom, cw, ch) {
  const speed = Math.sqrt(f.vx * f.vx + f.vy * f.vy);
  if (speed < 0.5) return;
  const scale = 2.0; // exaggerate for visibility
  const endX = f.x + f.vx * scale;
  const endY = f.y + f.vy * scale;
  const s1 = worldToScreen(f.x, f.y, camX, camY, zoom, cw, ch);
  const s2 = worldToScreen(endX, endY, camX, camY, zoom, cw, ch);
  ctx.strokeStyle = 'rgba(255,255,0,0.7)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(s1.x, s1.y);
  ctx.lineTo(s2.x, s2.y);
  ctx.stroke();
}

function drawHUD(ctx, f, frameIdx, totalFrames) {
  const actionNames = ['coast','thrust','left','left+thr','right','right+thr'];
  const a = f.action;
  // Decode MultiDiscrete [rot_left, rot_right, thrust] to action name
  let actName;
  if (a[0] === 0 && a[1] === 0 && a[2] === 0) actName = 'COAST';
  else if (a[0] === 0 && a[1] === 0 && a[2] === 1) actName = 'THRUST';
  else if (a[0] === 1 && a[1] === 0 && a[2] === 0) actName = 'LEFT';
  else if (a[0] === 1 && a[1] === 0 && a[2] === 1) actName = 'LEFT+THR';
  else if (a[0] === 0 && a[1] === 1 && a[2] === 0) actName = 'RIGHT';
  else if (a[0] === 0 && a[1] === 1 && a[2] === 1) actName = 'RIGHT+THR';
  else actName = `[${a}]`;

  const speed = Math.sqrt(f.vx * f.vx + f.vy * f.vy);
  const rotDeg = (f.rotation * 180 / Math.PI).toFixed(0);

  ctx.font = '12px monospace';
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(4, 4, 220, 110);

  ctx.fillStyle = '#00FF00';
  const lines = [
    `Step ${frameIdx}/${totalFrames}  Pickups: ${f.pickups_remaining}`,
    `Action: ${actName}  Reward: ${f.reward.toFixed(2)}`,
    `Pos: (${f.x.toFixed(0)}, ${f.y.toFixed(0)})`,
    `Vel: (${f.vx.toFixed(1)}, ${f.vy.toFixed(1)})  Spd: ${speed.toFixed(1)}`,
    `Rot: ${rotDeg}°`,
  ];
  // Closest wall distance
  if (f.wall8) {
    const minWall = Math.min(...f.wall8);
    lines.push(`Closest wall: ${minWall.toFixed(0)}px`);
  }
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], 8, 18 + i * 16);
  }
}

function drawMiniMap(ctx, walls, bounds, shipX, shipY, cw, ch) {
  const mmW = 160, mmH = 110;
  const mmX = cw - mmW - 10, mmY = 10;

  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(mmX, mmY, mmW, mmH);

  const mapW = bounds.max_x - bounds.min_x;
  const mapH = bounds.max_y - bounds.min_y;
  const scale = Math.min(mmW / mapW, mmH / mapH) * 0.9;
  const ox = mmX + mmW / 2 - (bounds.min_x + mapW / 2) * scale;
  const oy = mmY + mmH / 2 - (bounds.min_y + mapH / 2) * scale;

  // walls
  ctx.strokeStyle = 'rgba(0,255,0,0.4)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (const [x1, y1, x2, y2] of walls) {
    ctx.moveTo(ox + x1 * scale, oy + y1 * scale);
    ctx.lineTo(ox + x2 * scale, oy + y2 * scale);
  }
  ctx.stroke();

  // ship dot
  ctx.fillStyle = '#00FF00';
  ctx.beginPath();
  ctx.arc(ox + shipX * scale, oy + shipY * scale, 3, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Render a single frame of a replay onto a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {Object} replay  — { walls, bounds, pickups_initial, frames }
 * @param {number} frameIdx
 */
export function renderFrame(canvas, replay, frameIdx) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const f = replay.frames[frameIdx];
  if (!f) return;

  // Camera: center on ship, zoom to show a large area around it
  const mapW = replay.bounds.max_x - replay.bounds.min_x;
  const mapH = replay.bounds.max_y - replay.bounds.min_y;
  // Show ~3000px of world width, or fit the whole map if it's smaller
  const viewExtent = Math.max(3000, Math.min(mapW, mapH) * 0.8);
  const zoom = Math.min(W, H) / viewExtent;
  const camX = f.x;
  const camY = f.y;

  // Background
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, W, H);

  // Grid background (matching web/ style)
  ctx.strokeStyle = 'rgba(0, 255, 65, 0.06)';
  ctx.lineWidth = 1;
  const gridSize = 50;
  const startX = (-camX * zoom + W / 2) % (gridSize * zoom);
  const startY = (-camY * zoom + H / 2) % (gridSize * zoom);
  const step = gridSize * zoom;
  ctx.beginPath();
  for (let x = startX; x < W; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  for (let y = startY; y < H; y += step) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  ctx.stroke();

  drawMap(ctx, replay.walls, camX, camY, zoom, W, H);
  drawPickups(ctx, replay.pickups_initial, f.pickup_collected, camX, camY, zoom, W, H);
  if (f.path && f.path.length > 1) {
    drawPath(ctx, f.path, camX, camY, zoom, W, H);
  }
  drawRaycasts(ctx, f, camX, camY, zoom, W, H);
  drawShip(ctx, f, camX, camY, zoom, W, H);
  drawVelocityVector(ctx, f, camX, camY, zoom, W, H);
  drawMiniMap(ctx, replay.walls, replay.bounds, f.x, f.y, W, H);
  drawHUD(ctx, f, frameIdx, replay.frames.length);
}
