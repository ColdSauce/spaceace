#!/usr/bin/env python
"""Live web view of solver runs. Read-only: tails solve.py logs and the
ghost_actions sidecars; never touches the DB or the running processes.

Usage:
    uv run python scripts/watch_solver.py --log-dir /path/to/logs [--port 5051]

Then open http://localhost:5051
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RE_HEADER = re.compile(r"^L(\d+): (\d+) pickups; human ghost: ([\d.]+|None)s?")
RE_ETA = re.compile(r"spawn ETA ([\d.]+)s")
RE_SOLVE = re.compile(r"\[solve\] (.+?): (?:(\d+) ticks \(([\d.]+)s\)|no completion)")
RE_ROUND = re.compile(
    r"\[(refine|polish|suffix)\] round (\d+): (\d+) -> (\d+) ticks "
    r"\(([\d.]+)s, (?:-(\d+)|no gain), (-?[\d.]+)min left\)")
RE_BEST = re.compile(r"^best: (\d+) ticks = ([\d.]+)s")
RE_INCUMBENT = re.compile(r"incumbent sidecar: (\d+) ticks")


def parse_log(path: Path) -> dict | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    level = None
    d: dict = {"file": path.name, "human_s": None, "eta_s": None,
               "solves": [], "rounds": [], "final_s": None,
               "incumbent_ticks": None, "mtime": path.stat().st_mtime}
    for line in text.splitlines():
        m = RE_HEADER.match(line)
        if m:
            level = int(m.group(1))
            d["pickups"] = int(m.group(2))
            d["human_s"] = None if m.group(3) == "None" else float(m.group(3))
            continue
        m = RE_ETA.search(line)
        if m:
            d["eta_s"] = float(m.group(1))
            continue
        m = RE_SOLVE.search(line)
        if m:
            d["solves"].append({
                "label": m.group(1),
                "seconds": float(m.group(3)) if m.group(3) else None,
            })
            continue
        m = RE_ROUND.search(line)
        if m:
            d["rounds"].append({
                "idx": int(m.group(2)), "kind": m.group(1),
                "before": int(m.group(3)), "after": int(m.group(4)),
                "seconds": float(m.group(5)),
                "gain": int(m.group(6) or 0),
                "min_left": float(m.group(7)),
            })
            continue
        m = RE_BEST.match(line)
        if m:
            d["final_s"] = float(m.group(2))
            continue
        m = RE_INCUMBENT.search(line)
        if m:
            d["incumbent_ticks"] = int(m.group(1))
    if level is None:
        return None
    d["level"] = level
    return d


def sidecar_seconds(level: int) -> float | None:
    p = PROJECT_ROOT / "ghost_actions" / f"L{level}_tas.json"
    try:
        return json.loads(p.read_text())["ticks"] / 60.0
    except Exception:
        return None


def collect(log_dir: Path) -> dict:
    by_level: dict[int, dict] = {}
    for path in sorted(log_dir.glob("*.log")):
        d = parse_log(path)
        if d is None:
            continue
        cur = by_level.get(d["level"])
        if cur is None or d["mtime"] > cur["mtime"]:
            by_level[d["level"]] = d
    now = time.time()
    out = []
    for level, d in sorted(by_level.items()):
        d["banked_s"] = sidecar_seconds(level)
        age = now - d["mtime"]
        d["status"] = ("finished" if d["final_s"] is not None
                       else "live" if age < 180 else "stalled/stopped")
        d["age_s"] = round(age)
        out.append(d)
    return {"levels": out, "now": now}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>SpaceAce solver — live runs</title>
<style>
:root{
  --surface:#1a1a19; --panel:#222220; --text:#ffffff; --text-2:#c3c2b7;
  --muted:#7a7a72; --series:#3987e5; --good:#1baf7a; --grid:#33332f;
}
*{box-sizing:border-box} body{margin:0;background:var(--surface);color:var(--text);
  font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;padding:24px}
h1{font-size:16px;font-weight:600;margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:20px;font-size:12px}
.level{background:var(--panel);border-radius:10px;padding:18px 20px;margin-bottom:18px}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.head .lv{font-size:15px;font-weight:600}
.chip{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--grid);color:var(--text-2)}
.chip.live{border-color:var(--good);color:var(--good)}
.tiles{display:flex;gap:28px;flex-wrap:wrap;margin-bottom:14px}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .v small{font-size:13px;color:var(--text-2);font-weight:400}
.delta-good{color:var(--good)} .delta-neutral{color:var(--text-2)}
svg{display:block}
table{border-collapse:collapse;margin-top:10px;font-size:12px;width:100%}
td,th{padding:3px 10px 3px 0;text-align:right;color:var(--text-2)}
th{color:var(--muted);font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.06em}
td:first-child,th:first-child{text-align:left}
.gain{color:var(--good)}
.tip{position:fixed;pointer-events:none;background:#000;border:1px solid var(--grid);
  border-radius:6px;padding:6px 9px;font-size:12px;color:var(--text);display:none;z-index:9}
.empty{color:var(--muted)}
</style></head><body>
<h1>SpaceAce solver — live runs</h1>
<div class="sub">read-only view; refreshes every 3s · <span id="upd"></span></div>
<div id="root"></div><div class="tip" id="tip"></div>
<script>
const fmt = s => s==null ? "—" : s.toFixed(3)+"s";
function chart(rounds, human, banked){
  if(!rounds.length) return '<div class="empty">no refinement rounds yet (portfolio stage)</div>';
  const W=760,H=190,L=54,R=12,T=12,B=26;
  const xs=rounds.map(r=>r.idx), ys=rounds.map(r=>r.seconds);
  let lo=Math.min(...ys), hi=Math.max(...ys, rounds[0].before/60);
  if(human){lo=Math.min(lo,human); hi=Math.max(hi,human);}
  const pad=(hi-lo)*0.08+0.02; lo-=pad; hi+=pad;
  const X=i=>L+(W-L-R)*(xs.length<2?0.5:(i)/(xs.length-1));
  const Y=v=>T+(H-T-B)*(hi-v)/(hi-lo);
  let g='';
  const ticks=4;
  for(let i=0;i<=ticks;i++){const v=lo+(hi-lo)*i/ticks, y=Y(v);
    g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`+
       `<text x="${L-8}" y="${y+4}" text-anchor="end" fill="var(--muted)" font-size="10">${v.toFixed(1)}</text>`;}
  let ref='';
  if(human){const y=Y(human);
    ref=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="5 4"/>`+
        `<text x="${W-R}" y="${y-5}" text-anchor="end" fill="var(--text-2)" font-size="10">your PR ${human.toFixed(3)}s</text>`;}
  // step line: value holds until next round
  let dpts=rounds.map((r,i)=>[X(i),Y(r.seconds)]);
  let dstr='M'+dpts.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L');
  let dots=dpts.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="${rounds[i].gain>0?4:2.5}"
     fill="${rounds[i].gain>0?'var(--series)':'var(--panel)'}" stroke="var(--series)" stroke-width="2"
     data-i="${i}"/>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px" class="ch">
    ${g}${ref}<path d="${dstr}" fill="none" stroke="var(--series)" stroke-width="2"/>
    ${dots}
    <text x="${L}" y="${H-6}" fill="var(--muted)" font-size="10">round →</text></svg>`;
}
function render(data){
  const root=document.getElementById('root'); let html='';
  if(!data.levels.length) html='<div class="empty">no solve logs found</div>';
  for(const lv of data.levels){
    const best = lv.rounds.length ? lv.rounds[lv.rounds.length-1].seconds
               : (lv.solves.filter(s=>s.seconds).map(s=>s.seconds).sort((a,b)=>a-b)[0] ?? null);
    const shown = lv.final_s ?? best;
    const d = (lv.human_s!=null && shown!=null) ? lv.human_s - shown : null;
    const chipCls = lv.status==='live' ? 'chip live' : 'chip';
    const budget = lv.rounds.length ? Math.max(0, lv.rounds[lv.rounds.length-1].min_left) : null;
    html+=`<div class="level"><div class="head">
      <span class="lv">Level ${lv.level}</span>
      <span class="${chipCls}">${lv.status}${lv.status!=='finished'?' · updated '+lv.age_s+'s ago':''}</span>
      <span class="chip">${lv.file}</span></div>
      <div class="tiles">
        <div class="tile"><div class="k">current best</div><div class="v">${fmt(shown)}</div></div>
        <div class="tile"><div class="k">vs your PR</div><div class="v ${d>0?'delta-good':'delta-neutral'}">${d==null?'—':(d>0?'−':'+')+Math.abs(d).toFixed(3)+'s'} <small>${d>0?'ahead':''}</small></div></div>
        <div class="tile"><div class="k">banked (sidecar)</div><div class="v">${fmt(lv.banked_s)}</div></div>
        <div class="tile"><div class="k">budget left</div><div class="v">${budget==null?'—':budget.toFixed(1)+'<small>min</small>'}</div></div>
      </div>
      ${chart(lv.rounds, lv.human_s, lv.banked_s)}
      ${lv.rounds.length?`<table><tr><th>stage</th><th>round</th><th>ticks</th><th>time</th><th>gain</th></tr>`+
        lv.rounds.slice(-6).reverse().map(r=>`<tr><td>${r.kind}</td><td>${r.idx}</td><td>${r.after}</td>
        <td>${r.seconds.toFixed(3)}s</td><td class="${r.gain>0?'gain':''}">${r.gain>0?'−'+r.gain:'·'}</td></tr>`).join('')+`</table>`
      : `<table><tr><th>portfolio</th><th>result</th></tr>`+lv.solves.slice(-5).map(s=>
        `<tr><td>${s.label}</td><td>${s.seconds?s.seconds.toFixed(3)+'s':'no completion'}</td></tr>`).join('')+`</table>`}
    </div>`;
  }
  root.innerHTML=html;
  document.getElementById('upd').textContent='last fetch '+new Date().toLocaleTimeString();
  const tip=document.getElementById('tip');
  root.querySelectorAll('svg.ch').forEach((svg,si)=>{
    const lv=data.levels[si];
    svg.querySelectorAll('circle').forEach(c=>{
      c.addEventListener('mousemove',e=>{const r=lv.rounds[+c.dataset.i];
        tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-10)+'px';
        tip.innerHTML=`round ${r.idx} · ${r.kind}<br>${r.seconds.toFixed(3)}s (${r.after} ticks)`+
          (r.gain>0?`<br><span style="color:var(--good)">−${r.gain} ticks</span>`:'');});
      c.addEventListener('mouseleave',()=>tip.style.display='none');
    });
  });
}
async function tick(){
  try{ render(await (await fetch('/data')).json()); }catch(e){}
  setTimeout(tick, 3000);
}
tick();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=str, required=True)
    ap.add_argument("--port", type=int, default=5051)
    args = ap.parse_args()
    log_dir = Path(args.log_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/data":
                body = json.dumps(collect(log_dir)).encode()
                ctype = "application/json"
            else:
                body = PAGE.encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"watching {log_dir} -> http://localhost:{args.port}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
