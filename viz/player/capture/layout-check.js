// Layout check for the creature film (player.js), run before any capture.
// Steps through the whole film and reports, per beat and time range:
//   - callout boxes clipped by the frame or crossing the split divider
//   - callouts overlapping each other, the creature, or the world chips
//   - the creature clipped by the frame or its split panel
//   - caption text running into the meter or past the frame edge
//   - callouts jumping more than 40 px between frames (50 ms apart)
// Serve viz/ first (python -m http.server 8767 from viz/), then:
//   node layout-check.js "http://127.0.0.1:8767/player/index.html?layout=wide&play=0"
// Prints NO ISSUES / NO SIDE JUMPS when clean. Exit code 1 if anything is found.
// Flags in the first ~300 ms of a beat that say "crowding the frame edge" are
// the caption's 12 px slide-in while it fades up, not a layout fault.

const { chromium } = require("playwright");
const [url] = process.argv.slice(2);
const STEP = 100;
const inter = (a, b) => Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
                        Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
(async () => {
  const br = await chromium.launch({ headless: true });
  const p = await br.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto(url, { waitUntil: "load" });
  await p.waitForFunction("window.__ready === true", null, { timeout: 60000 });
  await p.evaluate("document.fonts && document.fonts.ready");
  const size = await p.evaluate("window.__size"); await p.setViewportSize({ width: size[0], height: size[1] });
  const beats = await p.evaluate("fetch('beats.json').then(r=>r.json())");
  const issues = {};
  const add = (key, t) => { (issues[key] = issues[key] || []).push(t); };
  for (let t = 0; t < beats.duration_ms; t += STEP) {
    await p.evaluate((tt) => window.__seek(tt), t);
    const d = await p.evaluate(() => {
      const r = (id) => { const e = document.getElementById(id); if (!e) return null; const b = e.getBoundingClientRect();
        return { x: b.left, y: b.top, w: b.width, h: b.height, op: parseFloat(getComputedStyle(e).opacity) }; };
      const cv = document.getElementById("world");
      return { callouts: window.__callouts || [], sprites: window.__sprites || [],
        world: r("world-card"), canvasW: cv.clientWidth, canvasBmp: cv.width,
        cap: r("caption"), lead: r("caption-lead"), sub: r("subline"), head: r("headline"),
        gauge: r("gauge"), chipA: r("chip-a"), chipB: r("chip-b"), end: r("endcard"), stage: r("stage") };
    });
    const beat = beats.beats.find((b) => t >= b.t0 && t < b.t1) || beats.beats[beats.beats.length - 1];
    const tag = (s) => `${beat.id}: ${s}`;
    const W = d.world, k = W.w / d.canvasW;               // canvas px -> page px (1 at scale 1)
    const VIEW = d.canvasW;                                // layout px of the canvas
    const mode = beat.world ? beat.world.mode : "none";
    if (W.op < 0.05 && mode === "none") { /* end card */ }
    const vis = d.callouts.filter((c) => c.a > 0.15);
    // canvas-space world box
    const VW = d.canvasW;
    for (const c of vis) {
      const ch = d.world.h / k;
      if (c.x < 2 || c.y < 2 || c.x + c.w > VW - 2 || c.y + c.h > ch - 2) add(tag(`label clipped by frame edge: "${c.text}"`), t);
      if (mode === "split") {
        const mid = VW / 2;
        if (c.x < mid + 4 && c.x + c.w > mid - 4) add(tag(`label crosses the split divider: "${c.text}"`), t);
      }
      for (const s of d.sprites) {
        const sb = { x: s.x - 10 * s.sc, y: s.y - 19 * s.sc, w: 20 * s.sc, h: 27 * s.sc };
        if (inter(c, sb) > 0) add(tag(`label covers the creature: "${c.text}"`), t);
      }
      for (const chip of [d.chipA, d.chipB]) {
        if (!chip || chip.op < 0.1) continue;
        const cb = { x: (chip.x - W.x) / k, y: (chip.y - W.y) / k, w: chip.w / k, h: chip.h / k };
        if (inter(c, cb) > 0) add(tag(`label overlaps a world chip: "${c.text}"`), t);
      }
    }
    for (let i = 0; i < vis.length; i++) for (let j = i + 1; j < vis.length; j++) {
      if (inter(vis[i], vis[j]) > 0) add(tag(`labels overlap: "${vis[i].text}" / "${vis[j].text}"`), t);
    }
    if (mode !== "none" && W.op > 0.5) {
      for (const s of d.sprites) {
        const ch = d.world.h / k;
        if (s.x - 10 * s.sc < 0 || s.x + 10 * s.sc > VW || s.y - 19 * s.sc < 0 || s.y + 8 * s.sc > ch) add(tag("creature clipped by frame edge"), t);
        if (mode === "split") {
          const lo = s.vx, hi = s.vx + s.vw;
          if (s.x - 10 * s.sc < lo || s.x + 10 * s.sc > hi) add(tag(`creature clipped by its split panel (${s.key})`), t);
        }
      }
    }
    if (d.gauge && d.gauge.op > 0.05 && d.cap && d.cap.op > 0.05) {
      const textBottom = Math.max(d.lead.y + d.lead.h, d.sub.y + d.sub.h);
      const textTop = Math.min(d.lead.y, d.sub.y);
      const gTop = d.gauge.y, gBot = d.gauge.y + d.gauge.h;
      const capL = Math.min(d.lead.x, d.sub.x), capR = Math.max(d.lead.x + d.lead.w, d.sub.x + d.sub.w);
      const hOverlap = capR > d.gauge.x && capL < d.gauge.x + d.gauge.w;
      if (hOverlap && textBottom > gTop - 10 && textTop < gBot) add(tag("caption text runs into the gauge"), t);
    }
    for (const e of [d.lead, d.sub]) {
      const vert = d.stage.h > 1800;   // 9:16 plates sit flush on the edges by design
      const lim = vert ? 1 : -16;
      if (e && d.cap.op > 0.05 && (e.x + e.w > d.stage.x + d.stage.w + lim || e.y + e.h > d.stage.y + d.stage.h + lim)) add(tag("caption text past or crowding the frame edge"), t);
    }
  }
  const fmt = (ts) => { const rs = []; let a = ts[0], b = ts[0];
    for (const x of ts.slice(1)) { if (x - b <= STEP) b = x; else { rs.push([a, b]); a = b = x; } } rs.push([a, b]);
    return rs.map(([a, b]) => `${(a / 1000).toFixed(1)}-${(b / 1000).toFixed(1)}s`).join(", "); };
  const keys = Object.keys(issues);
  if (!keys.length) console.log("NO ISSUES");
  for (const kk of keys) console.log(`${kk}  [${issues[kk].length} frames] ${fmt(issues[kk])}`);
  // Jumps: a callout box moving > 40 px relative to its anchor between frames.
  const last = {}, jumps = {};
  for (let t = 0; t < beats.duration_ms; t += 50) {
    await p.evaluate((tt) => window.__seek(tt), t);
    const cs = await p.evaluate("window.__callouts");
    for (const c of cs) {
      if (c.a < 0.3) continue;
      const rel = [c.x + c.w / 2 - c.ax, c.y + c.h / 2 - c.ay];
      const k = c.text.replace(/[0-9.+]+/g, "#");
      if (last[k] && t - last[k].t <= 50) {
        const j = Math.hypot(rel[0] - last[k].rel[0], rel[1] - last[k].rel[1]);
        if (j > 40) (jumps[k] = jumps[k] || []).push(`${(t / 1000).toFixed(2)}s moved ${j.toFixed(0)}px`);
      }
      last[k] = { rel, t };
    }
  }
  const jk = Object.keys(jumps);
  if (!jk.length) console.log("NO SIDE JUMPS");
  for (const k of jk) console.log(`jump: "${k}" ${jumps[k].length}x ${jumps[k].slice(0, 6).join(" | ")}`);
  await br.close();
  process.exitCode = keys.length || jk.length ? 1 : 0;
})().catch((e) => { console.error(e); process.exit(1); });
