/* ItaSoRL site loops - four seamless data loops for the "More ways to see it" row.
 *
 * Unlike brain.js (declared illustrative art), every number and every dot here is
 * read at load time from a committed file, and the page refuses to draw if a
 * derived mean drifts from the FINDINGS value it is published as:
 *
 *   ?loop=worlds  the recorded seed-0 survival brain in the real and the fake world
 *                 from the same start (viz/data/scene.json, written by viz/collect.py)
 *   ?loop=brains  per-seed probe readouts for four kinds of brain (FINDINGS 10.2, 10.8)
 *   ?loop=echoes  the survive-and-predict brains with behavior, senses, and both
 *                 subtracted (FINDINGS 10.4, 10.4.2 and its nonlinear addendum)
 *   ?loop=dial    the H2 graded seam: the fake rule blended back toward the real one
 *                 (FINDINGS 14, A1)
 *
 * Motion between measured states is a visual tween of dot positions only; a number
 * is drawn only while the picture sits on a measured state.
 *
 * Capture contract (viz/player/capture/capture-brain.js with CAP_SOURCE=artifacts):
 *   window.__ready, window.__size, window.__duration, window.__seek(t),
 *   window.__sceneSource = "artifacts". Every loop's frame at t = 0 equals its frame
 *   at t = __duration, so the encoded clip loops without a seam.
 */
(() => {
  "use strict";

  const Q = new URLSearchParams(location.search);
  const LOOP = Q.get("loop") || "brains";
  const W = 1280, H = 720;
  const cv = document.getElementById("c");
  const ctx = cv.getContext("2d");

  // ------------------------------------------------------------------ palette
  // Validated (dataviz validate_palette.js, dark, surface #100e1b):
  //   ACC/GRAY   emphasis pair; GRAY is a deliberate low-chroma de-emphasis
  //   REAL/FAKE  every check passes, including deutan/protan separation
  const BG0 = "#0a0910", BG1 = "#151228";
  const INK = "#ece9f5", SOFT = "#bdb7d6", MUTED = "#8c86a3";
  const ACC = "#9579e8", GRAY = "#6c6784", TEAL = "#35b5a2";
  const REAL = "#4a90e6", FAKE = "#df6a55", GOLD = "#e8b84b";
  const SANS = "'Hanken Grotesk', sans-serif";
  const HEAD = "'Space Grotesk', sans-serif";
  const MONO = "'IBM Plex Mono', monospace";

  // ------------------------------------------------------------------ numbers
  const T90_DF9 = 1.8331; // Student t 0.95 quantile, 9 df: every row has 10 seeds
  const BAR = 0.65;

  function stats(v) {
    if (v.length !== 10) throw new Error(`expected 10 seeds, got ${v.length}`);
    const n = v.length, m = v.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / (n - 1));
    const h = (T90_DF9 * sd) / Math.sqrt(n);
    return { seeds: v, mean: m, lo: m - h, hi: m + h, over: v.filter((x) => x >= BAR).length };
  }
  const f3 = (x) => x.toFixed(3);

  // Published values each derived mean must reproduce (3 dp), the same guard
  // viz/collect.py applies to the film's numbers.
  const EXPECT = {
    untrained: 0.488, predictor: 0.573, nopred: 0.601, nopred_ref: 0.730, survival: 0.752,
    minus_move: 0.726, minus_sense: 0.731, minus_both: 0.654,
    dial_surv_1: 0.752, dial_surv_0: 0.506,
  };
  function check(key, x) {
    if (f3(x) !== f3(EXPECT[key])) throw new Error(`number honesty: ${key} = ${f3(x)}, FINDINGS says ${f3(EXPECT[key])}`);
  }

  async function getJSON(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
    return r.json();
  }
  const ART = "../../../artifacts/";
  const bySeed = (cells, agent, key) =>
    cells.filter((c) => c.drift === "0.45" && c.agent === agent)
      .sort((a, b) => a.seed - b.seed).map((c) => c[key]);

  // ------------------------------------------------------------------ helpers
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const lerp = (a, b, k) => a + (b - a) * k;
  const easeOut = (k) => 1 - Math.pow(1 - clamp01(k), 3);
  const easeInOut = (k) => { k = clamp01(k); return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2; };
  const win = (t, a, b) => clamp01((t - a) / (b - a)); // 0 before a, 1 after b
  // Fixed vertical slot per seed, so a seed keeps its lane when it moves.
  const LANE = [-10, 6, -3, 11, -7, 2, 9, -12, -1, 4];

  function text(s, x, y, o = {}) {
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.font = o.font || `400 20px ${SANS}`;
    ctx.fillStyle = o.color || INK;
    ctx.textAlign = o.align || "left";
    ctx.textBaseline = o.base || "alphabetic";
    if (o.ls) ctx.letterSpacing = o.ls;
    ctx.fillText(s, x, y);
    ctx.restore();
  }
  function wrap(s, maxW, font) {
    ctx.save(); ctx.font = font;
    const out = []; let line = "";
    for (const w of s.split(" ")) {
      const t = line ? line + " " + w : w;
      if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t;
    }
    if (line) out.push(line);
    ctx.restore();
    return out;
  }
  function para(s, x, y, maxW, font, color, lh, alpha) {
    const lines = wrap(s, maxW, font);
    lines.forEach((l, i) => text(l, x, y + i * lh, { font, color, alpha }));
    return y + lines.length * lh;
  }
  function dot(x, y, r, color, alpha = 1, ring = true) {
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    if (ring) { ctx.lineWidth = 2; ctx.strokeStyle = BG0; ctx.stroke(); }
    ctx.restore();
  }
  function line(x0, y0, x1, y1, color, w, alpha = 1, dash = null) {
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.restore();
  }

  function background() {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, BG1); g.addColorStop(1, BG0);
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    const r = ctx.createRadialGradient(W * 0.72, H * 0.1, 40, W * 0.72, H * 0.1, 760);
    r.addColorStop(0, "rgba(149,121,232,0.10)"); r.addColorStop(1, "rgba(149,121,232,0)");
    ctx.fillStyle = r; ctx.fillRect(0, 0, W, H);
  }
  function header(kicker, title, sub, x = 72, maxW = 1136) {
    text(kicker, x, 62, { font: `500 15px ${MONO}`, color: TEAL, ls: "2.5px" });
    text(title, x, 114, { font: `600 42px ${HEAD}`, color: INK });
    return para(sub, x, 152, maxW, `400 21px ${SANS}`, SOFT, 29);
  }
  function footer(s, x = 72) {
    text(s, x, 700, { font: `400 13px ${MONO}`, color: MUTED });
  }

  // Probe-AUROC axis shared by the dot-plot loops.
  function axis(a) {
    const X = (v) => a.x0 + ((v - 0.4) / 0.5) * (a.x1 - a.x0);
    for (const v of [0.4, 0.6, 0.7, 0.8, 0.9]) line(X(v), a.yTop, X(v), a.yBot, "rgba(236,233,245,0.07)", 1);
    line(X(0.5), a.yTop, X(0.5), a.yBot, "rgba(236,233,245,0.30)", 1.2);
    line(X(BAR), a.yTop, X(BAR), a.yBot, TEAL, 1.6, 0.8, [7, 7]);
    text("the bar, set in advance: 0.65", X(BAR) + 10, a.yTop + 4, { font: `500 14px ${MONO}`, color: TEAL });
    for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) {
      text(v.toFixed(1), X(v), a.yBot + 26, { font: `400 15px ${MONO}`, color: MUTED, align: "center" });
    }
    text("coin flip", X(0.5), a.yBot + 48, { font: `500 14px ${MONO}`, color: SOFT, align: "center" });
    text("probe AUROC →", a.x1, a.yBot + 48, { font: `400 14px ${MONO}`, color: MUTED, align: "right" });
    return X;
  }
  function interval(X, y, lo, hi, mean, color, k = 1, alpha = 1) {
    const m = X(mean);
    line(lerp(m, X(lo), k), y, lerp(m, X(hi), k), y, color, 3.2, alpha);
    dot(m, y, 7.5, color, alpha);
  }

  // ------------------------------------------------------------------ loop: brains
  function makeBrains(d) {
    const T = 11000;
    const rows = [
      { r: d.untrained, label: "Untrained brain", detail: "random wiring, never trained", color: GRAY },
      { r: d.predictor, label: "Trained only to predict", detail: "guesses what it will sense next", color: GRAY },
      { r: d.nopred, label: "Trained to survive, no predictor", detail: `ran on CPU; with it, same CPU: ${f3(d.nopred_ref.mean)}`, color: GRAY },
      { r: d.survival, label: "Trained to survive and predict", detail: "the published brain", color: ACC },
    ];
    const ys = [290, 376, 462, 548];
    return {
      T,
      draw(t) {
        const s = t / 1000;
        background();
        header("ITASORL · WHAT EACH BRAIN'S MEMORY HOLDS", "Four kinds of brain, one fake world.",
          "Each dot is one brain trained from scratch, ten of each. Afterward a probe reads its memory: can it tell which world it lived in?");
        const X = axis({ x0: 560, x1: 1204, yTop: 246, yBot: 584 });
        const out = 1 - win(s, 9.0, 10.2);
        rows.forEach((row, i) => {
          const y = ys[i], emph = row.color === ACC;
          text(row.label, 72, y - 6, { font: `${emph ? 600 : 500} 23px ${SANS}`, color: emph ? INK : SOFT });
          text(row.detail, 72, y + 20, { font: `400 14px ${MONO}`, color: MUTED });
          const s0 = 0.5 + 1.25 * i;
          row.r.seeds.forEach((v, j) => {
            const k = win(s, s0 + 0.05 * j, s0 + 0.05 * j + 0.45);
            if (k <= 0) return;
            dot(X(v), y + LANE[j] - 34 * (1 - easeOut(k)), 4.6, row.color, 0.62 * k * out, false);
          });
          const kc = win(s, s0 + 0.65, s0 + 1.05);
          if (kc > 0) interval(X, y, row.r.lo, row.r.hi, row.r.mean, row.color, easeOut(kc), out);
          const kv = win(s, s0 + 0.8, s0 + 1.1);
          if (kv > 0) text(f3(row.r.mean), 520, y + 8, { font: `500 25px ${MONO}`, color: INK, align: "right", alpha: kv * out });
        });
        const kcap = win(s, 5.9, 6.4) * out;
        if (kcap > 0) {
          text(`Only the brain that both survives and predicts clears the bar on average. ${d.survival.over} of its 10 seeds are over it.`, 72, 662,
            { font: `500 22px ${SANS}`, color: INK, alpha: kcap });
        }
        footer("artifacts/expB2 · FINDINGS 10.2, 10.8 · L3 fingerprint, drift 0.45, hidden 8 · dot = one seed; bar = mean with 90% t-interval");
      },
    };
  }

  // ------------------------------------------------------------------ loop: echoes
  function makeEchoes(d) {
    const T = 12000;
    const steps = [
      { r: d.survival, label: "Raw readout", detail: "nothing removed" },
      { r: d.minus_move, label: "Minus how it moves", detail: "speed, energy, food, drag, every step" },
      { r: d.minus_sense, label: "Minus what it senses", detail: "this step's input and the last, linear" },
      { r: d.minus_both, label: "Minus both, nonlinear", detail: "senses and behavior, small neural net" },
    ];
    // [state, hold start, hold end]; transitions fill the gaps; 3 -> 0 wraps via 11.6-12.
    const holds = [[0, 0.0, 2.0], [1, 2.8, 4.8], [2, 5.6, 7.6], [3, 8.4, 10.6], [0, 11.4, 12.0]];
    function state(s) {
      for (let i = 0; i < holds.length; i++) {
        const [st, a, b] = holds[i];
        if (s >= a && s <= b) return { from: st, to: st, k: 0, hold: i };
        const nx = holds[i + 1];
        if (nx && s > b && s < nx[1]) return { from: st, to: nx[0], k: easeInOut((s - b) / (nx[1] - b)), hold: -1 };
      }
      return { from: 0, to: 0, k: 0, hold: 0 };
    }
    const ys = [296, 366, 436, 506];
    const Y = 400;
    return {
      T,
      draw(t) {
        const s = t / 1000;
        background();
        header("ITASORL · IS IT JUST AN ECHO?", "Subtract what the memory could be copying.",
          "The same ten survive-and-predict brains. Each control removes one thing the signal might be echoing; the probe reads what is left.");
        const X = axis({ x0: 580, x1: 1204, yTop: 262, yBot: 540 });
        const st = state(s);
        const active = st.k < 0.5 ? st.from : st.to;
        steps.forEach((step, i) => {
          const on = i === active;
          if (on) line(58, ys[i] - 24, 58, ys[i] + 22, TEAL, 4);
          text(step.label, 72, ys[i] - 4, { font: `${on ? 600 : 500} 22px ${SANS}`, color: on ? INK : MUTED });
          text(step.detail, 72, ys[i] + 20, { font: `400 13px ${MONO}`, color: on ? SOFT : "rgba(140,134,163,0.75)" });
          text(f3(step.r.mean), 540, ys[i] + 4, { font: `500 23px ${MONO}`, color: on ? INK : MUTED, align: "right" });
        });
        const A = steps[st.from].r, B = steps[st.to].r;
        A.seeds.forEach((v, j) => dot(X(lerp(v, B.seeds[j], st.k)), Y + LANE[j] * 2.4, 5.2, ACC, 0.6, false));
        interval(X, Y, lerp(A.lo, B.lo, st.k), lerp(A.hi, B.hi, st.k), lerp(A.mean, B.mean, st.k), ACC);
        if (st.hold >= 0) {
          const [, a, b] = holds[st.hold];
          const fin = a === 0 ? 1 : win(s, a, a + 0.3), fout = b === 12.0 ? 1 : 1 - win(s, b - 0.3, b);
          const r = steps[st.from].r;
          const cap = st.from === 3
            ? `${r.over} of 10 over the bar. The mean sits at the bar, not comfortably above it.`
            : `${r.over} of 10 brains over the bar.`;
          text(cap, 72, 648, { font: `500 22px ${SANS}`, color: INK, alpha: Math.min(fin, fout) });
        }
        footer("artifacts/expB2: behavior_audit_l3_h8_heldout, sensory_echo_l3_h8(_mlp) · FINDINGS 10.4, 10.4.2 · dot = one seed; bar = mean with 90% t-interval");
      },
    };
  }

  // ------------------------------------------------------------------ loop: dial
  function makeDial(d) {
    const T = 11000;
    const stops = d.alphas; // [1, .75, .5, .25, .1, 0]
    // Segments tile [0, T): hold at alpha 1 is split across the wrap, so the
    // frame at t = 0 and t = T is the same held state with its numbers showing.
    const seg = [];
    let t0 = 0;
    const push = (type, a, b, dur, fin, fout) => { seg.push({ type, a, b, t0, t1: t0 + dur, fin, fout }); t0 += dur; };
    push("hold", 0, 0, 0.65, false, true);
    for (let i = 1; i < stops.length; i++) {
      push("glide", i - 1, i, 0.55);
      push("hold", i, i, i === stops.length - 1 ? 1.6 : 0.7, true, true);
    }
    push("glide", stops.length - 1, 0, 1.6);
    push("hold", 0, 0, 11.0 - t0, true, false);
    function at(s) {
      for (const g of seg) if (s >= g.t0 && s < g.t1 + 1e-9) return g;
      return seg[0];
    }
    const XS0 = 72, XS1 = 500, YS = 318;
    // Six settings, evenly spaced: the dial is a labeled control, not a data axis.
    const sxi = (i) => XS0 + (i / (stops.length - 1)) * (XS1 - XS0);
    return {
      T,
      draw(t) {
        const s = t / 1000;
        background();
        header("ITASORL · WHERE THE SIGNAL LIVES", "Turn the flaw down. The signal follows.",
          "The fake world's learned rule is blended back toward the real one. At 0% the two worlds run the same physics.");
        const g = at(s);
        const k = g.type === "glide" ? easeInOut((s - g.t0) / (g.t1 - g.t0)) : 0;
        const pos = g.type === "glide" ? lerp(sxi(g.a), sxi(g.b), k) : sxi(g.a);
        let num = 0;
        if (g.type === "hold") {
          num = Math.min(g.fin ? win(s, g.t0, g.t0 + 0.2) : 1, g.fout ? 1 - win(s, g.t1 - 0.2, g.t1) : 1);
        }
        // Dial.
        text("Flaw left in the fake world", XS0, 236, { font: `500 21px ${SANS}`, color: SOFT });
        line(XS0, YS, XS1, YS, "rgba(236,233,245,0.18)", 6);
        line(pos, YS, XS1, YS, FAKE, 6, 0.9);
        stops.forEach((a, i) => {
          line(sxi(i), YS + 10, sxi(i), YS + 18, MUTED, 1.5);
          text(`${Math.round(a * 100)}%`, sxi(i), YS + 40, { font: `400 15px ${MONO}`, color: MUTED, align: "center" });
        });
        dot(pos, YS, 11, INK, 1);
        if (num > 0) text(`${Math.round(stops[g.a] * 100)}%`, XS1, 236, { font: `500 26px ${MONO}`, color: INK, align: "right", alpha: num });
        // Rows.
        const X = axis({ x0: 580, x1: 1204, yTop: 392, yBot: 586 });
        const rows = [
          { key: "surv", label: "Survive-and-predict brain", color: ACC, y: 450 },
          { key: "untr", label: "Untrained brain", color: GRAY, y: 536 },
        ];
        rows.forEach((row) => {
          const A = d[row.key][g.a], B = d[row.key][g.type === "glide" ? g.b : g.a];
          const emph = row.color === ACC;
          text(row.label, 72, row.y + 7, { font: `${emph ? 600 : 500} 22px ${SANS}`, color: emph ? INK : SOFT });
          A.seeds.forEach((v, j) => dot(X(lerp(v, B.seeds[j], k)), row.y + LANE[j] * 1.6, 4.8, row.color, 0.6, false));
          interval(X, row.y, lerp(A.lo, B.lo, k), lerp(A.hi, B.hi, k), lerp(A.mean, B.mean, k), row.color);
          if (num > 0) text(f3(A.mean), 540, row.y + 8, { font: `500 23px ${MONO}`, color: INK, align: "right", alpha: num });
        });
        if (num > 0 && (g.a === 0 || g.a === stops.length - 1)) {
          const cap = g.a === 0
            ? `Full flaw: the survive-and-predict brains read ${f3(d.surv[0].mean)}.`
            : `Seam gone: ${f3(d.surv[stops.length - 1].mean)}, a coin flip. The untrained brain never had it.`;
          text(cap, 72, 664, { font: `500 22px ${SANS}`, color: INK, alpha: num });
        }
        footer("artifacts/expH2/summary.json · FINDINGS 14 (A1 graded seam) · L3 fingerprint, drift 0.45 · dot = one seed; bar = mean with 90% t-interval");
      },
    };
  }

  // ------------------------------------------------------------------ loop: worlds
  function makeWorlds(scene) {
    const T = 12000, KMAX = 140;
    const A = scene.trajs.auth, S = scene.trajs.surr;
    if (S.length <= KMAX || A.length <= KMAX) throw new Error("recorded runs shorter than the loop");
    let split = -1;
    for (let i = 0; i <= KMAX; i++) {
      if (Math.hypot(A[i][0] - S[i][0], A[i][1] - S[i][1]) > 0.05) { split = i; break; }
    }
    const MX = 56, MY = 70, MS = 580;
    const P = (x, y) => [MX + x * MS, MY + y * MS];
    // Terrain baked once from the recorded height and water fields.
    const n = scene.grid_n;
    const terr = document.createElement("canvas");
    terr.width = n; terr.height = n;
    const tc = terr.getContext("2d");
    const img = tc.createImageData(n, n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const idx = j * n + i, h = scene.height[idx], wet = scene.wet[idx] > 0.5;
        const band = Math.floor(h * 7) / 7;
        const c = wet
          ? [lerp(18, 30, h), lerp(44, 74, h), lerp(70, 104, h)]
          : [lerp(22, 52, band), lerp(19, 42, band), lerp(38, 82, band)];
        img.data.set([c[0], c[1], c[2], 255], idx * 4);
      }
    }
    tc.putImageData(img, 0, 0);
    const key = (p) => `${p[0].toFixed(3)},${p[1].toFixed(3)}`;
    function trail(pts, k, color, alpha) {
      if (k < 1) return;
      ctx.save();
      ctx.globalAlpha *= alpha;
      ctx.strokeStyle = color; ctx.lineWidth = 3.2; ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.shadowColor = color; ctx.shadowBlur = 8;
      ctx.beginPath();
      for (let i = 0; i <= k; i++) { const [x, y] = P(pts[i][0], pts[i][1]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke();
      ctx.restore();
    }
    function head(p, color, label, dx, dy, alpha) {
      const [x, y] = P(p[0], p[1]);
      ctx.save();
      ctx.globalAlpha *= alpha;
      const g = ctx.createRadialGradient(x, y, 0, x, y, 26);
      g.addColorStop(0, color + "88"); g.addColorStop(1, color + "00");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 26, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      dot(x, y, 8.5, color, alpha);
      text(label, x + dx, y + dy, { font: `600 15px ${MONO}`, color: INK, alpha, align: dx < 0 ? "right" : "left" });
    }
    return {
      T,
      draw(t) {
        const s = t / 1000;
        background();
        // Map.
        ctx.save();
        ctx.beginPath(); ctx.roundRect(MX, MY, MS, MS, 18); ctx.clip();
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
        ctx.drawImage(terr, MX, MY, MS, MS);
        ctx.fillStyle = "rgba(10,9,16,0.18)"; ctx.fillRect(MX, MY, MS, MS);
        const k = s < 0.5 ? 0 : Math.min(KMAX, Math.floor(((s - 0.5) / 8.0) * KMAX));
        const fade = 1 - win(s, 10.0, 11.0);
        const back = win(s, 11.0, 11.5);
        const kk = s >= 11.0 ? 0 : k;
        const pa = new Set(scene.pellets_t.auth[kk].map(key));
        const ps = new Set(scene.pellets_t.surr[kk].map(key));
        for (const p of new Set([...pa, ...ps])) {
          const [x, y] = P(...p.split(",").map(Number));
          dot(x, y, 5, GOLD, pa.has(p) && ps.has(p) ? 0.95 : 0.4, false);
        }
        const trailA = s >= 11.0 ? 0 : fade;
        trail(A, kk, REAL, 0.9 * trailA);
        trail(S, kk, FAKE, 0.9 * trailA);
        const headA = s >= 11.0 ? back : (s >= 10.0 ? fade : 1);
        head(A[kk], REAL, "real", -16, -14, headA);
        head(S[kk], FAKE, "fake", 16, 24, headA);
        ctx.restore();
        ctx.save(); ctx.strokeStyle = "rgba(236,233,245,0.12)"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.roundRect(MX, MY, MS, MS, 18); ctx.stroke(); ctx.restore();
        // Text column.
        const x = 690, mw = 536;
        const live = s < 11.0 ? 1 : 0;
        text("ITASORL · ONE BRAIN, TWO WORLDS", x, 98, { font: `500 15px ${MONO}`, color: TEAL, ls: "2.5px" });
        text("Same brain, same start.", x, 148, { font: `600 40px ${HEAD}`, color: INK });
        text("One rule changed.", x, 196, { font: `600 40px ${HEAD}`, color: INK });
        let y = para("A recorded run of one trained survive-and-predict brain. In the fake world a small learned network replaces the rule for how things move. Nothing else differs.",
          x, 238, mw, `400 20px ${SANS}`, SOFT, 28);
        line(x, y + 12, x + 34, y + 12, REAL, 4); text("real world", x + 46, y + 18, { font: `500 17px ${MONO}`, color: INK });
        line(x + 210, y + 12, x + 244, y + 12, FAKE, 4); text("fake world", x + 256, y + 18, { font: `500 17px ${MONO}`, color: INK });
        // Measured distance between the two creatures, step by step.
        const cx0 = x, cx1 = x + mw, cy0 = y + 62, cy1 = y + 178;
        const DMAX = 0.6;
        const cxs = (i) => cx0 + (i / KMAX) * (cx1 - cx0);
        const cys = (v) => cy1 - (Math.min(v, DMAX) / DMAX) * (cy1 - cy0);
        text("how far apart, as a share of the world's width", cx0, cy0 - 12, { font: `400 14px ${MONO}`, color: MUTED });
        line(cx0, cy1, cx1, cy1, "rgba(236,233,245,0.18)", 1);
        for (const v of [0.2, 0.4]) {
          line(cx0, cys(v), cx1, cys(v), "rgba(236,233,245,0.07)", 1);
          text(v.toFixed(1), cx1 + 6, cys(v) + 5, { font: `400 13px ${MONO}`, color: MUTED });
        }
        if (kk >= 1) {
          ctx.save();
          ctx.globalAlpha *= fade * live;
          ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.lineJoin = "round";
          ctx.beginPath();
          for (let i = 0; i <= kk; i++) {
            const v = Math.hypot(A[i][0] - S[i][0], A[i][1] - S[i][1]);
            if (i === 0) ctx.moveTo(cxs(i), cys(v)); else ctx.lineTo(cxs(i), cys(v));
          }
          ctx.stroke();
          ctx.restore();
          const v = Math.hypot(A[kk][0] - S[kk][0], A[kk][1] - S[kk][1]);
          dot(cxs(kk), cys(v), 4.5, INK, fade * live);
        }
        text(`step ${kk}`, cx1, cy0 - 12, { font: `500 16px ${MONO}`, color: SOFT, align: "right", alpha: live * win(s, 0.2, 0.5) * fade });
        text("0", cx0, cy1 + 22, { font: `400 13px ${MONO}`, color: MUTED });
        text(`${KMAX} steps`, cx1, cy1 + 22, { font: `400 13px ${MONO}`, color: MUTED, align: "right" });
        const tsplit = 0.5 + (split / KMAX) * 8.0;
        const ks = win(s, tsplit, tsplit + 0.4) * fade * live;
        if (split >= 0 && ks > 0) {
          const sx = cxs(split);
          line(sx, cy0, sx, cy1, TEAL, 1.4, ks, [5, 5]);
          text(`The paths part at step ${split}.`, x, cy1 + 66, { font: `500 21px ${SANS}`, color: INK, alpha: ks });
        }
        const ke = win(s, 8.6, 9.1) * fade * live;
        if (ke > 0) {
          para("It moves differently in each world, so the memory probe is re-run with movement subtracted first (FINDINGS 10.4).",
            x, cy1 + 100, mw, `400 19px ${SANS}`, SOFT, 26, ke);
        }
        text("viz/data/scene.json (viz/collect.py) · seed-0 survival bundle, L3 fingerprint, drift 0.45 · steps 0 to 140", 56, 700,
          { font: `400 13px ${MONO}`, color: MUTED });
      },
    };
  }

  // ------------------------------------------------------------------ boot
  async function build() {
    if (LOOP === "worlds") return makeWorlds(await getJSON("../../data/scene.json"));
    if (LOOP === "dial") {
      const h2 = await getJSON(ART + "expH2/summary.json");
      const keys = ["a1.00", "a0.75", "a0.50", "a0.25", "a0.10", "a0.00"];
      const d = {
        alphas: keys.map((k) => h2.cells.survival[k].alpha),
        surv: keys.map((k) => stats(h2.cells.survival[k].per_seed)),
        untr: keys.map((k) => stats(h2.cells.untrained[k].per_seed)),
      };
      // The H2 artifact stores seeds to 4 dp, which rounds the alpha = 1 mean to
      // 0.7525. FINDINGS 14: at alpha = 1 the pools bit-match the headline run, so
      // that stop uses the headline's full-precision seeds once they agree.
      const h8 = await getJSON(ART + "expB2/heldout_l3_h8_summary.json");
      for (const [key, agent] of [["surv", "survival"], ["untr", "untrained"]]) {
        const full = bySeed(h8.cells, agent, "pool_target");
        if (!full.every((v, i) => Math.abs(v - d[key][0].seeds[i]) < 6e-5)) throw new Error(`alpha 1 ${agent} seeds differ from the headline run`);
        d[key][0] = stats(full);
      }
      check("dial_surv_1", d.surv[0].mean); check("dial_surv_0", d.surv[5].mean);
      return makeDial(d);
    }
    const h8 = await getJSON(ART + "expB2/heldout_l3_h8_summary.json");
    const survival = stats(bySeed(h8.cells, "survival", "pool_target"));
    check("survival", survival.mean);
    if (LOOP === "echoes") {
      const beh = await getJSON(ART + "expB2/behavior_audit_l3_h8_heldout.json");
      const se = (await getJSON(ART + "expB2/sensory_echo_l3_h8.json")).aggregate["d=0.45 survival"];
      const mlp = (await getJSON(ART + "expB2/sensory_echo_l3_h8_mlp.json")).aggregate["d=0.45 survival"];
      // Seed order must agree across artifacts before any seed is tracked between them.
      const same = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-6);
      if (!same(se.target.per_seed, survival.seeds) || !same(mlp.target.per_seed, survival.seeds)
          || !same(bySeed(beh.cells, "survival", "target"), survival.seeds)) {
        throw new Error("seed order differs between artifacts");
      }
      const d = {
        survival,
        minus_move: stats(bySeed(beh.cells, "survival", "resid_trace")),
        minus_sense: stats(se.resid_obs.per_seed),
        minus_both: stats(mlp.resid_obs_beh_mlp.per_seed),
      };
      check("minus_move", d.minus_move.mean); check("minus_sense", d.minus_sense.mean); check("minus_both", d.minus_both.mean);
      return makeEchoes(d);
    }
    const nowm = await getJSON(ART + "expB2/arch_baseline_l3_h8_nowm.json");
    const dev = await getJSON(ART + "expB2/device_control_l3_h8_wm_cpu.json");
    const d = {
      untrained: stats(bySeed(h8.cells, "untrained", "pool_target")),
      predictor: stats(bySeed(h8.cells, "predictor", "pool_target")),
      nopred: stats(nowm.arms["0.45"].survival.pool_target.per_seed),
      nopred_ref: stats(dev.arms["0.45"].survival.pool_target.per_seed),
      survival,
    };
    for (const kk of ["untrained", "predictor", "nopred", "nopred_ref"]) check(kk, d[kk].mean);
    return makeBrains(d);
  }

  window.__size = [W, H];
  window.__sceneSource = "artifacts";
  build().then(async (loop) => {
    await document.fonts.load(`600 42px ${HEAD}`);
    await document.fonts.load(`500 20px ${SANS}`);
    await document.fonts.load(`500 15px ${MONO}`);
    await document.fonts.ready;
    window.__duration = loop.T;
    // Any external seek (the capture rig) stops the live preview loop for good,
    // so a requestAnimationFrame redraw can never land between seek and screenshot.
    window.__seek = (t) => { window.__capturing = true; loop.draw(((t % loop.T) + loop.T) % loop.T); };
    loop.draw(0);
    window.__ready = true;
    if (!Q.has("still")) {
      const t0 = performance.now();
      const tick = () => { if (!window.__capturing) loop.draw((performance.now() - t0) % loop.T); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }
  }).catch((e) => {
    document.getElementById("err").textContent = String(e.message || e);
    console.error(e);
  });
})();
