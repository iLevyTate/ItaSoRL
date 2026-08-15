// Headless frame capture for the stylized "Two Minds" brain film (brain.js).
// Same deterministic __seek(t) contract as capture.js, minus the collect.py
// scene-source guard: the brain film is declared illustrative art
// (window.__sceneSource === "stylized"), so there is no recorded-world claim
// to protect. Refuses anything else so it can't silently encode the wrong page.
//
// Env: CHROME_EXE, CAP_URL, CAP_FPS (30), CAP_OUT (out.mp4), CAP_MODE (full|proof)

const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");

const CHROME = process.env.CHROME_EXE;
const URL = process.env.CAP_URL || "http://127.0.0.1:8766/index.html";
const FPS = parseInt(process.env.CAP_FPS || "30", 10);
const OUT = process.env.CAP_OUT || "out.mp4";
const MODE = process.env.CAP_MODE || "full";

function writeChunk(stream, buf) {
  return new Promise((resolve) => {
    if (stream.write(buf)) resolve();
    else stream.once("drain", resolve);
  });
}

(async () => {
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--force-color-profile=srgb", "--hide-scrollbars"],
  });
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
  });
  page.on("console", (m) => {
    const t = m.type();
    if (t === "error" || t === "warning") console.log(`[page:${t}]`, m.text());
  });

  await page.goto(URL, { waitUntil: "load" });
  await page.waitForFunction("window.__ready === true", null, { timeout: 30000 });
  await page.evaluate("document.fonts && document.fonts.ready");

  const src = await page.evaluate("window.__sceneSource");
  if (src !== "stylized") {
    console.error(`CAPTURE_ERROR wrong page: sceneSource="${src}", expected "stylized"`);
    await browser.close();
    process.exit(2);
  }

  const size = await page.evaluate("window.__size");
  const [W, Hh] = size;
  await page.setViewportSize({ width: W, height: Hh });
  await page.evaluate("window.__seek(0)");
  console.log(`viewport=${W}x${Hh}`);

  const duration = await page.evaluate("window.__duration");
  const frameCount = Math.round((duration / 1000) * FPS);
  console.log(`duration_ms=${duration} fps=${FPS} frames=${frameCount}`);

  if (MODE === "proof") {
    const proofDir = process.env.CAP_PROOF_DIR || "proof";
    fs.mkdirSync(proofDir, { recursive: true });
    const fracs = [0.0, 0.2, 0.34, 0.5, 0.7, 0.9];
    for (const f of fracs) {
      const t = Math.round(f * duration) % duration;
      await page.evaluate((tt) => window.__seek(tt), t);
      await page.screenshot({ path: `${proofDir}/frame_${String(t).padStart(6, "0")}.png` });
      console.log("proof frame t=", t);
    }
    await browser.close();
    console.log("PROOF_DONE");
    return;
  }

  const ff = spawn(
    "ffmpeg",
    [
      "-y",
      "-f", "image2pipe",
      "-framerate", String(FPS),
      "-i", "pipe:0",
      "-c:v", "libx264",
      "-preset", "medium",
      "-crf", "17",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      OUT,
    ],
    { stdio: ["pipe", "inherit", "inherit"] }
  );
  const ffDone = new Promise((res, rej) => {
    ff.on("close", (code) => (code === 0 ? res() : rej(new Error("ffmpeg exit " + code))));
  });

  const t0 = Date.now();
  for (let i = 0; i < frameCount; i++) {
    const t = Math.round((i * 1000) / FPS) % duration;
    await page.evaluate((tt) => window.__seek(tt), t);
    const buf = await page.screenshot({ type: "png" });
    await writeChunk(ff.stdin, buf);
    if (i % 60 === 0 || i === frameCount - 1) {
      const el = (Date.now() - t0) / 1000;
      const rate = i > 0 ? i / el : 0;
      console.log(`frame ${i + 1}/${frameCount}  ${rate.toFixed(1)}fps elapsed=${el.toFixed(0)}s`);
    }
  }
  ff.stdin.end();
  await ffDone;
  await browser.close();
  console.log("FULL_DONE ->", OUT);
})().catch((e) => {
  console.error("CAPTURE_ERROR", e);
  process.exit(1);
});
