// QA still capture sized to the renderer's own canvas (__size), scaled 0.5.
// Env: CHROME_EXE, SHOT_URL, SHOT_T (ms), SHOT_OUT.
const { chromium } = require("playwright");

const CHROME = process.env.CHROME_EXE;
const URL = process.env.SHOT_URL;
const T = parseInt(process.env.SHOT_T || "6000", 10);
const OUT = process.env.SHOT_OUT || "shot.png";

(async () => {
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--force-color-profile=srgb", "--hide-scrollbars"],
  });
  const page = await browser.newPage({ viewport: { width: 640, height: 480 }, deviceScaleFactor: 1 });
  await page.goto(URL, { waitUntil: "load" });
  await page.waitForFunction("window.__ready === true", null, { timeout: 30000 });
  const [w, h] = await page.evaluate("window.__size");
  await page.setViewportSize({ width: Math.round(w / 2), height: Math.round(h / 2) });
  await page.evaluate(() => {
    const c = document.getElementById("stage");
    c.style.width = "100vw"; c.style.height = "100vh";
    document.body.style.margin = "0";
  });
  await page.evaluate((tt) => window.__seek(tt), T);
  await page.screenshot({ path: OUT });
  console.log(`QA_SHOT_DONE t=${T} out=${OUT}`);
  await browser.close();
})().catch((e) => { console.error("SHOT_ERROR", e); process.exit(1); });
