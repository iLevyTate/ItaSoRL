// Renders docs/social-card/index.html to docs/social-card/social-preview.png (2560x1280,
// GitHub's 1280x640 at 2x). Needs Playwright's Chromium; set CHROME_EXE to its chrome
// binary and NODE_PATH to a node_modules holding playwright (viz/player/capture's works).
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_EXE || undefined });
  const page = await browser.newPage({ viewport: { width: 1280, height: 640 }, deviceScaleFactor: 2 });
  await page.goto(pathToFileURL(path.join(__dirname, "index.html")).href);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(__dirname, "social-preview.png") });
  await browser.close();
  console.log("wrote social-preview.png");
})();
