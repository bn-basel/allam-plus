// Model card images: the banner (release/assets/banner.html) and the charts (docs/charts/*.svg) as PNG at 2x, the
// format Hugging Face shows most reliably. Uses Playwright from the shared toolbox (C:\WIP\tools\toolbox).
//   node scripts/render_card_images.mjs
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const require = createRequire("C:/WIP/tools/toolbox/package.json");
const { chromium } = require("playwright");
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/, "$1")), "..");
const out = path.join(root, "release", "assets");
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 1280, height: 640 } });

// the banner: a fixed 1280 x 640 page
await page.goto(pathToFileURL(path.join(out, "banner.html")).href, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: path.join(out, "banner.png") });
console.log("banner.png");

// the charts: each SVG on a white page, captured at its own size (light colours, as most visitors see the card)
for (const name of ["card_summary", "phase4_served", "needle_phase2", "card_how"]) {
  const svg = path.join(root, "docs", "charts", `${name}.svg`);
  if (!fs.existsSync(svg)) { console.log(`skip ${name} (no svg)`); continue; }
  await page.setContent(`<html><body style="margin:0;background:#fff">${fs.readFileSync(svg, "utf8")}</body></html>`);
  await page.emulateMedia({ colorScheme: "light" });
  const el = await page.$("svg");
  await el.screenshot({ path: path.join(out, `${name}.png`), omitBackground: false });
  console.log(`${name}.png`);
}
await browser.close();
