#!/usr/bin/env node
/**
 * Captures screenshots of the blog post for verification.
 * Run: npx playwright install chromium && node screenshot-post.js
 */

const { chromium } = require('playwright');

const URL = 'http://127.0.0.1:4000/2026/02/two_ways_to_bet_on_a_trillion_dollar_market/';

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1200, height: 800 });

  try {
    await page.goto(URL, { waitUntil: 'networkidle' });

    // Screenshot 1: Top of page (title + first paragraphs)
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(300);
    await page.screenshot({ path: 'screenshot-top.png', fullPage: false });
    console.log('Saved screenshot-top.png');

    // Screenshot 2: Bullet list section (OpenAI acquisitions)
    await page.evaluate(() => {
      const list = document.querySelector('.page-content ul');
      if (list) list.scrollIntoView({ block: 'center' });
    });
    await page.waitForTimeout(300);
    await page.screenshot({ path: 'screenshot-bullets.png', fullPage: false });
    console.log('Saved screenshot-bullets.png');

    // Screenshot 3: Footnotes section at bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(300);
    await page.screenshot({ path: 'screenshot-footnotes.png', fullPage: false });
    console.log('Saved screenshot-footnotes.png');

    console.log('\nDone! Check screenshot-top.png, screenshot-bullets.png, screenshot-footnotes.png');
  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
