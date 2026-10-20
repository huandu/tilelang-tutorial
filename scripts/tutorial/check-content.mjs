/**
 * Validate the hand-authored tutorial inputs under `src/tutorial/`.
 *
 *   node scripts/tutorial/check-content.mjs            # everything
 *   node scripts/tutorial/check-content.mjs 01 02      # only these chapters
 *
 * Checks:
 *   - every chapter has a quiz file with 5–8 well-formed items
 *   - every figure spec is valid JSON, has an id matching its file name, uses a
 *     known `kind` and known `tone` values, and renders without throwing
 *   - every source figure has a matching spec (coverage)
 */

import { existsSync } from 'node:fs';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderFigure } from './svg.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');
const SRC = path.join(repo, 'src/tutorial');

const KINDS = new Set(['flow', 'layers', 'steps', 'compare', 'timeline', 'bars', 'grid', 'custom']);
const TONES = new Set(['default', 'accent', 'store', 'compute', 'muted', 'warn', 'header', 'cell']);

/** Titles/labels that show up when a spec was scaffolded but never filled in. */
const PLACEHOLDERS = new Set(['t', 'x', 'test', 'todo', 'tbd', 'demo', '示例', '待补', '—']);

function isPlaceholder(value) {
  return typeof value !== 'string' || PLACEHOLDERS.has(value.trim().toLowerCase());
}

const problems = [];
const warnings = [];

function fail(chapter, message) {
  problems.push(`${chapter}: ${message}`);
}

function warn(chapter, message) {
  warnings.push(`${chapter}: ${message}`);
}

function checkTones(chapter, spec, where) {
  const nodes = [
    ...(spec.nodes ?? []),
    ...(spec.layers ?? []),
    ...(spec.steps ?? []),
    ...(spec.items ?? []),
    ...(spec.shapes ?? []),
  ];
  for (const node of nodes) {
    if (typeof node.tone === 'string' && !TONES.has(node.tone)) {
      fail(chapter, `${where}: unknown tone "${node.tone}"`);
    }
  }
  for (const group of spec.groups ?? []) {
    if (typeof group.tone === 'string' && !TONES.has(group.tone)) {
      fail(chapter, `${where}: unknown group tone "${group.tone}"`);
    }
  }
}

/**
 * A spec must carry real content: at least two entries with real labels.
 * This is what catches scaffolded specs such as `{kind:'layers',layers:[{label:'x'}]}`.
 */
function checkSubstance(chapter, spec, where) {
  const collections = {
    nodes: spec.nodes ?? [],
    layers: spec.layers ?? [],
    steps: spec.steps ?? [],
    items: spec.items ?? [],
    rows: spec.rows ?? [],
    shapes: spec.shapes ?? [],
    arrows: spec.arrows ?? [],
  };
  const used = Object.entries(collections).filter(([, value]) => value.length > 0);
  if (used.length === 0) {
    fail(chapter, `${where}: spec has no nodes/layers/steps/items/rows/shapes`);
    return;
  }
  const [name, entries] = used[0];
  if (spec.kind === 'custom') return;
  if (entries.length < 2) {
    fail(chapter, `${where}: only ${entries.length} ${name} entry (need at least 2)`);
  }
  for (const entry of entries) {
    const label = entry.label ?? entry.right ?? '';
    if (typeof entry.cells === 'object') continue;
    if (isPlaceholder(label)) {
      fail(chapter, `${where}: placeholder label ${JSON.stringify(label)} in ${name}`);
      break;
    }
  }
}

async function checkFigures(chapter) {
  const sourceDir = path.join(SRC, 'source-figures', chapter);
  const specDir = path.join(SRC, 'figures', chapter);
  const sources = existsSync(sourceDir)
    ? (await readdir(sourceDir)).filter((file) => /\.(md|mmd|txt)$/.test(file))
    : [];
  if (sources.length === 0) return 0;
  if (!existsSync(specDir)) {
    fail(chapter, `no figure specs (expected ${sources.length})`);
    return 0;
  }
  const specs = (await readdir(specDir)).filter((file) => file.endsWith('.json'));
  const specIds = new Set(specs.map((file) => file.replace(/\.json$/, '')));
  for (const source of sources) {
    const id = source.replace(/\.(md|mmd|txt)$/, '');
    if (!specIds.has(id)) fail(chapter, `missing figure spec for ${source}`);
  }
  for (const file of specs) {
    const id = file.replace(/\.json$/, '');
    const raw = await readFile(path.join(specDir, file), 'utf8');
    let spec;
    try {
      spec = JSON.parse(raw);
    } catch (error) {
      fail(chapter, `${file}: invalid JSON (${error.message})`);
      continue;
    }
    if (spec.id !== id) fail(chapter, `${file}: id "${spec.id}" must equal the file name`);
    if (!KINDS.has(spec.kind)) fail(chapter, `${file}: unknown kind "${spec.kind}"`);
    if (isPlaceholder(spec.title) || String(spec.title).trim().length < 4) {
      fail(chapter, `${file}: title must be a real title (got ${JSON.stringify(spec.title)})`);
    }
    checkTones(chapter, spec, file);
    checkSubstance(chapter, spec, file);
    if (isPlaceholder(spec.caption)) {
      warn(chapter, `${file}: no caption (optional, but a one-line caption helps readers)`);
    }
    try {
      const svg = renderFigure(spec);
      if (!svg.startsWith('<svg') || !svg.endsWith('</svg>')) {
        fail(chapter, `${file}: renderer produced malformed SVG`);
      }
    } catch (error) {
      fail(chapter, `${file}: render failed (${error.message})`);
    }
  }
  const inlineCount = await countInlineSpecs(specDir, specs);
  if (specs.length - inlineCount > sources.length) {
    warn(
      chapter,
      `${specs.length - inlineCount} specs for ${sources.length} source figures (extra specs?)`,
    );
  }
  await checkInlineQuotes(chapter, specDir, specs);
  return specs.length;
}

/** Figures that stand in for a fenced block have no `source-figures/*` original. */
async function countInlineSpecs(specDir, specs) {
  let count = 0;
  for (const file of specs) {
    const spec = JSON.parse(await readFile(path.join(specDir, file), 'utf8'));
    if (typeof spec.inlineQuote === 'string' && spec.inlineQuote !== '') count += 1;
  }
  return count;
}

/**
 * Inline figures replace a fenced diagram that lives inside the chapter body,
 * so their `inlineQuote` must match exactly one fenced block in that chapter.
 * A stale quote would silently leave the raw Mermaid block on the page.
 */
async function checkInlineQuotes(chapter, specDir, specs) {
  // Inline figures may stand in for a block in the chapter body *or* in that
  // chapter's exercises / solutions, so all three files are searched.
  const texts = [];
  for (const suffix of ['', '.exercises', '.solutions']) {
    const file = path.join(SRC, 'content', `${chapter}${suffix}.md`);
    if (!existsSync(file)) continue;
    const markdown = await readFile(file, 'utf8');
    const blocks = [];
    let current = null;
    for (const line of markdown.split('\n')) {
      const fence = /^```(\S*)\s*$/.exec(line);
      if (current === null) {
        if (fence !== null) current = { lang: fence[1], lines: [] };
        continue;
      }
      if (line.trim() === '```') {
        blocks.push(current);
        current = null;
        continue;
      }
      current.lines.push(line);
    }
    const prefix = suffix === '' ? '' : `${suffix}.md#`;
    texts.push(...blocks.map((block) => `${prefix}${block.lines.join('\n')}`));
  }
  for (const file of specs) {
    const spec = JSON.parse(await readFile(path.join(specDir, file), 'utf8'));
    if (typeof spec.inlineQuote !== 'string' || spec.inlineQuote === '') continue;
    const hits = texts.filter((text) => text.includes(spec.inlineQuote)).length;
    if (hits !== 1) {
      fail(chapter, `${file}: inlineQuote matches ${hits} fenced blocks (need exactly 1)`);
    }
  }
}

async function checkQuiz(chapter) {
  const file = path.join(SRC, 'quiz', `${chapter}.json`);
  if (!existsSync(file)) {
    fail(chapter, 'missing quiz file');
    return 0;
  }
  let quiz;
  try {
    quiz = JSON.parse(await readFile(file, 'utf8'));
  } catch (error) {
    fail(chapter, `quiz: invalid JSON (${error.message})`);
    return 0;
  }
  if (quiz.chapter !== chapter) fail(chapter, `quiz.chapter must be "${chapter}"`);
  const items = quiz.items ?? [];
  if (items.length < 5 || items.length > 8) {
    fail(chapter, `quiz has ${items.length} items (want 5–8)`);
  }
  const seen = new Set();
  items.forEach((item, index) => {
    const id = item.id ?? `#${index + 1}`;
    if (seen.has(id)) fail(chapter, `duplicate quiz id ${id}`);
    seen.add(id);
    if (id !== `q${index + 1}`) fail(chapter, `quiz id ${id} out of order`);
    if (typeof item.stem !== 'string' || item.stem.trim().length < 8) {
      fail(chapter, `${id}: stem too short`);
    }
    if (!Array.isArray(item.options) || item.options.length !== 4) {
      fail(chapter, `${id}: needs exactly 4 options`);
    } else {
      if (new Set(item.options).size !== item.options.length) {
        fail(chapter, `${id}: duplicate options`);
      }
      for (const option of item.options) {
        if (typeof option !== 'string' || option.trim() === '') {
          fail(chapter, `${id}: empty option`);
        }
      }
    }
    if (!Number.isInteger(item.answer) || item.answer < 0 || item.answer > 3) {
      fail(chapter, `${id}: answer must be an integer 0–3`);
    }
    if (typeof item.explain !== 'string' || item.explain.trim().length < 10) {
      fail(chapter, `${id}: explain too short`);
    }
  });
  const spread = new Set(items.map((item) => item.answer));
  if (items.length >= 5 && spread.size < 3) {
    warn(chapter, `answers cluster on ${[...spread].join(',')}`);
  }
  return items.length;
}

async function main() {
  const filter = process.argv.slice(2);
  const meta = JSON.parse(await readFile(path.join(SRC, 'meta.json'), 'utf8'));
  const chapters = meta.chapters
    .map((chapter) => chapter.slug)
    .filter((slug) => filter.length === 0 || filter.some((value) => slug.startsWith(value)));

  let figures = 0;
  let quizItems = 0;
  for (const chapter of chapters) {
    figures += await checkFigures(chapter);
    quizItems += await checkQuiz(chapter);
  }

  for (const message of warnings) console.log(`warn  ${message}`);
  for (const message of problems) console.log(`FAIL  ${message}`);
  console.log(
    `\n${chapters.length} chapters · ${figures} figures · ${quizItems} quiz items · ` +
      `${problems.length} problems · ${warnings.length} warnings`,
  );
  if (problems.length > 0) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
