/**
 * Build the tutorial data from `src/tutorial/`.
 *
 *   node scripts/tutorial/build.mjs
 *
 * Inputs (committed):
 *   src/tutorial/meta.json                 chapter list and parts
 *   src/tutorial/content/*.md              imported chapter / exercise / solution markdown
 *   src/tutorial/source-figures/<slug>/*.md  the original figure markdown (reference material)
 *   src/tutorial/figures/<slug>/*.json     SVG specs (see scripts/tutorial/SPEC.md)
 *   src/tutorial/quiz/<slug>.json          auto-graded quiz items
 *
 * Outputs (committed, generated):
 *   src/data/tutorial/generated/<slug>.json
 *   src/data/tutorial/generated/index.json
 *
 * The script is deterministic: running it twice produces byte-identical output,
 * so `pnpm verify` stays meaningful.
 */

import { readFile, readdir, writeFile, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';
import { createHighlighter } from 'shiki';
import { renderFigure } from './svg.mjs';
import { buildReferenceIndex, createResolver } from './links.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');
const SRC = path.join(repo, 'src/tutorial');
const OUT = path.join(repo, 'src/data/tutorial/generated');

/** Parts of the book. Numbers are inclusive two-digit chapter numbers. */
const PARTS = [
  {
    id: 'part-1',
    label: '第一部分 · 宏观',
    title: '背景、设计思想与架构',
    blurb: '先建立正确的心智模型：TileLang 为什么存在、它处在软件栈的哪一层、与同类框架比强在哪。',
    from: '01',
    to: '06',
  },
  {
    id: 'part-2',
    label: '第二部分 · 实用',
    title: '从会用到敢用',
    blurb:
      '按 PyTorch tutorial 的节奏动手：写第一个 kernel、吃透核心原语、完成四类算子实战，最后训练并推理一个 7B 模型。',
    from: '07',
    to: '19',
  },
  {
    id: 'part-3',
    label: '第三部分 · 深入',
    title: '编译期、后端与芯片',
    blurb:
      '把 TileLang 从黑盒变成可读的编译器：IR 与 pass、布局推导、流水与同步、NVIDIA 与昇腾两条后端、性能方法论。',
    from: '20',
    to: '29',
  },
  {
    id: 'part-4',
    label: '第四部分 · 前沿与参与',
    title: '未合入的方向与如何贡献',
    blurb: '区分“已发布 / 只在 main / 只在 PR”三种状态，然后给出从 issue 到 PR 的完整贡献路径。',
    from: '30',
    to: '31',
  },
  {
    id: 'part-5',
    label: '第五部分 · DeepSeek 昇腾全栈',
    title: '一次真实开源的完整拆解',
    blurb:
      'DeepSeek 在 2026-09-30 开源了面向昇腾的一整套基础设施。用 TileLang 当主线，把这六个组件从算法、TileLang 写法、后端实现到 128 卡端到端串成一条完整链路。',
    from: '32',
    to: '39',
  },
];

const READING_WORDS_PER_MINUTE = 320;

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function slugifyHeading(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}

/** Count CJK characters (1 each) and latin words (1 each) for reading time. */
function countWords(text) {
  const cjk = text.match(/[\u3400-\u9fff]/gu)?.length ?? 0;
  const latin = text.match(/[A-Za-z][A-Za-z0-9_'-]*/g)?.length ?? 0;
  return cjk + latin;
}

/** First real paragraph of a chapter, used as the card summary. */
/**
 * Sentences that open a section but say nothing about the chapter itself. A
 * summary built from one of these reads as if the content were missing, so they
 * are skipped in favour of the first paragraph that actually describes the chapter.
 */
const SUMMARY_BOILERPLATE =
  /^(读完本章|读完后|本章|本节|在开始之前|在正式|前置知识|说明|注意|提示|以下|下面|我们|你应当|这一章)/;

function cleanParagraph(paragraph) {
  return paragraph
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[*`_>#]/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\$[^$]*\$/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** A real one-sentence summary of the chapter, used on cards and page headers. */
function chapterSummary(markdown) {
  const paragraphs = markdown
    .replace(/^\s*#\s+[^\n]*\n/, '')
    // Drop fenced blocks first: they contain blank lines, so splitting on
    // paragraph breaks before this would leak code fragments into the summary.
    .replace(/```[\s\S]*?```/g, '\n\n')
    .replace(/^\s*\|.*$/gm, '')
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph !== '');

  for (const paragraph of paragraphs) {
    if (/^(#{1,6}\s|```|\||>|\*|-|\d+\.)/.test(paragraph)) continue;
    const text = cleanParagraph(paragraph);
    if (text.length < 40) continue;
    if (SUMMARY_BOILERPLATE.test(text)) continue;
    if (/[：:]$/.test(text)) continue;
    return text.length > 150 ? `${text.slice(0, 148)}…` : text;
  }
  const fallback = cleanParagraph(paragraphs.find((paragraph) => !paragraph.startsWith('#')) ?? '');
  return fallback.length > 150 ? `${fallback.slice(0, 148)}…` : fallback;
}

// --------------------------------------------------------------------------- //
// Exercises and solutions
// --------------------------------------------------------------------------- //

const EXERCISE_HEADING = /^(#{2,5})\s*E(\d{1,2})\b(.*)$/;
const EXERCISE_BOLD = /^\*\*E(\d{1,2})\s*[（(]([^）)]*)[）)]\*\*\s*(.*)$/;

function classifyKind(text) {
  const value = text.replace(/[★\s]|需要硬件|不需要硬件|无需硬件|无硬件|\[|\]|（|）|\(|\)/g, '');
  for (const kind of ['代码阅读', '概念', '动手', '计算', '分析', '设计', '综合', '实验']) {
    if (value.includes(kind)) return kind;
  }
  return '综合';
}

function parseMeta(text) {
  const difficulty = (text.match(/★/g) ?? []).length || 1;
  const hardware =
    /需要硬件|需 NPU|需NPU|需要 GPU|需要 GPU|需要加速器/.test(text) &&
    !/不需要硬件|无需硬件|无硬件/.test(text);
  return { difficulty: Math.min(difficulty, 3), hardware, kind: classifyKind(text) };
}

/** Split a markdown file into `E<n>` blocks. Handles heading and bold styles. */
function splitByExercise(markdown) {
  const lines = markdown.split('\n');
  const items = [];
  let current = null;
  const intro = [];
  for (const line of lines) {
    const heading = EXERCISE_HEADING.exec(line);
    const bold = heading === null ? EXERCISE_BOLD.exec(line) : null;
    if (heading !== null) {
      if (current !== null) items.push(current);
      current = { id: `E${heading[2]}`, meta: heading[3].trim(), lines: [] };
      continue;
    }
    if (bold !== null) {
      if (current !== null) items.push(current);
      current = { id: `E${bold[1]}`, meta: `${bold[2]} ${bold[3]}`.trim(), lines: [bold[3]] };
      continue;
    }
    if (current === null) intro.push(line);
    else current.lines.push(line);
  }
  if (current !== null) items.push(current);
  return { intro: intro.join('\n'), items };
}

function parseExercises(markdown) {
  const { items } = splitByExercise(markdown);
  return items.map((item) => {
    const { difficulty, hardware, kind } = parseMeta(item.meta);
    return {
      id: item.id,
      kind,
      difficulty,
      hardware,
      title: item.meta
        .replace(/[（(][^）)]*[）)]/g, '')
        .replace(/[★\s]+/g, ' ')
        .trim(),
      markdown: item.lines.join('\n').trim(),
    };
  });
}

function parseSolutions(markdown) {
  const { items } = splitByExercise(markdown);
  const map = new Map();
  for (const item of items) map.set(item.id, item.lines.join('\n').trim());
  return map;
}

// --------------------------------------------------------------------------- //
// Figures
// --------------------------------------------------------------------------- //

/** "对应正文第 2.1 节" / "用于正文 §2.1 与 §4" / "见 §3.4.1" -> "2.1" */
function figureSection(sourceText) {
  const match =
    /(?:对应|用于|见|参考)\s*(?:正文|本章|第)*\s*(?:第|§)\s*(\d+(?:\.\d+)+)/.exec(sourceText) ??
    /§\s*(\d+(?:\.\d+)+)/.exec(sourceText);
  return match?.[1];
}

async function loadFigures(slug, sourceFigureDir) {
  const specDir = path.join(SRC, 'figures', slug);
  if (!existsSync(specDir)) return [];
  const files = (await readdir(specDir)).filter((file) => file.endsWith('.json')).sort();
  const figures = [];
  for (const file of files) {
    const spec = JSON.parse(await readFile(path.join(specDir, file), 'utf8'));
    let section = spec.section;
    if (section === undefined) {
      const base = file.replace(/\.json$/, '');
      for (const extension of ['.md', '.mmd', '.txt']) {
        const sourceFile = path.join(sourceFigureDir, `${base}${extension}`);
        if (!existsSync(sourceFile)) continue;
        section = figureSection(await readFile(sourceFile, 'utf8'));
        break;
      }
    }
    figures.push({
      id: spec.id ?? file.replace(/\.json$/, ''),
      title: spec.title ?? '',
      caption: spec.caption ?? '',
      notes: spec.notes ?? [],
      section,
      inlineQuote: spec.inlineQuote ?? '',
      svg: renderFigure({ ...spec, id: spec.id ?? file.replace(/\.json$/, '') }),
    });
  }
  return figures;
}

function figureMarkup(figure) {
  const notes =
    figure.notes.length === 0
      ? ''
      : `<ul class="tlfig-notes">${figure.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join('')}</ul>`;
  const caption =
    figure.caption === ''
      ? ''
      : `<figcaption class="tlfig-caption">${escapeHtml(figure.caption)}</figcaption>`;
  return `<figure class="tlfig-figure" id="fig-${figure.id}">${figure.svg}${caption}${
    notes
  }</figure>`;
}

/**
 * Is a markdown image target one of the book's diagram sources?
 *
 * The chapters mark every figure with `![说明](figures/<name>.md)`, pointing at
 * the Mermaid or ASCII file it was drawn from — never at an image file. The
 * figure itself is published as an inline SVG (see `figureMarkup`), so such a
 * target is a placeholder, not a picture.
 */
function isFigureSourcePath(src) {
  return /^(?:\.\.\/)*figures\/.+\.(?:md|mmd|txt)$/.test(src.trim());
}

/**
 * Turn references in already-rendered HTML into real links. Code spans are
 * handled first (that is how the book writes most of its citations), then the
 * few relative markdown links; fenced code blocks are protected so a snippet
 * that merely mentions a path is not rewritten.
 */
function rewriteReferenceLinks(html, resolve, slug, unresolved) {
  const protectedBlocks = [];
  const shielded = html.replace(/<pre[^>]*>[\s\S]*?<\/pre>/g, (block) => {
    protectedBlocks.push(block);
    return `\u0000PRE${protectedBlocks.length - 1}\u0000`;
  });

  // `label` replaces the scratch path the book was written against: a reader
  // can only act on the publisher's name, not on `../tilelang-corpus/...`.
  const anchorOpen = (resolved) => {
    const external = resolved.external ? ' target="_blank" rel="noopener noreferrer"' : '';
    const title =
      resolved.title === undefined || resolved.title === ''
        ? ''
        : ` title="原始出处：${escapeHtml(resolved.title)}"`;
    return `<a class="tutorial-ref" href="${resolved.href}"${external}${title}>`;
  };
  /** A `<code>` citation becomes a link whose visible text is the source name. */
  const link = (resolved, inner) => {
    const body =
      resolved.label === undefined || resolved.label === ''
        ? inner
        : `<code>${escapeHtml(resolved.label)}</code>`;
    return `${anchorOpen(resolved)}${body}</a>`;
  };

  /*
    A markdown link whose label is a code span, `` [`figures/x.md`](figures/x.md) ``,
    renders as `<a href="…"><code>…</code></a>`. The two passes below would each
    rewrite one half of it — the code span into a link, then the anchor's opening
    tag into a second one — and nest `<a>` inside `<a>`, which is invalid HTML
    that browsers repair by silently dropping the outer anchor. Rewriting the
    whole construct once, before either pass can see half of it, keeps the
    resolved link and its label exactly as the reader sees them today.

    The result is then shielded like a `<pre>` block: when the resolver has no
    shorter label to offer, the link keeps the author's path as its text, and
    that path is itself resolvable — the code-span pass below would wrap it a
    second time.
  */
  const rewrittenLinks = [];
  let out = shielded.replace(
    /<a href="([^"]+)"><code>([^<]+)<\/code><\/a>/g,
    (match, href, text) => {
      const resolved = resolve(decodeEntities(href), slug);
      if (resolved === null) {
        if (unresolved !== undefined && looksLikeReference(decodeEntities(text))) {
          unresolved.add(decodeEntities(text));
        }
        return match;
      }
      rewrittenLinks.push(link(resolved, `<code>${text}</code>`));
      return `\u0000LNK${rewrittenLinks.length - 1}\u0000`;
    },
  );
  out = out.replace(/<code>([^<]+)<\/code>/g, (match, text) => {
    const resolved = resolve(decodeEntities(text), slug);
    if (resolved === null) {
      if (unresolved !== undefined && looksLikeReference(text)) unresolved.add(text);
      return match;
    }
    return link(resolved, match);
  });
  /*
    `[../../VERSIONS.md](../../VERSIONS.md)` renders the raw path as the link
    text, which a reader cannot act on. When the text is just the path, swap it
    for the resolved label; links whose text was written by the author keep it.
  */
  out = out.replace(/<a href="([^"]+)">([^<]*)<\/a>/g, (match, href, text) => {
    if (/^(https?:|#|\/|mailto:)/.test(href)) return match;
    const target = decodeEntities(href);
    const resolved = resolve(target, slug);
    if (resolved === null) {
      if (unresolved !== undefined && looksLikeReference(href)) unresolved.add(href);
      return match;
    }
    const bare = decodeEntities(text).trim() === target.trim();
    const body =
      bare && resolved.label !== undefined && resolved.label !== ''
        ? `<code>${escapeHtml(resolved.label)}</code>`
        : text;
    return `${anchorOpen(resolved)}${body}</a>`;
  });
  // Links with markup inside keep their own text; only the tag is rewritten.
  out = out.replace(/<a href="([^"]+)">/g, (match, href) => {
    if (/^(https?:|#|\/|mailto:)/.test(href)) return match;
    const resolved = resolve(decodeEntities(href), slug);
    if (resolved === null) {
      if (unresolved !== undefined && looksLikeReference(href)) unresolved.add(href);
      return match;
    }
    return anchorOpen(resolved);
  });
  return out.replace(
    // NUL is a safe placeholder: it cannot occur in rendered markdown.
    // eslint-disable-next-line no-control-regex
    /\u0000(PRE|LNK)(\d+)\u0000/g,
    (_match, kind, index) =>
      (kind === 'PRE' ? protectedBlocks[Number(index)] : rewrittenLinks[Number(index)]) ?? '',
  );
}

/**
 * Is this `<code>` span a reference to project material (and so something the
 * resolver must handle), or just an ordinary inline code snippet?
 */
function looksLikeReference(text) {
  const value = text.trim();
  if (/\s/.test(value)) return false;
  return /(tilelang-corpus|_vendor|^\.\.\/|VERSIONS\.md|references\.md)/.test(value);
}

function decodeEntities(value) {
  return value
    .replace(/&#x([0-9a-fA-F]+);/g, (_m, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_m, dec) => String.fromCodePoint(Number.parseInt(dec, 10)))
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replaceAll('&amp;', '&');
}

/**
 * Replace the fenced Mermaid / ASCII diagram blocks that live *inside* a
 * chapter body with the same SVG figures used for `figures/*.md`. A spec opts in
 * with `inlineQuote`, a verbatim snippet of the block it replaces; the check
 * script verifies each quote matches exactly one block.
 */
function isInlineFigure(figure) {
  return figure.inlineQuote !== undefined && figure.inlineQuote !== '';
}

function injectInlineFigures(html, figures) {
  const pending = figures.filter(isInlineFigure);
  if (pending.length === 0) return html;
  const used = new Set();
  return html.replace(/<pre[^>]*>[\s\S]*?<\/pre>/g, (block) => {
    const text = decodeEntities(block.replace(/<[^>]+>/g, ''));
    for (const figure of pending) {
      if (used.has(figure.id)) continue;
      if (!text.includes(figure.inlineQuote)) continue;
      used.add(figure.id);
      return figureMarkup(figure);
    }
    return block;
  });
}

// --------------------------------------------------------------------------- //
// Markdown rendering
// --------------------------------------------------------------------------- //

async function createRenderer() {
  const highlighter = await createHighlighter({
    themes: ['github-light', 'github-dark'],
    langs: [
      'python',
      'cpp',
      'c',
      'bash',
      'shell',
      'json',
      'yaml',
      'markdown',
      'text',
      'diff',
      'ini',
      'toml',
    ],
  });

  const md = new MarkdownIt({ html: true, linkify: true, breaks: false });

  /** Shared Shiki rendering, so code listings match fenced blocks exactly. */
  const highlight = (code, lang) => {
    const known = highlighter.getLoadedLanguages().includes(lang) && lang !== '';
    try {
      return highlighter.codeToHtml(code, {
        lang: known ? lang : 'text',
        themes: { light: 'github-light', dark: 'github-dark' },
      });
    } catch {
      return `<pre class="shiki"><code>${escapeHtml(code)}</code></pre>`;
    }
  };

  // Server-side KaTeX: the chapters use $...$ and $$...$$ heavily.
  md.use(texmath, {
    engine: katex,
    delimiters: ['dollars', 'brackets'],
    katexOptions: { throwOnError: false, strict: false, output: 'html' },
  });

  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const info = (token?.info ?? '').trim();
    const lang = info.split(/\s+/)[0] ?? '';
    const code = token?.content ?? '';
    return highlight(code, lang);
  };

  const defaultLinkOpen =
    md.renderer.rules.link_open ??
    ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const href = token?.attrGet('href') ?? '';
    if (/^https?:/.test(href)) {
      token?.attrSet('target', '_blank');
      token?.attrSet('rel', 'noopener noreferrer');
    }
    return defaultLinkOpen(tokens, idx, options, env, self);
  };

  /*
    Figure placeholders never reach the page. The book writes each figure as
    `![说明](figures/<name>.md)`, but the build replaces it with the inline SVG
    rendered from that figure's spec; keeping the markdown image would publish
    `<img src="figures/…">`, which 404s for every reader. Real images (any
    extension the book might use later) keep the default rule.
  */
  const defaultImage =
    md.renderer.rules.image ??
    ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const src = tokens[idx]?.attrGet('src') ?? '';
    if (isFigureSourcePath(src)) return '';
    return defaultImage(tokens, idx, options, env, self);
  };
  return { md, highlighter, highlight };
}

/**
 * Render markdown the way the tutorial publishes it. Dropping the figure
 * placeholders above leaves an empty `<p>` where the author wrote `![…](…)`, so
 * the paragraph that held nothing but that placeholder goes too.
 */
function renderMarkdown(md, markdown) {
  return md.render(markdown).replace(/<p>\s*<\/p>/g, '');
}

/** Add stable ids to h2/h3 and collect the table of contents. */
function withHeadingIds(md, markdown) {
  const toc = [];
  const seen = new Map();
  const env = {};
  const tokens = md.parse(markdown, env);
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token?.type !== 'heading_open') continue;
    const level = Number(token.tag.slice(1));
    if (level !== 2 && level !== 3) continue;
    const inline = tokens[i + 1];
    const text = inline?.content ?? '';
    const base = slugifyHeading(text) || `section-${i}`;
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    const id = count === 0 ? base : `${base}-${count + 1}`;
    token.attrSet('id', id);
    toc.push({ id, text, level });
  }
  return { tokens, toc };
}

/**
 * Split the chapter markdown into blocks by `##`/`###` heading so a figure can
 * be attached to the section its source file names.
 */
function splitSections(markdown) {
  const lines = markdown.split('\n');
  const blocks = [];
  let current = { heading: '', number: null, lines: [] };
  for (const line of lines) {
    const match = /^(#{2,3})\s+(.*)$/.exec(line);
    if (match !== null) {
      blocks.push(current);
      const title = match[2] ?? '';
      const numberMatch = /^(\d+(?:\.\d+)*)/.exec(title.trim());
      current = {
        heading: line,
        number: numberMatch?.[1] ?? null,
        lines: [],
      };
      continue;
    }
    current.lines.push(line);
  }
  blocks.push(current);
  return blocks;
}

function renderChapter(md, markdown, figures) {
  // The page header already renders the chapter title, so drop the `# 第 NN 章 …`
  // line that every imported chapter file starts with.
  const body = markdown.replace(/^\s*#\s+[^\n]*\n/, '');
  const blocks = splitSections(body);
  // Figures that stand in for a fenced block inside the body are injected at
  // that block's position, so they must not also be attached to a section or
  // collected into the trailing figure appendix.
  const placed = figures.filter((figure) => !isInlineFigure(figure));
  const used = new Set();
  const html = [];
  const orphan = [];

  for (const block of blocks) {
    const body = block.lines.join('\n');
    if (block.heading !== '') {
      const { tokens, toc } = withHeadingIds(md, block.heading);
      html.push(md.renderer.render(tokens, md.options, {}));
      void toc;
    }
    if (body.trim() !== '') html.push(renderMarkdown(md, body));
    if (block.number === null) continue;
    for (const figure of placed) {
      if (figure.section === undefined || used.has(figure.id)) continue;
      if (figure.section === block.number || figure.section.startsWith(`${block.number}.`)) {
        // Attach to the most specific section: only take it if the next block
        // is not an even deeper sub-section of the same figure section.
        html.push(figureMarkup(figure));
        used.add(figure.id);
      }
    }
  }
  for (const figure of placed) {
    if (!used.has(figure.id)) orphan.push(figure);
  }
  if (orphan.length > 0) {
    html.push('<h2 id="figures">本章图示</h2>');
    html.push('<p>这些图没有在源文件里标注所属小节，集中列在这里，可在读完正文后回看。</p>');
    for (const figure of orphan) html.push(figureMarkup(figure));
  }
  return injectInlineFigures(html.join('\n'), figures);
}

// --------------------------------------------------------------------------- //

/**
 * Read the generated chapters back and fail the build on the rendering bugs a
 * reader would actually notice: KaTeX error spans, unrendered Mermaid/ASCII
 * fences, and reference links that still show a scratch path.
 */
async function assertRenderedOutput(index) {
  const problems = [];
  for (const chapter of index) {
    const payload = JSON.parse(await readFile(path.join(OUT, `${chapter.slug}.json`), 'utf8'));
    const parts = {
      正文: payload.html,
      资料: payload.referencesHtml,
      ...Object.fromEntries(
        (payload.exercises ?? []).map((exercise) => [
          `习题 ${exercise.id}`,
          `${exercise.prompt}${exercise.solution}`,
        ]),
      ),
    };
    for (const [where, html] of Object.entries(parts)) {
      if (typeof html !== 'string') continue;
      if (html.includes('katex-error')) problems.push(`${chapter.slug} · ${where}：KaTeX 渲染失败`);
      for (const block of html.match(/<pre[^>]*>[\s\S]*?<\/pre>/g) ?? []) {
        const text = decodeEntities(block.replace(/<[^>]+>/g, ''));
        if (
          /^\s*(flowchart|graph)\s+(TB|BT|TD|LR|RL)\b/m.test(text) ||
          /^\s*(sequenceDiagram|stateDiagram|classDiagram|erDiagram)\b/m.test(text)
        ) {
          problems.push(`${chapter.slug} · ${where}：还有未转成 SVG 的图表块`);
          break;
        }
      }
      for (const bad of html.matchAll(/<code>([^<]*tilelang-corpus[^<]*)<\/code>/g)) {
        // `.workspace/...` is prose about the writing environment, not a link.
        if (bad[1].trim().startsWith('.workspace/')) continue;
        problems.push(`${chapter.slug} · ${where}：仍显示本地路径 ${bad[1]}`);
      }
      // A figure placeholder that survived rendering ships as
      // `<img src="figures/…">` and 404s for every reader.
      for (const image of html.matchAll(/<img [^>]*src="([^"]*figures\/[^"]+)"/g)) {
        problems.push(`${chapter.slug} · ${where}：图表占位图没有被移除（${image[1]}）`);
      }
    }
  }
  if (problems.length > 0) {
    throw new Error(`渲染自检失败：\n  ${[...new Set(problems)].join('\n  ')}`);
  }
}

/**
 * The corpus' synthesis notes (`src/tutorial/notes/`) are the project's own
 * writing, so they are published alongside the chapters rather than linked to a
 * local scratch path.
 */
async function publishNotes(md, references, vendorRoot, sourceUrls, corpusRoot) {
  const dir = path.join(SRC, 'notes');
  if (!existsSync(dir)) return;
  const resolve = createResolver(references, {
    vendorRoot,
    corpusRoot,
    sourceUrls,
    exists: (target) => existsSync(target),
  });
  const index = [];
  for (const file of (await readdir(dir)).filter((name) => name.endsWith('.md')).sort()) {
    const raw = await readFile(path.join(dir, file), 'utf8');
    const base = file.replace(/\.md$/, '');
    const slug = base === 'README' ? '' : base.replace(/^\d+-/, '');
    // The note's own H1 becomes the page title, so drop it from the body.
    const title = /^#\s+(.+)$/m.exec(raw)?.[1]?.trim() ?? base;
    const body = raw.replace(/^\s*#\s+[^\n]*\n/, '');
    const html = rewriteReferenceLinks(renderMarkdown(md, body), resolve, '');
    await writeFile(
      path.join(OUT, `note-${slug === '' ? 'index' : slug}.json`),
      `${JSON.stringify({ slug, title, html }, null, 2)}\n`,
      'utf8',
    );
    if (slug !== '') index.push({ slug, title });
  }
  await writeFile(
    path.join(OUT, 'notes.json'),
    `${JSON.stringify({ notes: index }, null, 2)}\n`,
    'utf8',
  );
  // Same reason as `chapters.ts`: a static import survives Astro's chunk
  // extraction, `import.meta.glob` inside `getStaticPaths` does not.
  await writeFile(
    path.join(OUT, 'notes.ts'),
    [
      '// Generated by scripts/tutorial/build.mjs — do not edit by hand.',
      ...index.map((note, position) => `import note${position} from './note-${note.slug}.json';`),
      '',
      'export interface TutorialNote {',
      '  slug: string;',
      '  title: string;',
      '  html: string;',
      '}',
      '',
      'export const NOTES: Record<string, TutorialNote> = {',
      ...index.map((note, position) => `  '${note.slug}': note${position} as TutorialNote,`),
      '};',
      '',
      'export const NOTE_SLUGS = Object.keys(NOTES);',
      '',
    ].join('\n'),
    'utf8',
  );
  console.log(`published ${index.length} synthesis notes`);
}

async function main() {
  const meta = JSON.parse(await readFile(path.join(SRC, 'meta.json'), 'utf8'));
  const { md } = await createRenderer();
  await mkdir(OUT, { recursive: true });
  // `code/` and `references.json` hold data that cannot be recreated from a
  // fresh clone alone: the code archive and the reviewed upstream bibliography.
  // Keep both so the public repository remains a complete input for the build.
  for (const entry of await readdir(OUT)) {
    if (entry === 'code' || entry === 'references.json') continue;
    await rm(path.join(OUT, entry), { recursive: true, force: true });
  }

  const referencesFile = path.join(OUT, 'references.json');
  const corpusDir = path.join(repo, '.workspace/research/tilelang-corpus');
  let references;
  if (existsSync(path.join(corpusDir, 'sources/sources.yaml'))) {
    references = await buildReferenceIndex(corpusDir);
  } else if (existsSync(referencesFile)) {
    references = JSON.parse(await readFile(referencesFile, 'utf8'));
  } else {
    references = [];
  }
  // The pinned checkouts live in the git-ignored scratch space; a build without
  // them still works, it just resolves fewer bare paths.
  const vendorRoot = [
    path.join(repo, '.workspace/research/tilelang-book/_vendor'),
    path.join(repo, '_vendor'),
  ].find((candidate) => existsSync(candidate));
  // Upstream URLs for pages that have no `sources.yaml` entry, generated from
  // the crawler manifest by `scripts/tutorial/import-book.mjs`.
  const sourceUrls = existsSync(path.join(SRC, 'source-urls.json'))
    ? JSON.parse(await readFile(path.join(SRC, 'source-urls.json'), 'utf8'))
    : {};
  // Repository subtrees inside the corpus are cloned, so their contents can be
  // checked on disk exactly like the pinned checkouts.
  const corpusRoot = existsSync(corpusDir) ? corpusDir : undefined;
  /** Per-chapter code listings, for the `/code/<slug>/...` pages. */
  async function publishCode(slug) {
    const dir = path.join(SRC, 'code', slug);
    if (!existsSync(dir)) return [];
    const files = [];
    const walk = async (current, prefix) => {
      for (const entry of (await readdir(current, { withFileTypes: true })).sort((a, b) =>
        a.name.localeCompare(b.name),
      )) {
        const relative = prefix === '' ? entry.name : `${prefix}/${entry.name}`;
        if (entry.isDirectory()) {
          if (entry.name === '__pycache__') continue;
          await walk(path.join(current, entry.name), relative);
          continue;
        }
        if (!entry.isFile() || entry.name.endsWith('.pyc')) continue;
        const raw = await readFile(path.join(current, entry.name), 'utf8');
        files.push({ path: relative, content: raw, bytes: raw.length });
      }
    };
    await walk(dir, '');
    if (files.length === 0) return [];
    await mkdir(path.join(OUT, 'code'), { recursive: true });
    await writeFile(
      path.join(OUT, 'code', `${slug}.json`),
      `${JSON.stringify({ slug, files }, null, 2)}\n`,
      'utf8',
    );
    return files;
  }

  const index = [];
  const unresolved = new Set();
  const reportUnresolved = process.env.TUTORIAL_REPORT === '1';
  for (const chapter of meta.chapters) {
    const slug = chapter.slug;
    const chapterMd = await readFile(path.join(SRC, 'content', `${slug}.md`), 'utf8');
    const exercisesMd = await readFile(path.join(SRC, 'content', `${slug}.exercises.md`), 'utf8');
    const solutionsMd = await readFile(path.join(SRC, 'content', `${slug}.solutions.md`), 'utf8');

    const figures = await loadFigures(slug, path.join(SRC, 'source-figures', slug));
    const codeFiles = await publishCode(slug);
    const figureIds = new Set(figures.map((figure) => figure.id));
    const firstFigureId = figures[0]?.id;
    const codePaths = new Set(codeFiles.map((file) => file.path));
    const contextResolve = createResolver(references, {
      vendorRoot,
      corpusRoot,
      sourceUrls,
      exists: (target) => existsSync(target),
      figureIds,
      firstFigureId,
      codeFiles: codePaths,
    });
    const rawHtml = renderChapter(md, chapterMd, figures);
    const html = rewriteReferenceLinks(rawHtml, contextResolve, slug, unresolved);
    const displayed = figures.filter((figure) => !isInlineFigure(figure));

    const referencesPath = path.join(SRC, 'content', `${slug}.references.md`);
    const referencesHtml = existsSync(referencesPath)
      ? rewriteReferenceLinks(
          renderMarkdown(md, await readFile(referencesPath, 'utf8')),
          contextResolve,
          slug,
          unresolved,
        )
      : '';

    const solutions = parseSolutions(solutionsMd);
    const exercises = parseExercises(exercisesMd).map((exercise) => ({
      id: exercise.id,
      kind: exercise.kind,
      difficulty: exercise.difficulty,
      hardware: exercise.hardware,
      title: exercise.title,
      prompt: rewriteReferenceLinks(
        injectInlineFigures(renderMarkdown(md, exercise.markdown), figures),
        contextResolve,
        slug,
        unresolved,
      ),
      solution: rewriteReferenceLinks(
        injectInlineFigures(
          renderMarkdown(md, solutions.get(exercise.id) ?? '（源文件里没有这道题的参考答案。）'),
          figures,
        ),
        contextResolve,
        slug,
        unresolved,
      ),
    }));

    const quizPath = path.join(SRC, 'quiz', `${slug}.json`);
    const quiz = existsSync(quizPath)
      ? (JSON.parse(await readFile(quizPath, 'utf8')).items ?? [])
      : [];

    const { toc } = withHeadingIds(md, chapterMd);
    const words = countWords(chapterMd);
    const payload = {
      slug,
      number: chapter.number,
      title: chapter.title,
      summary: chapter.summary ?? chapterSummary(chapterMd),
      words,
      readingMinutes: Math.max(5, Math.round(words / READING_WORDS_PER_MINUTE)),
      toc,
      html,
      referencesHtml,
      codeFiles: codeFiles.map((file) => ({ path: file.path, bytes: file.bytes })),
      exercises,
      quiz,
      figures: displayed.map((figure) => ({
        id: figure.id,
        title: figure.title,
        section: figure.section ?? null,
      })),
    };
    await writeFile(
      path.join(OUT, `${slug}.json`),
      `${JSON.stringify(payload, null, 2)}\n`,
      'utf8',
    );

    index.push({
      slug,
      number: chapter.number,
      title: chapter.title,
      summary: payload.summary,
      readingMinutes: payload.readingMinutes,
      exerciseCount: exercises.length,
      quizCount: quiz.length,
      figureCount: displayed.length,
      inlineFigureCount: figures.length - displayed.length,
      toc: toc
        .filter((entry) => entry.level === 2)
        .map((entry) => ({ id: entry.id, text: entry.text })),
    });
    console.log(
      `  ${slug}: ${displayed.length}+${figures.length - displayed.length} figures, ${exercises.length} exercises, ${quiz.length} quiz, ${Math.round(words / 1000)}k words`,
    );
  }

  // A typed barrel for the published code listings: `import.meta.glob` inside a
  // page's `getStaticPaths` does not survive Astro's chunk extraction, but a
  // static import does.
  const codeDir = path.join(OUT, 'code');
  if (existsSync(codeDir)) {
    const slugs = (await readdir(codeDir))
      .filter((file) => file.endsWith('.json'))
      .map((file) => file.replace(/\.json$/, ''))
      .sort();
    const codeBarrel = [
      '// Generated by scripts/tutorial/build.mjs — do not edit by hand.',
      "import type { TutorialCodeFile } from '@lib/tutorial.ts';",
      ...slugs.map((slug, position) => `import code${position} from './${slug}.json';`),
      '',
      'export const CODE: Record<string, TutorialCodeFile[]> = {',
      ...slugs.map((slug, position) => `  '${slug}': code${position}.files as TutorialCodeFile[],`),
      '};',
      '',
      'export const CODE_SLUGS = Object.keys(CODE);',
      '',
    ].join('\n');
    await writeFile(path.join(codeDir, 'index.ts'), codeBarrel, 'utf8');
  }

  const versionsSource = path.join(SRC, 'VERSIONS.md');
  if (existsSync(versionsSource)) {
    const versionsHtml = rewriteReferenceLinks(
      renderMarkdown(md, await readFile(versionsSource, 'utf8')),
      createResolver(references, {
        vendorRoot,
        sourceUrls,
        exists: (target) => existsSync(target),
      }),
      '',
    );
    await writeFile(
      path.join(OUT, 'versions.json'),
      `${JSON.stringify({ html: versionsHtml }, null, 2)}\n`,
      'utf8',
    );
  }

  await publishNotes(md, references, vendorRoot, sourceUrls, corpusRoot);

  await writeFile(
    path.join(OUT, 'references.json'),
    `${JSON.stringify(references, null, 2)}\n`,
    'utf8',
  );
  await writeFile(
    path.join(OUT, 'index.json'),
    `${JSON.stringify({ parts: PARTS, chapters: index }, null, 2)}\n`,
    'utf8',
  );

  // A typed barrel so pages can import one chapter without glob gymnastics.
  const barrel = [
    '// Generated by scripts/tutorial/build.mjs — do not edit by hand.',
    "import type { TutorialChapter } from '@lib/tutorial.ts';",
    ...index.map((chapter, position) => `import chapter${position} from './${chapter.slug}.json';`),
    '',
    'export const CHAPTERS: Record<string, TutorialChapter> = {',
    ...index.map(
      (chapter, position) => `  '${chapter.slug}': chapter${position} as TutorialChapter,`,
    ),
    '};',
    '',
    'export const CHAPTER_SLUGS = Object.keys(CHAPTERS);',
    '',
  ].join('\n');
  await writeFile(path.join(OUT, 'chapters.ts'), barrel, 'utf8');
  console.log(`built ${index.length} chapters into src/data/tutorial/generated/`);
  await assertRenderedOutput(index);
  if (reportUnresolved) {
    if (unresolved.size === 0) {
      console.log('all references resolved');
    } else {
      console.log(`unresolved references (${unresolved.size}):`);
      for (const value of [...unresolved].sort()) console.log(`  ${value}`);
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
