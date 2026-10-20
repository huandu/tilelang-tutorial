/**
 * Resolve the references the tutorial text makes to local material.
 *
 * The book was written against a local corpus (`.workspace/research/tilelang-corpus/`)
 * and two pinned source checkouts (`_vendor/tilelang-v0.1.15/`, ...). None of that
 * exists for a reader of the website, so every such reference has to become a
 * link to something reachable:
 *
 *   _vendor/tilelang-v0.1.15/cuda/pipeline.py:129-142
 *     -> https://github.com/tile-ai/blob/v0.1.15/cuda/pipeline.py#L129-L142
 *   ../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md
 *     -> the upstream URL recorded in that corpus entry's sources.yaml
 *   code/03-compile-only/          -> /code/<chapter>/03-compile-only/
 *   figures/01-decision-map.md     -> #fig-01-decision-map
 *   ../../VERSIONS.md              -> /versions/
 *
 * Anything that cannot be resolved falls back to the reference index page so a
 * reader is never left with a dead end.
 */

import { existsSync, readdirSync, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

/** Pinned checkouts the book references, mapped to their upstream repository. */
export const VENDOR_REPOS = [
  {
    dir: 'tilelang-v0.1.15',
    repo: 'tile-ai/tilelang',
    ref: 'v0.1.15',
    note: '全书的分析基线',
  },
  {
    dir: 'tilelang-v0.1.14',
    repo: 'tile-ai/tilelang',
    ref: 'v0.1.14',
    note: '用于对照版本差异',
  },
  {
    dir: 'tilelang-ascend-v0.1.2.000',
    repo: 'tile-ai/tilelang-ascend',
    ref: 'TileLang-ascend-v0.1.2.000-release',
    note: '昇腾适配器，锁定 tag',
  },
];

/**
 * Corpus subtrees that are clones of an upstream repository rather than a
 * harvested page, so a path inside them can be linked on GitHub.
 */
const CORPUS_REPOS = [
  ['05-nvidia-backend/raw/cutlass/', 'NVIDIA/cutlass', 'main'],
  // The corpus keeps a `main` clone, so this subtree is pinned to the commit
  // recorded in its `PROVENANCE.md` rather than to the release tag.
  ['02-tilelang-core-repo/raw/', 'tile-ai/tilelang', '994b44eca1a8'],
  [
    '07-tilelang-ascend-adapter/raw/tilelang-ascend/',
    'tile-ai/tilelang-ascend',
    // Same reasoning as the core repo: the corpus holds a `main` clone.
    'cbc67f2697e8',
  ],
  ['07-tilelang-ascend-adapter/raw/tilelang-mlir-ascend/', 'tile-ai/tilelang-mlir-ascend', 'main'],
  ['07-tilelang-ascend-adapter/raw/TileFoundry/', 'tile-ai/TileFoundry', 'main'],
  ['12-operator-libraries/raw/TileOPs/', 'tile-ai/TileOPs', 'main'],
  ['12-operator-libraries/raw/TileRT/', 'tile-ai/TileRT', 'main'],
  ['12-operator-libraries/raw/tilelang-benchmark/', 'tile-ai/tilelang-benchmark', 'main'],
  ['10-community-and-talks/raw/tilelang-puzzles/', 'tile-ai/tilelang-puzzles', 'main'],
  ['06-huawei-ascend/raw/pto-spec/', 'PTO-ISA/pto-spec', 'main'],
  ['06-huawei-ascend/raw/PTO-Gym/', 'PTO-ISA/PTO-Gym', 'main'],
  ['06-huawei-ascend/raw/pyCircuit/', 'PTO-ISA/pyCircuit', 'main'],
  ['06-huawei-ascend/raw/pto-dsl/', 'huawei-csl/pto-dsl', 'main'],
  ['06-huawei-ascend/raw/pto-kernels/', 'huawei-csl/pto-kernels', 'main'],
  ['06-huawei-ascend/raw/megagdn-pto/', 'huawei-csl/megagdn-pto', 'main'],
  // DeepSeek's 2026-09-30 Ascend release: every subtree is a `main` clone, so
  // the ref is the commit recorded in that category's `PROVENANCE.md`.
  ['14-deepseek-ascend-stack/raw/TileKernels/', 'deepseek-ai/TileKernels', '66258df6175d'],
  ['14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/', 'deepseek-ai/DeepGEMM-Ascend', '8491bbb4b8c0'],
  ['14-deepseek-ascend-stack/raw/DeepEP-Ascend/', 'deepseek-ai/DeepEP-Ascend', '3b25377d04b2'],
  ['14-deepseek-ascend-stack/raw/DeepGEMM/', 'deepseek-ai/DeepGEMM', 'main'],
  ['14-deepseek-ascend-stack/raw/DeepEP/', 'deepseek-ai/DeepEP', 'main'],
  ['14-deepseek-ascend-stack/raw/FlashMLA/', 'deepseek-ai/FlashMLA', '2e5429fc5653'],
  ['14-deepseek-ascend-stack/raw/DeepSelect/', 'deepseek-ai/DeepSelect', 'bfa4507d935f'],
  ['14-deepseek-ascend-stack/raw/DeepJIT/', 'deepseek-ai/DeepJIT', 'main'],
  ['14-deepseek-ascend-stack/raw/Engram/', 'deepseek-ai/Engram', 'main'],
  ['14-deepseek-ascend-stack/raw/DeepSeek-V3.2-Exp/', 'deepseek-ai/DeepSeek-V3.2-Exp', 'main'],
  ['14-deepseek-ascend-stack/raw/DualPipe/', 'deepseek-ai/DualPipe', 'main'],
  ['14-deepseek-ascend-stack/raw/EPLB/', 'deepseek-ai/EPLB', 'main'],
  ['14-deepseek-ascend-stack/raw/clangd-ascend/', 'deepseek-ai/clangd-ascend', 'main'],
];

const CATEGORY_LABELS = {
  '01-tilelang-official-docs': 'TileLang 官方文档',
  '02-tilelang-core-repo': 'TileLang 主干仓库',
  '03-tilelang-papers': '论文',
  '04-language-and-ir': '语言与中间表示',
  '05-nvidia-backend': 'NVIDIA 后端与工具链',
  '06-huawei-ascend': '华为昇腾与 CANN',
  '07-tilelang-ascend-adapter': 'TileLang 昇腾后端',
  '08-frontier-unmerged': '前沿与未合入',
  '09-zhihu-da-zhi-ruo-yu': '知乎（乱序摸鱼）',
  '10-community-and-talks': '社区、课程与讲解',
  '11-related-compilers': '相邻编译器',
  '12-operator-libraries': '算子库与运行时',
  '13-e2e-training-and-serving': '端到端训练与推理',
  '14-deepseek-ascend-stack': 'DeepSeek 昇腾全栈开源',
};

/**
 * The corpus' own synthesis notes (`.workspace/.../90-synthesis/`) are written
 * by this book project, so they are republished under `/notes/` rather
 * than linked away.
 */
const SYNTHESIS_NOTES = {
  '01-knowledge-system.md': {
    slug: 'knowledge-system',
    title: 'TileLang 知识体系与学习路径（编者笔记）',
  },
  '02-compiler-internals-map.md': {
    slug: 'compiler-internals-map',
    title: 'TileLang 编译期内部地图（编者笔记）',
  },
  '03-ascend-route.md': {
    slug: 'ascend-route',
    title: '昇腾路线：现状、差异与入手点（编者笔记）',
  },
  'README.md': { slug: '', title: '编者笔记索引' },
};

/**
 * Paths the chapters cite specifically to prove that they are absent from the
 * pinned release. They must stay as code text instead of becoming 404 links.
 */
const KNOWN_MISSING_VENDOR_PATHS = new Set(['agent/core', 'src/transform/vendored']);

/** Human-readable form of the `:a-b,c` line suffix the book appends to paths. */
function lineLabel(lines) {
  if (lines === undefined || lines === '') return '';
  const parts = lines
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part !== '')
    .map((part) => part.replace('-', '–'));
  if (parts.length === 0) return '';
  return ` 第 ${parts.join('、')} 行`;
}

/** Keep publisher titles readable inside a sentence, full text in `title=`. */
function shortTitle(title) {
  return title.length > 52 ? `${title.slice(0, 51)}…` : title;
}

/** Minimal YAML reader for `sources.yaml`: the file is a flat list of scalars. */
function parseSources(text) {
  const entries = [];
  let current = null;
  let noteLines = [];
  let noteKey = null;
  const flushNote = () => {
    if (current !== null && noteKey !== null)
      current[noteKey] = noteLines.join(' ').replace(/\s+/g, ' ').trim();
    noteLines = [];
    noteKey = null;
  };
  for (const rawLine of text.split('\n')) {
    const line = rawLine.replace(/\t/g, '  ');
    if (/^\s*#/.test(line) || line.trim() === '') continue;
    const item = /^\s*-\s+id:\s*(.+)$/.exec(line);
    if (item !== null) {
      flushNote();
      if (current !== null) entries.push(current);
      current = { id: item[1].trim() };
      continue;
    }
    if (current === null) continue;
    const kv = /^\s{4}(\w+):\s*(.*)$/.exec(line);
    if (kv !== null) {
      flushNote();
      const [, key, value] = kv;
      if (value === '>-' || value === '>' || value === '|' || value === '') {
        noteKey = key;
      } else {
        current[key] = value.replace(/^["']|["']$/g, '');
      }
      continue;
    }
    if (noteKey !== null) noteLines.push(line.trim());
  }
  flushNote();
  if (current !== null) entries.push(current);
  return entries;
}

/** Build the reference index used by the site (also written to disk as JSON). */
export async function buildReferenceIndex(corpusDir) {
  const entries = [];
  const sourcesFile = path.join(corpusDir, 'sources/sources.yaml');
  if (existsSync(sourcesFile)) {
    const parsed = parseSources(await readFile(sourcesFile, 'utf8'));
    for (const source of parsed) {
      if (source.id === undefined || source.url === undefined) continue;
      const category = source.category ?? '';
      entries.push({
        id: source.id,
        category,
        categoryLabel: CATEGORY_LABELS[category] ?? category,
        title: source.title ?? source.id,
        org: source.org ?? '',
        year: source.year ?? '',
        url: source.url,
        trust: source.trust ?? '',
        tags: (source.tags ?? '').replace(/^\[|\]$/g, ''),
        localPath: category === '' ? '' : `${category}/raw/${source.id}.md`,
        note: source.note ?? '',
      });
    }
  }
  return entries;
}

/**
 * Split a trailing `:12-34, :56` line specification off a path. The book
 * sometimes lists several ranges after one path, and a stray comma must never
 * end up inside the file name.
 */
function splitLineSpec(target) {
  let rest = target.trim().replace(/[,\s]+$/, '');
  const specs = [];
  for (;;) {
    const match = /:\s*((?:\d+(?:-\d+)?)(?:\s*,\s*\d+(?:-\d+)?)*)$/.exec(rest);
    if (match === null) break;
    specs.unshift(match[1].replace(/\s+/g, ''));
    rest = rest.slice(0, match.index).replace(/[,\s]+$/, '');
  }
  return { filePath: rest, lines: specs.join(',') };
}

/**
 * File names in the Ascend docs contain spaces, and a raw space in `href` is
 * not a valid URL, so every path segment is encoded.
 */
function encodePath(filePath) {
  return filePath
    .replace(/^\/+/, '')
    .replace(/\/+$/, '')
    .split('/')
    .map((segment) => {
      let decoded = segment;
      try {
        decoded = decodeURIComponent(segment);
      } catch {
        // Leave malformed percent escapes untouched; encodeURIComponent below
        // will turn the literal percent sign into a valid path segment.
      }
      return encodeURIComponent(decoded);
    })
    .join('/');
}

function githubBlob(repo, ref, filePath, lines) {
  const clean = filePath.replace(/^\/+/, '').replace(/\/+$/, '');
  if (clean === '') return `https://github.com/${repo}/tree/${ref}`;
  const anchor =
    lines === undefined || lines === ''
      ? ''
      : (() => {
          const [firstRange = ''] = lines.split(',');
          const [first, last] = firstRange.split('-');
          if (first === '') return '';
          return last !== undefined && last !== '' && Number(last) >= Number(first)
            ? `#L${first}-L${last}`
            : `#L${first}`;
        })();
  return `https://github.com/${repo}/blob/${ref}/${encodePath(clean)}${anchor}`;
}

function githubTree(repo, ref, dirPath) {
  const clean = dirPath.replace(/^\/+/, '').replace(/\/+$/, '');
  return clean === ''
    ? `https://github.com/${repo}/tree/${ref}`
    : `https://github.com/${repo}/tree/${ref}/${encodePath(clean)}`;
}

/**
 * What a path inside a pinned checkout actually is. Submodules are checked out
 * as empty directories, and GitHub has no page for a gitlink path.
 */
function checkoutKind(target) {
  try {
    if (!statSync(target).isDirectory()) return 'file';
    return readdirSync(target).length === 0 ? 'submodule' : 'directory';
  } catch {
    return 'missing';
  }
}

/**
 * macOS filesystems are case-insensitive, so `TileLang` resolves to the
 * `tilelang` package directory locally while GitHub — where paths are
 * case-sensitive — has no such file. The real directory listing is the only
 * authority.
 */
function caseExact(target) {
  try {
    return readdirSync(path.dirname(target)).includes(path.basename(target));
  } catch {
    return false;
  }
}

/**
 * Turn a reference found in the text into a real link.
 *
 * `context.vendorRoot` points at the pinned checkouts, so bare repository paths
 * such as `tilelang/cuda/pipeline.py` can be resolved by existence check: if the
 * file is in exactly one checkout we link there, otherwise we leave it alone.
 */
export function createResolver(index, context) {
  const byLocalPath = new Map();
  /** Source files keyed by every trailing path fragment, for bare-name lookups. */
  const bySuffix = new Map();
  for (const entry of index) {
    if (entry.localPath !== '') byLocalPath.set(entry.localPath, entry);
    if (entry.localPath === '') continue;
    const parts = entry.localPath.split('/');
    for (let i = 0; i < parts.length; i += 1) {
      const key = parts.slice(i).join('/');
      const list = bySuffix.get(key) ?? [];
      list.push(entry);
      bySuffix.set(key, list);
    }
  }

  const exists = (target) => context.exists !== undefined && context.exists(target);
  /** Path inside one pinned checkout, or `undefined` when it is not available. */
  const inVendor = (dir, inside) =>
    context.vendorRoot === undefined
      ? undefined
      : path.join(context.vendorRoot, dir, inside.replace(/\/$/, ''));

  /** `_vendor/<dir>/<path>[:lines]` or a bare path that exists in one checkout. */
  function resolveVendor(target) {
    // The book sometimes writes `../../_vendor/<dir>/<path>` and URL-encodes
    // the spaces in doc file names; both forms mean the same checkout.
    const normalized = target
      .replace(/^(\.\.\/)+/, '')
      .replace(/%20/g, ' ')
      .replace(/\s*\/\s*/g, '/')
      .trim();
    const { filePath, lines } = splitLineSpec(normalized);
    // Commands, brace-expanded sets and deliberately elided paths are prose,
    // not one repository file.  Linking them produces plausible-looking 404s.
    if (
      /[{}]|&&|\|\||(^|\/)\.\.\.($|\/)/.test(filePath) ||
      /(^|\/)\s|\s($|\/)/.test(filePath) ||
      filePath.includes(':') ||
      (filePath.match(/_vendor\//g) ?? []).length > 1 ||
      [...KNOWN_MISSING_VENDOR_PATHS].some((known) => filePath.endsWith(known))
    ) {
      return null;
    }
    const at = filePath.indexOf('_vendor/');
    const relative = at < 0 ? filePath : filePath.slice(at + '_vendor/'.length);
    for (const repo of VENDOR_REPOS) {
      const prefix = `${repo.dir}/`;
      if (at >= 0 && (relative.startsWith(prefix) || relative === prefix.replace(/\/$/, ''))) {
        return vendorLink(repo, relative.slice(prefix.length), lines, true);
      }
    }
    // A bare candidate needs a path separator or a file extension: a lone word
    // like `TileLang` is prose, not a path.
    if (
      at < 0 &&
      /^[A-Za-z0-9_./-]+$/.test(filePath) &&
      !filePath.startsWith('..') &&
      /[./]/.test(filePath)
    ) {
      // Bare repository paths (`tilelang/cuda/pipeline.py`) and doc-relative
      // names (`get_started/overview.md`) both resolve by existence check. A
      // file present in several checkouts counts as the pinned baseline.
      for (const candidate of [filePath, `docs/${filePath}`]) {
        const hits = VENDOR_REPOS.filter((repo) => {
          const target = inVendor(repo.dir, candidate);
          return target !== undefined && exists(target) && caseExact(target);
        });
        if (hits.length === 0) continue;
        const link = vendorLink(hits[0], candidate, lines, true);
        if (link === null) continue;
        return {
          ...link,
          label: candidate === filePath ? undefined : `docs/${filePath}${lineLabel(lines)}`,
        };
      }
    }
    return null;
  }

  /**
   * Link one path inside a pinned checkout. GitHub 404s on `/blob/` for a
   * directory and on both `/blob/` and `/tree/` for a submodule gitlink, so the
   * target is walked up to the nearest page that exists. When the checkout is
   * available it has the final say; without it, obvious shell snippets are
   * still rejected.
   */
  function vendorLink(repo, inside, lines, requireReal) {
    const clean = inside.replace(/\/$/, '');
    // `.git` and friends are repository internals, not documentation.
    if (/(^|\/)\.git($|\/)|['"*|()<>]| -/.test(clean)) return null;
    const local = inVendor(repo.dir, clean);
    if (local !== undefined) {
      if (!caseExact(local)) return null;
      const kind = checkoutKind(local);
      if (kind === 'missing') return requireReal ? null : vendorFile(repo, clean, lines);
      if (kind === 'submodule') {
        const parent = path.dirname(clean);
        return vendorTree(repo, parent === '.' ? '' : parent);
      }
      if (kind === 'directory') return vendorTree(repo, clean);
    }
    return vendorFile(repo, clean, lines);
  }

  const vendorFile = (repo, inside, lines) => ({
    href: githubBlob(repo.repo, repo.ref, inside, lines),
    external: true,
    title: `${repo.repo} @ ${repo.ref}`,
  });

  const vendorTree = (repo, inside) => ({
    href: githubTree(repo.repo, repo.ref, inside),
    external: true,
    title: `${repo.repo} @ ${repo.ref}`,
  });

  /** A bare file name that matches exactly one corpus source. */
  function resolveSourceName(target) {
    const { filePath, lines } = splitLineSpec(target);
    if (!/^[A-Za-z0-9_./-]+\.\w+$/.test(filePath)) return null;
    const hits = bySuffix.get(filePath) ?? [];
    if (hits.length !== 1) return null;
    const only = hits[0];
    return {
      href: only.url,
      external: true,
      label: `${shortTitle(only.title)}${lineLabel(lines)}`,
      exact: true,
      title: only.title,
    };
  }

  /**
   * `../tilelang-corpus/<category>/raw/<file>` — the corpus is private scratch
   * material, so every citation becomes the *original* source: the publisher's
   * URL for a harvested page, a GitHub blob for a repo clone, or one of the
   * republished synthesis notes.
   */
  function resolveCorpus(target) {
    const { filePath, lines } = splitLineSpec(target);
    const marker = 'tilelang-corpus/';
    const at = filePath.indexOf(marker);
    if (filePath.includes('*')) return null;
    // Some citations list several ranges (`…:7-35, :94-118`) or use a
    // placeholder (`<类别>`) in prose; normalise both before matching.
    let local = filePath
      .slice(at < 0 ? 0 : at + marker.length)
      .replace(/:\d+(?:[-,]\d+)*/g, '')
      .replace(/[,\s]+$/, '');
    // `…/TileKernels/…` and `…/TileKernels/...` both mean "that directory".
    // Dropping the ellipsis lets the repo-subtree rules turn it into a tree link.
    if (/(?:…|\.\.\.)$/.test(local)) local = local.replace(/(?:…|\.\.\.)$/, '');
    const suffix = lineLabel(lines);

    // Without the `tilelang-corpus/` marker only these shapes are accepted:
    // the corpus' own bookkeeping files, and paths that start with a corpus
    // category (the synthesis notes write `01-…/raw/<file>` directly).
    if (at < 0) {
      const bookkeeping = /(^|\/)(README|SUMMARY|PROVENANCE|_index)\.md$/.test(local);
      const head = /^(\d\d-[a-z0-9-]+)\//.exec(local);
      const knownCategory = head !== null && CATEGORY_LABELS[head[1]] !== undefined;
      const elided = /^\d\d-\.\.\.\//.test(local);
      const sources = /^(\.\.\/)*sources\//.test(local);
      if (!bookkeeping && !knownCategory && !elided && !sources) return null;
    }

    // A bare `../tilelang-corpus/` (or `../tilelang-corpus/...`) means "the
    // corpus itself" — send the reader to the published reference index.
    if (local === '' || local === '...' || local === '…' || /[<>]/.test(local)) {
      return { href: '/references/', external: false, label: '资料库索引' };
    }
    // `01-.../get_started/Installation.md`: a deliberately elided path.
    const elided = /^(\d\d)-\.\.\.\/(.*)$/.exec(local);
    if (elided !== null) {
      const name = Object.keys(CATEGORY_LABELS).find((key) => key.startsWith(`${elided[1]}-`));
      return {
        href: '/references/',
        external: false,
        label: name === undefined ? '资料库索引' : `${CATEGORY_LABELS[name]} · 资料清单`,
      };
    }

    const synthesis = /^90-synthesis\/(.*)$/.exec(local);
    if (synthesis !== null) {
      const note = SYNTHESIS_NOTES[synthesis[1]];
      if (note !== undefined) {
        return {
          href: note.slug === '' ? '/notes/' : `/notes/${note.slug}/`,
          external: false,
          label: `${note.title}${suffix}`,
          exact: true,
        };
      }
    }

    const entry = byLocalPath.get(local);
    if (entry !== undefined) {
      return {
        href: entry.url,
        external: true,
        label: `${shortTitle(entry.title)}${suffix}`,
        exact: true,
        title: entry.title,
      };
    }

    // Provenance files belong to the private corpus, not to the cloned
    // upstream repository named in the surrounding path.
    if (/(^|\/)PROVENANCE\.md$/.test(local)) {
      return { href: '/references/', external: false, label: '资料库索引', exact: true };
    }

    for (const [prefix, repo, ref] of CORPUS_REPOS) {
      if (!local.startsWith(prefix) && local !== prefix.replace(/\/$/, '')) continue;
      const inside = local.slice(prefix.length);
      // The clone is on disk while the corpus is present, so a reference to a
      // file it does not actually contain can be dropped instead of 404ing.
      if (context.corpusRoot !== undefined) {
        const onDisk = path.join(context.corpusRoot, local);
        const kind = checkoutKind(onDisk);
        if (kind === 'missing') return null;
        const target =
          kind === 'directory'
            ? inside
            : kind === 'submodule'
              ? path.dirname(inside.replace(/\/$/, ''))
              : null;
        if (target !== null) {
          return {
            href: githubTree(repo, ref, target === '.' ? '' : target),
            external: true,
            label: `${repo.slice(repo.indexOf('/') + 1)}/${inside}${suffix}`,
            exact: true,
            title: `${repo} @ ${ref}`,
          };
        }
      }
      const label = `${repo.slice(repo.indexOf('/') + 1)}/${inside}${suffix}`;
      const basename = inside.replace(/\/$/, '').split('/').pop() ?? '';
      const extensionlessFile = new Set(['Dockerfile', 'LICENSE', 'Makefile', 'NOTICE', 'VERSION']);
      const directory =
        inside.endsWith('/') || (!basename.includes('.') && !extensionlessFile.has(basename));
      return directory
        ? {
            href: githubTree(repo, ref, inside),
            external: true,
            label,
            exact: true,
            title: `${repo} @ ${ref}`,
          }
        : {
            href: githubBlob(repo, ref, inside, lines),
            external: true,
            label,
            exact: true,
            title: `${repo} @ ${ref}`,
          };
    }
    const hiascend = /^06-huawei-ascend\/raw\/hiascend\/llms-content\/(.+?)\.md$/.exec(local);
    if (hiascend !== null) {
      // The harvested pages carry no `sources.yaml` entry, so the upstream URL
      // comes from the crawler manifest (`src/tutorial/source-urls.json`).
      const url = context.sourceUrls?.[local];
      if (url === undefined) {
        return { href: '/references/', external: false, label: '资料库索引', exact: true };
      }
      return {
        href: url,
        external: true,
        label: `hiascend.com · ${hiascend[1].split('/').pop()?.replace(/\.md$/, '') ?? ''}${suffix}`,
        exact: true,
        title: '华为昇腾社区文档',
      };
    }
    const arxiv = /^03-tilelang-papers\/raw\/arxiv-(\d{4})-(\d{4,5}).*\.md$/.exec(local);
    if (arxiv !== null) {
      return {
        href: `https://arxiv.org/abs/${arxiv[1]}.${arxiv[2]}`,
        external: true,
        label: `arXiv:${arxiv[1]}.${arxiv[2]}${suffix}`,
        exact: true,
      };
    }
    // Corpus bookkeeping files (`README.md`, `SUMMARY.md`, `PROVENANCE.md`,
    // `_index.md`) describe the corpus itself; the published index replaces them.
    if (/(^|\/)(README|SUMMARY|PROVENANCE|_index)\.md$/.test(local)) {
      return { href: '/references/', external: false, label: '资料库索引', exact: true };
    }
    // `sources/sources.yaml`、`sources/annotated-bibliography.md` 等也由索引页承载。
    if (/^(\.\.\/)*sources\//.test(local)) {
      return { href: '/references/', external: false, label: '资料库索引', exact: true };
    }
    // Anything else that still lives under a known corpus category is a
    // harvested page or a listing directory: the published index covers it.
    const category = /^(\d\d-[a-z0-9-]+)\//.exec(local);
    if (category !== null && CATEGORY_LABELS[category[1]] !== undefined) {
      return {
        href: '/references/',
        external: false,
        label: `${CATEGORY_LABELS[category[1]]} · 资料清单`,
        exact: true,
      };
    }
    return null;
  }

  function resolve(target, slug) {
    const cleaned = target.trim().replace(/^<|>$/g, '');
    if (cleaned === '' || cleaned.startsWith('#') || cleaned.startsWith('http')) return null;

    const vendor = resolveVendor(cleaned);
    if (vendor !== null) return vendor;

    const corpus = resolveCorpus(cleaned);
    if (corpus !== null) return corpus;

    const source = resolveSourceName(cleaned);
    if (source !== null) return source;

    if (/^(\.\.\/)*VERSIONS\.md$/.test(cleaned))
      return { href: '/versions/', external: false, label: '版本锁定' };
    // Author-facing book-production docs: the tutorial home carries the outline
    // and the reading plan, so both point there instead of at a dead file.
    if (/^(\.\.\/)*(WORK-PLAN|OUTLINE)\.md$/.test(cleaned))
      return { href: '/', external: false, label: '教程大纲与阅读计划' };
    if (/^(\.\.\/)*README\.md$/.test(cleaned))
      return { href: '/', external: false, label: '教程首页' };
    if (/^(\.\.\/)*references\.md$/.test(cleaned))
      return { href: '#references', external: false, label: '本章引用' };
    if (/^(\.\.\/)*exercises\.md$/.test(cleaned))
      return { href: '#exercises', external: false, label: '本章习题' };
    if (/^(\.\.\/)*solutions\.md$/.test(cleaned))
      return { href: '#exercises', external: false, label: '参考答案' };

    // `.mmd` is the Mermaid source a figure was drawn from; both point at the
    // rendered SVG on the page.
    const figure = /^(?:\.\.\/)*figures\/(.+)\.(?:md|mmd)$/.exec(cleaned);
    if (figure !== null) {
      const id = path.basename(figure[1]);
      if (context.figureIds !== undefined && context.figureIds.has(id)) {
        return { href: `#fig-${id}`, external: false, label: `图：${id}` };
      }
      return { href: '/', external: false, label: '教程首页' };
    }
    const figureDir = /^(?:\.\.\/)*figures\/?$/.exec(cleaned);
    if (figureDir !== null) {
      // Chapters whose figures are all inline have no trailing `#figures`
      // appendix. Point the directory reference at the first actual figure so
      // the anchor exists in every chapter that has figures.
      if (context.firstFigureId !== undefined) {
        return {
          href: `#fig-${context.firstFigureId}`,
          external: false,
          label: '本章图示',
        };
      }
      return null;
    }

    // Sibling links between the republished synthesis notes.
    const note = /^(?:\.\.\/)*(?:(\d\d)-)?([a-z-]+)\.md$/.exec(cleaned);
    if (note !== null) {
      const target =
        note[1] === undefined
          ? Object.values(SYNTHESIS_NOTES).find((value) => value.slug === note[2])
          : SYNTHESIS_NOTES[`${note[1]}-${note[2]}.md`];
      if (target !== undefined) {
        return {
          href: target.slug === '' ? '/notes/' : `/notes/${target.slug}/`,
          external: false,
          label: target.title,
        };
      }
    }

    const code = /^(?:\.\.\/)*code\/(.*)$/.exec(cleaned);
    if (code !== null) {
      const filePath = code[1].replace(/\/$/, '');
      if (filePath === '' || filePath === 'README.md') {
        return { href: `/code/${slug}/`, external: false };
      }
      if (context.codeFiles !== undefined && context.codeFiles.has(filePath)) {
        return { href: `/code/${slug}/${filePath}/`, external: false };
      }
      const dir = filePath.replace(/\/$/, '');
      const files = context.codeFiles === undefined ? [] : [...context.codeFiles];
      if (files.some((f) => f.startsWith(`${dir}/`))) {
        return { href: `/code/${slug}/${dir}/`, external: false };
      }
      /*
        The book often abbreviates a directory to its number (`code/01` for
        `code/01-thing/`). Match that against the real listing so the reader
        gets a link, and only when the abbreviation is unambiguous.
      */
      const dirs = new Set();
      for (const file of files) {
        const parts = file.split('/');
        for (let i = 1; i < parts.length; i += 1) dirs.add(parts.slice(0, i).join('/'));
      }
      const matches = [...dirs].filter((d) => d.startsWith(`${dir}-`));
      if (matches.length === 1) {
        const only = matches[0];
        return {
          href: `/code/${slug}/${only}/`,
          external: false,
          label: `code/${only}/`,
        };
      }
      return null;
    }
    return null;
  }

  return resolve;
}

export { CATEGORY_LABELS };
