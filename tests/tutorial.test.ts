import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { renderFigure } from '../scripts/tutorial/svg.mjs';

/*
  These tests read the generated tutorial JSON and pin its shape. The files are
  produced by `scripts/tutorial/build.mjs` from this repository's own content,
  so one narrowing assertion per file is cheaper than a full runtime schema —
  and the assertions below are exactly what would catch a generator regression.
*/
/* eslint-disable @typescript-eslint/no-unsafe-type-assertion */

interface Index {
  parts: { id: string; from: string; to: string }[];
  chapters: { slug: string; number: string; exerciseCount: number; quizCount: number }[];
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const generated = path.join(root, 'src/data/tutorial/generated');
const source = path.join(root, 'src/tutorial');

/** The generated JSON is trusted build output; assert once, here. */
function readJson(file: string): unknown {
  return JSON.parse(readFileSync(file, 'utf8')) as unknown;
}

const index = readJson(path.join(generated, 'index.json')) as Index;
const meta = readJson(path.join(source, 'meta.json')) as {
  chapters: { slug: string; number: string; title: string }[];
};

describe('tutorial index', () => {
  it('covers every imported chapter exactly once', () => {
    expect(index.chapters).toHaveLength(meta.chapters.length);
    expect(index.chapters.map((chapter) => chapter.slug).sort()).toEqual(
      meta.chapters.map((chapter) => chapter.slug).sort(),
    );
  });

  it('assigns every chapter to exactly one part', () => {
    for (const chapter of index.chapters) {
      const hits = index.parts.filter(
        (part) => chapter.number >= part.from && chapter.number <= part.to,
      );
      expect(hits, `chapter ${chapter.slug} matched ${String(hits.length)} parts`).toHaveLength(1);
    }
  });

  it('keeps the parts numbered, ordered and contiguous', () => {
    expect(index.parts.map((part) => part.id)).toEqual(
      index.parts.map((_part, position) => `part-${String(position + 1)}`),
    );
    for (const part of index.parts) {
      expect(part.from <= part.to).toBe(true);
    }
    const sorted = [...index.parts].sort((a, b) => a.from.localeCompare(b.from));
    for (let i = 1; i < sorted.length; i += 1) {
      const previous = sorted[i - 1];
      const current = sorted[i];
      if (previous === undefined || current === undefined) continue;
      expect(Number(current.from)).toBe(Number(previous.to) + 1);
    }
    // Every numbered chapter must fall inside some part.
    const covered = new Set(
      index.parts.flatMap((part) =>
        index.chapters
          .filter((chapter) => chapter.number >= part.from && chapter.number <= part.to)
          .map((chapter) => chapter.slug),
      ),
    );
    for (const chapter of index.chapters) expect(covered.has(chapter.slug)).toBe(true);
  });
});

describe('generated chapters', () => {
  for (const entry of index.chapters) {
    const slug = entry.slug;
    const chapter = readJson(path.join(generated, `${slug}.json`)) as {
      slug: string;
      html: string;
      referencesHtml: string;
      toc: { id: string; level: number }[];
      exercises: { id: string; prompt: string; solution: string }[];
      quiz: { id: string; options: string[]; answer: number; explain: string }[];
    };

    it(`${slug}: renders body, toc and exercises`, () => {
      expect(chapter.slug).toBe(slug);
      expect(chapter.html.length).toBeGreaterThan(3000);
      expect(chapter.toc.length).toBeGreaterThan(2);
      expect(chapter.exercises.length).toBeGreaterThanOrEqual(8);
      expect(chapter.exercises.length).toBe(entry.exerciseCount);
      for (const exercise of chapter.exercises) {
        expect(exercise.id).toMatch(/^E\d{1,2}$/);
        expect(exercise.prompt.length).toBeGreaterThan(10);
        expect(exercise.solution.length).toBeGreaterThan(10);
      }
    });

    it(`${slug}: quiz items are answerable`, () => {
      expect(chapter.quiz.length).toBe(entry.quizCount);
      if (chapter.quiz.length === 0) return;
      expect(chapter.quiz.length).toBeGreaterThanOrEqual(5);
      for (const item of chapter.quiz) {
        expect(item.options).toHaveLength(4);
        expect(item.answer).toBeGreaterThanOrEqual(0);
        expect(item.answer).toBeLessThan(4);
        expect(item.explain.length).toBeGreaterThan(10);
      }
    });

    /*
      The four bugs a reader actually reported on the first publish: math left as
      `$…$`, Mermaid fences left as code blocks, references left pointing at the
      private `.workspace/` corpus, and chapters with no prose summary.
    */
    it(`${slug}: renders math, figures and references for a reader`, () => {
      const readerVisible = [
        chapter.html,
        chapter.referencesHtml,
        ...chapter.exercises.flatMap((exercise) => [exercise.prompt, exercise.solution]),
      ].join('\n');
      expect(readerVisible).not.toContain('katex-error');
      expect(readerVisible).not.toMatch(/<code>[^<]*tilelang-corpus[^<]*<\/code>/);
      expect(readerVisible).not.toContain('href="/tilelang');
      // `$…$` that KaTeX did not consume means the delimiter never closed.
      expect(readerVisible).not.toMatch(/(?<!\\)\$[^$\n]{1,120}(?<!\\)\$/);
      for (const block of readerVisible.match(/<pre[^>]*>[\s\S]*?<\/pre>/g) ?? []) {
        const text = block
          .replace(/<[^>]+>/g, '')
          .replaceAll('&#x3C;', '<')
          .replaceAll('&lt;', '<');
        expect(text, `${slug}: a diagram fence was left as a code block`).not.toMatch(
          /^\s*(flowchart|graph)\s+(TB|BT|TD|LR|RL)\b/m,
        );
      }
    });
  }
});

describe('synthesis notes', () => {
  const notes = readJson(path.join(generated, 'notes.json')) as {
    notes: { slug: string; title: string }[];
  };

  it('publishes every corpus synthesis note', () => {
    expect(notes.notes.map((note) => note.slug)).toEqual([
      'knowledge-system',
      'compiler-internals-map',
      'ascend-route',
    ]);
  });

  it('renders each note with links instead of scratch paths', () => {
    for (const note of notes.notes) {
      const page = readJson(path.join(generated, `note-${note.slug}.json`)) as { html: string };
      expect(page.html.length).toBeGreaterThan(2000);
      expect(page.html).not.toMatch(/<code>[^<]*tilelang-corpus[^<]*<\/code>/);
    }
  });
});

describe('figure renderer', () => {
  const base = { id: 'demo', title: '演示图' };

  it('renders every supported kind as SVG', () => {
    const specs = [
      {
        ...base,
        kind: 'flow',
        nodes: [
          { id: 'a', label: '甲' },
          { id: 'b', label: '乙' },
        ],
        edges: [{ from: 'a', to: 'b' }],
      },
      { ...base, kind: 'layers', layers: [{ label: '上层' }, { label: '下层' }] },
      { ...base, kind: 'steps', steps: [{ label: '第一步' }, { label: '第二步' }] },
      {
        ...base,
        kind: 'compare',
        columns: ['A', 'B'],
        rows: [{ label: '行', values: ['1', '2'] }],
      },
      { ...base, kind: 'timeline', items: [{ label: '早' }, { label: '晚' }] },
      {
        ...base,
        kind: 'bars',
        items: [
          { label: '甲', value: 1 },
          { label: '乙', value: 2 },
        ],
      },
      { ...base, kind: 'grid', headers: ['x', 'y'], rows: [{ cells: ['1', '2'] }] },
      {
        ...base,
        kind: 'custom',
        width: 400,
        height: 200,
        shapes: [{ type: 'rect', x: 10, y: 10, width: 80, height: 40, label: '块' }],
      },
    ];
    for (const spec of specs) {
      const svg = renderFigure(spec);
      expect(svg.startsWith('<svg')).toBe(true);
      expect(svg.endsWith('</svg>')).toBe(true);
      expect(svg).toContain('role="img"');
    }
  });

  it('escapes markup in labels instead of injecting it', () => {
    const svg = renderFigure({
      ...base,
      kind: 'layers',
      layers: [{ label: '<script>alert(1)</script>' }, { label: '安全' }],
    });
    expect(svg).not.toContain('<script>');
    expect(svg).toContain('&lt;script&gt;');
  });

  it('rejects unknown kinds loudly', () => {
    expect(() => renderFigure({ ...base, kind: 'nope' })).toThrow(/unknown figure kind/);
  });
});
