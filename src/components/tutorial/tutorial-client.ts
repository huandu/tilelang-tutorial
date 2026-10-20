/**
 * Browser-side behaviour for the `/` tutorial.
 *
 * Everything here is local-only: reading progress, exercise self-assessment and
 * quiz scores live in `localStorage` under a single versioned key, and nothing
 * is ever sent to a server. The mark-up that this script drives is rendered by
 * `src/pages/*.astro`; the data attributes in those files are the
 * contract between the two.
 */

const STORAGE_KEY = 'tilelang-progress-v1';

type ExerciseVerdict = 'correct' | 'partial' | 'wrong';

interface QuizResult {
  score: number;
  total: number;
  at: number;
}

interface ChapterProgress {
  visitedAt?: number;
  read?: boolean;
  exercises?: Record<string, ExerciseVerdict>;
  /** Free-form answers typed into the exercise textareas, capped per entry. */
  notes?: Record<string, string>;
  quiz?: QuizResult;
}

interface Progress {
  chapters: Record<string, ChapterProgress>;
}

function emptyProgress(): Progress {
  return { chapters: {} };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function load(): Progress {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return emptyProgress();
    const parsed: unknown = JSON.parse(raw);
    if (!isPlainObject(parsed) || !isPlainObject(parsed.chapters)) return emptyProgress();
    // The payload is only ever written by `save()` below; a full runtime schema
    // would cost more code than the feature it protects.
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const chapters = parsed.chapters as unknown as Record<string, ChapterProgress>;
    return { chapters };
  } catch {
    return emptyProgress();
  }
}

function save(progress: Progress): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  } catch {
    // Storage can be unavailable (private mode, quota); progress is a nicety.
  }
}

function chapterEntry(progress: Progress, slug: string): ChapterProgress {
  return progress.chapters[slug] ?? {};
}

/** Read-or-create. The mutation lives here so callers pass their own local copy. */
function ensureChapter(progress: Progress, slug: string): ChapterProgress {
  const entry = chapterEntry(progress, slug);
  // `progress` is always a caller-local object here; see the call sites.
  // eslint-disable-next-line no-param-reassign
  progress.chapters[slug] = entry;
  return entry;
}

function touch(slug: string): void {
  if (slug === '') return;
  const progress = load();
  const entry = ensureChapter(progress, slug);
  entry.visitedAt = Date.now();
  save(progress);
}

/** Chapters count as read once every exercise in them has been attempted. */
function isRead(entry: ChapterProgress | undefined): boolean {
  if (entry === undefined) return false;
  if (entry.read === true) return true;
  const verdicts = Object.values(entry.exercises ?? {});
  return verdicts.length > 0;
}

// --------------------------------------------------------------------------- //
// Progress rendering
// --------------------------------------------------------------------------- //

function renderOverview(): void {
  const root = document.querySelector<HTMLElement>('[data-progress-overview]');
  if (root === null) return;
  const total = Number(root.dataset.total ?? '0');
  const progress = load();
  const read = Object.entries(progress.chapters).filter(([, entry]) => isRead(entry)).length;
  const ratio = total === 0 ? 0 : read / total;

  const ring = root.querySelector<SVGCircleElement>('[data-ring-value]');
  if (ring !== null) {
    const radius = Number(ring.getAttribute('r') ?? '52');
    const circumference = 2 * Math.PI * radius;
    ring.style.strokeDasharray = String(circumference);
    ring.style.strokeDashoffset = String(circumference * (1 - ratio));
  }
  for (const node of root.querySelectorAll<HTMLElement>('[data-progress-read]')) {
    node.textContent = String(read);
  }
  const percentNode = root.querySelector<HTMLElement>('[data-progress-percent]');
  if (percentNode !== null) percentNode.textContent = `${String(Math.round(ratio * 100))}%`;

  for (const card of document.querySelectorAll<HTMLElement>('[data-chapter-card]')) {
    const slug = card.dataset.chapterCard ?? '';
    const entry = progress.chapters[slug];
    card.dataset.read = isRead(entry) ? 'true' : 'false';
    const badge = card.querySelector<HTMLElement>('[data-card-status]');
    if (badge !== null) {
      if (isRead(entry)) badge.textContent = '已读完';
      else if (entry !== undefined) badge.textContent = '在读';
      else badge.textContent = '';
    }
  }
}

function renderChapterPanel(slug: string): void {
  const root = document.querySelector<HTMLElement>('[data-chapter-progress]');
  if (root === null) return;
  const entry = load().chapters[slug];
  const readNode = root.querySelector<HTMLElement>('[data-read-state]');
  if (readNode !== null) readNode.textContent = isRead(entry) ? '已读完' : '未读完';

  const exerciseNode = root.querySelector<HTMLElement>('[data-exercise-count]');
  if (exerciseNode !== null) {
    const done = Object.keys(entry?.exercises ?? {}).length;
    exerciseNode.textContent = `${String(done)} / ${root.dataset.exerciseTotal ?? '0'}`;
  }
  const quizNode = root.querySelector<HTMLElement>('[data-quiz-score]');
  if (quizNode !== null) {
    const quiz = entry?.quiz;
    quizNode.textContent =
      quiz === undefined
        ? '未作答'
        : `${String(quiz.score)} / ${String(quiz.total)}（${String(
            Math.round((quiz.score / Math.max(quiz.total, 1)) * 100),
          )}%）`;
  }
}

// --------------------------------------------------------------------------- //
// Exercises
// --------------------------------------------------------------------------- //

function wireExercises(slug: string): void {
  const items = document.querySelectorAll<HTMLElement>('[data-exercise]');
  if (items.length === 0) return;
  const progress = load();
  const entry = ensureChapter(progress, slug);

  const persist = (): void => {
    save(progress);
    renderOverview();
    renderChapterPanel(slug);
  };

  const persistNotes = (): void => {
    save(progress);
  };

  let noteTimer: number | undefined;
  const scheduleNotes = (): void => {
    if (noteTimer !== undefined) window.clearTimeout(noteTimer);
    noteTimer = window.setTimeout(persistNotes, 400);
  };

  const paint = (node: HTMLElement, verdict: ExerciseVerdict | undefined): void => {
    const element = node;
    const id = element.dataset.exercise ?? '';
    element.dataset.state = verdict ?? '';
    for (const button of element.querySelectorAll<HTMLButtonElement>('[data-verdict]')) {
      const value = button.dataset.verdict;
      button.classList.toggle('btn-active', value === verdict);
    }
    if (verdict !== undefined) {
      const answer = element.querySelector<HTMLElement>('[data-answer]');
      if (answer !== null) answer.hidden = false;
      const toggle = element.querySelector<HTMLButtonElement>('[data-reveal]');
      if (toggle !== null) toggle.textContent = '收起参考答案';
    }
    if (id !== '') entry.exercises = { ...(entry.exercises ?? {}), [id]: verdict ?? 'wrong' };
  };

  for (const item of items) {
    const id = item.dataset.exercise ?? '';
    const verdict = entry.exercises?.[id];
    if (verdict !== undefined) paint(item, verdict);

    const reveal = item.querySelector<HTMLButtonElement>('[data-reveal]');
    reveal?.addEventListener('click', () => {
      const answer = item.querySelector<HTMLElement>('[data-answer]');
      if (answer === null) return;
      answer.hidden = !answer.hidden;
      reveal.textContent = answer.hidden ? '查看参考答案' : '收起参考答案';
    });

    for (const button of item.querySelectorAll<HTMLButtonElement>('[data-verdict]')) {
      button.addEventListener('click', () => {
        const value = button.dataset.verdict;
        if (value !== 'correct' && value !== 'partial' && value !== 'wrong') return;
        paint(item, value);
        persist();
      });
    }

    const textarea = item.querySelector<HTMLTextAreaElement>('[data-exercise-notes]');
    if (textarea !== null) {
      textarea.value = entry.notes?.[id] ?? '';
      textarea.addEventListener('input', () => {
        entry.notes = { ...(entry.notes ?? {}), [id]: textarea.value.slice(0, 4000) };
        scheduleNotes();
      });
    }
  }
  persist();
}

// --------------------------------------------------------------------------- //
// Quiz
// --------------------------------------------------------------------------- //

function wireQuiz(slug: string): void {
  const root = document.querySelector<HTMLElement>('[data-quiz]');
  if (root === null) return;
  const questions = Array.from(root.querySelectorAll<HTMLElement>('[data-quiz-question]'));
  if (questions.length === 0) return;
  const submit = root.querySelector<HTMLButtonElement>('[data-quiz-submit]');
  const reset = root.querySelector<HTMLButtonElement>('[data-quiz-reset]');
  const result = root.querySelector<HTMLElement>('[data-quiz-result]');

  const mark = (reveal: boolean): number => {
    let score = 0;
    for (const question of questions) {
      const answer = Number(question.dataset.answer ?? '-1');
      const checked = question.querySelector<HTMLInputElement>('input[type="radio"]:checked');
      const chosen = checked === null ? -1 : Number(checked.value);
      if (reveal && chosen === answer) score += 1;
      for (const option of question.querySelectorAll<HTMLElement>('[data-option]')) {
        const index = Number(option.dataset.option ?? '-1');
        if (reveal) {
          option.dataset.verdict = index === answer ? 'correct' : index === chosen ? 'wrong' : '';
        } else {
          option.dataset.verdict = '';
        }
        const input = option.querySelector<HTMLInputElement>('input');
        if (input !== null) input.disabled = reveal;
      }
      const explain = question.querySelector<HTMLElement>('[data-quiz-explain]');
      if (explain !== null) explain.hidden = !reveal;
    }
    return score;
  };

  submit?.addEventListener('click', () => {
    const score = mark(true);
    if (result !== null) {
      result.hidden = false;
      result.textContent = `得分 ${String(score)} / ${String(questions.length)}（${String(
        Math.round((score / questions.length) * 100),
      )}%）。逐题解析已展开。`;
    }
    submit.disabled = true;
    const progress = load();
    ensureChapter(progress, slug).quiz = { score, total: questions.length, at: Date.now() };
    save(progress);
    renderOverview();
    renderChapterPanel(slug);
  });

  reset?.addEventListener('click', () => {
    mark(false);
    for (const input of root.querySelectorAll<HTMLInputElement>('input[type="radio"]')) {
      input.checked = false;
      input.disabled = false;
    }
    if (result !== null) {
      result.hidden = true;
      result.textContent = '';
    }
    // Inside `submit?.addEventListener` TypeScript narrows the button; here it
    // does not, so the guard is required.
    if (submit !== null) submit.disabled = false;
  });

  const previous = load().chapters[slug]?.quiz;
  if (previous !== undefined && result !== null) {
    result.hidden = false;
    result.textContent = `上次得分 ${String(previous.score)} / ${String(previous.total)}。重新作答后可以再次提交评分。`;
  }
}

// --------------------------------------------------------------------------- //
// Table of contents scroll-spy
// --------------------------------------------------------------------------- //

function wireToc(): void {
  const toc = document.querySelector<HTMLElement>('[data-toc]');
  if (toc === null) return;
  const links = Array.from(toc.querySelectorAll<HTMLAnchorElement>('a[href^="#"]'));
  const targets = links
    .map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))))
    .filter((element): element is HTMLElement => element !== null);
  if (targets.length === 0) return;

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const link of links) {
          const active = decodeURIComponent(link.hash.slice(1)) === entry.target.id;
          if (active) link.setAttribute('aria-current', 'true');
          else link.removeAttribute('aria-current');
        }
        break;
      }
    },
    { rootMargin: '-96px 0px -70% 0px', threshold: 0 },
  );
  for (const target of targets) observer.observe(target);
}

// --------------------------------------------------------------------------- //

function init(): void {
  const slug =
    document.querySelector<HTMLElement>('[data-chapter-slug]')?.dataset.chapterSlug ?? '';
  if (slug !== '') touch(slug);

  renderOverview();
  if (slug !== '') {
    renderChapterPanel(slug);
    wireExercises(slug);
    wireQuiz(slug);
    wireToc();
  }

  const markRead = document.querySelector<HTMLButtonElement>('[data-mark-read]');
  markRead?.addEventListener('click', () => {
    if (slug === '') return;
    const progress = load();
    const entry = ensureChapter(progress, slug);
    entry.read = !isRead(entry);
    save(progress);
    renderOverview();
    renderChapterPanel(slug);
    markRead.textContent = entry.read ? '取消「已读完」标记' : '标记本章已读完';
  });
  if (markRead !== null && slug !== '') {
    markRead.textContent = isRead(load().chapters[slug]) ? '取消「已读完」标记' : '标记本章已读完';
  }

  for (const reset of document.querySelectorAll<HTMLButtonElement>('[data-reset-progress]')) {
    reset.addEventListener('click', () => {
      save(emptyProgress());
      window.location.reload();
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init, { once: true });
} else {
  init();
}
