# Content maintenance

## Source layout

```text
src/tutorial/
├─ meta.json                     chapter list and parts
├─ content/<slug>.md             chapter prose
├─ content/<slug>.exercises.md   exercises
├─ content/<slug>.solutions.md   reference answers
├─ content/<slug>.references.md  per-chapter source list
├─ source-figures/<slug>/*       authored figure source
├─ figures/<slug>/*.json         renderer specs
├─ quiz/<slug>.json              multiple-choice questions
├─ code/<slug>/**                chapter examples
├─ notes/*.md                    editor notes
├─ VERSIONS.md                   pinned upstream versions
└─ source-urls.json              public URLs for harvested material
```

`meta.json` is the authoritative chapter order and grouping. A chapter normally
has one file for prose, exercises and answers. A quiz may be omitted only if the
corresponding page and metadata are also updated consistently.

## Regenerating the site data

```bash
pnpm tutorial:build
pnpm tutorial:check
pnpm test
```

The generator reads markdown with markdown-it, renders equations with KaTeX,
highlights code with Shiki and renders figures through `scripts/tutorial/svg.mjs`.
It rewrites local scratch paths to public source URLs and root-relative site
links. Do not hand-edit generated JSON or TypeScript.

## Adding a chapter

1. Add the chapter entry to `meta.json`.
2. Add prose, exercises, solutions and the references file.
3. Add source figures and matching JSON specs.
4. Add a quiz with 5–8 items.
5. Add a `code/<slug>/` directory if the chapter has runnable examples.
6. Run `pnpm tutorial:build` and inspect the chapter in the browser.
7. Run `pnpm verify`.

## Figures

Figure JSON schemas and examples are in
[`scripts/tutorial/SPEC.md`](../scripts/tutorial/SPEC.md). Every visual should
show a relationship that prose does not already carry in one sentence. Design
for a phone first: vertical stacks and 2×2 grids are better than wide rows.

Use semantic classes only. A figure must keep its meaning in both light and dark
themes and remain legible in the sepia reading mode.

## Quizzes

Quiz rules are in [`scripts/tutorial/QUIZ.md`](../scripts/tutorial/QUIZ.md).
Questions must be unambiguous, have exactly four options, exactly one correct
answer and an explanation that teaches the reasoning. Avoid trivia whose answer
changes with every upstream release unless the version is named.

## References

Technical claims should point to one of:

- an upstream source file at a pinned tag and line range;
- official documentation or a paper;
- a clearly labelled community source for hardware facts not in vendor docs.

If evidence is unavailable, write “未确认” instead of guessing. Do not expose
private scratch paths in reader-visible text.

## Chinese style

- Use `“”` for quotes.
- Lead with the judgement.
- Prefer complete sentences over long dash-linked clauses.
- Keep spaces between Chinese and Latin text or digits.
- Avoid slogan language, translationese and emoji.
- Explain a term the first time it appears.
