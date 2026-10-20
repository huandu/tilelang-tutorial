# Architecture

## Runtime shape

The project is a static Astro site. Every public route is rendered during the
build and served from Vercel's static output. There are no APIs, databases,
accounts or server-side sessions.

Browser state is deliberately local:

- `tilelang-progress-v1` stores chapters read, exercise self-assessment,
  free-form exercise notes and quiz scores.
- `tilelang-reader-theme` stores the preferred reading-mode colour.
- `theme` stores the normal site-wide light/dark choice.

No browser state is sent over the network.

## Data flow

```text
src/tutorial/*                 script output                 Astro pages
content and quiz   ────────▶  generated JSON / TS  ────────▶  static HTML
figure JSON specs  ──┐
code archive       ──┼──────▶  inline SVG, Shiki HTML
references.json    ──┘
```

`scripts/tutorial/build.mjs` is the only producer of
`src/data/tutorial/generated/`. It is deterministic and reads no private
workspace. The generated files are committed so a normal `astro build` does not
need to regenerate the corpus.

`src/lib/tutorial.ts` is the typed contract between generated data and pages.
It must not import Astro APIs. `src/lib/tutorial-reader.ts` contains pure
decisions used by the reading-mode browser script and has unit tests.

## Routes

| Route                      | Purpose                                                      |
| -------------------------- | ------------------------------------------------------------ |
| `/`                        | Tutorial home, progress and complete outline                 |
| `/<chapter-slug>/`         | Chapter prose, figures, exercises, quiz, code and references |
| `/notes/`                  | Editor notes index                                           |
| `/notes/<note-slug>/`      | Editor note                                                  |
| `/references/`             | Full public reference index                                  |
| `/versions/`               | Pinned versions and verification boundary                    |
| `/code/<chapter-slug>/...` | Directory and file browser for chapter code                  |

Chapter pages and utility pages outrank the root-level dynamic chapter route,
so `/notes/` and `/versions/` do not resolve as chapters.

## Reading mode

`TutorialLayout` mounts the reading controls on chapter pages that opt in. The
client writes `data-reader` and `data-reader-theme` on `<html>`; CSS hides site
chrome and elements marked with `data-reader-hide`, constrains the page to a
readable measure and switches DaisyUI tokens. Index, notes, code, version and
reference pages keep the normal shell so their footer spacing matches the site.

Fullscreen requires a user gesture, so reading mode is intentionally not
restored across navigation. Only the selected colour is persisted.

The floating entry control is independent of page layout. It hides as soon as
the footer enters the viewport, which prevents it from covering footer links
without adding compensation padding or empty space to the footer.

## Dependency boundaries

- Browser code may import pure modules only.
- Generator code runs in Node and may import Node built-ins, markdown-it, KaTeX,
  Shiki and the local SVG renderer.
- Pages may import generated data, tutorial types, components, layout and site
  metadata.
- There is no dependency on another repository or site package.

## Failure modes

- A missing figure spec is caught by `pnpm tutorial:check`.
- A malformed quiz is caught before the site builds.
- A stale generated JSON file is caught by tests and an ordinary build error.
- A private corpus path in reader-visible HTML is caught by `tests/tutorial.test.ts`.
- A broken root-level internal link should be treated as a content bug even if
  Vercel's static server would otherwise return 404.
