/**
 * Reading mode for the tutorial.
 *
 * Every page that renders through `TutorialLayout` gets reading mode for free:
 * the layout prints the controls and `src/components/tutorial/reader-client.ts`
 * drives them. This module holds the pure decisions (which colour styles exist,
 * how a stored value is validated, which DaisyUI theme backs each style) so the
 * browser script stays thin and the rules stay unit testable — see
 * `tests/tutorial-reader.test.ts`.
 */

/** The colour styles a reader can pick between. */
export const READER_THEMES = ['light', 'dark', 'sepia'] as const;

export type ReaderTheme = (typeof READER_THEMES)[number];

/**
 * `localStorage` key for the reader's own colour choice. It is deliberately
 * separate from the site-wide `theme` key: reading mode is a temporary mode, and
 * picking 护眼 must not rewrite the reader's site preference.
 */
export const READER_THEME_STORAGE_KEY = 'tilelang-reader-theme';

/** Button labels for the reader toolbar, in `READER_THEMES` order. */
export const READER_THEME_LABELS: Record<ReaderTheme, string> = {
  light: '浅色',
  dark: '深色',
  sepia: '护眼',
};

/** Narrows an unknown value, such as a stored string, to a reader colour style. */
export function isReaderTheme(value: unknown): value is ReaderTheme {
  return typeof value === 'string' && READER_THEMES.some((theme) => theme === value);
}

/**
 * The reader colour style that matches a site theme, used as the default when
 * the reader has no stored preference. `night` is the only dark site theme, so
 * entering reading mode from a dark site starts dark rather than flashing light.
 */
export function readerThemeFromSiteTheme(siteTheme: string | null | undefined): ReaderTheme {
  return siteTheme === 'night' ? 'dark' : 'light';
}

/**
 * The DaisyUI `data-theme` that backs a reader style.
 *
 * 浅色 / 深色 reuse the site's `winter` / `night` themes, so they stay visually
 * identical to the rest of the site. 护眼 rides on `winter` as well and only
 * overrides the design tokens (see `src/styles/tutorial-reader.css`), which
 * keeps code blocks on the light syntax palette instead of the dark one.
 */
export function siteThemeForReaderTheme(theme: ReaderTheme): 'winter' | 'night' {
  return theme === 'dark' ? 'night' : 'winter';
}
