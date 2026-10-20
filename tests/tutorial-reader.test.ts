import { describe, expect, it } from 'vitest';

import {
  READER_THEMES,
  READER_THEME_LABELS,
  READER_THEME_STORAGE_KEY,
  isReaderTheme,
  readerThemeFromSiteTheme,
  siteThemeForReaderTheme,
} from '../src/lib/tutorial-reader.ts';

describe('reader colour styles', () => {
  it('offers exactly the three documented styles', () => {
    expect([...READER_THEMES]).toEqual(['light', 'dark', 'sepia']);
    for (const theme of READER_THEMES) {
      expect(READER_THEME_LABELS[theme]).not.toBe('');
    }
  });

  it('keeps the reader preference out of the site theme key', () => {
    expect(READER_THEME_STORAGE_KEY).not.toBe('theme');
    expect(READER_THEME_STORAGE_KEY).toMatch(/^tilelang-/u);
  });

  it('accepts only the known styles', () => {
    expect(isReaderTheme('light')).toBe(true);
    expect(isReaderTheme('dark')).toBe(true);
    expect(isReaderTheme('sepia')).toBe(true);
    expect(isReaderTheme('night')).toBe(false);
    expect(isReaderTheme('')).toBe(false);
    expect(isReaderTheme(null)).toBe(false);
    expect(isReaderTheme(42)).toBe(false);
  });

  it('defaults the reader style from the current site theme', () => {
    expect(readerThemeFromSiteTheme('night')).toBe('dark');
    expect(readerThemeFromSiteTheme('winter')).toBe('light');
    // The no-JS first paint has no attribute yet; light is the safe default.
    expect(readerThemeFromSiteTheme(null)).toBe('light');
    expect(readerThemeFromSiteTheme(undefined)).toBe('light');
  });

  it('backs 浅色 and 护眼 with the light theme and 深色 with the dark one', () => {
    expect(siteThemeForReaderTheme('light')).toBe('winter');
    expect(siteThemeForReaderTheme('sepia')).toBe('winter');
    expect(siteThemeForReaderTheme('dark')).toBe('night');
  });
});
