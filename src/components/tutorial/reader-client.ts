/**
 * Browser-side behaviour for the tutorial reading mode.
 *
 * The markup it drives is printed by
 * `src/components/tutorial/ReaderControls.astro`; the data attributes and the
 * `data-reader` / `data-reader-theme` attributes on <html> are the contract
 * with `src/styles/tutorial-reader.css`. Everything here is local-only: the
 * chosen colour style lives in `localStorage`, and no request is ever made.
 *
 * Reading mode is deliberately *not* persisted across page loads. Fullscreen
 * needs a user gesture, so re-entering the mode on load would either fail
 * silently or trap the reader without a way back; the colour style is the only
 * part that survives a reload.
 */

import {
  READER_THEME_STORAGE_KEY,
  isReaderTheme,
  readerThemeFromSiteTheme,
  siteThemeForReaderTheme,
} from '@lib/tutorial-reader.ts';

import type { ReaderTheme } from '@lib/tutorial-reader.ts';

const root = document.documentElement;

function readStoredTheme(): ReaderTheme | undefined {
  try {
    const raw = localStorage.getItem(READER_THEME_STORAGE_KEY);
    return raw !== null && isReaderTheme(raw) ? raw : undefined;
  } catch {
    // Storage can be unavailable (private mode); the site theme is a fine default.
    return undefined;
  }
}

function storeTheme(theme: ReaderTheme): void {
  try {
    localStorage.setItem(READER_THEME_STORAGE_KEY, theme);
  } catch {
    // Persisting is best effort only.
  }
}

function requestFullscreen(onSuccess: () => void): void {
  try {
    // Called synchronously from the click handler, so the user gesture still
    // counts. iOS Safari has no element fullscreen; there the focus layout is
    // the whole effect, which is why the call must never throw.
    void document.documentElement
      .requestFullscreen({ navigationUI: 'hide' })
      .then(onSuccess)
      .catch(() => {
        // A refused request keeps the distraction-free layout as the fallback.
      });
  } catch {
    // No Fullscreen API at all.
  }
}

function releaseFullscreen(): void {
  try {
    if (document.fullscreenElement === null) return;
    void document.exitFullscreen().catch(() => {
      // Leaving the mode matters more than leaving fullscreen.
    });
  } catch {
    // Browsers without the Fullscreen API (older iOS Safari) have nothing to
    // release, and `document.exitFullscreen` is then missing too.
  }
}

function init(): void {
  const controls = document.querySelector<HTMLElement>('[data-reader-controls]');
  const enterButton = document.querySelector<HTMLButtonElement>('[data-reader-enter]');
  const exitButton = document.querySelector<HTMLButtonElement>('[data-reader-exit]');
  const toolbar = document.querySelector<HTMLElement>('[data-reader-toolbar]');
  if (controls === null || enterButton === null || exitButton === null || toolbar === null) {
    return;
  }

  const themeButtons = [
    ...document.querySelectorAll<HTMLButtonElement>('[data-reader-theme-option]'),
  ];

  let active = false;
  /** The site theme to restore on exit; captured when the mode is entered. */
  let previousTheme: string | null = null;
  /** A browser can refuse fullscreen; only then should Esc handling stay inert. */
  let fullscreenEntered = false;
  /** Hide the floating entry before it can overlap the footer's own links. */
  let footerVisible = false;

  const paintThemeButtons = (theme: ReaderTheme): void => {
    for (const button of themeButtons) {
      const pressed = button.dataset.readerThemeOption === theme;
      button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
      button.classList.toggle('btn-active', pressed);
    }
  };

  const applyTheme = (theme: ReaderTheme): void => {
    root.setAttribute('data-reader-theme', theme);
    root.setAttribute('data-theme', siteThemeForReaderTheme(theme));
  };

  const syncChrome = (): void => {
    enterButton.hidden = active || footerVisible;
    toolbar.hidden = !active;
  };

  const footer = document.querySelector<HTMLElement>('footer');
  const syncFooterVisibility = (): void => {
    const nextVisible = footer !== null && footer.getBoundingClientRect().top < window.innerHeight;
    if (footerVisible === nextVisible) return;
    footerVisible = nextVisible;
    syncChrome();
  };

  const leave = (): void => {
    if (!active) return;
    active = false;
    root.removeAttribute('data-reader');
    root.removeAttribute('data-reader-theme');
    if (previousTheme === null) root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', previousTheme);
    previousTheme = null;
    fullscreenEntered = false;
    syncChrome();
    releaseFullscreen();
    enterButton.focus();
  };

  const enter = (): void => {
    if (active) return;
    active = true;
    previousTheme = root.getAttribute('data-theme');
    const theme = readStoredTheme() ?? readerThemeFromSiteTheme(previousTheme);
    root.setAttribute('data-reader', 'on');
    applyTheme(theme);
    paintThemeButtons(theme);
    syncChrome();
    requestFullscreen(() => {
      fullscreenEntered = true;
    });
    toolbar.focus();
  };

  // The wrapper ships hidden so nothing shows without JavaScript; reveal it and
  // put the two states in sync before wiring the buttons.
  controls.hidden = false;
  syncChrome();
  window.addEventListener('scroll', syncFooterVisibility, { passive: true });
  window.addEventListener('resize', syncFooterVisibility);
  syncFooterVisibility();
  enterButton.addEventListener('click', enter);
  exitButton.addEventListener('click', leave);

  for (const button of themeButtons) {
    button.addEventListener('click', () => {
      const value = button.dataset.readerThemeOption;
      if (!isReaderTheme(value)) return;
      applyTheme(value);
      storeTheme(value);
      paintThemeButtons(value);
    });
  }

  // Pressing Esc leaves fullscreen without going through our button; leaving
  // fullscreen is treated as leaving the mode in both directions.
  document.addEventListener('fullscreenchange', () => {
    if (!active) return;
    if (document.fullscreenElement !== null) {
      fullscreenEntered = true;
      return;
    }
    if (fullscreenEntered) leave();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init, { once: true });
} else {
  init();
}
