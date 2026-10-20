/**
 * Minimal environment access for build-time configuration.
 *
 * The site is prerendered, so values are resolved while Astro builds the static
 * HTML. Vercel and the invoking shell already provide `process.env`; local
 * `.env` files are loaded as a convenience for development and preview builds.
 */

import { existsSync } from 'node:fs';

const LOCAL_ENV_FILES = ['.env.local', '.env'];
let localEnvLoaded = false;

/** Reads one variable, trimming surrounding whitespace and treating blank as unset. */
export function readEnvString(name: string): string | undefined {
  const value = process.env[name] ?? readLocalEnv(name);
  if (value === undefined) return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function readLocalEnv(name: string): string | undefined {
  loadLocalEnvFiles();
  return process.env[name];
}

function loadLocalEnvFiles(): void {
  if (localEnvLoaded) return;
  localEnvLoaded = true;

  for (const file of LOCAL_ENV_FILES) {
    try {
      if (existsSync(file)) process.loadEnvFile(file);
    } catch (error) {
      console.warn(`[env] could not read ${file}:`, error);
    }
  }
}
