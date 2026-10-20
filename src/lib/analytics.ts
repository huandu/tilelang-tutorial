/**
 * Google Analytics measurement wiring.
 *
 * The public measurement id is safe to expose in HTML. The environment variable
 * remains an override, while the built-in id only activates on an actual Vercel
 * production deployment so local and preview traffic stays out of the report.
 */

import { readEnvString } from './env.ts';

export const GA_MEASUREMENT_ID_ENV = 'GA_MEASUREMENT_ID';
export const DEFAULT_GA_MEASUREMENT_ID = 'G-ZQ5VYWGFV1';

const MEASUREMENT_ID_PATTERN = /^[A-Za-z][A-Za-z0-9-]{3,63}$/u;

export function isGaMeasurementId(value: string): boolean {
  return MEASUREMENT_ID_PATTERN.test(value);
}

/** Pure resolution rule, kept separate from the build environment for tests. */
export function resolveGaMeasurementId(
  configured: string | undefined,
  vercelEnv: string | undefined,
): string | undefined {
  if (configured !== undefined) {
    if (isGaMeasurementId(configured)) return configured;
    console.warn(`[analytics] ignoring ${GA_MEASUREMENT_ID_ENV}: "${configured}" is not a tag id`);
    return undefined;
  }

  return vercelEnv === 'production' ? DEFAULT_GA_MEASUREMENT_ID : undefined;
}

/** The tag id for this build, or `undefined` when analytics must stay off. */
export function readGaMeasurementId(): string | undefined {
  return resolveGaMeasurementId(readEnvString(GA_MEASUREMENT_ID_ENV), readEnvString('VERCEL_ENV'));
}
