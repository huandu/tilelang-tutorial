import { describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_GA_MEASUREMENT_ID,
  isGaMeasurementId,
  resolveGaMeasurementId,
} from '../src/lib/analytics.ts';

describe('Google Analytics configuration', () => {
  it('uses the configured measurement id everywhere', () => {
    expect(resolveGaMeasurementId('G-TEST123', 'preview')).toBe('G-TEST123');
  });

  it('enables the built-in tag only on Vercel production', () => {
    expect(resolveGaMeasurementId(undefined, 'production')).toBe(DEFAULT_GA_MEASUREMENT_ID);
    expect(resolveGaMeasurementId(undefined, 'preview')).toBeUndefined();
    expect(resolveGaMeasurementId(undefined, undefined)).toBeUndefined();
  });

  it('rejects malformed ids instead of emitting them into script and URLs', () => {
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(resolveGaMeasurementId('bad id"', 'production')).toBeUndefined();
    expect(warning).toHaveBeenCalledOnce();
    warning.mockRestore();
  });

  it('accepts Google tag id shapes', () => {
    expect(isGaMeasurementId(DEFAULT_GA_MEASUREMENT_ID)).toBe(true);
    expect(isGaMeasurementId('GT-TEST123')).toBe(true);
    expect(isGaMeasurementId('')).toBe(false);
  });
});
