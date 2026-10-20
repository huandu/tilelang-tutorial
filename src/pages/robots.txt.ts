import { SITE_URL } from '@lib/site.ts';

export function GET(context: { site?: URL | undefined }): Response {
  const site = context.site ?? new URL(SITE_URL);
  const sitemap = new URL('sitemap-index.xml', site);

  return new Response(['User-agent: *', 'Allow: /', `Sitemap: ${sitemap.href}`, ''].join('\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
