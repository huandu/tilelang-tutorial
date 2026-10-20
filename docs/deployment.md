# Deployment

## Production

- GitHub repository: `huandu/tilelang-tutorial`
- Vercel project: `tilelang-tutorial`
- Production URL: <https://tilelang-tutorial.huandu.me>
- Cloudflare DNS record:
  `CNAME tilelang-tutorial → 1506863d884e7057.vercel-dns-017.com` (DNS only)
- Build command: `pnpm run build`
- Install command: `pnpm install --frozen-lockfile`
- Node.js: 24.x on Vercel

The site is fully static. No environment variables, functions, storage bindings
or secrets are required.

## Continuous deployment

`main` is the production branch. A push runs GitHub Actions:

```text
install → content checks → astro check → lint → format → tests → build
```

After CI succeeds, Vercel builds from the same commit and updates the production
domain. Pull requests receive preview deployments.

## Local production check

```bash
pnpm verify
pnpm preview
```

Check at least:

- `/` returns the tutorial home.
- Every chapter URL is root-relative and ends with `/`.
- `/notes/`, `/references/` and `/versions/` do not collide with chapters.
- `/code/01-why-tile-dsl/` renders source.
- `robots.txt` points to `sitemap-index.xml`.
- A 404 renders inside the tutorial shell.

## Domain changes

The domain is managed in Cloudflare and attached to the Vercel project. Keep the
subdomain CNAME in DNS-only mode so Vercel can issue and renew its certificate.
Do not modify the apex `huandu.me` records when changing this subdomain.

After DNS propagation:

```bash
dig +short tilelang-tutorial.huandu.me CNAME
curl -I https://tilelang-tutorial.huandu.me/
```

Vercel should issue the TLS certificate automatically after DNS verification.

## Blog handoff

The former blog routes under `/tilelang/*` are permanently redirected to the
same path on this domain with the prefix removed. Examples:

```text
https://huandu.me/tilelang/01-why-tile-dsl/
  → https://tilelang-tutorial.huandu.me/01-why-tile-dsl/

https://huandu.me/tilelang/notes/compiler-internals-map/
  → https://tilelang-tutorial.huandu.me/notes/compiler-internals-map/
```

Canonical links, navigation and article prose in the blog point directly to the
new domain.
