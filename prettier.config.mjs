/**
 * Prettier configuration.
 *
 * `prettier-plugin-astro` handles `.astro` files and `prettier-plugin-tailwindcss`
 * sorts Tailwind utility classes. The Tailwind plugin must stay last.
 *
 * @see https://prettier.io/docs/configuration
 * @type {import('prettier').Config}
 */
const config = {
  arrowParens: 'always',
  bracketSameLine: false,
  bracketSpacing: true,
  endOfLine: 'lf',
  htmlWhitespaceSensitivity: 'css',
  overrides: [
    {
      files: '*.astro',
      options: { parser: 'astro' },
    },
    {
      files: ['*.md', '*.mdx'],
      options: { proseWrap: 'preserve' },
    },
    {
      files: ['*.yaml', '*.yml'],
      options: { singleQuote: false },
    },
  ],
  plugins: ['prettier-plugin-astro', 'prettier-plugin-tailwindcss'],
  printWidth: 100,
  proseWrap: 'preserve',
  quoteProps: 'as-needed',
  semi: true,
  singleQuote: true,
  tabWidth: 2,
  trailingComma: 'all',
  useTabs: false,
};

export default config;
