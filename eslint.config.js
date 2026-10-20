import js from '@eslint/js';
import { defineConfig, globalIgnores } from 'eslint/config';
import prettier from 'eslint-config-prettier';
import astro from 'eslint-plugin-astro';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import globals from 'globals';
import tseslint from 'typescript-eslint';

/**
 * Flat ESLint configuration, ordered from broadest to most specific.
 *
 * Layer order matters: base JS rules, then the `typescript-eslint` type-aware
 * presets, then Astro/a11y support, then targeted relaxations, and finally
 * `eslint-config-prettier` so formatting is owned by Prettier alone.
 *
 * `.astro` templates are parsed by `astro-eslint-parser`, which cannot fully
 * resolve types inside template expressions. A small set of type-aware rules is
 * therefore disabled for `.astro` only — see the comment on that block.
 */
export default defineConfig([
  globalIgnores([
    'dist/',
    'node_modules/',
    '.astro/',
    '.vercel/',
    '.workspace/',
    'pnpm-lock.yaml',
    // Imported verbatim from the book and produced by scripts/tutorial/build.mjs.
    'src/tutorial/content/',
    'src/tutorial/source-figures/',
    'src/tutorial/notes/',
    'src/tutorial/code/',
    'src/data/tutorial/generated/',
  ]),

  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,

  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    linterOptions: {
      reportUnusedDisableDirectives: 'error',
      reportUnusedInlineConfigs: 'error',
    },
    rules: {
      // Correctness
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'no-implicit-coercion': ['error', { boolean: false }],
      'no-param-reassign': ['error', { props: true }],
      'no-unneeded-ternary': 'error',
      'no-useless-rename': 'error',
      'object-shorthand': ['error', 'always'],
      'prefer-const': ['error', { destructuring: 'all' }],
      'prefer-template': 'error',

      // Type safety
      '@typescript-eslint/consistent-type-exports': [
        'error',
        { fixMixedExportsWithInlineTypeSpecifier: true },
      ],
      '@typescript-eslint/consistent-type-imports': [
        'error',
        { fixStyle: 'inline-type-imports', prefer: 'type-imports' },
      ],
      '@typescript-eslint/explicit-function-return-type': [
        'error',
        { allowExpressions: true, allowTypedFunctionExpressions: true },
      ],
      '@typescript-eslint/explicit-module-boundary-types': 'error',
      '@typescript-eslint/method-signature-style': ['error', 'property'],
      '@typescript-eslint/no-import-type-side-effects': 'error',
      '@typescript-eslint/no-unnecessary-condition': [
        'error',
        { allowConstantLoopConditions: 'only-allowed-literals' },
      ],
      '@typescript-eslint/no-unsafe-type-assertion': 'error',
      '@typescript-eslint/prefer-readonly': 'error',
      '@typescript-eslint/require-array-sort-compare': ['error', { ignoreStringArrays: true }],
      '@typescript-eslint/strict-boolean-expressions': [
        'error',
        {
          allowAny: false,
          allowNullableBoolean: false,
          allowNullableEnum: false,
          allowNullableNumber: false,
          allowNullableObject: false,
          allowNullableString: false,
          allowNumber: false,
          allowRuleToRunWithoutStrictNullChecksIKnowWhatIAmDoing: false,
          allowString: false,
        },
      ],
      '@typescript-eslint/switch-exhaustiveness-check': [
        'error',
        { considerDefaultExhaustiveForUnions: true },
      ],
    },
  },

  // Astro components: recommended Astro rules plus strict accessibility checks.
  ...astro.configs.recommended,
  ...astro.configs['flat/jsx-a11y-strict'],

  {
    files: ['**/*.astro'],
    languageOptions: {
      parserOptions: {
        // `astro-eslint-parser` supports the legacy `project` option only.
        project: true,
        projectService: false,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: { 'jsx-a11y': jsxA11y },
    rules: {
      ...jsxA11y.flatConfigs.strict.rules,

      // Template expressions in `.astro` are not fully typed by
      // `astro-eslint-parser`, so the `no-unsafe-*` family reports on every
      // rendered expression. They stay enabled for `.ts`/`.js` sources below.
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/restrict-template-expressions': 'off',

      // `{value && <markup />}` is the idiomatic guard in Astro templates and
      // `value` is already `T | undefined` from the component props.
      '@typescript-eslint/strict-boolean-expressions': [
        'error',
        {
          allowNullableBoolean: true,
          allowNullableNumber: true,
          allowNullableObject: true,
          allowNullableString: true,
        },
      ],

      // Frontmatter values are typed through `Props`, not through inference.
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-unnecessary-condition': 'off',

      // Astro specific conventions.
      'astro/no-set-html-directive': 'error',
      'astro/prefer-class-list-directive': 'error',
      'astro/prefer-object-class-list': 'error',
      'astro/prefer-split-class-list': 'error',
      'astro/sort-attributes': 'error',
    },
  },

  // Tests may lean on non-null assertions and looser return types.
  {
    files: ['**/*.test.ts', '**/*.spec.ts', 'tests/**/*.ts'],
    rules: {
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },

  // Config files are plain scripts with dynamic imports.
  {
    files: ['**/*.config.{js,mjs,ts}', 'eslint.config.js'],
    rules: {
      '@typescript-eslint/explicit-function-return-type': 'off',
      // Config files import packages that ship no type declarations
      // (e.g. `eslint-plugin-jsx-a11y`), so member access cannot be typed.
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
    },
  },

  // `scripts/` holds plain JavaScript that Node runs directly, with no type
  // annotations to check. The base rules still apply; the rules that need type
  // information are turned off rather than fed guesses.
  {
    files: ['scripts/**/*.{js,mjs}'],
    ...tseslint.configs.disableTypeChecked,
    rules: {
      ...tseslint.configs.disableTypeChecked.rules,
      // Plain JavaScript has no type annotations to declare.
      '@typescript-eslint/explicit-function-return-type': 'off',
    },
  },

  prettier,
  {
    /*
      The tutorial generator scripts are plain Node ESM. They are still
      type-checked by `astro check`, but the TypeScript-syntax rules below
      cannot apply to a `.mjs` file that carries no type annotations.
    */
    files: ['scripts/**/*.{mjs,js}'],
    rules: {
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/restrict-template-expressions': 'off',
    },
  },
]);
