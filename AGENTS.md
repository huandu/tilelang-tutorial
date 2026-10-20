# AGENTS.md

本文件适用于整个 `tilelang-tutorial` 仓库。它是 AI 代理和贡献者的操作约定。

## 项目目标

这是一个完全独立的 TileLang 中文教程 Astro 应用，发布在
<https://tilelang-tutorial.huandu.me>。它不得依赖 `ai-thoughts-blog` 的代码、
资源、构建产物或环境变量。

站点完全预渲染。浏览器功能包括阅读进度、习题自评、章节自测和阅读模式，
所有状态只保存在本机 `localStorage`，不得引入账户系统或服务端持久化。

## 技术与命令

- Astro 7，`output: 'static'`，Vercel adapter。
- TypeScript 5.9，继承 `astro/tsconfigs/strictest`。
- Tailwind CSS 4、DaisyUI 5（`winter` / `night`）、typography。
- pnpm 11，Node.js 22.12 或更新版本。
- ESLint flat config、Prettier、Vitest 5。

```bash
pnpm install
pnpm dev
pnpm tutorial:build
pnpm tutorial:check
pnpm check
pnpm lint
pnpm format:check
pnpm test
pnpm build
pnpm verify
```

`pnpm verify` 是提交门禁。修改源内容后先运行 `tutorial:build`，再运行完整门禁。

## 目录边界

```text
src/tutorial/               手写内容与不可由外部仓库恢复的源数据
src/data/tutorial/generated 生成数据，和源文件一起提交
scripts/tutorial/           生成器、链接改写、SVG 渲染与校验
src/lib/                    站点元数据、类型和纯函数
src/components/tutorial/    Astro 组件与浏览器脚本
src/pages/                  文件路由；章节位于根路径
src/styles/                 Tailwind、图表与阅读模式样式
docs/                       维护文档
tests/                      只测试纯模块与生成数据形状
```

保持以下分工：

- `src/lib/tutorial.ts` 不得导入 Astro。
- `scripts/tutorial/build.mjs` 必须是确定性的；同一输入连续运行两次应产生相同输出。
- 浏览器脚本只能导入 `src/lib/tutorial-reader.ts` 这类纯模块，不得导入 Node
  文件系统、服务端配置或生成器。
- `src/tutorial/content/`、`source-figures/`、`code/`、`notes/` 和生成数据
  不参与 Prettier/ESLint 源码格式化，但必须通过内容校验和测试。

## 路由

- `/`：教程首页、进度、阅读路径和全部章节。
- `/<chapter-slug>/`：章节正文、习题、自测、代码入口和引用。
- `/notes/`、`/notes/<slug>/`：编者笔记。
- `/references/`、`/versions/`：资料索引和版本锁定。
- `/code/<chapter-slug>/...`：章节代码目录和文件。

不要重新引入 `/tilelang` 前缀。站内链接必须使用根路径。

## 内容约定

- 引号统一为 `“”`，不要使用 `「」` 或 ASCII 双引号。
- 先给判断，再给证据。不要用“这一节是”“接下来是”起句。
- 中文要自然，避免翻译腔、空泛形容词和口号式连接词。
- 中文与拉丁字母、数字之间留空格。
- 破折号和冒号尽量少用。能断成完整句子时就用句号。
- 不使用 emoji。
- 技术结论必须能追溯到锁定版本的源码或公开资料。无法确认的内容明确写“未确认”。
- 区分三类状态：release 已具备、只在 `main`、只在 PR/RFC。

## 图表

- 图只在需要表达关系时使用：并列字段、单源多消费者、时间序列、层级或状态变化。
- 手写图采用 `src/tutorial/figures/<slug>/NN.json` 规格，由
  `scripts/tutorial/svg.mjs` 生成响应式内联 SVG。
- 颜色必须来自 DaisyUI 语义 token 或 `tlfig-*` 样式，不得写死十六进制。
- 先为手机设计：纵向堆叠、2×2 网格、短标签优先于长句。
- 每个源图都应有规格；`pnpm tutorial:check` 会检查覆盖、占位符和渲染错误。

## 样式

- 先使用 DaisyUI 组件，再用 Tailwind 处理布局与间距。
- 颜色只使用语义 token：`base-100`、`base-content`、`primary`、`success` 等。
- 不使用 `dark:` 变体或 `slate-*`、`blue-*` 等原始色板。
- `success`、`warning`、`error` 主要用作表面色；状态优先用 badge、alert 或背景。
- 阅读模式通过 `<html>` 上的 `data-reader` 与 `data-reader-theme` 驱动，
  只隐藏带 `data-reader-hide` 的导航元素。

## TypeScript 与测试

- 对象形状优先使用 `interface`。
- 开启 `exactOptionalPropertyTypes`；可选的透传属性写成 `prop?: T | undefined`。
- `.ts` 代码遵守 `strict-boolean-expressions`，不要依赖隐式真值判断。
- 修改纯函数时扩展对应测试；修改生成器时更新生成数据形状测试。
- 测试不得访问网络，也不得依赖私有 `.workspace/`。

## 发布

- GitHub：`huandu/tilelang-tutorial`。
- Vercel：`tilelang-tutorial`，生产域名为
  `tilelang-tutorial.huandu.me`。
- `main` 的每次推送都通过 CI 并触发生产部署。
- 部署细节见 [docs/deployment.md](./docs/deployment.md)。

## 完成定义

一项改动完成，至少满足：

1. 内容和代码边界符合本文件。
2. 受影响的生成数据已重新生成并提交。
3. `pnpm verify` 通过。
4. 新增外部链接可公开访问，站内链接遵循根路径。
5. 文档、README 或 AGENTS 中的命令与结构同步更新。
