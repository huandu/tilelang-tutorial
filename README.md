# TileLang 系统教程

[![CI](https://github.com/huandu/tilelang-tutorial/actions/workflows/ci.yml/badge.svg)](https://github.com/huandu/tilelang-tutorial/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-tilelang--tutorial.huandu.me-2563eb)](https://tilelang-tutorial.huandu.me)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-3da639)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-2563eb)](./LICENSE)

一套面向“会用 PyTorch、但没写过 kernel”的研发人员的 TileLang 中文系统教程。
教程从性能直觉和编程模型讲起，经过算子实战、编译期与两条硬件后端，
最后拆解 DeepSeek 面向昇腾开源的整套基础设施。

在线阅读：**<https://tilelang-tutorial.huandu.me>**

## 内容

- 39 章，五部分递进：宏观模型、实用开发、编译期与后端、前沿与贡献、
  DeepSeek 昇腾全栈。
- 每章附可交互习题、参考答案、章节自测和可浏览的示例代码。
- 阅读进度、习题自评和测验分数只保存在浏览器 `localStorage`，没有服务端持久化。
- 238 张由 JSON 规格生成的响应式 SVG 图，跟随浅色、深色与护眼阅读模式切换。
- 三篇编者笔记与完整的上游资料索引。
- 全书以 TileLang `v0.1.15` 和 TileLang-Ascend `v0.1.2.000-release` 为分析基线；
  只在 `main` 或 PR 中存在的能力会单独标注。

## 技术栈

Astro 7、TypeScript 5.9、Tailwind CSS 4、DaisyUI 5、KaTeX、Shiki、
markdown-it、Vitest 5 和 pnpm 11。站点完全预渲染，由 Vercel 发布。

这个仓库不依赖作者的博客、私有资料库或任何外部构建脚本。章节正文、图规格、
题库、代码归档和引用元数据都在仓库内。

## 本地运行

需要 Node.js 22.12 或更新版本，以及 pnpm 11。

```bash
pnpm install
pnpm dev
```

开发服务器默认运行在 <http://localhost:4321>。

## 常用命令

```bash
pnpm dev               # 本地开发
pnpm tutorial:build     # 从 src/tutorial/ 重新生成页面数据
pnpm tutorial:check     # 校验题库、图表规格和原图覆盖
pnpm check              # Astro / TypeScript 诊断
pnpm lint               # ESLint，零警告
pnpm format:check       # Prettier 检查
pnpm test               # Vitest
pnpm build              # 生产构建
pnpm verify             # 提交前完整门禁
```

`pnpm verify` 是唯一发布门禁。CI 和人工提交都必须先通过它。

## 目录

```text
src/
├─ components/tutorial/   # 章节卡片、进度、阅读模式与浏览器交互
├─ data/tutorial/         # 生成后的章节、代码、笔记和引用数据
├─ layouts/               # 站点与教程外壳
├─ lib/                   # 站点元数据、教程类型和阅读模式纯函数
├─ pages/                 # 根路径页面、章节、笔记、代码与资料路由
├─ styles/                # Tailwind 入口、图表与阅读模式样式
└─ tutorial/              # 章节 Markdown、题库、图规格和示例代码
scripts/tutorial/         # 确定性数据生成器、SVG 渲染器与内容校验
docs/                     # 架构、内容维护和部署说明
tests/                    # Vitest 套件
```

章节 URL 直接位于站点根路径。例如第 1 章是 `/01-why-tile-dsl/`，
对应代码是 `/code/01-why-tile-dsl/`。博客旧地址 `/tilelang/<path>` 会由
博客的 Vercel 配置永久重定向到新站。

## 修改内容

章节目录由 `src/tutorial/meta.json` 定义。正文、习题、答案、图规格和题库的
维护方式见 [docs/content.md](./docs/content.md)。

修改 `src/tutorial/` 后必须运行：

```bash
pnpm tutorial:build
pnpm tutorial:check
pnpm verify
```

生成结果是确定性的，应和源文件一起提交。不要在
`src/data/tutorial/generated/` 中手工编辑 JSON 或 TypeScript。

## 贡献

内容纠错、可访问性改进、构建优化和图表改进都欢迎。提交前请阅读
[CONTRIBUTING.md](./CONTRIBUTING.md) 与 [AGENTS.md](./AGENTS.md)。

安全问题和敏感信息不要通过公开 issue 报告，请按
[SECURITY.md](./SECURITY.md) 私下联系维护者。

## 许可

教程正文、习题、笔记和图表采用
[Creative Commons Attribution 4.0 International](./CONTENT-LICENSE.md) 授权。
网站源码、生成器和示例代码采用 [MIT](./LICENSE) 授权。

引用或转载教程内容时，请保留作者署名并链接到原始页面。
