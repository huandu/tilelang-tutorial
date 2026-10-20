# 教程源内容

这个目录是网站内容的唯一事实来源。39 章正文、习题、答案、图规格、题库、
示例代码和编者笔记都在这里，不依赖仓库外的私有工作区。

| 文件或目录 | 作用 |
| --- | --- |
| `meta.json` | 章节顺序与五部分分组 |
| `content/` | 章节正文、习题、答案与引用清单 |
| `source-figures/` | 图表原始说明 |
| `figures/` | 可渲染的 SVG JSON 规格 |
| `quiz/` | 章节自测题 |
| `code/` | 可浏览的章节示例代码 |
| `notes/` | 三篇编者笔记 |
| `VERSIONS.md` | 锁定版本与验证边界 |
| `source-urls.json` | 资料索引的上游 URL |

修改任何输入后，在仓库根目录运行：

```bash
pnpm tutorial:build
pnpm tutorial:check
pnpm verify
```

完整维护流程见仓库根目录的 `docs/content.md`。生成数据不要手工编辑。
