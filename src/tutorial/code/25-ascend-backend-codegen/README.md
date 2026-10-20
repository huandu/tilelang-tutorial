# 第 25 章 代码目录

本章的示例围绕「**看生成物**」展开：能离线做的部分，全部是**只读地解析锁定版本的
源码**；必须上机的部分，单独标出来，并且**没有运行过**。

## 验证级别总表

| 目录 | 作用 | 需要硬件吗 | 验证级别 |
| --- | --- | --- | --- |
| `01-route-map/` | 从三棵树里抽出四条昇腾编译路线的入口、codegen 目标、编译器命令 | 否 | **CPU 逻辑验证**（已在本机运行） |
| `02-scope-dma-pipes/` | 抽出 `T.alloc_*` 的 scope → CCE 限定符映射、`T.copy` 的 DMA 通路表、流水线 bitmask，并做一致性检查 | 否 | **CPU 逻辑验证**（已在本机运行） |
| `03-api-presence-matrix/` | 逐条 API 在两棵树里的存在性矩阵（含 `T.tile.*`、`T.alloc_ub`、`T.dual_copy`） | 否 | **CPU 逻辑验证**（已在本机运行） |
| `04-cite-check/` | 校验 `chapter.md` 里每一个 `path:line` 引用都能落到真实文件与真实行号 | 否 | **CPU 逻辑验证**（已在本机运行） |
| `05-emit-ascend-c/` | 一个真实 kernel，在有昇腾环境的机器上打印生成的 Ascend C 并逐段核对 | **是**（Ascend NPU + CANN + bisheng） | **仅静态核对**（本机无卡，未运行） |

## 定位 `_vendor/` 与 corpus

所有脚本按「脚本所在目录向上数 4 层」定位书根
（`…/tilelang-book/chapters/25-ascend-backend-codegen/code/<example>/` → 上 4 层 =
`tilelang-book/`），再拼：

* `_vendor/tilelang-v0.1.15/`（M 树，主干）
* `_vendor/tilelang-ascend-v0.1.2.000/`（A 树，适配器）
* `../tilelang-corpus/`（资料库，只读）

只要章节目录还在 book 下，脚本就能直接用相对路径找到这些树，无需传参。
如果换了位置，用 `--book-root` 显式指定。

## 运行顺序建议

```bash
python3 01-route-map/route_map.py
python3 02-scope-dma-pipes/scope_dma_pipes.py
python3 03-api-presence-matrix/api_matrix.py
python3 04-cite-check/check_citations.py

# 只有在有昇腾环境的机器上：
python3 05-emit-ascend-c/print_ascend_c.py --target ascend     # M 树
```

## 纪律

* 不修改 `_vendor/`，不修改 corpus。这些脚本只读。
* 任何输出都不是「跑过 kernel」的证据；`NN-` 编号脚本只证明「源码里确实这么写」。
