# 05-control-flow · 控制流

对应正文第 7 节。

## 目的

1. `if / elif / else`、三元表达式、`while`、`break` / `continue` 的写法；
2. `T.if_then_else` 与 `T.Select` 的区别（「只求值一边」 vs 「两边都求值」），
   以及为什么越界保护只能用前者；
3. 控制流对 **向量化**（守卫挡住宽访存）与 **流水**（可调度语句与可重放 Bind）
   的影响；
4. 记录一处**文档与源码不一致**：`all_of` / `any_of` 接受的是 buffer，不是条件列表。

## 文件

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `control_flow_demo.py` | TileLang 示例 | **仅静态核对** |

## 运行前提（未在本机运行）

* 运行前提：TileLang `v0.1.15` + NVIDIA GPU（`buffer_predicates` 用到 shared
  上的 uint8 buffer）。
* 静态检查命令（本机实际执行过）：

  ```bash
  python3 -m py_compile control_flow_demo.py    # 通过
  ```

* 在有卡的机器上如何验证：

  ```python
  # 1) 越界保护：比较 if_then_else 版与 Select 版的生成代码
  k = safe_load_vs_select.compile(N=1000)
  print(k.get_kernel_source())   # if_then_else 版应带条件；Select 版无保护

  # 2) 守卫 vs 无守卫：比较两者的 CUDA 源码里有没有向量化访存
  g = guard_costs_vectorization.compile(N=2047)
  f = guard_free_full_blocks.compile(N=2047)
  print(g.get_kernel_source())
  print(f.get_kernel_source())
  ```

* 性能口径：本示例不做性能断言。
* 作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过 +
  `T.if_then_else` / `T.Select` / `T.any_of` / `T.all_of` / `T.loop_break` 的
  定义都在 `_vendor/tilelang-v0.1.15/` 里找到并读过）。**未在硬件上运行**。

## 已知的文档 / 源码分歧

`docs/programming_guides/control_flow.md:33-38` 展示的是

```python
if T.all_of(i < M, j < N):
```

但 v0.1.15 里 `tilelang/language/logical.py:51` 的定义是
`def all_of(buffer: BufferLikeType)`，测试也是按缓冲区调用
（`testing/python/language/test_tilelang_language_all_of.py:56`）。
请按**源码**写：多条件的与用 `and`（会变成 `tir.And`）。
`control_flow_demo.py` 里的 `buffer_predicates` 给的是可用的缓冲区写法。

## 原文出处

* `_vendor/tilelang-v0.1.15/tilelang/language/tir/ir.py:172-186, 228`
* `_vendor/tilelang-v0.1.15/tilelang/language/logical.py:12-87`
* `_vendor/tilelang-v0.1.15/testing/python/language/test_tilelang_language_all_of.py:56,105`
* `_vendor/tilelang-v0.1.15/tilelang/language/customize.py:163-169`
* `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:139-168`
* `_vendor/tilelang-v0.1.15/docs/programming_guides/control_flow.md:15-140`
