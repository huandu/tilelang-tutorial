# 示例 3：同一段数据流，四种调度（调度是注解，不是重写）

## 目的

第 03 章 2.3 节说：调度空间的四个维度在 v0.1.15 里**都有入口，也都有缺省**。这个示例把
这句话变成四个可以逐行 diff 的 kernel：

| 变体 | 只改了什么 | 属于哪个维度 | 改动的源码行 |
| --- | --- | --- | --- |
| `v0` 基线 | —— | —— | 全部缺省 |
| `v1` 线程划分 | `T.gemm(..., policy=T.GemmWarpPolicy.FullRow)` | thread binding | 1 行 |
| `v2` 显式布局 | `T.annotate_layout({...make_swizzled_layout(B_s)})` | layout | 3 行 |
| `v3` 手工流水 | `T.Pipelined(..., stage=[0,0,1], order=[0,1,2])` | pipeline | 1 行 |

四个 kernel 的**数据流完全一样**：搬入 A、B 的分块 → 乘加 → ReLU → 搬出。区别只在调度注解。
这正是论文摘要那句「encapsulated them as a set of customization annotations and primitives」
的直接体现——调度不是重写代码，是在代码上加注解。

## 与官方示例的关系

* `v0` / `v1` 的写法来自 `examples/flash_attention/example_mha_fwd_bshd.py:24-90`
  （`T.gemm(..., policy=T.GemmWarpPolicy.FullRow)`）；
* `v2` 的写法来自 `examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py:344-348`；
* `v3` 的写法来自 `_vendor/tilelang-v0.1.15/docs/programming_guides/software_pipeline.md:44-60`。

## 运行前提

* **NVIDIA GPU**（sm70 及以上，Tensor Core 可用）+ CUDA 版 PyTorch + TileLang v0.1.15。
* 不需要昇腾 NPU。

没有 GPU 时可做：

```bash
python3 -m py_compile schedule_as_annotation.py
python3 schedule_as_annotation.py --static-only
```

## 运行命令

```bash
python3 schedule_as_annotation.py               # 编译并对比四个变体的输出
python3 schedule_as_annotation.py --static-only  # 无 GPU 时
```

## 期望输出

**有 GPU 时**（读者自行验证）：

1. 四个变体的输出都与 `torch.relu(A @ B)` 在 `rtol=atol=1e-2` 下一致；
2. **本示例不打印性能数字**。若你想比较四种调度的快慢，请自己计时并把条件写清楚
   （同一张卡、同一 shape、同一 dtype、同一 warmup/重复次数口径）。作者没有做过这项测量，
   因此这里不给出任何结论。

**无 GPU 时**：打印四个变体的调度维度对照表与 API 核对清单，退出码 0。

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile` 通过；四个变体用到的 API（`T.Pipelined` 的
`stage` / `order` 参数、`T.annotate_layout`、`T.GemmWarpPolicy`、
`tilelang.layout.make_swizzled_layout`）逐个在 `_vendor/tilelang-v0.1.15/` 中核对过签名。
**没有**在硬件上运行，**没有**做过任何性能对比。

## 陷阱提醒

* `v3` 故意**不**传 `num_stages`。官方 docstring 明确建议：「Avoid setting `num_stages`
  together with manual `order`/`stage` unless you intentionally need an explicit depth
  override」（`_vendor/tilelang-v0.1.15/tilelang/language/loop.py:168-170`）。
* `v2` 的 swizzle **是否真正生效**取决于 buffer 最后一维的字节数与对齐。官方示例里有一个
  判据 `shape[-1] * DataType(dtype).bits == 512`（即最后一维 64 字节）才会 swizzle，
  否则退化成恒等映射（`examples/gemm/example_gemm_intrinsics.py:15-17`）。本示例只演示
  「注解写在哪、怎么写」，**没有**在硬件上核实生成代码里的访存模式；读者若在意效果，
  应当用 `T.use_swizzle` / 布局可视化（`docs/tools/layout_visualization.md`）自己确认。
* 四个变体的**正确性应当一致**，只是调度不同；如果某个变体结果不对，那是调度注解写错了，
  不是数据流写错了——这正是把两者分开之后更容易定位问题的地方。
