# 示例 03：用注解接管布局（需要 NVIDIA GPU 才能运行）

## 目的

对应正文第 3.3 节「用户如何干预」。在一个普通 tiled GEMM 上，显式做两件事：

```python
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
T.use_swizzle(panel_size=10, enable=True)
```

并同时打开编译器的 layout visualization（`pass_configs` 里的两个键），这样能把
「你钉住的」和「编译器推导出来的」放在一起比。

累计子 `C_local` 故意**不**钉：它是 `T.Fragment`（每个 (thread, register slot) 一个
元素），只有在必须跟某个特定消费者对齐时才值得手写。

## 运行前提

* NVIDIA GPU（能编译 CUDA）+ CUDA toolkit。
* PyTorch（CUDA 版）。
* TileLang `v0.1.15`，且 `import tilelang` 可用。
* Matplotlib 可选（只有想把 layout 图导出成 SVG/PNG 时才需要）。

## 运行命令

```bash
python3 annotate_swizzle_gemm.py
```

## 期望输出

1. **正确性判据**：脚本用 `torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)`
   校验结果，通过则打印 `correctness check passed`。
2. **可观测的布局证据**（不需要读性能计数器）：
   * 编译期会打印形如

     ```text
     C_local inferred layout:
       Shape: [128, 128] -> [..]
       Thread: <线程表达式>
       Index:  [<寄存器槽表达式>]
       Replicate:  1
     ```

     的文本（`TL_LAYOUT_VISUALIZATION_ENABLE=True` 时始终打印）；
   * `TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg"` 会在 `./tmp/` 下给每个二维 inferred
     fragment 落一张 `<buffer>_layout.svg`；
   * `kernel.get_kernel_source()` 返回生成的 CUDA 源码，可搜索 `__shared__`、
     XOR 形式的地址表达式来确认 swizzle 生效。
3. **性能口径**：本示例**不报告**任何 FLOPS，因为写作机没有 GPU。要测性能，另配
   计时脚本（`tilelang.profiler`）或外部 benchmark，并注明 GPU 型号、CUDA、
   TileLang 版本与重复次数。

## 验证级别

**仅静态核对**：

* `python3 -m py_compile annotate_swizzle_gemm.py` 通过；
* 所用 API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对存在性与签名，见下表；
* **未运行**（本机无 GPU、无 TileLang）。因此 `README` 里的「期望输出」是**待验证的
  预期**，不是已观测结果。

## API 对照表（已在 v0.1.15 中核对）

| 调用 | 定义位置 | 说明 |
| --- | --- | --- |
| `T.annotate_layout({buffer: layout})` | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:45-66` | 写入 `layout_map` block attr；对 Fragment buffer 要求 layout 是 `Fragment` |
| `T.use_swizzle(panel_size, order, enable)` | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:30-42` | 线程块栅格化顺序（`rasterization2DRow/Column/MLX`），不是共享内存 swizzle |
| `make_mma_swizzle_layout(shared_buf, is_smooth=False)` | `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/layout/mma_layout.py:336`（经 `tilelang/cuda/intrinsics/__init__.py:14` 导出） | MMA 用的 shared 布局 |
| `PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE` / `..._FORMATS` | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:252,255` | 打开/配置 inferred layout 可视化 |
| `@tilelang.jit(pass_configs=...)`、`T.const`、`T.empty` | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py:1-26` | 本示例的骨架直接取自官方 GEMM 示例 |
| `kernel.get_kernel_source()`、`kernel.compile(...)` | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532`、`tilelang/jit/__init__.py:507` | 打印生成源码 |

## 与本章「常见坑」的对照

把上面两行注解删掉再编译一次，对比 `C_local inferred layout` 与生成的 CUDA，就能亲眼
看到「编译器自己也会选一个 swizzle」。这也是正文第 5 节反复强调的一点：**注解是覆盖，
不是必需**；大多数情况下先让编译器推，出问题再覆盖。
