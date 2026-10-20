# 07 · API 名核对（可复现）

## 目的

回答两个问题，并且**可复现**：

1. 本章示例里用到的每个 `T.xxx`（含 `T.tile.` 与 `T.simd.` 子命名空间）在锁定版本
   的语言面里到底存不存在？
2. “主干 v0.1.15 没有 `T.alloc_ub`”“`T.tile` 命名空间只在适配器仓库里”这两条
   结论，能不能被机器复现？

## 运行前提

Python 3.9+，只用标准库（`argparse`、`io`、`re`、`tokenize`）。
**不需要 GPU / NPU / PyTorch / TileLang。**

核对功能需要锁定版本的源码在给定相对位置；找不到时脚本打印 `skipped` 并以 0 退出。

## 运行命令

在 `code/07-api-audit/` 目录下：

```bash
# 主干方言：本章的 NVIDIA 示例 + 主干昇腾示例，含两个负向断言
python3 api_audit.py --root .. \
    --vendor ../../../../_vendor/tilelang-v0.1.15 --label mainline \
    --only elementwise_variants.py norm_kernels.py rope_kernels.py \
           rmsnorm_rope_quant.py ascend_mainline_kernels.py \
    --absent alloc_ub --absent-namespace tile

# 适配器方言：只有昇腾适配器骨架
python3 api_audit.py --root ../06-ascend \
    --vendor ../../../../_vendor/tilelang-ascend-v0.1.2.000 --label adapter \
    --only ascend_adapter_sketch.py
```

## 预期输出（实测数据）

在 macOS（Apple Silicon）、Python 3.11 上实际运行，两次都通过（退出码 0）：

```text
# 主干
扫描到的名字集合: 2071 个
示例文件        : 5 个，共 71 处 T.* 用法
## 一层的 `T.` 属性访问
  用到 30 个：Cast, Kernel, Parallel, Pipelined, Serial, SimdVF, SimtVF, Tensor,
  abs, alloc_fragment, alloc_reducer, alloc_shared, annotate_buffer_versions,
  ceildiv, clamp, clear, copy, finalize_reducer, float32, if_then_else, int32,
  max, prim_func, reduce_absmax, reduce_max, reduce_sum, reducer_init,
  reducer_update, rsqrt, simd
  ✓ 全部命中
## 断言：这些名字不存在
  ✓ alloc_ub 确实不在该方言的语言面里
## 断言：这些子命名空间没有被导出
  ✓ tile 没有被任何语言面 `__init__.py` 导出

# 适配器
扫描到的名字集合: 1056 个
## 一层的 `T.` 属性访问
  用到 11 个：Kernel, Scope, Tensor, alloc_ub, barrier_all, cast, copy,
  prim_func, reduce_sum, serial, tile
  ✓ 全部命中
## 两层
  ✓ tile.add ← ascend_tile.py        ✓ tile.broadcast ← ascend_tile.py
  ✓ tile.cast ← ascend_tile.py       ✓ tile.fill ← ascend_tile.py
  ✓ tile.mul ← ascend_tile.py        ✓ tile.rsqrt ← ascend_tile.py
```

## 重要边界

1. 这是**启发式静态检查**，不是类型检查。名字集合由扫描 `tilelang/language/**`
   （含 `cuda/language`、`ascend/language`）里的 `def`、模块级赋值、`__all__`
   与 `from … import …` 拼出来；靠 `getattr` 之类动态注入的名字可能漏掉。
2. 扫描用 `tokenize`，**跳过注释与字符串字面量**——否则正文与 docstring 里提到的
   名字会产生大量假阳性。
3. “命中”不代表签名对得上。参数名、dtype 约束、`clear` 语义仍以正文引用的源文件为准。
4. 负向断言（`--absent` / `--absent-namespace`）只说明该名字没有被**语言面**
   导出到 `T.` 上，不排除它在别处被定义。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过，上面的输出是
本机实际输出。**未在任何加速器上运行。**
