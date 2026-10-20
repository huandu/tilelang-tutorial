# 06 · 昇腾侧的同款实现

## 目的

同一批算子（向量加、RMSNorm）在昇腾上的两个写法，用来对照第 6、9 章讲的
“GPU 与 NPU 的抽象差异”在代码上具体长什么样：

| 文件 | 语言面 | UB 的写法 | 线程级并行 | 向量计算 |
| --- | --- | --- | --- | --- |
| `ascend_mainline_kernels.py` | 主干 v0.1.15 的 `tilelang.ascend.language` | `T.alloc_shared` | `T.SimtVF(threads=…)` | `T.Parallel` / `T.simd.` |
| `ascend_adapter_sketch.py` | 适配器 v0.1.2.000 的 `tilelang.language` | `T.alloc_ub` | `T.Scope("V")` + `T.barrier_all()` | `T.tile.` |

**两个文件的 API 不能混用。** 本章用 `07-api-audit/api_audit.py` 把这条结论做成了
可复现的检查（见下）。

## 运行前提

* `ascend_mainline_kernels.py`：昇腾 950 NPU + CANN + torch_npu，
  并且 TileLang 是用 `USE_ASCEND=ON USE_CUDA=OFF` 从源码构建的 v0.1.15
  （安装步骤见 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:10-40`）。
* `ascend_adapter_sketch.py`：昇腾 NPU + CANN + torch_npu + `tilelang-ascend`
  适配器 **v0.1.2.000-release**。

两者都不需要 NVIDIA GPU。

## 运行命令

```bash
cd 06-ascend
python3 -m py_compile ascend_mainline_kernels.py ascend_adapter_sketch.py   # 本机可做

# 真机（主干）
python3 ascend_mainline_kernels.py
# 真机（适配器）
python3 ascend_adapter_sketch.py
```

复现“两个方言的 API 不通用”这条结论：

```bash
cd ../07-api-audit

# 主干：示例里的名字全部命中，并断言 alloc_ub 与 tile 命名空间不存在
python3 api_audit.py --root ../06-ascend \
    --vendor ../../../../_vendor/tilelang-v0.1.15 --label mainline \
    --only ascend_mainline_kernels.py \
    --absent alloc_ub --absent-namespace tile

# 适配器：T.alloc_ub 与 T.tile. 全部命中
python3 api_audit.py --root ../06-ascend \
    --vendor ../../../../_vendor/tilelang-ascend-v0.1.2.000 --label adapter \
    --only ascend_adapter_sketch.py
```

## 预期输出

**正确性判据**：

* `vector_add` 与 `a + b` 在 `rtol=atol=1e-2` 内一致（官方示例用的就是这个口径）；
* `rms_norm_ascend` 与
  `x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)`
  在 1e-2 内一致；同时 `RSTD` 输出与 `torch.rsqrt(x.square().mean(-1) + eps)` 对齐。

**性能口径**：两个 kernel 都是纯访存型，用
`do_bench(..., backend="msprof")` 量延迟，再算有效带宽
（向量加是 `3 * N * 4` 字节，RMSNorm 是 `(2 * N + d + batch) * sizeof`）。
**本机没有 NPU，没有任何 msprof 数字。**

## 昇腾实现的四个要点（正文 §3.4 的代码版）

1. **UB 就是 `T.alloc_shared`**：主干里没有 `T.alloc_ub`
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-23` 明确写了
   "Allocate a UB buffer"）。适配器把它改名为 `T.alloc_ub`。
2. **`T.Kernel` 没有 `threads=`**：主干是“一维 AI Core 网格”
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-159`），
   线程域要另外用 `T.SimtVF` 开。适配器是
   `T.Kernel(n, is_npu=True) as (cid, vid)`，多一个向量子核维度。
3. **归约的写法不同**：SimtVF 里推荐 `T.alloc_reducer` / `T.reducer_update` /
   `T.finalize_reducer`（epoch 语义，经得起线程域物理复制）；
   适配器仍然可以直接用 `T.tile.` 的向量指令配合 `T.reduce_sum`。
4. **tile 由 UB 容量反推**：官方 RMSNorm 示例先算 `ub_budget`，据此挑 `rows`，
   再决定是复用输入 buffer 还是另开输出 buffer
   （`_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:17-33`）。
   这和 CUDA 上调 shared memory 尺寸是同一种思路，只是容量小得多。

## 作者验证级别

**仅静态核对**：`python3 -m py_compile` 通过；两个文件各自用到了正确的语言面
（`07-api-audit` 的两轮检查都通过，包含“主干没有 `T.alloc_ub`、没有 `T.tile`”
的负向断言）。**未在昇腾 NPU 上编译或运行过**，不含任何实测数字。
