# 05 · Blackwell TCGEN05 GEMM（累加器在 TMEM）

## 目的

演示 Blackwell（sm_100）上第 5 代 Tensor Core 的用法差异：**累加器不在寄存器里，
而在 Tensor Memory（TMEM）里**，且 MMA 是异步的，需要 mbarrier 完成通知。

原型是官方示例 `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/gemm_tcgen5mma.py`。

## 运行前提（需要真实硬件）

1. NVIDIA Blackwell GPU（`sm_100a`，例如 B200）；
2. CUDA toolkit 12.8 或更新（`sm_100a` 需要 12.8+）；
3. Python 3.9+，`torch`（CUDA 版）、`tilelang==0.1.15`。

## 运行命令

```bash
python3 tcgen05_gemm.py
```

## 期望输出（正确性判据）

* `torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)` 通过；
* 生成代码里应能看到 `T.alloc_tmem` 对应的 TMEM 分配，以及 `tcgen05` 系列的
  指令与 `mbarrier_wait_parity`；
* 脚本会打印与 PyTorch 的对比延迟，**本脚本不声称任何具体性能数字**。

## 作者实际做到的验证级别

**仅静态核对**：

* `python3 -m py_compile tcgen05_gemm.py` 通过；
* 用到的 API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对：
  `T.alloc_tmem` → `tilelang/language/allocate.py:254-300`；
  `T.alloc_barrier` → `tilelang/language/allocate.py:194-222`；
  `T.tcgen05_gemm` → `tilelang/cuda/language/gemm_op.py` 的 `__all__`；
  `T.mbarrier_wait_parity` → `tilelang/language/builtin.py:457-496`；
* **未在 GPU 上运行过。**

## 一个 API 层面的注意点

官方示例原型用的是 `T.tcgen05_gemm(...)`，但在 v0.1.15 里这个名字**不在**
`tilelang.language` facade 的导出名单上：`tilelang/cuda/language/gemm_op.py:20-30`
的 `__all__` 列了它，可 `tilelang/cuda/language/__init__.py:88` 只把
`gemm, gemm_blockscaled, gemm_sp` 引进包命名空间，而
`tilelang/language/__init__.py:11` 用的是受 `__all__` 限制的 star-import。
详见正文 4.3.6 节。

因此本示例改用**已导出**的同步形式 `T.gemm(A_shared, B_shared, C_tmem, mbar=mbar)`，
由编译器隐式插入 `mbarrier_wait_parity`。两种写法的语义相同。

## 为什么这几个写法是这样

* `T.alloc_tmem([block_M, block_N], accum_dtype)` 的列数必须是 2 的幂且 ≥ 32
  （`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:254-300`）。
* `mbar = T.alloc_barrier(1)`：TCGEN05 完成通知的 arrive 数是 1（单线程发射）。
* `T.gemm(..., mbar=mbar)` 会在发射后**隐式**插入一次 `mbarrier_wait_parity`，
  所以本示例直接写这个同步形式，不需要再手写等待。
* 结尾必须 `T.copy(C_tmem, C_local)` 把结果从 TMEM 搬回寄存器，再做任何后处理
  ——只有 TCGEN05 与 TMEM load/store 能访问 TMEM。

## 进阶（未在本例中实现）

* `T.tcgen05_gemm(A_s, B_s, C_tmem, mbar=mbar)` + `T.mbarrier_wait_parity`：显式的
  异步写法，等待由自己管（本例用的是等价的同步形式 `T.gemm(..., mbar=mbar)`）；
* `T.gemm_blockscaled(..., mbar=..., k_start=..., sf_a_granularity_k=128,
  sf_b_granularity_k=128)`：MXFP8 等 block-scaled 格式（见
  `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py`）；
* `use_2cta=True` + `cluster_dims=(2,1,1)`：2-CTA TCGEN05（见
  `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/gemm_tcgen5mma_ws_persistent.py`）。
