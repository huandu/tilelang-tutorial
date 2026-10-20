# 04 · NV 侧 `T.copy` / `T.gemm` 骨架

## 目的

本章正文反复出现两个原语，这里给出它们在**一个完整 kernel** 里的样子，方便对照
§2.2 的映射表与 §4.1/§4.3 的机制说明：

* `nv_gemm_copy_sketch.py` —— 一个 `T.copy` + `T.gemm` + `T.copy` 的经典
  shared-memory GEMM，逐处标注每个原语在正文里对应哪一节。
* `gemm_primitive_variants.py` —— 同一份 GEMM 在 `transpose_A/B`、`policy`、
  `clear_accum` 上的四种组合；用于第 E4/E5 题的对照阅读。

**这两个文件都只做静态核对**：没有 GPU，也没有跑过。

## 运行前提

* NVIDIA GPU（`gemm_primitive_variants.py` 里的 wgmma 分支需要 Hopper 及以上才能
  真正走到，但它只是 `T.gemm`，具体指令由编译器选）
* CUDA toolkit + PyTorch(CUDA)
* TileLang `v0.1.15`（`pip install tilelang==0.1.15` 或从锁定 commit 构建）

## 运行命令

```bash
cd 04-copy-gemm-sketches
python3 -m py_compile nv_gemm_copy_sketch.py gemm_primitive_variants.py

# 有卡机器上：
python3 nv_gemm_copy_sketch.py
python3 gemm_primitive_variants.py
```

## 期望输出（正确性判据，不是实测结果）

* `nv_gemm_copy_sketch.py`：`torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)`
  必须通过（fp16 输入 / fp32 累加）。
* `gemm_primitive_variants.py`：四种组合都必须与
  `torch.matmul` 的对应写法一致；`clear_accum` 变体用 `k == 0` 清零，
  结果必须与「先 `T.clear` 再累加」等价。
* 性能口径（若要做）：用 `tilelang.profiler.do_bench` 测主循环时间，并**同时**
  记录 GPU 型号、驱动、CUDA 版本与 TileLang 版本；不要把单机数字当成基线。

## 作者验证级别

**仅静态核对**：两个文件都通过 `python3 -m py_compile`；其中每个 TileLang API 都在
`_vendor/tilelang-v0.1.15/` 里逐个核对过参数名与默认值。
**未在硬件上运行。**

## 已知的静态核对结论

* `T.gemm` 在 v0.1.15 的签名是
  `(A, B, C, transpose_A=False, transpose_B=False, policy=GemmWarpPolicy.Square,
  clear_accum=False, annotations=None)`，**没有** `accum_dtype`
  （`tilelang/language/gemm_op.py:160-169`）。
* `T.GemmWarpPolicy` 从 `tilelang/language/common.py:78,197` 导出，因此
  `T.GemmWarpPolicy.Square` 可用。
* `T.fill`/`T.clear` 从 `tilelang/language/fill_op.py` 导出
  （`tilelang/language/common.py:86`）。
