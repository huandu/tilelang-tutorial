# 04 · 手写 warp-specialized GEMM

## 目的

演示 `T.ws(role)` + `T.tma_copy(..., barrier=...)` + `T.barrier_wait/arrive`
三者如何组成一个 producer/consumer 流水线。原型是官方示例
`_vendor/tilelang-v0.1.15/examples/warp_specialize/example_warp_specialize_gemm_copy_0_gemm_1.py`，
本章的版本只做了一点改动（加上 `main()` 的检查与说明注释）。

## 运行前提（需要真实硬件）

1. NVIDIA Hopper 或更新的 GPU（`sm_90+`，因为 `T.tma_copy` / mbarrier 需要 TMA）；
2. CUDA toolkit 12.x，且 `nvcc` 可在 PATH 中找到；
3. Python 3.9+，`torch`（CUDA 版）、`tilelang==0.1.15`；
4. `tilelang` 需要能找到 CUTLASS headers（随 wheel 一起安装）。

## 运行命令

```bash
python3 ws_gemm.py
```

## 期望输出（正确性判据）

* `torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)` 通过，打印
  `Kernel output matches PyTorch reference.`；
* 之后的 `do_bench` 只打印延迟，**本脚本不声称任何具体性能数字**——性能取决于卡型、
  tile 配置和时钟，必须在目标机器上实测。

## 作者实际做到的验证级别

**仅静态核对**：

* `python3 -m py_compile ws_gemm.py` 通过；
* 用到的每个 API 都在 `_vendor/tilelang-v0.1.15/` 中核对过存在性与签名
  （`T.ws` → `tilelang/cuda/language/warpgroup.py:19-59`；
  `T.alloc_barrier` → `tilelang/language/allocate.py:194-222`；
  `T.tma_copy` → `tilelang/language/copy_op.py:232-300`；
  `T.barrier_wait` / `T.barrier_arrive` → `tilelang/language/builtin.py:885-908`）；
* **未在 GPU 上运行过。** 本机没有 NVIDIA GPU。任何「跑得对/跑得快」的结论都必须由读者
  在自己的机器上得出。

## 阅读提示

* `T.ws(0)` 展开成 `tx < 128`，`T.ws(1)` 展开成 `128 <= tx < 256`
  （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/warpgroup.py:19-59`）。
* 两个 mbarrier 的 `arrive_count=128` 对应一个 128 线程的 warp group。
* `num_stages=0`：多缓冲不由 `T.Pipelined` 插入，改由 mbarrier 的奇偶翻转承担。
* consumer 分支里的 `T.gemm(A_shared, B_shared, C_local)` 在 sm_90 上会走 WGMMA
  （`M=128 >= 64`、256 线程 = 8 warps，8 % 4 == 0）——见正文 4.3 节。
