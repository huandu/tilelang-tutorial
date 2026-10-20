# quickstart-gemm：用官方示例走一遍完整编译流程

## 目的

把正文图 2 的 13 步在真实运行时上打点。kernel 主体取自
`_vendor/tilelang-v0.1.15/examples/quickstart.py:8-48`，本章加了三样东西：

1. `debug_root_path=...`，让每一阶段产出的源码落到磁盘；
2. 在 4 个位置打印进度，对应图 2 的 `[1-3] / [4] / [9] / [12-13]`；
3. 用 `torch.testing.assert_close` 做一次正确性判据。

## 运行前提

| 项 | 要求 |
| --- | --- |
| 硬件 | NVIDIA GPU（或把 target 换成其它受支持的后端） |
| 驱动 | 与所用 CUDA toolkit 匹配的驱动 |
| Python 包 | `torch`（CUDA 版）、`tilelang`（版本 `0.1.15`，`python -c "import tilelang; print(tilelang.__version__)"` 可核对） |
| 磁盘 | `--out` 指定的目录可写 |

## 运行命令

```bash
python3 -c "import tilelang, torch; print(tilelang.__version__, torch.cuda.is_available())"
python3 quickstart_gemm.py --out /tmp/tl-debug
python3 quickstart_gemm.py --out /tmp/tl-debug --block-m 64 --block-n 64 --block-k 32
```

## 期望输出（在有卡的机器上）

```text
[1-3] elaborating PrimFunc for M=N=K=1024 ...
      IRModule functions: [...]
[4] compiling (cache lookup happens inside tilelang.cache.cached) ...
[9] generated device source (first 20 lines):
      ... __global__ void ...
[9] debug dumps written under /tmp/tl-debug
[12-13] calling the compiled kernel through the adapter
      result matches torch.relu(a @ b)
```

并应在 `--out` 目录里出现：

```text
tilelang_jit_kernel_<name>.c    # 生成的设备源码
tilelang_jit_program_<name>.py  # 回写的 TIR（prim_func.script()）
```

**正确性判据**：脚本正常退出（退出码 0），且最后一行打印
`result matches torch.relu(a @ b)`。**性能口径**：本脚本不测性能，也**不打印任何
延迟/吞吐数字**——插桩打印会干扰计时，性能测量请用
`kernel.get_profiler().do_bench()`（见 `examples/quickstart.py:82-87`）。

## 无卡时会怎样（作者的实际情况）

本机是 macOS Apple Silicon，**没有 NVIDIA GPU，也没有昇腾 NPU**，Python 环境里
没有安装 torch/tilelang。因此本示例的验证级别是：

* `python3 -m py_compile quickstart_gemm.py` —— **通过**；
* 正文与脚本里出现的每个 TileLang API（`tilelang.jit` 的 `out_idx`/`debug_root_path`
  参数、`T.prim_func`、`T.Kernel`、`T.const`、`T.alloc_shared`、`T.alloc_fragment`、
  `T.clear`、`T.Pipelined`、`T.copy`、`T.gemm`、`T.Parallel`、`T.max`、
  `get_tir`、`compile`、`get_kernel_source`）—— 逐个在 `_vendor/tilelang-v0.1.15/`
  里核对过存在性与签名；
* **没有运行过**。脚本在没有 GPU 的环境里会打印原因并以退出码 3 结束，不会伪造结果。

## 昇腾版本怎么做

把 `target` 换成 `"ascend"`（`tilelang/ascend/target.py:57-59` 注册了
`ascend` / `asc` 两个名字），并把 `torch.randn(..., device="cuda")` 换成
`torch_npu` 的空张量。需要 CANN 与 `torch_npu`（见
`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:12-25`）。本章不作运行声明。
