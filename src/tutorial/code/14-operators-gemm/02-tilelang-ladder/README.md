# 示例 02：GEMM 五级阶梯（①–⑤）

## 目的

把第 14 章第 2 节的阶梯写成可对照运行的代码。每一级只改一件事：

| 文件 / `--rung` | 级 | 相对上一级的变化 |
| --- | --- | --- |
| `gemm_ladder.py --rung 0` | ① | 朴素分块，直接读 global |
| `gemm_ladder.py --rung 1` | ② | 加 shared memory + `T.gemm`（串行 K 循环） |
| `gemm_ladder.py --rung 2` | ③ | `T.serial` → `T.Pipelined(num_stages=N)` |
| `gemm_ladder.py --rung 3` | ④ | 加 `T.annotate_layout` + `T.use_swizzle` |
| `gemm_intrinsic.py` | ⑤ | 手动 `ldmatrix_a/b` + `mma` + `stmatrix` |

## 运行前提

级别 ①–④：

* NVIDIA GPU（CUDA toolkit）、PyTorch with CUDA、TileLang `v0.1.15`。
* Python 3.9+。

级别 ⑤（`gemm_intrinsic.py`）：

* 同上；另外它用到的 `TensorCoreIntrinEmitter` 与
  `tilelang.cuda.intrinsics` 都是 CUDA 专用模块。

## 运行命令

```bash
# 正确性（每一级都可以单独跑）
python3 gemm_ladder.py --rung 0 --m 1024 --n 1024 --k 1024
python3 gemm_ladder.py --rung 1 --m 4096 --n 4096 --k 4096
python3 gemm_ladder.py --rung 2 --m 4096 --n 4096 --k 4096 --num-stages 3
python3 gemm_ladder.py --rung 3 --m 4096 --n 4096 --k 4096 --num-stages 3

# 打印生成的 CUDA 源码（对照第 12 章）
python3 gemm_ladder.py --rung 3 --show-source

# 计时（会打印 TFLOP/s，并提醒你补全口径）
python3 gemm_ladder.py --rung 2 --bench

# 第 ⑤ 级
python3 gemm_intrinsic.py
```

## 期望输出

* **正确性判据**：与 PyTorch 的 `a @ b`（或第 ⑤ 级的 `a @ b.T`）比较，
  `rtol=1e-2, atol=1e-2` 通过。
* **性能口径**：`--bench` 用 `do_bench(warmup=25, rep=100, return_mode="mean")`
  （v0.1.15 的默认写法），报 `2MNK/latency` 得到的 TFLOP/s。使用时必须
  同时写明 backend、是否 flush L2 —— 见 `../06-bench-methodology/`。
* **本机没有 GPU，以上输出均未产生。** 作者实际做到的是静态核对：
  `python3 -m py_compile` 通过；`T.gemm`、`T.Pipelined`、`T.annotate_layout`、
  `T.use_swizzle`、`T.alloc_shared`、`T.alloc_fragment`、`T.alloc_local`、
  `TensorCoreIntrinEmitter` 逐个在 `_vendor/tilelang-v0.1.15/` 中比对过签名。

## 已知局限

* 级别 ①（朴素版）只是**正确性基准**，不要拿它的延迟跟后面的级别比规模——
  它的访存量是 $O(MNK)$ 量级。
* 级别 ⑤ 的 `block_row_warps` / `warp_row_tiles` / `chunk` 等都是写死的
  “debug config”，和官方示例一致；它演示的是**结构**，不是最优配置。
* 官方对 TCGEN05（Blackwell）路线另有写法（`T.alloc_tmem` +
  `T.tcgen05_gemm` + 显式 `mbarrier`），见
  `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/`；本示例不做展开。
