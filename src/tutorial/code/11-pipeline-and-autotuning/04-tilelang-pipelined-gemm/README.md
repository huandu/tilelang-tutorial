# 04 · T.Pipelined 的三种写法（TileLang）

## 目的

同一个 GEMM 主体，三种调度写法，用来对照正文第 3 节与第 1 节：

| 函数 | reduction 循环 | 说明 |
| --- | --- | --- |
| `matmul_auto` | `T.Pipelined(n, num_stages=3)` | 只给深度，`PipelinePlanning` 自己推 stage/order |
| `matmul_manual` | `T.Pipelined(n, stage=[0,0,1], order=[0,1,2])` | 手工指定；故意不给 `num_stages`（深度由 `max(stage)+1` 推出） |
| `matmul_plain` | `T.serial(n)` | 完全不流水，作为“不做流水会怎样”的对照组 |

原型：`_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_autotune.py`。

## 运行前提

* NVIDIA GPU（sm_80 及以上）+ 可用 CUDA toolkit；
* PyTorch（CUDA 版）与 TileLang `v0.1.15`；
* 不需要昇腾 NPU。

## 运行命令

```bash
python3 pipelined_gemm.py                        # 默认 4096³，128×128×32，num_stages=3
python3 pipelined_gemm.py --stages 1
python3 pipelined_gemm.py --block-m 256 --block-n 256 --block-k 64 --stages 4
```

脚本先打印预算对比（`stages=1` 与 `stages=<目标>` 各自需要的 shared 字节数与是否超预算），
再编译 `matmul_auto`，并把生成源码里与流水相关的行（循环头、`cp.async`、barrier、`mma`、
`__syncthreads`）过滤出来打印。

`--block-m 256 --block-n 256 --block-k 64 --stages 4` 是**故意**给一个超预算的配置：
期望看到 `[OVER]`，并且在编译或启动阶段报 shared memory 相关的错误。把 `--stages 3`
换回去应当恢复正常——这条对照就是正文第 4.5 节与第 5.1 节说的“stage 数不是越大越好”。

## 期望输出（正确性 + 观察项）

1. **正确性**：在脚本里补一次与 `torch.matmul` 的对照，判据是 `rtol=1e-2, atol=1e-2`
   （fp16 输入、fp32 累加）。本目录没有把对照写进脚本，是因为它需要 GPU 才能执行，
   读者请自行加上并记录结果。
2. **生成码观察项（下面是预期，需要你在自己机器上核对，作者没有跑过）**：
   * `matmul_plain` 的循环体里，`__syncthreads()` 应当出现在“拷贝完”和“gemm 完”
     两处，搬运与计算之间没有重叠；
   * `matmul_auto` / `matmul_manual` 的稳态循环体里，异步拷贝与其等待应当分处不同位置，
     shared buffer 的索引应当带一个随迭代变化的偏移（多缓冲版本维）；
   * 多缓冲版本里，同一份 buffer 的声明尺寸应当变成 `num_stages × ...`。

   如果观察结果与上面不符，那本身就是一个值得记下来的发现——请以你机器上的实际生成码
   为准，并注明 TileLang 版本、GPU 型号与编译参数。
3. **性能口径**：如果要测，用 `kernel.get_profiler().do_bench()`，并同时记录
   tile 尺寸、`num_stages`、`threads`、GPU 型号与 CUDA 版本；本目录**不提供任何
   性能数字**，因为没有卡可测。

## 作者实际做到的验证级别

**仅静态核对**：

* `python3 -m py_compile pipelined_gemm.py` 通过；
* 逐个核对了本文件用到的 API 在 `_vendor/tilelang-v0.1.15/` 中存在且签名一致：
  `tilelang.jit`（`tilelang/jit/__init__.py:575-588`）、`T.Pipelined`
  （`tilelang/language/loop.py:112-186`）、`T.serial`（同文件 `:189`）、
  `T.Kernel`（`tilelang/language/kernel.py:431`）、`T.alloc_shared`
  （`tilelang/language/allocate.py:49`）、`T.alloc_fragment`（同文件 `:81`）、
  `T.copy`（`tilelang/language/copy_op.py:59`）、`T.gemm`
  （`tilelang/language/gemm_op.py:160`）、`T.clear`
  （`tilelang/language/fill_op.py:48`）、`T.ceildiv`
  （`tilelang/language/tir/ir.py:216`，经 `tilelang.language` 再导出）。

**未在硬件上运行。** 这里没有任何实测数字，也没有声称跑过。
