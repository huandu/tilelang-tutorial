# 图 5：两种并行模型的映射

**用途**：正文 §4.5、§7.2。

事实来源：`.../TileKernels/tile_kernels/moe/topk_gate_cuda.py:22,34-38` 与
`.../moe/topk_gate_asc.py:11,30,46,75-81,106-151`；
`_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-172`
（昇腾 `T.Kernel` 没有 `threads=`）；`.../tilelang/language/kernel.py:431`
（通用 `T.Kernel`）。

```text
  C U D A    （SIMT）
  ─────────────────────────────────────────────────────────────
   grid: num_tokens blocks                   ← T.Kernel(num_tokens, threads=32)
   ┌─────────────┐  ┌─────────────┐
   │ block 0     │  │ block 1     │   ...     （每 block = 一个 token）
   │ ┌──┬──┬──┐  │  └─────────────┘
   │ │T0│T1│..│32│
   │ └──┴──┴──┘  │   ← 32 个线程，每个线程 1 个标量
   └─────────────┘
   跨线程通信：__shfl / shared / T.alloc_reducer（编译器实现）
   代码里看不到「宽度」：T.Parallel(num_aligned_experts)


  A S C E N D   S I M D
  ─────────────────────────────────────────────────────────────
   grid: num_vec_cores cores                 ← T.Kernel(num_vec_cores)
   ┌──────────────────────┐  ┌──────────────────────┐
   │ Vector core 0        │  │ Vector core 1        │  ...
   │  vreg0: lane 0..63   │  └──────────────────────┘
   │  vreg1: lane 0..63   │   ← 一条指令吃 64 个 lane
   │  ... 最多 6 个 vreg  │
   └──────────────────────┘
   跨 lane 归约：S.vcmax / S.vcmin / S.vcadd（显式指令）
   代码里看得见「宽度」：vector_length = 64、T.int32x64、PAT_ALL
   专家数超过一组能覆盖的范围 → 自己分组（快路径 / 慢路径）


  同一件事在两台机器上的形状
  ─────────────────────────────────────────────────────────────
   CUDA    : token → block → warp(32) → 每个线程几个专家
   Ascend  : token → Vector core → 6 个 vreg × 64 lane → 分组遍历

   一句话：CUDA 的并行是「32 个独立的小计算」，
           Ascend 的并行是「一条 64 宽度的向量指令」。
```

## 读图提示

* 昇腾的「核」这一层**没有线程域**：`T.Kernel(N)` 只给一个主索引，
  想用线程就进 `T.SimtVF(threads=...)`（`rand/randn_asc.py:34` 是例子）。
* 「专家数超过一组」这件事只有昇腾版需要处理（`num_groups > 1` 的慢路径），
  因为每条 SIMD 指令的宽度是硬件固定的。
* 这解释了为什么昇腾侧会有 `vector_length`、`max_vregs_per_group`
  这类**看起来像超参数、其实是语法参数**的常量。
