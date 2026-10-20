# 第 28 章 习题

说明：

* 难度 `★`（读完本章即可做）/ `★★`（需要动手复算或读源码）/ `★★★`（需要综合判断）。
* “需要硬件”一栏里写“否”的题，只用标准库 Python 在本机就能完成；
  写“是”的题必须在一台有 NVIDIA GPU 或昇腾 NPU 的机器上才能得出
  **实测**结论，但在无卡机器上仍然可以完成它的设计部分。
* 所有计算题都可以用 `code/` 下的脚本复算；先自己算，再对答案。
* 只用到本章及之前各章的知识。

---

**E1（概念，★，无需硬件）** 写出本章“先算再写”的五步流程，并解释为什么
“定 tile 尺寸”必须排在“定流水结构”之前。举一个把两者顺序颠倒会出问题的具体例子。

**E2（概念，★，无需硬件）** roofline 的三个输入分别是什么？为什么 TileOPs 的
规范要求 roof key 必须由算子的作者在代码里**声明**，而不能从正在跑的 kernel
或者输入 dtype 推断出来？

**E3（计算，★★，无需硬件）** 用 `TileOPs/src/tileops/perf/profiles/h200.yaml` 的
数值，算出 `cuda_core.fp32` 的平衡点，并写出你的算式（theoretical、calibration、
带宽各用了哪个数）。然后回答：为什么给一个逐元素 / 归约类的算子用
`tensor_core.fp16` 的 roof 是错的？用本章案例算子的 AI（3.102）说明这个错误
会把平衡点带偏多少倍。

**E4（代码阅读，★★，无需硬件）** 读
`_vendor/tilelang-v0.1.15/tilelang/tools/Analyzer.py` 的第 10 行与第 187–204 行。
回答：

1. 在一个 compute capability 9.0（Hopper）的机器上调用
   `Analyzer.analysis(tir, CUDA("cuda"))`，`result.expected_tflops` 是什么？
2. 此时 `result.estimated_time` 等于什么？它漏掉了哪一半？
3. 为什么说这个工具“不会报错，只会给你半个答案”？
4. 如果你在一个 sm_89 的机器上跑同一个 kernel，`expected_tflops` 是不是一个
   可以拿来和实测 TFLOPS 比较的数？为什么？

**E5（代码阅读，★★，无需硬件）** 读
`_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/default.py:537-574`，
写出 `compute_tile_dict` 里与 `reg_usage`、`block_per_SM`、`num_wave` 有关的四行公式。
然后代入 `block_M=block_N=128`、`block_K=32`、fp16 输入、fp32 累加
（`smem_cap=227 KB`、`reg_cap=65536`、`sm_partition=4`、`SM 数=132`），回答：
`reg_usage` 与 `smem_cost` 各是多少？`block_per_SM` 的三项分别是多少，
哪一项是 binding？

**E6（动手，★★，无需硬件）** 运行

```bash
python3 code/03-carver-cost-model/carver_cost_model.py --smem-cap 101376 --cores 108 --topk 8
```

把它与默认参数（`--smem-cap 232448 --cores 132`）的结果对比，回答：

1. top-8 的 tile 有没有变化？为什么？
2. `num_wave` 变化了多少？
3. 这两个参数里，哪一个更能解释结果的差异？

**E7（动手，★，无需硬件）** 运行

```bash
python3 code/01-op-ledger/op_ledger.py --case gemm --m 128 --n 128 --k 128
```

然后分别用 `--roof tensor_core.fp16` 和 `--roof cuda_core.fp32` 各跑一次。
写出三次输出的 `bound` 与平衡点，并解释“同一个算子在两张表里有两种身份”这件事，
以及它对设计决策意味着什么。

**E8（代码阅读，★★★，无需硬件）** 读
`_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.cc:692-707`。

1. 写出 `CubeOperations` 的完整公式。
2. fp16/bf16 输入时 `k_group` 是多少？fp32 在 HF32 模式下呢？普通 fp32 呢？
3. 一个 `M=256, N=200, K=128` 的 bf16 tile，实际按多少 ops 计费？
   与 `2*M*N*K` 相比多付了多少百分比？
4. 这个对齐代价在 NVIDIA 侧的 `mma` 形状约束里有没有对应物？

**E9（动手，★★，无需硬件）** 运行

```bash
python3 code/07-ascend-latency-model/ascend_pipe_model.py
```

1. 从输出里读出方形 tile 的 Cube/MTE2 翻转点在哪两个边长之间。
2. 用本章的闭式解 $MN/(M+N) \ge 81.92$ 算出理论翻转点，与第 1 问对比。
3. 为什么在 `M=N=128` 那一行，MTE2 的 II 恰好是 128 cycles 而不是
   `2·(128·16 + 16·128)/100 = 82`？
4. 官方昇腾 950 的 GEMM+ReLU 例子用的是什么 tile？代入模型后 binding 是哪条 pipe？

**E10（概念，★★，无需硬件）** 写出 fusion / split 的四个判据问题。然后用
`code/05-fusion-split/` 的输出说明：为什么 split-K 与 split-KV 这两个选择会
“搬运更多字节反而胜出”？它们的共同原因是什么？

**E11（动手，★★★，无需硬件）** 运行

```bash
python3 code/06-case-norm-rope-quant/case_norm_rope_quant.py --rows-per-block 16
```

回答：

1. 设计 B（16 行一 block）的总流量是多少？相对理想值是多少倍？
2. 设计 C（40 行一 block）的总流量是多少？为什么它已经**无法再改善**查表部分？
3. 如果把 `q_heads` 改成 64（`--q-heads 64`），设计 C 的 `rows/blk` 会变成多少？
   这时它的片上占用会变成多少？还放得下 192 KB 的预算吗？

**E12（设计，★★★，需要硬件验证）** 本章 6.5 节选了设计 C（一个 token 一 block）。
现在假设产品需求变成：**量化必须与 norm+RoPE 分开**（因为量化要复用同一个 scale
做后续的 KV cache 写入）。请设计第三版：

1. 写出它的预算表（payload / lookup / 总流量 / 片上占用 / grid）；
2. 说明它相比设计 C 多搬运了多少字节，这些字节是不是不可避免的；
3. 给出“在有卡机器上如何验证你的判断”的步骤，包括要测量什么、判据是什么。

**E13（概念，★★，无需硬件）** 从本章第 7 节的反模式清单里挑六条，逐条写出它的
**定量依据**（数字或公式），并说明如果不遵守它，你会先看到什么现象。

**E14（动手，★★，无需硬件）** 用

```bash
python3 code/04-tile-size-decision/tile_decision.py --diagnose \
        --block-m 128 --block-n 256 --block-k 64 --stages 3 --threads 256
```

回答：

1. 这份配置被哪一条约束卡住？为什么不是寄存器？
2. 如果保持 `block_M=128, block_N=256` 不变，把 `stages` 降到 2，
   `blocks_per_SM` 会变成多少？为什么？
3. 如果改用 `--instr mma`，这份配置还合法吗？为什么？
4. 综合第 1–3 问，给出这份配置的下一步动作（只写一个动作，并说明判据）。
