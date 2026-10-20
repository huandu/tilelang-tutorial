# 图 3：三类资源的预算与口径

**用途**：正文 §4.1、§4.2。

**事实来源**：

* 算力三档（bf16 432.5 / fp8 865 / fp4 1730.1 TFLOPS）：
  **本书复算**，公式 `flops_per_cycle × 1650 MHz × 32 核`，参数见
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:46-71`；
  与 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:90-100` 的 `Hardware limit` 列吻合。
* 片内带宽（HBM 3500 GB/s、L2 4500 GB/s）：同一份 `HWDesc` 里的硬件模型常量。
* 卡间带宽（dispatch 313–375、combine 272–347 GB/s）：
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-21`，条件是 Ascend 950DT + CANN 9.2.0 + PoC HDK、
  netlayer 1、16384 tokens/rank、hidden 7168、top-6/256 experts、
  10 warmup + 50 samples、刷 cache。

```text
类别        本次引用到的量                          来源性质
──────────────────────────────────────────────────────────────
算力        bf16 432.5 / fp8 865 / fp4 1730.1     本书复算（源码硬件模型）
            TFLOPS（单卡，32 AI Core）

片内带宽     HBM 3500 GB/s、L2 4500 GB/s           源码里的模型常量（单卡）

卡间带宽     dispatch 313–375 GB/s                  厂商实测（PoC HDK）
            combine  272–347 GB/s

唯一的跨类比较（只用于判类）：
    3500 GB/s ÷ 374 GB/s ≈ 9.4   → 跨卡搬一次 ≈ 本卡搬十次
```

**要点**

* 三行来自**三种不同性质**的来源：模型常量 ×2、实测 ×1。不能当成同一种
  “芯片规格”来互相运算（正文 §4.4）。
* “90–95% 的物理 payload 带宽上限”用的是**逻辑 payload**口径，不含 padding、
  不含 barrier（第 38 章 §7.4）。
* 这张图里的任何数字都不能与 NVIDIA 侧的旧数字放一起比（第 32 章 §4.3）。
