# 图 6：Engram 论文的 scaling law 结论（来自论文，非本书实测）

对应正文第 1.3 节。要点：**论文把“容量在 MoE 与 Engram 之间怎么分”变成一个
U 型曲线问题，并在 27B 规模上报告了系统性收益。**

> 注意： **本图所有数字都来自 Engram 论文本身**（`paper-deepseek-engram.pdf` 的文本抽取
> `14-deepseek-ascend-stack/raw/paper-deepseek-engram.md`）。**本章作者没有复现任何
> 训练，也没有加速器可做验证。** 这些数字的测量条件是论文自述的：
> 262B tokens、激活参数 3.8B、与严格 iso-parameter 且 iso-FLOPs 的 MoE-27B 对比。

## 图 6a：容量分配是 U 型的

```text
  验证 loss
   1.730 │ ● ρ=100%（纯 MoE）
   1.725 │   ╲
   1.720 │     ● ρ≈90%
   1.715 │        ╲
   1.710 │           ● ρ≈80%   ← 论文报告的最优点附近
   1.705 │             ╲
   1.700 │               ● ρ≈40%（Engram 拿走大部分容量）
         └──────────────────────────────────────────────►  分配比例 ρ
           100%  MoE 占满全部稀疏容量
             0%  全部给 Engram（条件计算容量被挤空）

   论文的两个端点解释：
     ρ → 100%（纯 MoE）：没有专门放静态模式的记忆，只能靠深度与计算去重建
     ρ → 0%  （纯 Engram）：缺少条件计算能力，需要动态推理的任务掉下来

   论文报告：把约 20%–25% 的稀疏容量让给 Engram 最好；10B 这一档，
   验证 loss 从 1.7248（ρ=100%）降到 1.7109（ρ≈80%，Δ=0.0139）。
```

## 图 6b：27B 规模上的主要收益（论文 Table 1，相对 MoE-27B）

```text
   benchmark            提升（百分点）
   BBH            +5.0   ████████████████████████
   CMMLU          +4.0   ███████████████████
   ARC-Challenge  +3.7   ██████████████████
   MMLU           +3.4   ████████████████
   DROP           +3.3   ████████████████
   HumanEval      +3.0   ███████████████
   MATH           +2.4   ███████████
   GSM8K          +2.2   ██████████
   MMLU-Pro       +1.8   ████████

   长上下文（论文正文）：
     Multi-Query NIAH   84.2 → 97.0
     Variable Tracking  77.0 → 89.0

   无限内存regime：把槽位从 2.58e5 扫到 1.0e7（最多约 130 亿参数），
   验证 loss 与槽位数在 log 空间里呈线性（严格幂律）——即「加记忆就继续变好」，
   不需要额外计算。

   系统效率：把 100B 参数的表卸载到 host memory，论文报告开销 < 3%。
```

## 读图要点

1. **U 型的两个端点都是“缺一条腿”。** 左边缺记忆（静态知识要靠算力重建），
   右边缺条件计算（动态推理掉下来）。这与“MoE 与 Engram 是互补的两条稀疏轴”
   是同一句判断的两种说法。
2. **增益最大的不是知识题。** 论文自己强调：知识类（MMLU/CMMLU）涨，但推理类
   （BBH/ARC-C）与代码数学（HumanEval/MATH）涨得更多。它把原因归到
   “Engram 让骨干网络的前几层不必再做静态模式重建，等于加长了有效深度”。
   这是论文的机制分析（LogitLens + CKA），不是本书的结论。
3. **“加表就变好”是这一章的工程意义所在。** 因为寻址确定性，表可以放到 host
   甚至 NVMe，容量不再受 HBM 限制；`O(1)` 指的是**每个 token 取的行数固定**，
   不是总复杂度固定（见图 1 与 `code/04`）。
4. **这些数字的边界很清楚**：都是论文自报、作者未复现；本章也没有能力在
   Ascend 950 上复现一次 27B 预训练。引用它们只是为了说明“为什么值得为这类
   访存算子单独写 kernel”，而不是当作本项目的性能承诺。

## 来源

* U 型曲线与分配比例：`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:364-430`
  （含 `1.7248 → 1.7109`、`ρ ≈ 75%–80%`、`20%–25%`）。
* 无限内存 regime 与幂律：`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:431-465`
  （含槽位区间 `2.58e5`–`1.0e7`、约 130 亿参数）。
* 27B 规模收益表：`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:470-520`（Table 1）；
  摘要口径：`:26-40`。
* 长上下文：`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:92-95` 与 `:670-682`。
* host memory 卸载 `< 3%`：`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:39,97-99`。
* 仓库侧的对照说明（“仓库里只有 demo”）：`../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/README.md:78-90`。

**本图不含任何本书的实测数据。**
