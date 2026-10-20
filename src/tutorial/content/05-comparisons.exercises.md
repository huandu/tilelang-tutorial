# 第 05 章 习题

难度标记：`★` 概念确认，`★★` 需要阅读源码或资料，`★★★` 需要综合判断。
“硬件”一栏写“否”表示无需 GPU/NPU，纯读代码/资料即可作答。

---

## E1 ★ 抽象层次

用本章的“三个决策”（谁搬数据、谁切 tile、谁同步）填空：

| 对象 | 谁搬数据 | 谁切 tile | 谁插同步 |
| --- | --- | --- | --- |
| CUDA | ? | ? | ? |
| Triton | ? | ? | ? |
| TileLang（NVIDIA 后端） | ? | ? | ? |

并回答：为什么说 TVM 不在这张表的任何一行里？

（硬件：否）

---

## E2 ★ tile 的定义

写出 Triton 2019 论文对 tile 的定义原文（英文），并注明来源文件与行号。
然后说明：为什么“tile”这个定义同时适用于 Triton 和 TileLang，却**不**适用于 CUDA。

（硬件：否）

---

## E3 ★★ 2019 Triton 的能力边界

有人读完 Triton 的 2019 论文后说“Triton 原生支持张量核，所以写 GEMM 比 CUDA 容易”。
请用论文原文指出这句话里哪一部分**至少对 2019 年的 Triton 不成立**，并给出出处。

（硬件：否）

---

## E4 ★★ 读 TileLang 源码：谁在插同步

打开 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py`，找出 NVIDIA 后端 pass 顺序里
与“流水”和“同步”有关的两处（提示：一个在 `129` 行附近，一个在 `258` 行附近）。
写出这两段的 pass 名，并解释为什么 TileLang 的用户不需要写 `__syncthreads()`。

（硬件：否）

---

## E5 ★★★ TVM 的角色

本章说“TVM 是底盘，不是竞争者”。请给出至少两条**源码级**证据支持这个说法，并解释
如果把 TVM 换成“TileLang 自己写的 IR 框架”，TileLang 会失去什么。

（硬件：否）

---

## E6 ★★ Ascend C 与 TileLang@Ascend 的分工

依据 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` 里的对照表，列出至少三件
“用 TileLang 写时可以跳过、但用 Ascend C 写时必须自己做”的事，并指出这张表在文件里的
行号范围。

（硬件：否）

---

## E7 ★★ PTO-DSL 与 TileLang@Ascend 的关系

PTO-DSL 的 README 自己说明了它和 tilelang-ascend 的关系。请引用原文，说明：
(1) 谁更“低层”；(2) 举出一个“在 tilelang 里一行、在 PTO-DSL 里要自己写”的例子。

（硬件：否）

---

## E8 ★★ 统一例子：融合 epilogue 的位置

在本章的统一例子（`C = relu(A @ B)`）里，TileLang 用 `T.Parallel` 对 `C_local` 做 ReLU。
请回答：

1. 为什么在 CUDA 里这个 ReLU 通常是“另一个 kernel”或“GEMM 的 epilogue stage”，
   而在 TileLang 里可以就是同一个 kernel 里的几行？
2. 查 `_vendor/tilelang-v0.1.15/examples/quickstart.py`，写出这段融合 ReLU 的行号区间。

（硬件：否）

---

## E9 ★★ 昇腾上的 persistent kernel

PTO-DSL 的 matmul 教程指出：NPU 算子的 `grid` 通常等于硬件核数（persistent），而不像
普通 CUDA/Triton kernel 那样随输入规模增长。请引用该教程的说法，并说明 TileLang@Ascend
示例里对应的是哪个原语（提示：`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md`）。

（硬件：否）

---

## E10 ★★★ 决策：该选谁

下面三个场景，各选一个工具并给出理由（必须引用本章的证据，不许只凭印象）：

1. 你在一台昇腾 950 上要把一个自定义 attention 前向写到接近 CANN 算子库的水平，
   并且希望自动处理多缓冲与同步。
2. 你要在 NVIDIA H100 上把 GEMM 的 fp8 blockscaled 路径对齐 CUTLASS，并保留对
   epilogue 的完全控制。
3. 你要给 TileLang 加一个新的 pass，改善某个 layout 推导的结果。

（硬件：否）

---

## E11 ★★ 资料缺口

本章明确标注了“资料不足”的地方。请列举**至少三处**，每处说明：缺什么资料、为什么缺了
就无法下结论、建议从哪里补。（不要求你补齐资料，只要求写清缺口。）

（硬件：否）

---

## E12 ★★★ 代码阅读：CuTe DSL 与 TileLang 的区别

读
`../tilelang-corpus/07-tilelang-ascend-adapter/raw/TileFoundry/examples/minicpm3_4b-cutedsl/kernels/cute_gemv.py:36-113`。
找出其中属于“线程/指令级”的三个 API（例如带 `tidx` 的参数），并说明：如果同样一件事
用 TileLang 写，哪几个概念会从代码里消失。为什么这**不**意味着 CuTe DSL“更差”？

（硬件：否）

---

## E13 ★★ 社区数字的引用规范

有一位社区作者写道“TileLang 性能已经和主线 Ascend C 打得有来有回，代码行数差了好几倍”。
请按本章的引用规则改写这句话，使其成为一句**可被审计**的表述（要写清：来源是什么性质、
是否厂商材料、有没有测量口径、本书如何使用它）。

（硬件：否）

---

## E14 ★★★ 动手：画坐标系

用 Mermaid 或 ASCII 画一张图，把本章七个对象分别放到“线程/指令级”“tile 级”
“编译器底盘”三层，并标出 TileLang 与每一个对象的连线类型（竞争 / 互补 / 底盘）。
图里每个连线上写一个本章用过的证据编号（如 R2、S1、S11）。

（硬件：否）

---

## E15 ★★ 读源码：Z3 在昇腾调度里的位置

打开 `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py`，
说明这个文件开头为什么强调 Z3“not thread-safe”，以及它在整个昇腾调度流水里对应
`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py` 的哪个 pass。

（硬件：否）
