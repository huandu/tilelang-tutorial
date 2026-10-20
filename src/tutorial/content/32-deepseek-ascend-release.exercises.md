# 第 32 章 习题

说明：

* 难度：`★`（读完正文即可作答）/ `★★`（需要综合本章多处内容）/ `★★★`（需要动手、
  多步推理，或需要对照仓库源码）。
* 「需要硬件」一栏中，`否` 表示只要一台能跑 Python 的机器；`是（NPU）` 表示必须用
  真实的昇腾 NPU。**本章所有题目都标 `否`**：这是一章导览，全部题目都能静态完成。
* 所有题目只依赖第 32 章及之前章节的知识。涉及后续章节的地方会显式标注。
* 引用路径时请写成 `../tilelang-corpus/...` 或 `_vendor/...`，并给出行号；这是本教材
  的硬性要求（见正文 §8）。

---

## E1 概念题 ★ 不需要硬件

正文 §2.1 给出了官方文章对六个组件（TileLang 昇腾版、DeepGEMM-Ascend、DeepEP-Ascend、
TileKernels、FlashMLA、DeepSelect）的一句话定位。

1. 把这六个组件分别归入「语言/编译」「计算库」「算子库」「通信库」四层。
2. TileKernels 的官方定位里有一个定语很关键（「**常规**向量计算和访存算子」）。说出
   这个定语排除了什么——也就是哪些算子的极限性能**不**由 TileKernels 负责。
3. 为什么说「只开源一个算子库算不上全栈」？用 §1.2 的三层来说。

---

## E2 概念题 ★ 不需要硬件

正文 §2.2 引用了官方对 TileLang 昇腾版的措辞：「对昇腾 Ascend C 底层指令进行封装、
提供高级语言的编程方式、同时不损失硬件性能」。

1. 把这句话拆成三个互相牵制的目标，各用一句话解释。
2. 为什么不能把「不损失硬件性能」直接当成性能承诺？正文建议用什么材料去核验真实的
   性能数字？
3. §2.3 的那句承诺（「训练中用到的每一个 TileLang 算子，在昇腾上都有对应的高性能
   实现」）容易被过度推论成两个错误的结论。写出这两个错误结论。

---

## E3 概念题 ★★ 不需要硬件

正文 §4.2 把「一一对应」拆成三层可迁移性。

1. 写出这三层，并各配一个来自 §4 的例子。
2. 「实现技巧不可迁移」这一层，用 DeepGEMM-Ascend 的 README 原话举一个具体的技术点。
3. 有人看完对照表后说：「既然 API 一样，那我用 NV 上调试好的参数直接跑昇腾就行。」
   用本章内容指出这句话错在哪。

---

## E4 概念题 ★★ 不需要硬件

正文 §4.3 给了三类「并不一一对应」。对每一类各举一个例子，并写出出处（路径 + 行号）：

1. **格式差异**——以 scaling factor 的打包方式为例，分别写出 NVIDIA SM100 与昇腾的
   打包单位与存放顺序。
2. **功能子集差异**——FlashMLA 在昇腾侧缺了什么？DeepSelect 在昇腾侧限制了哪些参数？
3. **语义差异**——DeepEP-Ascend 的 README 给了哪两条约束/提醒？

---

## E5 代码阅读题 ★ 不需要硬件

读 TileKernels 的路由算子入口
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_kernel.py:7-43`）：

```python
topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
if num_tokens == 0:
    return topk_idx

if is_ascend():
    from tile_kernels.moe.topk_gate_asc import get_topk_gate_kernel_asc

    kernel = get_topk_gate_kernel_asc(num_experts, num_topk, get_num_vec_cores())
else:
    kernel = get_topk_gate_kernel_cuda(num_experts, num_topk, use_pdl=get_pdl())

kernel(scores, topk_idx)
```

1. 这段代码里的后端选择发生在什么时候（编译期 / 导入期 / 每次调用）？判据是什么？
2. 为什么昇腾分支的 `import` 写在 `if` 里面，而不是文件顶部？如果把它挪到顶部，
   在没有昇腾环境的机器上会发生什么？
3. `get_num_vec_cores()` 与 CUDA 分支里的 `use_pdl` 分别反映了两个平台的什么差异？
   （提示：见第 6 章「执行单元」与第 19 章 AIC/AIV 分工。）
4. 这个入口对应的正文结论是哪一句？

---

## E6 代码阅读题 ★★ 不需要硬件

DeepGEMM-Ascend 里唯一用 TileLang 写的算子是 mHC prenorm
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py`）。

1. 写出它的 import 语句，并说明这属于哪一棵「树」。
2. 从文件里找出至少四个**昇腾专属**的 TileLang 原语，说明各自的作用。
3. 该文件的注释里提到了 AutoSchedule 的一条限制（`AutoSchedule rejects …`）。它在讲
   什么？这条限制说明「用 TileLang 写」并不等于「不用管调度」——为什么？
4. 结合 §4.4，说明为什么 GEMM 主体没有用 TileLang，而 mHC prenorm 用了。请分别给出
   「用」和「不用」的判据。

---

## E7 代码阅读题 ★★ 不需要硬件

DeepSelect 的后端选择与平台差异写在
`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/deep_select/interface.py` 里。

1. 它的平台判据是什么（第 10 行）？与 TileKernels 的判据是否相同？
2. 列出文档字符串里至少四条「平台相关」的参数限制（参数名 + 限制），并说明它们属于
   §4.3 的哪一类差异。
3. 为什么这个仓库的 `topk()` 签名在两个平台上是一样的，但行为不完全一样？用一个具体
   参数说明「签名相同、语义不同」。
4. 这个例子对「一一对应」的理解有什么补充？

---

## E8 概念题 ★★ 不需要硬件

正文 §8.3 用 DeepEP-Ascend 的 `Ongoing` 一节区分了三种状态。

1. 写出这三种状态的判据。
2. 从 `Ongoing` 原文里各举一个例子（已发布 / 实验性 / 未实现）。
3. 「有 API」与「有实现」不是一回事，请说明在工程上你该如何处理一个只有 API 预留的
   功能（至少写两条具体动作）。
4. 如果一个组件 README 里**没有** `Ongoing` 一节，你能得出「它没有未实现的功能」这个
   结论吗？为什么？（提示：想一想 DeepSelect。）

---

## E9 动手题 ★★ 不需要硬件

本章的 `code/01-verify-checklist/` 里有一个纯标准库脚本，它能把一个仓库 README 里的
`Requirements` / `News` / `Ongoing` 等小节抽出来。

1. 运行它，对 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md`
   抽一遍，把输出贴出来。
2. 根据输出，写出该组件「能不能在 Ascend 950 + CANN 9.2.0 上用」的结论，并给出
   **至少两条**支持这个结论的原文证据。
3. 换一个 README（例如 TileKernels）再跑一次，指出两次输出的结构差异，并解释为什么
   抽取脚本必须「找不到就明说，而不是猜」。

---

## E10 动手题 ★★ 不需要硬件

用 §8.5 的六个问题，给自己选一个组件做一次完整的核验。要求：

1. 明确你选的是哪个组件、在哪个 commit 快照上（从 `PROVENANCE.md` 里抄）。
2. 逐题回答六个问题，每题都要给路径 + 行号或明确写「本快照未标注」。
3. 最后产出一句「可以直接写进结论」的话（格式参考 §8.7 的结尾）。
4. 指出你的核验里**仍然存疑**的地方。

---

## E11 概念题 ★★★ 不需要硬件

正文 §7.2 用一个定性公式解释了为什么 128 卡会把通信推成主角：

$$
\text{通信量} \approx \text{tokens} \times \text{topk} \times \text{hidden} \times b
$$

1. 用一句话说明这个公式里每一项的含义。
2. 为什么卡数增加会让「通信/计算」的比值上升？请从「token 被路由到别的卡的概率」和
   「每卡本地工作量」两个角度各说一句。
3. 官方文章关于超节点只有一句话（`deepseek-article-20260930.md:13`）。写出这句话
   **没有**给出的三类信息（例如拓扑、带宽、型号等），并说明本章是怎么处理这个信息缺口的。

---

## E12 综合题 ★★★ 不需要硬件

假设你要在昇腾上跑一次 MoE 推理（不是训练），并且手上只有这份快照。

1. 按执行顺序列出你会用到的组件，并给每一步标注它属于 §6.1 的哪一段。
2. 对每一个组件，写出你进它的 README 后**第一件要确认的事**。
3. 如果 `topk` 的输出 dtype 你想要 fp32 输入、或者想要排序输出，会遇到什么障碍？
   指出是哪个组件、哪一条限制、出处在哪里。
4. 最后写一句话总结：「我只凭这份 README 快照能确定什么、不能确定什么。」
