# 第 36 章 习题

共 14 题。难度 `★`（基础）/ `★★`（需要推导或读源码）/ `★★★`（综合）。
标 **[无硬件]** 的题在本机就能完成；标 **[需硬件]** 的题需要昇腾 NPU，
本章不要求你拥有它——这类题只要求你写出「前提 / 命令 / 期望输出 / 验证级别」。

习题只使用本章与前置章节（第 15、19、22、23、25、35 章）的知识。

---

**E1（概念，★，[无硬件]）** 写出单个 Q token 的稀疏 attention 前向的四步公式，
并给出每一步结果的 shape。然后回答：为什么第一步是「gather」而不是普通矩阵乘？

**E2（概念，★，[无硬件]）** 教科书的 online softmax 只维护一个 max `m`，
本章的 kernel 维护 `running_max` 与 `running_max_for_softmax` 两个。
（a）说明这两个量各自的语义；（b）解释为什么**多一个** max 反而能省时间。

**E3（动手，★★，[无硬件]）** 运行 `code/01-head-roofline/head_roofline.py`，
把输出与报告 `:104-111` 的表逐行对照。然后回答：
为什么 `h_q = 32, topk = 1024` 判为 memory bound，而 `h_q = 64, topk = 1024`
判为 compute bound？写出你比较的两个数。

**E4（数值 + 概念，★★，[无硬件]）** 报告说 `RESCALE_THRES` 一般取 6。
（a）代码里 `config.h:43` 的注释写的是
`(cur_max - max_for_scale) * sm_scale <= RESCALE_THRESHOLD`，
`run()` 里预计算的却是 `RESCALE_THRESHOLD / sm_scale`。解释这两者是等价的，
并说明为什么必须这么转换。
（b）取 `sm_scale = 1/sqrt(512)`，算一下：这个阈值相当于允许**原始 P**（未乘
`sm_scale`）的滞后有多大？这个数和 `exp(6)` 有什么关系？

**E5（代码阅读，★★，[无硬件]）** 读 `config.h:28` 对 `MMA_M = ku::ceil(H_Q, 32u)`
的注释。它给了两条「为什么不能小于 32」的理由。复述这两条，并说明
`MMA_M_PER_V = 16` 与「一个 L0C fractal 的高」的关系（提示：`FRACTAL_H = 16`）。

**E6（概念，★★，[无硬件]）** 报告算出 `B_TOPK > 73.14`（`:142`），
但最终代码里 `B_TOPK = 64`（`config.h:55`）。这两件事矛盾吗？为什么？
如果你把 73.14 当成代码的硬约束，会在读代码时得出什么错误结论？

**E7（数值推导，★★★，[无硬件]）** 报告说：若要在 rescale 时通过「乘单位矩阵」
把 accumulator 放回 L0C，需要关掉 HF32，此时 Cube 只有 256 FMA/cycle/AICore，
一个 `64 × 64 × 512` 的矩阵乘要 8192 周期。请复算这 8192，并解释
「为什么是 `64 × 64 × 512` 而不是 `64 × 512 × 512`」。

**E8（数值推导，★★，[无硬件]）** 用报告 `:142` 的两个公式：
MMA 周期 `= B_TOPK · 16`，FixPipe 周期 `= 1024 + B_TOPK · 2`。
（a）取 `B_TOPK = 64`，分别算出两者，并指出哪个大；
（b）报告说 `P[i]` 的空窗「大约 1024 周期」（`:184`）。用 (a) 的结果说明
这个量级为什么和 FixPipe 那一项可比。

**E9（代码阅读，★★，[无硬件]）** 阅读 `kernel_body.h:1306-1319` 的
`systolic_arr_activate`（尤其是 `issue_kv_gather_later` 那段注释）。
（a）说明什么条件下代码会把 `run_set_is_rescale_triggered_flag` 提到
`run_issue_kv_gather` **之前**；
（b）结合 §4.4 的「顺序发射、顺序执行」，解释如果**不**这么做会发生什么。

**E10（概念，★★，[无硬件]）** 报告中 KV 的 ND→L1「随路转换」方案为什么失败了？
请用「BIU 只在什么条件下触发访存合并」这句话回答，并说明为什么
「先到 UB 用 SIMD VF 转 NZ」能绕开这个问题。

**E11（动手，★★★，[无硬件]）** 运行
`code/03-online-softmax-skip-scale/online_softmax_skip_scale.py`。
（a）说明它验证了什么、**没有**验证什么；
（b）把脚本里的 `RESCALE_THRES` 改成 0（每块都 rescale）和改成 `1e30`（永不 rescale），
观察 **rescale 次数**与误差怎么变；再解释脚本里那段「病态工况」（后半段 KV 被放大）
为什么让 `1e30` 那一档在 float32 下变成 inf/nan。

**E12（代码阅读，★★，[无硬件]）** `kv_nd2nz_for_prefill_vf`
（`kernel_body.h:277-331`）用的是「column（head dim）major」的外层循环。
代码注释（`:294-296`）说这是被 `vsstb` 的 post-update 约束反推出来的。
（a）复述这条约束；（b）如果改成 row-major，会违反什么？

**E13（综合，★★★，[无硬件]）** 把下面四个技术点按 §6 的 A/B/C 三类归类，
并**逐个写出判据**（不要只给类别）：
（a）online softmax 的 rescale；
（b）用 `T.simd.vsstb` 做 ND2NZ 的写回；
（c）gather2 的 `burst_count = 2`；
（d）KV 的 FP4 反量化里每 16 维共享一个 scale 的 `UNPK4` 转换。

**E14（概念，★★，[无硬件]）** 本章的 §5.4（icache）与 §5.5（L1 bank 配平）
这两条优化，分别照顾的是哪一类资源？请说明为什么其中一条**不能**被
第 23 章的自动调度器覆盖，而另一条**可以**被「地址分配」这类编译期优化覆盖。

---

> 想加做一道真机题的话：在满足 `FlashMLA/README.md:85-88`（Ascend 950 +
> CANN ≥ 9.2.0 + `torch_npu`）的机器上跑 `python tests/test-sparse-prefill.py`，
> 按本章「前提 / 命令 / 期望输出 / 验证级别」的四段式写一份复现记录，
> 并明确区分「你实测的数字」与「上游自报的数字」。**本章作者没有做过这件事。**
