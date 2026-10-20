# 第 36 章 习题参考答案

答案按 E1–E14 一一对应。每题给“思路 → 参考答案 → 常见错误”三段。
所有数字都能在 `code/` 的脚本里复算；引用一律给路径与行号。

---

## E1 四步公式与 shape

**思路**：本章 §2.1 已经给了四步，直接抄结论但要写全 shape，
并解释第一步为什么不是矩阵乘（它没有可乘的维度，是按坐标取行）。

**参考答案**：

* `gathered_kv = KV[indices]`，结果 `[topk, d_qk]`；
* `P = Q @ gathered_kv^T`，结果 `[h_q, topk]`；
* `S = softmax(P * sm_scale, dim=-1)`，结果 `[h_q, topk]`；
* `out = S @ gathered_kv[:, :d_v]`，结果 `[h_q, d_v]`。

第一步是 **gather（按索引取行）**，不是矩阵乘：`indices` 的每个元素是一个行号，
`gathered_kv[k] = KV[indices[k]]`，没有任何“乘”的语义。
它之所以重要，是因为它把“连续存储”变成了“离散访问”，于是：
① 不能随路做 layout 转换（本章 §3.4）；② 每个 token 要单独发一次搬运请求，
撞上 MTE2 的队列深度上限（§3.3）。

报告位置：`:29-37`。

**常见错误**：把 `P` 写成 `[topk, h_q]`（应该是 `[h_q, topk]`）；
忘记 `out` 只用 `gathered_kv[:, :d_v]` 而 `P` 用了全部 `d_qk`。

---

## E2 为什么要有两个 max

**思路**：区分“真实 max”与“当前在用的 max”。

**参考答案**：

（a）语义（报告 `:45`）：

* `running_max`：到目前为止**每一行 P 的真实最大值**，只增不减；
* `running_max_for_softmax`：**上一次做 rescale 时使用的那个 max**，
  也就是 `S = exp(P − m_s)` 里的 `m_s`。它允许落后于 `running_max`。

（b）多一个 max 让它能**省掉 rescale**。rescale 的代价在昇腾上很高：
要把 `out_accum` 乘一个标量，而 `out_accum` 在 UB 里，于是要么搬 L0C 结果
（FixPipe），要么在 UB 上做一遍读改写（报告 `:160`）。
如果每来一个新块就把 `m_s` 追上 `m*`，这笔开销每块都要付。
改成“只在滞后超过 `RESCALE_THRES` 时更新”之后，绝大多数块直接跳过
（报告 `:69`：“如果没有任何一行 i 触发了这个条件，就跳过这一步”）。

**常见错误**：把 `running_max_for_softmax` 说成“上一步的 `running_max` 的副本”
它是**上次 rescale 时**的值，可能落后很多块。

---

## E3 复现算存比表

**思路**：跑脚本，逐行对；关键是**比较 `cube_cycles` 与 `mem_cycles` 的大小**。

**参考答案**：

`python3 code/01-head-roofline/head_roofline.py` 会打印与报告 `:104-111`
完全一致的八行（脚本里带 `--assert-report` 选项可自动断言）。

* `h_q = 32, topk = 1024`：`cube_cycles = 8192`，`mem_cycles = 11765`。
  `8192 < 11765` → **memory 更贵** → memory bound。
* `h_q = 64, topk = 1024`：`cube_cycles = 16384`，`mem_cycles = 12457`。
  `16384 > 12457` → **compute 更贵** → compute bound。

所以判据就是“`cube_cycles` 与 `mem_cycles` 谁大”。
`h_q = 64, topk = 512` 时是 `8192` vs `6921`，余量不大，报告因此说
`head = 64` 时“compute 和 memory 大致均衡”（`:114`）。

**常见错误**：把 `totk` 记成 `d`；忘了 `cube_cycles` 里那个 `(d+d)` 是两个 GEMM
（QKᵀ 与 PV）各 `d` 维；忘了 `mem_cycles` 要除以**每核每周期**的 L2 带宽
（5 TB/s ÷ 32 核 ÷ 1.65 GHz ≈ 94.7 B/cycle），而不是 5 TB/s。

---

## E4 阈值作用在 `P * sm_scale` 上

**思路**：把“乘 `sm_scale`”移进比较的两边。

**参考答案**：

（a）条件是 `(cur_max − max_for_scale) * sm_scale > RESCALE_THRESHOLD`。
两边同除以正数 `sm_scale`：
`(cur_max − max_for_scale) > RESCALE_THRESHOLD / sm_scale`。
所以把 `RESCALE_THRESHOLD / sm_scale` **预先算好**再和未缩放的差值比较，
结果完全一样，但在 kernel 里每次比较只用一次减法，不用先乘 `sm_scale`。
代码正是这么做的：`AuxParams.rescale_threshold_div_sm_scale`
（`kernel_body.h:1660-1662`），比较处用
`row_max_max_delta > aux_params.rescale_threshold_div_sm_scale`
（`kernel_body.h:1189`）。

（b）`sm_scale = 1/sqrt(512) ≈ 0.04419`，所以
`RESCALE_THRESHOLD / sm_scale ≈ 6 / 0.04419 ≈ 135.8`。
也就是说，**允许原始 P 的滞后达到约 136**；一旦乘上 `sm_scale`，它对应的
就是 6。这解释了报告 `:67-68` 那句“`exp(running_max − running_max_for_softmax)`
总是 ≤ exp(6)”。因为用于 exp 的自变量已经是缩放后的值。

**常见错误**：说“阈值是 6，所以 `exp(P)` 最多放大 6 倍”。
放大倍数对应的是 `exp(δ)`（δ 是缩放后的滞后），所以最大放大是
`exp(6) ≈ 403` 倍，不是 6 倍。E4(b) 的 135.8 是**未缩放** P 上的容忍量，
不要把这两个数混起来。

---

## E5 `MMA_M` 为什么不能小于 32

**思路**：抄注释里的两条理由，再对上 `FRACTAL_H`。

**参考答案**：`config.h:28` 的注释给了两条：

1. `MMA_M = 32` 时这个配置**已经**是 memory-bound（L2 带宽受限），
   再小只会更偏 memory bound，没有收益；
2. `MMA_M = 32` 对应 `MMA_M_PER_V = 16`，而 16 正好是**一个 L0C fractal 的高**
   （`config.h:71` 的 `FRACTAL_H = 16`）；`MMA_M` 小于 32 会让
   FixPipe → UB 的拷贝“出问题”。

也就是说 32 是**功能下限**（fractal 整除），64 是**性能取值**（打满 Cube 吞吐，
报告 `:116`）。这两个约束一起锁定了“一个 AI Core 64 个 head”。

**常见错误**：把 `FRACTAL_H = 16` 说成 CPU 缓存行；它是 Cube 侧 ZN/NZ
排布里 fractal 的行数，和 `L12L0_TILE_SIZE` / `FRACTAL_W` 是同一套 layout 概念
（见第 21 章）。

---

## E6 `B_TOPK > 73.14` 与 `B_TOPK = 64`

**思路**：看清那条不等式是为**哪个配置**推的。

**参考答案**：不矛盾。报告 `:142` 那段的前提是
“当只使用一个 CUBE Core 和一个 VECTOR Core（1C1V）的时候”，
FixPipe 带宽 128 B/cycle/AI Core 是**整个 1C1V 单元**的口径；
推出的 `B_TOPK > 73.14` 是“1C1V 下 FixPipe 不成为瓶颈”的条件。
而最终方案是 **1C2V**（报告 `:144`，`config.h:53`），
每个 VECTOR Core 只负责 `B_TOPK_PER_V = 32` 个 token、
`MMA_M_PER_V = 32` 个 head，FixPipe 的压力与 1C1V 不同，
所以可以取 `B_TOPK = 64`。

把 73.14 当硬约束，会得出“代码 `B_TOPK = 64` 是错的”这种荒谬结论，
或者会去错误地解释为什么代码“违反”了报告，实际上报告正是用这条不等式
**否掉 1C1V**，然后才选了不需要满足它的 1C2V。

**常见错误**：把“1C1V 的 bound”和“代码的配置”混为一谈；
忘记报告里这段的标题就是“能不能用 1C1V”。

---

## E7 复算 8192 周期

**思路**：MAC 数 ÷ FMA-per-cycle。

**参考答案**：`64 × 64 × 512 = 2,097,152` 次乘加。
HF32 关闭后 Cube 是 `256 FMA/cycle/AICore`，
所以 `2,097,152 / 256 = 8192` 周期。与报告 `:128` 的数字一致。

为什么是 `64 × 64 × 512`（M×K×N）而不是 `64 × 512 × 512`：
“把 accumulator 放回 L0C”这件事只需要一个**单位矩阵**参与，
单位矩阵是 `64 × 64`（因为它只需要在 64 个 head 的方向上做恒等变换），
乘上去的代价是 `M × K × N = 64 × 64 × 512`，而不是完整的再算一遍 attention
（那会是 `64 × 512 × 512`）。报告原话是“至少需要做一个 `64x64x512` 的 float 矩阵乘法”
（`:126`）。

**常见错误**：把 `64 × 512` 当成单位矩阵的规模；
忘了关 HF32 后是 256 而不是 4096 FMA/cycle（HF32 打开时是 4096，快 16 倍）。

---

## E8 FixPipe 与 MMA 周期

**思路**：直接代入 `B_TOPK = 64`。

**参考答案**：

（a）`B_TOPK = 64`：

* MMA 周期 `= 64 × 16 = 1024`；
* FixPipe 周期 `= 1024 + 64 × 2 = 1152`。

`1152 > 1024`，所以在这个配置下 **FixPipe 比 MMA 还慢**。这正是报告
说 1C1V 不行的依据（要让 FixPipe 不成为瓶颈就得
`B_TOPK · 16 > 1024 + B_TOPK · 2`，即 `B_TOPK > 73.14`）。

（b）报告 `:184` 说 `P[i]` 从就绪到被用掉“大约 1024 周期”。
这个量与 (a) 里 MMA 的一个块周期（1024）和 FixPipe 的常数项（1024）都同量级，
所以“用 `QKᵀ[i+1]` 顶上去”确实能填满这段空窗，一个块的矩阵乘时间
刚好和搬运 + softmax 的时间相抵。报告没有给这段的完整推导，
本节只是说明量级一致。

**常见错误**：把 `B_TOPK · 16` 里的 16 当成“16 个 token”；
它是 `(512+512)/4096 × 64 = 16`，即“每个 token 每个 head 摊到 16 个周期”。

---

## E9 `issue_kv_gather_later`

**思路**：读注释的两条理由，再看条件。

**参考答案**：

（a）条件：即将处理的那个块是**一个 index 组的组首**，即
`kv_block_idx % (INDEX_BLOCK_SIZE / B_TOPK) == 0`
（`kernel_body.h:1306-1310`）。此时 `run_process_index` 会真的干活
（`kernel_body.h:983-990` 里对“不是组首就 return”的判断），
所以把 `run_set_is_rescale_triggered_flag` 插在它前面是“免费的”。

（b）注释（`kernel_body.h:1307-1314`）给的两条理由：
1. `run_process_index` 正要占用 VF，此时发 `set_is_rescale_triggered_flag`
   不会让 VF 闲置；
2. `run_issue_kv_gather` **必须等** `run_process_index` 完成（它要用后者算出的
   MTE2 地址参数）。如果先发 gather 再发 flag，scalar core 就要停下来等
   `process_index`，而 scalar core 一停，**cube core 就会空闲**
   （这正是 §4.4 说的“阻塞式同步被顺序发射放大”）。

**常见错误**：以为这只是“顺序调整、影响很小”；
实际上它决定 CUBE 是否会出现气泡，属于必须解释的理由。

---

## E10 ND2NZ 为什么不能随路做

**思路**：抓住“源和目标都连续”这一条。

**参考答案**：正常做法是 GM→L1 时随路 ND2NZ，但有两个障碍：

1. KV 是按 token **稀疏 gather** 的，一次一个 token（1 KB），
   不是一段连续的矩阵，无法触发“随路转换”这种整块搬运的路径；
2. “第一个想法”是把一个 token 看成 32 个 32 B 块、在 L1 里以
   `B_TOPK * 32B` 为 stride 落位，但报告的实验显示：BIU
   “只会在**源和目标都连续**的时候才会触发访存合并”，
   而这里只有源连续、目的是 strided 的 → **无法合并访存**，
   L2 带宽利用率很低，瓶颈回到 L2 带宽（报告 `:152`）。

“先到 UB 用 SIMD VF 转 NZ”能绕开它，因为：UB 是一次**连续**的 MTE2 落点
（每个 token 一整块进去），而 ND→NZ 的重排是在**片上**由 VF 完成的，
重排发生在 UB 内部，不再需要访存合并来拉伸带宽（报告 `:154`）。

**常见错误**：把失败原因说成“MTE2 队列不够”；
这一条的失败原因是**访存合并**，不是队列深度（队列深度是 §3.3 bound ① 的事）。

---

## E11 跑 NumPy skip-scale 对拍

**思路**：脚本做两件事，用一个“一次算完”的参考实现做基准，
再用分块的 online softmax（带阈值）逼近它。

**参考答案**：

（a）**验证了什么**：在给定的输入下，“分块 + skip-scale”的 online softmax
与“一次算完”的 softmax **在数值上一致到 float64 精度**
（脚本打印最大绝对误差与最大相对误差）。也就是说，本章 §2.2 的递推与
§2.3 的阈值逻辑是**算法正确**的。
**没有验证什么**：任何硬件行为（FixPipe 带宽、MTE2 队列、icache、bank conflict）、
任何性能数字、任何浮点格式（脚本用 float64/float32 而非 bf16 的 Cube 行为）。
它**不是**对 FlashMLA 实现的复现。

（b）`RESCALE_THRES = 0`：每个块都 rescale（脚本打印 `rescale = 块数`），
数值上仍然正确，但 kernel 里每一步都要搬一次 `out_accum`，性能最差。
`RESCALE_THRES = 1e30`：永不 rescale，等价于一直用第一块的 max 做分母。
在脚本的**正常工况**下它看起来“也没错”。因为 `sm_scale = 1/sqrt(d)` 之后
`P·sm_scale` 本来就在 O(1) 量级，`exp` 不会溢出；但在脚本随后构造的
**病态工况**（后半段 KV 整体放大，导致后到的块把 logit 顶得很高）里，
float32 下 `S = exp(P − m_s)` 直接溢出成 inf/nan。
这就解释了报告 `:67-68` 为什么必须给 `δ` 设上界：**这不是精度问题，是溢出问题**。

**常见错误**：把脚本的对拍结论说成“证明 FlashMLA 算子是正确的”；
它只验证了算法的等价性。

---

## E12 为什么 ND2NZ 用 column-major 循环

**思路**：看 `vsstb` 的 post-update 语义。

**参考答案**：

（a）约束：`vsstb` 的 post-update **只接受正的 `offset`**
（`kernel_body.h:294-296`），也就是说写指针只能**一直往前走、不能回退**。

（b）为了让“写指针单调递增”，必须找一个**只增不减**的遍历顺序。
如果按 row-major（先走 KV token，再走 head 维度）遍历，写回位置会在
NZ 排布里来回跳，指针需要回退；所以改用 **column（head dim）major**：
外层按 head 维的 chunk 走，内层按 token 走，指针单调前进。
而且外层步长取 2（同时处理两个相邻 256 B），是为了打满两个 UB bank group
（`kernel_body.h:296`）。

**常见错误**：以为循环顺序是“为了缓存友好”这种通用理由；
它是被**某条指令的 post-update 语义**直接约束出来的。

---

## E13 四个技术点的分类

**思路**：用 §6.1 的判据：“换一条语义相近的指令，结论还成立吗？”

**参考答案**：

（a）**online softmax 的 rescale** → **A 类**。
判据：它只需要“逐元素乘 + 行归约 + 条件分支”，用 tile 级原语
（`T.Parallel`、`T.reduce_max` 等）就能描述，不依赖任何特定指令的语义。

（b）**用 `T.simd.vsstb` 做 ND2NZ 的写回** → **B 类**。
判据：M 树里**有**这个原语（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/
simd.py:482`，带 `update` 参数对应 post-update），所以不用退回 Ascend C；
但你需要显式进入 `T.SimdVF` 区域并指定 stride/mask，不能再指望自动调度。

（c）**gather2 的 `burst_count = 2`** → **C 类**。
判据：收益完全来自这一个参数位（报告 `:216`）；`T.copy` 这一层抽象
没有暴露 `burst_count` / `src_stride`。换成“语义相近的搬运指令”，
结论就不成立了（没有 burst 就没有省请求）。

（d）**FP4 每 16 维共享 scale 的 `UNPK4` 转换** → **C 类**。
判据：这是**带数据分布语义**的格式转换
（`asc_loadalign_unpack4_postupdate` + `asc_e2m1x22bfloat16`，
`kernel_body.h:383-388`）。`T.simd.*` 覆盖了通用 `vld`/`vsstb`，
但没有覆盖这个“一个字节里两个 e2m1 要按低/高 nibble 拆到 128 个有序 BF16”
的专门转换。

**常见错误**：把 (b) 归到 C 类（以为“`vsstb` 是 CCE intrinsic 所以必须手写”）
在 M 树上它是可表达的；真正落在 C 类的是**围绕它的协议与参数**，
不是指令名本身。

---

## E14 icache 与 L1 bank 配平

**思路**：一条是“指令侧”，一条是“地址侧”。

**参考答案**：

* **§5.4 icache**：照顾的是**指令缓存**（32K/16K/8K，报告 `:238`）。
  它对应的动作是“不要展开循环、不要过度内联、把两段循环 fuse”。
  这类决策依赖“生成的机器码有多大”，而第 23 章的调度器**不建模代码体积**
  （它估的是延迟与资源），所以**不能被自动调度覆盖**，必须由人判断。
* **§5.5 L1 bank 配平**：照顾的是**L1 的地址映射**（两个 bank：0–256K、
  256K–512K，报告 `:242`）。它对应的动作是“把每个 Tensor 拆成 lo/hi 两份，
  在地址上错开”。这本质上是一个**地址分配 / 存储布局**问题，
  完全可以被编译期优化覆盖（第 21 章的 layout 推导、
  第 25 章 §2.2 的 scope → 地址映射都是这类问题），所以**可以由编译器做**，
  只是当前版本（v0.1.15）还没有自动做这件事，需要手写。

**常见错误**：把两条都说成“硬件限制所以必须手写”。
区分点是：**icache 依赖代码体积（编译器不建模），bank 依赖地址映射（编译器
可以建模但尚未实现）**。
