# 第 28 章 算子设计方法论与性能模型

## 学习目标

读完本章，你应当能够：

1. 拿到一个没见过的新算子，按固定顺序走完“**先算再写**”的五步，建立性能模型 →
   决定 tile 尺寸与存储位置 → 决定流水结构 → 决定是否需要特殊指令 / 特殊布局 →
   最后才写代码，并在每一步给出可检验的判据，而不是“试试看”。
2. 把 roofline 用对：算出 `(flops, bytes)`、算术强度与机器平衡点，说清它给出的
   是**下界**而不是预测；并当场指出至少六种常见误读把它们带偏了多少。
3. 说清 tile 级解析性能模型在做什么：TileSight 的三层结构（intra-tile / inter-tile /
   cross-device），以及 TileLang 自带的 carver 里 **traffic、片上容量、寄存器、
   驻留块数、波数** 这五个量是怎么算出来、怎么被用来排序的；并用
   `tilelang.tools.Analyzer` 的**实际实现**说明一个工具的输出边界在哪里，
   它在什么情况下会静默地给出半个答案。
4. 拿到一份 tile 配置，能指出它被**哪一条**约束卡住（片上容量 / 寄存器 / 指令形状 /
   并行度），并给出“先粗算再微调”的下一步。
5. 对 fusion、split、特殊指令三类决定给出判据而不是结论，并说明同样的判据在
   NVIDIA 与昇腾上为什么会得出不同答案。
6. 独立完成一个融合算子的完整设计循环：预算表 → 第一版 → 性能诊断 → 第二版。

## 前置知识

* 第 2 章：`(flops, bytes)` 记账法、算术强度、机器平衡点、roofline 是下界。
  本章 2 节是它的“进阶用法 + 常见误读”版。
* 第 11 章：`T.Pipelined` 的 stage/order、多缓冲的容量约束、
  “`num_stages ≥ 1 + ceil(latency / compute)`”这条判据；以及 3.9 节对 carver 的
  初步介绍。本章 3.2 节会把 carver 的 cost model 拆到公式一级。
* 第 12 章：`lower_trace` / `pass_diff`，以及 Nsight Compute 与 msProf 的口径。
  本章给出的所有“诊断”都是这一步的输入。
* 第 13 章：RMSNorm 与 RoPE 的实现与融合；第 27 章：fp8 量化与 block scaling。
  本章第 6 节的案例把它们拼在一起。
* 第 14、15 章：GEMM 与 attention 的性能账本与变体家族。
* 第 21 章：layout 推导与布局冲突的代价；第 23 章：昇腾的 latency 估计与 SMT 调度。

**本章的验证边界（务必先读）**：写作环境是 macOS (Apple Silicon)，**没有 NVIDIA
GPU，也没有昇腾 NPU**。因此：

* 本章给出的所有数字都只有三类来源：**公式推导**（`code/` 下的七个脚本，纯标准库，
  可在任意机器复算）、**厂商公开文档**、**锁定版本源码里的常量**。
* **没有任何一条性能数字来自本机实测。**
* 凡涉及具体 kernel 的行为，本章只写“应该怎么验证”，不写“实测结果是多少”。

---

## 1. 问题的由来：为什么必须“先算再写”

### 1.1 三种几乎人人都会踩的失败

一个没有经历过系统方法训练的算子作者，最常见的三种失败长这样：

**失败一：调参调在了一个不存在最优解的空间里。** 第 13 章的账本已经说明，RMSNorm
这类算子的算术强度在 1 FLOP/Byte 量级，比任何一条 roof 的平衡点都低两个数量级。
在这类算子上调 `block_K`、调 `num_stages`，能拿到的收益上限是百分之几；
而真正决定性能的是**访存量**，把三次访存合成一次，就是 3 倍的差距。
`code/01-op-ledger` 的 `--sweep` 把这件事量化了：同一个 `sin(cos(x))`，
融合前后分别是 14.905 µs 与 29.810 µs 的下界（H200 标定值口径）。

**失败二：tile 尺寸是在编译失败的报错里学会的。** `block_M=256, block_N=256,
block_K=64, num_stages=4` 在 Hopper 上要 256 KB 的 shared memory，而单 block 上限是
227 KB。这不是“慢一点”，是**编译或启动阶段直接失败**
（`code/04-tile-size-decision` 复现了这条边界）。同一份配置的累加器还要 256 个
寄存器/线程，已经越过 255 的硬上限。这两条都是**写代码之前就能算出来的**。

**失败三：把一个带宽问题当成算力问题。** 一个 kernel 实测只有峰值带宽的 30%，
于是去查张量核利用率、去换 `T.gemm` 的 policy。真实原因是每个 block 只搬 256 字节，
描述符与调度开销远大于载荷。这也是 `code/06` 里第一版设计失败的方式。

三种失败的共同点是：**它们都不是“写错了代码”，而是“想错了问题”。**

### 1.2 设计流程：代码是最后一步

本章把前面二十几章的实操经验收敛成一条固定的流程（图 1）：

```text
① 建模       (flops, bytes) -> AI -> 平衡点 -> 判类：memory-bound 还是 compute-bound
             ↓
② 定 tile    四条约束（片上容量 / 寄存器 / 指令形状 / 并行度）-> tile 尺寸 + 存储位置
             ↓
③ 定流水     搬运延迟与计算周期的比 -> 要不要流水、要几份缓冲
             ↓
④ 定特殊路径 默认降级路径够不够快 -> 需不需要张量核 / TMA / swizzle / fractal / dual_copy
             ↓
⑤ 写代码     ①–④的结论变成 tile 尺寸、scope、num_stages 与注解
             ↓
⑥ 对账       实测与模型差多少、差在哪一条约束上 -> 回到最可疑的那一步
```

后面五节一节一步。请特别注意 ② 与 ③ 的顺序：**先定 tile，再定流水。**
第 11 章 4.5 节已经说明“能加多少 stage 是 tile 尺寸的函数”，反过来也一样，
一个为流水优化过的 `block_K` 如果放不进片上，前面的工作全部作废。

---

## 2. 第一步：把一个算子变成 `(flops, bytes)`

### 2.1 账本模板

第 2 章给了四条记账规则。做设计时把它变成一张固定的表格，逐行填：

```text
项                     flops          bytes
读入  A, B              -             Σ 每个输入对象一次
写出  C                 -             Σ 每个输出对象一次
中间量                  -             0（规则 1：不单独计）
逐元素算术              1 FLOP / 次     -
矩阵乘                  2 FLOP / 乘加   -
超越函数                1 FLOP / 次     -（这是账本口径，不是硬件成本）
特殊输出（scale 等）     -             按真实字节数计
```

最后一行是最容易漏的。在 `code/06` 的案例里，每一行都要写一个 fp32 的量化 scale，
它只有 0.655 MB，占理想流量的 1%；但如果你的设计让它在每个 head 都重算一遍，
它就会变成一个新的大头。**漏掉一行不会让账本错得很离谱，但会让后面的判据建立在
错误的基数上。**

`code/01-op-ledger/op_ledger.py` 就是这张表：它把每个 term 的 flops 与 bytes 分别打印，
再给总数、算术强度、四条 roof 的平衡点与 `SOL time = max(memory, compute)`。

### 2.2 roofline 的正确用法

roofline 只有一个公式和一条判据：

$$
\mathrm{AI} = \frac{\text{flops}}{\text{bytes}}, \qquad
\text{平衡点} = \frac{\text{峰值算力}}{\text{峰值带宽}}, \qquad
\text{下界} = \max\!\left(\frac{\text{bytes}}{\text{带宽}},
\frac{\text{flops}}{\text{峰值算力}}\right)
$$

正确的用法有且只有三条：

1. **用它判类，不要用它预测。** 它给出的是“再好也不可能快过这个数”。
   真实实现一定更慢（第 2 章 2.5 节）。
2. **roof 必须由设计者声明，而且声明的是“最优实现会用哪个单元”。**
   TileOPs 的 roofline 文档把这条写成了规范：roof key“state the unit an
   **optimal** implementation would use, declared by the op author in code. It is
   never inferred from the running kernel”
   （`../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md:63-70`）。
3. **换任何一台机器、任何一个 shape，都要重算。** AI 随规模变化（第 14 章 1.2 节），
   标定后的峰值与带宽也随芯片变化。

用 H200 的标定 profile（R2：`TileOPs/src/tileops/perf/profiles/h200.yaml`，
`effective = theoretical × calibration`，定义见 R1 §5.1）算出四条 roof 的平衡点：

| roof | theoretical | calibration | effective | 平衡点 |
| --- | --- | --- | --- | --- |
| `tensor_core.fp8` | 1979.0 TFLOPS | 0.6977 | 1380.75 TFLOPS | 306.67 |
| `tensor_core.bf16` | 989.5 TFLOPS | 0.7680 | 759.94 TFLOPS | 168.78 |
| `tensor_core.fp16` | 989.5 TFLOPS | 0.7370 | 729.26 TFLOPS | 161.97 |
| `cuda_core.fp32` | 67.0 TFLOPS | 0.8548 | 57.27 TFLOPS | 12.72 |

带宽：4800 GB/s × 0.9380 = **4502.4 GB/s**（R2:15-18）。

### 2.3 roofline 的八种常见误读

下面每一条都由 `code/02-roofline-misuse/roofline_misuse.py` 在同一台机器、
同一个算子上量化。该算子就是本章第 6 节的案例：QK-Norm + RoPE + fp8 量化，
189.2 MFLOP / 67.76 MB，AI = 2.79 FLOP/Byte。

**M1：把 datasheet 峰值当成可达到的峰值。** H200 的 fp16 张量核 datasheet 是
989.5 TFLOPS，标定后是 729.26 TFLOPS（持续值只有 0.6684，受 700 W 限制；
burst 值 0.7370）。用 datasheet 算，compute time 被低估 **1.36×**。
这不是“厂商虚标”：同一个 `h200.yaml` 里，HBM 的标定值是 0.9380，
而纯 copy 混合只有 0.8183，**标定是按访问类型分别测的**。

**M2：用张量核的 roof 给逐元素算子定价。** 一个 norm/RoPE/quantize 的 kernel 跑在
CUDA core 上，不是张量核上。用 `tensor_core.fp16` 定价，平衡点从 **12.72 变成
161.97（12.7×）**。在当前这个算子上两种读法都得“memory-bound”，所以看起来很无害；
但只要逐元素的工作量再重一点（长归约、激活链），这个 12.7 倍的偏移就足以让判类翻转，
把你送去优化错误的东西。

**M3：按输入 dtype 声明 roof。** 一个 fp8 的 attention，它的 fp8 只是**存储**格式；
它的 QK/PV 收缩可能跑在 fp16 或 bf16 上，softmax 一定跑在 CUDA core 上。
同一份代码，三个上限：306.67 / 161.97 / 12.72。TileSight 论文的摘要正是从这个缺口
开始的：现有工具只能给“coarse roofline bounds”（R4）。

**M4：用一个 stage 的 AI 代表融合后的整算子。** 同一个算子，写成两个 kernel 是
134.87 MB（AI 1.40），融合成一个 kernel 是 67.76 MB（AI 2.79）。
拿拆分版的 AI 去评判融合版，会**低估它的算术强度 2.0 倍**，进而把后面每一个判据
都往“memory-bound”的方向偏。账本是按**调用**定义的：输入各读一次、输出各写一次、
中间量不计，调用次数一变，AI 就变。

**M5：把超越函数按 1 FLOP 记账。** 账本口径如此（第 2 章规则 2），但硬件不是。
本 corpus 里唯一一份可以引用的 ops/cycle 表在昇腾侧
（`_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.h:57-68`）：

```text
mul / add / sub / min_max / cmp / logic     51 ops/cycle
div / mod                                   13 ops/cycle
special_func（rsqrt / exp 等）               8 ops/cycle
```

一个超越函数是**乘法的 6.38 倍成本**。把它代进一个 8192 宽的 softmax 行
（每 score 5 个操作、其中 1 个是 `exp`）：不称重是 0.8k cycles，按 pipe 称重是
1.7k cycles，**2.08×**。TileOPs 的文档把这条写得很直白：这个度量
“does not certify an SFU-bound kernel as at its limit”（`roofline.md:34`）。
注意：NVIDIA 侧在本 corpus 里**没有**等价的 ops/cycle 表，所以本章不给 NV 侧的倍数。

**M6：只数 payload，不数查表。** 案例算子里，norm 权重与 RoPE 的 cos/sin 表合计
4.19 MB，占理想流量的 **6.2%**，看起来可以忽略。但设计一旦把它们按“每行读一次”
来写，这 4.19 MB 会变成 251.66 MB，占实际流量的 **79.8%**（第 6 节）。
**小数据不是问题，被反复读的小数据才是问题。**

**M7：在模型没有抓手的尺度上读数字。** 同一个 kernel body，在 4096 token 的 prefill
shape 上 SOL 是 15.051 µs，在一个 decode step（1 token、40 行 128 元素）上 SOL 是
**0.003 µs**。TileOPs 把这种行标成 latency-bound：“`sol_time` and measured time
both below the latency floor: launch overhead dominates and the model has no
traction”（`roofline.md:40-41`）。当整个算子的 SOL 只有几百纳秒时，
“效率”是错的指标，该看的是**发射次数**和**最小能填满设备的网格**。

**M8：把 >100% 的效率当成好 kernel。** TileOPs 的判读分档是
“≥80% 视为到顶、>105% 视为 anomaly”，并且每一行隐含的速率都会与 roof 的
**理论**上限比较；越界在物理上不可能，所以那是**公式错误**，不是性能
（`roofline.md:156-163`）。最常见的成因就是少算了一行字节，比如忘了 scale 张量
（案例里 0.655 MB）。

### 2.4 多层 roofline：拐点在哪里

roofline 还有一个不常被提到但很有用的用法：**用平衡点反推“从多大规模开始，
调 tile 才有意义”。**

第 14 章 1.2 节已经做过一次：fp16 方阵 GEMM 的 AI ≈ 2M/(3b) 随边长线性增长，
代入平衡点就能解出“M 大于多少才跨过平衡点”。同一招可以套在本章的算子上。
`code/01-op-ledger` 的 `--sweep` 给出了一整张翻转表：

| 算子 | AI | 在 `tensor_core.fp16` 下的判类 | 在 `cuda_core.fp32` 下的判类 |
| --- | --- | --- | --- |
| gemm 4096³ fp16 | 1365.33 | compute | compute |
| gemm 512³ fp16 | 170.67 | compute | compute |
| gemm 128³ fp16 | 42.67 | memory | **compute** |
| attention 32×4096×4096 d128 | 2068.12 | compute | compute |
| attention 32×1×8192 d128 | 1.01 | memory | memory |
| rmsnorm 8192×4096 | 1.00 | memory | memory |
| 元素链 n=16.8M（融合） | 0.50 | memory | memory |
| 案例算子 | 3.10 | memory | memory |

第三行就是 M2 讲的那件事在一个真实形状上发生：`gemm 128³` 在张量核 roof 下是
memory-bound，在 CUDA-core roof 下是 compute-bound。**它的真实身份取决于你打算
用哪个单元实现它**。这正是“roof 要由设计者声明”的含义。

---

## 3. 第二步：tile 级解析模型

手算的账本粒度是“整个算子”。要往下走一步，就必须能回答“**这个 tile 配置**会怎样”。
这一类模型统称 tile 级解析性能模型。本节讲四种，并说清各自能答什么。

### 3.1 TileSight：把 tile 从编程原语提升为分析原语

TileSight 的自我定位写在它的论文摘要里（
`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2607-22432.md`）：

> programmers still rely on **coarse roofline bounds, opaque ML predictors, or
> post-hoc profilers** to understand kernel execution. … We present TileSight, a
> tile-centric performance-modeling tool that elevates the tile from a
> programming primitive to an **analysis primitive**.

它的结构分三层，三层共用 tile 这个抽象：

| 层 | 建模对象 | 机制 |
| --- | --- | --- |
| intra-tile（核内） | 计算与访存的流水重叠 | 把工作表达成跨 network / memory / compute 三条 pipeline 的**资源向量** |
| inter-tile（核间） | cache 层级 | 调度有依赖和有序关系的动作以暴露合法重叠；从 **tile 复用距离**推断多级 cache 命中率 |
| cross-device（跨卡） | 节点间通信 | 把远程张量访问映射成放置（placement），再用 alpha-beta 阶段代价路由 |

论文自报的数字：在 A100 / H200 / B200 / B6000 上，单卡 kernel 延迟预测的
pooled MAPE 是 **12.35%**；L2 命中率预测在所有 GPU 上与实测相差约 **1 个百分点**；
最多 32 卡时，融合的分布式 kernel 是 16.18% wMAPE，端到端 vLLM serving 是
13.52% wMAPE；在优化任务上，它选出的 tile 配置与厂商/专家基线有竞争力。

**证据边界（必须说清）**：本 corpus 里 TileSight 只有**论文摘要**
（`arxiv-2607-22432.md` 共 12 行）与**仓库 README 的转录**
（`../tilelang-corpus/12-operator-libraries/raw/github-tile-ai-tilesight.md`，73 行）。
**没有源码，也没有全文。** 所以本章只能引用它的**分层结构与自报误差**，
不能引用它的具体公式。它的 README 摘要里能确认的另有两点：
（a）模型跑在 CPU 上，`CUDA`/`HIP` 探针单独测量目标 GPU；
（b）入口形如 `sight.analyze(program, H200_SXM().set_to_microbench(), options=...)`，
结果里可以取 `result.launches["main"].kernel_body_s`。
缺料已在 `references.md` 的“待补资料”里登记。

**它解决的问题是什么**：如果你已经有 tile 形状、数据移动和软件流水的描述，
你可以在**不写代码**的情况下问“换一个 tile 会怎样”。这正是本章第 4、6 节
手推预算表的自动化版本。

### 3.2 carver：TileLang 自带的 tile 配置推荐器

TileLang v0.1.15 里有一个可以 `from tilelang import carver` 用起来的框架
（`_vendor/tilelang-v0.1.15/tilelang/carver/README.md`）。它做三件事：

1. **建模硬件。** `TileDevice` 挂着一组能力参数
   （`_vendor/tilelang-v0.1.15/tilelang/carver/arch/arch_base.py:6-27`）。
   CUDA 设备模型的实现是
   `_vendor/tilelang-v0.1.15/tilelang/carver/arch/cuda.py:139-167`，
   其中几个数值得记住：

   ```text
   smem_cap        = cuda_driver.get_shared_memory_per_block()   # 运行时查询
   compute_max_core= device.multi_processor_count                # 运行时查询
   reg_cap         = 65536                                       # :157
   max_smem_usage  = 2 * smem_cap                                # :158
   sm_partition    = 4                                           # :159
   transaction_size= [32, 128]                                   # 字节，:162
   bandwidth       = [750, 12080]                                # MB/s，:167
   ```

   注意 `bandwidth` 那两行的注释：“will be used for **recommend basic tile size**”，
   并且明确写着只有设备**之间的比例**才有意义
   （`_vendor/tilelang-v0.1.15/tilelang/carver/arch/cuda.py:163-166`，见 S3）。
   它不是一块 HBM 的可用带宽。

2. **枚举 tile 结构。** `MatmulTemplate` / `GeneralReductionTemplate` /
   `ElementwiseTemplate` / `GEMVTemplate` / `FlashAttentionTemplate` / `ConvTemplate`
   把“循环该怎么分块”参数化
   （`_vendor/tilelang-v0.1.15/tilelang/carver/template/`）。
   调用 `recommend_hints(topk=20)` 拿到 top-k 建议。

3. **按 cost model 排序。** 核心在
   `_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/default.py`。把它拆成公式：

   **traffic**（`:398-432`）。对每个节点的每个输入张量，按传播后的区域算
   `coalesced_tensor_shape(region, full_shape, 读事务元素数) × 每元素字节`
   （`roller/policy/common.py:41-52`）。对普通 GEMM，这个量最终等于
   $b\,MNK\left(\frac{1}{block\_N}+\frac{1}{block\_M}\right) + b\,MN$，
   `code/03-carver-cost-model --self-test` 就是在核对这一点，误差为 0。

   **片上容量**（`:452-495`）。用一个 best-fit 分配器
   （`roller/bestfit.py:22-61`，32 字节对齐）把每个节点要缓存张量的 footprint
   拼起来，得到 `smem_cost`。

   **寄存器**（`:564`）：

   ```text
   reg_usage = 2 * max over nodes of ( prod(tile) * dtype_bits / 32 )
   ```

   这是一个**块级**的累加器占用量代理：`prod(tile)` 是输出块的元素数，
   乘上位宽再乘 2。位宽取哪个 dtype 取决于 `node.get_dtype()` 在真实输入上返回
   什么（累加 dtype 还是输入 dtype），本章的复算取 fp32 累加，并把这一点登记为
   假设，见 `code/03-carver-cost-model/README.md` 的“已知局限”。

   **驻留与波数**（`:556-574`）：

   ```text
   if smem_cost > smem_cap or reg_usage > reg_cap: 该候选直接丢弃
   block_per_SM = min(max_smem_usage // smem_cost,
                      reg_cap        // reg_usage,
                      sm_partition)
   num_wave     = ceil(grid_size / (block_per_SM * compute_max_core))
   ```

   **排序键**（`:111-112`）：

   ```text
   priority = (traffic + 1) * num_wave
   ```

   也就是“搬得少、波数少”的候选优先。搜索本身是最优先遍历
   （`dfs_smem_tile`，`:96-134`），已访问 tile 数量上限 2000。

`code/03-carver-cost-model/carver_cost_model.py` 把这套算术在 CPU 上复算了一遍。
在 4096³ fp16 / fp32 累加、132 SM、227 KB shared 的设定下：

```text
candidates visited : 729   valid : 154
rejected on regs     :  122
rejected on smem     :  453
model priority values in use : 48
ties at the top     : 8
```

三件事直接读出来：

* **寄存器比 shared memory 更早成为瓶颈**：122 个候选被 `reg_usage > reg_cap` 丢掉。
* **模型对 `block_K` 没有分辨率**：`128×256×16` 与 `128×256×128` 的 traffic 与
  `priority` 完全相同。所以“先定 M/N，再定 K”不只是经验，是这套 cost model 的结构。
* **模型会给出大量并列**：154 个可行候选只产出 48 个不同的 `priority`，前 8 名并列。
  第 11 章 3.9 节说“并列就是该去测量的信号”，这里给出了它的量级。

**carver 的定位是“排序”，不是“定案”。** 它挡掉明显不可行的候选、给出一个起点集合；
它不替代测量。还有一个现实约束：**carver 在没有 GPU 的机器上用不起来**。
`CUDA.__init__` 会调用 `tvm.runtime.cuda(0)` 并在没有设备时抛
`RuntimeError("Cannot find cuda device 0.")`
（`_vendor/tilelang-v0.1.15/tilelang/carver/arch/cuda.py:144-146`），
而 `auto_infer_current_arch()` 在没有 CUDA / MPS 时会退到 CPU 模型
（`.../carver/arch/__init__.py:47-56`），CPU 模型只填了平台名
（`.../carver/arch/cpu.py:12-19`），没有 `smem_cap`。这就是本章用复算脚本而不是
直接调用 carver 的原因。

顺带一提，carver 不只是离线工具：
`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:403` 会在需要
设置 L2 persistence 时调用 `tilelang.carver.arch.driver.get_persisting_l2_cache_max_size`。

### 3.3 `tilelang.tools.Analyzer`：一个必须知道边界的工具

v0.1.15 自带一个从 TIR 直接估 FLOPs 与访存的工具
（`_vendor/tilelang-v0.1.15/tilelang/tools/Analyzer.py`）。用法是：

```python
from tilelang.carver.arch import CUDA
from tilelang.tools import Analyzer

tir = kernel.get_tir(block_M=128, block_N=128, block_K=32)
result = Analyzer.analysis(tir, CUDA("cuda"))
# result.total_flops / total_global_bytes / expected_tflops / estimated_time
```

它的实现很短，因此边界也很清楚。逐条读源码：

* `total_flops` **只统计 `tl.gemm` / `tl.tileop.gemm`**
  （`Analyzer.py:90-106,148-151`）。逐元素算术、`T.reduce_*`、`T.Parallel` 里的运算
  一律计 0。
* `total_global_bytes` **只统计跨 global 边界的 `tl.copy` / `tl.tileop.copy`**
  （`:58-88`）。直接读写 global buffer 的语句不在内。
* 算力表只有三个条目（`:10`）：

  ```python
  ARCH_CONFIGS = {"80": (128, 1.41, 2, 108),
                  "86": (128, 1.70, 2, 84),
                  "89": (128, 2.52, 2, 128)}
  ```

  `get_peak_tflops` 用 `device.compute_capability[:2]` 查表，查不到就返回 `None`
  （`:187-195`），而 `calculate` 在 `peak_tflops is None` 时把
  `estimated_time` **静默退化成只有访存时间**（`:201-204`）。
  换句话说：**在 Hopper（90）和 Blackwell（100/103）上，这个工具不会报错，
  它只会给你半个答案。**
* 那三个条目的第三个字段是 `flops_per_cycle = 2`，也就是**每 cycle 两次浮点运算**，
  这是 CUDA core 的 FMA 口径，不是张量核口径。官方文档自己写了：
  “Treat `expected_tflops` and `estimated_time` as comparison aids, not
  device-specific benchmark results”
  （`_vendor/tilelang-v0.1.15/docs/tools/analyzer.md:110-112`）。
* `expected_bandwidth_GBps` 直接来自 carver 的设备模型
  （`Analyzer.py:198`：`self.device.bandwidth[1] / 1000`）。用 CUDA 模型
  （`.../carver/arch/cuda.py:167`）代进去就是 **12.08 GB/s**。这个数只有设备间比较的意义，
  拿它去除字节数得到的绝对时间没有物理含义。

官方文档的“Limitations”一节
（`_vendor/tilelang-v0.1.15/docs/tools/analyzer.md:98-112`）把这些都写清楚了。
本章要在它之上再加一句方法论：**一个工具的输出边界，是你在用它之前就该读的，
不是在被它的数字误导之后才去读的。** 判据很简单，先问“它统计了哪些指令”，
再问“它的峰值表覆盖了我的 compute capability 吗”，最后问
“它的带宽和我的机器是同一个量纲吗”。

对这个工具的正确用法是：**只要 GEMM-only 的 FLOPs 与跨 global 的字节数**，
拿它们去和手算的账本对账（`code/01` 的 `--case gemm` 就是同一个数）。
不要用它去估任何一个融合算子的时间。

### 3.4 昇腾侧：指令级延迟模型 + SMT 求解

昇腾侧不是“流量 × 波数”的启发式。官方 pass 序列是
（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:103-116`）：

```python
mod = ascend_transform.EstimateLatency()(mod)
mod = ascend_transform.AutoSchedule()(mod)
mod = ascend_transform.AssignCore()(mod)
mod = ascend_transform.PrepareMultiBuffer()(mod)
mod = ascend_transform.ResolveCore()(mod)
mod = ascend_transform.InsertSync()(mod)
mod = ascend_transform.MaterializeMultiBuffer()(mod)
```

`EstimateLatency` 的模型是一条**按物理 pipe 计费**的规则。注释直接写在源码里
（`_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.cc:326-340`，见 S29）：

```text
1. Memory latency: walk read/write regions, classify (src,dst) scope
   pair to a physical path, charge `base + bytes/bandwidth`.
2. Compute latency:
     Cube   -> (M*K*N) / cube_throughput_<dtype> + base
     Vector -> max(op-count via OperationCounter, bytes / valu_bandwidth)
     Scalar -> one cycle per atomic task
3. II by pipe: take the max of the bandwidth- and throughput-limited
   initiation intervals for each active physical resource.
4. Total latency = memory_latency + compute_latency
   under sequential assumption; pipeline overlap is handled by the
   scheduler (Z3) via the II we emit.
```

两处细节值得单独指出，因为它们直接决定你写的 tile 能不能被正确估价：

* **Cube 的 ops 不是 M×N×K。** `CubeOperations`
  （`estimate_latency.cc:692-707`）是
  `2 * align(M,16) * align(N,16) * align(K, k_group)`，其中 `k_group = 256 / bits`
  （fp16/bf16 为 16；fp32 在 HF32 模式下为 8，否则为 1）。乘加的 2 已经含在内，
  M 和 N 都按 16 对齐。**对齐是要花钱的**：一个 `N=200` 的 tile 按 208 计费。
* **总延迟是 `memory_latency + compute_latency`**（`:941`），
  重叠由 Z3 通过 II 处理。第 23 章 5.1 节提醒过：这个顺序假设让总延迟偏小，
  而偏小的方向是危险的方向。

常量表在 `_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.h:27-158`，
其中与 tile 尺寸关系最大的
几个：

```text
cube_throughput_fp16 = 8192 ops/cycle      cube_base_latency  = 8
cube_throughput_bf16 = 8192                cube_unit_m/k/n    = 16
cube_throughput_fp32 = 512                 cube_throughput_fp8 = 16384
aic_mte2_to_l1_bandwidth   = 100 B/cycle   mte2_gm_to_l1_base_latency = 190
mte2_gm_to_l1_min_ii       = 64 cycles     l1_to_l0a/b_bandwidth = 256
```

`code/07-ascend-latency-model/ascend_pipe_model.py` 用这些常量做了本章最有用的一次
推导：把 Cube 的 II 与 GM→L1 的 MTE2 的 II 相比，`K` 与两个因子 2 会消掉，得到一条
**纯形状判据**：

$$
\frac{MN}{M+N} \;\ge\; \frac{\texttt{cube\_throughput}}{\texttt{mte2\_bandwidth}}
   \;=\; \frac{8192}{100} = 81.92
$$

方形 tile 解出来是 **边长 ≥ 164**。脚本里的实际扫掠与之吻合：

| M=N | Cube II | MTE2 II | binding |
| --- | --- | --- | --- |
| 64 | 16 | 128 | MTE2 |
| 128 | 64 | 128 | MTE2 |
| **192** | **144** | **128** | **Cube** |
| 256 | 256 | 164 | Cube |
| 512 | 1024 | 328 | Cube |

而官方昇腾 950 的 GEMM+ReLU 例子用的正是 `block_M=256, block_N=224, block_K=128`
（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:74`）。代入模型：

```text
Cube II 1792   MTE2 II 1229   MTE1 II 256   -> binding Cube (latency/period = 1.004)
```

模型没有针对这个例子做任何拟合，它落在 Cube 侧。这是自洽性的一个旁证，
**不是**验证（见下方“模型的边界”）。

### 3.5 四种模型的定位

| 模型 | 粒度 | 输入 | 输出 | 什么时候用 | 什么时候会给出错误答案 |
| --- | --- | --- | --- | --- | --- |
| 手算账本（`code/01`） | 整个算子 | shape、dtype、算法 | AI、下界 | 决定“这个算子值不值得优化、往哪个方向优化” | roof 声明错（M2/M3） |
| TileSight（R4、R5） | tile 与 schedule | program（tile、数据移动、流水） | kernel body 延迟、cache 命中率 | 已有几个候选方案，要在不写代码的前提下排序 | 论文自报 12.35% MAPE；**本 corpus 只有摘要，无源码** |
| carver-roller（S1–S20） | tile 配置 | PrimFunc + 设备模型 | top-k 候选与排序 | 不知道从哪个 tile 开始试 | 模型无分辨率的维度上给并列（`block_K`） |
| `Analyzer`（S21–S27） | 整个 TIR | PrimFunc + 设备模型 | FLOPs、跨 global 字节数 | 给 GEMM 对账 | sm_90/100 无算力条目 ⇒ 只给访存时间；非 `T.gemm`/`T.copy` 一律不计 |
| 昇腾 `EstimateLatency`+Z3（S28–S34） | 语句 / 任务 | TIR + 标定常量 | 每任务 latency、II、开始时间 | 预测调度结果、判断哪条 pipe 是瓶颈 | 换芯片要重标常量；模型偏小方向的误差 |

**模型的边界**：`code/07` 是**教学版**。真实的 `EstimateAscend` 还要算 Vector 的
ops/cycle、`ScaleBytesByLanes` 的 lanes 缩放、每个物理路径的完成延迟，以及基于
pipe mask 的 II 合成（`estimate_latency.cc:750-970`）。本章只保留 Cube、MTE1、MTE2
三条。第 23 章讲完整的 pass。

---

## 4. 第三步：tile 尺寸怎么定

### 4.1 四条约束

tile 尺寸不是一个自由度，是四个约束的交集：

**C1 片上容量。** 做 $S$ 级流水，需要同时保留 $S$ 份输入 tile：

$$
\text{SMEM} = S \cdot b \cdot \bigl(block\_M \cdot block\_K + block\_K \cdot block\_N\bigr)
\le \text{可用片上容量}
$$

在 Hopper 上，单 block 的上限是 227 KB（每 SM 可配到 228 KB）。
**超了是编译或启动失败，不是变慢。**

**C2 寄存器压力。** 累加器必须在计算期间常驻寄存器：

$$
\text{regs/thread} \approx \frac{block\_M \times block\_N}{\text{threads}} \times \frac{\text{accum\_bytes}}{4}
$$

CUDA 最佳实践文档给了这把尺子的刻度：compute capability 7.0 的 SM 有 65536 个
32 位寄存器、最多 2048 个驻留线程，也就是**每线程 32 个寄存器才能到 100% occupancy**，
而且寄存器分配以“每 warp 256 个”为单位向上取整
（`nvidia-cuda-best-practices.md:3126`）。单线程的硬上限是 255。

**C3 指令形状。** 张量核指令有固定形状：`mma` 密集 f16 是 `m16n8k8` / `m16n8k16`
（`nvidia-ptx-isa.md:23476-23477`），`wgmma` 密集 f16 是 `m64nNk16`，N 从 8 到 256
（同文件 `:28132`），一个 warpgroup 是 4 个连续 warp 即 128 线程（`:28113-28114`）。
`block_M` 不是 64 的倍数，就用不上 `wgmma` 的 m64。昇腾侧的对应约束是 Cube 的
16×16 分形与各级对齐要求（UB 32 B、L1 32 B、L0A/L0B 512 B、L0C 64 B）。

**C4 并行度。** 输出块数除以“每 SM 驻留块数 × SM 数”就是波数。最后一波填不满，
机器就空转。第 14 章 4.1–4.2 节的 split-K / stream-K 就是为这一条准备的。

### 4.2 四条约束在两边的对照

| 约束 | NVIDIA（sm_90） | 昇腾 |
| --- | --- | --- |
| 片上容量 | shared 228 KB/SM、227 KB/block；CUDA 保留 1 KB、静态分配 48 KB 需 opt-in（R7） | L1 / L0A / L0B / L0C / UB 五块独立，各有对齐；容量随型号，需 `GetCoreMemSize` 查询（R11、R12）；L1 512 KB 与 UB 192 KB 都是**示例条件值**（R13、R14） |
| 寄存器 | 65536/SM、255/线程、按每 warp 256 个取整（R8） | 不走这条线；约束落在 L0/UB 容量与 MTE 带宽上 |
| 指令形状 | `mma` 16×8×16、`wgmma` 64×N×16、warpgroup=128（R9） | Cube 分形 16×16；对齐 UB 32 B / L1 32 B / L0A·L0B 512 B / L0C 64 B（R11） |
| 并行度 | 输出块数 / (驻留块数 × SM 数) | 常是 persistent 的一维固定块数（官方例子 `num_blocks=32`，S35），并行度由**核数**而不是 SM 数决定 |

### 4.3 决策流程：先粗算，再微调

`code/04-tile-size-decision/tile_decision.py` 把流程实现成了检查机。顺序不能换：

**第 1 步：从“容量允许的最大 tile”开始。** 因为 DRAM 流量
$b\,MNK(1/block\_N + 1/block\_M) + b\,MN$ 随 tile 增大而单调下降。
在 4096³ 的例子里，traffic 最优是 256×256（1107.3 MB），
其次是 128×256 / 256×128（1644.2 MB），再其次是 128×128（2181.0 MB）。

**第 2 步：查指令形状是否整除。** 不整除就没有意义，先改尺寸。

**第 3 步：算寄存器与驻留块数。** 256×256 + fp32 累加 + 512 线程 =
128 regs/thread，`65536 // (128 × 512) = 1`，也就是只剩 1 block/SM；
128×256 + 256 线程同样是 128 regs/thread，但每 block 一半的寄存器，
于是能驻留 2 块。这一步的输出是“谁在卡我”：

```text
block=( 128, 256, 64)  stages=3  threads=256  instr=wgmma
  C1 shared/stage     48.0 KB -> total    144.0 KB
  C2 accumulators   128/thread ->  128 regs/thread (warp-rounded),  32768 regs/block
     occupancy: regs -> 2 blk/SM, smem -> 1, threads -> 8  => 1 blk/SM, binding=smem
  C4 grid    512 tiles over   8-warp blocks -> 4 wave(s), last wave  87.9% full
```

**第 4 步：算波数与尾波填充率。** 尾波明显不满时，先考虑 split 类变体而不是继续加大 tile。

**第 5 步：只有到这一步才动 `block_K` 与 `num_stages`。** 它们不影响流量
（carver 的复算已经证明 `block_K` 在 traffic 与 priority 上完全没有分辨率），
只影响流水与占用。第 11 章 4.5 节给出了这一维的判据
（`num_stages ≥ 1 + ceil(latency / compute)`）。

### 4.4 什么时候“大 tile”对，什么时候“小 tile + 多波”对

这不是风格问题，是两条约束谁先撞到的问题：

| 情形 | 该往哪走 | 判据 |
| --- | --- | --- |
| 算术强度高（大 GEMM、prefill attention） | 大 tile | 流量是主项，且 `blocks/SM` 只要 ≥ 1 就能靠 tile 内并行拉开 |
| 第一波就填不满设备（块数 < SM 数） | 先加并行度，必要时 split | 小块再大也填不满 |
| 属于“少量块 × 深 K”（瘦长 GEMM） | split-K | 第 14 章 4.1 节 |
| 最后一波明显不满 | stream-K 或调整分块 | 第 14 章 4.2 节 |
| 逐元素 / 归约类，AI 远低于平衡点 | tile 只受容量与向量宽度约束，**不要**为了流量去加大 tile | 流量由“读几次”决定，不由 tile 尺寸决定 |

最后一行是本章与第 14 章最重要的区别：**第 14 章那套“让 tile 越大越好”的表，
只对 compute-bound 的算子成立。** 对一个 AI ≈ 3 的算子，加大 `block_N` 不会减少
任何一次访存，只会让寄存器先爆掉。

---

## 5. 第四步：fusion / split / 特殊指令

### 5.1 融合的账本：四个问题

融合的收益只有一个来源：**少搬一趟中间量**。省下的字节数就是中间量的大小乘 2
（原来要写一次、读一次）。但融合要付出三样代价，每一样都可能把它否掉。
所以判据是四个问题（`code/05-fusion-split/fusion_split.py`）：

1. **融合省多少字节？** 若中间量只有几百 KB 而输入输出是几十 MB，省的可以忽略。
2. **融合后要同时活着多少片上数据？** 中间结果必须留在寄存器或片上，
   不能落回 HBM。否则等于没融合。
3. **融合会不会把 grid 缩到装不满设备？** 两个各自有 10 万块的逐元素算子，
   融合后可能只剩几千块。
4. **融合引入了什么新工作？** 额外的输出、额外的归约、额外的同步。

第 3、4 条在实际工程里最常被忽略。`code/05` 的六个取舍里，有两个是**故意**让搬运
更多字节的一方胜出：

```text
decision                                        A MB      B MB  delta MB    A grid    B grid  winner
skinny GEMM 1024x1024x16384, split-K=8           4.2      71.3     -67.1        64       512  split-K=8
decode attention, sq=1, skv=128k, split-KV=64    2147.5    2149.6      -2.1        32      2048  split-KV=64
```

两行的原因相同：**64 块和 32 块都填不满 132 个 SM**，而多搬 67.1 MB / 2.1 MB 换来
512 / 2048 块。判据必须写出来，就是因为它在不同的输入下会给出相反的答案。

### 5.2 什么时候必须拆

和 5.1 对称，拆的判据是“融合的哪一条代价被突破了”：

* **片上容不下。** 这是硬约束。典型是“把整个 attention 融进一个 kernel”，
  当 KV 长度让 `N×N` 的 score 矩阵放不下时，就必须回到 online softmax 的分块形态
  （第 15 章 2.5 节）。
* **并行度被压到设备以下。** 此时要么拆，要么用 split 类变体把并行度买回来。
* **需要两种完全不同的执行体。** 昇腾上 Cube 与 Vector 是两块执行体；
  NV 上张量核与 CUDA core 也在争同一份寄存器。跨执行体的融合收益要重新算一遍。
* **中间的数值精度必须回落到内存。** 例如需要把 fp32 中间量存成 fp32 才能保证
  数值正确性，那就必须写出去，融合的意义大打折扣。

### 5.3 特殊指令与特殊布局的判据

默认降级路径（`T.copy` → 普通访存、`T.gemm` → `mma`）先试。只有出现下面这类
**可观测的**信号时，才引入特殊路径：

| 信号 | 说明默认路径撞到了什么 | 可能的手段 |
| --- | --- | --- |
| shared memory 的 bank conflict 高 | 布局与访存模式不匹配 | `T.use_swizzle` / `T.annotate_layout`（第 21 章） |
| 搬运指令占比高、计算单元空转 | 搬运没有被重叠 | TMA（Hopper）/ `T.Pipelined` 多缓冲（第 22、24 章） |
| 累加器寄存器爆掉 | 累加必须落在同一处 | Blackwell 的 `T.alloc_tmem` + `T.tcgen05_gemm`（第 24 章） |
| 布局转换出现在热路径上 | 两个 op 要的布局不同 | 提前 `T.annotate_layout` 统一（第 21 章） |
| 昇腾上 Cube 与 Vector 互相等 | 核间没有流水 | 把两者放进同一个 `T.Pipelined`，或用 `dual_copy`（第 25 章） |

**反过来说：没有信号就不要用。** 特殊路径都会引入新的约束（对齐、形状、同步），
而这些约束会在下一个 shape 上咬人。第 14 章 2.5 节的官方 intrinsic 例子自述是
“a debug config”，就是这条规则的注脚。

### 5.4 昇腾与 NVIDIA 的方法论差异

同一个算子、同一套判据，在两边会得出不同答案。原因有三条，都能在锁定源码里看到。

**差异一：容量的形状不一样。** NV 侧是“一个 shared + 一个寄存器堆”，
tile 尺寸是“一个数放不放得下”。昇腾侧是 L1 / L0A / L0B / L0C / UB 五块独立存储，
每块有自己的容量与对齐（对齐要求：UB 32 B、L1 32 B、L0A 512 B、L0B 512 B、
L0C 64 B、BiasTable 64 B、Fixpipe 64 B，来源见 R11）。于是“tile 尺寸”在昇腾上
其实是“每条数据住在哪一块、按什么对齐”。

**差异二：执行体的数量不一样。** NV 上一个 block 里的线程是一个 SIMT 整体。
昇腾上一个 AI Core 是 **1 个 Cube + 2 个 Vector**。这直接改变“融合”的定义：
在 NV 上把 epilogue 融进 GEMM 是省一趟 HBM；
在昇腾上还多一层意思，**用 Vector 的时间盖住 Cube 的时间**。
官方 GEMM+ReLU 例子就是 `T.dual_copy(C_l0c, C_ub)` 之后在 `T.SimtVF(threads=128)`
里做 ReLU，再 `T.dual_copy` 写回
（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:62-104`，见 S35）。前面第 15
章 4.5 节也讲过 attention 上“`QKᵀ` 的结果必须绕一趟”。

**差异三：同步的单位不一样。** NV 上是线程屏障加异步拷贝自带的等待，用户几乎不写。
昇腾上是 **pipe 到 pipe 的事件**（Set/Wait flag），`InsertSync` 统一插入成对 flag
（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:113`，见 S34）。
所以在昇腾上加一条新数据流，
要问的问题不是“要不要加 barrier”，而是“这条依赖值不值得一次 flag 往返”。

**这三条差异如何改变设计选择**，可以用本章第 6 节的案例看：

| 决策 | NV 侧的理由 | 昇腾侧的理由 |
| --- | --- | --- |
| 要不要融合这三步 | 省 83.9 MB 的往返 | 同样省 83.9 MB，而且省掉两次 kernel 之间的同步 |
| tile 取多大 | 让 `blocks/SM` 至少为 1，同时让流量接近下界 | 让 GM→UB 的搬运能盖住 Vector 的计算，同时 UB 装得下 |
| 需不需要特殊指令 | 默认 `T.copy` + `T.gemm` 先试 | 默认路径已经含 `dual_copy` / `SimtVF` 的编排，要先确认它没被绕过 |

---

## 6. 第五步：把一个融合算子从头设计一遍

本节把前面五节走一遍。所有数字来自 `code/06-case-norm-rope-quant/`，
可以在任意机器上复算。

### 6.1 问题定义

推理侧的 attention 前处理，每层每个 token 都要跑：

```python
q, k = proj(x)                      # bf16, [tokens, heads, head_dim]
q, k = rmsnorm(q, k)                # 沿 head_dim，每个 (token, head) 一次
q, k = rope(q, k)                   # 同一 token 的所有 head 共用一张 cos/sin 表
q8, k8, scale = fp8_quant(q, k)     # 每个 (token, head) 一个 fp32 scale
```

参数：`tokens=4096`、`q_heads=32`、`kv_heads=8`（GQA）、`head_dim=128`、
输入 bf16、输出 fp8 e4m3。

### 6.2 预算表

**第一步，账本（这部分与设计无关）：**

```text
rows (token, head) 对    : 163840
元素数                   : 20971520
flops                    : 210.2 MFLOP
理想字节                 : 67.76 MB  （payload 63.57 + 查表 4.19）
算术强度                 : 3.102 FLOP/Byte
访存下界 @4502.4 GB/s    : 15.051 us
```

**判类**：3.102 远低于 `cuda_core.fp32` 的平衡点 12.72，更远低于张量核的 161.97。
这个算子是 memory-bound，而且是**极端** memory-bound。结论直接出来：

> 不要为它引入张量核，不要为它调 `num_stages`。
> 唯一值得做的是**减少它实际搬运的字节数**。

**第二步，tile 尺寸。** 因为它是逐行的归约 + 逐元素变换，tile 的 C1/C2 只受
“一个 block 要同时活着多少行”约束，不受指令形状约束（不涉及张量核）。
所以这一节的问题变成：**一个 block 处理几行？**

**第三步，流水。** 每行的归约依赖整行的数据，所以“先读完整行再算”是天然的
两级结构。这在第 11 章的框架里就是 2 级流水：搬运本行与计算上一行重叠。
不需要更多级，因为下一行不依赖上一行的结果。

**第四步，特殊指令。** 这里没有任何一条 5.3 节的信号。默认路径就是答案。

### 6.3 第一版：一行一个 block

最直觉的写法：grid = 163840，每个 block 处理一个 (token, head) 的 128 个元素，
`threads=128`，每个线程一个元素，在 UB/shared 里做一次树形归约求平方和，
然后读 RoPE 表、旋转、求 absmax、量化、写回。

它的预算表长这样：

```text
design                     rows/blk    grid  waves  on-chip KB  payload MB  lookup MB  total MB  vs ideal
A: one row per block              1  163840    156         1.9       63.57     251.66    315.23     4.65x
```

逐项看：

* **片上占用只有 1.9 KB**，离任何容量上限都远。这一版**不是**被 C1 卡住的。
* **片上放得下、形状也没有问题**，所以 C1/C3 这两条最常见的约束在这里都不是问题。
* **总共要搬 315.23 MB，是理想值的 4.65 倍。**

这个数字就是这一版被否掉的全部理由。

### 6.4 性能诊断

诊断的关键是**把超出的部分拆开**：

```text
A: one row per block
    total  315.23 MB = payload  63.57 MB + lookup  251.66 MB   (lookup is  79.8% of all traffic)
    over the ideal by  247.46 MB  -> the achievable time floor rises from 15.05 us to 70.01 us
```

三个结论，一个比一个具体：

1. **payload 完全正确**：63.57 MB，与理想值一字不差。也就是说数据的读、写、scale
   都数对了。
2. **多出来的 247.46 MB 全是查表**：norm 权重（每行读一次 q 和 k 的两份 128 元素权重）
   加上 RoPE 表（每行读一次 128 个 cos 与 128 个 sin，fp32）。
3. **因此下界从 15.05 µs 抬到 70.01 µs。** 这一版的实测时间无论怎么调，都注定在
   70 µs 这个量级之上，**调参救不了它，只有改设计能救它**。

如果手上有一台有卡的机器，这里该做的验证是：

* 用 `Analyzer` 拿 GEMM-only 的对照（这一步不适用于本算子，因为它没有 `T.gemm`），
  用第 12 章的口径拿实测时间与实测 DRAM 流量；
* 检查实测 DRAM 流量是不是接近 315 MB。**如果是**，模型的诊断成立，
  而且它同时说明 L2 没有兜住这些查表，163840 个 block 之间的复用距离太远；
* **如果实测流量接近 68 MB**，说明 L2 全部兜住了，那么这一版的真实问题在别处
  （描述符开销、发射次数、尾部效应），诊断要换一条路走。

这正是“模型告诉你去看哪里，测量告诉你看到的是不是真相”。

### 6.5 第二版

**改动的唯一目标是让查表被摊开。** 具体做法：一个 block 处理**连续的若干行**，
并且**让这些行落在同一个 token 的同一个 head 组内**。这样 norm 权重只读一次、
RoPE 表只读一次，整组行共享。

两个候选：

```text
B: 8 rows per block               8   20480     20         8.0       63.57      31.46     95.03     1.40x
C: one whole token per block     40    4096      4        36.2       63.57       6.29     69.86     1.03x
```

* **B（8 行一 block）**：grid 从 163840 降到 20480，查表从 251.66 MB 降到 31.46 MB，
  总流量 95.03 MB，是理想值的 1.40 倍。片上只要 8 KB。
* **C（一整个 token）**：一个 block 处理 32 个 q head 加 8 个 kv head，共 40 行。
  grid 降到 4096，查表只剩 6.29 MB，总流量 69.86 MB，**只比理想值高 3%**。
  片上 36.2 KB，仍然远在预算之内。

选 C，理由是三条账都过：流量 1.03×、片上 36.2 KB、grid 4096（远大于 132 SM）。

### 6.6 结账：两版之间到底发生了什么

| | 第一版 A | 第二版 C | 变化 |
| --- | --- | --- | --- |
| 一个 block 的行数 | 1 | 40 | 按 token 分组 |
| grid | 163840 | 4096 | ÷40 |
| 片上占用 | 1.9 KB | 36.2 KB | ×19（仍宽裕） |
| 总流量 | 315.23 MB | 69.86 MB | ÷4.5 |
| 相对理想值 | 4.65× | 1.03× | — |
| 下界（4502.4 GB/s） | 70.01 µs | 15.52 µs | ÷4.5 |

**第二版没有换算法、没有换机器、没有调任何参数，只改了一个数：一个 block 处理几行。**
而这个数是从 6.4 节的诊断里直接读出来的，不是试出来的。

补两条这一版仍然存在的风险，用来提醒“预算表不是保证”：

1. **40 行一 block 意味着一个 block 要索引两套 norm 权重**（q 一套、k 一套）
   和一张按 token 索引的 RoPE 表。如果实现里把这个索引写错，正确性会立刻崩，
   而且症状是“某些 head 的输出不对”这种最难查的形态。
2. **grid 只有 4096，尾波与负载均衡变得重要。** 4096 块在 132 SM 上，
   每 SM 驻留 8 块时是 4 波，尾波填充 4096/(132×8)=3.88 → 最后一波约 88%。
   再加大行数会把这一项继续推高收益，但同时减少并行度。**这就是 5.1 节第 3 个
   问题在真实案例里的形态。**

---

## 7. 反模式清单

下面每一条都能在本章前面找到定量依据。把它们当成设计评审时的检查项。

1. **先写代码再调参。** 如果一个算子的 AI 比平衡点低两个数量级，调参空间本来就不存在
   （第 1.1 节、`code/01`）。**先算。**
2. **用 datasheet 峰值算效率。** 在 H200 上会把 compute time 低估 1.36×（M1）。
3. **用张量核的 roof 给逐元素算子定价。** 平衡点差 12.7×（M2）。
4. **按输入 dtype 声明 roof。** fp8 是存储格式，不是计算单元（M3）。
5. **用一个 stage 的 AI 代表整个融合算子。** 会把 AI 读成一半（M4）。
6. **把超越函数当成和乘法一样便宜。** 昇腾上差 6.38 倍，softmax 行整体差 2.08 倍（M5）。
7. **只算 payload，不算查表 / scale / 索引。** 案例里查表从 6.2% 变成 79.8%（M6）。
8. **在小到没有意义的尺度上读效率。** decode shape 的 SOL 可以只有 0.003 µs（M7）。
9. **把 >100% 的效率当成好 kernel。** 那是公式错误（M8）。
10. **忽略 epilogue 成本。** 融合的收益要减去额外输出、额外归约、额外同步
    （5.1 节的第 4 个问题）。写出去再读回来的中间量，等于没融合。
11. **tile 太大导致 occupancy 崩掉。** 256×256 + fp32 累加 + 512 线程 = 128 regs/线程，
    只剩 1 block/SM；再大一点（256×256 累加器 = 256 regs/线程）直接越界（4.3 节）。
12. **反过来，tile 太小导致每 block 只搬几百字节。** 描述符与发射开销成为主项，
    而且 L2 复用距离变远（案例的第一版）。
13. **只在 `M×N` 上调，不知道 `block_K` 对流量没有分辨率。** carver 的复算显示
    154 个可行候选里前 8 名并列（3.2 节）。**先定 M/N，再定 K。**
14. **把模型的输出当成判决。** `code/07` 里 Cube 的 8 cycle 完成开销就把 stage 公式
    从 2 顶到 3，而官方例子用的是 2。模型给判据，测量给结论。
15. **忽略同步的成本。** 昇腾上是 pipe 到 pipe 的 flag 往返；给每条依赖都加一次同步，
    流水就被切碎了（5.4 节）。
16. **在需要特殊指令之前就用特殊指令。** 没有 5.3 节的信号就不要引入新约束；
    特殊路径的约束会在下一个 shape 上咬人。

---

## 8. 本章小结

把这一章压缩成六句话：

1. **顺序是“建模 → tile → 流水 → 特殊路径 → 代码”。** 代码是第五步，
   而第 ① 步的产物（AI 与平衡点）决定后面四步值不值得做。
2. **roofline 是判类工具，不是预测工具。** 它的三个输入都要自己声明：
   哪些字节算进去、用哪条 roof、哪个 shape。声明错了，后面的每一步都会错。
3. **tile 级解析模型有三个层次。** 手算账本告诉你“往哪个方向优化”；
   TileSight / carver 这一类模型告诉你“哪几个候选值得试”；
   编译器内建的估计器（`Analyzer`、`EstimateLatency`+Z3）告诉你
   “编译期已经替你算了什么”，而它们的边界必须逐条读源码确认。
4. **tile 尺寸是四条约束的交集**：片上容量、寄存器、指令形状、并行度。
   超了会直接失败或直接掉性能，所以要在写代码之前算。
   顺序是先 `M/N` 再 `block_K`。
5. **fusion / split 的判据是四个问题**：省多少字节、要多少片上、会不会缩掉并行度、
   引入什么新工作。有四条互斥的答案，所以判据必须写出来。
   同样的判据在 NV 与昇腾上会得出不同答案，因为容量形状、执行体数量与同步单位都不同。
6. **模型和测量各管一半。** 模型告诉你“差多少、差在哪一条约束上”，
   测量告诉你“实际差的是不是这一条”。案例里第一版到第二版的 4.5 倍差距，
   完全来自模型诊断出的“查表被重复读了 40 次”，而不是来自任何一次调参。

---

## 延伸阅读

* **R1**，roofline 的规范定义与它的诚实性规则：
  `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md`
  尤其是 §1.2、§1.4、§4.3、§5.1。本章 2.2–2.3 节的判据大量引用了它。
* **R2**，机器参数：`.../TileOPs/src/tileops/perf/profiles/h200.yaml`。
  换机器时从这里换起；R3（`h20_3e.yaml`）是“profile 里会缺 roof”的现成反例。
* **R4、R5**，TileSight 的论文摘要与仓库 README。完整论文与源码在本 corpus 中缺失，
  见 `references.md`“待补资料”第 1 条。
* **S1–S20**，carver 的源码：`_vendor/tilelang-v0.1.15/tilelang/carver/`。
  从 `roller/policy/default.py` 的 `compute_tile_dict` 开始读最省力。
* **S21–S27**，`tilelang.tools.Analyzer` 的实现与限制：
  `_vendor/tilelang-v0.1.15/tilelang/tools/Analyzer.py` 与
  `_vendor/tilelang-v0.1.15/docs/tools/analyzer.md`。
* **S28–S35**，昇腾的指令级延迟模型：
  `_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.h` 与
  `estimate_latency.cc`，以及 `tilelang/ascend/README.md` 的官方例子；
  第 23 章是它的完整讲解。
* **R11、R12**，昇腾的对齐表与容量查询接口。
* **本章七个可复算示例**：`code/README.md`，以及
  `code/06-case-norm-rope-quant/` 的案例。
