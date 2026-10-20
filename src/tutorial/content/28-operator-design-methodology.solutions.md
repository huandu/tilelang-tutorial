# 第 28 章 习题答案

每题先写**解题思路**，再给**参考答案**，最后列**常见错误**。
所有数字都可以用 `code/` 下的脚本复算；文中给出的都是脚本的实际输出。

---

## E1

**思路**：流程题的关键不是背顺序，而是说清顺序背后的依赖关系。

**参考答案**：五步是

```
① 建模（(flops, bytes) → AI → 平衡点 → 判类）
② 定 tile 尺寸与存储位置（四条约束）
③ 定流水结构（延迟/周期比 → 要不要多缓冲、几份）
④ 定是否需要特殊指令 / 特殊布局
⑤ 写代码（并把 ①–④ 的结论变成参数）
```

② 必须在 ③ 之前，因为**能加多少 stage 是 tile 尺寸的函数**。第 11 章 4.5 节给了
这条关系的具体形态：流水相关的片上占用上界是
`num_stages × (A_tile + B_tile)`。`128×128×32` 的 tile 每级 16 KB，
4 级流水只要 64 KB；`256×256×64` 的 tile 每级 64 KB，4 级流水要 256 KB，
**直接超过 227 KB 的单 block 上限**。

具体反例：先按「流水越深越好」把 `num_stages` 定成 4，再回头把 tile 调到
`256×256×64`——这时编译直接失败，而失败原因不是那一行代码，是两小时前的顺序错了。

**常见错误**：把顺序说成「tile 和流水可以互相调」。它们不是对等的：
tile 决定占用**量级**，`num_stages` 只是在剩余空间里分配，量级定错后面怎么调都不对。

---

## E2

**思路**：先把三个输入列出来，再回答「为什么不能自动推断」——要同时说清
「不能从运行中的 kernel 推」和「不能从 dtype 推」两个方向。

**参考答案**：三个输入是

1. `bytes`：这个算法最少要搬多少字节（每个输入读一次、每个输出写一次、中间量不计）；
2. `flops`：这个算法最少要算多少次（矩阵乘的乘加记 2，其余按账本口径）；
3. roof：用哪个单元的理论峰值，以及对应的带宽。

TileOPs 的规范（`roofline.md:63-70`）要求 roof key「state the unit an **optimal**
implementation would use, declared by the op author in code」，并且明确禁止两种推断：

* **不能从运行中的 kernel 推断**：「that would price a kernel on the wrong unit
  against the wrong ceiling and hide exactly the gap the metric exists to expose」。
  一个跑在 CUDA core 上、本该用张量核的 GEMM，会被它自己当前的选择「洗白」。
* **不能从 dtype 推断**：「an fp8-backend attention takes fp16/bf16 tensors」。
  存储格式不是计算单元。

**常见错误**：把 roof 说成「由硬件决定」。硬件只决定有哪些 roof 可用；
选哪一条是**对算法的判断**——同一个算子换实现，roof 就该换。

---

## E3

**思路**：三步：拿 theoretical × calibration 得 effective，再用 effective/带宽。

**参考答案**：

```
cuda_core.fp32  theoretical = 67.0e12 FLOPS     (h200.yaml:20-21)
                calibration = 0.8548            (h200.yaml:22-26)
                effective   = 67.0e12 × 0.8548 = 57.27e12 FLOPS
hbm             theoretical = 4800e9 B/s        (h200.yaml:15)
                calibration = 0.9380            (h200.yaml:16-18)
                effective   = 4800e9 × 0.9380 = 4502.4e9 B/s
平衡点 = 57.27e12 / 4502.4e9 = 12.72 FLOP/Byte
```

用 `tensor_core.fp16`（effective 729.26 TFLOPS）得到的平衡点是 161.97，
是前者的 **12.7 倍**。

案例算子的 AI 是 3.102，两种读法都得 memory-bound，所以这个错误在这里看起来无害。
但 `code/01` 的 `--sweep` 给出了它真的会翻案的形状：`gemm 128³` 的 AI 是 42.667，
在 `tensor_core.fp16` 下判 memory-bound，在 `cuda_core.fp32` 下判 compute-bound。
价格估错 12.7 倍，足以让判类翻转。

**常见错误**：只用 theoretical 不算 calibration（那就是习题 M1 的错误），
或者忘了带宽也要乘 calibration（0.9380 是四种访问混合的**包络**，
纯 copy 只有 0.8183）。

---

## E4

**思路**：读两段代码：算力表第 10 行，和 `calculate` 里查表失败的分支。

**参考答案**：

```python
# Analyzer.py:10
ARCH_CONFIGS = {"80": (128, 1.41, 2, 108), "86": (128, 1.70, 2, 84), "89": (128, 2.52, 2, 128)}

# Analyzer.py:187-195
arch_key = device.compute_capability[:2]
if arch_key not in ARCH_CONFIGS:
    logger.info(...)
    return None            # <- 注意是 info 日志，不是异常

# Analyzer.py:201-204
compute_time = self.total_flops / (peak_tflops * 1e12) if peak_tflops else None
estimated_time = max(mem_time, compute_time) if peak_tflops else mem_time
```

1. Hopper 的 `compute_capability` 以 `90` 开头，不在表里，所以
   `result.expected_tflops is None`。
2. `estimated_time = mem_time`（只有访存时间）。它**漏掉了算力那一半**：
   `max(mem, compute)` 退化成 `mem`。
3. 因为它是 `logger.info` 之后返回 `None`，不是抛异常。工具「正常返回」，
   调用方拿到一个看起来合理的秒数，不知道它少算了一半。
4. 不算一个可比较的数。表里的第三个字段 `flops_per_cycle = 2` 是
   **CUDA core 的 FMA 口径**（每 cycle 两次浮点运算），不是张量核口径。
   官方文档自己写着「Treat `expected_tflops` and `estimated_time` as comparison
   aids, not device-specific benchmark results」（`docs/tools/analyzer.md:110-112`）。

**常见错误**：以为「表里没有就会报错」。不报错才是危险的。

---

## E5

**思路**：先抄公式，再代数字，逐项比较。

**参考答案**：

```text
b = 2 字节（fp16 输入）          accum_bits = 32
smem_cost  = block_M*block_K*b + block_K*block_N*b
           = 128*32*2 + 32*128*2 = 16384 B
reg_usage  = 2 * max(prod(tile) * bits / 32)
           = 2 * max(128*128, 128*32, 32*128) * 32 / 32
           = 2 * 16384 = 32768

by_smem      = max_smem_usage // smem_cost = (2*227*1024) // 16384 = 28
by_reg       = reg_cap // reg_usage         = 65536 // 32768          =  2
by_partition = sm_partition                 =                           4
block_per_SM = min(28, 2, 4) = 2        binding = regs
grid_size    = (4096/128) * (4096/128) = 1024
num_wave     = ceil(1024 / (2 * 132)) = 4
```

**binding 是寄存器项。** shared memory 还能放 28 块，寄存器只允许 2 块。

**常见错误**：

* 用 `smem_cap` 而不是 `max_smem_usage` 算 `by_smem`。carver 用的是
  `max_smem_usage = 2 * smem_cap`（`carver/arch/cuda.py:158`），
  因为它假设每 SM 可以有两个 block 各用一半的 shared。
* 把 `reg_usage` 当成「每线程寄存器数」。它是**块级**的，量纲与
  `reg_cap`（每 SM 65536 个寄存器）一致。
* 忘了 `reg_usage` 的 `2 *`。

---

## E6

**思路**：跑两次，比较三组数：top-8 的 tile 身份、`num_wave`、拒绝原因分布。

**参考答案**：命令与实际输出

```bash
python3 code/03-carver-cost-model/carver_cost_model.py --smem-cap 101376 --cores 108 --topk 8
```

```text
candidates visited : 729   valid : 150                 （默认：154）
rejected on regs     :   46                            （默认：122）
rejected on smem     :  533                            （默认：453）
model priority values in use : 44                      （默认：48）
  1      128     256      16      1644.2     12.0      1     5    8.22e+09  regs
```

1. **top 的 tile 身份没有变化**：还是 `128×256` 与 `256×128` 系列（`block_K` 从 16
   到 128）。因为在这一组形状上，binding 的是寄存器而不是 shared：
   把 shared 砍到 99 KB 只是把更多大 tile 从「被寄存器拒绝」变成「被 shared 拒绝」。
2. `num_wave` 从 4 变成 5。这一行是 `128×256`，对应的 grid 是
   `(4096/128) × (4096/256) = 32 × 16 = 512`；`block_per_SM` 仍为 1，
   所以默认配置下是 `ceil(512/(1×132)) = 4`，换设备后是
   `ceil(512/(1×108)) = 5`。
3. **`--cores` 更能解释 `num_wave` 的差异**（它出现在分母里），
   而 `--smem-cap` 主要决定哪些候选存活。`priority = (traffic+1) × num_wave`，
   所以换了 SM 数之后整个 `prio` 的量级都会变，但**排序**往往不变——
   这也说明 `prio` 是一个排序键而不是时间。

**常见错误**：以为改了设备参数就一定会改 top-1。模型在无分辨率的维度上给并列，
换设备参数只会换掉并列者之外的部分。

---

## E7

**思路**：同一个算子跑两次，只换 `--roof`。

**参考答案**：

```bash
python3 code/01-op-ledger/op_ledger.py --case gemm --m 128 --n 128 --k 128 --roof tensor_core.fp16
python3 code/01-op-ledger/op_ledger.py --case gemm --m 128 --n 128 --k 128 --roof cuda_core.fp32
```

| roof | peak | 平衡点 | AI | 判类 | SOL |
| --- | --- | --- | --- | --- | --- |
| `tensor_core.fp16` | 729.26 TFLOPS | 161.97 | 42.667 | memory-bound | 0.022 µs |
| `cuda_core.fp32` | 57.27 TFLOPS | 12.72 | 42.667 | compute-bound | 0.073 µs |

**AI 是一个数（42.667），判类有两个答案。** 这不是模型自相矛盾，而是说
「这个算子是 memory-bound 还是 compute-bound」这句话本身不完整——
完整的问题是「**用哪个单元实现它**，它是 memory-bound 还是 compute-bound」。

对设计决策的意义：如果这个 128³ 的 GEMM 是要复用一个大 GEMM 的 kernel，
它会用张量核，于是判类告诉你「问题不在算力，在流量」；
如果它必须用 CUDA core 实现（例如形状不支持张量核、或者 dtype 不支持），
那么它就是 compute-bound，优化方向应该是减少算术，而不是减少访存。

**常见错误**：只说「平衡点不同」，不说「因为 AI 不变，所以判类翻转的直接原因是
roof 变了 12.7 倍」。

---

## E8

**思路**：逐行读函数，注意 `align` 作用在哪些维度、`k_group` 怎么算。

**参考答案**：

```cpp
// estimate_latency.cc:692-707
int64_t k_group = 256 / shape.input_dtype.bits();
if (dtype.is_float() && dtype.bits() == 32) k_group = shape.hf32 ? 8 : 1;
int64_t m = align(shape.m, 16);
int64_t n = align(shape.n, 16);
return 2 * m * n * align(shape.k, k_group);
```

1. `ops = 2 × align(M,16) × align(N,16) × align(K, k_group)`。
2. fp16/bf16：`k_group = 256/16 = 16`。fp32 且 HF32 模式：`k_group = 8`。
   普通 fp32：`k_group = 1`。
3. `M=256, N=200, K=128` bf16：

```text
align(256,16)=256  align(200,16)=208  align(128,16)=128
ops = 2*256*208*128 = 13,631,488
2*M*N*K             = 13,107,200
多付 4.0%
```

4. 有对应物，但方向相反。NVIDIA 的 `mma` / `wgmma` 是**形状约束**
   （`m16n8k16` / `m64nNk16`）：形状不合法就编不过，不合法就不会有性能问题。
   昇腾这边是**计费约束**：形状可以不是 16 的倍数，但超出的部分照样收钱。
   所以昇腾的 tile 尺寸要额外避免「差一点点」的形状（例如 200 而不是 192/208）。

**常见错误**：

* 以为 M 不需要对齐。对齐的是 `m` 和 `n` 两个维度。
* 把 `cube_throughput_fp16 = 8192` 理解成 8192 次乘加。
  函数里那个 `2 *` 已经说明它是 **8192 次浮点运算/cycle**，
  也就是 4096 次乘加/cycle。

---

## E9

**思路**：先看扫掠表找翻转区间，再用闭式解算理论值，最后解释两个数为什么不必完全相等。

**参考答案**：

1. 翻转在 `M=N=128`（binding = MTE2，128 vs 128 时 cube 64）与
   `M=N=192`（binding = Cube，144 vs 128）之间。
2. 闭式解：`MN/(M+N) ≥ 81.92`，方形时 `M ≥ 163.8`。
3. 因为 MTE2 的 II 是
   `max(ceil(bytes / bandwidth), descriptors × mte2_gm_to_l1_min_ii)`
   = `max(82, 2 × 64) = 128`。在 tile 小的时候，
   **描述符的最小发射间隔（128 cycles）而不是带宽（82 cycles）是瓶颈。**
   这是一条独立于带宽的约束，闭式解里没有它。
4. 官方例子用 `block_M=256, block_N=224, block_K=128`
   （`tilelang/ascend/README.md:74`），代入后 binding 是 **Cube**
   （Cube II 1792 vs MTE2 II 1229）。

**常见错误**：看到 128→192 的跳变就以为闭式解错了。闭式解是**渐近**结论，
在小 tile 上被描述符发射间隔顶住是正常现象——而这条恰好说明
「带宽够用」不等于「搬运够快」。

---

## E10

**思路**：四个问题先写全，再回到两行输出上找共同点。

**参考答案**：四个问题是

1. 融合省多少字节（拆分字节数 − 融合字节数）？
2. 融合后要同时活着多少片上数据？超过预算就是硬否决。
3. 融合会不会把 grid 缩到装不满设备？
4. 融合引入了什么新工作（额外输出、额外归约、额外同步）？

两行输出：

```text
skinny GEMM 1024x1024x16384, split-K=8           4.2      71.3     -67.1        64       512  split-K=8
decode attention, sq=1, skv=128k, split-KV=64    2147.5    2149.6      -2.1        32      2048  split-KV=64
```

共同原因是**第 3 个问题**：这两个算子在「不切」的形态下只有 64 块和 32 块，
而机器有 132 个 SM。也就是说，**不切的时候一半以上的机器是空着的**，
此时流量根本不是限制。所以它们愿意付 67.1 MB / 2.1 MB 去换 512 / 2048 块。

注意这两行的 `delta MB` 都是**负的**——「省下的字节」是负数，即切分之后搬得更多。
如果只看第 1 个问题，结论会是「不要切」，而那是错的。

**常见错误**：把「fusion 总是好的」当成前提。四个问题是**并列**的判据，
不是按优先级排序的——任何一条都能否决融合。

---

## E11

**思路**：改参数重跑，比较三行；第三问还要看片上占用随行数的增长。

**参考答案**：

```bash
python3 code/06-case-norm-rope-quant/case_norm_rope_quant.py --rows-per-block 16
```

```text
A: one row per block              1  163840    156         1.9       63.57     251.66    315.23     4.65x
B: 16 rows per block             16   10240     10        15.1       63.57      17.83     81.40     1.20x
C: one whole token per block     40    4096      4        36.2       63.57       6.29     69.86     1.03x
```

1. B（16 行）总流量 **81.40 MB，1.20×**。
2. C（40 行）总流量 **69.86 MB，1.03×**。它无法再改善查表部分，因为查表的
   重复次数已经是 1——`lookup_reads_per_token = 1.0`，也就是每个 token 的
   norm 权重和 RoPE 表各读了一次，**这是算法本身要求的最小值**，不是设计缺陷。
   剩下的 1.03 倍来自 `rows % rows_per_block` 的取整与权重按 block 计数。
3. `--q-heads 64` 时 `rows/blk = 64 + 8 = 72`，片上占用变成 **64.3 KB**，
   仍远低于 192 KB 预算（此时 payload 也涨到 114.43 MB，总流量 120.72 MB，
   1.02×）。

**常见错误**：

* 认为行数越大越好，一路加到 4096。行数受**片上容量**和**并行度**双向夹击：
  行数 × head_dim × (输入 + 输出 + fp32 暂存) 是线性的，
  而 grid = rows / rows_per_block 是反比的。
* 忘了「一个 token 的 q 与 k 用不同的 norm 权重」。如果模型里把两套权重当成一套，
  结果会偏乐观。

---

## E12

**思路**：这是设计题。先想清楚「多出来的东西是**中间量**还是**最终输出**」，
再决定它可不可省。

**参考答案**：

**1. 预算表（设计 C′：norm+RoPE 一个 kernel，量化一个 kernel）**

```text
第一个 kernel：读 q,k(bf16) + 写 q,k(bf16) + 读查表
  payload = 41.94 + 41.94 = 83.89 MB
  lookup  = 6.29 MB（design C 的值，因为粒度不变）
  小计    = 90.18 MB

第二个 kernel：读 q,k(bf16) + 写 q8,k8(fp8) + 写 scale
  payload = 41.94 + 20.97 + 0.655 = 63.57 MB
  lookup  = 0
  小计    = 63.57 MB

合计      = 153.75 MB  （vs 设计 C 的 69.86 MB，2.20x）
```

另外，第二个 kernel 的 grid 仍然是 4096（40 行一 block），所以并行度没有变差。

**2. 多搬的字节是不是不可避免的？**

多出来的 83.89 MB 全部是「把 q,k 以 bf16 写出去再读回来」。**只要两个阶段的
粒度是 40 行，它就是不可避免的**——因为第二个 kernel 无法访问第一个 kernel 的
寄存器。

唯一的例外是**让第二个 kernel 不要单独存在**：把 scale 的计算提前到第一个 kernel，
把 q8/k8 也一起写出去，只在后续需要重新量化时（例如写 KV cache）再另起一次调用。
那样就回到了设计 C。**所以正确的判据不是「哪个设计流量小」，
而是「产品需求真的要求两阶段吗」**：如果是，那 2.20 倍就是这一条需求的代价，
要明确写进设计文档，而不是当成实现细节。

**3. 在有卡机器上如何验证**

```text
要测的量
  1) 两个 kernel 各自的 wall time（第 12 章的口径：预热 + 多次重复 + L2 flush）
  2) 两个 kernel 各自的 DRAM 读/写字节数（Nsight Compute 的 dram__bytes，或 msProf 的对应指标）
  3) 第一个 kernel 的实测流量

判据
  a) 两个 kernel 的 DRAM 字节数之和除以 4502.4 GB/s，与 153.75 MB / 带宽 的下界比：
     如果实测时间 < 2× 下界，说明诊断成立，多出来的字节确实被搬了。
  b) 如果第一个 kernel 的实测读字节明显小于 90.18 MB，
     说明它的中间结果被 L2 兜住了一部分——此时应改为比较 kernel 之间的
     L2 命中率，而不是总字节数。
  c) 对照实验：把两个 kernel 用一次 launch 融合（设计 C′ 的 batch 版），
     看总时间是否下降约 83.89 MB / 带宽 ≈ 18.6 µs。若下降明显小于这个数，
     说明时间被别的东西（启动开销、尾部效应）主导，瓶颈判断需要重做。
```

**常见错误**：

* 只给一个数字，不说「多出来的字节是中间量，还是必须落盘的输出」。
  这是本题的核心区分。
* 把「验证」写成「跑一遍看看快不快」。验证要有**判据**：测什么量、和什么比、
  差多少算证实、差多少算证伪。

---

## E13

**思路**：任选六条，每条都要给出**可复算的数字或公式**，不能只有结论。

**参考答案**（一条示例写法，其余同理）：

| 反模式 | 定量依据 | 不遵守会先看到什么现象 |
| --- | --- | --- |
| 先写代码再调参 | 案例算子 AI = 3.102，远低于 `cuda_core.fp32` 平衡点 12.72 | 调了几天 `block_K` / `num_stages`，收益只有个位数百分比 |
| 用 datasheet 峰值算效率 | H200 fp16 张量核 989.5 → 729.26 TFLOPS，差 1.36× | 「效率只有 55%」，其实已经接近可达上限 |
| 用张量核 roof 给逐元素算子定价 | 平衡点 161.97 vs 12.72，差 12.7× | 把 `gemm 128³`（AI 42.667）判成 memory-bound |
| 超越函数按 1 FLOP 记账 | 昇腾 51 vs 8 ops/cycle，差 6.38×；softmax 行整体 2.08× | exp 密集的 kernel 实测比「算力上限」慢好几倍，且加大 tile 无效 |
| 忽略 epilogue / 查表 | 案例查表从理想 4.19 MB 变成 251.66 MB（79.8%） | 实测 DRAM 流量远高于算出来的下界，而算力利用率正常 |
| tile 太大导致 occupancy 崩掉 | 256×256 + fp32 累加 + 512 线程 = 128 regs/线程，`65536/(128×512) = 1` blk/SM | 每 SM 只有一块在跑，延迟藏不住，`num_wave` 不高但吞吐很低 |
| 模型当判决 | `code/07` 里 Cube 的 8 cycle 开销把 stage 公式从 2 顶到 3，官方例子用 2 | 按模型把 `num_stages` 定成 3，反而多占 48 KB 且没有收益 |

（其余各条的定量依据见正文第 2.3、3.2、4.3、5.1、6.4 节。）

**常见错误**：把反模式写成「不要这样做」的道德判断，而不是「这样做会付出多少代价」
的定量判断。本章的每一条反模式都能给出数字。

---

## E14

**思路**：跑三次，比较 binding、`blocks_per_SM` 与合法性。

**参考答案**：

```bash
python3 code/04-tile-size-decision/tile_decision.py --diagnose \
        --block-m 128 --block-n 256 --block-k 64 --stages 3 --threads 256
```

```text
C1 shared/stage     48.0 KB -> total    144.0 KB
C2 accumulators   128/thread ->  128 regs/thread (warp-rounded),  32768 regs/block
   occupancy: regs -> 2 blk/SM, smem -> 1, threads -> 8  => 1 blk/SM, binding=smem
C4 grid    512 tiles over   8-warp blocks -> 4 wave(s), last wave  87.9% full
```

1. **被 shared memory 卡住**（binding = smem）。`stages=3` 要 144 KB，
   `228 KB / 144 KB = 1` 块。寄存器那一项允许 2 块（`65536 / 32768 = 2`），
   shared 只允许 1 块，所以不是寄存器。
2. `--stages 2`：

```text
C1 shared/stage     48.0 KB -> total     96.0 KB
   occupancy: regs -> 2 blk/SM, smem -> 2, threads -> 8  => 2 blk/SM, binding=regs
C4 grid    512 tiles -> 2 wave(s), last wave 93.9% full
```

   `blocks_per_SM` 变成 2，binding 换成寄存器。
3. `--instr mma`：**合法**。`mma` 的要求是
   `block_M % 16 == 0`（128 ✓）、`block_N % 8 == 0`（256 ✓）、
   `block_K % 16 == 0`（64 ✓）、`threads % 32 == 0`（256 ✓）。
   `wgmma` 的额外要求是 `block_M % 64 == 0` 且 `threads % 128 == 0`，
   这份配置也满足，所以它同时合法于两条路径——`--instr` 只是换了检查标准，
   不改变诊断结果。
4. **下一步动作**：把 `stages` 从 3 降到 2。

   判据：`stages=3` 时 `blocks_per_SM = 1`，`stages=2` 时是 2，
   而尾波填充率从 87.9% 升到 93.9%。也就是说降一级流水换来的是**翻倍的驻留块数**
   和更好的尾波——这在延迟隐藏上通常比多一级缓冲更值。

   但这一步必须**测**：如果这个 GEMM 的 `K/block_K = 64` 轮循环里搬运延迟远大于
   计算周期，2 级流水就不够，此时应该改用更小的 `block_K`
   而不是加回第 3 级（第 11 章 4.5 节的判据）。

**常见错误**：

* 看到 `--instr mma` 输出不同就以为配置变了。真正变的只有检查标准，
   输出里的 binding 差异来自同一份配置在不同约束下的不同限制项。
* 直接给出「改 `block_K`」作为下一步。`block_K` 不影响 DRAM 流量，
  也不在这份诊断的约束里；先动它属于「在无分辨率的维度上调参」。
