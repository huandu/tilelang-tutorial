# 第 35 章 习题

说明：

* 难度：`★`（读完正文即可作答）/ `★★`（需要综合本章多处内容）/
  `★★★`（需要动手、多步推理，或对照源码）。
* 「需要硬件」一栏里，`否` 表示一台能跑 Python 的机器即可；
  `否（但要读源码）` 表示不需要硬件，但必须在 corpus 快照里核对；
  `是（NPU）` 表示必须用真实昇腾 NPU（都是选做）。
* 所有题目只依赖第 01–35 章的知识；涉及后文处会显式标注。
* 写作环境没有 GPU/NPU，所以涉及实测的答案只给**怎么测**与**判据**，
  不给「测出来是多少」。
* 本章的性能数字一律来自 `DeepGEMM-Ascend/README.md` 的表格，口径见正文 §5.1。

---

## E1 概念题 ★ 不需要硬件

1. DeepGEMM-Ascend 说它「完全兼容 DeepGEMM 的 API」。请从**包名、函数名、参数
   形态、后缀含义**四个方面各写一句，说明这个「兼容」具体指什么；
2. 这个兼容承诺**不覆盖**哪一部分？（提示：README 的一句限定语）
3. 写出昇腾侧 scaling factor 的打包关系式（含数值范围与主序），
   并说明它与 NVIDIA SM90、SM100 各自的格式有什么不同。
4. 除了 SF 格式，正文 §2.2 还指出**两处**「README 未提、但代码里写明」的差异，
   它们分别是什么？

---

## E2 概念题 ★ 不需要硬件

正文 §2.3 把 MAD 原语抽象拆成三个头文件 + 一个出口函数。

1. `tile_ptr.hpp` 里的 `operand_ptr` 比裸指针多了什么？`as_mad_aligned` 在做什么？
2. `copy.hpp` 替使用者挡掉了哪四类琐碎工作？各举一个函数名。
3. `sync.hpp` 的 `PipeEvent<kSrc, kDst>` 为什么只存一个整数 `base`，
   却比裸 flag id 更不容易写错？
4. `mad(...)` 里判断走 `asc_mmad_mx` 还是 `asc_mmad` 的条件是什么？为什么是这个条件？
5. 这四个文件的产物（Ascend C 代码）与 TileLang 的产物是不是同一层？说明理由。

---

## E3 性能表读图题 ★★ 不需要硬件

下面是 `DeepGEMM-Ascend/README.md` 的两组数据（口径：Ascend 950DT、CANN 9.20、
`bench_msprof`、冷 L2）：

| Type | M | N | K | Latency (us) | Computation (TFLOPS) | Hardware limit | Utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8×FP8 | 4096 | 7168 | 16384 | 1117.6 | 861 | 865 | 99.5% |
| BF16×BF16 | 4096 | 7168 | 16384 | 2229.9 | 431 | 432 | 99.8% |

| M | N | K | Latency (us) | Computation (TFLOPS) | Memory BW (GB/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7168 | 16384 | 54.8 | 549 | 2456 |
| 4096 | 7168 | 16384 | 1117.6 | 861 | 186 |

1. 同一形状下 BF16 的延迟约为 FP8 的 2 倍，用正文 §5.1/§5.2 的哪个结论解释？
2. 上面第二张表里，`M=128` 与 `M=4096` 各自的瓶颈是什么？请用「算术强度」
   与「机器的平衡点」两步推出来（平衡点可用 `865 TFLOPS / 3500 GB/s` 估算）。
3. 为什么 `M=128` 那一行的算力利用率只有 63%，作者却认为它「已经接近极限」？
4. 如果要优化 `M=128, N=7168, K=16384` 这一行，你会在 `GemmConfig` 的哪几项上
   动手？说明每一项想改变什么。

---

## E4 代码阅读题 ★ 不需要硬件（但要读源码）

下面是 `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/ascend/copy.hpp:371-380`：

```cpp
template <typename TA, typename TB, typename TC>
__aicore__ inline void mad(l0c_ptr<TC> l0c, l0a_ptr<TA> l0a, l0b_ptr<TB> l0b,
                           uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           asc_unit_flag_mode unit_flag, bool clear_C = false) {
    if constexpr (sizeof(TA) == 1 || sizeof(TB) == 1) {
        asc_mmad_mx(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n,
                    static_cast<uint8_t>(unit_flag), true, false, clear_C);
    } else {
        asc_mmad(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n,
                 static_cast<uint8_t>(unit_flag), true, false, clear_C);
    }
}
```

1. 哪些 dtype 会走 `asc_mmad_mx`？fp4 打包成什么类型、`sizeof` 是多少？
2. 两个分支里 `shape_k` 与 `shape_n` 的传参顺序与函数签名里的 `(shape_m, shape_n,
   shape_k)` 不同，这不是笔误——请说明为什么（提示：`asc_mmad` 自己的形参顺序）。
3. `clear_C` 默认是 `false`。在 GEMM 的 K 循环里它应该取什么？如果忘了会有什么症状？
   （提示：第 19 章的 P9）
4. `unit_flag` 的作用是什么？在 `hc_prenorm.py:287-291` 里它被设成
   `T.Select(最后一块, 3, 2)`，这个「最后一块用 3」的意图是什么？

---

## E5 代码阅读题 ★★ 不需要硬件（但要读源码）

下面是 `DeepGEMM-Ascend/csrc/jit_kernels/transform_sf.hpp:45-55`（节选）：

```cpp
const uint32_t src_shape_m = sf.size(-2);
const uint32_t src_shape_k = sf.size(-1);
const uint32_t dst_shape_m = shape_m;
const uint32_t dst_shape_k = ceil_div(src_shape_k, is_float_input ? 2 : 1);
auto physical_sizes = sf.sizes().vec();
physical_sizes[physical_sizes.size() - 2] = dst_shape_k;
physical_sizes[physical_sizes.size() - 1] = dst_shape_m;
auto output = torch::empty(physical_sizes, sf.options().dtype(at::kShort)).transpose(-2, -1);
```

设输入 `sf` 是 fp32、形状 `[MN, K/32]`（`gran_k = 32`）。

1. 分别算出 `src_shape_k`、`dst_shape_k`、`dst_shape_m` 与**输出张量的形状**。
2. 输出张量的 `stride(-2)` 与 `stride(-1)` 分别是多少？这对应哪个 `Major`？
3. 一个 `dst` 元素（`int16`）里装了哪两个输入元素？高位与低位各是谁？
4. 如果调用方交上来的 `sf` 是 fp32 但尾数不为 0，会在哪里失败、报什么？
5. 为什么这个转换 kernel 的输入可以接受 fp32，也可以接受已经打包好的 `int16`？
   （看 `csrc/apis/layout.hpp:37-65` 的 `check_sf_layout`）

---

## E6 概念题 ★★ 不需要硬件

正文 §4 把「一次 shape 的旅程」拆成五阶段。

1. 写出五个阶段的名称与各自的核心数据结构/产物。
2. `GemmDesc.compiled_dims` 的默认值是什么？为什么默认把 N、K 固化而把 M 留成
   运行期？（提示：MoE 推理的 M 是什么）
3. `DG_PRINT_CONFIGS=1` 与 `DG_JIT_DEBUG=1` 有什么区别？它们各自的消费点在哪一行？
4. DeepJIT 的缓存 key 包含哪些成分？它和 TileLang（第 26 章）的磁盘缓存 key
   最本质的差异是什么？
5. `DG_JIT_*` 与 `DJ_JIT_*` 的关系是什么？为什么说「`import deep_gemm` 之后再设
   环境变量可能已经晚了」？

---

## E7 计算题 ★★ 不需要硬件

`csrc/jit_kernels/heuristics.hpp:46-48,52-71` 里：

```cpp
constexpr int get_mad_flops_per_cycle(int elem_bits) const {
    return bf16_flops_per_cycle * (sizeof(uint16_t) * 8) / elem_bits;
}
// .bf16_flops_per_cycle=8192, .aic_clock_mhz=1650, .num_ai_cores=32,
```

1. 分别算出 bf16（16）、fp8（8）、fp4（4）三种情况下的「每核每周期 FLOPs」。
2. 用 `每核每周期 × 主频 × 核数` 算出三档峰值（TFLOPS），
   并与 README 的 Hardware limit 列（432 / 865 / 1730）对照。
3. 为什么 fp4 的峰值正好是 bf16 的 4 倍？这是硬件实测结论还是模型假设？
4. 如果某天这台机器的 `aic_clock_mhz` 变成 1800，三档峰值各变成多少？
   README 里那些 Utilization 会跟着变吗？（说明理由）

---

## E8 判断题 ★★ 不需要硬件

下面是四个真实需求。请对每一个判断「先走 TileLang」还是「下钻 Ascend C + MAD 原语」，
并给出**至少两条依据**（依据要落在正文 §5 / §6 / §7 的具体事实上）。

1. **A**：一个 `[8192, 24, 28672]` 的瘦长 GEMM + 行平方和，形状固定，目标是打满带宽；
2. **B**：一个 `[4096, 7168, 16384]` 的 fp8 dense GEMM，要逼近硬件算力峰值，
   形状与量化格式已经冻结；
3. **C**：一个还在迭代、形状每周都变的 attention 变体，只要求先跑通、能对拍；
4. **D**：一个 MQA 打分算子，`head_dim=128`，profile 显示 `aic_fixpipe` 已经 99%。

---

## E9 代码阅读题 ★★ 不需要硬件（但要读源码）

`hc_prenorm.py:26-41` 用三个约束决定 K 方向切几份：

```python
num_k_splits_by_tokens = max(1, num_aic_cores * num_l0c_stages // num_token_blocks)
num_k_splits_by_hidden = num_hidden_blocks & -num_hidden_blocks
num_k_splits_by_cores  = num_aic_cores & -num_aic_cores
split_limit = min(num_k_splits_by_tokens, num_k_splits_by_hidden, num_k_splits_by_cores)
num_k_splits = 1 << (split_limit.bit_length() - 1)
```

1. `x & -x` 算出来的是什么数？为什么这三个约束都要求它是 2 的幂？
2. 三个约束各自避免什么坏情况？（提示：尾块、整除、并行度）
3. `hc_prenorm.py:133-134` 与 `:358-359` 打开的原子加，为什么只对 `split_id != 0`
   生效？这样做的代价是什么？接口给了什么开关来消除这个代价？
4. 这个 kernel 的 N 是 24，但 `block_mhc_mult3 = (24 + 15) // 16 * 16 = 32`。
   为什么要「物理 32、逻辑 24」？如果直接用 24 会碰到什么问题？

---

## E10 概念题 ★★ 不需要硬件

1. MegaMoE 把哪五段融进一个 kernel？融合的收益是什么（至少两条）？
2. 融合的代价至少写出三条，并各给一个源码依据。
3. `csrc/apis/mega_moe.hpp:159` 有一句
   `DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores)`。这句断言对使用者意味着
   什么？为什么会有这条限制？
4. 官方说 MQA logits「FIX-pipe bound rather than compute bound，饱和到 99%」。
   什么是 FixPipe？为什么这个算子会卡在它上面（给出两条形状上的理由）？
5. 同一个 MQA 算子，decode（Q=256）与 prefill（Q=4096）两行的带宽差了一个数量级。
   请解释为什么「瓶颈是形状的函数」。

---

## E11 动手题 ★★ 不要在 NPU 上做（CPU 可完成）

用本章 `code/02-perf-ledger/perf_ledger.py` 作为起点，写一段脚本：

1. 遍历 README 的 FP8 推理表（14 行）；
2. 每行用 `Computation / 865` 算出算力利用率；
3. 按「算力利用率 < 50% 判为带宽主导，否则判为算力主导」输出一张两列的清单；
4. 再对每一行算出「算术强度」`2·M·N/((M+N)·1)`（fp8 时每元素 1 字节），
   与平衡点 `865e12/3500e9 ≈ 247` 比较，看分类是否与第 3 步一致。

**要求**：把两处不一致的行单独列出来，并解释可能的原因（提示：K 很小的时候，
行尾的 128 字节扇区放大与 SF 读取会改变真实字节数）。

---

## E12 综合题 ★★★ 不需要硬件

一个同事说：「既然 DeepGEMM-Ascend 能把 GEMM 做到 99.8%，
那我们全组都改用 Ascend C + MAD 吧，TileLang 只适合写 helloworld。」

请写一段 300 字以内的回复，要求：

1. 指出这句话**对**的部分；
2. 指出它**错**的部分，并至少用本章两个具体事实反驳
   （提示：hc_prenorm 为什么用 TileLang；TileLang 侧官方分工对照表说了什么）；
3. 给出三条「该选哪条路」的判据。

---

## E13 概念题 ★ 不需要硬件

1. 为什么本章反复强调「冷 L2」这个口径？如果换成热 L2，下面哪一列会明显变化、
   哪一列几乎不变：`Computation`、`Memory bandwidth`、`Hardware limit`？
2. 本章说「本机没有 NPU，所有代码结论都标注『静态阅读』」。
   在这种约束下，一个作者**可以**做到哪三件事来保证结论可信？
   **不可以**做什么？（提示：参考正文开头的验证边界与第 19 章的写法）

