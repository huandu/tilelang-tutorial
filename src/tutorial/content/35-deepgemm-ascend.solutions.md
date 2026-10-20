# 第 35 章 习题答案

说明：每题给“解题思路 + 参考答案 + 常见错误”。引用正文时用“§N.M”，
引用源码时给 corpus 路径与行号，引用数值时标明它来自 README 表格还是本书复算。
涉及真机实测的题只给判据与流程。

---

## E1

**思路**：§2.1 讲“兼容什么、不覆盖什么”，§2.2 讲“唯一不兼容点”与两处代码差异。

**参考答案**

1. 四个方面：
   * **包名**：两边都叫 `deep_gemm`，`import deep_gemm` 不变（`README.md:9`）。
   * **函数名**：`fp8_fp4_gemm_{nt,nn,tn,tt}`、`bf16_gemm_*`、`m_grouped_*_contiguous`、
     `k_grouped_*_tn_contiguous`、`fp8_fp4_mqa_logits`、`fp8_fp4_mega_moe` 等一一对应
     （注册表 `csrc/apis/gemm.hpp:537-662`、`csrc/apis/attention.hpp:209+`、
     `csrc/apis/mega_moe.hpp:206-225`）。
   * **参数形态**：低精度 GEMM 一律传 `(数据, scale)` 元组；`c` 是可选累加项；
     `recipe`/`recipe_a`/`recipe_b` 与 `compiled_dims` 同名同义（`gemm.hpp:190-236`）。
   * **后缀含义**：`nt/nn/tn/tt` 说的是**逻辑形状**而不是内存布局；
     `m_` / `k_` 前缀指 grouped 的轴（`gemm.hpp:45-58`、`:243-249`、`:311-317`）。
2. 不覆盖的是**工具函数全集**。README 只说“kernel API 见 DeepGEMM 的接口”
   （`README.md:52-53`）；NVIDIA 侧有 `set_tc_util` / `set_pdl` /
   `set_ignore_compile_dims` / `get_mn_major_tma_aligned_tensor`（`DeepGEMM/README.md:146-160`），
   Ascend 侧换成 `set_npu_arch` / `set_mk_alignment_for_contiguous_layout` /
   `get_paged_mqa_logits_metadata` 等（`README.md:54-60`）。
3. 打包关系：`sf_out[m, p] : int16 == (sf[m, 2p] << 8) | sf[m, 2p+1]`，
   其中 `m ∈ [0, MN)`、`p ∈ [0, ceil(K/64))`，物理上按 m（MN）连续。
   对照：SM90 用 FP32 scale，SM100 把 **4 个** UE8M0 打进一个 `torch.int`
   （`DeepGEMM/README.md:65-68`），昇腾是 **2 个**打进 `int16`。
4. 两处：masked grouped GEMM 未实现（`csrc/apis/gemm.hpp:591,615`）；
   `fp4×fp8` 不再支持（`gemm.hpp:184-188`）。

**常见错误**：把“兼容”理解成“二进制/数值完全一致”。兼容的是**接口契约**，
不是数值路径（BF16 那一条，两边 K 方向累加顺序不同，`tests/utils.py:18-30` 明确
允许阈值而不是逐位相等）。

---

## E2

**思路**：§2.3 逐条给了文件名与函数名，照它回答即可，但要说清“为什么这样设计”。

**参考答案**

1. `operand_ptr` 存的是 `addr + shape_mn + shape_k + major`（`ascend/tile_ptr.hpp:19-40`），
   比裸指针多了**形状与主序**。`as_mad_aligned()` 把形状向上对齐到 MAD 能接受的尺寸
   （MN→`FRAC_MN=16`，K→`get_frac_k<T>()`），让调用方不必自己算对齐（同文件 `:50-55`）。
2. 四类：**通路与格式**（`copy_gm_to_l1` 处理 ND↔NZ，`copy_l0c_to_gm` 处理 NZ→ND）；
   **对齐与尾块**（`fill_tail_block`、`copy_gm_to_ub` 的 `right_padding`）；
   **地址/索引计算**（`copy_l1_to_l0a(dst, src, mn_idx, k_idx)`）；
   **参数转换**（`infer_quant_mode<T, ST>()` 编译期推出量化模式）。
3. 因为它把 **PIPE 方向编码进类型**：`notify(PipeEvent<PIPE_MTE2, PIPE_MTE1>{0})` 一旦
   把源/目的写反，模板参数就对不上，编译期就报错；裸 flag id 写反只会变成运行期的
   偶发错误（第 19 章 P10 的症状）。
4. 条件是 `sizeof(TA) == 1 || sizeof(TB) == 1`（`ascend/copy.hpp:373-379`）。
   因为 fp8 是 1 字节、fp4 打包后也是 1 字节（`float4_e2m1x2_t`），
   两者都要走带 micro-scaling 的 `asc_mmad_mx`；bf16 是 2 字节，走普通 `asc_mmad`。
5. 是同一层。两者最终都由 bisheng/ld.lld 编译成 Ascend C 设备代码；
   差别只在“谁生成那段 Ascend C”，TileLang 在编译期，DeepGEMM-Ascend 在运行期
   拼源码（§2.4 图 2）。

**常见错误**：以为 `asc_mmad_mx` 是“更快的 mmad”。它是**带 scale 的形态**，
和速度无关；fp4 走它是因为有 scale 要参与运算。

---

## E3

**思路**：先算强度，再比平衡点，最后看表格里的两列互相印证。

**参考答案**

1. 用“每周期 MAD 数 ∝ 1/elem_bits”这条硬件模型（`heuristics.hpp:46-48`）：
   fp8 每周期 16384 FLOPs，bf16 每周期 8192，正好差 2 倍 → 同形状延迟约 2 倍。
   README 的 Hardware limit 列（432 vs 865）也正好差 2 倍。
2. 平衡点 ≈ `865e12 / 3500e9 ≈ 247 FLOP/byte`。
   `M=128`：强度 `2·M·N/((M+N)·1B) = 2·128·7168/(7296) ≈ 251` → **贴着平衡点**，
   读 B 的时间与算的时间同量级，瓶颈是**带宽**。
   `M=4096`：强度 `≈ 2·4096·7168/11264 ≈ 5213` → **远高于平衡点**，瓶颈是**算力**。
3. 因为它的分母选错了。它按算力算只有 63%，但真实约束是“把 B 读进来”，
   那一行 `Memory BW = 2456 GB/s` 已经接近机器带宽（`HWDesc.gm_bw_gbps = 3500`，
   复算见 `code/02-perf-ledger/`），所以它在该维度上已经接近极限。
4. 应在这些项上动手：
   * `resident_b` / `l2_ctrl_b`：让 B 在 L1/L2 里多留一会儿，减少重复搬运
     （`heuristics.hpp:196-197`、`:535-545`）；
   * `block_n`：把 N 方向切大一点，提高 B 的复用；
   * `num_l1_stages`：加深流水，让搬运与 MAD 重叠得更充分；
   每一项的目标都是同一个：**提高算术强度或提高搬运效率**，而不是“让 MAD 更快”。

**常见错误**：因为 `M=128` 利用率低就认定“写得不好”。
利用率的分子分母要先对齐到同一条流水线。

---

## E4

**思路**：逐行读那 10 行代码，配合 `common.hpp` 的常数。

**参考答案**

1. `sizeof(TA)==1 || sizeof(TB)==1`：fp8（`float8_e4m3_t`，1 字节）与
   fp4（`float4_e2m1x2_t`，1 字节、两个半字节）。所以 fp8×fp8、fp8×fp4、fp4×fp4
   都走 `asc_mmad_mx`。
2. 因为 `asc_mmad(_mx)` 自己的形参顺序是 `(c, a, b, m, k, n, ...)`。它要的是
   `(shape_m, shape_k, shape_n)`；`mad()` 的签名用的是人类习惯的
   `(shape_m, shape_n, shape_k)`，所以调用时要换位。这一点从 `mad()` 的调用行
   与 `copy.hpp:288-310` 的 `asc_copy_l0c2gm(dst, src, n_elems, m_elems, ...)`
   也能互相印证：这个库的底层 API 普遍是 `(n, m)` 顺序。
3. K 循环里应取 `k == 0`（第一块清零）。忘了清的后果：结果里混进上一轮 L0C 的残值，
   表现为“第一次调用对、后续全错”或“换了输入但输出没变”，正是第 19 章 P9。
4. `unit_flag` 是覆盖更新控制（相邻两块能否合并搬运/L0C 写回）。
   `hc_prenorm.py:287-291` 在**最后一块隐藏维 + 最后一个 MAD** 时用 3、
   其余用 2，意图是让最后一块“收尾”而不是继续 accumulation ，具体的枚举语义以
   CANN 的 `asc_unit_flag_mode` 为准（本章只按源码行号引用，未做真机验证）。

**常见错误**：把 `clear_C` 当成可选性能开关。它是**正确性**开关。

---

## E5

**思路**：严格按代码里出现的顺序代数值。

**参考答案**

1. `src_shape_k = K/32`（输入是 fp32、每 32 个 K 一个 scale）；
   `dst_shape_k = ceil((K/32)/2) = ceil(K/64)`；`dst_shape_m = shape_m = MN`。
   输出张量形状是 `[MN, ceil(K/64)]`（因为先按 `[dst_shape_k, dst_shape_m]` 分配、
   再 `.transpose(-2,-1)`）。
2. `stride(-2) == 1`、`stride(-1) == MN`（物理上 MN 连续），对应 `Major::MN`
   （`csrc/utils/layout.hpp:44-47`）。
3. 高位是 `sf[m, 2p]`，低位是 `sf[m, 2p+1]`（`deep_gemm/include/deep_gemm/
   transform_sf.hpp:251-280` 的 `shiftright 23` / `shiftright 15` / `or`）。
   注意这里的“m”在 MN-major 打包实现里对应**相邻两行**。
4. 在 device 侧的 `check_validation` 报错（`transform_sf.hpp:13-21`）：
   要求 fp32 非负且低 23 位为 0，否则 `ascendc_assert` 失败。
5. `check_sf_layout`（`csrc/apis/layout.hpp:37-65`）先判断输入是不是已经满足
   “short + MN-major + gran_mn==1 + 正确 stride”；满足就直接返回**不转换**，
   否则才走 `launch_transform_sf`。所以两条输入路径都合法，且转换是**幂等**的。

**常见错误**：以为 `.transpose(-2,-1)` 改变了物理内存。它只改 stride；
物理布局是由 `physical_sizes` 决定的。

---

## E6

**思路**：§4.1 的表 + §4.2/§4.6 的说明。

**参考答案**

1. ① 描述问题（torch 张量 → `GemmDesc`）；② 选 config（`GemmDesc` → `GemmConfig`）；
   ③ 生成源码（`GemmConfig` → 一段 C++ 字符串）；④ 编译与缓存（源码 → 设备二进制）；
   ⑤ launch（kernel + `gm_ptr` + `LaunchOptions`）。
2. 默认 `"nk"`（`csrc/apis/gemm.hpp:543`）。因为 MoE 推理时每个专家收到的 token 数
   （M）随 batch 变化，而 N（专家输出维）、K（专家输入维）是固定的，
   把固定的固化进编译期常量能去掉运行期分支，把变化的留给运行期能避免反复重编。
3. `DG_PRINT_CONFIGS=1` 只打印每个 shape 选中的 config；`DG_JIT_DEBUG=1` 是超集，
   还打印编译命令、加载耗时等。两者的消费点相同：
   `csrc/jit_kernels/heuristics.hpp:714-718`（同一个 `if` 里）。
4. DeepJIT 的 key 覆盖源码、追踪到的 include、编译器版本、有效编译选项，
   以及调用方提供的 dependency signature（`DeepJIT/README.md:12`）。
   与 TileLang 相比最本质的差异：**DeepJIT 的输入已经是源码字符串，key 里没有 IR 这层**；
   TileLang 的 key 覆盖 IR 与 pass 配置（第 26 章 §4.4）。
5. 每个 `DG_JIT_*` 未设时回退到 `DJ_JIT_*`（`DeepJIT/include/deep_jit/utils/env.hpp:65-70`）；
   而且这些值在 JIT 运行时**第一次创建时被快照**（`README.md:62-63`）。
   另外 `ASCEND_HOME_PATH` 是 **import 时**读的（`deep_gemm/__init__.py:15-17`），
   所以两者都必须赶在对应时机之前设置。

**常见错误**：以为改环境变量总能生效。这个库有“快照”语义。

---

## E7

**思路**：硬算，然后与 README 对照；最后一问考的是“除式的分子分母”。

**参考答案**

1. bf16 = `8192 × 16 / 16 = 8192`；fp8 = `8192 × 16 / 8 = 16384`；
   fp4 = `8192 × 16 / 4 = 32768` FLOPs/cycle/核。
2. 峰值 = 每核每周期 × 1650 MHz × 32 核：
   bf16 = 8192 × 1.65e9 × 32 = **432.5 TFLOPS**；
   fp8 = **865.0 TFLOPS**；fp4 = **1730.1 TFLOPS**。
   与 README 的 432 / 865 / 1730 吻合（差 < 0.2%，复算脚本见
   `code/02-perf-ledger/`）。
3. 是**模型假设**：它假设 Cube 的吞吐与元素位宽成反比（`8192 × 16 / bits`）。
   源码注释也只写了 `4096 multiply-adds; each counts as two FLOPs`
   （`heuristics.hpp:57`）。这是硬件模型里的假设，不是本机实测。
4. 三档都乘以 `1800/1650 = 1.0909`：bf16 471.7、fp8 943.3、fp4 1887.3 TFLOPS。
   README 的 Utilization 会变（分母变成新峰值），但 **Computation 那一列不会**，
   它来自实测延迟。所以“利用率的绝对值”依赖于你用的硬件模型版本。

**常见错误**：把 432 当成“产品规格书的峰值”。它其实是源码里那个模型的输出，
与 README 恰好一致。

---

## E8

**思路**：按 §7.3 的三问（瓶颈类型 → 形状是否冻结 → 访存是否规整）逐条过。

**参考答案**

1. **A → 先走 TileLang**。依据：① 形状就是本章 hc_prenorm 的形状，
   算术强度 ≈ 24 FLOP/byte，是 memory-bound（§5.7 复算）；②
   第 6 节说明 TileLang 版已把带宽打到 3463/3500 ≈ 98.9%。
   反例条件：若访存变得稀疏或需要精细 L1 bank 配平，再考虑下钻。
2. **B → 下钻 Ascend C + MAD（或至少准备好）**。依据：① 这是 5.2 节那张表的
   形状，目标 99.5% 是靠 `mad_m/n/k`、L1/L0 stage、`resident_*`、`direct_store`、
   L2 hint 这一整套手工参数拿到的；② 形状已冻结，重编代价可接受。
   如果只是“先追上 90%”，用 TileLang 也可以（§7.3 问题 3）。
3. **C → 先走 TileLang**。依据：① 形状每周变，DeepGEMM-Ascend 的配置几乎全是
   编译期常量，改了就要重 JIT；② 需求是“跑通 + 对拍”，TileLang 的
   `@tilelang.jit` + 布局注解迭代成本更低。
4. **D → 先定位，再决定；不要先改 MAD 形状**。依据：① `aic_fixpipe` 是
   `L0C -> out` 这条流水线（`deep_gemm/testing/bench.py:39`），已经 99%，
   此时优化 MAD 形状没有意义；② README 的 MQA 表证明了这一点，
   prefill 的算力只有 78.7%，瓶颈在收尾。可行的方向是减少搬出量、
   或把归约挪到别的流水线（第 36 章会展开）。

**常见错误**：用“性能目标高不高”当唯一判据。正确判据是“瓶颈在哪条流水线”+
“形状是否冻结”。

---

## E9

**思路**：`x & -x` 是经典的取“最大 2 的幂因子”，三个约束各自对应一种切分失败。

**参考答案**

1. `x & -x` 取的是 `x` 里最大的 2 的幂因子（例如 `24 & -24 = 8`）。
   要求 2 的幂是为了让 `num_k_splits` 也能取成 2 的幂（`1 << (bit_length-1)`），
   从而**均匀**分配，不留尾块。
2. 三个约束：`by_tokens` 保证每个 split 至少能分到 `num_l0c_stages` 个 token 块，
   否则并行度白给；`by_hidden` 保证 K 方向的块数能被整分；
   `by_cores` 保证可用核数能被整分。
3. 因为 split 0 做**普通写**、其余 split 做**原子加**，最后 D 里就是完整结果，
   不需要额外的跨核规约 kernel。代价是原子加的累加顺序不保证，
   结果**不可复现**。接口给的开关是 `use_deterministic_algorithms(True)`
   （`csrc/jit/jit.hpp:51-57`），它把 store 拆成 `num_k_splits` 个串行相位
   （`hc_prenorm.py:95-102`），用速度换可复现。
4. 因为物理缓冲区要按 16 对齐（MAD/L1 的搬运粒度），直接把物理形状写成 24 会
   在对齐上失败或引入额外转换。做法是“物理 32、逻辑 24”：
   `T.copy` 与 `T.gemm` 都只覆盖 `0:mhc_mult3`，多出来的 8 行只在容量计算里存在
   （`hc_prenorm.py:158-163` 的注释）。这与第 25 章讲的“OOB padding / 尾块”
   是同一类工程手法。

**常见错误**：把 `use_deterministic_algorithms` 当成“打开就更准”。
它不改数值精度，只改累加顺序的确定性。

---

## E10

**思路**：§5.6 的收益/代价清单 + §5.5 的 FIX-pipe 分析。

**参考答案**

1. 五段：EP dispatch、Linear1（grouped GEMM）、SwiGLU + 量化、Linear2（grouped GEMM）、
   EP combine。收益：① 中间结果不再写回 HBM 再读出（`layout/mega_moe.hpp:26-33`
   的四个进度阶段说明数据在片上/对称缓冲区里交接）；② 通信与计算可以重叠
   （`deep_gemm/include/deep_gemm/mega_moe.hpp:61-107` 把 AIC/AIV0/AIV1 的 pipe
   event 全部显式编号）。
2. 代价（任选三条）：① 必须多进程 + HCCL 对称缓冲区（`csrc/apis/mega_moe.hpp:44-62`）；
   ② 形状必须提前注册（`:27-42`、`:72-110`）；③ 核数固定 32（`:159`）；
   ④ 每次 launch 要清 progress 区（`csrc/jit_kernels/mega_moe.hpp:43-46`）；
   ⑤ 只支持 fp8×fp4 + swiglu + fast_math（`:145-150`）。
3. 意味着**用了 MegaMoE 就不能改核数**，否则断言失败。原因是它的核间分工
   （谁搬 A、谁搬 B、谁做 combine 归约）是按 `kNumAICores = 32`
   静态排好的，event 编号与 L1 布局表都是编译期常量。
4. FixPipe 是**唯一一条从 L0C 出去的流水线**（`L0C → GM/L1/UB`，搬运途中可做
   格式/类型转换）。MQA 卡在它上面，两条形状上的理由：
   ① `head_dim` 只有 64/128，一次 MAD 能换来的额外计算很少；
   ② 输出 logits 是 `#[Q] × max_seqlen_k` 的稠密矩阵（`csrc/apis/attention.hpp:97-98`），
   搬出量与输入同量级。两头一夹，`L0C→out` 成为上界。
5. 因为 Q 少时 KV 基本只用一次（强度低、带宽高），Q 多时 KV 被复用很多次
   （强度高、回到算力/FIX 约束）。README 里 decode 那一行带宽 1349–2006 GB/s，
   prefill 那两行只有 277–283 GB/s，同一个算子，换个 Q 就是另一种瓶颈。

**常见错误**：把“融合”当成纯收益。融合一定伴随形状/进程/缓冲区的约束收紧。

---

## E11

**思路**：先按算力利用率分类，再按算术强度分类，最后看谁不一致。

**参考答案**（关键代码与结论）

```python
balance = 865e12 / 3500e9          # ≈ 247 FLOP/byte
for row in INFER_FP8:              # 用 code/02-perf-ledger 的表
    util = row.compute_tflops / 865 * 100
    intensity = 2 * row.m * row.n / ((row.m + row.n) * 1)   # fp8：1 B/element
    print(row.m, row.n, row.k, f'{util:.1f}%',
          '带宽主导' if util < 50 else '算力主导',
          '带宽侧' if intensity < balance else '算力侧')
```

预期：`M=128` 的 7 行全部落在“带宽主导 / 带宽侧”（算力利用率 12.1%–63.5%，
其中 `M=128,N=576` 只有 12.1%）；`M=4096` 的 7 行全部落在“算力主导 / 算力侧”
（71.6%–99.5%）。

可能不一致的行与原因：

* `M=128, N=7168, K=16384`（63.5%）与 `M=128, N=24576, K=1536`（54.1%）
  的算力利用率已经超过 50%，但强度仍低于平衡点。因为**强度公式忽略了 SF**：
  `gran_k=32` 时 SF 占 `1/32` 的字节，K 越小占比越高，
  真实强度还要再低一点；
* `M=128, N=32768, K=512`：K 只有 512，行尾对齐到 128 字节扇区时会有明显放大，
  实测带宽比模型高（正文 §5.3 提到 FP8 推理表的比值稳定在 1.09 左右）。

**常见错误**：用一个阈值（50%）代替强度计算。阈值只是把连续量离散化的粗筛，
真正的判据是“强度 vs 平衡点”。

---

## E12

**思路**：先承认真实结论（99.8% 是真的），再指出“真但不能推广”，最后给判据。

**参考答案**

> 对的部分：dense GEMM 上 Ascend C + MAD 确实做到了 98.3%–99.8%（README 表格），
> 这是真结论，不是宣传。
>
> 但不能推广到全部算子，有三条反证。第一，同一个仓库的 hc_prenorm
> 就是用 TileLang 写的（README Acknowledgements：“The mHC kernel is backed by
> Tilelang”），因为它是 memory-bound：算术强度 ≈ 24 FLOP/byte，
> TileLang 版本已经把带宽打到 3463/3500 ≈ 98.9%，再下钻 Ascend C 收益极小、
> 维护成本却很高。第二，99.8% 是在**形状与量化格式冻结**的前提下拿到的，
> DeepGEMM-Ascend 的配置几乎全是编译期常量，形状一周一变的光景下重编成本会爆炸。
> 第三，那个 99.8% 的分母是“Cube 算力”，对 MQA 这类 FIX-pipe bound 的算子，
> 算力利用率再高也没用。
>
> 三条判据：① 瓶颈在哪条流水线（算力 / 带宽 / FixPipe / 通信）；
> ② 形状与算法是否已冻结；③ 访存是否规整（稀疏、不规则尾块、L1 bank 配平
> 会让编译器帮不上忙）。前两条指向 Ascend C，其余先走 TileLang。

**常见错误**：用“哪个快”当唯一判据。正确顺序是先定瓶颈，再定工具。

---

## E13

**参考答案**

1. 因为 `DG_JIT` 这一族是**吞吐型**算子：如果 L2 命中，B 就不会真的从 HBM 读，
   带宽数字会虚高，结论也会错。冷 L2 的做法见
   `deep_gemm/testing/bench.py:263-266`（分配 2×L2 大小的缓冲）与
   `:334-346`（每次迭代前 `zero_()` 把它写一遍，把上一轮的 B 挤出去）。
   换成热 L2：`Memory bandwidth` 会明显变化（很可能下降，因为大量访问命中 L2
   而不是 HBM）；`Computation` 基本不变（Cube 的吞吐与缓存无关）；
   `Hardware limit` 完全不变（它是硬件模型，不是测量）。
2. 可以做的三件事：① **逐行核对**引用的源码与行号（本章全部结论都是这么来的）；
   ② 用**纯 CPU 可复算的整数模型**把结论钉住（本章 `code/02`、`code/03`）；
   ③ 把每个数字的**口径**写清（硬件、版本、工具、缓存状态），
   并显式标注“仅静态阅读 / 未运行”。
   不可以做：把没跑过的结果写成跑过、把 README 的数字说成自己测的、
   或给一个无法追溯的口径（第 19 章、本章开头的验证边界都是这个纪律）。

