# 第 35 章 DeepGEMM-Ascend：把 GEMM 压到硬件极限

## 学习目标

读完本章，你应当能够：

1. 说清 DeepGEMM-Ascend「完全兼容 DeepGEMM 的 API」这句话**到底兼容了什么**：
   同一套包名与函数名、同一种 `(A, sfA)` 调用约定、同一族算子的语义；以及**唯一**被
   官方标注为不同的地方——昇腾上 scaling factor（下称 SF）的存储格式（每两个 UE8M0
   打包成一个 `int16`、按 MN-major 存放），并能写出这个打包关系。
2. 指认出「MAD 原语抽象」这一层在代码里的**具体位置**（`deep_gemm/include/deep_gemm/ascend/`
   下的 `tile_ptr` / `copy` / `sync` 三个头），并逐项说出它替使用者挡掉了哪四类琐碎工作：
   fractal（NZ）布局、对齐约束、地址计算、参数转换；同时能把它与第 25 章讲的 TileLang
   昇腾后端摆在一起，回答「同一层抽象、两种做法」各自的做法是什么。
3. 读懂 JIT 与启发式这条链：`GemmDesc → GemmConfig → kernel 源码 → DeepJIT 编译/缓存 →
   launch`，能指出 `heuristics.hpp` 里的硬件模型（`HWDesc`）与代价函数分别在看哪些量，
   并说清 `set_num_sms` / `set_npu_arch` 与 `DG_JIT_*` / `DG_PRINT_CONFIGS` 的作用。
4. 对本章的六类算子各说出它的**瓶颈类型**（算力、HBM 带宽、FixPipe、通信），
   并会用表格里的「Computation」与「Memory bandwidth」两列做判断：为什么同一个
   FP8 GEMM，`M=128` 时是带宽问题、`M=4096` 时是算力问题。
5. 读一遍由 TileLang 实现的 `hc_prenorm.py`，说清这个「瘦长 GEMM + 平方和」为什么
   用 TileLang 就够、它靠什么把 28672 长的 K 拆开、以及它为什么**没有**落到
   Ascend C 那条路上。
6. 拿到一个昇腾 GEMM 需求时，能按本章的决策表判断该走 TileLang 还是该下钻到
   Ascend C + MAD 原语，并给出至少两条依据。

## 前置知识

* 第 14 章：GEMM 的**性能账本**（FLOPs、字节、算术强度）与**五级阶梯**。本章不再重复
  这些定义，直接使用「算术强度」「compute-bound / memory-bound」「tile 尺寸决定片上
  访存」这三个结论。
* 第 25 章：昇腾后端与硬件特性。Cube（AIC）/Vector（AIV）、GM/L1/L0A/L0B/L0C/UB 的
  容量与角色、MTE1/MTE2/MTE3/FixPipe 四条搬运流水线、`T.gemm` 只会降到 L0 MAD，
  以及「同步是 pipe 到 pipe 的事件」。
* 第 16 章：MoE 的四段结构（router / dispatch / grouped GEMM / combine）。本章的
  M-Grouped GEMM 与 MegaMoE 直接建立在这张图上。
* 第 26 章：TileLang 自己的运行时与 JIT（`@tilelang.jit`、磁盘缓存 key、host ABI、
  `torch` 适配器）。本章第 4.6 节会把它与 DeepJIT 对照着读。
* 第 27 章：fp8 / fp4 / MX block scaling / E8M0（UE8M0）的基本概念。本章会用到
  「granularity 1×32」「每 32 个 K 元素一个 scale」这些说法。

不需要你会写 Ascend C，也不需要你有昇腾硬件。第 2.3 节会读 MAD 抽象层的头文件，
第 6 节会读一段 TileLang kernel——两段都可以只靠源码读懂。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守三条硬规矩：

1. **本章所有代码结论一律标注「静态阅读」**。引用的 DeepGEMM-Ascend 源码全部逐行读过，
   行号与文件内容一致；但**没有任何一行 C++ 被编译过、没有被任何 NPU 执行过**。
2. **性能数字全部来自 README 的表格**，并写清测量条件：硬件 `Ascend 950DT`、
   CANN `9.20`、工具 `bench_msprof`、**冷 L2**（每次迭代前把 2 倍 L2 大小的缓冲区清零）、
   形状按 DeepGEMM test suite
   （来源：`DeepGEMM-Ascend/README.md:84-87`）。本章**没有一条实测数字**；
   凡是「本书复算」的地方都明确写出是复算而不是测量。
3. 本章出现的量只有三类来源：
   * **官方表格**（上一条）；
   * **本书复算 / 推导**——`code/02-perf-ledger/` 与 `code/03-config-search/` 里的纯
     Python 程序，能在写作机上跑出结果，正文会标注「本书复算」；
   * **源码事实**——带 `文件:行号`，标注「静态阅读」。

三个记号先约定好：

| 记号 | 指什么 |
| --- | --- |
| `DeepGEMM-Ascend/…` | corpus 里的 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/…`；站点会把它换成上游 GitHub 链接（commit `8491bbb4b8c0`） |
| `DeepGEMM/…` | corpus 里的 NVIDIA 侧对照物 `…/raw/DeepGEMM/…`（commit `057ca5964aae`） |
| `DeepJIT/…` | corpus 里的 `…/raw/DeepJIT/…`（commit `3732a3b9a0a2`） |

**本章与第 25 章的关系**：第 25 章讲「TileLang 自己怎么把 TileLang 编译到昇腾」；
本章讲「一个生产级 GEMM 库为什么选择**绕开**编译器、直接把芯片摊开写」，以及这条路
与 TileLang 的分工边界在哪里。两条路都是昇腾的路，不是二选一的世界观之争。

---

## 1. 问题的由来：TileLang 什么时候不够用

### 1.1 一个具体场景

第 14 章的五级阶梯把 GEMM 从「朴素分块」推到 `T.gemm` + 流水 + layout 注解，
第 25 章又说明了 `T.gemm` 在昇腾上最终会降成 L0 上的一条 MAD 指令。那么最自然的问题
是：**既然编译器已经能生成 L0 MAD，为什么还要有一个手写 MAD 的库？**

DeepGEMM-Ascend 在 README 第一段就给了自己的理由（`DeepGEMM-Ascend/README.md:3-11`）：

* 它对昇腾的**矩阵乘加原语（MAD）**做了一个**轻量抽象**，隐藏 fractal 布局、
  对齐约束、地址计算、参数转换这些细节；
* 它大量使用昇腾特有的优化手段，例如**稀疏数据加载**、**基于协程的流水线**；
* 它把这些实现作为「昇腾上极致性能优化」的参考；
* 官方给出的成绩是「在很宽的矩阵形状范围内达到硬件极限性能」——第 5 节会逐张表核对
  「硬件极限」这个说法。

注意这里的措辞：**没有一句是说「TileLang 不好」**。官方对 TileLang 的表态在
Acknowledgements 里（`DeepGEMM-Ascend/README.md:190`）：

> The JIT runtime is provided by DeepJIT ... **The mHC kernel is backed by Tilelang.**

也就是说：**同一个仓库里，两种方法并存**。GEMM / MQA logits / MegaMoE 走 Ascend C +
MAD 抽象；mHC 的 prenorm GEMM 走 TileLang。本章要回答的正是「为什么在同一份代码里
要分两路走」，边界画在哪里。

### 1.2 本章要回答的五个问题

1. **API 契约**：所谓「完全兼容 DeepGEMM 的 API」，兼容到什么程度？调用方需要改什么？
2. **抽象层次**：MAD 原语抽象包了什么？它和第 25 章讲的 TileLang 昇腾方言是**同一层**
   还是两层？「同一层抽象、两种做法」分别是什么做法？
3. **JIT 与启发式**：一个 shape 进来，怎么变成一组 tile 参数，再变成一段 kernel 源码，
   最后变成一次 launch？
4. **瓶颈分类**：六类算子的瓶颈分别在算力、带宽、FixPipe、还是通信？怎么从性能表读出来？
5. **决策**：什么情况下该用 TileLang，什么情况下必须下钻到 Ascend C + MAD？

### 1.3 一条重要的写作纪律

本章读的材料是一个**已经做到 98%–99.8% 硬件利用率**的库。读这种代码最大的风险是
「照抄结论而不看条件」：看到 99.8% 就以为「Ascend C 一定比 TileLang 快」。

正确的读法是相反的：**先问这个数字是在什么形状、什么量化格式、什么口径下得到的**，
再看它用了哪些手段才拿到。第 5 节会逐张表做这件事，第 7 节才据此给出决策表。

---

## 2. 概念与模型

### 2.1 package 名相同意味着什么

最先把一个容易误读的点钉住。DeepGEMM（NVIDIA）与 DeepGEMM-Ascend 的 Python 包名
**都是 `deep_gemm`**（`DeepGEMM-Ascend/README.md:9`：「DeepGEMM Ascend 和 DeepGEMM
使用相同的包名」）。这意味着：

* 调用方的 import 语句不变：`import deep_gemm`；
* 调用方调用的函数名不变：`deep_gemm.fp8_gemm_nt(...)`、`deep_gemm.bf16_gemm_nt(...)`、
  `deep_gemm.m_grouped_fp8_gemm_nt_contiguous(...)` 等；
* 参数顺序与关键字名不变（这是「API 兼容」里最容易漏掉、也最容易被上游改动打坏的部分）。

Ascend 版里这些名字是**逐条注册**出来的，注册点集中在 `csrc/python_api.cpp:30-42`，
它把 8 个命名空间里的注册函数挂到同一个扩展模块 `_C` 上：

```text
deep_gemm::config::register_apis(m);        // set_num_sms / set_npu_arch / ...
deep_gemm::epilogue_class::register_apis(m);
deep_gemm::gemm_api::register_apis(m);      // fp8_fp4_gemm_* / bf16_gemm_* / m_grouped_* / k_grouped_*
deep_gemm::attention_api::register_apis(m); // fp8_fp4_mqa_logits / paged 版
deep_gemm::einsum::register_apis(m);
deep_gemm::layout::register_apis(m);        // transform_sf_into_required_layout
deep_gemm::aclnn_api::register_apis(m);     // aclnn_* 参考实现
deep_gemm::mega_moe::register_apis(m);
```

（来源：`DeepGEMM-Ascend/csrc/python_api.cpp:30-42`；扩展模块名由 `TORCH_EXTENSION_NAME`
定义为 `_C`，同文件 `:25-27`。）Python 侧再把这些名字重新导出为 `deep_gemm.<name>`
（`DeepGEMM-Ascend/deep_gemm/__init__.py:25-79`），并额外补上 `tf32_hc_prenorm_gemm`
这个 TileLang 算子（同文件 `:100-101`）。

**兼容的边界**：README 说「kernel API 见 DeepGEMM 的接口文档」
（`DeepGEMM-Ascend/README.md:52-53`），也就是说**兼容承诺覆盖的是 kernel 接口**，
不是工具函数的全集。事实上两边的工具函数并不一样：

NVIDIA 版有 `set_tc_util`（张量核利用率）、`set_pdl`（Programmatic Dependent Launch）、
`set_ignore_compile_dims`、`set_block_size_multiple_of`、
`get_mn_major_tma_aligned_tensor` 等（`DeepGEMM/README.md:146-160`）；Ascend 版换成
`set_npu_arch`、`set_mk_alignment_for_contiguous_layout`、
`transform_k_grouped_sf_into_required_layout`、`get_paged_mqa_logits_metadata` 等
（`DeepGEMM-Ascend/README.md:54-60`）。两边共有的是 `set_num_sms` / `get_num_sms`、
`use_deterministic_algorithms`、`transform_sf_into_required_layout` 这几个
（逐条对照两份 README 的工具函数清单）。

### 2.2 唯一被官方标注的不兼容点：SF 的存储格式

这是全章最需要精确的一处。README 里唯一带 `NOTE` 标号的不兼容说明是
（`DeepGEMM-Ascend/README.md:48-50`）：

> The scaling factor format on Ascend differs from NVIDIA's: **each pair of UE8M0
> scaling factors along the K dimension is packed into an `int16`, and the packed
> values are stored in MN-major order** for optimal hardware efficiency.

NVIDIA 侧对照：SM90 用 FP32 scale；SM100 用打包 UE8M0，**4 个 UE8M0 打进一个
`torch.int`**（`DeepGEMM/README.md:65-68`）。所以差异不只是「打包率由 4 变 2」，
还有**主序方向**：NVIDIA 要求 MN-major + TMA 对齐（`DeepGEMM/README.md:95-104` 的
`get_mn_major_tma_aligned_tensor`），昇腾也要求 MN-major，但打包单位与对齐要求不同。

这个格式在代码里可以逐行核对，三处互相印证。

**第一处：输入校验**。`check_sf_layout` 规定 SF 输入必须是 fp32 或 `short`，
`gran_k` 只能是 32，短整型输入时最后一维的长度是 `ceil(k/(32*2))`
（`DeepGEMM-Ascend/csrc/apis/layout.hpp:37-65`）：

```cpp
constexpr uint32_t sf_pack_k = 2;                                   // 两个 ue8m0 一个 int16
const auto shape_div_k = sf.scalar_type() == at::kFloat ? gran_k : (gran_k * sf_pack_k);
DJ_HOST_ASSERT(sf.size(-1) == ceil_div(k, shape_div_k));
DJ_HOST_ASSERT(gran_k == 32, "gran_k must be 32");
```

**第二处：接口注释里的输出形状**（`csrc/apis/layout.hpp:19-36`）：

> Output is always **short** (packed ue8m0x2), **MN-Major**, and repeated for gran_mn
> if needed. … sf: `[..., MN, ceil(K / gran_k / 2)]`, short, MN-Major。

**第三处：输出怎么分配**。`launch_transform_sf` 先把物理形状设成
`[..., dst_shape_k, dst_shape_m]`（MN 在最内层、连续），再转置后交出去
（`csrc/jit_kernels/transform_sf.hpp:45-55`）：

```cpp
const uint32_t dst_shape_k = ceil_div(src_shape_k, is_float_input ? 2 : 1);
physical_sizes[size-2] = dst_shape_k;   // K 方向被压成一半（两个 ue8m0 = 1 个 int16）
physical_sizes[size-1] = dst_shape_m;   // MN 是连续维
auto output = torch::empty(physical_sizes, ...).transpose(-2, -1);
```

于是取回来的张量形状是 `[MN, K/64]`、且 `stride(-2) == 1`（MN 连续）——这正是
`Major::MN` 的定义（`DeepGEMM-Ascend/csrc/utils/layout.hpp:44-47`：
`get_major_type_ab` 用 `stride(-1)==1 ? Major::K : Major::MN`）。

**打包关系**（可以用两行记住）：

```text
sf_out[m, p] : int16  ==  (sf[m, 2p] << 8) | sf[m, 2p+1]     每个 sf 是一个 ue8m0 字节
m ∈ [0, MN)，p ∈ [0, ceil(K/32/2)) = [0, ceil(K/64))
物理内存按 m 连续（MN-major），地址 ≈ (p * MN + m) * 2 字节
```

这个合并在 device 侧的 SIMD 代码里写得非常直白：`pack_validate_mn_major_simd_impl`
把沿 MN 相邻两行的 fp32 分别 `shiftright 23` 与 `shiftright 15`，再 `or` 到一起
（`DeepGEMM-Ascend/deep_gemm/include/deep_gemm/transform_sf.hpp:251-280`）：

```cpp
asc_shiftright_scalar(sf0, sf0, 23, mask_all_b32);   // 取 fp32 的指数位（ue8m0）
asc_shiftright_scalar(sf1, sf1, 15, mask_all_b32);   // 第二个 scale 放到低 8 位
asc_or(sf_packed, sf0, sf1, mask_all_b32);           // 合成一个 16 位
asc_storealign_pack_postupdate(dst, sf_packed, store_offset, mask_all_b32);
```

（fp32 输入先被校验：指数以外的 23 位尾数必须为 0，否则直接报错，见
`deep_gemm/include/deep_gemm/transform_sf.hpp:13-21` 的 `check_validation`；
`MX_SF_DIVISOR = 64` 与 `sf_elements_per_pair = 64` 定义在
`deep_gemm/include/deep_gemm/common.hpp:64` 与 `csrc/jit_kernels/heuristics.hpp:69`。）

![图 1：SF 的 int16 打包与 MN-major 布局](figures/01-sf-layout-int16-mn-major.md)

（源描述见 [`figures/01-sf-layout-int16-mn-major.md`](figures/01-sf-layout-int16-mn-major.md)；
SVG 规格 `src/tutorial/figures/35-deepgemm-ascend/01-sf-layout-int16-mn-major.json`。
上半格表示「逻辑上每个 K 方向相邻的两个 ue8m0」；下半格表示「物理内存里
MN 是连续维」，两者合成一个 `[MN, ceil(K/64)]` 的 int16 矩阵。）

**读者最容易踩的一脚**：`tests/test_mega_moe.py:22-36` 造 SF 的方式是
「先生成 `uint8` 的指数数组 `[rows, cols/32]`，再 `.view(torch.int16)`」。这不是类型
转换，而是**重新解释**：字节数不变，元素数减半，正好完成「两字节一个 int16」的打包。
自己造输入时如果写了 `sf.to(torch.int16)`，得到的是数值转换而不是打包，结果一定错。

**还有两处「读源码比读 README 更准」的差异**（都不是数值语义不同，而是**没实现**）：

* masked grouped GEMM 在 Ascend 后端**没有实现**：`csrc/apis/gemm.hpp:591` 与 `:615`
  各留了一行注释「`m_grouped_*_nt_masked` is not implemented in this backend」；
  NVIDIA 版则有 `m_grouped_fp8_gemm_nt_masked`（`DeepGEMM/README.md:82-88`）。
* `fp4×fp8`（A 是 fp4、B 是 fp8）**不再支持**，支持的是 `fp8×fp4`、`fp8×fp8`、
  `fp4×fp4` 三种组合（`csrc/apis/gemm.hpp:184-188`；`csrc/apis/aclnn.hpp:96-104`
  用一张表列出 ACLNN 侧的支持矩阵）。

### 2.3 MAD 原语抽象：三个头文件，四类脏活

README 说它「对昇腾的矩阵乘加原语提供了一个轻量抽象，隐藏分形矩阵布局、对齐约束、
地址计算、参数转换等细节」（`DeepGEMM-Ascend/README.md:5-9`，英文原文同段）。
这层抽象就是三个头文件，入口是 `deep_gemm/include/deep_gemm/ascend.hpp:1-12`，
它自己把分工写在注释里：

```cpp
// Device primitives, split by abstraction layer:
//   tile_ptr — operand_ptr / result_ptr (shape-carrying device pointers) + l*_ptr aliases
//   copy     — GM/L1/L0/UB data movement, scale-factor load, mad, quant-mode inference
//   sync     — intra-block / flag / ffts synchronization
```

**（1）`tile_ptr.hpp`：把「裸地址」换成「带形状的指针」。** 这类抽象解决的是
「地址计算」与「对齐约束」。`operand_ptr` 里存的是 `addr + shape_mn + shape_k`，
而不是一个裸指针（`ascend/tile_ptr.hpp:19-57`）：

```cpp
template <typename ptr_t, typename elem_t, Major MAJOR = Major::K>
struct operand_ptr {
    uintptr_t addr;
    const uint64_t shape_mn, shape_k;   // GEMM operand dimensions (logical, layout-invariant)
    static constexpr Major major = MAJOR;
    static constexpr uint32_t k_per_slot = is_fp4<dtype_t>() ? 2u : 1u;
    ...
    constexpr __aicore__ operand_ptr as_mad_aligned(uint64_t shape_mn, uint64_t shape_k) const {
        constexpr auto FRAC_K = get_frac_k<dtype_t>();
        return operand_ptr<ptr_t, elem_t, MAJOR>(aligned(shape_mn, FRAC_MN), aligned(shape_k, FRAC_K), addr);
    }
};
```

注意 `as_mad_aligned`：它把形状**向上对齐**成 MAD 能接受的尺寸（MN 对齐到
`FRAC_MN = 16`，K 对齐到 `get_frac_k<T>()`，见
`deep_gemm/include/deep_gemm/common.hpp:62,66-79`）。调用方只管写「我要算 M×K」，
「MAD 实际要多大」由这一层算——这就是「对齐约束」被隐藏掉的意思。
四个别名一目了然地标出这块内存住在哪一级
（`ascend/tile_ptr.hpp:93-97`）：

| 别名 | 限定符 | 物理位置 |
| --- | --- | --- |
| `l1_ptr<T, M>` | `__cbuf__` | L1（Cube 的输入中转） |
| `l0a_ptr<T>` / `l0b_ptr<T>` | `__ca__` / `__cb__` | L0A / L0B |
| `l0c_ptr<T>` | `__cc__` | L0C（fp32 累加器） |
| `ub_ptr<T>` | `__ubuf__` | UB（Vector 侧） |

**（2）`copy.hpp`：把「一次搬运」写成带语义的函数。** 它覆盖了 GM/L1/L0/UB 之间的
所有通路，并且**在函数内部**处理 ND↔NZ 格式转换、半字节打包（fp4 是 2 个/字节）、
转置、尾块填充、L2 cache hint。几个代表性条目：

| 抽象 | 位置 | 它替你做了什么 |
| --- | --- | --- |
| `copy_gm_to_l1(...)` | `ascend/copy.hpp:102-132` | K-major 走 `asc_copy_gm2l1_nd2nz`（ND→NZ 分形）；MN-major 走转置版本；fp4 时 `pack_factor = 2` 自动减半 |
| `copy_l1_to_l0a(...)` | `ascend/copy.hpp:134-145` | 判断要不要 transpose，并处理 8-bit 类型下 `m_step` 必须偶数的约束 |
| `copy_l0c_to_ub/gm(...)` | `ascend/copy.hpp:266-310` | 收尾：NZ→ND、量化模式（`asc_quant_mode`）、ReLU、是否需要 transpose |
| `fill_tail_block(...)` | `ascend/copy.hpp:180-198` | 尾块补零（`aligned(actual_k, 64)` 之外的部分） |
| `infer_quant_mode<T, ST>()` | `ascend/copy.hpp:232-264` | 由「输出 dtype × 累加器 dtype」**编译期**推出量化模式（`F322BF16`、`VQF322FP8_PRE`、`REQ8` …） |

`infer_quant_mode` 是「参数转换」这条脏活最典型的例子：Ascend C 的 L0C→GM/UB
搬运要带一串枚举参数（量化模式、ReLU 前处理模式、unit flag 模式），手工填非常容易错。
抽象层把它降成**模板参数推导**——你写 `copy_l0c_to_gm<bf16, float>(...)`，
模式就被推出来了（默认参数见 `ascend/copy.hpp:266-300`）。

**（3）`sync.hpp`：把「裸 flag id」换成「有方向的类型」。** `PipeEvent<kSrc, kDst>`
是一个**只带一个整数 base** 的空结构，但它的模板参数写明了这条事件的 PIPE 方向
（`ascend/sync.hpp:11-27`）：

```cpp
template <pipe_t kSrc, pipe_t kDst> struct PipeEvent { uint32_t base; };
template <pipe_t kSrc, pipe_t kDst, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void notify(PipeEvent<kSrc, kDst> event, uint32_t stage_idx = 0);
template <pipe_t kSrc, pipe_t kDst, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void wait(PipeEvent<kSrc, kDst> event, uint32_t stage_idx = 0);
```

`notify` / `wait` 最终落到 `asc_sync_notify(kSrc, kDst, event)` /
`asc_sync_wait(...)`（同文件 `:16-27`），也就是第 25 章讲的 Set/Wait flag。
抽象带来的两点收益是**可读性**（事件常量写成 `kL1AFull{0}`、`kSFEmpty{3}` 这种名字，
   见第 5.6 节 MegaMoE 的事件表）与**防止方向写反**（源/目的 PIPE 是类型的一部分）。

**（4）`mad(...)`：整个抽象层的出口，只有一行。** 所有 GEMM 最终都收敛到
`deep_gemm/include/deep_gemm/ascend/copy.hpp:371-380`：

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

两件事值得记住：

* **8-bit / 4-bit 走 `asc_mmad_mx`**（MX = micro-scaling，带 scale 的形态），
  16-bit（bf16）走 `asc_mmad`。判据就是 `sizeof(TA)==1 || sizeof(TB)==1`
  ——fp8 是 1 字节、fp4 打包成 1 字节，都落进 MX 分支。
* `clear_C` 是**累加器要不要清零**。它是显式参数，不是自动判断——对应第 19 章讲的
  「L0C 不会自动清零」这个坑。在 GEMM 主循环里它一般取 `k == 0`（第一节 K 块清零），
  例如 `deep_gemm/include/deep_gemm/fp8_gemm.hpp` 的 K 循环里就是按这个语义传的。

### 2.4 一张图看清「同一层、两种做法」

把第 25 章和本章的内容叠在一起看，昇腾上做 GEMM 有三层可以站：

![图 2：昇腾 GEMM 的四层抽象与两条实现路线](figures/02-abstraction-layers.md)

（图 2 的源描述见 [`figures/02-abstraction-layers.md`](figures/02-abstraction-layers.md)，
SVG 规格见 `src/tutorial/figures/35-deepgemm-ascend/02-abstraction-layers.json`。）

读这张图的三个要点：

1. **MAD 原语抽象和 TileLang 的昇腾方言站在同一层**：两者都是「让使用者写 tile 级的
   数据流，而不是写 flag 和地址」。这就是本章题眼里的「同一层抽象」。
2. **但两者往下走的方式不同**：TileLang 往下是**编译期**——`T.copy` / `T.gemm` / 流水注解
   经过 AutoSchedule、InsertSync、codegen 变成 Ascend C 或 PTO
   （第 25 章 §4）；DeepGEMM-Ascend 往下是**运行期字符串拼接**——C++ 里用
   `std::format` 拼出一小段 `#include` 头文件的 C++ 源码，交给 DeepJIT 现场编译
   （第 4 节）。一个把「怎么降级」交给编译器，一个把「已经降好的模板」交给模板展开。
3. **两者的产物都在同一层**：都是 Ascend C / bisheng 能编译的设备代码。所以它们不是
   互斥的技术，而是**在同一层上的两种工程选择**——这正是第 7 节决策表的立足点。

---

## 3. 用法：一次调用的全流程

### 3.1 最小调用（照 `tests/` 的写法）

本节的代码照抄官方测试的调用姿势，完整可复制的版本在
[`code/01-python-api/`](code/01-python-api/)（**仅静态核对，未运行**）。

先说一个**最容易踩的前提**：DTK/CANN 的 home 由 `ASCEND_HOME_PATH` 或
`ASCEND_TOOLKIT_HOME` 给出（`DeepGEMM-Ascend/README.md:62`）；包在 import 时读这两个
环境变量，读不到就退回 `/usr/local/Ascend/ascend-toolkit/latest`
（`deep_gemm/__init__.py:15-17`）。**这个探测发生在 import 时**，所以环境变量必须在
`import deep_gemm` 之前设好——否则你会得到一个「编译命令里找不到 bisheng」的报错，
而不是一句清楚的「路径没设」。

BF16 是最短的入口，因为不需要 scale factor：

```python
import torch, torch_npu, deep_gemm

m, n, k = 4096, 7168, 16384
a = torch.randn(m, k, dtype=torch.bfloat16, device='npu')
b = torch.randn(n, k, dtype=torch.bfloat16, device='npu')   # NT：B 是 [N, K]
d = torch.empty(m, n, dtype=torch.bfloat16, device='npu')

deep_gemm.bf16_gemm_nt(a, b, d)          # d = a @ b.T
```

FP8 版本多一个 SF 参数，调用约定是**(数据, scale) 元组**（
`csrc/apis/gemm.hpp:190-236` 的 `fp8_fp4_gemm` 签名）：

```python
import deep_gemm, torch

# A: [M, K] fp8_e4m3 ; sfA: [ceil(M/gran_m), ceil(K/32)]  fp32 或 int16（打包 ue8m0）
# B: [N, K] fp8_e4m3 ; sfB: [ceil(N/gran_n), ceil(K/32)]
# D: [M, N] bf16 或 fp32
deep_gemm.fp8_gemm_nt((a, sf_a), (b, sf_b), d, recipe_a=(1, 32), recipe_b=(1, 32))
```

这里有几个名字必须先认准（都能在 `csrc/apis/gemm.hpp:184-241` 的注释里逐条对上）：

* **后缀 `nt/nn/tn/tt` 说的是逻辑形状**，不是内存布局。`nt` = A 不转置
  `[M,K]`、B 转置 `[N,K]`；`tn` = A `[K,M]`、B `[K,N]`。注册表见
  `csrc/apis/gemm.hpp:539-570`。
* **`recipe` / `recipe_a` / `recipe_b` 是量化粒度**：三元的 `(gran_m, gran_n, gran_k)`
  与两个二元的 `(gran_mn, gran_k)` 互斥；不传时默认 `(1, 32)`
  （`csrc/apis/layout.hpp:113-128` 的 `get_recipe`）。`gran_k` 只能是 32。
* **SF 可以传 fp32 或打包好的 `int16`**；传 fp32 时只有 8 位指数被使用，
  23 位尾数必须为 0，否则 kernel 直接 panic
  （`csrc/apis/gemm.hpp:193-200` 的 NOTE，与 device 侧 `check_validation` 对应）。
* **`d` 是输出、`c` 是可选累加项**：传了 `c` 就算 `C += A@B`，否则算 `D = A@B`
  （`csrc/apis/gemm.hpp:201-202`）。

MoE 的 grouped 形态是 `m_grouped_*_contiguous`，多一个「每个专家分到多少行」的
`grouped_layout`（`[G]` int32，前缀和语义）：

```python
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout)   # b: [G, N, K]
deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a, sf_a), (b, sf_b), d, grouped_layout)
```

两个约束要记住：**M-grouped 必须用 `use_psum_layout=True`**（默认就是 True，
见 `csrc/apis/gemm.hpp:571-590`），**并且 BLOCK_M 固定为
`get_mk_alignment_for_contiguous_layout()` = 256**（`csrc/jit/jit.hpp:59-69`
把 `kMkAlignment` 写死成 256，`set_mk_alignment...` 想改成别的值会直接 panic；
heuristics 里对 M-grouped 强制 `block_m = 256`，见
`csrc/jit_kernels/heuristics.hpp:124-130`）。所以每个专家的行数必须对齐到 256，
不足的部分由调用方补零。

### 3.2 工具函数：哪些是「设一次管全局」的

工具函数的全集在 `DeepGEMM-Ascend/README.md:54-60`，实现与注册在
`csrc/apis/runtime.hpp:12-58`（config 类的部分）和 `csrc/jit/jit.hpp:18-74`
（`Runtime` 类）。挑出四个真正会改变行为的：

| 函数 | 作用 | 源码 | 默认行为 |
| --- | --- | --- | --- |
| `set_num_sms(n)` / `get_num_sms()` | 限制 kernel 可以用的 AI Core 数（对齐到 `kNumAICores=32` 的上限） | `csrc/jit/jit.hpp:34-41` | `0` 表示「问设备要」 |
| `set_npu_arch(n)` / `get_npu_arch()` | 覆盖/查询 JIT 编译用的 `dav-*` 架构号 | `csrc/jit/jit.hpp:25-32` | `0` 表示「问设备要」，取自 `ACL_DEV_ATTR_NPU_ARCH`（`DeepJIT/include/deep_jit/backend/ascend/device.hpp:49,56`） |
| `use_deterministic_algorithms(bool)` | 打开后，K 方向被 split 的 kernel 改成「先在核内规约、再原子写」，结果逐位可复现 | `csrc/jit/jit.hpp:51-57`；消费方见 `hc_prenorm.py:95-102` | 关 |
| `get_theoretical_mk_alignment_for_contiguous_layout()` | 返回 256，供调用方给 MoE 行数补零用 | `csrc/jit/jit.hpp:71-73` | 恒为 256 |

**注意一处与本章任务描述不符的地方，需要更正**：在锁定快照里**没有**
`DG_NPU_ARCH` 这个环境变量。架构只有两个来源——设备上报（`get_npu_arch()`），
或者 Python 侧显式 `set_npu_arch(int)`。作者在全仓库里 grep 过
`DG_NPU_ARCH`，无命中；唯一与 arch 相关的源码是上面那三处。
如果官方后续加了同名环境变量，请以源码为准（登记见 [`references.md`](references.md)
的「待核」小节）。

### 3.3 环境变量：`DG_JIT_*` → `DJ_JIT_*` 的回退

环境变量的清单在 `DeepGEMM-Ascend/README.md:62-78`，语义定义在
`DeepJIT/include/deep_jit/utils/env.hpp:56-80` 的 `Env` 类：**每个 `DG_JIT_*`
在未设时回退到对应的 `DJ_JIT_*`**（`Env("DG").get("JIT_CACHE_DIR")` 先找
`DG_JIT_CACHE_DIR`，再找 `DJ_JIT_CACHE_DIR`，见 `env.hpp:65-70`）。
而且它们**在 JIT 运行时第一次创建时被快照**（README 原话
「snapshotted when the JIT runtime is first created」）——也就是说，
`import deep_gemm` 之后再 `os.environ[...] = ...` 可能已经晚了。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `DG_JIT_CACHE_DIR` | `$HOME/.dj` | 编译产物缓存目录；可以是冒号分隔的多目录，**按顺序查找、命中即用，未命中写进第一个** |
| `DG_JIT_DEBUG` | `0` | 打开后打印编译命令、加载耗时、每个 shape 选中的 config（消费点见 `csrc/jit_kernels/heuristics.hpp:714-718`） |
| `DG_PRINT_CONFIGS` | `0` | 只打印每个 shape 的 config（与上一条共用同一个打印点） |
| `DG_JIT_PRINT_COMPILER_COMMAND` | `0` | 打印 bisheng / 反汇编命令 |
| `DG_JIT_PRINT_LOAD_TIME` | `0` | 打印 kernel 加载耗时 |
| `DG_JIT_KERNEL_DEBUG_INFO` | `0` | 给 bisheng 加调试信息 |
| `DG_JIT_LAUNCH_TIMEOUT` | `300`（秒） | 单次 launch 超时；`0` 表示不限 |
| `ASCEND_LAUNCH_BLOCKING` | — | 每次 launch 后同步设备，把异步 launch 错误变成行内报错（实现见 `DeepJIT/include/deep_jit/backend/ascend/kernel.hpp:219`） |

另外三个只影响单个 kernel 的变量不在 README 清单里，但确实存在，值得知道：
`DG_JIT_WITH_LINEINFO`（给编译加行号信息，`csrc/jit/jit.hpp:87-88`）、
`DG_TRANSFORM_SF_LD_CACHE_HINT` / `DG_TRANSFORM_SF_ST_CACHE_HINT`
（SF 转换 kernel 的 L2 提示，`csrc/jit_kernels/transform_sf.hpp:70-71`）、
以及测试专用的 `DG_TEST_MAX_WORKERS`（并行预编译的线程数，
`deep_gemm/testing/par_compile.py:27`）。

### 3.4 对拍手段：`aclnn_*` 参考实现

一个为了「把 GEMM 压到极限」而写的库，怎么证明自己是对的？答案是**再造一份「不求快、
只求是官方算子」的实现，然后把两者对齐**。这就是 `aclnn_*_gemm_*` 系列的用途
（README 原话：「ACLNN reference GEMMs used to cross-check the kernels in tests」，
`DeepGEMM-Ascend/README.md:59`）。

它的实现很短：把头文件里的 `aclnnQuantMatmulV5` / `aclnnMatmul` 包成和 kernel 同形的
Python 函数（`csrc/apis/aclnn.hpp:19-47` 的 `make_quant_operand`、
`:122-208` 的 `aclnn_fp8_fp4_gemm`、`:209-260` 的 `aclnn_bf16_gemm`），
注册名是 `aclnn_fp8_fp4_gemm_{nt,nn,tn,tt}` 与 `aclnn_bf16_gemm_{nt,nn,tn,tt}`
（`csrc/apis/aclnn.hpp:262-294`）。顺带一提，它还把 `cublaslt_gemm_*` 这两个
**NVIDIA 侧的名字**别名到同一批 ACLNN 实现上（同文件 `:291-294`）——这是
「API 兼容」在工具函数层面的又一个体现。

测试里对拍是这样用的（`tests/common.py:82-123`）：

```python
d = torch.randn_like(d_ref)
gemm_dg(test, a, b, c, d, ...)          # 被测：deep_gemm.<kernel>
torch.npu.synchronize()
has_ref = gemm_aclnn(test, a, b, c, d, ...)   # 参考：deep_gemm.aclnn_<kernel>
if has_ref is not False:
    assert_equal(d, d_ref, ..., threshold=...)  # BF16 用相似度，FP8 要求逐位相等
```

**这份对拍代码最有价值的地方不是「它测了」，而是「它明说了哪些情况不测」**。
`gemm_aclnn` 里连着四条提前返回（`tests/common.py:82-117`），每条都写清了原因：

1. 非 Normal（m-/k-grouped）**没有** ACLNN 对应算子，直接跳过；
2. `K` 不是 64 的倍数 → ACLNN 不支持；
3. `gran_mn != 1` 的 block scaling → ACLNN 的 MX 路径只支持 1×32，直接拒绝
   （「Unsupported groupSize」）；
4. `ceil(k/32) <= 2` 且某个操作数是 MN-major → ACLNN 会把 `[k/64, n, 2]` 的
   scale 前导维误读成 `n=1`，属于对方实现的缺陷，跳过。

第 3、4 条尤其值得记住：**「参考实现不支持」不等于「被测实现错了」**。一个成熟库的
测试会把这些差异显式登记出来，而不是靠调阈值蒙混过去。判据本身也分两档
（`tests/utils.py:18-30`）：

* `threshold == 0` → `torch.equal`，要求**逐位相等**（FP8 路径：同一条 Cube + cast
  路径，本来就该一样）；
* `threshold > 0` → 用余弦相似度导出的 `calc_diff`（BF16：Cube 与 ACLNN 的 K 方向
  累加顺序不同，最多差约 1 ULP，不该要求逐位相等，`tests/utils.py:8-15`）。

---

## 4. 实现机制：一次 shape 的旅程

### 4.1 五个阶段

![图 3：从一个 shape 到一次 launch](figures/03-jit-flow.md)

（源描述见 [`figures/03-jit-flow.md`](figures/03-jit-flow.md)；SVG 规格
`src/tutorial/figures/35-deepgemm-ascend/03-jit-flow.json`。）

五个阶段的落点都能在代码里找到：

| 阶段 | 输入 | 输出 | 代码位置 |
| --- | --- | --- | --- |
| ① 描述问题 | torch 张量 | `GemmDesc` | `csrc/apis/gemm.hpp:60-182`、`csrc/jit_kernels/gemm_config.hpp:39-95` |
| ② 选 config | `GemmDesc` | `GemmConfig` | `csrc/jit_kernels/heuristics.hpp:575-719` |
| ③ 生成源码 | `GemmConfig` | 一段 C++（`#include` + 模板实例化） | `csrc/jit_kernels/bf16_gemm.hpp:16-72`、`fp8_gemm.hpp` 同构 |
| ④ 编译 + 缓存 | 源码字符串 | 设备二进制 | `csrc/jit/jit.hpp:78-91` + DeepJIT |
| ⑤ launch | kernel + 参数 | 一次下发 | `csrc/jit_kernels/bf16_gemm.hpp:73-77` |

### 4.2 ① `GemmDesc`：把张量翻译成「问题」

`GemmDesc` 是一个**与硬件无关的问题描述**（`csrc/jit_kernels/gemm_config.hpp:39-95`）：
指针、dtype、`m/n/k`、主序（`major_a/major_b`）、外维 stride、是否累加、
epilogue 类型、gemm_type（Normal / M-grouped / K-grouped / Batched）、以及
`compiled_dims`（哪些维度要**编译期固化成常量**）。

它由 `get_gemm_desc()` 从 torch 张量推出来（`csrc/apis/gemm.hpp:60-182`），
推的关键是**主序**。主序不是猜的，是从 stride 里读的：
`get_major_type_ab` 检查「最后两维必须恰好有一个 stride 为 1」，
然后 `stride(-1)==1 → Major::K`，否则 `Major::MN`
（`csrc/utils/layout.hpp:37-47`）；输出 `d` 则强制要求 `stride(-1)==1`
（`check_major_type_cd`，同文件 `:49-52`）。如果最后两维 stride 都是 1，直接 panic
「Ambiguous major axis」——因为这时候「谁是内层」是歧义的，编译器不允许你蒙。

`compiled_dims` 值得单独说一句。默认值是 `"nk"`
（`csrc/apis/gemm.hpp:543`）：N 和 K 会被**编译期常量**烧进 kernel，M 保持运行期。
理由和 NVIDIA 版一致（`DeepGEMM/README.md:98` 的 `set_ignore_compile_dims`）：
MoE 推理时每个专家收到的 token 数（M）是动态的，N/K 是固定的。之后
`get_compiled_dim(dim, 'k', compiled_dims)` 负责把「要固化」的维度取出来，
没被选中的维度取 0，表示「用运行期的值」（`csrc/jit_kernels/gemm_config.hpp:16-21`）。
这个 0 会一直传到内核模板参数里（例如 `bf16_gemm.hpp:48-50`：
`SHAPE_K != 0 ? SHAPE_K : shape_k_runtime`）。

### 4.3 ② `GemmConfig`：十个数字，决定性能

`GemmConfig` 是**执行配置**（`csrc/jit_kernels/gemm_config.hpp:98-121`），一共几组：

* **tile 尺寸**：`block_m/block_n/block_k`（GM→L1 的数据块）、
  `mad_m/mad_n/mad_k`（L1→L0 的 MAD 块）；
* **流水深度**：`num_l1_stages`（A/B 在 L1 的级数）、`num_l0_stages`（L0 AB 级数）、
  `num_epilogue_stages`（UB 收尾级数）、`num_l1_sf_stages` + `sf_k_blocks`
  （SF 在 L1 的级数与每次搬多少 K 块）；
* **开关**：`direct_store`（L0C 直写 GM，绕过 UB）、`resident_a/resident_b/resident_sf`
  （A/B/SF 常驻 L1，不随 wave 重载）、`dual_aiv_dequant`（两个 AIV 一起做 fp4 反量化）；
* **L2 提示**：`l2_ctrl_a` / `l2_ctrl_b` / `l2_ctrl_store_cd`；
* **launch**：`deep_jit::ascend::LaunchOptions`（`num_blocks`、`num_ubuf_bytes`、
  `num_launch_timeout_secs`）。

打印出来的样子（`gemm_config.hpp:163-172` 的 formatter）是：

```text
GemmConfig(block=256x256x512, mad=128x128x256, l1_stages=3, l0_stages=2,
           epi_stages=8, l1_sf_stages=2, sf_k_blocks=4, l2_ctrl=.../.../...,
           direct_store=true, resident=.../.../..., dual_aiv_dequant=false, launch_options=...)
```

把 `DG_PRINT_CONFIGS=1` 打开，每个新 shape 第一次被编译时就会打印一行这样的配置
（打印点 `csrc/jit_kernels/heuristics.hpp:714-718`，用了 `std::unordered_set`
去重）。这是**读这个库最省事的入口**：不用猜它为什么快，先看它选了什么。

### 4.4 ③ 启发式：搜索空间不大，但每一步都有依据

选配置的主函数是 `select_gemm_config`（`heuristics.hpp:575-719`）。它的骨架是
「**枚举 → 过滤非法候选 → 用代价模型挑最小**」：

```text
for block_m in [m_alignment, min(512, align(m, m_alignment))] :   # 从大到小
    for block_n in [n_alignment, min(512, L0C/(4*block_m))] :     # 从大到小
        快速下界剪枝（memory / compute 任一超过当前最优就 continue）
        make_gemm_candidate(...) -> 可行？-> cost -> 优于最好就留下
```

（`heuristics.hpp:600-650`；`block_cap` 对 4-bit 或 dequant 路径是 256，其余 512，
见 `:587`。）

四个关键约束，它们解释了「为什么 tile 尺寸不能随便取」：

1. **L0C 放得下**：`block_m * block_n * 4B ≤ L0CSizeBytes`
   （`heuristics.hpp:100-102`，`L0CSizeBytes = 256KB`，
   `deep_gemm/include/deep_gemm/common.hpp:57`）。注意这里是**fp32 累加**，
   所以 `256×256` 就已经用掉 256KB 的一个整块了。
2. **MAD 块还要再切**：`mad_m * mad_n * 4B ≤ L0CSizeBytes / 2`
   （`heuristics.hpp:125`），留一半给双缓冲——这样 MAD 和 L0C→UB 的搬运能重叠。
3. **L0A/L0B 放得下且至少两级**：`mad_k` 由 L0A/L0B 的容量反推
   （`heuristics.hpp:131-135`）：
   `l0a_k = L0A*8/(num_l0_stages * mad_m * elem_bits)`，对 L0B 同理，
   再取两者与 `block_k/num_l0_stages` 的最小值，然后向下对齐到 `k_alignment`。
4. **L1 放得下 A/B/SF 的全部 stage**：`num_l1_stages` 由 L1 剩余容量除以
   「A+B 一级的字节数」得到（`heuristics.hpp:164-172`）；
   带 SF 时，L1 的 pipe event 还要在 operator 与 SF 之间**共享**，
   所以 stage 数被压到 `min(4, num_pipe_events - num_l1_sf_stages)`。

**代价模型在算什么**：`get_gemm_cost`（`heuristics.hpp:252-465`）把时间拆成
「搬 A/B/SF 的时间」+「L1→L0 的时间」+「MAD 的时间」+「收尾写出的时间」，
各自用 `HWDesc` 里的带宽/频率换算：

```cpp
// csrc/jit_kernels/heuristics.hpp:52-71（节选）
.gm_bw_gbps=3500, .l2_bw_gbps=4500, .aic_clock_mhz=1650,
.bf16_flops_per_cycle=8192,        // 4096 multiply-adds; each counts as two FLOPs
.gm_to_l1_bytes_per_cycle=256, .l1_to_l0_bytes_per_cycle=256,
.l1_to_l0_sf_bytes_per_cycle=32,   // Measured SF transfer rate.
.l0c_to_ub_bytes_per_cycle=128, .ub_to_l1_bytes_per_cycle=128,
.num_pipe_events=8, .num_intra_block_events=16, .num_ai_cores=32,
.l2_size_bytes=128*1024*1024, .l2_sector_bytes=kL2SectorBytes,
.sf_elements_per_pair=64, .fractal_row_bytes=32,
```

**这段硬件模型可以直接验证「硬件极限」那一列**。`get_mad_flops_per_cycle(elem_bits)`
的定义是（`heuristics.hpp:46-48`）：

```cpp
return bf16_flops_per_cycle * (sizeof(uint16_t)*8) / elem_bits;   // 16 / elem_bits
```

于是单芯片峰值 = `flops_per_cycle × 主频 × 核数`：

| dtype | 每核每周期 FLOPs | 复算峰值 | README 的 Hardware limit 列 |
| --- | --- | --- | --- |
| bf16 | 8192 | 8192 × 1650 MHz × 32 = **432.5 TFLOPS** | 432 |
| fp8 | 16384 | **865.0 TFLOPS** | 865 |
| fp4 | 32768 | **1730.1 TFLOPS** | 1730 |

（**本书复算**，脚本在 [`code/02-perf-ledger/`](code/02-perf-ledger/)；
`DeepGEMM-Ascend/README.md:95-104` 的表格列。）三者与 README 完全吻合——
这说明「Hardware limit」这一列**不是宣传口径，而是源码里那个硬件模型算出来的**，
后面的利用率才有意义。

**最后还有四个「补丁」**，是主搜索之后按代价择优打开的开关
（`heuristics.hpp:652-712`）：

* `direct_store`：满足对齐条件时，L0C 直写 GM 而不过 UB
  （`can_direct_store`，`heuristics.hpp:504-516`）；
* `resident_a/resident_b/resident_sf`：M 或 N 只有一块（或 K 被编译期固化）时，
  把 A/B/SF 常驻在 L1，省掉每个 wave 的重载（`heuristics.hpp:525-545`）；
* `dual_aiv_dequant`：fp8×fp4 路径上让两个 AIV 分摊 fp4 解码与 NZ 重排
  （`heuristics.hpp:549-557`、`:692-700`）；
* K 方向的 tile 再比较：当 K 被固化时，保持 M/N 不变、把 `block_k` 翻倍再比一次
  （`heuristics.hpp:655-676`）。

### 4.5 ④⑤ 生成源码、编译、launch

选定 config 之后，生成源码的做法**不是模板文件，而是字符串拼接**。以 bf16 为例
（`csrc/jit_kernels/bf16_gemm.hpp:16-72`）：

```cpp
auto config = select_gemm_config(desc);
const auto code = std::format(R"(
#include <deep_gemm/bf16_gemm.hpp>

namespace deep_gemm {{
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&bf16_gemm_impl<
        {}, {}, ...>);
}}
}} // namespace deep_gemm
)", desc.major_a, desc.major_b, /* …23 个参数… */);
const auto kernel = jit->compile("bf16_gemm", code);
jit->launch(kernel, config.launch_options, /* gm_ptr 包裹的指针与 stride */);
```

把参数列出来看，就明白为什么「`mad_m/mad_n/mad_k` 这些数字」值得单独成为一组配置：
**它们全部是模板参数**，会变成内核里的 `constexpr`。于是所有边界判断、对齐补齐、
尾块处理都在编译期完成，运行期没有分支——这是「简洁又高效」的机制来源。

launch 时传的是 `gm_ptr<T, Major>{outer_stride, addr, batch_stride}` 这种**带 stride
的句柄**（`common.hpp:133-182`），不是裸指针。原因在文档注释里：GM 上的操作数
允许是「主序方向一致、但外维 stride 不等于紧致尺寸」的视图
（`csrc/jit_kernels/gemm_config.hpp:68-75`），甚至可以是 `.permute()` 后的 batched
张量。内核侧拿到的就是「形状 + stride」，不需要知道它是怎么被 view 出来的。

### 4.6 缓存：DeepJIT 做的事，与第 26 章的 TileLang 对照

DeepJIT 是一个 **header-only 的 C++20 JIT 运行时**，同时支持 CUDA 与昇腾
（`DeepJIT/README.md:3-7`）。它负责的正是第 26 章讲 TileLang 时讲过的那些事：
源码与 include 的哈希、内存缓存、磁盘缓存、懒初始化、launch 选项。

| 维度 | TileLang（第 26 章） | DeepJIT（本章） |
| --- | --- | --- |
| 缓存 key | 由 IR、目标、pass config 等算出 | 源码 + 追踪到的 include + 编译器版本 + 编译选项 + **调用方给的 dependency signature**（`DeepJIT/README.md:12`） |
| 磁盘布局 | 目录 + 原子发布 | 唯一临时目录编译，`fsync` 后**原子 rename** 发布，多个进程可以同时编译同一条目并复用结果（`DeepJIT/README.md:36-44`） |
| 共享缓存 | 支持跨机器复用 | 同样支持；`DJ_JIT_CACHE_DIR=/a:/shared` 时按序查找、只往第一个写（`DeepJIT/README.md:38-52`） |
| 编译诊断 | dump IR、生成代码 | dump Ascend 汇编（`DG_JIT_DUMP_ASM`）、带行号编译（`DG_JIT_WITH_LINEINFO`） |
| 与框架的接口 | `@tilelang.jit` + `torch` 适配器 | pybind11 注册进 `_C`；启动用当前 `torch_npu` 流（`csrc/python_api.cpp:21-23`） |

一个值得注意的差异：**DeepJIT 的输入是「已经生成好的源码字符串」**，
所以它的缓存 key 里没有 IR 这一层；而 TileLang 的缓存 key 覆盖的是 IR 与 pass 配置。
这意味着两者的失效条件不同——改一行 C++ 头文件会让 DeepJIT 缓存失效，
改一个 `TL_*` pass 开关会让 TileLang 缓存失效。

**预编译**：因为「第一次调用某个 shape 会触发一次真实编译」，测试套件用
`par_compile()` 提前把要测的形状全部编出来（`deep_gemm/testing/par_compile.py:16-36`）。
它的机制很巧妙——在 `dry_run` 上下文里把 `set_dry_run(True)`，此时每个 kernel 的
`runtime->get_dry_run()` 为真，于是走 `compile_without_load()` 只编译不加载
（`bf16_gemm.hpp:67-70`、`transform_sf.hpp:92-95`、`runtime.hpp:46-54`）。

---

## 5. 六类算子，各卡在哪一条流水线上

本节是整章的分析核心。方法只有两步：**先看表格里的两个比值**（Computation ÷
Hardware limit、Memory bandwidth ÷ 芯片带宽），**再回去看代码里对应的机制**。

### 5.1 先把「口径」钉死

在读数之前必须先把三件事说清楚，否则 98% 会变成一个空的形容词。

**（1）硬件与软件版本**：`Ascend 950DT` + `CANN 9.20`；Requirements 里写明
需要 `bin/bisheng`、`bin/ld.lld`、`torch_npu`、Python ≥ 3.10、支持 C++20
`<format>` 的编译器（`DeepGEMM-Ascend/README.md:19-31`）。

**（2）工具与冷 L2**：官方用 `bench_msprof`，并且**冷 L2**——每次迭代前把一块
「2 倍 L2 大小」的缓冲区清零，从而把上一轮的 B 从 L2 里挤出去
（`deep_gemm/testing/bench.py:263-268` 的 `_make_l2_flush_buffer`，
`:334-346` 的 `flush_buffer.zero_()`；`flush_l2=True` 是 `bench_msprof` 的默认值，
`:289`）。这解释了为什么下面的带宽列能高到 3400 GB/s 量级：
如果 L2 命中，B 就不会真的从 HBM 读。
每个 shape 先跑 `num_warmups=10` 次预热、再测 `num_tests=30` 次取均值
（默认值 `bench.py:287-288`，循环在 `:330` 与 `:343`，均值由 `_mean_field`
在 `:202-221` 计算）。

**（3）利用率的分母是**「上面那个用 `HWDesc` 复算出来的硬件峰值」，
不是厂商宣传的型号峰值。三档分别是 bf16 432.5、fp8 865、fp4 1730.1 TFLOPS
（复算脚本见 `code/02-perf-ledger/`，公式见 `heuristics.hpp:46-48`）。

### 5.2 Dense GEMM：98.3%–99.8% 是什么意思

先从最干净的一张表看起（`DeepGEMM-Ascend/README.md:95-104`，`M=4096, N=7168, K=16384`）：

| Type | Latency (us) | Computation (TFLOPS) | Hardware limit | Utilization |
| --- | --- | --- | --- | --- |
| FP4×FP4 | 565.6 | 1701 | 1730 | 98.3% |
| FP8×FP4 | 1117.8 | 861 | 865 | 99.5% |
| FP8×FP8 | 1117.6 | 861 | 865 | 99.5% |
| BF16×BF16 | 2229.9 | 431 | 432 | 99.8% |

![图 4：五张性能表的利用率对照](figures/04-perf-utilization.md)

（源描述见 [`figures/04-perf-utilization.md`](figures/04-perf-utilization.md)；
SVG 规格 `src/tutorial/figures/35-deepgemm-ascend/04-perf-utilization.json`。）

读这张表要读出四件事：

1. **同一个形状，四种量化格式，算子强度差 4 倍**：bf16 2229.9us → fp8 1117.6us
   （正好一半）→ fp4 565.6us（再一半）。这与「每周期 MAD 数 ∝ 1/elem_bits」
   的硬件模型一致（`heuristics.hpp:46-48`）。
   **换句话说，这里没有「格式越低位越省时间」的玄学，就是同一台 Cube 的三种吞吐。**
2. **fp8×fp4 与 fp8×fp8 的延迟几乎相同**（1117.8 vs 1117.6us）。因为 B 的
   fp4 需要先在片上反量化再送 L0，反量化不是瓶颈，MAD 才是。
3. **99% 级别意味着「几乎没有被访存挡住」**。这个形状的算术强度极高：
   `2·M·N·K / ((M+N)·K·elem_bytes)` ≈ `2MN/(M+N)/elem_bytes`，
   对 fp8 而言约 `2·4096·7168/11264/1 ≈ 5210 FLOP/byte`——
   而机器的比值大约是 `865e12 / 3500e9 ≈ 247 FLOP/byte`。**算力强度是机器平衡点的
   20 倍以上**，所以它天生 compute-bound，跑不满才奇怪。
4. **只有 FP4×FP4 掉到 98.3%**。差的那 1.7% 落在哪？最可能的地方是 fp4 路径额外
   多出来的工作：既要加载/转换 scale，又要处理打包的半字节。这是**从表格能提出、
   但无法只凭表格证明**的假设——按本章纪律，它只是一个待验证的猜想
   （验证方法见 `code/02-perf-ledger/README.md` 的「怎么在真机上确认」一节）。

### 5.3 FP8 推理 GEMM：为什么 M=128 和 M=4096 是两种问题

第二张表是 14 行的 FP8 推理形状（`README.md:103-118`）。挑两组对照着读：

| M | N | K | Latency (us) | Computation (TFLOPS) | Memory BW (GB/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7168 | 16384 | 54.8 | 549 | 2456 |
| 128 | 32768 | 512 | 10.4 | 415 | 1829 |
| 4096 | 7168 | 16384 | 1117.6 | 861 | 186 |
| 4096 | 24576 | 1536 | 362.2 | 854 | 137 |

**M=128 的两行**：算力只到 415–549 TFLOPS（48%–63%），但带宽到了 1829–2456 GB/s。
为什么？因为 M 小时，**权重矩阵 B 只被用一次**：

```text
算术强度 ≈ 2·M·N·K / ((M + N)·K·B_elem) = 2·M·N / ((M+N)·B_elem)
M=128, N=7168, B_elem=1B  →  2·128·7168 / (7296·1) ≈ 251 FLOP/byte
```

251 FLOP/byte 已经接近机器平衡点 247——**处在这个点上，读 B 的时间和算的时间同量级，
两者互相遮挡，谁也吃不掉谁**。所以 M=128 的瓶颈是「把 B 搬进来」，不是 Cube。

**M=4096 的两行**：算力 854–861 TFLOPS（98.7%–99.5%），带宽只剩 137–186 GB/s。
同样的 B 被 32 倍多的行复用：

```text
M=4096, N=7168, B_elem=1B  →  2·4096·7168 / (11264·1) ≈ 5213 FLOP/byte
```

比平衡点高 20 倍，于是完全 compute-bound。

**这一段的工程结论**（也是第 7 节决策表的输入之一）：

* 小 M 的推理形状（decode、batch=1 的 prefill）**优化方向是减少 B 的搬运**：
  权重常驻、L2 hint、更大 block 让 N 方向复用更充分。代码里的对应机制就是
  `resident_b` 与 `l2_ctrl_b = NOTALLOC_KEEP`（`heuristics.hpp:196-197`、
  `:535-545`）。
* 大 M 的形状**优化方向是让 Cube 不空转**：更深的 L1/L0 流水、
  `direct_store` 省掉一次 UB 往返（`heuristics.hpp:522-528`）。
* 两件事都做完了，剩下的差距才轮到「换更激进的排布」。

**读表提醒**：表格的 GB/s 列是官方 README 直接给出的；本书用
「A+B+SF 的字节数 ÷ 延迟」复算时，**14 行的比值全部落在 1.089–1.095**
（也就是表格列比模型高约 9%，且整张表高度一致——脚本
`code/02-perf-ledger/` 会把这个比值逐行打印）。本书**只报告这个事实、不做解释**：
快照里没有给出该列的计算公式，所以无法判断这 9% 是 profiler 侧的实测 GM 流量、
还是别的原因。**引用时以表格列为准，不要用复算值去替它**；差异已登记在
[`references.md`](references.md) 的「待核」小节。
作为对照，第 5.7 节的 HC Prenorm 表**能**被同样方法复现（比值 0.99–1.00），
说明「复算不出来」是这两张表的区别，而不是本书方法的问题。

### 5.4 M-Grouped GEMM：MoE 专家的形状

第三张表（`README.md:120-133`）是 MoE 的 grouped GEMM，`FP8×FP4`、BF16 输出。
它的组织方式与第 16 章讲的 「contiguous layout」完全一致：

```text
A: [M, K]                —— 所有专家的 token 首尾相接
B: [G, N, K]             —— 每个专家一份权重
grouped_layout: [G] int32 —— 每个专家分到多少行
D: [M, N]
```

（形状约定见 `csrc/apis/gemm.hpp:45-58` 的注释与 `:274-306` 的
`m_grouped_fp8_fp4_gemm_contiguous`。）

读数值（`#Groups=4, M per group=8192` 与 `#Groups=8, M per group=4096` 两组）：

| 组数 | 每组 M | N | K | Latency (us) | Computation | Memory BW |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 8192 | 6144 | 7168 | 3722.4 | 818 | 104 |
| 4 | 8192 | 7168 | 3072 | 1778.0 | 856 | 98 |
| 8 | 4096 | 6144 | 7168 | 3689.4 | 831 | 136 |
| 8 | 4096 | 4096 | 4096 | 1356.5 | 861 | 180 |

**八行全部在 818–862 TFLOPS（94.6%–99.7%），带宽只有 98–180 GB/s。**
结论：只要每组的 M 足够大（4096/8192 就是「足够大」），grouped GEMM 与 dense
GEMM 是同一类问题——**compute-bound**。

这条结论很重要，因为它直接回答了第 16 章留下的一个疑问：MoE 里最贵的那一步
（专家 GEMM）在昇腾上**不是通信问题，而是算力问题**；一旦 M per group 掉到几百
（例如某个专家只分到几十个 token），强度会掉到 5.3 节那种「带宽主导」的区间，
这才是 M-Grouped 真正的工程难点。官方测试选的 `M per group = 4096/8192`
代表**训练/预填充**，不代表 decode。

### 5.5 MQA Logits：卡在 FixPipe 上

第四张表是 DeepSeek Lightning Indexer 用的 MQA 打分
（`README.md:135-144`）。官方给了一句非常明确的定性：
**「The kernel is FIX-pipe bound rather than compute bound, saturating the FIX pipe
at 99% utilization.」**

| Type | Format | #Q | #K | #Heads | Dim | Latency (us) | Computation | Memory BW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prefill | FP8 | 4096 | 8192 | 32 | 128 | 302.6 | 681 | 283 |
| Prefill | FP4 | 4096 | 8192 | 32 | 128 | 277.4 | 743 | 277 |
| Decode | FP8 | 256 | 8192 | 32 | 128 | 150.9 | 565 | 2006 |
| Decode | FP4 | 256 | 8192 | 32 | 128 | 124.2 | 708 | 1349 |

**什么是 FixPipe**：第 25 章讲过，FixPipe 是**唯一一条从 L0C 出去的流水线**——
`L0C → GM/L1` 或 `L0C → UB`，搬运途中可以顺手做格式与类型转换（
corpus 的 `…atlas_ascendc_10_0008_html.md`，第 19/25 章引用过）。
在 profiling 结果里它就是 `KernelProfile.aic_fixpipe = "L0C -> out"`
（`deep_gemm/testing/bench.py:39`）。

**为什么会卡在它上面**：MQA logits 的形状是「每个 (q, kv) 对上做一个 head_dim
长度的小点积，然后沿 head 归约成一个标量」。于是：

* 每个 MAD 产出的 L0C 数据，**能换来的额外计算非常少**——head_dim 只有 64/128，
  而点积的结果要立刻被搬出去；
* 输出 logits 是 `#[Q] × max_seqlen_k` 的一个**稠密大矩阵**（
  `csrc/apis/attention.hpp:97-98` 用 `torch::empty_strided` 按 max_seqlen_k
  分配，行 stride 由 `:17-22` 的 `get_logits_stride` 对齐到 512 字节），
  而且它比输入小不了多少；
* 两头一夹，`L0C→out` 的搬出速度就成了上界。**这就是「FIX-pipe bound」**。

顺带说明「99%」怎么读：它是 `aic_fixpipe` 这条 pipe 的占用率接近饱和，
而不是 Cube 打满。用第 19 章教的判读法：**看到 `aic_fixpipe` 高、`aic_mad` 不高，
就不要再调 MAD 的形状，而要想「能不能少搬、或者搬到别处再算」**。

还有一条可观测量与第 5.3 节相反，值得对照：**Decode（Q=256）那两行的带宽反而
高到 1349–2006 GB/s**，因为 Q 少时 KV 基本只用一次。同一个算子，
prefill 是 FIX-bound，decode 是带宽 + FIX 混合——**再一次说明「瓶颈」是形状的函数**。

代码侧对应的是 `MQALogitsConfig`：`split_kv=512`（KV 方向按 512 切）、
`mad_m = 64/num_heads*num_heads`、`num_cd_stages=2`、`num_logits_stages=3`
（`csrc/apis/attention.hpp:61-70`，paged 版同构，见
`csrc/jit_kernels/mqa_logits.hpp:90-103`）。其中 `num_cd_stages` / `num_logits_stages`
控制的正是「L0C→out」这条最容易堵的路径有几级缓冲。

### 5.6 MegaMoE：把五段融成一个 kernel

第五张表是 MegaMoE（`README.md:146-158`）。官方对它的定义是
**「fuses EP dispatch, two grouped GEMMs, SwiGLU, and combine」**，
在 EP8、top-k=6、带一个 shared expert 的设置下，8 个 rank 取平均。

![图 5：MegaMoE 的五段融合](figures/05-mega-moe-fusion.md)

（源描述见 [`figures/05-mega-moe-fusion.md`](figures/05-mega-moe-fusion.md)；
SVG 规格 `src/tutorial/figures/35-deepgemm-ascend/05-mega-moe-fusion.json`。）

**动机**：对照第 16 章的 MoE 四段结构，如果不融合，一次 MoE 至少是
`dispatch → GEMM1 → SwiGLU → 量化 → GEMM2 → combine` 六次 kernel 下发，
中间每一段都要把结果写回 HBM 再读出来，还要在 host 侧做同步。
融合成一个 kernel 之后，中间结果可以留在片上或对称缓冲区里，
而且**通信与计算可以重叠**——这正是它存在的理由。

**结构**：代码里把进度分成四个阶段
（`deep_gemm/include/deep_gemm/layout/mega_moe.hpp:26-33`）：

```cpp
enum ProgressStage : uint32_t {
    kDispatched,    // Linear1 inputs ready
    kLinear1Ready,  // Quantized Linear2 inputs ready
    kLinear2Ready,  // BF16 expert outputs ready
    kCombineSent,   // Sends complete; output ring reusable
};
```

三个执行体分工写在 `deep_gemm/include/deep_gemm/mega_moe.hpp:61-107`：
AIC 负责两段 GEMM（`kL1AFull` / `kSFFull` / `kL0AFull`…）、
AIV0 负责 fp4 权重解码成 fp8 NZ 与 GEMM1 的收尾、
AIV1 负责挑路（top-k 权重）、收尾读出与 combine 的归约。
连**每一条 pipe event 的编号分配**都写成了 `constexpr` 常量
（`kL1AFull{0}`、`kSFEmpty{3}`、`kSharedL1BFull{5}`、`kEpilogueFull{0}`…），
并附了张 L1/L0 的字节级布局表——
**这就是「把芯片摊开写」最直观的样子**：

```text
// deep_gemm/include/deep_gemm/mega_moe.hpp:50-59
// Memory  Region       Offset  Stages x size
// L1      A                 0      3 x 64          (KiB)
//         routed B        192      4 x 64
//         shared B        192      2 x 128 (M=128), 3 x 64 (M=256)
//         A SF            448      2 x 16
//         B SF            480      2 x 16
// L0A     A                 0      2 x 32
// L0B     B                 0      2 x 32
// L0C     FP32 accum        0      2 x 64 (M=128), 1 x 256 (M=256)
```

**代价**（这些是真正要记住的）：

1. **必须是多进程 + 对称内存**：要先用 HCCL 注册一块对称缓冲区
   （`SymmBuffer`，见 `csrc/apis/mega_moe.hpp:44-62` 与
   `deep_gemm/mega/__init__.py:9-60`），所有 rank 的分配顺序必须一致。
2. **形状必须提前钉死**：缓冲区大小由
   `num_ranks/num_experts/num_max_tokens_per_rank/num_topk/hidden/intermediate_hidden`
   算出来（`csrc/apis/mega_moe.hpp:27-42`、`:72-110`），运行期不能变。
3. **核数被写死**：`DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores)`
   （`csrc/apis/mega_moe.hpp:159`，`kNumAICores = 32`）——**用了 MegaMoE 就不能
   `set_num_sms` 降到别的值**，否则直接断言失败。
4. **每次 launch 要清进度缓冲**：`aclrtMemsetAsync` 清零 progress 区
   （`csrc/jit_kernels/mega_moe.hpp:43-46`）——这是一笔固定开销，token 很少时占比可观。
5. **能力被裁到最窄**：只支持 `fp8xfp4` + `swiglu` + `fast_math=True`
   （`csrc/apis/mega_moe.hpp:145-150`、`deep_gemm/mega/__init__.py:28-29`）。

**读表**（8 行）：

| #Experts | Hidden | Interm | Tokens | Latency (us) | Computation | Memory BW | **Comm BW** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 96 | 5120 | 2304 | 64 | 138.7 | 228.5 | 1850.0 | 37.5 |
| 96 | 5120 | 2304 | 256 | 225.5 | 562.5 | 1257.4 | 91.6 |
| 96 | 7168 | 3072 | 64 | 222.2 | 266.4 | 2136.9 | 32.8 |
| 96 | 7168 | 3072 | 256 | 359.2 | 659.2 | 1425.4 | 80.5 |
| 384 | 5120 | 2304 | 4096 | 2567.7 | 790.3 | 567.5 | 128.8 |
| 384 | 7168 | 3072 | 16384 | 17904.2 | 846.3 | 269.3 | 103.3 |

三条读法：

1. **Tokens 越大越接近算力极限**：384 专家那几行到了 790–846 TFLOPS
   （91%–98%），而 96 专家 + 64 token 只有 228–266 TFLOPS。小 batch 时
   固定开销（清 progress、装权重、通信起步）占主导。
2. **Memory BW 一列在小 token 时高达 1850–2137 GB/s**：这与 5.3 节的
   M=128 是同一个道理——B（专家权重）当次只用一次。
3. **Communication bandwidth 单独一列，且只有 33–135 GB/s**：
   与第 38 章要讲的 EP8/EP128 通信带宽相比，这一列描述的是**稳态通信速率**，
   不是通信占总时间的比例——两者要分开看。总延迟里到底有多少被通信占用，
   需要按 `comm_bytes / comm_bw` 单独算，而 **README 没有给出 comm_bytes，
   所以本书不给出任何时间占比**（这正是「多一列带宽不等于多一个结论」的例子）。
   能确定的只有一件事：它被**单独列出来**，说明官方认为它与另外两列同等重要。

### 5.7 HC Prenorm GEMM：给 TileLang 的那一块

第六张表是本章的 TileLang 落点（`README.md:160-170`）。官方描述：
**「Prenorm GEMM for the DeepSeek mHC module, which nearly saturates the HBM
write bandwidth.」**

| M | N | K | Latency (us) | Computation | Memory BW (GB/s) |
| --- | --- | --- | --- | --- | --- |
| 13 | 24 | 28672 | 6.7 | 3 | 519 |
| 137 | 24 | 28672 | 9.6 | 20 | 1110 |
| 512 | 24 | 28672 | 15.4 | 46 | 2082 |
| 4096 | 24 | 28672 | 72.7 | 78 | 3276 |
| 8192 | 24 | 28672 | 136.7 | 82 | 3463 |

这张表的形状一眼就能定性：**N=24**。它比任何一条流水线的宽度都小，
所以「GEMM 的形状」在这里几乎没有意义——真正在动的是 `A`（`[M, 28672]` bf16）
和 `sqr_sum`（`[M]` fp32）这两件东西。

算术强度（`2·M·N·K` vs `M·K·2 + N·K·4 + M·N·4`）在这里是：

```text
M=8192, N=24, K=28672:
  FLOPs   = 2·8192·24·28672            ≈ 1.13e10
  字节     ≈ 8192·28672·2 (A)  + 24·28672·4 (B)  + 8192·24·4 (D) ≈ 4.73e8
  强度     ≈ 23.9 FLOP/byte              ← 远低于 5.3 节算出的机器平衡点 247
```

所以它必是 **memory-bound**，而且官方报的 3463 GB/s 很接近
`HWDesc.gm_bw_gbps = 3500`（`heuristics.hpp:55`）。

**一个必须写清的细节**：官方描述说它「nearly saturates the HBM **write**
bandwidth」，但按测试代码的计字节口径（`count_bytes(a, b, d, sqr_sum)`，
`tests/test_hyperconnection.py:103-107`），上面 `M=8192` 那一行的 473 MB 里
**99.4% 是 A 的读**，写只有不到 1 MB。本书用这个口径复算：
`4.734e8 B / 136.7e-6 s = 3463 GB/s`，与表格列**完全吻合**（
`code/02-perf-ledger/` 会打印每一行的复算值）。
因此「write bandwidth」这句措辞与示例代码的计数口径不一致——
**表格数字本身是可复算的，「写带宽」这个定语存疑**，已登记在
[`references.md`](references.md)。这恰好又是一个「读文档也要读代码」的例子。

**这个 kernel 由 TileLang 实现**（`README.md:190`；代码在
`deep_gemm/tilelang_ops/hc_prenorm.py`，Python 侧出口
`deep_gemm.tf32_hc_prenorm_gemm`，`deep_gemm/__init__.py:100-101`）。
第 6 节逐段读它，回答「为什么这里 TileLang 够用」。

---

## 6. 逐段读 `hc_prenorm.py`：TileLang 够用的那一块

这一节把本章唯一的 TileLang kernel 从头读到尾。完整可复制的骨架在
[`code/04-hc-prenorm-tilelang/`](code/04-hc-prenorm-tilelang/)（**仅静态核对**）。

### 6.1 它要算什么

接口的 docstring 一句话说清（`hc_prenorm.py:367-377`）：

```python
"""Fused hyperconnection prenorm forward.
D = float(A) @ B.T and sqr_sum = sum(float(A) ** 2, dim=-1), computed in one kernel.
    a: [M, K] BF16 input.
    b: [N, K] FP32 weight.
    d: [M, N] FP32 output, or [num_splits, M, N] when num_splits is given.
    sqr_sum: [M] FP32 output, or [num_splits, M] when num_splits is given.
"""
```

三件事值得注意：

1. **输入是 bf16、权重是 fp32，做的是 fp32 精度的 GEMM**。所以这不是「低精度 GEMM」，
   而是「读一堆 bf16、把它展宽成 fp32、和 fp32 权重相乘」。kernel 名里的 `tf32`
   指的是这个「以 fp32 精度累加」的语义，而不是某个 TensorFloat 格式。
2. **两个输出**：`D = [M, 24]` 和 `sqr_sum = [M]`。后者是沿 K 的平方和——
   这正是 mHC 模块做归一化需要的那一项（mHC 的算法背景见第 34 章）。
   把两件事放进一个 kernel，是因为它们**共享同一趟对 A 的读取**——
   如果拆成两个 kernel，A 就要从 HBM 读两遍。
3. **K split 是在 kernel 内部规约的**。`num_splits` 参数只接受 `None` 或 `1`
   （`hc_prenorm.py:382-387`），注释写得很清楚：
   「This backend reduces the K split inside the kernel」。也就是说 TileLang 版本
   自己把「多个核分别算 K 的一段、再原子加」这件事包在了里面，
   调用方看不到多出来的那维。

### 6.2 `get_config()`：把「怎么切」算清楚

`get_config(mhc_hidden_size, mhc_mult3, num_tokens)` 负责在**编译前**把切分方案
定下来（`hc_prenorm.py:13-57`）。它的思路是「三个约束各自求一个 2 的幂，
取最小」：

```python
num_aic_cores = _C.get_num_sms()                       # 我可用的核数（默认 32）
num_k_splits_by_tokens = max(1, num_aic_cores * num_l0c_stages // num_token_blocks)
num_k_splits_by_hidden = num_hidden_blocks & -num_hidden_blocks   # 2 的幂：整除 K 方向块数
num_k_splits_by_cores  = num_aic_cores & -num_aic_cores           # 2 的幂：整除核数
split_limit = min(三个)
num_k_splits = 1 << (split_limit.bit_length() - 1)
```

（`hc_prenorm.py:26-41`。）这里 `& -x` 是「取 x 里最大的 2 的幂因子」的惯用写法。
三个约束分别保证：

* K 方向的块数能被核数整除（不留尾块）；
* 核数能被均分；
* 每个 split 至少能分到 `num_l0c_stages` 个 token 块，否则并行度白给。

接下来两个容量常数是**硬编码**的（`hc_prenorm.py:15-18`）：
`block_num_tokens = 128`、`block_hidden = 256`、`l0b_size_bytes = 64KB`、
`l0c_size_bytes = 256KB`，用来算 L0B/L0C 各能放几级
（`hc_prenorm.py:30-31`）：

```python
num_l0b_stages = l0b_size_bytes // (block_mhc_mult3 * block_hidden * 4)
num_l0c_stages = l0c_size_bytes // (block_num_tokens * block_mhc_mult3 * 4)
```

**与第 19 章的数字对照**：第 19 章用的 Atlas A2 是 L0B 64KB、L0C 128KB；
这里写的是 L0C **256KB**，与 DeepGEMM-Ascend 自带的
`deep_gemm/include/deep_gemm/common.hpp:57`（`L0CSizeBytes = 256*1024`）一致，
两者都是 Ascend 950 系列的口径。**同一本教材里出现两组容量数字不是矛盾，
是两台机器**——查数字前先看它写的是哪台。

`mhc_mult3`（mHC 那一路的 `N`）被对齐到 16 的倍数，并且断言 `≤64 且 8 的倍数`
（`hc_prenorm.py:21,27`）；`block_hidden=256` 必须整除 `mhc_hidden_size`
（`:22-24`）。于是 N=24 在这里实际被当成 **32**（`(24+15)//16*16`）来分配缓冲区，
只在真正的数据搬运与 MAD 区域里用回 24（`hc_prenorm.py:158-163` 的注释：
「Keep physical buffers block_mhc_mult3-aligned, but restrict data movement and MAD
regions to the logical mhc_mult3」）。**这是本章最典型的 TileLang 写法**：
物理形状对齐、逻辑范围收窄，一句注释就把「为什么 buffer 比逻辑大」交代清楚。

### 6.3 kernel 骨架：一个 MixedKernel，两级流水

`@tilelang.jit` 修饰的 `tf32_hc_prenorm_gemm_kernel`（`hc_prenorm.py:59-359`）
结构如下（行号都是 `hc_prenorm.py` 的）：

```text
60- 93  函数签名与张量声明（num_tokens 是 T.dynamic，mhc_mult3/hidden 是 T.const）
105     with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
111-120   分配 UB / L1 / L0A / L0B / L0C（用 T.alloc_shared / alloc_l1 / alloc_l0a / ...）
122-131   T.annotate_buffer_versions({...})        ← 每个 buffer 几级
133-135   非 0 号 split 打开 float32 原子加；设置 hf32 舍入
136- 292 主循环：
           136  for chunk_idx in T.Serial(num_token_chunks)
           154  └ for local_hidden_block in T.Pipelined(..., num_stages=2)
           160      ├ 搬 fn 权重：GM → L1 → L0B
           170      └ for token_stage in T.Serial(...)      ← token 方向
           184          ├ T.copy x → x_ub（bf16，L2 hint notalloc_keep）
           195          ├ with T.SimdVF(latency=640):  展宽 + NZ 打包 + 行平方和
           268          ├ T.dual_copy(x_nz_ub → x_l1)
           273          └ for hidden_mad: 搬 x_l1→x_l0，T.gemm(...) → out_mul_l0
294-356  with T.PerCoreTask(): 两个 store 阶段（D 与 sqr_sum）
358-359  收尾：关掉原子加
```

它用到的原语全部是第 19 / 25 章讲过的 M 树 API：
`T.MixedKernel`（AIC + AIV 混核）、`T.alloc_shared`（UB）、`T.alloc_l1`、
`T.alloc_l0a/l0b/l0c`、`T.copy`、`T.dual_copy`、`T.gemm`、`T.Pipelined`、
`T.SimdVF` + `T.simd.*`、`T.annotate_layout`、`T.annotate_buffer_versions`、
`T.PerCoreTask`、`T.ascend_sync_inter_arrive/wait`、`T.set_atomic`。
**没有一处是 Ascend C**。

### 6.4 第一个关键点：为什么先展宽、再打 NZ

最密的一段是 `T.SimdVF(latency=640)` 里的向量代码（`hc_prenorm.py:195-266`）。
它在做三件事，**一趟遍历同时完成**：

```python
# hc_prenorm.py:220-249（节选，去掉寄存器的初始化）
for pass_id in T.Serial(block_hidden // 128):
    packed = T.simd.vld(x_ub[row, pass_id * 128])          # 读 128 个 bf16
    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)    # 交错成两个 64×fp32
    lo = T.reinterpret(lo_bits, 'float32x64')              # bf16 → fp32（展宽）
    hi = T.reinterpret(hi_bits, 'float32x64')
    row_acc = T.simd.vadd(row_acc, T.simd.vmul(lo, lo, f32_mask), f32_mask)   # 平方和
    row_acc = T.simd.vadd(row_acc, T.simd.vmul(hi, hi, f32_mask), f32_mask)
    nz_ptr = T.simd.vsstb(lo, nz_ptr, nz_stride, f32_mask, update=True)       # 按 NZ 布局写出
    nz_ptr = T.simd.vsstb(hi, nz_ptr, nz_stride, f32_mask, update=True)
```

`vintlv(zero, packed)` 是「把 16 位元素交错成 32 位槽位」的写法：一个 bf16 占 16 位，
把它与 `0` 交错后重新解释成 fp32，就完成了 bf16→fp32 的展宽。
紧接着用 `vsstb` 把结果按 NZ 布局写进 `x_nz_ub`——**展宽与重排合成一条流水**。

值得对照的是：这段代码**本来就是 TileLang**，而不是「先写 CUDA 再迁移」。
它用了 `T.simd.*` 这一层显式 SIMD 原语，而不是靠 `T.Parallel` 自动张量化——
这正是第 25 章 §6.2 提醒的那件事：「自动调度不保证把任意 `T.Parallel` 张量化，
要精确控制就写 `T.SimdVF` + 显式 `T.simd.*`」。

`sqr_sums` 最后写回 `sqrsum_ub`，并用 `if local_hidden_block != 0` 判断要不要
把上一块隐藏维的结果累加进来（`hc_prenorm.py:263-266`）——因为 K 方向被拆成了
`num_hidden_blocks_per_split` 块，平方和要跨块累加。

### 6.5 第二个关键点：K split 用原子加合并

`get_config` 把这些核分成 `num_k_splits` 组，每组负责 K 的一段；
`split_id = core_id // cores_per_split`（`hc_prenorm.py:106`）。
第 0 组之外的组**打开原子加**（`hc_prenorm.py:133-134`）：

```python
if split_id != 0:
    T.set_atomic('add', 'float32')
...
if split_id != 0:
    T.set_atomic_none()
```

于是 store 阶段（`hc_prenorm.py:294-356`）不需要任何跨核规约：
0 号组普通写、其余组原子加，最后 D 里就是完整的和。
**代价是：非确定性模式下每次运行的累加顺序可能不同**，结果不可复现。
所以接口额外提供 `deterministic` 开关，打开后把 store 阶段拆成
`num_k_splits` 个**串行相位**，每个 split 依次写（`hc_prenorm.py:95-102`、
`:294-324`）——用速度换「逐位可复现」。

这不是 TileLang 发明的技巧，而是同一算法的两种实现约定；TileLang 版本能把两者
写在**同一个 kernel 里用参数选**，靠的是 `T.PerCoreTask` + `T.Serial` 这两件原语：

```python
# hc_prenorm.py:294-324（节选）
with T.PerCoreTask():
    for store_phase in T.Serial(num_store_phases):
        store_this_phase = T.if_then_else(deterministic, split_id == store_phase, ...)
        ...
        T.ascend_sync_inter_arrive('PIPE_FIX', out_mul_flag)
        T.ascend_sync_inter_wait('PIPE_FIX', out_mul_flag)
```

注意最后两行：这是**核间**同步（`inter`），flags 用 `(chunk_idx*num_store_phases
+ store_phase) % 4` 循环复用 4 个 event。这类「手写核间同步」在第 19 章的坑清单里
（P10：flag 必须严格配对）属于高危区；TileLang 在这里只是把 API 摊开给作者用，
**并没有让这件事变简单**。这也是判断「该不该用 TileLang」的一个信号：
当一个 kernel 需要大量手工核间同步时，TileLang 省下来的工作量已经不多。

### 6.6 为什么这里 TileLang 够用

把上面几点合起来，答案有四条，每条都能对上源码：

1. **它是 memory-bound，不是 compute-bound**（§5.7 的算术强度 ≈ 24 FLOP/byte）。
   对一个被 HBM 带宽卡住的 kernel，**优化空间主要在「少读一次、少写一次」**，
   而不是「把 Cube 榨到 99%」。TileLang 现有的 `T.copy` / `T.Pipelined` /
   `annotate_buffer_versions` 恰好就是干这个的。
2. **形状规整**：K = 28672 能被 256 整除、M 是动态但按 128 分块、N 是 24（对齐到 32）。
   没有稀疏、没有不规则的尾块、没有需要精细控制 L1 bank 的访存模式。
   第 7 节决策表里的「规整访存」这一类，正是 TileLang 的舒适区。
3. **它需要的所有原语 TileLang 都有，而且都是高层语义**：混核（`T.MixedKernel`）、
   SIMD 向量区（`T.SimdVF` + `T.simd.*`）、L1/L0 分配、NZ 布局注解、
   多缓冲版本注解、`T.gemm` 到 L0C。**没有一处需要「绕过编译器」**。
4. **它的两个输出共享一趟读**——这正是融合的收益，而「把两个算子融进一个 kernel」
   是 TileLang 最擅长表达的事情（第 13 章的融合主题延续到这里）。

反过来说，**这个 kernel 里也有 TileLang 帮不上忙的地方**：`T.SimdVF` 里那段
手工交错与 NZ 写出、`T.PerCoreTask` 里的核间 flag、`set_atomic` 的开关时机，
都是作者自己设计的。TileLang 在这里提供的是「**把芯片原语摊给你用的语法**」，
而不是「自动帮你把这件事做对」。第 7 节会把这条界线写成决策表。

---

## 7. 决策表：什么时候用 TileLang，什么时候下钻

### 7.1 先摆事实

在给结论之前，先把本章已经核实过的事实列出来——决策表的每一格都必须落在其中一条上，
不能凭空写：

| 事实 | 出处 |
| --- | --- |
| DeepGEMM-Ascend 的 GEMM / MQA / MegaMoE 用 Ascend C + MAD 抽象；mHC prenorm 用 TileLang | `DeepGEMM-Ascend/README.md:190`；`deep_gemm/tilelang_ops/hc_prenorm.py` |
| 前者拿到 98.3%–99.8%（dense）、94.6%–99.7%（M-grouped）、FIX 99%（MQA） | `README.md:90-144` |
| 后者的形状是 `[M, 24, 28672]`，算术强度 ≈ 24 FLOP/byte，带宽 3463/3500 GB/s | §5.7；`README.md:160-170` |
| TileLang 侧的主干（v0.1.15）与适配器（v0.1.2.000）都在；`T.gemm` 只降 L0 MAD | 第 19、25 章 |
| TileLang-Ascend 官方自己的分工对照表说：自动调度**不保证**把任意 `T.Parallel` 张量化 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:141-153`（第 19 章 S4 用 130-155 的宽范围核验过同一份文档） |

### 7.2 决策图

![图 6：TileLang 与 Ascend C + MAD 的分工决策](figures/06-tilelang-vs-ascendc.md)

（源描述见 [`figures/06-tilelang-vs-ascendc.md`](figures/06-tilelang-vs-ascendc.md)；
SVG 规格 `src/tutorial/figures/35-deepgemm-ascend/06-tilelang-vs-ascendc.json`。）

### 7.3 决策表

**问题 1：这个算子的瓶颈在算力，还是在访存？**

| 瓶颈 | 建议 | 依据 |
| --- | --- | --- |
| 访存 / 带宽（算术强度远低于机器平衡点） | **先试 TileLang** | §5.7 的 hc_prenorm：强度 ≈ 24 FLOP/byte，TileLang 版已接近 HBM 峰值带宽 |
| 算力（算术强度远高于平衡点，目标是榨 Cube） | **准备下钻 Ascend C + MAD** | §5.2 的 dense GEMM：98.3%–99.8% 是用 `mad_m/n/k`、L1/L0 stage、L2 hint、`resident_*`、`dual_aiv_dequant` 这一整套手工参数拿到的 |
| 既不是访存也不是算力（FixPipe / 通信） | **先定位到具体 pipe，再决定** | §5.5 MQA 卡在 `aic_fixpipe`；§5.6 MegaMoE 卡在通信与计算的配比 |

**问题 2：访存模式规整吗？**

| 访存模式 | 建议 | 依据 |
| --- | --- | --- |
| 规则的 rect tile、对齐、无稀疏 | TileLang | hc_prenorm 的 K/M/N 全部规整；`T.copy` + 布局注解足够 |
| 不规则尾块、NaN/inf 特判、需要精细 L1 bank 配平 | Ascend C + MAD | DeepGEMM-Ascend 用 `set_l0c_nz2dn` / `set_mte2_nz` 直接配置搬运参数（`ascend/copy.hpp:9-20`）、`fill_tail_block` 手工补零（`:180-198`） |
| 稀疏加载（只搬需要的那部分） | Ascend C + MAD | README 明确把「sparse data loading」列为它相对编译器路线的优势之一（`README.md:6-7`） |

**问题 3：开发效率和可移植性，哪个更重要？**

| 你的处境 | 建议 | 依据 |
| --- | --- | --- |
| 算法还在快速迭代，形状会变 | TileLang | 改一个 `T.copy` 的 shape 就能重编；DeepGEMM-Ascend 的很多配置是**编译期常量**（`GemmConfig` 全是模板参数），换形状要重新 JIT |
| 形状已经冻结，要榨最后 2% | Ascend C + MAD | 98% 之上的每一分都来自手工参数；代码里甚至留了一条注释说这个启发式「不够好、应该按代价枚举更合适」（`heuristics.hpp:139-140`） |
| 上限需要「可复现、可审计」 | 两条路都要，但要对齐口径 | DeepGEMM-Ascend 用 `aclnn_*` 对拍（§3.4）；TileLang 侧用第 14 章的对齐方法 |

**问题 4：这个算子需要跨设备/跨团队复用吗？**

| 需求 | 建议 | 依据 |
| --- | --- | --- |
| 同一份算法要在 NVIDIA 与昇腾都跑 | TileLang（见第 33 章） | DeepGEMM-Ascend 的路线是「同一 API、两份实现」——NVIDIA 的 `DeepGEMM` 是另一套 C++/CUDA；TileLang 的路线是「一份算法、两个后端」 |
| 只做昇腾、且要极致 | Ascend C + MAD | 本章全篇 |

### 7.4 三条容易走错的边界

1. **「用 TileLang 就不需要懂硬件」是错的**。`hc_prenorm.py` 里逐字出现的
   `T.alloc_l1` / `T.alloc_l0a/b/c`、`T.SimdVF`、`T.ascend_sync_inter_arrive`
   说明：TileLang 把硬件摊开给你用，你仍然要决定「这块数据放哪一级、
   谁先谁后」。**TileLang 省掉的是模板代码与参数推导，不是硬件判断。**
2. **「用 Ascend C 一定更快」也是错的**。§5.7 已经说明：
   hc_prenorm 是带宽问题，TileLang 版本已经打满带宽；这里换成 Ascend C + MAD，
   **能拿回来的额外收益非常小**，却要付出维护成本。
3. **两者的可组合性不同**。DeepGEMM-Ascend 的 kernel 有明确的 Python 接口和
   `aclnn_*` 参考实现（§3.4），可以像积木一样插进任何 PyTorch 流程；
   TileLang kernel 更容易和**同一份 TileLang 代码库**里的其它算子共享 layout 与
   调优参数（第 33、34 章的 TileKernels 就是这样组织的）。选择前先想清楚
   「这个算子要和谁拼在一起」。

---

## 8. 常见坑

按「读文档 / 读代码 / 自己写」三类整理。每条都给出**症状 → 根因 → 怎么查**。

**P1｜以为 SF 是数值转换而不是字节重解释。**
症状：用 `sf.to(torch.int16)` 造输入，结果大片错误或直接 panic。
根因：昇腾 SF 格式是把**两个 ue8m0 字节**放进一个 `int16`，
不是把 fp32 数值转成 int16。查：`tests/test_mega_moe.py:22-36` 的
`sf_exponent.view(torch.int16)`；或用 `transform_sf_into_required_layout`
把 fp32 交给库去转（`csrc/apis/layout.hpp:68-92`）。

**P2｜把 `dav-*` 架构当成环境变量。**
症状：想 `export DG_NPU_ARCH=...` 却发现没有效果。
根因：快照里没有这个环境变量；架构要么由设备上报，要么用
`deep_gemm.set_npu_arch(n)`（`csrc/jit/jit.hpp:25-32`）。

**P3｜环境变量设晚了。**
症状：`DG_JIT_CACHE_DIR` / `DG_JIT_DEBUG` 明明设了却不生效。
根因：`DG_JIT_*` 在 **JIT 运行时第一次创建时被快照**
（`README.md:62-63`），`ASCEND_HOME_PATH` 则在 **import 时**被读
（`deep_gemm/__init__.py:15-17`）。查：把 `os.environ[...] = ...` 挪到
`import deep_gemm` 之前。

**P4｜想用 masked grouped GEMM。**
症状：`AttributeError`，或者找不到 `m_grouped_*_nt_masked`。
根因：Ascend 后端没有实现（`csrc/apis/gemm.hpp:591`、`:615`）。
查：改走 contiguous + 自己补零，或者用别的算子。

**P5｜`compiled_dims` 与动态 M 打架。**
症状：每个新 M 都触发一次编译，编译时间爆炸。
根因：默认 `compiled_dims="nk"` 只固化 N、K，M 保持动态——
这是**对的**（MoE 的 M 本来就会变）；编译变慢应查是不是把 `m` 也写进了
`compiled_dims`（`csrc/apis/gemm.hpp:543`、`gemm_config.hpp:16-21`）。

**P6｜M-grouped 的行数没对齐到 256。**
症状：结果错行，或断言失败。
根因：M-grouped 强制 `block_m = 256`，且
`set_mk_alignment_for_contiguous_layout` 只接受 256
（`csrc/jit/jit.hpp:59-69`、`heuristics.hpp:106`、`:124-130`）。
查：用 `get_theoretical_mk_alignment_for_contiguous_layout()` 拿对齐值再补零。

**P7｜在 MegaMoE 上 `set_num_sms`。**
症状：直接断言失败。
根因：`DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores)`
（`csrc/apis/mega_moe.hpp:159`）——MegaMoE 的核间分工写死了 32 核。

**P8｜用 ACLNN 对拍时以为「对不上就是 kernel 错」。**
症状：对拍失败，但参考实现其实没在算你要的那个。
根因：ACLNN 只支持 1×32 的 scale、要求 K 是 64 的倍数、对 MN-major 的单 K-pair
scale 有已知缺陷（`tests/common.py:82-117` 的四条提前返回）。
查：先确认对拍本身适用，再下结论。

**P9｜把 `aclnn_*` 当成性能基线。**
症状：拿 `aclnn_bf16_gemm_*` 的耗时和 kernel 比，得出奇怪的结论。
根因：它是**参考实现**（README 原话「used to cross-check the kernels in tests」），
不是为性能调的。性能基线应看 README 的表格与它的测量条件。

**P10｜拿 dense 的 99% 去推 M=128 的推理性能。**
症状：按 dense 表预期小 M 也能到 99%，实测差很远。
根因：§5.3——M=128 是带宽问题，利用率天然低（48%–63%）。
查：先算算术强度，再决定该看哪一列。

**P11｜把 hc_prenorm 的「write bandwidth」当字面意思。**
症状：按「写带宽」做容量规划，发现对不上。
根因：该表按 `count_bytes(a, b, d, sqr_sum)` 计（§5.7），99% 是 A 的读。
查：自己在 `code/02-perf-ledger/` 里换口径复算，再决定引用哪一种。

**P12｜在 HC kernel 上追求确定性却不接受开销。**
症状：`use_deterministic_algorithms(True)` 之后慢了一截。
根因：确定性模式把 store 拆成 `num_k_splits` 个串行相位
（`hc_prenorm.py:95-102`）。查：训练用非确定 + 固定种子，
验证/复现用确定模式，别把两者混着调。

---

## 9. 本章小结

一页纸复习：

1. **API 契约的内容**：包名、函数名、参数顺序、`(A, sfA)` 元组约定、
   `nt/nn/tn/tt` 逻辑形状后缀全部一致；注册点
   `csrc/python_api.cpp:30-42`。兼容承诺只管 **kernel 接口**，工具函数两边不同。
2. **唯一被官方标注的不兼容点是 SF 存储格式**：K 方向每两个 UE8M0 打进一个
   `int16`、MN-major 存放；`gran_k` 只能是 32；fp32 输入要求低 23 位为 0。
   另有两条「README 未提但代码写明」的覆盖差异：masked grouped 未实现、
   `fp4×fp8` 不再支持。
3. **MAD 抽象 = 三个头文件 + 一个出口**：`tile_ptr`（带形状的指针、自动对齐）、
   `copy`（通路、NZ 转换、量化模式推导、尾块补零）、`sync`（带方向的 PIPE 事件）、
   `mad(...)`（8-bit/4-bit 走 `asc_mmad_mx`，16-bit 走 `asc_mmad`，`clear_C` 显式传）。
4. **一次 shape 的旅程**：`GemmDesc`（问题）→ `select_gemm_config`（在 L0C/L0A-L0B/L1
   容量约束下搜索，用 `HWDesc` 的带宽/频率算代价）→ 字符串拼出模板实例化 →
   DeepJIT 编译与缓存（`DG_JIT_CACHE_DIR`，默认 `$HOME/.dj`）→ launch。
   `DG_PRINT_CONFIGS=1` 是读这个库最快的入口。
5. **「硬件极限」不是宣传词**：`get_mad_flops_per_cycle` × 主频 × 32 核 =
   bf16 432.5 / fp8 865 / fp4 1730 TFLOPS，与 README 的 Hardware limit 列吻合。
6. **六类算子的瓶颈**：dense 与大 M 的 grouped 是算力（98%+）；小 M 的 FP8 推理、
   hc_prenorm 是带宽；MQA logits 是 FixPipe；MegaMoE 是「小 batch 时固定开销 +
   通信」，大 batch 时回到算力。
7. **TileLang 的落点是 hc_prenorm**：memory-bound、形状规整、原语齐备、
   两个输出共享一趟读。它照样要手写 SIMD 交错、NZ 写出与核间 flag——
   TileLang 给的是语法，不是自动化。
8. **决策的主轴是瓶颈类型**：访存主导 → 先试 TileLang；算力主导且形状冻结 →
   Ascend C + MAD；FixPipe / 通信主导 → 先定位到具体 pipe 再决定。

---

## 延伸阅读

* **[R1] DeepGEMM-Ascend 自述**：`DeepGEMM-Ascend/README.md`
  —— 定位、Requirements、接口、工具函数、环境变量、六张性能表、
  以及那句「The mHC kernel is backed by Tilelang」。
* **[R2] MAD 抽象层**：`DeepGEMM-Ascend/deep_gemm/include/deep_gemm/ascend.hpp`
  与其下的 `ascend/tile_ptr.hpp`、`ascend/copy.hpp`、`ascend/sync.hpp`
  —— 本章第 2.3、4 节的全部依据。
* **[R3] JIT 与启发式**：`DeepGEMM-Ascend/csrc/jit/jit.hpp`、
  `csrc/jit_kernels/heuristics.hpp`、`csrc/jit_kernels/gemm_config.hpp`
  —— 硬件模型、候选生成、代价函数、config 选择。
* **[R4] 各算子的 device 实现**：`deep_gemm/include/deep_gemm/{bf16_gemm,fp8_gemm,
  fp8_dequant_gemm,mqa_logits,mega_moe,transform_sf}.hpp`。
* **[R5] TileLang 落点**：`DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py`
  （本章第 6 节逐段读）与 `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py`
  （MegaMoE 的 TileLang 基线实现，见 `tests/test_mega_moe.py:74-94`）。
* **[R6] 对拍与口径**：`DeepGEMM-Ascend/tests/`（`common.py`、`generators.py`、
  `utils.py`、`test_*.py`）与 `deep_gemm/testing/bench.py`（`bench_msprof`、
  L2 flush、`KernelProfile`）。
* **[R7] NVIDIA 对照物**：`DeepGEMM/README.md`
  —— 接口约定、工具函数、以及 SM90/SM100 对 SF 的不同要求。
* **[R8] JIT 运行时**：`DeepJIT/README.md`
  —— 双后端、缓存 key、共享缓存、原子发布。
* **[R9] 前置章节**：第 14 章（GEMM 性能账本与五级阶梯）、第 16 章（MoE 四段结构，
  含 grouped GEMM）、第 25 章（昇腾后端与 codegen）、第 26 章（TileLang 运行时与 JIT）、
  第 27 章（低精度与 block scaling）。

完整的引用清单（含核验状态与上游链接）见 [`references.md`](references.md)；
每个代码示例的运行前提与验证级别见 [`code/README.md`](code/README.md)；
图的说明见 [`figures/README.md`](figures/README.md)。
