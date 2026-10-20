# 第 27 章 低精度、量化与数值正确性

## 学习目标

读完本章，你应当能够：

1. 说出 fp16 / bf16 / fp8（e4m3、e5m2）/ fp4 e2m1 的**有效位、`u = 2^-p`、
   最大有限值、最小规格化值**，并解释「为什么 bf16 的精度约等于 int8」
   「为什么 fp8 必须配 scale 才能用」。
2. 讲清 MX / block-scaled 的三要素：**块粒度**、**E8M0 scale 的语义与值域**、
   **scale 的打包与索引方式**；并说明 NVIDIA 与昇腾在这一步上的差异。
3. 写出 int8 / int4 对称量化与反量化的公式，判断给定数据上误差的**上界与量级**，
   并说明「粒度决定误差、位宽只是其中一个因素」。
4. 在 TileLang 里找到低精度的表达方式：dtype 系统、`T.int4` / fp8 的使用、
   `T.gemm_blockscaled` 的参数语义，以及五个官方示例各自的结构。
5. 为一次低精度对拍**先定义容差再写代码**：选对参照、选对指标，
   并区分数值误差、逻辑错误、参照错误、累加器不足这四类「对不上」。
6. 在给定形状与层的场景里判断该不该量化、量化到什么程度，并说清
   量化对 attention 与 MoE 的影响来自哪里。

## 前置知识

* 第 9 章：`T.copy` / `T.gemm` / `T.clear` 的语义，以及 `T.alloc_shared`
  / `T.alloc_fragment` 的区别。
* 第 12 章第 3.7 节：**容差模型**——`u = 2^-p`、长度 K 的归约误差 `u/sqrt(K)`、
  `atol` / `rtol` / `max_mismatched_ratio` 的用法。本章所有的容差推导都建在它上面。
* 第 14 章第 5 节：fp8 / int4 / dequantize / block-scaled 四类 GEMM 的**结构**与
  调用方式；第 14 章第 7 节的对齐方法论。本章补上它们的**数值**部分。
* 第 10 章：shared memory 与 fragment 的区别；本章第 4 节会用到。
* 第 6 章（可选）：NV 与昇腾的存储层级对照；本章 4.2 节涉及 L1 / L0 / TMEM。

**本章的验证边界（务必先读）**：写作环境是 macOS (Apple Silicon)，
**没有 NVIDIA GPU，也没有昇腾 NPU**。因此：

* 本章 `code/01`–`code/06` 六个示例**全部在 CPU 上实际运行过**（Python 3.11.7 +
  numpy 1.24.1），输出的每一个数字都可复算，验证级别是「CPU 逻辑验证」；
* 凡涉及硬件行为的结论（scale 放在 TMEM、粒度是 32 还是 128、fp8 的 K 对齐），
  来源都是**锁定版本的源码**或 **corpus 里的厂商文档**，`references.md` 逐条给出
  路径与核验方式；
* **本章没有任何一条性能数字**，也没有任何一条「在加速器上运行过」的声明。
  凡是有硬件才能确认的事，本章只写「怎么验证」，不写结论。

---

## 1. 问题的由来

### 1.1 低精度不是「省一点」，而是三笔账同时改写

fp16 换成 fp8，表面上只改了一个 dtype，实际上三笔账同时变了：

| 账 | fp16 → fp8 的变化 | 受益的算子类型 |
| --- | --- | --- |
| 访存 | 权重/激活的字节数减半 | memory-bound（decode 阶段的 GEMV、norm、elementwise） |
| 算力 | 张量核的 fp8 峰值通常是 fp16 的 2 倍 | compute-bound（prefill 的 GEMM、attention 的 QK/PV） |
| 容量 | 同样显存能放下的权重翻倍 | 长上下文推理、大专家数的 MoE |

所以「该不该量化」从来不是一个精度问题，而是一个**先看账、再看精度**的问题
（第 14 章第 1 节的性能账本在这里同样适用）。真正需要认真对待的是第三笔账的对价：
**低精度会改变数值结果**，而改变多少必须能提前算出来、事后能验证。

### 1.2 误差有三个来源，必须分开讨论

把「结果对不上」一律归为「低精度导致」，是最常见的误判。低精度算子的误差至少有
三个独立来源，量级可以差好几个数量级：

1. **表示误差**：数据存成低精度时被舍入。上界是 `u = 2^-p`
   （fp16 `4.9e-4`、bf16 `3.9e-3`、fp8 e4m3 `6.3e-2`、fp4 e2m1 `2.5e-1`）。
2. **累加误差**：乘积累加时的舍入。它**取决于累加器类型与归约顺序**，与输入精度无关。
   本章 `code/03` 实验 C 里，同一份 fp16 输入换成 fp32 累加器，中位误差从
   `1.24e-4` 掉到 `1.24e-9`。
3. **算法误差**：量化本身引入的信息损失。例如权重从 fp16 量化到 int4 之后，
   无论 kernel 写得多准，输出都不可能等于量化前的结果——差别是**设计决定**，
   不是 bug。

第 12 章讲了前两类在 fp16/bf16 下的模型；本章把它推广到 fp8/fp4/整数，
并加上第三类：**怎么给「量化本身」定一个参照**。

### 1.3 这一章的组织

沿着「数值 → 表达 → 硬件 → 验证 → 取舍」推进：

* 第 2 节：把格式的范围与精度**算出来**（不是抄参数），并推出量化误差的上界；
* 第 3 节：TileLang 怎么表达这些 dtype，五个官方示例各自的结构；
* 第 4 节：块缩放的硬件约束——scale 放哪、怎么打包、粒度是多少；
* 第 5 节：正确性验证方法论，重点是**参照与指标的选择**；
* 第 6 节：性能与精度的权衡，落到「什么形状、什么层适合量化」；
* 第 7 节：常见陷阱清单。

---

## 2. 数值基础

本节所有数字都由 `code/01-number-formats/number_formats.py` **枚举位模式**算出；
读者可以在任意装了 numpy 的机器上复算。不要把这些数字当「参数表」抄下来——
要记住的是它们**怎么来的**。

### 2.1 浮点格式的表示与取舍：fp16 与 bf16

一个二进制浮点数由三部分构成：符号位 `s`、指数 `e`、尾数 `m`，值为
`(-1)^s × (1.m)₂ × 2^(e - bias)`。

**有效位数 `p` 决定精度，指数位数决定范围。** 16 位里两者此消彼长：

| 格式 | 符号/指数/尾数 | 有效位 `p` | `u = 2^-p` | max 有限值 | min 规格化数 | 动态范围 |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 1/5/10 | 11 | 4.883e-04 | 6.55e+04 | 6.104e-05 | 12.0 个数量级 |
| bf16 | 1/8/7 | 8 | 3.906e-03 | 3.39e+38 | 1.175e-38 | 78.6 个数量级 |

这张表解释了 bf16 的全部设计动机：**它把 fp32 的指数位原封不动搬过来，只砍尾数**。
于是 fp32 → bf16 只需要截断低 16 位，溢出风险与 fp32 相同；代价是相对精度比 fp16
低 8 倍（`2^-8` 对 `2^-11`）。

工程上的两条直接推论：

* **需要宽的动态范围时用 bf16**：梯度的量级跨度远大于激活（有极小值也有离群值），
  fp16 更容易下溢到次正规区甚至 0；
* **需要高精度时用 fp16**：激活与前向权重通常已被归一化到量级接近 1 的范围，
  这时 `u` 小 8 倍是实打实的收益。

### 2.2 fp8：e4m3 与 e5m2 是一个显式的取舍

fp8 只有 8 位，于是指数与尾数的争夺更激烈，标准给出了两个版本：

| 格式 | 符号/指数/尾数 | 有效位 `p` | `u = 2^-p` | max 有限值 | min 规格化数 | 动态范围 |
| --- | --- | --- | --- | --- | --- | --- |
| e4m3 | 1/4/3 | 4 | 6.250e-02 | **448** | 0.01562 | 5.4 个数量级 |
| e5m2 | 1/5/2 | 3 | 1.250e-01 | 5.734e+04 | 6.104e-05 | 9.6 个数量级 |

有几个 e4m3 特有的细节，写 kernel 时必须知道：

1. **e4m3 没有 inf**，指数全 1 也是有限值——`448 = 1.75 × 2^8` 就是它的最大值。
   这一点与 IEEE 惯例不同（`references.md` R7：
   “The `e4m3` encoding does not support infinity”）。
2. **NaN 只有两个编码**：`0x7f` 与 `0xff`。其余位模式都是有限数。
3. 因为最大只有 448，**e4m3 几乎总是配 scale 使用**。锁定版本里的 MXFP8 量化就是这么做的：

   ```python
   # _vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:556-559
   x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
   sf = ceil_to_ue8m0(x_amax / 448.0)
   x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
   ```

   `amax / 448` 再向上取到 2 的幂，保证除以 scale 之后**一定落在 e4m3 的值域内**。
4. **e4m3 还有负零**。TileLang 里为此专门有一个清理函数
   `fp8_remove_negative_zeros_`（`tilelang/utils/tensor.py:25-29`）：
   在逐位比较的场景下，`-0.0` 与 `+0.0` 的位模式不同，会被判成不相等。

于是「e4m3 还是 e5m2」就有了清楚的判据：**前向与权重用 e4m3（精度优先），
反向的梯度用 e5m2（范围优先）**。`determine_fp8_type()` 的默认值也是 e4m3
（`tilelang/language/fp8.py:9-27`）。

### 2.3 MX / block-scaled：块粒度 + E8M0

给整个张量配一个 scale（per-tensor）太粗，给每个元素配一个（per-element）开销太大。
**MX（Microscaling）格式的答案是折中：每 32 个元素共享一个 8 位指数 scale。**

三个要素缺一不可：

**（a）块粒度。** MX 的原生粒度是 **32 个元素**。这一点在两处独立文档里一致：

* PTX 对 `.kind::mxf8f6f4` 的块缩放规定：每个 A 的行按 32 个 K 元素分块，
  scale 存在 Tensor Memory 里（`nvidia-ptx-isa.md:32061-32130`）；
* CATLASS 的常量 `MX_SCALE_GROUP_NUM = 32`，注释是
  “Data count for one MX-scale factor per group”
  （`catlass/include/catlass/catlass.hpp:48`）。

**（b）scale 的语义：E8M0。** 8 位指数、0 位尾数，也就是**只能是 2 的整数次幂**：

* 可表示值 `2^-127 .. 2^127`（255 个编码，`0xff` 留给 NaN）；
* **没有 0、没有负数**——它是缩放因子，不是数据格式；
* 相邻值之比恒为 2，所以「用 scale 缩放」**不引入任何尾数误差**，只搬运指数。

这条性质是 MX 能同时保住范围与精度的关键：`x / 2^k` 在二进制下只是指数减 `k`，
是**精确**操作。E8M0 的编码还有个实现上的便利——它正好就是 fp32 的
biased exponent 字节，所以源码里直接从位里抠：

```python
# _vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:543-546
def ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)
```

* 加 `(bits & 0x7FFFFF).ne(0)`：尾数非 0 时指数加一，实现「向上取整到 2 的幂」；
* `clamp(1, 254)`：跳过 0（次正规区）与 255（NaN）。

**（c）打包与索引。** 4 个 scale 字节被打进 1 个 `uint32`，全局按 group-major 展平：

```text
word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
SFA[g * M + m]        # g 是「4 个字节一组」的组号，m 是行号
```

`code/05-mx-scale-layout/mx_scale_layout.py` 把这套打包/解包复刻了一遍，并证明：
字节序写反会让结果偏差 5.58（相对 L2），用行主序索引会偏差 2.47——
都是「和真值同一个量级」的错误，属于逻辑错误而不是精度问题。

**块粒度的实际取值可以比 32 粗。** 锁定版本的 SM100 MXFP8 示例把粒度设成 **128**：
一个字节覆盖 128 个 K 元素，四个字节覆盖 512 个 K 元素，正好一个 `uint32`。
`sf_load_period = sf_granularity_k * 4 // block_K = 128 * 4 / 128 = 4`，
所以每 4 次 K 迭代搬一个 word
（`examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:1-47`）。

> **对接口的含义**：粒度是一个必须显式写进接口的量。NVIDIA 硬件原生 32、
> TileLang 的 SM100 示例 128、昇腾 CATLASS 固定 32。
> 生产端与消费端取不同的粒度，结果会直接错（见 `code/05` 第 3 行：偏差 2.77）。

### 2.4 int8 / int4 的量化与反量化

整数量化没有指数位，范围完全由 scale 撑起来。最常用的是**对称量化**（记 `qmax`
为整数格式的正向最大值，int8 取 127、int4 取 7）：

```text
s = absmax / qmax                        # scale
q = clip(round(x / s), -qmax-1, qmax)    # 量化（round 为就近取整）
x_hat = q * s                            # 反量化
```

误差是**均匀分布**的：量化步长等于 `s`，最大误差 `s/2`，均方根 `s/√12`。
相对到 absmax 就是 `u_int = 1/(2·qmax)`，即

* int8：`1/(2·127) = 3.937e-3`；
* int4：`1/(2·7) = 7.143e-2`。

把这两个数与第 2.1、2.2 节的浮点表对起来看，会得到一个反直觉但很有用的结论
（`code/02` 与 `code/03` 都实测到了）：

| 对标 | 相对精度 | 结论 |
| --- | --- | --- |
| int8 (3.9e-3) ≈ bf16 (3.9e-3) | 同量级 | **int8 存权重在精度上不比 bf16 差** |
| int4 (7.1e-2) ≈ fp8 e4m3 (6.3e-2) | 同量级 | **4bit 定点约等于 8bit 浮点** |

差别在**范围的分配方式**。int4 只有一个全局 absmax，一个离群值就会把整个 scale
拉大，让绝大多数元素只用到很粗的档位；fp8 有指数，大小值各用各的精度。
这就是「int4 更需要细粒度 scale」的根本原因。

**粒度到底有多重要？** `code/02-quant-dequant/quant_dequant.py` 在同一份
256×1024 数据上实测（相对 L2 是对全张量的均方口径）：

| 方案 | 相对 L2 | 逐元素 max err | SQNR |
| --- | --- | --- | --- |
| int8 per-tensor | 4.951e-02 | 3.406e-01 | 26.1 dB |
| int8 per-channel | 7.674e-03 | 3.393e-01 | 42.3 dB |
| int4 per-group-64 | 1.083e-01 | 6.049e+00 | 19.3 dB |
| MXFP8 (e4m3, blk 32) | 2.685e-02 | 3.808e+00 | 31.4 dB |
| MXFP4 (e2m1, blk 32) | 1.177e-01 | 1.586e+01 | 18.6 dB |

注意第一行：per-tensor int8 的相对 L2 是 `4.95e-2`，**比它的理论上界 3.9e-3 大一个
数量级**。原因是这份数据 per-tensor 的 absmax 是 86.5，而 per-row absmax 的中位数
只有 3.44——差 25 倍。上界是**相对 absmax** 的，而 L2 是全体元素的均方。
换成 per-channel 之后 absmax 回到同一量级，L2 立刻降到 `7.67e-3`，落在上界附近。

> **一句话记住**：量化的精度由「每多少个元素共享一个 scale」决定，
> 位宽只是其中一个因素。选粒度比选位宽更值得花时间。

### 2.5 累加精度与舍入误差的传播

第 12 章给出的模型是：以 `S = Σ|a_i·b_i|` 为基准，长度 K 的归约

* 最坏情形相对误差约等于 `u`（各项误差同号、完全不抵消）；
* 均方根相对误差约为 `u/√K`（各项误差独立、随机抵消）。

本章把 `u` 换成每个格式的「等价 u」（浮点用 `2^-p`，整数用 `1/(2·qmax)`），
在 `code/03-error-model/error_model.py` 里做了三组实验。

**实验 B：缩放律对浮点成立，对 fp4 失效。** 实测中位数 / 预算：

| 格式 | K=64 | K=256 | K=1024 | K=4096 |
| --- | --- | --- | --- | --- |
| fp16 | 0.53 | 0.54 | 0.58 | 0.48 |
| bf16 | 0.52 | 0.41 | 0.41 | 0.47 |
| e4m3 | 0.47 | 0.45 | 0.47 | 0.44 |
| int8 | 1.47 | 1.27 | 1.32 | 1.30 |

fp16/bf16/e4m3 稳定在 0.5 附近（与第 12 章的结论一致），int8 在 1.3 附近
（均匀量化比浮点更接近最坏情形）。但 e2m1（fp4）在同一实验里是 **2.58**：
4 位浮点的相对间距随指数变化太剧烈，`u` 这个「相对上界」对小数值根本不成立。
**用 fp4 时不要套 fp16 的容差公式，必须在真实数据分布上实测。**

**实验 C：累加器是独立的一档误差。** 输入完全相同（fp16），只换累加方式
（K=4096，20 次试验）：

| 方案 | 相对误差中位数 | 相对误差最大值 |
| --- | --- | --- |
| fp16 顺序累加 | 1.238e-04 | 5.695e-04 |
| **fp32 累加** | **1.243e-09** | 3.431e-09 |
| 32 段 fp16 部分和 + fp32 汇总 | 2.063e-05 | 7.009e-05 |

第三行就是 split-K 的抽象：**每段的舍入误差只与该段长度有关，跨段用 fp32 连接
不会再放大**。这条结论可以直接搬到 kernel 上：K 很长时，宁可把 K 切开、
用 fp32 汇总，也不要让低精度累加器一路累到底。

**累加器的选择在编译器里是有约束的**，不是随手填的：

* TCGEN5MMA（Blackwell）的 fp16/bf16 路径要求**累加器必须是 fp32**，
  且 K 是 16 的倍数（`src/op/tcgen5_meta.h:42-45`）；
* 同一路径的 f8f6f4 家族要求 K 是 32 的倍数，累加器可以是 fp32 **或 fp16**
  （`src/op/tcgen5_meta.h:79-82`）——**后者能编译通过，但会明显变差**，
  这是本章第 7 节的陷阱之一；
* WGMMA（Hopper）的 fp8 路径要求 A 不转置、B 转置，且 K 是 32 的倍数
  （`src/cuda/op/gemm.cc:49-50,62-63`）。

---

## 3. TileLang 里的表达

### 3.1 dtype 系统：一个入口，三种写法

TileLang 的 dtype 不接受「自己发明的名字」。`tilelang/dtypes.py` 只有三行，
把语言面的定义整体再导出（`_vendor/tilelang-v0.1.15/tilelang/dtypes.py:1-3`）：

```python
from tilelang.language.dtypes import *
from tilelang.language.dtypes import dtype, AnyDType, get_tvm_dtype
```

真正的定义在 `tilelang/language/dtypes.py`。要点：

* **一个 dtype 可以写成三种形态**：字符串（`'float8_e4m3fn'`）、TileLang 的 dtype
  对象（`T.float8_e4m3fn`）、框架 dtype（`torch.float8_e4m3fn`）。
  统一入口是 `get_tvm_dtype()`（`:349-353`）。
* **登记表决定「能不能用」**。`_STR_TO_TVM_DTYPE_CALL`（`:121-142`）里列出的是
  语言面认识的名字：`int4` / `int8` / `int16` / `int32` / `int64`、`float16` /
  `float32` / `float64` / `bfloat16`、`float8_e4m3` / `float8_e4m3fn` /
  `float8_e4m3fnuz` / `float8_e5m2` / `float8_e5m2fnuz` / `float8_e8m0fnu`、
  `tfloat32`。名字不在这张表里，`T.dtype("...")` 会直接抛 `TypeError`（`:245-255`）。
* **低精度家族的向量形态是自动派生的**：`float8_e4m3x2 / x4 / x8 ...`、
  `float4_e2m1fnx16` 等（`:449-560` 的类声明区）。命名规则就是 `<dtype>x<lanes>`，
  多个标量用 `T.float8_e4m3x2(a, b)` 打包成 `Shuffle`（`:148-152`）。
* **FP4 有两种存储**，这一点很容易踩坑：
  * `float4_e2m1fn`：紧凑的 4 bit 打包（`type_code == 17`），对应 CUTLASS 的
    `float_e2m1_t`，用于 mxf4 / mxf4nvf4 的 packed SMEM；
  * `float4_e2m1_unpacked`：8 bit 一个槽位（`type_code == 131`），只用于 tcgen05
    的 `f8f6f4` / `mxf8f6f4` 混合精度路径与 TMA 的 `16U4_ALIGN16B`。

  两者的判断函数是 `is_float4_e2m1fn()` / `is_float4_e2m1_unpacked()` /
  `is_float4()`（`:268-308`）。
* **混合 dtype 的 GEMM 目前只允许 f8f6f4 家族**：
  `validate_gemm_ab_dtypes()` 在 A、B dtype 不同时直接抛错，
  除非两边都属于 `is_f8f6f4_family()`（`:319-341`）。

还有一条**只在运行时才会暴露**的约束：**torch 自己不一定有这些 dtype**。
`__dtype_as_torch__`（`:187-238`）在遇到 `int4` / `uint4` / `float4_e2m1fn` 时会
退化成 `int8` / `uint8` / `float4_e2m1fn_x2` 作为**存储**类型，并打一条 debug 日志；
而 `float8_e8m0fnu` 要求 `torch >= 2.8.0`，`float4_e2m1fn_x2` 同样要求
`torch >= 2.8.0`。也就是说：**TileLang 认识这个 dtype，不代表你的 torch 认识**。
遇到「编译过了但张量创建失败」，先查版本。

### 3.2 fp8 相关的辅助模块

`tilelang/language/fp8.py` 只有 49 行，但它是「同一份代码跑在不同后端上」的关键：

```python
# _vendor/tilelang-v0.1.15/tilelang/language/fp8.py:9-27
def determine_fp8_type(fp8_format="e4m3", device=None) -> str:
    """Select the correct FP8 dtype string for the current platform.
    - CUDA defaults to FP8 E4M3FN / E5M2.
    - ROCm uses FNUZ except gfx950 (OCP), which prefers non-FNUZ when available.
    """
```

同样的 `"e4m3"` 请求在三个平台上会得到三个不同的字符串：

| 平台 | `determine_fp8_type("e4m3")` | 说明 |
| --- | --- | --- |
| CUDA | `T.float8_e4m3fn` | OCP 版本，就是本章第 2.2 节讲的那个 |
| ROCm（除 gfx950） | `T.float8_e4m3fnuz` | FNUZ：没有负零、NaN 只有一个 |
| ROCm（gfx950） | `T.float8_e4m3fn` | 新硬件回到 OCP |

注意 `dtype.as_torch()`（`language/dtypes.py:191-204`）用的是同一套判据，
所以「TileLang 的 dtype 字符串」与「torch 的 dtype」在 fp8 上是**平台相关**的。
写跨后端代码时不要硬编码 `torch.float8_e4m3fn`，用 `determine_torch_fp8_type()`。

### 3.3 五个量化 GEMM 示例各自在讲什么

第 14 章第 5 节讲了它们的**结构**，这里补上每个示例里与数值相关的细节。

**（a）`examples/gemm_fp8/example_tilelang_gemm_fp8.py`：只换 dtype。**

```python
# :16-35，节选
def matmul(A, B, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    ...
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
```

* 累加器显式写成 `T.float32`——这是 fp8 的正确用法；
* 对拍用的不是逐元素比较，而是**相似度**（`:7-11` 的 `calc_diff`，
  形如 `1 - 2<x,y>/(|x|²+|y|²)`），判据是 `diff < 1e-3`（`:53`）。
  这是一个「量级判据」：它只关心两个结果的方向是否一致，天然允许逐元素的量化噪声。
  第 5 节会讲这种指标的适用范围与风险。

**（b）`examples/gemm_int4/example_tilelang_gemm_int4.py`：整数域是精确的。**

```python
# :19-36，节选
A: T.Tensor((M, K), T.int4)
B: T.Tensor((N, K), T.int4)
C: T.Tensor((M, N), T.int32)
...
T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

* 累加器是 `int32`：int4 × int4 的乘积最大 `8 × 8 = 64`，K 累加不会溢出 int32，
  所以**整数域的乘法累加是精确的**，没有累加误差；
* 因此它的对拍判据是 `rtol=0, atol=0`（`:101`）——**逐位相等**。
  这是本章里唯一可以用精确相等的地方，因为它根本没有舍入；
* 打包在宿主侧完成（`:59-67`）：`(a & 0x0F) | ((b & 0x0F) << 4)`，
  两个 int4 的低 4 位拼进一个字节。解码时靠**算术**右移做符号扩展还原。

**（c）`examples/dequantize_gemm/`：把反量化融合在搬入时。**

README 里的原型就是最小可运行版本：

```python
# examples/dequantize_gemm/README.md
for i, j in T.Parallel(block_N, block_K):
    B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
        num_bits, B_local[i, j // 2], j % 2, dtype=in_dtype,
    )
T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)
```

几个值得单独看的变体：

* `example_dequant_gemm_w4a8.py:10-22` 的 `_tir_u8_to_i4_to_i8` 用了一个漂亮的
  位技巧做符号扩展：先把 4 位值左移 4 位，再把它按 `int8` 算术右移 4 位，
  就把 4 位有符号数变成了 8 位有符号数——比任何分支判断都快。
* `example_dequant_gemm_fine_grained.py:43-44` 按「一次搬运 128 位」反推每个线程
  该处理多少元素：`local_size = 128 // bits(in_dtype)`。这是 dequantize GEMM
  最关键的工程参数：解码的粒度要跟访存事务对齐，否则向量化会失效。
* `examples/dequantize_gemm/quantize/` 下是 host 侧的量化工具（`mxfp.py`、
  `nvfp4.py`、`quantization.py`），其中 `nvfp4.py:180-196` 的
  `blockscaled_chunk_kmajor_word_offset()` 专门处理 CUTLASS
  `BlockScaledBasicChunk` 的 K-major 排布——**scale 布局不只有一种**。

**（d）`examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py`：块缩放的样子。**

```python
# :37-52 与 :134-146，节选
SFA: [(K / sf_granularity_k) / 4) * M] in uint32     # group-major packed E8M0
SFB: [(K / sf_granularity_k) / 4) * N] in uint32
T.tcgen05_gemm_blockscaled(..., k_start=k * block_K,
                           sf_a_granularity_k=sf_granularity_k,
                           sf_b_granularity_k=sf_granularity_k)
```

这几行里包含了全部约定：粒度（`sf_granularity_k`）、打包（4 个字节一个 `uint32`）、
布局（`/4) * M` 这种下标表达式就是 group-major）。
这个示例还有 1CTA / 2CTA / 2CTA persistent 三个变体，其中 2CTA 变体直接断言
`sf_granularity_k == 128`（`:195`）——**粒度不是随便填的，硬件与模板都会限制它**。

**（e）`examples/bitnet-1.58b/`：端到端的 1.58 bit。**

这个目录不是单个 kernel，而是一个完整的模型复现（BitNet b1.58-3B）。它的量化规则
在 `utils_quant.py` 里，是「三值权重 + 8 位激活」的经典写法：

```python
# examples/bitnet-1.58b/utils_quant.py:16-30
def weight_quant(weight, num_bits=1):
    s = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s        # 三值：-1 / 0 / +1

def activation_quant(x, num_bits=8):
    Qn, Qp = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s            # per-token absmax
```

两个细节值得记住：

* **权重的 scale 用「平均绝对值」而不是 absmax**（`s = 1/mean|w|`）。这是因为三值
  量化只有 3 个档位，用 absmax 会让绝大多数权重落到 0；
* **激活的 scale 是 per-token（`dim=-1`）**，因为激活的离群值通常集中在少数通道，
  per-tensor 会被单个离群值拖死。

这条 1.58 bit 路线的性能数字在 `examples/bitnet-1.58b/README.md` 里
（A100、Llama-3B 结构）：fp16 基线 177.67 → int8×int2 实现 64.18
（batch=1、in_seq=1）。**这是上游样例给出的数字，不是本机实测**，
条件与复现脚本都在那个目录里，`references.md` 已登记。

### 3.4 昇腾侧的表达

昇腾适配器仓库（`_vendor/tilelang-ascend-v0.1.2.000/`）给低精度路径补了两样东西：

**（a）dtype 名字不同。** 昇腾方言里 fp8 的字符串是 `e4m3_float8` / `e5m2_float8`，
映射到 `float8_e4m3_t` / `float8_e5m2_t`（`tilelang/language/ascend.py:26`、
`tilelang/utils/tensor.py:26-40`）。写跨后端代码时不要硬编码名字。

**（b）量化 GEMM 的写法。** `examples/quant_batch_matmul/example_quant_batch_matmul.py`
是最完整的例子，它的数据流是：

```python
# :49-88，节选
A_L1 = T.alloc_L1([block_M, block_K], in_dtype)        # int8
B_L1 = T.alloc_L1([block_K, block_N], in_dtype)        # int8
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)    # int32
for bk in T.serial(k_num):
    T.copy(A[bb, bm * block_M, bk * block_K], A_L1)
    T.copy(B[bb, bk * block_K, bn * block_N], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))
T.copy(C_L0, workspace_1[...])
T.copy(workspace_1[...], c_ub)                          # 搬到 UB 上做 epilogue
for bm_v, bn_v in T.Parallel(block_M_2, block_N):
    c_scale[bm_v, bn_v] *= scale_ub[bn_v]               # per-channel / per-tensor scale
```

注意这条路径的形状：**int8 输入 → int32 在 L0C 精确累加 → 搬到 UB 乘 scale → cast 到
fp16/bf16 输出**。这是「位置 C：累加之后反量化」的原型，也是最省的一种做法：
整数累加精确，scale 只乘一次。它的对拍判据是 `rtol=1e-2, atol=1e-2`（`:116`）。

**（c）昇腾的 fp8 GEMM 有平台门槛。** `examples/gemm/example_gemm_fp8_pto.py:10-16`
写得很直白：FP8 的 TMATMUL 路径需要 A5（Ascend 950）的 Cube 核，
A2/A3 没有这条路径，示例会直接打印 `[SKIP]` 跳过。

---

## 4. 实现机制：块缩放的硬件约束

### 4.1 NVIDIA：scale 住在 TMEM 里

块缩放看起来只是「多乘一个数」，但在 Blackwell 上它牵动了整条数据通路。
PTX 对 `tcgen05.mma` 的块缩放规定（`nvidia-ptx-isa.md:32061-32130`）：

```text
(A * scale_A) * (B * scale_B) + D
where `scale_A` and `scale_B` are matrices residing in Tensor Memory.
```

四个由此推出的硬约束：

**（1）scale 必须在 TMEM，不能在 SMEM。**
所以示例里有一条专门的通路把 scale 从 SMEM 搬进 TMEM：

```python
# examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:131-134
T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem)
T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem)
```

`tcgen05_cp_warpx4` 的语义在 `tilelang/language/builtin.py:1440-1471` 写得很清楚：
每个 128 word 的块映射到 **4 个 TMEM 列**，列偏移自动前进。

**（2）为什么是「每 4 个 K 块打包」。** TMEM 的一个单元格是 32 位，E8M0 是 8 位，
所以一个单元格正好放 4 个 scale。指令描述符里用两个 2 位字段选子列：

```python
# tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py:629-632
runtime_sf_a_id = ((_sf_k_start + ki * micro_size_k) // int(sf_a_granularity_k)) % 4
runtime_sf_b_id = ((_sf_k_start + ki * micro_size_k) // int(sf_b_granularity_k)) % 4
runtime_instr_desc = base_instr_desc | (runtime_sf_a_id << 29) | (runtime_sf_b_id << 4)
```

`% 4` 就是「在 word 的 4 个字节里选第几个」。**这条公式把 `sf_granularity_k` 与
存放布局绑死了**：粒度写错，选到的字节就错。

**（3）SFB 要复制到 32 个 lane 分区，SFA 不一定。** PTX 的原文是
“Scale factor for `B` matrix needs to be duplicated to all 32 lane partitions of
tensor memory. But for the `A` matrix, the duplication is not necessary if the
scales are copied into Tensor Memory via `tcgen05.st` on `sm_107f` architecture.”
于是示例里的 TMEM 形状是不对称的：

```text
SFA_tmem: [128 lanes, 4 columns]
SFB_tmem: [128 lanes, 8 columns]     # 两个 128 列的 N chunk
```

（`examples/blockscaled_gemm_sm100/mxfp8_illustrated.md` 的 Scale-Factor Layout 一节）
同一份文档还说明了 `tcgen05_sf_warp_transpose` 的作用：把 `4 x 32` 的 word 视图
改写成 `32 x 4`，以匹配 `tcgen05.cp.32x128b.warpx4` 的源模式。

**（4）粒度可以比 32 粗，但不能比 32 细。** 加粗是合法的——把同一个值写进 4 个连续的
子块即可；但**细过 32 不行**，因为 `scale_vec::1X` 只有 4 个子列可用。

### 4.2 昇腾：固定粒度 32，反量化发生在 L1 → L0

昇腾侧的 MX 路径与 NVIDIA 的结构完全不同，但有一处一致：**粒度是 32**。

**（a）CATLASS 的模板。** 常量定义在 `catlass/include/catlass/catlass.hpp:48`：

```cpp
constexpr uint32_t MX_SCALE_GROUP_NUM = 32;  // Data count for one MX-scale factor per group
```

它在模板里的用法可以在 `block_mmad_mx_tla.hpp` 看清楚：

```cpp
// include/catlass/gemm/block/block_mmad_mx_tla.hpp:101,107-109
static constexpr uint32_t L1_SCALE_FACTOR_K = DispatchPolicy::L1_SCALE_FACTOR_K;
static constexpr uint32_t L1SCALEA_TILE_SIZE =
    L1_TILE_M * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleA) * L1_SCALE_FACTOR_K;
```

`L1_SCALE_FACTOR_K` 是「L1 上一次驻留多少个 K 条带的 scale」，而
`MX_SCALE_GROUP_NUM`（= 32）决定每个条带需要多少个 scale。也就是说：
**想加粗粒度，只能改模板参数，不能像 NVIDIA 那样在 kernel 里用不同的
`sf_granularity_k` 绕过去**。

**（b）反量化发生在搬运指令里。** 昇腾的 Cube 直接吃低精度操作数，
scale 在 L1 → L0 的搬运阶段就作用到每个 K 块上。CATLASS 的 MXFP8 FlashAttention
文档把这条路径画得很清楚（`mxfp8_flash_attention_infer.md`）：

```text
Q(fp8_e4m3)  K(fp8_e4m3)   mxScaleQ(e8m0)  mxScaleK(e8m0)
      |            |              |              |
      v            v              v              v
   GM→L1        GM→L1      GM→L1 mxScaleA     GM→L1 mxScaleB
      |            |              |              |
      +------------+--------------+--------------+
                   v
        L1→L0A / L1→L0B（反量化在这里发生）
                   v
           Cube Mmad (mx_fp8_e4m3)
```

同一份文档还给了两个关键数字：**block size = 32**、scale 格式 **e8m0**。
这与 NVIDIA 侧的 PTX 规定一致，说明「MX 的粒度是 32」是跨厂商的约定，
而 TileLang 的 128 是**具体 kernel 的实现选择**。

**（c）昇腾 950 上的 dtype。** 两个现成的例子：

* `examples/59_ascend950_a8w4_mx_matmul/`：A 是 `float8_e4m3_t`、B 是
  `float4_e2m1x2_t`（E2M1 打包）、scale 是 `float8_e8m0_t`
  （`a8w4_mx.cpp:57`），公式写作 `C = (MxScaleA x A) * (MxScaleB x B) + Bias`；
* `examples/54_ascend950_fp4_mx_matmul/`：A、B 都是 `float4_e2m1x2_t`，
  L1 分块 256×256×448、L0 256×256×128——README 特意提醒「勿随意增大 L1 的 K，
  否则 `L1TileShape exceeding the L1 space`」。

**（d）TileLang 主干里的昇腾块缩放接口。** `tilelang/ascend/language/gemm_op.py`
把粒度写死在 32：

```python
# tilelang/ascend/language/gemm_op.py:121-156，节选
"""Ascend block-scaled MXFP8 GEMM, shadowing the common ``T.gemm_blockscaled``.
...
the SFA region slice, and MX scales cover 32 K elements per factor.
"""
annotations["sf_a_granularity_k"] = 32
annotations["sf_b_granularity_k"] = 32
```

对照 NVIDIA 侧的 `T.gemm_blockscaled(..., sf_a_granularity_k=..., sf_b_granularity_k=...)`
（`tilelang/language/gemm_op.py:275-320`），可以看清楚这个抽象的分层：

| 接口 | 粒度 | 谁决定 |
| --- | --- | --- |
| `T.gemm_blockscaled` | 调用方传入 | 前端（target-neutral），后端决定支持与否 |
| `T.tcgen05_gemm_blockscaled` | 调用方传入 | 必须落到 SM100 的 TCGEN5MMA，否则编译失败 |
| 昇腾方言的 `T.gemm_blockscaled` | **固定 32** | 昇腾后端 |

还有一个设计细节值得注意：**没有块缩放实现的后端不会静默降级**。
`src/op/gemm_blockscaled.cc:94-99` 里的报错信息是：

```text
Block-scaled GEMM is not supported by the <backend> backend (target=...)；
a dense GEMM lowering would silently drop the SFA/SFB scale factors.
```

这是对的：块缩放**不是一个可选的优化**，丢掉 scale 会直接改变结果。
如果你在某个后端上看到这条错误，它是在阻止你写出一个静默出错的 kernel。

### 4.3 反量化融合在哪一级

把「W 是低精度、要乘 scale 才能用」这件事落到实处，只有三个位置可选（图 4）：

| 位置 | 做法 | 代表 | 适用条件 |
| --- | --- | --- | --- |
| A. 搬入时（prologue） | 在 `T.copy` 之后用位运算解码，再送进普通 `T.gemm` | `examples/dequantize_gemm/*` | scale 与 K 相关（group 沿 K）或格式需要解包（int4/fp4） |
| B. 主循环里（L1→L0） | 搬运指令按块粒度读 scale 并作用 | CATLASS `MmadMx`、SM100 的 `tcgen05.mma.block_scale` | 硬件提供块缩放 MMA |
| C. 累加之后（epilogue） | 整数/浮点累加完后乘一次 scale | 昇腾 `example_quant_batch_matmul` | **scale 与 K 无关**（per-tensor / per-channel） |

三者的取舍不是性能偏好，而是**由 scale 的粒度倒推的**：

* 位置 C 之所以便宜，是因为整数累加精确、scale 只乘一次；但它**只能处理与 K
  无关的 scale**。一旦 scale 是 per-K-block，累加完再乘就已经晚了——
  每个 K 块必须乘自己的 scale，误差结构完全不同。
* 位置 B 是唯一能省掉「解压 + SMEM 往返」的方案，代价是粒度被硬件与模板固定。
* 位置 A 是通用兜底：能处理任意粒度，但 SMEM 要按解压后的尺寸开
  （4 bit 解到 16 bit 就是 4 倍空间）。

**位置 C 有一个容易忽略的前提**：只有当累加器是**整数**时，「整数累加精确」才成立。
如果权重是 fp8、累加器是 fp16，那么位置 C 的 scale 乘法仍然精确，
但累加本身已经有误差了——这正是第 2.5 节实验 C 测出来的那一档。

---

## 5. 正确性验证方法论

第 12 章讲了「先定容差再谈对错」。低精度算子在这个基础上多了两个问题：
**参照怎么选**、**指标怎么选**。

### 5.1 三层参照：先决定「跟什么比」

低精度算子的对拍必须先回答一个问题：**参照实现自己做不做量化？** 三种选择对应
三种完全不同的判据（`code/04-vs-reference/reference_matching.py` 把三种都跑了一遍）：

| 参照 | 相比对象 | 测的是什么 | 判据 |
| --- | --- | --- | --- |
| **绝对参照**（量化感知的 fp64/fp32 实现） | 实现了同一套量化规则的参考代码 | kernel 是否写对 | 可以很紧（本例到 1e-16） |
| **相对参照**（另一个 kernel / 框架实现） | 另一个做了同样量化的实现 | 两个实现是否一致 | 松容差 + 不匹配比例 |
| **量化前参照**（直接把 fp32 结果当参照） | 量化之前的数据 | **量化损失了多少信息** | 只能用于评估精度，不能当对拍判据 |

`code/04` 的实测（X 是 int8、W 是 int4 per-group-64，判据为
`rtol=1e-2, atol=1e-3, max_mismatched_ratio=1e-3`）：

```text
参照                                      相对 L2      不匹配比例    结论
A 量化感知参照（fp64）                        3.837e-16    0.000e+00    通过
B 反量化位置错（只乘最后一组 scale）             2.097e-01    9.666e-01    失败
C 参照漏乘 scale（逻辑错误）                   9.643e-01    9.999e-01    失败
D 用量化前的 fp32 权重当参照                   1.078e-01    9.397e-01    失败（但不是 bug）
E 被测与参照 C 犯同一个错                      9.906e-13    0.000e+00    通过（假阳性）
E' 同一个错误的实现 vs 正确的 A                 2.668e+01    9.999e-01    失败
```

两行最关键：

* **D 行失败，但不是 bug。** 94% 的元素不匹配、相对 L2 是 `1.08e-1`——
  这就是 int4 权重量化应有的误差。把它当判据，你只能把 `atol` 放到毫无作用的程度。
* **E 行通过，但也不对。** 两个实现犯了同一个错（都漏了 scale），对拍当然通过；
  而 E' 行证明这个实现相对正确参照差了 26.7 倍。**「两边一致」不等于「两边都对」。**

结论：**每个量化 kernel 都要有一个绝对参照**，再拿与框架实现的对拍作第二道防线。

### 5.2 误差指标：三个一起看，缺一不可

锁定版本里的两个对拍工具给的是同一组参数（`rtol` / `atol` /
`max_mismatched_ratio`）：

| 工具 | 位置 | 默认值 |
| --- | --- | --- |
| `torch_assert_close` | `tilelang/utils/tensor.py:205-293` | `rtol=1e-2, atol=1e-3, max_mismatched_ratio=1e-3` |
| `Profiler.assert_allclose` | `tilelang/profiler/__init__.py:104-115` | `atol=1e-2, rtol=1e-2, max_mismatched_ratio=1e-2` |

这些参数各负责一件事，用错了会漏掉整类问题：

| 指标 | 能发现 | 发现不了 |
| --- | --- | --- |
| `rtol` / `atol`（逐元素） | 个别元素的巨大偏差、近零值的绝对偏差 | 全张量统一的小偏移（每个元素都没超差，但整体错） |
| `max_mismatched_ratio` | 允许极少数离群点，避免被单个元素卡死 | 大面积的系统性错误（比例远超阈值） |
| **相对 L2** | 整体是否「同一个量级」 | 少数离群点（被均方稀释） |
| **逐元素 max err** | 有没有大的离群点 | 整体的小偏差 |

**实践中按这个顺序看**：

1. 相对 L2：判断是「量级问题」还是「噪声问题」。接近 1 就是逻辑错误。
2. 逐元素 max err：找出离群元素，看它们是否集中在某一行/列/专家上。
3. 不匹配比例：看错的是「全部」还是「局部」。

`code/04` 的输出就是按这个顺序排的，一眼就能看出 B/C/E' 是逻辑错误
（相对 L2 分别是 0.21 / 0.96 / 26.7），D 是数值误差（相对 L2 0.11，但 max err 只有 1.04）。

### 5.3 哪些误差是「正常的」

把本章的模型与实测汇成一张基线表。**判据是「相对 L2 是否落在预期量级」**，
不是「是否等于 0」：

| 场景 | 预期相对 L2 量级 | 依据 |
| --- | --- | --- |
| fp16 输入 + fp32 累加 | 1e-5 ~ 1e-4 | 第 12 章实测中位数约为 `0.5 × u/√K` |
| bf16 输入 + fp32 累加 | 1e-4 ~ 1e-3 | 同上 |
| fp8 e4m3 输入 + fp32 累加 | 1e-3 ~ 1e-2 | 本章 `code/03` 实验 A（K=64 时 3.7e-3） |
| int8 per-channel + int32 累加 | 1e-3 ~ 1e-2 | 本章 `code/02`（7.67e-3） |
| int4 per-group-64 + int32 累加 | 5e-2 ~ 2e-1 | 本章 `code/02`（1.08e-1） |
| MXFP8（e4m3，32 粒度块缩放） | 1e-2 ~ 5e-2 | 本章 `code/02`（2.69e-2） |
| MXFP4（e2m1） | 1e-1 ~ 3e-1 | 本章 `code/02`（1.18e-1）；模型在此失效 |
| 纯整数路径（int8×int8 → int32） | 0（逐位相等） | `examples/gemm_int4` 的 `rtol=0, atol=0` |

最后一行值得单独强调：**整数域没有舍入，所以可以要求逐位相等**。
如果你的 int 路径对不上，那就一定是逻辑错误，没有任何「容差」的余地。
这条判据比任何经验值都可靠。

### 5.4 区分数值误差与逻辑错误

图 5 给了完整的判定树。这里给出三个**最快能用的判据**：

**判据一：看「误差 / 结果的量级」。** 如果是 1e0 量级（每个元素都错），
那是逻辑错误；如果是 1e-2 量级且与格式的 `u` 相符，那是数值误差。

**判据二：看误差是否与某个维度相关。** 把误差 reshape 回 `(M, N)`，分别沿两个方向
取最大绝对值：

* 误差集中在某些行/列/专家 → 索引错误（scale 用错轴、group 索引错）；
* 误差与行列无关、分布均匀 → 数值误差。

**判据三：换参照再算一遍。** 如果换成量化感知的 fp64 参照后误差变成 1e-15，
那原来的「误差」全部来自量化本身；如果仍然很大，那是 kernel 的问题。

一个有用的补充信号：**误差与 K 的关系**。数值误差随 K 按 `1/√K` 缩小
（第 12 章与本章实验 B）；如果误差**随 K 增大而变大**，那几乎一定是
累加器精度不足或归约顺序有问题，而不是输入精度的问题。

---

## 6. 性能与精度的权衡

### 6.1 什么形状适合量化

量化的收益来自「少搬字节、少算时间、少占显存」，所以要回到第 14 章的性能账本：

* **memory-bound 的形状**（算术强度低于机器平衡点）：
  收益几乎与位宽成线性。decode 阶段 batch=1 的 GEMV、归一化、逐元素算子都在这里。
  权重从 16 bit 降到 4 bit，访存量降到 1/4，时间通常也接近 1/4。
  这是**权重量化**（W4A16、W8A16）最有价值的场景。
* **compute-bound 的形状**（大 M、大 K 的 prefill GEMM）：
  收益来自张量核的峰值差异。fp8 的峰值通常是 fp16 的 2 倍，
  但**只有当 K 足够大、且精度损失可接受时**才拿得到。
  激活也一起量化（W8A8、W4A8）才吃到这部分收益。
* **容量受限的形状**：MoE 的专家数、长上下文的 KV cache。
  这类场景里量化往往是**唯一**能把模型放进显存的办法，精度是次要矛盾。

一条粗判据：**先看这个算子的算术强度与显存压力，再看它的数值敏感度。**
形状决定收益上限，数值敏感度决定你敢走多远。

### 6.2 量化对 attention 与 MoE 的影响

**Attention。** 三个矩阵乘各有各的敏感度：

| 步骤 | 量化对象 | 关键约束 | 依据 |
| --- | --- | --- | --- |
| QK^T | Q、K | 需要 per-block scale；softmax 之前整体平移不改变结果，所以 scale 的绝对量级不敏感 | CATLASS 的 MXFP8 FA 用 block=32 的 e8m0 |
| P·V | P | **P 是 softmax 的输出，值域固定在 [0,1]**，必须 cast 回 fp8/fp16，且常常要配一个静态量化系数 | `mxfp8_flash_attention_infer.md`：`P_fp8 = quantize(P_fp32 × pScale)` |
| 输出 | O | 累加仍在 fp32，Online Softmax 也在 fp32 完成 | 同上：“Online Softmax 在 FP32 精度下完成计算” |

两个具体的坑：

* **P 的量化会放大误差**：softmax 之后 P 的小值很多，低精度下这些小数被压成 0
  或相邻档位，等效于对注意力的尾部做了截断。这正是要用 per-block scale 而不是
  per-tensor 的原因。
* **Online Softmax 的 rescale 与累加器精度耦合**：`O` 的累加在 fp32 完成，
  但每次 rescale 都要把 `O` 乘上 `exp(m_old - m_new)`，低精度累加器在这里会快速
  累积误差。**attention 的累加器用 fp32 几乎是硬要求。**

**MoE。** 三条影响路径：

1. **专家权重的量化收益最大**：MoE 的参数量集中在专家上，量化后同样的显存放得下
   更多专家，或同样专家数下 batch 更大。
2. **路由权重的量化要单独考虑**：路由的 top-k 选择对数值很敏感，
   通常保留 fp32/fp16 不量化；一旦量化，可能选出「不同的专家」。
   这种**离散**错误的表现不是误差变大，而是整段输出结构不同。
3. **分组 GEMM 的 scale 布局更复杂**：CATLASS 的 grouped MX 示例里，
   SFA 是「按拼接后的 M 做 group-major」，而 SFB 是「按 `(E, N)` 做 group-major」：

   ```text
   SFA[sf_group * M_total + m]
   SFB[sf_group * E * N + eid * N + n]
   ```

   （`examples/blockscaled_gemm_sm100/mxfp8_illustrated.md`）
   这里有三个下标维度（组、专家、行列），是 scale 布局最容易写错的地方。

### 6.3 一份检查清单

决定量化一个算子之前，按顺序回答：

1. **这个算子是哪一类**（memory-bound / compute-bound / 容量受限）？收益上限是多少？
2. **量化谁**（只有权重、还是权重 + 激活、还是连 KV cache 一起）？
3. **scale 的粒度**（per-tensor / per-channel / per-group（多大）/ per-block
   （K 方向多大））？粒度与硬件模板兼容吗？
4. **scale 的布局**（group-major 还是行主序、字节序、一个 word 覆盖多少 K）；
   与生产端（量化工具）的约定一致吗？
5. **累加器类型**（fp32 / int32）？中途有没有 cast 回低精度？
6. **参照实现**（量化感知的绝对参照 + 框架对拍）准备好了吗？
7. **容差**（`rtol` / `atol` / `max_mismatched_ratio`）是从 `u` 推出来的，还是抄的？
8. **端到端指标**（ppl / 准确率）有没有测？逐层误差都在预算内不等于模型没退化。

---

## 7. 常见陷阱

这一节把本章与 corpus 里的错误模式汇成清单，每条都给出可复现的凭据。

**（1）scale 布局写错。** 三种典型错误在 `code/05-mx-scale-layout/mx_scale_layout.py`
里都能复现，相对 L2 分别是 5.58（字节序）、2.47（行主序）、2.77（粒度不匹配）。
其中**行主序的 bug 会藏起来**：当 K 不超过 4 个粒度块（只有 1 个 word 组）时，
两种索引恰好等价——脚本输出末尾那个 `True` 就是这个情况。
**对策**：layout 单测必须用至少 2 个组的 shape。

**（2）累加器精度不足。** 最隐蔽的一种：**fp8 输入配 fp16 累加器能通过编译**。
`src/op/tcgen5_meta.h:79-82` 明确允许 f8f6f4 家族配 `c_dtype.is_float16()`。
它能跑，但第 2.5 节实验 C 说明这一档误差可以差 4 个数量级。
**对策**：fp8/fp16 输入的累加器一律写 `T.float32`；K 很长时再考虑分段累加。

**（3）反量化融合位置错误。** 把 per-K-block 的 scale 提到循环外只乘一次，
在 `code/04` 的 B 行可以复现（相对 L2 0.21）。这类错误的特征是
**误差与 K 的位置相关**：K 越大的部分错得越多。
**对策**：把 scale 的下标（组号）与 K 的循环变量画在一起，确认每一块乘的是自己那一组。

**（4）精度验证用错参照。** `code/04` 的 D 行：拿量化前的 fp32 结果当参照，
会有 94% 的元素不匹配，这会逼着你把容差放到没有鉴别力。
**对策**：参照实现里的量化规则与 kernel 完全一致；把「量化损失」与「实现误差」
拆成两个测试。

**（5）把 fp4 当成「更小的 fp8」来估容差。** 第 2.5 节实验 B：e2m1 的实测误差
是 `u/√K` 预算的 2.58 倍，模型在这个精度下已经失效。
**对策**：fp4 的容差必须在真实数据分布上实测，不要套公式。

**（6）E8M0 取整方向写反。** scale 必须**向上**取到 2 的幂
（`ceil_to_ue8m0`），否则 `x/scale` 可能超过 e4m3 的最大值 448 而溢出成 NaN
（e4m3 没有 inf，溢出直接进 NaN 编码 `0x7f`）。
**对策**：量化完之后断言 `max(|xq|) <= 448`。

**（7）用错了 fp8 的变体。** CUDA 上是 `e4m3fn`、ROCm 上默认是 `e4m3fnuz`，
两者的位模式不同（FNUZ 没有负零）。硬编码 `torch.float8_e4m3fn` 在 ROCm 上
会得到错误结果而不是报错。**对策**：用 `determine_fp8_type()` /
`determine_torch_fp8_type()`。

**（8）把 fp8 的 `-0.0` 当普通零。** 逐位比较时 `-0.0 != +0.0`。
`tilelang/utils/tensor.py:25-29` 提供了 `fp8_remove_negative_zeros_`，
但对拍脚本自己写比较逻辑时容易忘。

**（9）int4 打包时丢掉符号位。** `pack_int4` 用的是「低 4 位 + 补码」：
`(a & 0x0F) | ((b & 0x0F) << 4)`。解码必须做**算术**右移（不是逻辑右移），
否则 `-1` 会变成 `15`。`example_dequant_gemm_w4a8.py:10-22` 里的
「先左移 4 位再算术右移 4 位」就是这个技巧。

**（10）跨实现的粒度约定不一致。** NVIDIA 原生 32、TileLang SM100 示例 128、
昇腾 CATLASS 固定 32。生产端与消费端取不同的粒度，`code/05` 里对应 2.77 的相对 L2。
**对策**：把粒度写进 checkpoint 的元数据，而不是靠「约定」。

**（11）只测逐层误差，不测端到端指标。** 每一层的相对 L2 都在预算内，
误差仍可能在几十层里累积到影响 ppl。`examples/bitnet-1.58b/eval_ppl.py`
与 `eval_correctness.py` 就是上游为这件事准备的脚本。

---

## 8. 本章小结

**数值基础（第 2 节）。** `u = 2^-p`：fp32 6.0e-8、fp16 4.9e-4、bf16 3.9e-3、
fp8 e4m3 6.3e-2、fp4 e2m1 2.5e-1。整数对称量化的等价 `u` 是 `1/(2·qmax)`：
int8 3.9e-3（约等于 bf16）、int4 7.1e-2（约等于 fp8 e4m3）。
**粒度决定误差，位宽只是其中一个因素**：per-tensor int8 的实测相对 L2
（4.95e-2）比上界大一个数量级，而 per-channel 立刻回到上界附近（7.67e-3）。

**MX 与块缩放。** 块粒度（MX 原生 32）+ E8M0 scale（2 的整数次幂，
`2^-127 .. 2^127`，没有 0）+ 每 4 个字节打包进一个 `uint32`（group-major 展平）。
E8M0 的缩放只搬指数，是精确操作；这也是它适合做块缩放的原因。

**误差传播。** 第 12 章的 `u/√K` 对 fp16/bf16/e4m3 成立（实测比值 0.44–0.58），
对 int8 偏保守（1.27–1.47），对 fp4 失效（2.58）。**累加器是独立的一档误差**：
同一份 fp16 输入，fp16 累加 1.24e-4、fp32 累加 1.24e-9、
32 段 fp16 + fp32 汇总 2.06e-5。

**TileLang 的表达。** dtype 有统一入口（`get_tvm_dtype`），fp8 的名字由
`determine_fp8_type()` 按平台决定；`T.int4` + `T.int32` 是唯一可以要求
逐位相等（`rtol=0, atol=0`）的路径；`T.gemm_blockscaled` 把粒度作为显式参数，
而昇腾方言把它固定为 32。

**硬件约束。** NVIDIA：scale 住在 TMEM，用 `tcgen05_cp_warpx4` 写入 4 个子列，
`SFA_ID / SFB_ID = (k_start / granularity) % 4` 选字节，SFB 还要复制到 32 个 lane
分区。昇腾：粒度固定 32，反量化发生在 L1→L0。没有块缩放实现的后端会
**直接编译失败**，不会静默退化成普通 GEMM。

**验证方法论。** 先选参照（量化感知的绝对参照是必须的，因为「两边一致」不等于
「两边都对」），再看三个指标（相对 L2 判量级、逐元素 max 找离群、不匹配比例看
是全局还是局部），最后用「误差 / 结果量级」分类：1e0 是逻辑错误，
1e-2 量级且与 `u` 相符是正常数值误差。

**取舍。** memory-bound 看带宽收益、compute-bound 看张量核峰值、容量受限看能不能
放下；attention 的 P 需要 per-block scale，MoE 的路由权重通常不能量化，
分组 GEMM 的 SFB 布局是 `(E, N)` group-major。

## 延伸阅读

* 容差模型的完整推导与实验：第 12 章第 3.7 节；
  `chapters/12-debugging-and-profiling/code/02-tolerance-model/`。
* 四类低精度 GEMM 的结构与调用方式：第 14 章第 5 节。
* 本章引用的每条源码与资料：`references.md`。
* 可复算的数值实验：`code/01`–`code/05`；源码事实核对：`code/06`。
* 图 1–图 5：`figures/`。
