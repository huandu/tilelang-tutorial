# 第 27 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链条较长或需要动手）。
* “需要硬件”一栏里，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）`/`是（NPU）`
  表示必须用真实加速器。本机（macOS、无 GPU/NPU）只能做标 `否` 的题。
* 所有题目只依赖第 27 章及更早的知识；涉及后续章节的地方会显式提示。
* 可用 `code/` 下的六个脚本复算，它们互不依赖。

---

## E1 概念题 ★ 不需要硬件

1. 写出 fp16、bf16、fp8 e4m3、fp8 e5m2、fp4 e2m1 的“符号/指数/尾数”位数、
   有效位数 `p` 与 `u = 2^-p`。
2. 为什么 bf16 选“8 位指数 + 7 位尾数”，而不是像 fp16 那样“5 位指数 + 10 位尾数”？
   请从“溢出/下溢”与“相对精度”两个角度各写一句。
3. 用第 2.4 节的公式算出 int8 与 int4 对称量化的等价 `u`，并指出它们分别与哪个
   浮点格式的相对精度同级。
4. 一句话回答：**量化误差主要由位宽决定，还是由 scale 的粒度决定**？给出本章里的
   一个实测证据。

---

## E2 概念题 ★ 不需要硬件

fp8 有两个版本，e4m3（最大 448，`u = 6.25e-2`）与 e5m2（最大 57344，
`u = 1.25e-1`）。

1. 为什么 e4m3 **没有 inf**、只有两个 NaN 编码？这对量化流程提出什么要求？
2. 训练里“前向/权重用 e4m3、反向梯度用 e5m2”这条经验的依据是什么？
3. 如果在 CUDA 上硬编码 `torch.float8_e4m3fn`，代码搬到 ROCm（非 gfx950）上会出现
   什么现象？为什么说这是“静默错误”而不是“报错”？
4. “e4m3 有负零”这件事会在什么样的对拍代码里造成误判？锁定版本提供了哪个函数
   来规避它？

---

## E3 概念题 ★★ 不需要硬件

参考第 2.3 节与图 2，回答 MX / block-scaled 的三个要素：

1. 块粒度在 PTX 与 CATLASS 里分别是多少？给出你在 corpus / `_vendor` 里找到的
   具体位置。
2. E8M0 的值域是多少？它为什么“不能表示 0”？这对“整块全零”的输入意味着什么？
3. 为什么说“用 E8M0 缩放不引入尾数误差”？请从二进制表示的角度解释。
4. 锁定版本的 SM100 MXFP8 示例用的是 128 粒度，而不是 32。请说明：
   (a) 为什么 128 是合法的；(b) 这一选择与 `sf_load_period` 的关系；
   (c) 昇腾侧能不能做同样的选择。

---

## E4 概念题 ★★ 不需要硬件

一份权重矩阵 `W`（形状 4096×4096）要做 int8 对称量化，三种粒度可选：
per-tensor、per-channel（每输出行一个 scale）、per-group-64（沿 K 每 64 个元素一个）。

1. 分别写出三种方案需要存多少个 scale（浮点数个数）以及相对权重的开销比例。
2. 假设 `W` 的行幅度差异很大（各行 absmax 相差 20 倍），三种方案的相对 L2 误差
   会怎么排序？为什么？
3. 用 `code/02-quant-dequant/quant_dequant.py` 复算，验证你的排序；
   并解释输出里 per-tensor 的相对 L2（4.95e-2）为什么**大于**它的理论上界 3.9e-3。
4. 如果换成 int4（`qmax = 7`），三个数字会各自变成多少量级？哪种粒度才能让
   相对 L2 落在 1e-2 以内？

---

## E5 概念题 ★★ 不需要硬件

1. 写清楚“输入舍入误差”与“累加误差”的区别，并各举一个本章的实验证据。
2. 第 12 章的模型说，长度 K 的归约误差约为 `u/√K`。本章实验 B 里 fp16 的
   “实测 / 预算”在 0.48–0.58 之间，而 e2m1（fp4）是 2.58。请解释后者为什么失效。
3. split-K 把一次长归约拆成若干段。用本章实验 C 的三个数字说明它为什么能保住精度，
   并指出它与“把累加器换成 fp32”在收益来源上的区别。
4. 为什么说“fp8 输入配 fp16 累加器”是一个陷阱？给出锁定版本里允许这种组合的
   源码位置，并说明它为什么能通过编译。

---

## E6 代码阅读题 ★★ 不需要硬件

读 `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py`
的下列片段，回答问题。

```python
# :37-47
SFA: [sf_k_groups * M] in uint32
SFB: [sf_k_groups * N] in uint32
...
k_iters = T.ceildiv(K, block_K)
sf_load_period = sf_granularity_k * 4 // block_K
sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)
```

上游 docstring 的示意式标成了
`[(K / sf_granularity_k) / 4) * M]`，多了一个右括号；下面实际分配代码使用的是
`sf_k_groups * M`，这里按后者引用。

1. 当 `K = 4096`、`sf_granularity_k = 128`、`block_K = 128` 时，
   `k_iters`、`sf_load_period`、`sf_k_groups` 分别是多少？
2. `SFA` 的长度是多少（用 M 表示）？它为什么是 `uint32` 而不是 `uint8`？
3. 第 `:109-120` 行的 `if k % sf_load_period == 0` 这个条件意味着什么？
   如果把它改成每次迭代都搬 scale，功能上还正确吗？代价是什么？
4. 第 `:509-525` 行的 `unpack_sf_u32_1d` / `pack_sf_u8_to_u32_1d` 互为逆操作。
   请写出 `pack` 的表达式，并说明 `packed.T.contiguous().reshape(-1)` 这一步
   改变了什么（即“group-major”的含义）。

---

## E7 代码阅读题 ★★ 不需要硬件

读 `examples/dequantize_gemm/example_dequant_gemm_w4a8.py` 的下列片段：

```python
# :10-22
def _tir_u8_to_i4_to_i8(nbit, val, pos, dtype):
    mask = tirx.const((1 << nbit) - 1, T.uint8)
    i4 = (val >> (pos.astype(T.uint8) * tirx.const(nbit, T.uint8))) & mask
    i8_shifted = tirx.reinterpret(T.int8, i4 << tirx.const(4, T.uint8))
    i8 = i8_shifted >> tirx.const(4, T.int8)
    return i8
```

1. 这三行分别做了什么？为什么不能直接写 `i8 = val >> (pos * 4)`？
2. 用 Python 验证：`val = 0xF0`、`pos = 0` 与 `pos = 1` 时结果分别是多少？
   （提示：模拟 uint8 的位运算与 int8 的算术右移。）
3. `examples/gemm_int4/example_tilelang_gemm_int4.py:59-67` 的 `pack_int4` 是反方向
   的操作。它在打包时有没有处理符号位？如果没有，解码端靠什么还原负数？
4. 为什么这条路径的对拍可以用 `rtol=0, atol=0`，而 fp8 路径不能？

---

## E8 代码阅读题 ★★★ 不需要硬件

读下面两段代码，回答 scale 在 NVIDIA 硬件上“怎么放、怎么取”。

```python
# tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py:629-632
runtime_sf_a_id = ((_sf_k_start + ki * micro_size_k) // int(sf_a_granularity_k)) % 4
runtime_sf_b_id = ((_sf_k_start + ki * micro_size_k) // int(sf_b_granularity_k)) % 4
runtime_instr_desc = base_instr_desc | (runtime_sf_a_id << 29) | (runtime_sf_b_id << 4)
```

```python
# tilelang/language/builtin.py:1440-1452（docstring 节选）
"""Copy one or more packed scale-factor chunks from shared memory to tensor memory.
...
For 1D packed ``uint32`` scale buffers, each 128-word chunk maps to
4 TMEM columns and the column offset is advanced automatically."""
```

1. `% 4` 这个运算的物理含义是什么？为什么是 4 而不是 8 或 16？
2. 设 `sf_a_granularity_k = 128`、`block_K = 128`、`k_start = 256`。
   当 `ki = 0,1,2,3` 时（`micro_size_k = 32`），`runtime_sf_a_id` 分别是多少？
   由此说明“一个 scale 字节能服务几个 K32 原子”。
3. 如果把 `sf_a_granularity_k` 从 128 改成 32 而生产端仍按 128 打包，会发生什么？
   （用 `code/05-mx-scale-layout/mx_scale_layout.py` 的第 3 行结果佐证。）
4. 第二段代码里“每 128 word 的块映射到 4 个 TMEM 列”这一句，
   与第 1 问的“4”是同一个 4 吗？请解释它们的关系。

---

## E9 代码阅读题 ★★ 不需要硬件

读昇腾侧 `_vendor/tilelang-ascend-v0.1.2.000/examples/quant_batch_matmul/example_quant_batch_matmul.py`
的片段：

```python
# :49-88，节选
A_L1 = T.alloc_L1([block_M, block_K], in_dtype)      # int8
B_L1 = T.alloc_L1([block_K, block_N], in_dtype)      # int8
C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)  # int32
...
for bm_v, bn_v in T.Parallel(block_M_2, block_N):
    c_scale[bm_v, bn_v] *= scale_ub[bn_v]
```

1. 这份实现把反量化放在哪一级？对照图 4 的三个位置（prologue / 主循环 / epilogue），
   说明它属于哪一类，以及为什么这个位置在这里是合法的。
2. 如果把 `scale` 改成“沿 K 每 32 个元素一个”的 per-block scale，
   这段代码还能用吗？为什么？
3. 它的对拍判据是什么？请从“误差来源”的角度解释这个判据是不是合理。
4. 结合第 4.2 节，说明昇腾的 MX 路径（`MX_SCALE_GROUP_NUM = 32`）与这份
   per-channel 量化示例在“粒度”上的差别，以及各自适合什么场景。

---

## E10 动手题 ★★★ 不需要硬件

`code/05-mx-scale-layout/mx_scale_layout.py` 里有一个“行主序 bug 会藏起来”的现象。

1. 直接运行脚本，记录第 2 行（行主序）的相对 L2，以及最后那句
   “group-major 与 row-major 的解包结果相同”的取值。
2. 把默认的 `--k 1024` 改成 `--k 512` 再跑一次，第 2 行的结果变成多少？为什么？
3. 用 `--k 2048` 再跑一次。解释结果随 K 变化的原因（提示：看
   `sf_k_groups = ceil(ceil(K/gran)/4)`）。
4. 给这个脚本加一条防御：当 `sf_k_blocks <= 4` 时打印一条警告，说明此时无法用
   这个用例检测出行主序 bug。写出你的实现思路与关键代码。

---

## E11 动手题 ★★★ 不需要硬件

1. 用 `code/03-error-model/error_model.py --trials 400` 跑一遍，
   记录 fp16 与 int8 在四个 K 上的“实测 / 预算”。
2. 把 `EQUIVALENT_U["int8"]` 分别改成 `1/(2*sqrt(12)*127)` 与 `1/127`，
   再跑一次。哪一个是更好的拟合？请从“均匀量化误差的统计性质”解释。
3. 用 `code/02-quant-dequant/quant_dequant.py --rows 64 --cols 4096` 跑一遍
   MXFP8，观察它与默认参数下的相对 L2 有什么差别，并解释原因
   （提示：块数变了，每块的 absmax 也随之变化）。
4. 写一段 10 行以内的 numpy，验证“E8M0 缩放是精确的”：
   对任意 fp32 数值 `x` 与任意 E8M0 scale `s`，`x / s` 的结果在 fp32 下没有舍入误差
   （除指数下溢/上溢）。给出你的验证方式与结论。

---

## E12 动手题 ★★★ 不需要硬件

你要为一个“fp8 e4m3 权重 + per-channel scale + fp32 累加”的 GEMM 写对拍。

1. 根据本章的模型，写出这个场景下预期的相对 L2 量级，并说明推导过程。
2. 设计三个参照（量化感知的绝对参照、框架相对参照、量化前参照），分别说明
   它们能证明什么、不能证明什么。
3. 用 numpy 实现这个场景的最小版本（不需要 TileLang）：随机 M×K 的 fp16 激活与
   N×K 的权重，权重按 per-channel 量化到 e4m3，做 `Y = X @ Wq.T`，
   与三个参照比较，打印相对 L2 / 逐元素 max err / 不匹配比例。
4. 你会把 `rtol` / `atol` / `max_mismatched_ratio` 设成多少？给出数字与理由。
5. 故意在 kernel 侧引入一个 bug（例如把 scale 作用在 N 的错位上），
   说明输出指标会怎么变，以及你会用哪个指标先发现它。

---

## E13 分析题 ★★★ 不需要硬件

下面五组观测都来自“和 PyTorch 参考实现对拍”的场景。对每一组判断它最可能是
哪一类问题（正常数值误差 / 逻辑错误 / 参照选错 / 累加器精度不足），并给出你的
第一步动作。

1. W 是 int4 per-group、X 是 fp16；相对 L2 = `1.1e-1`，逐元素 max err = `1.0`，
   不匹配比例 = `0.94`。
2. W 是 int4、累加器是 int32；相对 L2 = `9.6e-1`，不匹配比例 = `0.99`，
   差异在 N 方向上呈现“每个通道一个固定倍数”的模式。
3. 输入是 fp16、累加器也是 fp16（K=8192）；相对 L2 = `3e-4`，
   把累加器换成 fp32 后降到 `1e-8`。
4. 相对 L2 = `5.6`，且误差与 K 的位置强相关（K 越大误差越大）。
5. 相对 L2 = `2.7e-3`，不匹配比例 = `2e-4`，逐元素 max err = `1.8e-2`，
   误差在整张张量上均匀分布，没有明显的行列结构。

---

## E14 设计题 ★★★ 不需要硬件

你要把一个 70B 级别的 MoE 模型（每层 64 个专家、路由 top-8）部署到单卡上，
显存是硬约束。回答：

1. 你会先量化什么、后量化什么？按收益从大到小排序，并说明每一项的判据。
2. 路由权重（gate）与专家权重应该用不同的量化策略吗？为什么？
3. 分组 GEMM 的 scale 布局与普通 GEMM 有什么不同？参考
   `examples/blockscaled_gemm_sm100/mxfp8_illustrated.md` 给出 SFA / SFB 的下标表达式。
4. 你会用哪些指标来确认“没有量化坏”？逐层误差与端到端指标各占什么位置？
