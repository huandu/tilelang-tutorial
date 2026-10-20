# 第 02 章 习题答案

与 [exercises.md](exercises.md) 的 E1–E13 一一对应。每题给“解题思路 → 参考答案 →
常见错误”。所有数字都可以用 `code/` 下的两个脚本复算。

---

## E1（★）

**思路**：回到“算术强度 vs 平衡点”这一个判据，逐阶段看它主要花在哪。

**参考答案**

| 阶段 | 归类 | 理由 |
| --- | --- | --- |
| 前向 | compute-bound | 主体是 GEMM 与 attention，长序列下 FLOPs 远大于必须搬动的字节数（8B 模型、seq 4096：66.1 TFLOP 对 28.45 GB） |
| 反向 | compute-bound | 每个前向 GEMM 的反向是它的 2 倍 FLOPs（$\mathrm{d}A=\mathrm{d}C B^\top$、$\mathrm{d}B=A^\top \mathrm{d}C$ 各 $2MNK$），合计约 3 倍前向 FLOPs |
| 优化器 | memory-bound | 每参数读 grad/param/m/v、写 param/m/v；bf16 梯度 + fp32 状态是 26 字节/参数，算术只有十几 FLOP |
| 通信 | 纯搬运（memory/网络 bandound） | ring all-reduce 搬 $2(N-1)/N$ 倍梯度数据，0 FLOP |

来源：优化器的张量与公式见
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_aolapi_context_ops-nn_aclnnApplyAdamW_md.md:25-60`
（`varRef`/`mRef`/`vRef` 三个输入输出张量 + `grad`）。

**常见错误**：把反向归成 memory-bound（只想到“要读激活”，忽略了它同样是 2 倍 FLOPs 的
GEMM 主场）；或者忘记“反向的 FLOPs 是前向的 2 倍”是对 GEMM 精确、对 attention 偏乐观。

---

## E2（★）

**思路**：数“每读一个字节做多少次乘加”。

**参考答案**

* prefill：$S$ 个 query 与 $S$ 个 key 做 $S\times S$ 的分数矩阵，每个分数上的 QK 与 PV
  各是一次 $d$ 维的乘加对。以 $S=4096$、$d=128$ 计，读入的 Q/K/V 只有
  $4\cdot S\cdot d$ 那么多个数（每 head），而要做 $4S^2d$ 次乘加，算术强度上千。
  这就是 prefill 是 compute-bound 的原因。
* decode：query 只有 1 个 token，但要把整段 KV cache 读一遍。每个 KV 元素参与
  $O(1)$ 次乘加（因为 query 只有 1 行），算术强度降到 1 FLOP/Byte 左右
  （KV 长 32768、32 head、$d=128$ 时是 1.01）。

一句话：**prefill 的 query 与 key 同阶，decode 的 query 是 1。**

**常见错误**：只说“decode 的 batch 小”，batch 小不是原因；即使 batch=32，
每读一个 KV 元素也只多做 32 次乘加，仍然远低于平衡点。

---

## E3（★★）

**思路**：本章用到的两个“规格 ≠ 实际”事实都要给出条件；然后只做两个除法。

**参考答案**

两个事实（都来自
`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml`，
标定条件是该文件注释里写的：2048 MB working set、5 次 ×200 轮、显存 3201 MHz、
算力受 700 W 限制）：

1. fp16 张量核的规格是 989.5 TFLOPS，标定值（burst）是 0.7370 → 729.3 TFLOPS；
   持续值只有 0.6684；
2. HBM 规格是 4800 GB/s，标定值是 0.9380（取各种访问模式的最大值）→ 4502 GB/s；
   纯 copy 混合只有 0.8183 → 3928 GB/s。

厂商规格下的平衡点：$989.5/4.8 = 206.1$ FLOP/Byte（标定后是 162.0）。

会不会改变判据？对本章默认表格里的算子**不会**：它们的算术强度要么远低于 162
（0.17–42），要么远高于 206（1365、2068）。但**边界上的算子会翻转**：
fp16 GEMM 的算术强度约为 $N/3$（$N$ 为边长），边长 512 时是 170.7，
在标定平衡点 162 下算 compute-bound，在厂商平衡点 206.1（以及 H100 的 333）下
算 memory-bound。这说明“拿哪个数当机器参数”会直接影响结论。

**常见错误**：只算两个平衡点就收工，没有去找“落在两个数之间”的算子；
或者把 0.9380/0.7370 说成“实测性能”，它们是对**微基准**的标定，不是本章的实测。

---

## E4（★★）

**思路**：把话题拉回账本。

**参考答案**

设链上有 $k$ 个算子、每个算子的张量有 $N$ 个元素、每元素 $b$ 字节：不融合的访存是
$2kbN$（每个算子一读一写），融合后是 $2bN$（整条链一读一写）。

* **这条链本身的收益**是 $k$ 倍，确实随 $k$ 增长；
* **端到端收益**取决于 $\dfrac{\text{省下的字节}}{\text{模型的总访存}}$，分母是整段程序，
  不是这条链。

场景：一个模型 99% 的访存与算力都在 GEMM 与 attention 上（它们的算术强度远高于平衡点），
只剩一条 100 个算子的逐元素链。把它融成一个 kernel，这条链自己快了 100 倍，
但端到端收益不到 1%。

**常见错误**：把“链条内部的比值”当成“端到端收益”。另外要提醒的是，减少 kernel 数量
还能降低启动开销（3.1 节），但那是另一笔账，且只在 kernel 很小时才重要。

---

## E5（★★）

**思路**：LayerNorm 的访存只和“张量被读/写几趟”有关，算术按每元素 8 FLOP 记。

**参考答案**（$M=4096,H=4096$，$N=MH=16.777\text{M}$ 元素，fp16）

1. 融合：读 $x$ + 写 $y$ = $2N\cdot2 = 67.11$ MB；FLOPs $=8N=134.2$ MFLOP；
   算术强度 $=2.0$ FLOP/Byte；时间下界 $=67.11\text{MB}/4502\text{GB/s}=14.9$ µs。
2. 三趟拆分：读 $x$ 三次 + 写 $y$ 一次 = $4N\cdot2 = 134.22$ MB；算术强度 $1.0$；
   时间下界 $29.8$ µs。
3. 比值 2.0×，两者都是 memory-bound（2.0 与 1.0 都远低于平衡点 162）。
4. $H$ 从 4096 改成 512（$N$ 缩小 8 倍）：算术强度**不变**（还是 2.0/1.0，
   因为它只取决于“读几趟写几趟”），但绝对下界缩小 8 倍（1.9 µs / 3.7 µs）。
   此时要注意：下界落到几微秒的量级，**kernel 启动开销可能已经占主导**，
   模型给出的下界不再有意义（这正是 TileOPs roofline 文档里 latency-bound 那一类）。

**常见错误**：认为“$H$ 变小会让算子的算术强度变低”，算术强度与张量规模无关，
只与访存/算术的**比例**有关；规模影响的是绝对时间，以及“模型还成不成立”。

---

## E6（★★）

**思路**：先写通用式，再代数字。

**参考答案**

1. $N=2048$：FLOPs $=2N^3=1.718\times10^{10}$；bytes $=3N^2b=3\cdot2048^2\cdot2 = 25.17$ MB；算术强度 $=682.7$ FLOP/Byte，**compute-bound**（>333）。
2. 通用式：$\text{AI}=\dfrac{2N^3}{3N^2b}=\dfrac{2N}{3b}=\dfrac{N}{3}$（fp16，$b=2$）。
   令 $N/3=333 \Rightarrow N\approx 999$。也就是说只有边长小于约 1000 的 GEMM 才会
   掉进 memory-bound 区（这里用的是“A/B/C 各读或写一遍”的记账口径，真实实现因为有
   cache 与片上复用，边界会移动）。
3. 复算：

   ```bash
   python3 -c "import op_traffic as t; \
   print(t.judge(t.gemm(2048,2048,2048), t.MACHINES['h100-spec']))"
   ```

   （在该目录下运行）会得到 `compute-bound`；把 2048 换成 512 会得到 `memory-bound`。

**常见错误**：把 $b$ 写成 1（fp8）或 4（fp32）却不说明；或者忘记 $3N^2$ 里的系数 3
（A、B、C 三块）。

---

## E7（★★★）

**思路**：把“$S\times S$ 中间量”的每一趟都写出来，再算比值。

**参考答案**（$B=1$、32 head、$d=128$、fp16，每元素 2 字节）

1. 落 $S\times S$ 的版本：

   * Q/K/V 各读一遍 + O 写一遍：$4\cdot32\cdot S\cdot128\cdot2$ 字节；
   * 分数矩阵：写一次、softmax 读一次写一次、PV 读一次，共 6 趟，
     每趟 $32\cdot S^2\cdot2$ 字节。

   $S=8192$ 时，每趟 $32\cdot8192^2\cdot2=4$ GiB，6 趟 $=24$ GiB $=25.77$ GB；
   加上 Q/K/V/O 的 268.4 MB，总计约 **26.04 GB**。
2. flash 风格：只有 Q/K/V/O 四趟 = **268.4 MB**。
3. 比值约 **97×**。$S$ 翻倍到 16384 时，分数矩阵那 6 趟变成 4 倍（103.1 GB），
   而 Q/K/V/O 只变成 2 倍（536.9 MB），比值升到约 **193×**，**比值随 $S$ 线性增长**，
   因为分子是 $O(S^2)$、分母是 $O(S)$。

   一句话：中间量的字节数与“query 数 × key 数”成正比，而输入输出的字节数只与
   “token 数”成正比。

**常见错误**：只数 2 趟（写+读）而不是 6 趟；或者把 Q/K/V/O 也算成 $O(S^2)$
（它们是 $O(S\cdot d)$，$d$ 与 $S$ 无关）。

---

## E8（★★）

**思路**：逐行对着公式读代码。

**参考答案**

1. 三行 `scores_bytes` 分别是：
   * 第一行 `2.0 * bh * s * s * elem`：**写**一次分数矩阵（`2.0` 是“字节/元素”的系数
     fp16 两字节，代码里用 `2.0 * ... * elem` 表示“读一个、写一个”的通用写法）；
   * 第二行：softmax **读**一次、**写**一次；
   * 第三行：PV **读**一次。
   （三行都是 `2.0 * bh * s * s * elem` 这个表达式，但语义分别是写、读+写、读，
   注释里写明了这一点。）
2. `4.0 * s * s * d` 是 QK 与 PV 两个矩阵乘的 FLOPs（各 $2S^2d$）；
   `5.0 * s * s` 是 softmax 的算术（scale、running max、减法、exp、求和，每分数 5 FLOP，
   这个口径来自 TileOPs roofline 文档）。
3. 如果 softmax 与 PV 融合、但分数矩阵仍然落显存：分数矩阵只需**写一次 + 读一次**
   （融合后的 kernel 读回分数后直接做完 softmax 与 PV），
   即把 `scores_bytes` 改成 `2.0 * bh * s * s * elem * 2`。

**常见错误**：把 `bh` 当成 batch（它是 batch × head）；或者以为 `elem` 是 1
（fp16 是 2）。

---

## E9（★★）

**思路**：读 `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py` 的 94–140 行。

**参考答案**

1. `execution_backend` 的合法取值是
   `Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"]`；不传时读环境变量
   `TILELANG_EXECUTION_BACKEND`（默认 `"auto"`）。`out_idx` 是“哪些输入/输出张量应该作为
   返回值交给调用方”的下标（`list[int] | int | None`）。
2. `target` 接受字符串、字典或 `tvm.target.Target`；docstring 给的例子是
   `{"kind": "cuda", "arch": "sm_90"}`。不传时读 `TILELANG_DEFAULT_TARGET`（默认 `"auto"`）。
3. 一句话：**TileLang 把自己定位在“单个 kernel 的编译与执行”这一层**，
   输入是一个 PrimFunc（一个算子），输出是一个可以通过多种方式（cython / torch /
   tvm_ffi / nvrtc / CuTe DSL）被宿主调用的 `JITKernel`。它不接收计算图，
   也没有图级的算子替换概念。依据就是函数签名：参数只有 `func/out_idx/execution_backend/
   target/pass_configs/compile_flags`，没有任何与“图”“整模型”相关的参数。

**常见错误**：把 `out_idx` 理解成“输出个数”；或者以为 `target="auto"` 会做某种自动选择
（它只是把选择交给环境变量/默认值）。

---

## E10（★★★）

**思路**：这段文字是在解释“为什么这个 key 的每个细节都必须这样”。

**参考答案**

1. 原文说：“The key is a `str` because dynamo treats string operator arguments as
   compile-time constants, while an `int` is generalized to an unhashable `SymInt`.”
   也就是说：用 `str` 时 Dynamo 会把它当编译期常量；用 `int` 时会被泛化成
   `SymInt`，而 `SymInt` 不可哈希，缓存就做不了。
2. 因为 Inductor 会把 fake tensor 的输出形状**烤进**它写到磁盘上的编译产物
   （“inductor bakes the fake's output shape into an artifact it caches on disk”）。
   如果同一个 key 在不同实例之间复用了，key 就会解析到**另一个实例**，
   从而命中一份形状不对的缓存产物。这解释了为什么 key 必须保证唯一。
3. 因为被 Dynamo 追踪的那部分代码是要被捕获成图的，图里只能有“可被追踪的算子调用”；
   而 kernel 构造、缓存查找、启动都发生在 TileLang builder 里，这是追踪器无法捕获的
   副作用。文档把这条写成不变量：“a dynamo-traced `forward` must not construct kernels
   or enter a TileLang builder. Everything past the operator body is untraced, so cache
   lookup, kernel construction and launch belong there.”

**常见错误**：答成“`str` 更好读”，理由与可读性无关，是 Dynamo 的常量折叠与
哈希语义决定的。

---

## E11（★★）

**思路**：跑两组参数，逐行比对判据列。

**参考答案**

```bash
python3 code/01-op-traffic-model/op_traffic.py                      # H200 标定，平衡点 162
python3 code/01-op-traffic-model/op_traffic.py --machine h100-spec  # H100 规格，平衡点 333
```

默认表格里的算子（逐元素加、逐元素链 ×2、LayerNorm ×2、GEMM 4096³、attention prefill ×2、
attention decode、AdamW、all-reduce，算术强度从 0.17 到 2068）**没有一个翻转**：
它们要么远低于 162/333，要么远高于。翻转只出现在边界算子上，例如
fp16 GEMM 边长 512（算术强度 170.7）：在平衡点 162 下是 compute-bound，
在 333（H100）或 206.1（H200 厂商规格）下是 memory-bound。

为什么逐元素加永远不可能变成 compute-bound？它的访存是 $3Nb$ 字节（读 $x$、读 $b$、写 $y$），
算术只有 $N$ FLOP，算术强度恒为 $1/(3b)=1/6$ FLOP/Byte（fp16），**与规模无关**。
要让它变成 compute-bound，机器的平衡点得低于 1/6，也就是带宽比算力贵 6 倍，
现实中不存在这样的加速器。

**常见错误**：以为“大一点就变成 compute-bound”。算术强度是**比例**，
对逐元素、归约这类算子与规模无关；只有 GEMM、attention 这类“计算量随某个维度超线性增长”
的算子，算术强度才会随规模增长。

---

## E12（★★★）

**思路**：固定模型与机器，只改序列长度。

**参考答案**（`python3 code/02-step-budget/step_budget.py --seq-len <S>`）

| seq_len | 前向 | 反向 | 优化器 | 通信 |
| --- | --- | --- | --- | --- |
| 512 | 12.1% | 24.2% | **56.2%** | 7.6% |
| 4096 | 27.9% | 55.9% | 14.3% | 1.9% |
| 32768 | 32.9% | 65.8% | 1.1% | 0.1% |
| 131072 | 33.3% | 66.6% | 0.1% | 0.0% |

1. 占比变化最快的是**优化器**：从 56.2% 掉到 0.1%。
2. 原因：优化器的访存量（8B 参数约 208.78 GB）与“每个参数”有关，**与序列长度无关**；
   而前向/反向的 FLOPs 随序列长度（attention 部分甚至是 $S^2$）增长。
   分母长大，分子不变，占比自然被稀释。
3. 几乎不随序列长度变化的**绝对时间**是优化器（46.375 ms）与通信（6.243 ms），
   注意通信的占比在变，是因为它在**总时间**里的份额变了，它自己的绝对量没变。

顺带一个结论：短序列（batch 小、序列短）的训练里，训练时间的相当一部分花在
“与序列长度无关”的优化器与通信上，这时优化算子的收益上限就是那 40%–60% 之外的余量。

**常见错误**：把“占比下降”当成“绝对时间下降”。绝对时间没变，变的是分母。

---

## E13（★★★）

**思路**：照 `layernorm` 的写法，先定“读什么、写什么”，再定“每元素几个 FLOP”，
最后交给 `judge`。

**参考答案**（示例代码，可直接接在 `op_traffic.py` 后面）

```python
def silu(n: float, dtype: str = "fp16") -> Cost:
    elem = ELEM_BYTES[dtype]
    # sigmoid = neg + exp + add + recip = 4 FLOP；再乘 x = 5 FLOP/元素
    # （计数口径见 TileOPs roofline.md：超越函数在账本上记 1 FLOP）
    return Cost(
        name=f"SiLU ({n/1e6:.1f}M 元素, {dtype})",
        flops=5.0 * n,
        nbytes=2.0 * n * elem,          # 读 x，写 y
        assumption="每元素 5 FLOP（sigmoid 4 + 乘 1）；读 x、写 y",
    )


def rope(b: float, heads: float, s: float, d: float, dtype: str = "fp16") -> Cost:
    elem = ELEM_BYTES[dtype]
    n = b * heads * s * d
    # 每两个元素配成一对：各 2 次乘 + 1 次加/减 = 每对 6 FLOP，即每元素 3 FLOP
    return Cost(
        name=f"RoPE 旋转部分 ({b:.0f}x{heads:.0f}x{s:.0f}x{d:.0f}, {dtype})",
        flops=3.0 * n,
        nbytes=2.0 * n * elem,          # 读 x、写 y；cos/sin 预先算好，忽略
        assumption="每对元素 4 乘 2 加减 = 6 FLOP，即每元素 3 FLOP；cos/sin 预计算",
    )
```

运行结果（H200 标定参数，$N=16.777$M 元素，两者形状取 $B{=}1$、32 head、$S{=}4096$、$d{=}128$，
即 $N=16.777$M）：

| 算子 | FLOPs | 访存 | 算术强度 | 时间下界 | 判据 |
| --- | --- | --- | --- | --- | --- |
| SiLU | 83.9 MFLOP | 67.11 MB | 1.25 | 14.9 µs | memory-bound |
| RoPE（旋转部分） | 50.3 MFLOP | 67.11 MB | 0.75 | 14.9 µs | memory-bound |

两个都是 memory-bound，时间下界完全由“读一遍、写一遍”决定。
如果把它们与残差相加融合进同一个 kernel，**访存量不变**（还是读入一次、写出一次），
融合的收益体现在“省掉了它们各自读写的中间张量”，而不是省掉这两次读写本身。
换句话说：**这几个算子单独看都是 2 趟访存；把它们串起来才会变成 2k 趟。**

**常见错误**：

* 忘记超越函数的计数规则（TileOPs 的口径是账本上记 1 FLOP，不是硬件成本）；
* 把 RoPE 的 `cos/sin` 也算进访存（它们是预计算的，与 token 数同阶但小得多，
  可忽略。但要在答案里说明这个假设）；
* 认为融合会减少“输入→输出”的读写次数。融合减少的是**中间张量**的往返。
