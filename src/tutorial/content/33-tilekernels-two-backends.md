# 第 33 章 TileKernels（上）：一份算法，两个后端

## 学习目标

读完本章，你应当能够：

1. 说清 TileKernels 的**三层目录约定**，`*_kernel.py`（平台无关的 Python 包装：参数校验、
   输出分配、后端分派）、`*_cuda.py` / `*_asc.py`（两份 `@tilelang.jit` 的 `T.prim_func`）、
   `torch/` + `testing/`（参考实现与对拍工具），并解释每一层**该放什么、不该放什么**，
   以及为什么这样切能让同一份 Python API 同时喂给两个后端。
2. 逐行读懂 `moe/topk_gate_kernel.py` 的后端分派：`is_ascend()` 怎么判断、为什么昇腾分支
   **延迟 import**、`get_num_vec_cores()` / `get_pdl()` 这类运行期探测值如何变成
   `@tilelang.jit` 的**编译期常量**并进入缓存键。
3. 把 `topk_gate` 的 CUDA 版与 Ascend 版**逐段对照**：并说得出
   `T.Kernel(num_tokens, threads=32)` 与 `T.Kernel(num_vec_cores)` 的差别在哪、
   `T.alloc_fragment` / `T.alloc_reducer` / `T.alloc_shared` 各自解决什么问题、
   Ascend 版的 `T.annotate_buffer_versions`、`T.SimdVF()`、`T.Pipelined(..., num_stages=...)`、
   `T.view`、`vector_length = 64` 分别在补偿什么。
4. 解释**为什么两边的并行单位不同**（NVIDIA 的线程/线程块 vs 昇腾的 AI Core / Vector Core /
   SIMD lane），以及为什么 Ascend 版必须**显式**指定缓冲区版本数。
5. 说清量化算子族（per-token / per-block / per-channel cast、cast_back、SwiGLU 融合）三种
   **缩放粒度**的语义差别、fp8/fp4 的换算口径，以及“为什么要把 SwiGLU 和量化融合”，
   并把它们接到第 27 章的数值正确性口径上。
6. 用 **RoPE 与 Rand** 两个小算子说明“双后端”不是大算子的专利；并区分本章中
   **哪些结论是从源码结构直接读出的、哪些必须真机验证**。

## 前置知识

* 第 07 章：`@tilelang.jit`、`T.Kernel`、`T.copy` 的语义，以及 JIT 与缓存。
* 第 09 章：`T.copy` / `T.Parallel` / `T.reduce_*` / `T.fill` 的语义与降级行为。
* 第 11 章：`T.Pipelined` 与多缓冲、`num_stages` 的含义。
* 第 13 章：逐元素、归一化、RoPE 的算子结构与“融合”的动机。
* 第 19 章：昇腾上的最小 kernel、AIC/AIV、UB/L1/L0/L0C、Set/Wait 同步，以及
  **“两棵树”（适配器 A 树 / 主干 M 树）**这张对照表。本章直接建立在第 19 章之上：
  第 19 章教你写**一个**昇腾 kernel，本章教你让**同一个算法**同时拥有两块芯片的实现。
* 第 25 章：昇腾后端 codegen 与 `T.SimdVF` / `T.Pipelined` / AutoSchedule 的原理。
* 第 27 章：低精度与量化的数值口径（block scaling、累加精度、舍入）。本章的量化算子族
  是它在真实库里的落地，本章不重复推导。

不需要你会写 CUDA，也不需要你会写 Ascend C；也不需要你手上有 NVIDIA GPU 或昇腾 NPU
本章所有示例的最高验证级别是**仅静态核对**，下面先说清楚这意味着什么。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**既没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守与第 19 章相同的两条硬规矩：

1. **所有 kernel 示例的最高验证级别是“仅静态核对”**。代码能通过 `python3 -m py_compile`
   或 `python3 -m ast`，调用的 API 逐个在锁定版本源码里比对过**存在性与签名**，
   但**没有任何一条性能数字、任何一个“Kernel Output Match!”来自本机实测**。
2. 任何需要真实硬件才能产生结果的步骤，一律写成**前提 / 命令 / 期望输出 / 验证级别**
   四段式；“期望输出”一律表述为“在满足前提的机器上应当看到”。

本章引用的代码有两类来源，请先认准：

| 记号 | 路径 | 说明 |
| --- | --- | --- |
| **TK 库** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/` | TileKernels 仓库快照，commit `66258df6175d`（2026-09-30） |
| **M 树** | `_vendor/tilelang-v0.1.15/` | TileLang 主干 `v0.1.15`（本章用它核对 API） |

**一条特别重要的边界**：TileKernels 的 Ascend 后端用的是**主干 M 树的昇腾方言**
（`from tilelang.ascend import language as T` / `target='ascend'`），**不是**第 19 章里那个
适配器 A 树（`tilelang.language` + `target='ascendc'/'pto'`）。凡是本章出现“A 树”的地方，
都只是拿第 19 章的对照来帮助理解；TileKernels 一行 A 树代码都没有。

本章出现的数字只有三类来源：**TileKernels 自述文件/上游文档**（带 corpus 路径）、
**上游源码里的常量**（带文件与行号）、**本书的算术推导**（由 `code/` 里的纯 Python 程序复算）。
三者都会显式标注。凡是只有宣传口径、没有条件说明的数字，本章一律不采用。

---

## 1. 问题的由来：为什么是 TileKernels

### 1.1 它在这次开源里的位置

本次（2026-09-30）官方发布说明这样描述这批昇腾组件：

> 本次开源还包括昇腾平台核心计算与通信组件，为不同使用场景提供可复用的基础能力：
> DeepGEMM 加速通用矩阵运算；DeepEP 提供高效的大规模跨设备通信；**TileKernels 提供数据处理
> 所需的常规向量计算和访存算子**；FlashMLA 提供稀疏注意力算子……在多项关键测试用例中，
> 上述组件的计算与通信性能已接近硬件上限。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:12`）

这段话把 TileKernels 的定位说得很准：它**不是**去抢 GEMM 或注意力的风头，而是补上
“其余那些不显眼、但每次前向反向都要跑几十个”的算子，MoE 路由、量化、RoPE、
超连接门控、随机数。这类算子的共同点是：**算法简单、访存密集、形状琐碎**，在真实
训练/推理里占的**总时间**不低，而且极难靠手写 CUDA / Ascend C 逐个维护。

TileKernels 自述文件给出的能力清单是：

> TileKernels is a library of dozens of highly optimized kernels implemented in
> [TileLang] ... It provides kernels for several common operations in LLM training and
> inference, including mixture-of-experts routing, Engram, quantization, and manifold
> hyper-connections. Most kernels achieve performance close to the hardware's compute or
> memory bandwidth limits. **All of these kernels have already been used in our internal
> training and inference workloads.**
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:3`；中文对照见
> 同文件 `:5`）

> 注意： “性能接近硬件上限”“已用于内部训练与推理”是**厂商自述**，本章不把它当作可复现的
> 结论。本章关心的是**代码结构**：这些算子确实存在、确实同时有 CUDA 版与 Ascend 版，
> 而且两份实现共享同一个 Python API。

同一天的 `News` 一节把本章的主题直接写成了标题：

> **[2026-09-30] Huawei Ascend support**: Added Huawei Ascend support ... Following the
> NVIDIA path, the kernels now ship a **second backend that is selected automatically at
> runtime, so the same Python APIs run on both NVIDIA GPUs and Huawei NPUs.**
> （`.../TileKernels/README.md:9`）

自述的依赖门槛也一并抄在这里，后文会用到（`.../TileKernels/README.md:21-31`）：

| 组件 | 要求 |
| --- | --- |
| Python | ≥ 3.12 |
| PyTorch | ≥ 2.13 |
| **TileLang** | **≥ 0.1.15** |
| CUDA 后端 | NVIDIA SM90 / SM100，CUDA Toolkit ≥ 13.1 |
| Ascend 后端 | **Ascend 950 NPU，CANN ≥ 9.2.0** |

注意“TileLang ≥ 0.1.15”这一条：本书锁定的主干正好是 `v0.1.15`，也就是说
**TileKernels 要求的就是本书讲解的那个版本**，不存在“书里的 API 与库里的 API 对不上”
的问题。（`VERSIONS.md`）

### 1.2 为什么值得为它单开一章：它是全书中心问题的实验室

全书反复问的一个问题是：**同一份算法，在不同芯片上到底差多少？**（第 06 章的抽象对照、
第 19 章的昇腾实战、第 25 章的昇腾后端，都在不同层次回答它。）TileKernels 把这个问题的
答案**做成了一份可以 `git clone` 的代码**：同一个仓库里，每个算子都有
`xxx_cuda.py` 和 `xxx_asc.py` 两个文件，它们**实现同一个算法**，但**用 TileLang 写成了
两种完全不同的形状**。

这给本章带来三个别的章节给不了的东西：

1. **“平台无关”与“平台相关”的边界，第一次变成了可见的文件边界。** 你不用猜哪里是
   抽象的尽头，`*_kernel.py` 之外的部分，就是必须为芯片重写的部分。
2. **对照是“同题同解”，不是“同题异解”。** 两个文件的目标函数（算法、数值口径、
   输出布局）完全相同，差异全部来自硬件，因此每一条差异都能干净地归因。
3. **能看出“重写”的工作量到底有多大。** 你可以数一数：同一件事，两边各写了多少行、
   哪些行是逐字相同的、哪些行是必须重写的。

### 1.3 本章范围与路线图

本章只覆盖 TileKernels 里**体量小、但结构完整**的那一半：

| 目录 | 本章讲什么 |
| --- | --- |
| `tile_kernels/moe/`（topk_gate 家族、normalize_weight） | 核心对照样本 |
| `tile_kernels/quant/`（per-token / per-block / per-channel cast、cast_back、SwiGLU fwd/bwd） | 算子族地图 + 融合动机 |
| `tile_kernels/transform/`（RoPE） | 小算子也能双后端 |
| `tile_kernels/rand/` | 一个**诚实的反例**：只有一个后端 |

Engram 与流形超连接（mHC）留给第 34 章。路线图：

```
§2 三层结构        §3 后端分派       §4 top-k gate 逐段对照（本章重心）
   │                   │                          │
   └───────────┬───────┴──────────────┬───────────┘
               ▼                      ▼
        §5 量化算子族          §6 RoPE / Rand
               └──────────┬───────────┘
                          ▼
                 §7 方法论落点（哪些重写、少在哪）
                          ▼
                    §8 小结 + 常见坑
```

---

## 2. 概念与模型：三层结构

### 2.1 三种文件后缀就是三层

打开 TileKernels 的 `tile_kernels/`，你会看到对每个算子都成立的一个命名规律。
以本章的主角 top-k gate 为例：

```
tile_kernels/moe/
├── topk_gate_kernel.py    ← 平台无关：参数校验、输出分配、后端分派
├── topk_gate_cuda.py      ← CUDA 后端：一个 @tilelang.jit 的 T.prim_func
├── topk_gate_asc.py       ← Ascend 后端：另一个 @tilelang.jit 的 T.prim_func
├── moe_topk_gate_forward_{kernel,cuda,asc}.py
├── moe_topk_gate_backward_{kernel,cuda,asc}.py
├── normalize_weight_{kernel,cuda,asc}.py
└── sqrt_softplus.py       ← 两边共享的数学近似（按后端给不同的实现）
```

同一规律在别的目录也一样成立：`quant/per_token_cast_{kernel,cuda,asc}.py`、
`transform/rope_{kernel,cuda,asc}.py`。（`.../TileKernels/tile_kernels/` 目录树，
见 `README.md:73-89` 的 Project Structure 一节。）

外层的两个目录补上这个闭环：

* **`tile_kernels/torch/`**：同一批算子的**纯 PyTorch 参考实现**。例如
  `torch/topk.py:18-21` 的 `stable_topk` 就是“`torch.sort(descending=True, stable=True)`
  取前 k”，三行，慢，但绝对正确。它是两个 kernel 共同的“标准答案”。
  `tile_kernels/torch/__init__.py:1-25` 把这些参考实现集中导出。
* **`tile_kernels/testing/`**：对拍与计时工具。`testing/numeric.py:33-38` 的 `calc_diff`
  是“相对误差”（下文会用到）、`testing/numeric.py:5-30` 的 `assert_equal` 是**逐字节**
  比对、`testing/generator.py` 负责生成被测形状。测试文件（`tests/**`）则把这些拼起来。

### 2.2 每一层该放什么

这是本章第一条要记牢的规则。三层各自的职责边界**很不平均**，大量代码在最外层，
而两个后端各自只留“纯计算”：

| 层 | 放什么 | **不放**什么 |
| --- | --- | --- |
| `*_kernel.py` | 1) 输入校验（`dim`/`contiguous`/`dtype`/shape 上限）<br>2) 把“可选参数是否存在”折叠成布尔开关，供 JIT 做特化<br>3) 输出张量分配（`torch.empty`）<br>4) **后端分派**（`if is_ascend(): ... else: ...`）<br>5) 空输入短路（`if num_tokens == 0: return`）<br>6) 调用 `kernel(...)` | 任何 `T.prim_func`、任何 `T.*` 原语、任何硬件概念 |
| `*_cuda.py` | 一个 `@tilelang.jit(...)` 工厂函数，返回一个 `T.prim_func`；里面只有算法与 CUDA 侧的并行映射 | 参数校验、`torch.empty`、后端判断 |
| `*_asc.py` | 同左，但用 `tilelang.ascend.language` 与 `simd as S`；只有算法与昇腾侧的并行映射 | 同上 |

用一句话概括：**外层的每一行都在回答“要不要算、算什么形状、用哪种芯片”，
内层的每一行都在回答“这块芯片上怎么算”。**

三类“平台相关但又不属于 kernel”的常客，都有各自固定的去处：

| 内容 | 放哪 | 例子 |
| --- | --- | --- |
| 数学近似（不同芯片精度不同） | 单文件、按后端分函数 | `moe/sqrt_softplus.py:23-25` 的 `sqrt_softplus_forward_cuda` 与 `:29+` 的 Ascend 多项式版本 |
| 运行期探测（核数、PDL、UB 容量） | `config.py` | `config.py:24-83` |
| 输出量纲/形状推导、缩放因子分配 | 共享的配置模块 | `quant/common.py` |

### 2.3 一次调用的完整链路

把上面拼起来，一次 `topk_gate(scores, num_topk)` 的完整链路是：

（对应 `figures/01-three-layers.md` 的图 1）

```text
  用户代码            外层（平台无关）                 内层（平台相关）
  ────────            ────────────────                 ────────────────
  topk_gate(s, k)
        │
        ▼
   [校验] dim/contiguous/float32, k<=min(E,32)
        │
        ▼
   [分配] topk_idx = torch.empty((T,k), int64)
        │
        ▼
   [空输入短路] num_tokens == 0 -> 直接返回
        │
        ▼
   [分派] is_ascend()?
     ├─ 否 ──► get_topk_gate_kernel_cuda(E, k, use_pdl=get_pdl())
     │             └─► @tilelang.jit -> 编译/取缓存 -> kernel(...)
     │                    └─► CUDA: T.Kernel(num_tokens, threads=32)  ...
     └─ 是 ──►（延迟 import）get_topk_gate_kernel_asc(E, k, get_num_vec_cores())
                   └─► @tilelang.jit -> 编译/取缓存 -> kernel(...)
                          └─► Ascend: T.Kernel(num_vec_cores)  ...
        │
        ▼
   return topk_idx
```

这张图里有两个细节，很多人第一遍会看漏，但它们是整个设计的支点：

* **分派发生在“工厂函数被调用”这一步，不是“kernel 被执行”这一步。** 两个
  `get_*_kernel` 都是 `@tilelang.jit` 装饰的**工厂**：你调它，它返回一个已经编译好的
  kernel。因此 `is_ascend()` 决定的是**去哪个工厂**，而型号相关的一切（核数、线程数）
  在工厂**被调用时**就被烧进编译产物里了。
* **短路发生在最前面。** `num_tokens == 0` 时直接返回空张量，不编译、不启动。
  这不是优化，而是正确性的一部分：空的 shape 会让后续的 `T.ceildiv`/`align` 产生
  奇怪的结果，也会让 Ascend 版的 `T.device_assert` 在没有任何行可处理时失去意义。

### 2.4 为什么值得这样分层

如果只是“代码整洁”，不值得单开一节。这三层切法真金白银地解决了四个问题：

1. **后端隔离**：`*_cuda.py` 里根本不 import 昇腾的东西，`*_asc.py` 里也根本不 import
   CUDA 的东西。因此“两份实现”在物理上就是两个文件，不存在条件分支缠绕在 kernel 里。
   代价是外层必须知道“现在是谁”，这正是 `config.py` 的职责。
2. **可测性**：`torch/` 提供慢但正确的参考，`testing/` 提供对拍与误差口径。测试文件
   只调用**外层 API**（见 §3.4 的 `tests/moe/test_topk_gate.py`），所以**同一份测试同时
   覆盖两个后端**。这是“一份算法、两个后端”在工程上最划算的地方。
3. **可缓存**：所有“形状相关”的值（`num_experts`、`num_topk`、`num_vec_cores`）都是
   编译期常量，作为工厂函数的**参数**传进去。同一组参数第二次调用直接命中 JIT 缓存
   （第 07 / 26 章），不会重复编译。
4. **可增后端**：加一个后端 = 加一个 `_xxx.py` + 在外层加一个分支。不需要动算法，
   也不需要动测试。

### 2.5 常见坑：这一层最常见的一种错

把**运行期才有的量**（比如 `num_tokens`）当成编译期常量写进 kernel 的 `T.Kernel`
参数里，是这一层最典型的错误。TileKernels 的写法是：

```python
# .../TileKernels/tile_kernels/moe/topk_gate_cuda.py:13
num_tokens = T.dynamic('num_tokens')

@T.prim_func
def topk_gate_kernel(
    scores: T.Tensor[(num_tokens, num_experts), T.float32],
    topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
):
    with T.Kernel(num_tokens, threads=num_threads) as pid:   # 第 22 行
        ...
```

`T.dynamic('num_tokens')` 声明“这个维度运行时才知道”；`T.Kernel(num_tokens, ...)` 于是
变成一个**运行期**网格。这样同一个编译产物可以服务任意 token 数，不必为每个 batch
重新编译；代价是网格尺寸不再是编译期常量，编译器能做的静态检查少了一部分。
`num_experts` / `num_topk` 这类量则相反，它们在 `topk_gate_kernel.py:38` 被当作
`get_topk_gate_kernel_asc(...)` 的参数，**是**编译期常量，因为算子要为它们做布局特化。

---

## 3. 运行期后端选择

### 3.1 `config.py`：全部运行期探测的家

整个后端选择的依据集中在 `tile_kernels/config.py`。先看最核心的四行：

```python
# .../TileKernels/tile_kernels/config.py:24-31
@functools.lru_cache(maxsize=1)
def is_ascend() -> bool:
    return os.path.exists('/dev/davinci_manager')

@functools.lru_cache(maxsize=1)
def get_device() -> str:
    return 'npu' if is_ascend() else 'cuda'
```

**判据是“设备文件是否存在”，不是“设备是否可用”。** `/dev/davinci_manager` 是昇腾
驱动暴露给用户态的设备节点；它在昇腾机器上一定存在，在 CUDA 机器上一定不存在。
这么写有两个后果，都要记住：

* 它是**进程级、一次性的**结论（`lru_cache(maxsize=1)`）。一个进程不可能同时是
  CUDA 机器和昇腾机器，所以缓存不会出错；但这也意味着**同一进程里无法切换后端**，
  想同时对比两个后端只能开两个进程。
* 它**不检查 NPU 是否真的可用**（驱动装了但没卡、卡被占满，它照样返回 `True`）。
  真正的可用性检查发生在后端自己启动 kernel 的时候。

接着是“有多少个核”这件事。这里有一个必须看清楚的区别：

```python
# .../TileKernels/tile_kernels/config.py:39-71
@functools.lru_cache(maxsize=None)
def get_device_num_sms() -> int:
    if is_ascend():
        prop = torch.npu.get_device_properties(torch.npu.current_device())
        return prop.cube_core_num            # 昇腾：Cube 核数
    else:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        return prop.multi_processor_count    # NVIDIA：SM 数

# Pick the accessor that matches the kernel type:
#   - Mix kernel (AIC + AIV): use `get_num_ai_cores`
#   - pure cube kernel:       use `get_num_cube_cores`
#   - pure vector kernel:     use `get_num_vec_cores`
get_num_ai_cores = get_num_sms
get_num_cube_cores = get_num_sms

def get_num_vec_cores() -> int:
    return get_num_sms() * 2
```

`config.py:62-65` 那段注释就是设计意图的自述：**同一个“核数”，在三种核上的含义不同**。
昇腾的一个 AI Core 由 1 个 Cube 核（AIC）和 2 个 Vector 核（AIV）组成（第 19 章 §2.1），
所以纯向量算子要开的并行单位是 `cube_core_num * 2`。这就是为什么
`get_num_vec_cores()` 要乘 2，而 `get_num_cube_cores` 直接等于 `get_num_sms`。

再往下还有两个“容量”探测和一个“开关”：

```python
# .../TileKernels/tile_kernels/config.py:74-83
@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.shared_memory_per_multiprocessor      # 只对 CUDA 有意义

@functools.lru_cache(maxsize=None)
def get_max_ub_per_vector_core(use_simt: bool = True) -> int:
    assert is_ascend()
    return 216 * 1024 if use_simt else 248 * 1024      # 单位：字节
```

`get_max_ub_per_vector_core` 是本文件里**唯一一个硬编码的硬件常量**：每个 Vector 核的
UB 预算按 `use_simt` 取 216 KB 或 248 KB。注意这是**库自己写下的口径**，不是我们从
官方文档核到的容量（第 19 章引的 Atlas A2 口径是 192 KB，两者不是同一款芯片、
也不是同一套预留策略）。RoPE 与 Rand 就是靠这个值反推 buffer 大小的（§6）。

最后是 PDL 开关。它是本章唯一一处 CUDA 专属能力：

```python
# .../TileKernels/tile_kernels/config.py:105-112（另有 96-103 的 reset）
def set_pdl(use_pdl: bool) -> None: ...
def get_pdl() -> bool: ...
```

`use_pdl` 默认 `False`（`config.py:7`），用户可显式打开。PDL 是 NVIDIA 的
Programmatic Dependent Launch：允许后一个 kernel 在前一个 kernel 还没完全结束时
就开始加载（`T.pdl_sync()` / `T.pdl_trigger()`，定义见
`_vendor/tilelang-v0.1.15/tilelang/cuda/language/pdl.py:10-21`）。**它只存在于 CUDA
方言里**：`tilelang.language` 通过 `from tilelang.cuda.language import *` 把它带进来
（`_vendor/tilelang-v0.1.15/tilelang/language/__init__.py:11`），而
`tilelang.ascend.language` 只建立在 `tilelang.language.common` 之上
（`.../tilelang/ascend/language/__init__.py:5,49-56`），因此昇腾分支**根本没有** `pdl_sync`。
这就是 `get_pdl()` 只会出现在 `_cuda.py` 的分派参数里、不会出现在 `_asc.py` 里的原因。

### 3.2 从 `topk_gate` 入口逐行读

下面是整个文件，只有 43 行，但三层结构的三条规则全在里面：

```python
# .../TileKernels/tile_kernels/moe/topk_gate_kernel.py:1-43（逐行）
import torch

from tile_kernels.config import get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.moe.topk_gate_cuda import get_topk_gate_kernel_cuda


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Select the top-k experts per token from scores. ... """
    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts, f'num_topk ({num_topk}) must be <= num_experts ({num_experts})'
    assert num_topk <= 32, f'num_topk ({num_topk}) must be <= 32'

    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    if is_ascend():
        from tile_kernels.moe.topk_gate_asc import get_topk_gate_kernel_asc

        kernel = get_topk_gate_kernel_asc(num_experts, num_topk, get_num_vec_cores())
    else:
        kernel = get_topk_gate_kernel_cuda(num_experts, num_topk, use_pdl=get_pdl())

    kernel(scores, topk_idx)
    return topk_idx
```

逐行拆：

| 行 | 在做什么 | 为什么在这里 |
| --- | --- | --- |
| `:1-4` | 导入：`config` 的三件套 + **CUDA 工厂** | CUDA 工厂在**模块顶层**导入；这意味着**即使在 NPU 机器上 `import tile_kernels` 也会导入 `topk_gate_cuda`**。它只是“定义了一个函数”，不会编译，也不会碰 CUDA 设备。但这是一个值得注意的实现选择（见坑 P2） |
| `:26` | 校验 `2D / contiguous / float32` | 后端无关。kernel 假设输入已经是连续 fp32，把转换交给调用者，避免在 kernel 里塞进一个 layout 分支 |
| `:27-29` | 取形状、校验 `k <= E` 且 `k <= 32` | `k <= 32` 是**算法约束**：两个后端都用“32 路”作为一次向量/一轮 warp 能覆盖的宽度（§4.5）。约束写在共享层，两个后端就不必各写一遍 |
| `:31` | 分配 int64 输出 | 输出 dtype 是 API 契约（`:18` 的 Returns），不是后端细节 |
| `:32-33` | 空输入短路 | 见 §2.3 |
| `:35-38` | 昇腾分支：**函数内** import + 传三个编译期常量 | import 是延迟的；`get_num_vec_cores()` 在这一刻求值，于是**核数被烧进了编译产物** |
| `:39-40` | CUDA 分支：传 `num_experts, num_topk, use_pdl` | CUDA 版不需要核数（网格由 `num_tokens` 决定） |
| `:42` | 同一个调用形式 | 两个后端返回的是**同一种东西**：接受 `(scores, topk_idx)` 的可调用对象 |

`:42` 这一行是整章最值得记的一行。**上游的 kernel 从不“知道”下游有两种后端**，
它只知道“有个东西接受两个张量”。这就是第 03 章讲的“接口与 schedule 解耦”
在真实库里的样子。

### 3.3 与第 19 章对照：TileKernels 用的是 M 树

第 19 章花了很大篇幅讲“昇腾有**两棵树**，写法不通用”。TileKernels 的选择非常干脆：

| 维度 | 第 19 章的 A 树（适配器） | 第 19 章的 M 树（主干） | **TileKernels 选的** |
| --- | --- | --- | --- |
| 导入 | `import tilelang.language as T` | `import tilelang.ascend.language as T` | **M 树**：`from tilelang.ascend import language as T`（`topk_gate_asc.py:2`） |
| `target` | `"ascendc"` / `"pto"` | `"ascend"` | `@tilelang.jit` 默认（`rand/randn_asc.py:7` 里显式写 `target='ascend'`） |
| 目标设备 | A2 / A3 | Ascend 950 | **Ascend 950**（`README.md:30` 明写） |
| UB 分配 | `T.alloc_ub` | `T.alloc_shared` | **`T.alloc_shared`**（`topk_gate_asc.py:31`） |
| 矩阵乘 | `T.gemm_v0` | `T.gemm(..., unit_flag_ctrl=...)` | `T.gemm` |
| 同步 | 手写 `barrier_all` / `set_flag` | **默认自动插入** | 默认自动（`_asc.py` 里几乎没有同步代码） |
| 调度 | 手写为主 + `TL_ASCEND_*` | AutoSchedule 默认开 | AutoSchedule |
| 混核 | `is_npu=True` + `vid` | `T.MixedKernel(N, sids=2)` | 两者都用 |

为什么选 M 树？两个可核实的理由：

* **它要 Ascend 950**（`README.md:30`），而 950 正是 M 树的验证目标（第 19 章 §2.6）。
* **它要自动调度**。这个库里的 Ascend kernel 通篇几乎看不到同步指令，依赖全部由
  AutoSchedule + InsertSync 推导（第 22 / 23 章）。手写 `barrier` 在几十个算子上是
  不可维护的。你会在 §4.7 看到这条选择的代价：**用户必须自己声明缓冲区版本数**
  （`T.annotate_buffer_versions`），因为“几份 buffer”是调度器无法替你猜的语义。

还有一条更硬的判据，可以直接从“有没有这个模块”看出来：TileKernels 的 Ascend 文件里
那句 `from tilelang.ascend.language import simd as S`（例如 `topk_gate_asc.py:3`），
**只在 M 树里成立**，锁定的 v0.1.15 有 `tilelang/ascend/language/simd.py`（57909 字节，
`def vintlv` 在 `:1052`、`def vdintlv` 在 `:1067`），而 A 树
（`_vendor/tilelang-ascend-v0.1.2.000/`）连 `tilelang/ascend/` 这个子包都没有，
跨 lane 归约/重排原语（`vcmax` / `vintlv` 等）在它的整个 `tilelang/` 里一个都搜不到
（它的昇腾向量表面在 `tilelang/language/ascend.py`）。**换句话说：没有 M 树，
这些 kernel 的写法根本无从谈起。** 核对过程与命令见
[`references.md`](references.md) 末尾的“附”一节。

### 3.4 同一套测试同时覆盖两个后端

这是“三层结构”最实际的收益。`tests/moe/test_topk_gate.py:52-59` 全部内容如下：

```python
# .../TileKernels/tests/moe/test_topk_gate.py:52-59
@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_topk_gate(params):
    scores = generate_test_data(params)
    num_topk = params['num_topk']

    topk_idx_ref = torch_stable_topk(scores, num_topk)          # tile_kernels/torch/topk.py:18
    topk_idx = tile_kernels.moe.topk_gate(scores, num_topk)     # 外层 API
    assert_equal(topk_idx, topk_idx_ref)                        # testing/numeric.py:5
```

这段测试里有三件值得学的事：

1. **它只碰外层 API**（`tile_kernels.moe.topk_gate`），所以同一份测试在 CUDA 机器上就跑
   CUDA 实现、在昇腾机器上就跑昇腾实现。测试文件里没有任何 `if is_ascend()`。
2. **它的判据是**逐字节相等（`assert_equal` 先转成 `uint8` 视图再 `torch.equal`，
   `testing/numeric.py:18-21`），不是“误差小于某个数”。top-k 的**索引**是离散量，
   没有“差不多相等”这回事；这直接推高了两个后端的实现要求，**tie-break 规则必须
   完全一致**（§4.8）。
3. **`torch/topk.py:18-21` 的参考实现特意把 stride 改成 `(num_topk, 1)`**
   （`as_strided`），因为 `assert_equal` 默认比对 stride（`testing/numeric.py:14`）。
   这类“慢但严格”的参考实现，是双后端库里唯一能被两边同时信任的共同标准。

### 3.5 常见坑清单（外层层）

| 编号 | 坑 | 症状 | 正确做法 |
| --- | --- | --- | --- |
| **P1** | 用“能否 `import torch_npu`”之类的方式判断后端 | 装了包但没卡时误判 | 用设备节点/运行期能力探测；TileKernels 用 `/dev/davinci_manager`（`config.py:26`） |
| **P2** | 在模块顶层 import **两个**后端的 kernel 模块 | 在昇腾机器上导入 CUDA dialect 也可能踩到 CUDA-only 的东西 | 共享层顶层只 import CUDA 工厂，昇腾工厂**延迟 import**（`topk_gate_kernel.py:4` vs `:36`）。见下方注释 |
| **P3** | 把 `num_vec_cores` 当成运行期值传进 kernel | 每个 batch 重新编译，或静默用错核数 | 在工厂调用时求值，作为编译期常量传入（`topk_gate_kernel.py:38`） |
| **P4** | 用 CUDA 的 `get_num_sms()` 去开昇腾向量核 | 只用了一半的向量核 | 纯向量算子用 `get_num_vec_cores()`（`config.py:70-71`） |
| **P5** | 以为两份实现的参数表一致 | 昇腾版多一个 `num_vec_cores`，CUDA 版多一个 `use_pdl` | 参数表**本来就应该**不同：那是各自的编译期特化旋钮 |
| **P6** | 在昇腾分支里传 `use_pdl` | 报 `AttributeError`，因为 M 树昇腾方言没有 `pdl_sync` | PDL 是 CUDA-only 能力（`cuda/language/pdl.py`） |

> **关于 P2 的诚实说明**：`topk_gate_kernel.py:4` 的顶层 import 之所以没有出问题，
> 是因为 `topk_gate_cuda.py` 只有“函数定义 + `@tilelang.jit` 装饰”，不执行任何 CUDA
> 操作。但同一个仓库里的 `quant/per_block_cast_asc.py:7-9` 特意留了一条注释
> （“Re-bind below the star import”），说明**共享模块的 `import *` 可能把错误的 `T`
> 带进后端文件里**。这是这一层最隐蔽的一类 bug，写自己的库时要留意。

---

## 4. 核心对照：top-k gate 的两个后端

这一节是本章的重心。我们把 `moe/topk_gate_{cuda,asc}.py` 拆成**同一串语义步骤**，
每一步先看 CUDA 怎么写、再看 Ascend 怎么写、最后解释**为什么必须不一样**。

两个文件都很短（CUDA 68 行、Ascend 157 行），意思不是“昇腾实现更啰嗦”，而是
**昇腾版把 CUDA 版交给编译器的那些决定（谁是并行单位、缓冲区有几份、向量有多宽）
全部显式写了出来**。这是本章最想让你记住的一句话。

### 4.1 先把算法说清楚

`topk_gate(scores, num_topk)` 的语义是：对每一行（一个 token 的专家打分），选出
分数最高的 `num_topk` 个专家的**下标**，返回 `int64` 的 `[num_tokens, num_topk]`。

API 契约里有两句关键的话（`.../TileKernels/tile_kernels/moe/topk_gate_kernel.py:22-24`）：

> - Always return the smaller index when there are ties.
> - The output is always contiguous.

“并列时取字典序更小的下标”这一条决定了整个实现形态。`torch.topk` 的并列语义是
**未定义**的，所以参考实现必须换成 `torch.sort(..., stable=True)`（`torch/topk.py:18-21`）。
而在 kernel 里，这条契约把算法钉死成一件事：

```text
for k in 0 .. num_topk-1:
    找到当前未被选中的最大值 m
    在所有「值等于 m」的下标里取最小的那个 idx        ← tie-break
    把 idx 记到输出第 k 列
    把 idx 位置的分数置为 -inf                       ← 不参与下一轮
```

也就是**“重复取 max，并列取最小下标”**。它的代价是 `num_topk` 轮串行，好处是
不需要排序、不需要辅助空间、tie-break 规则天然明确。两个后端都实现这一段，
区别只在“一轮怎么并行地做掉”。

> 另一个必须记住的实现细节：**输入必须是有限的**。两个后端都会检查
> `nan/inf`，并且在发现时通过 `T.device_assert` 让整个 kernel 失败
> （CUDA `topk_gate_cuda.py:40-46`；Ascend `topk_gate_asc.py:66-71,155`）。
> 原因也来自 tie-break 契约：如果输入里有 `NaN`，“最大值”和“相等”都失去定义。

### 4.2 CUDA 版逐段读

完整文件（`.../TileKernels/tile_kernels/moe/topk_gate_cuda.py`）：

```python
# :1-15  ── 导入与工厂签名
import tilelang
from tilelang import language as T

from tile_kernels.utils import align

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel_cuda(num_experts: int, num_topk: int, use_pdl: bool = False):
    num_tokens = T.dynamic('num_tokens')
    num_threads = 32
    num_aligned_experts = align(num_experts, num_threads)

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:      # :22
            if use_pdl:
                T.pdl_sync()                                        # :24
            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            amax_fragment = T.alloc_fragment((1,), T.float32)
            idx_fragment = T.alloc_fragment((num_aligned_experts,), T.int32)
            idx_reducer = T.alloc_reducer((1,), T.int32, op='min')  # :28
            idx_result = T.alloc_fragment((1,), T.int32)
            nonfinite_reducer = T.alloc_reducer((1,), T.int32, op='sum')
            nonfinite_result = T.alloc_fragment((1,), T.int32)
            topk_idx_shared = T.alloc_shared((num_topk,), T.int32)  # :32

            for i in T.Parallel(num_aligned_experts):               # :34-38
                if i < num_experts:
                    scores_fragment[i] = scores[pid, i]
                else:
                    scores_fragment[i] = -T.infinity(T.float32)

            # Invalid input check, single deferred assert to keep the load loop vectorized
            T.reducer_init(nonfinite_reducer)                       # :41-46
            for i in T.Parallel(num_aligned_experts):
                if i < num_experts and not T.isfinite(scores_fragment[i]):
                    T.reducer_update(nonfinite_reducer[0], 1)
            T.finalize_reducer(nonfinite_reducer, nonfinite_result)
            T.device_assert(nonfinite_result[0] == 0, msg='Score input for topk gate must be finite')

            for i in T.Parallel(num_aligned_experts):               # :48-49
                idx_fragment[i] = i

            # Get topk via repeatedly finding max
            for k in T.unroll(num_topk):                            # :52-62
                T.reduce_max(scores_fragment, amax_fragment)
                T.reducer_init(idx_reducer)
                for i in T.Parallel(num_aligned_experts):
                    if scores_fragment[i] == amax_fragment[0]:
                        T.reducer_update(idx_reducer[0], idx_fragment[i])
                T.finalize_reducer(idx_reducer, idx_result)
                topk_idx_shared[k] = idx_result[0]
                for i in T.Parallel(num_aligned_experts):
                    if idx_fragment[i] == idx_result[0]:
                        scores_fragment[i] = -T.infinity(T.float32)

            T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)   # :64
            if use_pdl:
                T.pdl_trigger()                                          # :66

    return topk_gate_kernel
```

逐段解释。

**(a) 并行单位：`T.Kernel(num_tokens, threads=32)`（`:22`）。** 网格是
`num_tokens` 个 block，**每个 block 处理一行、只有 32 个线程**（一个 warp）。
“一行一个 block”让下面的所有跨线程通信都退化成 warp 内通信，
这是后面能用 `alloc_reducer` 的前提。注意 `num_tokens` 是 `T.dynamic` 的
（`:13`），所以网格尺寸运行时才知道。

**(b) 片上存储：fragment 存分数与下标，shared 存输出（`:25-32`）。**
`T.alloc_fragment` 是“按线程切分的寄存器数组”（第 10 章），所以
`scores_fragment` 逻辑上是 32 个 fp32，物理上每个线程拿几个。
`idx_fragment` 同理，装的是“这个位置对应哪个专家下标”。
只有 `topk_idx_shared`（`num_topk ≤ 32` 个 int32）是 shared memory，
因为最后一次拷贝要由全 warp 协作完成。

**(c) 载入与补齐：越界位置填 `-inf`（`:34-38`）。**
`num_aligned_experts = align(num_experts, 32)`（`:15`）把专家数向上对齐到 32。
为什么能这么随意地补齐？因为“并列取最小下标”只会在**真实存在的专家**里发生，
而 `-inf` 永远不可能成为最大值，除非一行的分数全是 `-inf`，那就被 `:40-46` 的
有限性检查拦住了。**这两段是互相依赖的**，缺一个都会让 `-inf` 补齐变得不安全。

**(d) 有限性检查：deferred assert（`:40-46`）。** 注意注释的原话，
“single deferred assert to keep the load loop vectorized”。做法是：
用 `T.Parallel` 并行地**累加**“非有限值的个数”（`op='sum'` 的
`T.alloc_reducer`），最后用一个 `T.device_assert` 断言它为 0。
如果改成在 `T.Parallel` 循环体里逐个 `T.device_assert`，编译器就没法把这个循环
向量化/展开成规整的访存了。**“把断言折叠成一次，换取循环的规则性”**是这一层
最实用的一个模式，值得单独记住。

关于 `alloc_reducer` 本身：它是 v0.1.15 的一等公民，
文档字符串写得很清楚（`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:296-320`）：

> Allocate a reducer: a first-class deferred reduction epoch handle. ...
> The reducer lives in the virtual ``local.reducer`` scope and may only be accessed
> through the epoch operations ... **Physical storage and the cross-thread
> communication plan are chosen by the compiler**; the physical layout can never change
> how many times a logical contribution is combined.

也就是说，“这 32 个线程各自数出来的 0/1 怎么合并成一个数”**不由你决定**，
由编译器按当前后端选（warp shuffle、shared memory 都行）。你要写的只是
“初始化 / 每轮贡献 / 收尾”三步（`reducer_init` / `reducer_update` /
`finalize_reducer`，定义在 `.../tilelang/language/reduce_op.py:358,398,434`）。

**(e) 一轮 top-k（`:52-62`）：三次并行操作 + 一个标量。**
一轮的实际步骤是：

1. `T.reduce_max(scores_fragment, amax_fragment)`：全 warp 求最大值
   （`amax` = absolute max 的命名沿用自量化，这里就是 max）。
2. 用 `op='min'` 的 reducer 收集“所有等于最大值的下标里的最小值”，
   **这一句就是 tie-break 契约的落点**（`:54-58`）。
3. 把选中的位置置 `-inf`，下一轮不再考虑（`:60-62`）。

第 2 步用 `T.alloc_reducer(..., op='min')` 而不是手写 shuffle 归约，
是这份代码最有教学价值的地方：`min` 与 `sum` 的**通信量一样**，但语义完全不同。
把“语义”而不是“通信模式”写出来，编译器才能在昇腾上换成完全不同的实现。

**(f) 输出：int32 shared → int64 global（`:64`）。**
`T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)`：
源是 int32、目标是 int64，`T.copy` 在两侧 dtype 不同时会做逐元素转换
（标量路径见 `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:122-127`，
区域路径在 `copy.cc` 里做同样的转换）。索引非负且 `< 2^31`，所以转换精确。
`disable_tma=True` 是 CUDA 侧的关键字提示（`copy_op.py:77` 的 `tilelang.cuda.language.copy`
才带它）；对这种“只搬几十个字节”的小拷贝，走 TMA（张量内存加速器）的建描述符开销
不划算，所以显式关掉。

**(g) 两个开关：`TL_DISABLE_WARP_SPECIALIZED`（`:7-11`）与 PDL（`:23-24,65-66`）。**
前者关掉 warp specialization：这个 kernel 只有 32 个线程、逻辑上就是“一个 warp 干一件事”，
再去做 producer/consumer warp 拆分只会引入无意义的同步。后者是 CUDA 专属的
Programmatic Dependent Launch，让相邻 kernel 能重叠启动（§3.1）。
**顺序很讲究**：`pdl_sync()` 在读输入**之前**、`pdl_trigger()` 在写输出**之后**，
这样“等上游写完”和“通知下游可读”都落在正确的边界上。

### 4.3 Ascend 版逐段读

昇腾版长得多，我们分三段看：**编译期配置**、**缓冲区与流水**、**SIMD 主体**。

**(a) 编译期配置：显式的向量宽与分组（`:1-23`）。**

```python
# .../TileKernels/tile_kernels/moe/topk_gate_asc.py:1-23
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.utils import align, ceil_div

@tilelang.jit
def get_topk_gate_kernel_asc(num_experts: int, num_topk: int, num_vec_cores: int):
    num_stages = 6
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)

    # Adaptive: use only as many vregs as needed when num_experts <= 256.
    max_vregs_per_group = 6
    num_vregs_per_group = ceil_div(min(num_experts, max_vregs_per_group * vector_length), vector_length)
    experts_per_group = num_vregs_per_group * vector_length
    num_groups = ceil_div(num_experts, experts_per_group)
    num_aligned_experts = num_groups * experts_per_group

    # Round up to a multiple of vector_length for UB alignment of the index output.
    num_aligned_topk = align(num_topk, vector_length)
    num_tokens = T.dynamic('num_tokens')
```

四个常量决定了后面的一切：

* `vector_length = 64`：**一条向量寄存器（vreg）装 64 条 fp32 SIMD lane**。
  这是硬件的显式宽度，不是编译器猜的，第 19 章讲的“SIMD lane 才是昇腾上的
  『线程』”在这里落成了代码。
* `max_vregs_per_group = 6`：一组最多用 6 个 vreg，即一组最多覆盖
  `6 × 64 = 384` 个专家。加上注释里那句“when num_experts <= 256”，
  可以推出作者的实际区间：`num_experts ≤ 384` 时 `num_groups == 1`，
  走**快路径**（`:73-105`）；超过就走**慢路径**（`:106-151`）。
* `num_aligned_experts`：把专家数向上对齐到 `experts_per_group` 的整数倍，
  让“按组遍历”的循环上界是常量。
* `num_aligned_topk = align(num_topk, vector_length)`：**顺带解决了 int64 视图的空间问题**
  （见 (c) 的 `T.view`）。`num_topk ≤ 32` 时它至少是 64，而 int64 视图需要
  `2 × num_topk ≤ 64` 个 int32 槽，刚好放得下。

**(b) 网格、缓冲与流水（`:30-48`）。**

```python
# :30-48
with T.Kernel(num_vec_cores) as core_id:
    scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
    out_ub = T.alloc_shared((num_aligned_topk,), T.int32)
    invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
    topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
    T.annotate_buffer_versions({scores_ub: num_stages, out_ub: num_stages, invalid_logits_ub: 1})

    with T.SimdVF():
        T.clear(out_ub)
        T.clear(invalid_logits_ub)

    for w in T.Pipelined(
        T.ceildiv(num_tokens, num_vec_cores),
        num_stages=num_stages,
        annotations={'multi_buffer_eligible': [scores_ub, out_ub]},
    ):
        row = w * num_vec_cores + core_id
        if row < num_tokens:
            T.copy(scores[row, :], scores_ub[:num_experts])
            ...
```

这里有五个必须逐个说清的决定：

1. **网格是 `num_vec_cores`，没有 `threads=`（`:30`）。** 对比 CUDA 的
   `T.Kernel(num_tokens, threads=32)`：昇腾的 grid 就是“Vector 核的编号”，
   每个核处理一行；`with T.Kernel(N) as bx` 里的线程域是**不存在**的
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-172` 的
   docstring：“There is no SIMT thread domain at this scope, so this dialect has no
   `threads` parameter ... Use `T.SimtVF(threads=...)` inside the body to run
   thread-parallel code”）。
2. **`T.alloc_shared` 在这里是 UB，不是 CUDA 的 shared memory。** M 树的昇腾方言故意
   让 `alloc_shared` 指向 UB（`.../tilelang/ascend/language/__init__.py:8-18` 的注释：
   “alloc_shared deliberately shadows the common surface: Ascend UB takes bool buffers
   in the requested scope”）。三个 UB 缓冲各自承担一个角色：
   `scores_ub` 放分数、`out_ub` 放输出、`invalid_logits_ub` 放**跨所有迭代累计**的
   非法标志。这正是为什么它的版本数是 `1` 而另外两个是 `num_stages`（`:35`）。
3. **`T.view` 把 int64 输出当成 int32 来写（`:34`）。** `T.view` 的契约是
   “shape 与 dtype 必须保持总位数不变、源必须紧凑且 `elem_offset` 为 0”
   （`_vendor/tilelang-v0.1.15/tilelang/language/customize.py:101-140`）。
   `[num_tokens, num_topk] int64` → `[num_tokens, num_topk*2] int32` 满足这个条件。
   于是“写一个 int64 下标”变成“写两个 int32 槽：低位写下标、高位保持 0”。
   注意到 UB 的写入用 `'ONEPT_B32'`（单 lane 写，`:101,139`），而高位之所以是 0，
   是因为 `T.clear(out_ub)` 在最前面把整个缓冲区清过零。**这不是顺手为之，
   而是 int64 视图正确性的必要条件**。
4. **`T.annotate_buffer_versions` 显式声明版本数（`:35`）。** 语义见
   `.../tilelang/ascend/language/annotations.py:19-65`：值为 `1` 表示
   “退出多缓冲资格”；`≥2` 表示“固定用这么多份”；也可以只给模式
   （`"auto"` / `"iteration"` / `"counter"`）。这里 `scores_ub` 与 `out_ub`
   各要 6 份，`invalid_logits_ub` 只要 1 份。**为什么必须手写？** 因为这三个缓冲区
   讲的是三件不同的语义：前两个是“每个迭代一份、可以重叠”的流式数据，
   第三个是“整个 kernel 一份”的累积器。调度器可以从依赖图推出“有几处读写”，
   但推不出“这份数据是每轮重置还是全程共享”。**那必须由你告诉它**。
5. **流水结构：`T.Pipelined(count, num_stages=..., annotations={'multi_buffer_eligible': [...]})`
   （`:41-45`）。** 外层是“每核要处理多少个 token”的循环，`num_stages=6` 表示
   流水深度 6 级；`multi_buffer_eligible` 进一步告诉调度器“这两个缓冲可以
   化成多份以支撑流水”。注意工作划分是 `row = w * num_vec_cores + core_id`
   （`:46`），**按核交错**分配 token（核 0 处理第 0、V、2V… 行），而不是
   把 token 连续切给每个核。这样每次迭代搬进 UB 的都是“一整行”，
   与 `T.copy(scores[row, :], scores_ub[:num_experts])`（`:48`）的连续读天然匹配。

**(c) SIMD 主体：显式的向量槽位（`:50-105` 快路径）。**

```python
# :50-71  ── 局部变量、掩码、非有限值检测
with T.SimdVF():
    mask_32b = S.pset(32, 'PAT_ALL')
    one = S.pset(32, 'PAT_VL1')

    neg_inf = S.vdup(-T.infinity(T.float32), T.float32, mask_32b)
    int_max = S.vdup(T.max_value(T.int32), T.int32, mask_32b)

    scores_vec = S.alloc_local((num_vregs_per_group,), T.float32)
    index_vec  = S.alloc_local((num_vregs_per_group,), T.int32)
    tmp_idx_vec = S.alloc_local((num_vregs_per_group,), T.int32)

    lanes = S.vci(0, T.int32)
    zero = S.vdup(0.0, T.float32)
    zeros = S.vdup(0, T.int32)
    invalid = T.alloc_var(T.int32x64, init=S.vld(invalid_logits_ub[0]))
    for j in T.unroll(ceil_div(num_experts, vector_length), explicit=True):
        valid = S.vcmps(S.vadds(lanes, j * vector_length), num_experts, op='lt')
        logit = S.vsel(S.vld(scores_ub[j * vector_length]), zero, valid)
        nonfinite = S.vcmps(S.vsub(logit, logit), 0.0, op='ne')
        invalid = S.vmax(invalid, S.vsel(S.vdup(1, T.int32), zeros, nonfinite))
    S.vsts(invalid_logits_ub[0], S.vcmax(invalid), one)
```

* `S.pset(32, 'PAT_ALL')` / `S.pset(32, 'PAT_VL1')`：**谓词掩码**，分别表示
  “全 64 个 32 位 lane 都置位”和“只置第 1 个 lane”。SIMD 原语大多要显式带掩码，
  因为硬件是 64 lane 一起算的（第 19 章 §7.5）。
* `S.vci(0, T.int32)` 生成 lane 下标向量 `[0,1,...,63]`，用它和常量比较就能造出
  “有效 lane”掩码，替代了 CUDA 版里“越界填 `-inf`”的补齐动作，
  **昇腾版靠掩码，CUDA 版靠补齐**，这是两种机器上等价的两种做法。
* `nonfinite = S.vcmps(S.vsub(logit, logit), 0.0, op='ne')`：**用 `x - x != 0`
  统一检测 NaN 和 ±inf**。有限数相减得 0；`NaN`/`inf` 相减得 `NaN`，
  而 `NaN != 0` 为真。这是比 `isfinite` 更省指令的写法，代价是可读性，
  读代码时要反过来推才知道它在干什么。
* `invalid` 是一个 `T.int32x64` 向量寄存器，初值从 `invalid_logits_ub` 读入、
  每行 `S.vmax` 累积、每轮写回。**这就是“一个跨迭代累积的标志位”在 SIMD 上的形状**：
  用 `1` 打标、用 `max` 归约，最后在循环外一次性 `T.device_assert`
  （`:155`），和第 4.2(d) 的 deferred assert 是完全同一个思路，只是这里
  “累积”发生在 UB 里而不是 reducer 里。

接下来是快路径的主体：

```python
# :73-105  ── num_groups == 1：一组 vreg 装下所有专家
if num_groups == 1:
    for i in T.unroll(num_vregs_per_group, explicit=True):
        scores_vec[i] = S.vld(scores_ub[i * vector_length])
        index_vec[i] = S.vci(T.int32(i * vector_length), T.int32)

    num_exp = S.vdup(T.int32(num_experts), T.int32, mask_32b)
    for i in T.unroll(num_vregs_per_group, explicit=True):
        scores_vec[i] = S.vsel(scores_vec[i], neg_inf, S.vcmp(index_vec[i], num_exp, mask_32b, 'lt'))

    acc_max = S.alloc_var(T.float32)
    acc_min_idx = S.alloc_var(T.int32)
    for k in T.serial(num_topk):
        acc_max = scores_vec[0]
        for i in T.unroll(1, num_vregs_per_group, explicit=True):
            acc_max = S.vmax(acc_max, scores_vec[i], mask_32b)
        mx = S.vdupv(S.vcmax(acc_max, mask_32b), mask_32b)

        for i in T.unroll(num_vregs_per_group, explicit=True):
            tmp_idx_vec[i] = S.vsel(index_vec[i], int_max, S.vcmp(scores_vec[i], mx, mask_32b, 'eq'))
        acc_min_idx = tmp_idx_vec[0]
        for i in T.unroll(1, num_vregs_per_group, explicit=True):
            acc_min_idx = S.vmin(acc_min_idx, tmp_idx_vec[i], mask_32b)
        idx = S.vdupv(S.vcmin(acc_min_idx, mask_32b), mask_32b)

        S.vsts(out_ub[2 * k], idx, one, 'ONEPT_B32')

        for i in T.unroll(num_vregs_per_group, explicit=True):
            scores_vec[i] = S.vsel(neg_inf, scores_vec[i], S.vcmp(index_vec[i], idx, mask_32b, 'eq'))
```

把这段和 CUDA 版的 `:52-62` 摆在一起，你会看到**同一个算法的两种“一轮”**：

| 步骤 | CUDA（一个 warp，32 线程） | Ascend（一个 Vector 核，64 lane × N 个 vreg） |
| --- | --- | --- |
| 求最大值 | `T.reduce_max`（编译器选 shuffle 或 shared） | `S.vmax` 把所有 vreg 折成一个，再 `S.vcmax` 做跨 lane 归约 |
| 取并列最小下标 | `op='min'` 的 `T.alloc_reducer` | 先 `S.vsel` 把非并列位置换成 `INT_MAX`，再 `S.vmin` + `S.vcmin` |
| 写第 k 个结果 | 写 shared，最后统一拷贝 | `S.vsts(..., one, 'ONEPT_B32')` 直接写单个 lane |
| 屏蔽已选 | `T.Parallel` + 条件赋值 | `S.vsel` 用掩码把并列位置换成 `neg_inf` |

`S.vdupv(v, mask)` 的语义是“把标量 `v` 广播成一个向量”；`S.vcmax` / `S.vcmin` 是
“跨 lane 归约”；`S.vsel(a, b, mask)` 是“mask 为真取 `a`、否则取 `b`”
（定义见 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py:922,969,976,1429`）。
也就是说，因为硬件没有“32 个线程各自算完再交换”这回事，
**“归约”在昇腾上必须由一个显式指令完成**；CUDA 上它可能是一串 shuffle，
你不需要知道具体是什么。

**(d) 慢路径与写回（`:106-157`）。** 当 `num_groups > 1` 时，一组 vreg 装不下所有专家，
于是每一轮 `k` 被拆成三个阶段：**Phase 1** 逐组求局部最大值并累积全局最大值；
**Phase 2** 逐组求出“等于全局最大值的 lane 里最小的下标”；**Phase 3** 把这个中奖
lane 在 UB 里置成 `neg_inf`（`:114-149`）。Phase 3 每次都要写回 UB，所以作者在循环末尾
插了一句 `S.mem_bar('VST_VLD')`（`:151`）。**这是全文件唯一一句手写的同步**，
因为“刚写进 UB 的值，下一次迭代要读回来”这条依赖，在 SIMD 指令流里必须显式保证
（第 19 章 §2.4）。

最后两行收尾：

```python
# :153-155
T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])
...
T.device_assert(invalid_logits_ub[0] == 0, msg='topk_gate input contains nan or inf!')
```

`:153` 的拷贝搬的是 `num_topk * 2` 个 int32（即 `num_topk` 个 int64）到
int32 视图的那一行。注意 `:155` 的断言**在 `T.Pipelined` 循环之外**：
`invalid_logits_ub` 只有 1 个版本、被所有迭代共享，所以在所有迭代结束后统一检查。
这也是“为什么它的版本必须是 1”的答案。**它是个累积器，不是流水数据**。

### 4.4 两个后端并排对照

（对应 `figures/02-topk-gate-compare.md` 的图 2）

| 维度 | CUDA 版 `topk_gate_cuda.py` | Ascend 版 `topk_gate_asc.py` |
| --- | --- | --- |
| 网格 | `T.Kernel(num_tokens, threads=32)` `:22` | `T.Kernel(num_vec_cores)` `:30` |
| 并行单位 | 1 block / token，1 warp / block | 1 Vector 核 / token，64 lane / 指令 |
| 一行的存放 | `T.alloc_fragment`（寄存器）`:25-27` | `T.alloc_shared`（UB）`:31` |
| “归约”的表达 | `T.alloc_reducer(op='min'/'sum')` + `reducer_init/update/finalize` `:28,41-45,54-58` | `S.vmax`/`S.vmin`/`S.vcmax`/`S.vcmin` `:89,98-99` |
| 越界元素 | 补齐成 `-inf` `:37-38` | 掩码屏蔽 `:81,120` |
| 非法值检测 | reducer 累加 + 循环外一次 `T.device_assert` `:41-46` | `x-x != 0` 打标 + `vmax` 累积 + 循环外断言 `:66-71,155` |
| 网格剩余处理 | 网格 = token 数，天然整除 | `if row < num_tokens` 守卫 `:47` |
| 多迭代工作划分 | 无（一 block 一行） | 按核交错：`row = w*V + core_id` `:46` |
| 流水 | 无（一行一批做完） | `T.Pipelined(..., num_stages=6)` `:41-45` |
| 缓冲版本 | 由编译器决定 | **显式** `T.annotate_buffer_versions` `:35` |
| 显式同步 | 无 | 一句 `S.mem_bar('VST_VLD')` `:151` |
| 向量宽度 | 由编译器选（`T.Parallel` + 向量化） | **写死** `vector_length = 64` `:11` |
| int64 输出怎么写 | `T.copy` 做 int32→int64 转换 `:64` | `T.view` 成 int32 视图、逐 lane 写 `:34,101` |
| 启动/结束提示 | `T.pdl_sync()` / `T.pdl_trigger()` `:23-24,65-66` | 无（PDL 是 CUDA-only） |
| pass_configs | 关掉 warp specialization `:8-10` | 无（AutoSchedule 默认） |

这张表现在可以回答本章最开始那个问题的一半了：**同一份算法，两个后端的差异不是
“算法差异”，而是“并行映射差异”**。上表里没有一行改变算法语义，每一行都在回答
“这块芯片上的并行单位是什么、数据放哪、什么时候同步、缓冲区有几份”。

### 4.5 为什么两边的并行单位根本不同

（对应 `figures/05-parallel-model-map.md` 的图 5）

上半张表里最容易被当作“风格差异”的一行，其实是整个对照的根因：

| | NVIDIA CUDA 后端 | 昇腾 Ascend 后端 |
| --- | --- | --- |
| 你**必须**指定的 | 网格尺寸 + 每 block 线程数 | 只用网格（核）尺寸 |
| “线程”存在吗 | 是（`threads=32`），线程有独立寄存器和栈 | **在核这一层不存在**；只在 `T.SimtVF(threads=...)` 里出现 |
| 一条指令作用于 | 32 个线程，每个线程 1 个标量（SIMT） | 64 条 lane，一次算 64 个标量（SIMD） |
| 每个“并行单位”拿到什么 | 逻辑上是一个标量：线程 t 处理元素 t | 逻辑上是一个 64 元素的向量 |
| 跨单位通信 | `__shfl_sync`、shared memory、`T.alloc_reducer` | `S.vcmax`/`S.vcmin`/`S.vcadd` 等**跨 lane 归约指令**，或走 UB |
| 代码里看得见宽度吗 | 看不见（`T.Parallel(32)` 就是一个循环） | **看得见**（`vector_length = 64`、`T.int32x64`、`PAT_ALL`） |

把它讲成一句话：

> CUDA 的并行是**“32 个独立的小计算”**，SIMD 宽度是编译器在向量化时自己选的；
> 昇腾的并行是**“一条 64 宽度的向量指令”**，你就得按这个宽度写代码。

这解释了两个后端里最刺眼的两处不同：

* CUDA 版用 `T.Parallel(num_aligned_experts)` 处理“每个线程管几个专家”，
  写法完全不依赖 32 这个数；换成 64 线程数的实现，源码几乎不用改。
* Ascend 版必须显式管理“一组几个 vreg、一组覆盖多少个专家、分几组”
  （`:14-18`）。因为**每条指令一次只吃 64 个 lane**，专家数超过一组能覆盖的范围
  就得自己分组（`:106-151` 的慢路径）。这不是风格，是硬件的形状。

对读者最实用的一条推论是：**在昇腾上，“tile 大小”不是个性能参数，而是个语法参数。**
你在 CUDA 上写 `block_N = 128` 只是想让 GEMM 好看；你在昇腾上写
`experts_per_group = 384` 直接决定了代码走哪条分支。

### 4.6 为什么 Ascend 版要显式管缓冲区版本数

（对应 `figures/03-ascend-ub-pipeline.md` 的图 3）

Ascend 版把“一行”拆成了 **GM → UB（MTE2 搬运）→ 计算（Vector）→ UB → GM（MTE3 搬运）**
四段，并且用 `num_stages = 6` 把它们叠成流水：当第 `w+1` 行正在被搬进 UB 时，
第 `w` 行正在被计算、第 `w-1` 行正在被写回。这就是 `T.Pipelined` 的全部意义
（第 11 章）。

但要流水，就必须回答一个问题：**同时在做三件事，UB 上要放几份 `scores_ub`？**
答案显然是“至少 3 份”，写第 `w+2` 行、算第 `w+1` 行、读第 `w` 行同时发生。
`T.annotate_buffer_versions({scores_ub: num_stages, ...})` 就是在回答它。

为什么调度器不能自己算？看 `annotations.py:19-45` 的 docstring 就明白了：

> When using `T.Persistent` with `num_stages > 1`, the Z3 auto-scheduler computes
> optimal buffer version counts for each on-chip buffer. Each mapping value may be:
> `1` opt out of multi-buffer eligibility ...; `num_versions >= 2` use a fixed version
> count ...; `mode` select a mode while leaving the version count to the scheduler.
> ... A fixed count is useful when you want to **guarantee** double-buffering for a
> critical buffer regardless of the Z3 solver's choice.

也就是说，**调度器确实会算，但算的是“在给定依赖下最少要几份”**。而
“我想更深地重叠”是**性能意图**，不是依赖约束，除非你明确说出来。
再加上一点：昇腾的 UB 容量是硬上限（`config.py:81-83` 里库自己的口径是每核
216 KB / 248 KB），份数越多越可能撞上容量墙。所以这份文件里三件事，
`num_stages`、`annotate_buffer_versions`、`multi_buffer_eligible`，是**互相制约**的
三个旋钮，必须由写 kernel 的人一起定。

对照 CUDA 版：CUDA 版根本没有“一行一行流”的结构，一个 block 处理一行、
做完就退出，**不存在缓冲区复用的时间窗口**，所以从来没有这个问题。
这就是“同一份算法，Ascend 版必须显式管缓冲版本”的全部原因：
**流水是昇腾侧为了掩盖搬运延迟而引入的结构，而 CUDA 侧用的是“网格并行度”掩盖它。**

### 4.7 int64 输出在两边怎么落地

一个很具体但很有意思的差异。API 承诺输出 `int64`，而两块芯片的原生索引宽度都是
32 位（CUDA 侧 `idx_local`/`idx_reducer` 全是 `T.int32`；Ascend 侧全在 32 位 lane 上算）。
两边的解法不同：

| | 做法 | 依赖的前提 |
| --- | --- | --- |
| CUDA `:64` | `T.copy` 从 int32 shared 拷进 int64 global，靠逐元素转换 | `T.copy` 的 dtype 转换（`copy_op.py:122-127`） |
| Ascend `:34,101,153` | 把 int64 张量 `T.view` 成 `[num_tokens, 2*num_topk] int32`，低位写索引、高位靠 `T.clear` 保持 0 | `T.view` 的“总位数不变 + 源紧凑 + `elem_offset==0`”契约（`customize.py:101-140`） |

两种做法都成立，但**约束不一样**：CUDA 版依赖 copy 的隐式转换；
Ascend 版依赖“高位是 0”这个不变量，而它是由 `T.clear(out_ub)` 建立的。
如果哪天有人把 `T.clear(out_ub)` 删掉（比如为了省指令），
**索引会变成“低位正确、高位是垃圾”的巨型数字**，而且这种错误在
小 shape 的测试里可能看不出来。这是 §4.3(b) 里那句“这不是顺手为之”的分量所在。

### 4.8 常见坑（kernel 层）

| 编号 | 坑 | 为什么危险 |
| --- | --- | --- |
| **K1** | 在 `T.Parallel` 循环体里逐个 `T.device_assert` | 打断循环的规则性；CUDA 版特意折叠成一次 assert（`:40`） |
| **K2** | 昇腾版忘了 `T.clear(out_ub)` | int64 视图的高位变垃圾（§4.7） |
| **K3** | 把 `invalid_logits_ub` 也加上多版本 | 它就不再是“全程一个累积器”，`T.device_assert` 会漏检前面迭代的坏值（`:35`） |
| **K4** | 在昇腾分支里写 `T.pdl_sync()` | M 树昇腾方言没有这个 API（§3.1） |
| **K5** | 用 `-inf` 补齐昇腾的越界 lane | 昇腾版靠掩码（`:81,120`）；补 `-inf` 会与“非法值检测”互相干扰 |
| **K6** | 以为两边 tie-break 规则可以不同 | 测试是**逐字节**比对（`testing/numeric.py:18-21`），任何 tie 差异都会 red |
| **K7** | 把 `num_vec_cores` 改成运行期值 | 会让 `experts_per_group` 等常量退化，慢路径与快路径的选择也随之漂移 |
| **K8** | 在 SIMD 段里忘了 `S.mem_bar('VST_VLD')` | 刚写 UB 的值下一轮读回来时可能还没落地（`:151`） |

---

## 5. 量化算子族：三种粒度，两套后端

（对应 `figures/04-quant-family.md` 的图 4）

`tile_kernels/quant/` 是本章第二大的一块，也是最容易“看名字以为懂了、其实没懂”的一块。
它一共只有 **三种缩放粒度**、**两条数据方向**、**一个融合**：

| 名字 | 缩放粒度 | 语义 |
| --- | --- | --- |
| `per_token_cast` | 每个 token 一组（沿通道切） | `sf_block = (1, num_per_channels)`：一行里的第 `j` 列用 `sf[i, j // num_per_channels]` |
| `per_block_cast` | 每个 2D 块一组 | `sf_block = (num_per_tokens, num_per_channels)`：块内共享一个尺度，主流配置是 `(32, 32)` / `(128, 128)` |
| `per_channel_cast` | 每个通道组跨多个 token | `sf_block = (num_per_tokens, 1)`：一列上连续 `num_per_tokens` 行共享一个尺度 |
| `cast` | 无（逐元素） | 纯 dtype 转换，可选随机舍入 |
| `cast_back` / `per_token_cast_back` | 与输入侧同构 | 反量化：`x * sf` 还原成 bf16/fp32 |
| `swiglu_forward_and_per_token_cast` | per-token | SwiGLU 前向 + 量化融合成一个 kernel |

三类粒度的差别**不是“块大小”的差别，而是“尺度作用在哪个维度”的差别**：

| | 尺度沿哪个维度变化 | 典型用途 |
| --- | --- | --- |
| per-token | 只随**行与列块**变化，块大小 1×K | 激活（每个 token 的动态范围不同） |
| per-block | 行列都变 | 权重与激活，MXFP4/MXFP8 类分块量化 |
| per-channel | 只随**列块**变化，块大小 M×1 | 需要跨多个 token 统计的通道尺度 |

### 5.1 共享的配置层：`quant/common.py`

三个 cast kernel 的“什么算合法、尺度怎么存”全部集中在 `quant/common.py`。
核心是两行：

```python
# .../TileKernels/tile_kernels/quant/common.py:19-33
@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float8_e4m3fn
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False
    use_e4m3_sf: bool = False

    @property
    def with_sf(self) -> bool:
        return self.torch_dtype in [torch.float8_e4m3fn, torch.int8, torch.uint32]
```

`sf_block` 就是粒度；`with_sf` 就是把“要不要缩放因子”这件事折叠成三个布尔开关，
供 JIT 做特化（§2.2 的第 2 条职责）。目标格式只有两个：`e4m3` 与 `e2m1`
（`common.py`、`config.py:34-36`）。fp4 用 `torch.int8` 承载**两个一包**，
所以有一个专门的换算：

```python
# .../TileKernels/tile_kernels/quant/common.py:174-185
def get_logical_hidden(hidden, dtype):
    """Compute hidden size when `torch.int8` is used for packing FP4"""
    return hidden if dtype != torch.int8 else hidden * 2

def get_physical_hidden(hidden, dtype):
    return hidden if dtype != torch.int8 else hidden // 2
```

**逻辑宽度与物理宽度是两个概念**：fp4 张量在 PyTorch 里只有 `hidden/2` 个 int8，
但算法上要按 `hidden` 个元素算。所有 cast kernel 的第一个动作都是
`hidden = get_logical_hidden(hidden, x.dtype)`（例如 `per_token_cast_kernel.py:31`）。
第 27 章讲的“量化产物的 layout”在这里有了具体形状。

还有一处显式的**后端差异**，而且它会造成两种张量布局：

```python
# .../TileKernels/tile_kernels/quant/common.py:70-75
def get_packed_ue8m0_pack_factor() -> int:
    return 2 if is_ascend() else 4

def get_packed_ue8m0_torch_dtype() -> torch.dtype:
    return torch.int16 if is_ascend() else torch.int32
```

“UE8M0”是把尺度因子压成 8 位指数（只存 `2^e` 的 `e`）。
打包时 CUDA 一次塞 4 个（int32），昇腾一次塞 2 个（int16）。
两个数都是 32 位字长，但**同一份权重在两种设备上的 sf 张量形状不同**，
所以“把 CUDA 上量化好的权重拿到昇腾上跑”需要重新打包。这是真实部署里必须知道的一条。

### 5.2 换算口径：`get_sf_and_inv`

所有 cast kernel 的数值核心是同一个宏（`:251-282`）：

```python
# .../TileKernels/tile_kernels/quant/common.py:251-282（节选）
@T.macro
def get_sf_and_inv(amax, out_config, sf_inv_dtype=T.float32):
    clamped_amax = T.max(amax, out_config.clamp_min_value)   # 防 0 → 除零
    max_value = T.max_value(out_config.dtype)                # e4m3:448 / e2m1:6
    sf = T.alloc_var(T.float32)
    if out_config.round_sf:
        # round_sf only uses ceil(log2(sf)) ...
        sf = clamped_amax * (T.float32(1) / max_value)
    else:
        sf = clamped_amax / max_value
        if out_config.use_e4m3_sf:
            sf_fp8 = T.cast(sf, T.float8_e4m3fn)
            return sf_fp8, 1 / T.cast(sf_fp8, T.float32)
        else:
            return sf, max_value / clamped_amax

    # Round into 2's power
    bits = T.reinterpret(sf, T.uint32)
    # clamp_min_value ensures sf is positive and normal before exponent extraction.
    # `(bits - 1) >> 23 + 1` gives ceil(log2).
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    ...
```

四件事值得说：

1. **`clamp_min_value` 是必需的，不是保险。** 全零行会让 `amax = 0`，
   于是 `sf = 0`、`sf_inv = ∞`，输出整行变成 `NaN`。库给的默认下限写在
   `config.py:10-15`：`e4m3` 非幂次舍入时是 `1e-4`，其他情况按目标格式的
   最大可表示值再乘一个小的比例。`clamp_min_value` 还承担第二个职责，
   保证 `sf` 是**正规格数**，否则下面按位取指数的技巧就不成立
   （注释原话：“ensures sf is positive and normal before exponent extraction”）。
2. **`round_sf` 是“把尺度取成 2 的幂”**，用位运算实现 `ceil(log2(...))`。
   这么做的动机是：`x / sf` 与 `x * (1/sf)` 在 2 的幂下**结果完全一致**，
   于是可以把除法换成乘倒数，且不会引入额外误差（注释：
   “for supported formats this reciprocal multiply preserves the same power-of-two bin
   as division”）。这是 MX 格式能在硬件上做快的原因之一。
3. **`use_packed_ue8m0` 时返回的不是浮点数，而是 `uint8(exp_sf + 127)`**，
   也就是直接吐指数的偏置表示，后面的 kernel 会把它按位打包。
4. **`use_e4m3_sf` 只在 CUDA 上可用**（`common.py:118`、`cast_back_kernel.py:41`、
   `per_token_cast_kernel.py:33` 各有一处断言）。昇腾侧不支持“用 fp8 存尺度”这条路。

> 这一节的全部数值细节，第 27 章已经完整推导过；本节只补一件事：
> **这些口径被写成了两个后端共用的一个宏**。两个 kernel 调用同一个
> `get_sf_and_inv`，所以“舍入方式、clamp、幂次化”在 CUDA 与昇腾上**不可能不一致**。
> 这是三层结构在数值正确性上的价值：把“容易写错的算术”放进共享层。

### 5.3 三条 cast 路径在两边怎么分派

三个 `*_kernel.py` 的分派逻辑几乎是同一个模板。以 `per_block_cast` 为例：

```python
# .../TileKernels/tile_kernels/quant/per_block_cast_kernel.py:42-61（节选）
if is_ascend():
    from tile_kernels.quant.per_block_cast_asc import get_per_block_cast_kernel_asc

    kernel = get_per_block_cast_kernel_asc(
        hidden=hidden, in_config=in_config, out_config=out_config,
        sf_only=sf_only, cast_only=(sf is not None),
        num_vec_cores=get_num_vec_cores(),
    )
else:
    kernel = get_per_block_cast_kernel_cuda(
        hidden=hidden, in_config=in_config, out_config=out_config,
        sf_only=sf_only, cast_only=(sf is not None),
        use_pdl=get_pdl(),
    )
```

对比 §3.2 的 `topk_gate`：**结构完全一样**（延迟 import / 展开布尔开关 /
核数 vs PDL 二选一）。三处细节值得注意：

* `in_config` / `out_config` 是 dataclass 实例，直接当编译期常量传进 JIT 工厂。
  这要求它们是**不可变且可哈希**的。所以 `BaseCastConfig` 被声明成
  `@dataclass(frozen=True)`（`common.py:19`）。**“让配置能进缓存键”是一个设计约束，
  不是随手的选择。**
* `cast_only=(sf is not None)` 把“调用方已经给了尺度因子、我只需要搬数据”
  这个运行期分支折叠成一个编译期布尔。于是同一个算法生成两个特化版本，
  而不是在每个元素上分支。
* `sf_only` 同理：只要尺度不要数据（MoE 的 aux loss 之类场景会这么用）。

同一层的**形状约束**在三个算子上并不相同，而且**昇腾侧更紧**：

| 约束 | per-token | per-block | per-channel |
| --- | --- | --- | --- |
| CUDA/共享层 | `num_per_channels ∈ {16,32,64,128}`、`hidden % 64 == 0`（`per_token_cast_kernel.py:32`） | `num_tokens % 128 == 0`、`hidden % 128 == 0`（`per_block_cast_kernel.py:33-34`）、输出需 16 字节对齐（`:67`） | `num_per_tokens ∈ {32,128}`、`num_tokens % 128 == 0`、`hidden % 128 == 0`（`per_channel_cast_kernel.py:41-44`） |
| 昇腾侧额外 | — | **必须是 `(32, 32)`**（`per_block_cast_asc.py:22`），且 `block_k` 取 `align(gcd(512或1024, hidden), 128)`（`:26-27`） | **`num_per_tokens` 只能是 32**（`per_channel_cast_asc.py:21`）；输入若带尺度，`num_per_channels` 也只能是 32（`:23-24`） |

这些数字不是“性能偏好”，而是**实现里写死的常量**。所以“同一份 Python API 跑在两个
后端上”有一个隐含前提：**调用方要落在两边都支持的参数子集里**。库用断言把这件事
变成编译期错误，而不是运行期错答案。这是正确做法，但读者必须知道它存在。

### 5.4 cast_back：反量化与“不许越界的组合”

`cast_back_kernel.py:26-58` 是“同一个套路的反向版”：输入 `(data, sf)`，输出 bf16/fp32。
它多出三条**组合约束**，都是因为反量化必须重建原尺度：

```python
# .../TileKernels/tile_kernels/quant/cast_back_kernel.py:41-44
assert not (in_config.use_e4m3_sf and is_ascend()), 'E4M3 SF is only supported on CUDA'
assert not (x_block_size[1] == 1 and in_config.use_tma_aligned_col_major_sf), (
    'cast_back does not support col-major scaling factors when num_per_channels == 1'
)
```

第二条很有意思：当缩放块只有 1 列宽时，列主序布局下“同一个块”的字节会被拆散，
重建尺度需要跨行 gather，代价高。**这不是硬件限制，是这套实现选择不做。**
库里宁可断言失败，也不生成一个慢十倍的通用路径。这是“库”与“框架”的分工差别。

顺带一个真实的**不一致**：`cast_back_kernel.py:46` 用的是
`if x.device.type == 'npu':`，而不是其他地方统一的 `is_ascend()`
（对比 `cast_kernel.py:40`、`swiglu_forward_kernel.py:48`、`rand/randn_kernel.py:31`）。
两者在正常使用下等价，但**判据来源不同**（一个看张量、一个看机器）。
写自己的库时应挑一种并保持一致；上游这处差异我们照实标注，不做“统一化”的推测。

### 5.5 为什么要把 SwiGLU 和量化融在一起

SwiGLU 前向的定义很短：

```text
输入 x: [num_expanded_tokens, 2*hidden]      # 左右两半拼在一起
gate = x[:, :hidden]；up = x[:, hidden:]
out  = swish(gate) * up      -> [num_expanded_tokens, hidden]
```

库里的实现直接做了这件事（`quant/swiglu_forward_kernel.py:24-27`，
`hidden = hidden // 2`），并且在**同一个 kernel 里**把结果量化：

```python
# .../TileKernels/tile_kernels/quant/swiglu_forward_kernel.py:40-45（节选）
# Swiglu forward : hidden -> hidden // 2
assert (fmt, num_per_channels) in [('e4m3', 128), ('e4m3', 32), ('bf16', None), ('fp32', None)]
out_config = get_cast_output_config(fmt, (1, num_per_channels if ... else 1), ...)
```

融合的动机是一条纯粹的访存账。分开做的话：

```text
方案 A（两个 kernel）：
  读 x (2H)  →  写 swiglu 结果 (H)  →  读 (H)  →  写 fp8 结果 (H/2) + sf
  总访存 ≈ 2H + H + H + H/2 = 4.5H（按 fp16 计）

方案 B（一个 kernel）：
  读 x (2H)  →  写 fp8 结果 (H/2) + sf
  总访存 ≈ 2.5H
```

（这是**本书的记账**，忽略 sf 与缓存命中，只用来展示量级；它**不是**任何机器上的测量。）
中间那趟 `H + H` 的往返被整个消掉了。**SwiGLU 是 memory-bound 算子，所以省下的
就是实打实的时间**。这正是第 13 章“融合的意义”在真实库里的样子。

昇腾版还多带了一个 Tailored 的能力：它把 `topk_weights` 和
`routed_scaling_factor` 也一起吃进来（`swiglu_forward_kernel.py:13-14` 的参数、
`:84` 的调用），也就是说**“MoE 路由权重 × SwiGLU × 量化”三段合成一段**。
于是 `swiglu_forward_asc.py:36-56` 里出现了三组 UB 缓冲
（`xl_ub/xr_ub/out_ub`，外加 `values_ub`、`amax_ub`、`inverse_ub`、`sf_dense_ub`
以及 `use_fast_certificate` 分支下的第二套 `_fb_ub`），
**融合的代价是片上存储压力**，而这条压力的边界就是 UB 容量。

### 5.6 这条线接到第 27 章

把本节压缩成三句可以直接对到第 27 章的话：

1. **口径共享**：`get_sf_and_inv` 是两个后端唯一的算术权威，
   所以“clamp / 幂次化 / 打包”没有第二种解释。
2. **精度由格式决定**：fp8 只有 `e4m3`、fp4 只有 `e2m1`，
   累加一律 fp32（`common.py` 里 `sf_value_dtype = T.float32` 一类写法），
   与第 27 章的结论一致。
3. **布局是后端相关的**：UE8M0 的打包因子在 CUDA 是 4、昇腾是 2，
   列主序 sf 在昇腾不需要 16 字节对齐而 CUDA 需要（`common.py:215-217` 的注释：
   “CUDA: col-major SF must be aligned to 16 bytes for TMA load; Ascend needs no
   alignment”）。**权重跨设备搬运时必须重打包**。

### 5.7 常见坑（量化层）

| 编号 | 坑 | 为什么危险 |
| --- | --- | --- |
| **Q1** | 忘了 `get_logical_hidden` | fp4 的 `hidden` 差一倍，shape 断言或结果错 |
| **Q2** | 把 CUDA 上打包的 UE8M0 sf 直接喂给昇腾 | 打包因子 4 vs 2，形状与取值都错（`common.py:70-75`） |
| **Q3** | 在昇腾上开 `use_e4m3_sf` | 断言失败（`common.py:118`） |
| **Q4** | 在昇腾上给 `per_block_cast` 传 `(128,128)` | 只支持 `(32,32)`（`per_block_cast_asc.py:22`） |
| **Q5** | 想当然地认为“断言里的形状限制是性能建议” | 它们是实现常量，越界直接报错 |
| **Q6** | 全零行没有 clamp | `sf=0`、`sf_inv=inf`、输出 NaN（`common.py:252-254`） |

---

## 6. RoPE 与 Rand：双后端不只是大算子的事

### 6.1 RoPE：同一个旋转，两种向量映射

RoPE（旋转位置编码）的算法只有几行：把 head 维度按“前半 / 后半”配对
（NeoX 口径）或“奇偶配对”（GPT-J 口径），每组做一次二维旋转：

```text
out_first  = first * cos + second * (-sin)   （或共轭时取反）
out_second = first * sin + second *  cos
```

接口层（`transform/rope_kernel.py:9-38`）把 3D `[tokens, heads, dim]` 与 4D
`[batch, seqlen, heads, dim]` 统一成 4D，再做后端分派（`:80-115`）。约束是：
`rot_dim ∈ {64, 128}`、dtype 是 bf16/fp32、原地修改、`cos_sin_cache` 必须是 fp32
连续张量。同一份分派有两个后端工厂：`get_rope_kernel_cuda` / `get_rope_kernel_asc`。

两个实现的差异非常集中，可以直接对上 §4.5 的并行模型：

| | CUDA `rope_cuda.py` | Ascend `rope_asc.py` |
| --- | --- | --- |
| 网格 | `T.Kernel(ceildiv(seqlen, tokens_per_cta), batch, threads=threads)`（`:59`） | `T.Kernel(num_vec_cores)`（`:120`） |
| 并行宽度怎么定 | 由 `nheads` 与 `half_dim` 反推 `threads`（`:40-45`） | 写死 `SIMD_LANES = 64`（`:9`） |
| 一行放哪 | `T.alloc_local`（寄存器，`:62-65`） | `T.alloc_shared`（UB，`:121-122`） |
| 缓冲区版本 | 不需要 | `T.annotate_buffer_versions({x_ub: num_stages, cos_sin_ub: num_stages})`（`:123`） |
| 与容量的关系 | 只受寄存器数量约束 | 从 UB 容量**反推** tile 大小（`:31-36`） |
| 两半怎么配对 | 两次带偏移的索引（`:98-100`） | `S.vintlv` / `S.vdintlv` 把两半**交错进一个向量**（`:168,176`） |

最后一行是昇腾实现里最“机器特有”的一步，值得展开。**NeoX 口径要求把
`x[..., :half]` 与 `x[..., half:]` 配对乘加**；在 CUDA 上这就是两次带不同偏移的
读（`rope_cuda.py:98-100`），硬件各发一次访存、每个线程各算一个标量，
“谁和谁配对”不关硬件的事。而在 SIMD 上，一次乘加只能作用于**同一个 vreg 的
64 个 lane**：要参与同一次 FMA 的两个元素，必须落在两个 vreg 的**同一个 lane 位置**上。
于是“谁和谁配对”从一个索引问题变成了一次 **lane 重排**：

* 打包的 bf16 加载（`load2(..., packed_bf16=True)`）把 128 条 bf16 lane 按
  **偶数 / 奇数**拆成两个 vreg，`vcvt(..., part=0/1)` 就是 `PART_EVEN` / `PART_ODD`
  （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py:1247-1249`）。
  但 NeoX 要的是“前半 / 后半”，所以先 `S.vintlv` 重排成两半、旋转之后再
  `S.vdintlv` 拆回偶奇交给打包写回（`rope_asc.py:182-192`，`rotary_dim = 128` 分支）。
* GPT-J（`interleaved=True`）的搭档本来就是相邻的（`2i, 2i+1`），
  所以打包路径**不需要**这一步；反过来，fp32 路径下 `load2` 天然给出前后两半
  （`rope_asc.py:79-81`），于是需要重排的是 GPT-J 而不是 NeoX。

**注意这与旋转公式本身无关**：公式一行没变，变的是操作数怎么排进 lane。
这正是“同一份算法、两种并行映射”最纯粹的一个例子。

另外两处细节说明昇腾实现必须**自己算容量**：

```python
# .../TileKernels/tile_kernels/transform/rope_asc.py:30-37
num_vec_cores = get_num_vec_cores()
dtype_size = dtype.bytes
bytes_per_row = nheads * rotary_dim * dtype_size + rotary_dim * 4
max_ub_size = get_max_ub_per_vector_core(use_simt=False)
num_stages = min(3, max_ub_size // bytes_per_row)
ub_block_tokens = min(SIMD_LANES, max_ub_size // (num_stages * bytes_per_row))
block_tokens = 1 << (ub_block_tokens.bit_length() - 1)
can_bulk_copy = x_stride_0 == nheads * x_stride_1 or x_stride_1 == rotary_dim
```

“一行要占多少字节”→“UB 能装几行”→“取 2 的幂”→“能不能整块搬”，
这条链条全部由写 kernel 的人维护。CUDA 版完全没有这一段：
register 不够编译器会 spill、会报错，不需要你预先算。

正确性口径见 `tests/transform/test_rope.py:36-46`：调用外层
`tile_kernels.transform.apply_rotary(...)`，与
`tile_kernels/torch/rope.py` 的 `rotary_embedding_ref` 用 `calc_diff <= 1e-8` 比对
（`testing/numeric.py:33-38` 的相对误差）。测试覆盖 3D/4D、两种配对（`interleaved`）、
共轭、`positions` 的三种模式（None / int32 / int64）、以及尾切
（`query[..., -dim:]`），**同一份测试，两个后端**，还是那句话。

### 6.2 Rand：一个诚实的“只有一个后端”

`tile_kernels/rand/` 是本章唯一没有三件套的目录：

```
tile_kernels/rand/
├── randn_kernel.py   ← 外层：统一 API
├── randn_asc.py      ← 只有 Ascend 实现
└── seed.py           ← host 侧的种子管理
```

它甚至在第 1 行就写明了动机：

```python
# .../TileKernels/tile_kernels/rand/randn_kernel.py:1
# Temporary replacement for the extremely slow torch_npu.randn.
```

外层的行为是“在 CUDA 上直接让路”：

```python
# .../TileKernels/tile_kernels/rand/randn_kernel.py:29-40（节选）
shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
device = device or get_device()
if not is_ascend():
    return torch.randn(shape, dtype=dtype, device=device)     # CUDA：交给 PyTorch

assert torch.device(device).type == 'npu', 'Ascend randn only supports NPU tensors'
assert dtype in (torch.bfloat16, torch.float32), ...
# reserve_seed runs on the host, so graph replay would reuse the captured seed.
assert not torch.npu.is_current_stream_capturing(), (
    'Ascend randn does not support NPU graph capture now: graph replay would reuse the same random seed'
)
```

昇腾实现本身很短（`randn_asc.py:7-42`），却是本章**唯一一个把全部四个昇腾要点同时
出现**的例子：`T.Kernel(num_vec_cores)`、`T.annotate_buffer_versions`、
`T.SimtVF(threads=512)`、`T.Persistent(..., num_stages=...)`：

```python
# .../TileKernels/tile_kernels/rand/randn_asc.py:20-40（节选）
with T.Kernel(num_vec_cores) as core_id:
    output_ub = T.alloc_shared((T.int64(B),), dtype)
    T.annotate_buffer_versions({output_ub: num_stages})

    for pid in T.Persistent([num_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
        offset = T.int64(pid) * B
        with T.SimtVF(threads=threads):
            T.rng_init(seed + pid)
            for serial in T.unroll(B // tile_size, unroll_factor=unroll_factor):
                for tx in T.Parallel(tile_size):
                    output_ub[serial * tile_size + tx] = T.rng_rand_float(dist='normal')
        T.copy(output_ub, output[offset])
```

这一段回答了“昇腾上还能不能用 SIMT”：**能，但只能在 `T.SimtVF` 这个子作用域里**，
而生成随机数这种“每个元素独立、不需要向量归约”的工作正是它的用武之地。
`T.rng_init` / `T.rng_rand_float` 在 CUDA 与昇腾两个方言里都存在
（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/random.py:4-70` 与
`.../tilelang/ascend/language/random.py:15-38`），差别只在默认序列号
（`ascend/language/__init__.py:71-75` 的注释说明了原因）。

还有一条**必须诚实指出**的限制，就写在上面的断言里：**昇腾 randn 不能进 NPU 图**。
原因是 `reserve_seed` 在 host 上执行（`rand/seed.py:13-20`），图捕获时会把这个值
固化下来，replay 得到的随机数就完全一样了。库选择**显式失败**而不是悄悄给出错误结果，
这是库的正确做法，也提醒我们：**“同一份 API”不等于“所有使用方式都同构”**。
第 19 章讲过 ACLGraph 的收益与边界，这里是它在真实库里的一个反例。

---

## 7. 方法论落点：哪些是共享的，哪些必须重写

### 7.1 一份清单

把本章读到的所有东西归一次类。

**平台无关（写一次，两个后端共用）**：

| 内容 | 证据 |
| --- | --- |
| 函数与参数签名、返回值契约 | `topk_gate_kernel.py:7-25`（docstring 里明确写了 tie-break 规则） |
| 输入校验与形状上限 | `topk_gate_kernel.py:26-29` |
| 输出张量分配与 dtype | `topk_gate_kernel.py:31` |
| 空输入短路 | `topk_gate_kernel.py:32-33` |
| `sf_block` 等配置语义、尺度换算算术 | `quant/common.py:19-33,251-282` |
| 逻辑宽度 vs 物理宽度 | `quant/common.py:174-185` |
| 缩放因子形状与分配 | `quant/common.py:188-233` |
| 数学近似的**接口**（不是实现） | `moe/sqrt_softplus.py`：同名函数、两个后端各一份实现 |
| 参考实现与对拍工具 | `tile_kernels/torch/**`、`tile_kernels/testing/**` |
| 测试用例本身 | `tests/**`：只调用外层 API |

**必须重写（每个后端一份）**：

| 内容 | topk_gate 里的证据 |
| --- | --- |
| 并行映射（网格/线程/核） | `T.Kernel(num_tokens, threads=32)` vs `T.Kernel(num_vec_cores)` |
| “一块数据放哪一级片上存储” | `alloc_fragment` vs `alloc_shared`(UB) |
| 归约怎么写 | `T.alloc_reducer(op=...)` vs `S.vcmax`/`S.vcmin` |
| 同步 | 隐式/PDL vs `S.mem_bar` + 自动 flag |
| 缓冲版本数 | 编译器决定 vs `T.annotate_buffer_versions` |
| 向量宽度 | 编译器选 vs `vector_length = 64` |
| 越界处理 | 补 `-inf` vs 掩码 |
| 分块/分组策略 | 无 vs 快路径/慢路径 |
| 容量反推 tile 大小 | 不做 vs `rope_asc.py:31-36` |

一句话总结这张对照：

> **共享的是“算法与契约”，重写的是“数据怎么在芯片上流动”。**

上半张表里没有一行提到 CUDA 或昇腾；下半张表里也没有一行提到这个算子叫什么。
如果你在写自己的库，这两张表就是拆文件的标准。

### 7.2 “重写”的量比手写 CUDA / Ascend C 少在哪

先说一个**反直觉的事实**：TileKernels 的 Ascend 实现**并不比 CUDA 实现短**
（`topk_gate` 68 行 vs 157 行），而且它里面用的 `S.*` 原语，语义上已经非常接近
Ascend C 了（`vld`/`vsts`/`vcmax` 这些名字就是硬件指令）。所以“少写”的**不是
机器级代码本身**。少写的是这四类东西：

1. **Host 侧样板与类型胶水**。没有 `cudaMalloc` / stream / kernel launch 配置，
   也没有 ACL / `aclrtMalloc` / `TPipe` / `TQue` 那套样板。两个后端共享同一份
   `torch.empty` 分配与同一个调用形式（第 26 章讲了 ABI 这一层怎么做的）。
2. **tiling 与容量记账**。TileLang 的 layout inference 替你做 L0/L1 分块与
   swizzle（第 21 章），你只在**真的会影响 UB 容量**的时候才手算
   （`rope_asc.py:31-36`、`swiglu_forward_asc.py` 的多套 buffer）。
   手写 Ascend C 时，这套账是每写一个算子都要重做一遍的。
3. **同步的正确性**。M 树默认自动插入 flag（第 22 章），
   `topk_gate_asc.py` 全文件 157 行里只有**一句**手写同步（`:151` 的 `mem_bar`）。
   手写 Ascend C 时，`set_flag`/`wait_flag` 的配对是主要 bug 来源
   （第 19 章 §2.4、§11）。
4. **多后端并存的组织成本**。加一个后端 = 加一个文件 + 外层加一个分支；
   算法、契约、测试、参考实现全部复用。这是“一份算法、两个后端”真正的收益所在，
   **不是省了打字，而是省了“第二份实现悄悄跑偏”的风险**。

### 7.3 边界在哪：什么时候 TileLang 不够用

第 33 章必须回答“为什么某些部分没有用 TileLang”。在 TileKernels 里，
答案不是“某些算子用了别的语言”，而是**在四个具体位置往下沉了一层**：

| 下沉点 | 位置 | 下沉到哪 | 为什么 |
| --- | --- | --- | --- |
| 内联设备函数与 PTX | `quant/swiglu_forward_cuda.py:11-16`（`prefetch.global.L2` 的 inline PTX），在 `:91` 用 `T.import_source(...)` 注入 | CUDA C++ / PTX | `__device__` 级别的指令级控制（L2 预取提示）没有对应的 TileLang 原语 |
| SIMD 内在函数 | Ascend 侧所有 `S.*`（`tilelang.ascend.language.simd`，v0.1.15 里 `simd.py` 有 2000+ 行） | 昇腾向量指令 | 想做 lane 级控制（掩码、单 lane 写、跨 lane 归约）就必须下沉 |
| 数学近似 | `moe/sqrt_softplus.py:9-20` 的多项式系数；`quant/rstd.py` 的补偿 rsqrt | 手写数值方法 | 精度目标决定了实现，与芯片无关，但两个后端需要不同的指令序列（`sqrt_softplus.py:23-25` 的 CUDA 版与 `:29+` 的 Ascend 多项式版），所以要分开写 |
| 矩阵乘 | `T.gemm` / `T.gemm_blockscaled`（M 树） | 由后端 codegen 落到 Cube / MMA / WGMMA | 这里**不需要**下沉，TileLang 已经把这条路径做好了，这是它相对手写 Ascend C 的主要优势（第 14 / 24 / 25 章） |

注意第一行和第三行的性质完全不同。第一行是**“TileLang 没暴露的能力”**；
第三行是**“TileLang 暴露了，但同一份数学在不同芯片上要写成不同序列”**。
分清这两者，才不会误判“TileLang 能不能干这件事”。

`S.*` 的位置（第二行）最容易被误解。它不是“TileLang 失败了”，
`tilelang.ascend.language.simd` 是**官方方言的一部分**，在锁定版本里是一等公民。
真实的情况是：**TileLang 在 Ascend 侧选择了一条“该下沉就下沉、但保持同一套 IR 与调度”
的路线**（第 25 章会讲它背后的两种 codegen 路线）。所以 `topk_gate_asc.py` 里
`T.Pipelined` + `S.vmax` 混着写，是完全符合设计意图的用法。

### 7.4 与第 32、34 章的边界

* 第 32 章给出这次开源的清单与硬件/版本边界（包括 Ascend 950 + CANN ≥ 9.2.0）。
  本章不重复。
* 第 34 章接着讲 TileKernels 的另一半：Engram 与流形超连接（mHC）。
  那批算子的特点是**“不是矩阵乘”**，条件记忆查表、Sinkhorn 归一化、
  多层重计算。本章的两条经验（三层结构、平台无关/平台相关的清单）会在那里再次生效。
* 第 35–38 章讲其余组件（DeepGEMM-Ascend / FlashMLA / DeepSelect / DeepEP-Ascend），
  它们的 TileLang 含量远低于本章，正好用来对照“TileLang 主线”的边界。

---

## 8. 本章小结

### 8.1 一页复习

* **三层结构**：`*_kernel.py`（校验 + 分配 + 分派）/ `*_cuda.py` / `*_asc.py`，
  外加 `torch/`（参考实现）与 `testing/`（对拍工具）。
  外层的每一行回答“要不要算、算什么形状、用哪块芯片”；内层的每一行回答
  “这块芯片上怎么算”。
* **后端选择**：`config.py:24-26` 用 `/dev/davinci_manager` 是否存在来判定，
  `lru_cache` 缓存到进程级；核数在工厂调用时求值（`get_num_vec_cores() = sms × 2`），
  于是成为编译期常量并进入 JIT 缓存键；PDL 是 CUDA-only，
  所以它只出现在 CUDA 分支的参数里。
* **top-k gate 对照**：CUDA 是“1 block / 行 + 32 线程 + `T.alloc_reducer`”，
  Ascend 是“1 Vector 核 / 行 + 64 lane + `S.vcmax`/`S.vcmin` + 显式缓冲版本”。
  算法与 tie-break 契约完全相同，差异全部是并行映射。
* **为什么并行单位不同**：CUDA 的“线程”是真实的执行单位（SIMT），
  Ascend 在核这一层没有线程域（SIMD，64 lane/指令），
  所以在昇腾上“宽度”是语法的一部分（`vector_length = 64`、`T.int32x64`）。
* **为什么 Ascend 要显式管缓冲版本**：因为 `T.Pipelined` 引入了
  “多个迭代同时在飞”的状态，份数是**性能意图**，调度器只能保证下限。
* **量化算子族**：三种粒度（per-token / per-block / per-channel）由
  `sf_block` 表达；`get_sf_and_inv` 是两个后端共用的算术权威；
  UE8M0 打包因子 CUDA 是 4、昇腾是 2，**跨设备要重打包**。
* **RoPE / Rand**：RoPE 展示了“配对”从索引问题变成向量重排问题
  （`S.vintlv`/`S.vdintlv`）；Rand 展示了“同一份 API”不等于“所有用法同构”
  （昇腾 randn 不能进图，库选择显式报错）。
* **方法论**：共享算法与契约，重写数据流动。TileLang 省掉的是 host 样板、
  tiling 记账、同步插入与多后端组织成本，**不是**机器级代码本身，
  必要时它会主动下沉（inline PTX 预取、SIMD 内在函数）。

### 8.2 全部常见坑（汇总）

| 层 | 编号 | 一句话 |
| --- | --- | --- |
| 外层 | P1–P6 | 后端判据、延迟 import、编译期常量、核数访问器、参数表差异、CUDA-only 能力（§3.5） |
| kernel | K1–K8 | 断言折叠、`T.clear`、缓冲版本语义、`-inf` vs 掩码、tie-break、`mem_bar`（§4.8） |
| 量化 | Q1–Q6 | 逻辑宽度、跨设备打包、`use_e4m3_sf`、形状常量、clamp（§5.7） |

### 8.3 本章没有验证的东西（务必记住）

以下全部需要真机才能确认，本章**只做了静态阅读**：

1. **任何性能结论**。两个后端的快慢、`num_stages = 6` 是否最优、
   `vector_length = 64` 与 `max_vregs_per_group = 6` 的选择，一概未验证。
2. **数值正确性**。“tie-break 完全一致”“int32→int64 转换精确”
   “`x-x != 0` 能检出全部非有限值”都是从源码推出的**论证**，不是运行结果。
3. **`T.view` 在昇腾上的实际可用性**（本章只核对了它的契约与调用方式）。
4. **两个后端的产物是否真的语义等价**。这需要一台 CUDA 机器和一台昇腾机器，
   跑 `tests/**` 的同一组用例，比对 `tile_kernels/torch/**` 的参考实现。
5. `get_max_ub_per_vector_core()` 的 216/248 KB 是库自述的预算口径，
   我们没有在官方文档里找到完全一致的对应条目。

反过来，本章**从源码结构直接读出**的结论（不依赖运行）是可靠的：
文件分层、分派逻辑、API 契约、形状约束、缓冲版本号、
以及“哪些行是逐字共享、哪些行是逐后端重写”。

## 延伸阅读

* 本章的引用清单、核验方式与版本锁定：见 [`references.md`](references.md)。
* 习题与参考答案：见 [`exercises.md`](exercises.md) 与 [`solutions.md`](solutions.md)。
* 图的源描述：见 [`figures/`](figures/)（含 `README.md` 的来源表）。
* 代码示例：见 [`code/`](code/)（每个示例都有 README 说明运行前提与验证级别）。
* 相关章节：第 07 章（JIT 与缓存）、第 11 章（流水线）、第 19 章（昇腾实战与两棵树）、
  第 25 章（昇腾后端 codegen）、第 26 章（运行时与 ABI）、第 27 章（量化数值口径）、
  第 32 章（这次开源的清单）、第 34 章（TileKernels 下半：Engram 与 mHC）。
