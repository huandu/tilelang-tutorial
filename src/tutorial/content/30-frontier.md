# 第 30 章 前沿探索：只有 main / 未合入的方向

## 学习目标

读完本章，你应当能够：

1. 分清**三种状态**，“v0.1.15 已具备”“仅 main”“仅未合入 PR/RFC”，并说出
   第四种诚实结论“无法确认”在什么情况下必须使用。
2. 用**四步判定流程**独立核对任何一个能力属于哪一档，并写出可复现的验证命令，
   而不是引用一句“据说支持了”。
3. 说清三个**架构级变化**（多后端解耦 issue #2115、TVM → apache/tvm + tvm-ffi
   迁移 issue #2196、CuteDSL 后端 issue #1454）各自的动机、影响面与当前状态，
   并指出其中**哪一条实际上早已随 release 发布**。
4. 说清昇腾 950 的**两条后端路线**：PR #3308 与 PR #3310 分别把什么带进了哪棵树，
   以及为什么“`target="pto"` 在 v0.1.15 里会编译失败”。
5. 解释 RFC #2897（reducer）为什么是一个**最容易被误判**的条目：议题还开着，
   但它的主体语义已经进了 v0.1.14 与 v0.1.15；同时说清 RFC #2708（atomic 内存序）
   与 RFC #3178（cache policy）为什么**确实**还停留在 RFC 阶段。
6. 用 `git` / `gh` 给自己搭一套上游跟踪流程，并说出本地 `_vendor/` 快照
   （`--depth 1`）在比较版本时的具体限制。

## 前置知识

* 第 20 章“IR 与编译期管线精读”与第 29 章“编译器工程”：本章反复提到
  “pass 挂在 pipeline 的哪一步”，这需要你已经能读 `tilelang/<chip>/pipeline.py`。
* 第 24 章“NVIDIA 后端”§4.8 与第 25 章“昇腾后端”§4.8–§4.10：CuTe DSL 后端、
  AscendNPU IR 路线、PTO 路线在那里已经讲过；本章只讲它们的**状态与走向**，
  不重复机制。
* 第 4 章“架构总览”：`tilelang/backend/` 这套注册表结构在那里首次出现。
* 会读 Python 与 C++17，会用 `rg` 与 `git`。**不需要 GPU，也不需要 NPU**：
  本章所有判断都来自源码与资料快照，不来自运行结果。
* 手边有 [VERSIONS.md](../../VERSIONS.md) 这一页，最好再打开
  `_vendor/tilelang-v0.1.15/` 与 `../tilelang-corpus/`。

本章回答一个问题：**当你在网上看到一个 v0.1.15 教程里没写的能力，
怎么在十分钟内判断出它能不能用、什么时候能用。**

---

## 1. 问题的由来：release 之外还有两个世界

### 1.1 为什么教材必须冻结版本

[VERSIONS.md](../../VERSIONS.md) 把全书钉在 `TileLang v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，2026-09-30）与
`TileLang-Ascend v0.1.2.000-release`（commit `89a9364e…`，2026-09-08）上。
理由写得很直白：`main` 每天都在动，如果每章按自己的日期读 `main`，
不同章节会得出互相矛盾的结论，读者也没法复核。

但冻结的代价是：**教材描述的世界与社区正在讨论的世界，中间隔着一层时差**。
这一层时差里有两个世界：

```text
  v0.1.15（你可以依赖）  →  main（下一个 release 会带上）  →  PR / RFC（方向，不是能力）
```

第 1 章讲“为什么需要 tile DSL”时用的是 2026 年的资料；第 15 章写 attention
时用 `examples/` 里的 FlashAttention；而你打开 GitHub，看到的是
`T.PersistentKernel`、`AutoSimtVF`、Ascend 算子生成 agent。**这些都不在
v0.1.15 里**。本章就是那层“翻译”。

### 1.2 三个真实会踩到的场景

**场景 A：抄了 PR 里的写法，编译不过。** 你在 PR #3325 的正文里看到
`T.PersistentKernel(OUT_TILES, num_cores=28)`，写进自己的 kernel，报
`AttributeError`。原因不是你的错：这个 API 只在那个 PR 的分支上存在。

**场景 B：读到一篇说“reducer 会算错”的文章，于是不敢用。** 文章引的是
issue #2897 里那个最小错码例子，`sum([1..8])` 返回 `576` 而不是 `36`。
但那个例子用的是 legacy 写法 `alloc_reducer(..., replication="all")`；
v0.1.15 里的 v2 API 已经把这个语义问题解决了（见 §5.1）。**照文章的说法弃用
reducer，是过度反应；照文章的说法直接写 legacy 写法，则确实有风险。**

**场景 C：昇腾 950 上有两个后端，不知道选哪个。** issue #3317 正是这个提问，
标题就是“It seems there are two Ascend 950 backends, which is suggested to use”。
它被关闭了，但 corpus 抓到的页面里**没有维护者回答**（见 §4.1）。

三个场景的共同点：**问题的答案不在“谁说得对”，而在“代码里到底有什么”**。
本章不教你任何新功能，只教你把这层翻译做对。

---

## 2. 概念与模型：三种状态、四步判定

### 2.1 三种状态的定义

| 状态 | 定义 | 你能不能依赖 | 写作规范 |
| --- | --- | --- | --- |
| **v0.1.15 已具备** | 符号在 `_vendor/tilelang-v0.1.15/` 里能搜到 | 能 | 可以直接写进正文与代码 |
| **仅 main** | 不在 v0.1.15，但在 corpus 的 main 快照（commit `994b44e`）里 | 不能 | 必须显式标注，且只能出现在本章 |
| **仅未合入 PR/RFC** | 两棵树里都搜不到，只在 PR/RFC 页面出现 | 不能 | 只能当方向讨论，不能当用法 |
| **无法确认** | 资料未覆盖，或页面状态与代码快照互相矛盾 | 不能 | 必须写“无法确认”并说明原因 |

第四档不是兜底话术。本章 §4.1 就会遇到一个真实案例：PR #3310 的页面快照显示
它被标记为 draft，而 corpus 的 main 快照里已经有了它的合并 commit。这时唯一
诚实的写法是：**以代码为准陈述事实，并注明两个快照各自的日期**。

### 2.2 证据的强度分级

同一件事，不同来源的可信度完全不同。本章只接受前三档：

| 强度 | 证据 | 说明 |
| --- | --- | --- |
| 强 | 锁定版本里的源码、`ls` 出来的目录、git commit | 可复核，可引用行号 |
| 中 | PR/RFC 正文、commit message、README | 是作者自述，可能滞后于实际代码 |
| 弱 | 二手转述、教程、讨论串 | 只能用来找线索，不能作为结论 |

举一个“中”档证据会骗人的例子：PR #3308 的改动列表里有一大段
`[Ascend] Remove the PTO and VMI content`，把 PTO 相关文件删掉，
commit message 里写着“PTO was a second Ascend codegen backend … Both are dropped
from this branch, leaving the AscendC path as the only Ascend codegen”。
如果只读这段，你会得出“主干放弃了 PTO”。但同一份 PR 序列里，
PR #3310 又把 PTO 作为独立后端加了回来，**PR 正文描述的是那条分支的中间状态，
不是主干的最终状态**。这类判断只能回到代码。

### 2.3 四步判定流程

```bash
# 第 1 步：在锁定 release 里找「符号」，不是找「说法」
rg -n '<symbol>' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src

# 第 2 步：与上一个 release 对照，判断是不是本次新增
rg -n '<symbol>' _vendor/tilelang-v0.1.14/tilelang _vendor/tilelang-v0.1.14/src

# 第 3 步：在 corpus 的 main 快照里找
rg -n '<symbol>' ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang \
                    ../tilelang-corpus/02-tilelang-core-repo/raw/src

# 第 4 步：都没有 → 只可能是未合入的 PR/RFC，注明快照日期
```

流程图见 [figures/03-verify-flow.md](figures/03-verify-flow.md)，
三种状态的直观定义见 [figures/01-three-states.md](figures/01-three-states.md)。

本章把这两步做成了脚本：[`code/01-state-audit/state_audit.py`](code/01-state-audit/state_audit.py)
对 13 条能力逐条判定并自检，[`code/02-tree-diff/tree_diff.py`](code/02-tree-diff/tree_diff.py)
做两棵树的文件级差异。两者都只用标准库，离线可跑。运行后你会看到
`self-check: ok` 与 `selftest: ok`。如果哪次不 ok，说明版本换了，应当停下来重核。

### 2.4 三个反复出现的误判模式

**误判一：把术语表当成功能表。** `08-frontier-unmerged/` 这个 corpus 目录名
描述的是**收集动机**（前沿与未合入话题），不是每一条的落地状态。里面的
RFC #2897 主体已经发布，而 issue #2196 确实一点没动。

**误判二：把 roadmap 当成待办清单。** issue #1454 是“CuteDSL Backend”的
roadmap，列了六项。其中“Initial Commit”“GEMM V2”“Blackwell Support”
早就完成了，roadmap 条目只是**没有回填勾选**。

**误判三：把“issue 很老”当成“早已落地”。** issue #2115 提于 2026-04-28，
它的 PR 清单基本都合入了；issue #2196 提于 2026-05-13，却完全没落地。
编号与日期都不构成证据，只有代码构成证据。

---

## 3. 架构级变化：解耦、迁移、新后端

### 3.1 多后端解耦（issue #2115）：已落地，而且是“早就落地”

**它解决什么问题。** issue 原文的第一句就是动机：
“TileLang currently supports multiple backends (CUDA, ROCm, etc.), but the
codebase has become deeply coupled with specific backends. This tight coupling
makes it difficult to add new backends and maintain existing ones.”
换句话说：不是功能缺失，而是**加一个后端要动共享代码**。

**影响面。** 这条路线重构了“后端”这个概念本身：

| 关注点 | v0.1.15 里的位置 |
| --- | --- |
| 后端注册表 | `tilelang/backend/module.py:238` `register_backend`、`:275` `get_backend`、`:285` `list_backends` |
| target 探测与归一化 | `tilelang/backend/target.py` 的 `register_target_detector` / `register_target_normalizer` |
| device / host codegen 抽象 | `tilelang/backend/device_codegen.py`、`tilelang/backend/host_codegen.py` |
| 每个后端的 pipeline | `tilelang/{cuda,ascend,rocm,metal,cpu,webgpu}/pipeline.py` |
| 语言方言 | `tilelang/cuda/language/`、`tilelang/ascend/language/` |
| 给后端的集成规范 | `.agents/skills/tilelang-backend/SKILL.md` |

**当前状态：v0.1.15 已具备，而且 v0.1.14 里就有。** 这一点很容易被写错，
issue 标题里的“RoadMap”让人以为它还在进行中，但 issue 正文列出的那串 PR
（`#2114`、`#2121`、`#2138`、`#2189`、`#2291`、`#2297`、`#2323`、`#2442`、
`#2446`、`#2734`、`#2870`、`#2318`、`#2900`）是**已完成清单**，不是计划。

验证：

```bash
rg -n 'def register_backend' _vendor/tilelang-v0.1.15/tilelang/backend/module.py
ls _vendor/tilelang-v0.1.15/tilelang/{cuda,ascend,rocm,metal,cpu,webgpu}/pipeline.py
```

**对使用者的影响有两层。**

写 kernel 的人：`T.Kernel` 现在由**方言**拥有，同名 API 在不同方言里语义不同。
最直接的例子是昇腾：`tilelang/ascend/language/kernel.py:139-171` 的 `Kernel`
在文档字符串里就写明“this dialect has no `threads` parameter”，
它只接受一个 1-D 网格 extent，线程域只在 `T.SimtVF` 内部存在；
而 CUDA 侧 `tilelang/cuda/language/kernel.py` 的 `Kernel` 是另一套。

改编译器的人：加后端不再是“在 `src/target/` 里堆一个 codegen”，而是四件套
language dialect、target/BackendContext 贡献、PassPipeline、Host/Device codegen。
`.agents/skills/tilelang-backend/SKILL.md` 把这个契约写成了必须逐条读的清单。

### 3.2 TVM → apache/tvm + tvm-ffi（issue #2196）：**确实没落地**

**它解决什么问题。** TileLang 的 IR 是 TVM TIR，而它长期依赖自己维护的 TVM 分叉。
RFC #2196 的提问是：“当前 TileLang/tvm 这个 submodule 分叉是**有意的长期分叉**，
还只是一个历史 pin？”提问者还带来了一个下游迁移分支作为先行样本。

**关键事实。** RFC 正文给出的仓库考古值得记住，因为它是理解“为什么改编译器这么难”
的钥匙：

* `tile-ai/tilelang@main` 的 submodule 指向 `TileLang/tvm`，但**不是**指向它的
  `main`，而是指向 `tilelang_main` 分支；
* `TileLang/tvm` 的 `main` 是 2025-08-23 的时间点，比 apache/tvm 的 main 领先
  约 648 个 apache commit、本地 0 个 commit，RFC 的判断是“它实质上是一个
  **冻结的 apache/tvm pin**”；而真正被 TileLang 使用、并被 submodule 指向的是
  `tilelang_main`：它比 apache/tvm 落后约 420 个 commit，同时有 178 个本地 commit；
* 这些本地 commit 里包含 **Z3 集成**：TVM core 里 `Analyzer` 持有 `z3_prover`，
  `USE_Z3` 默认 OFF；下游迁移分支的做法是把 prover 搬到
  `src/transform/vendored/z3_prover.cc`，让 `tvm::arith::Z3Prover(analyzer)`
  变成一个兼容 shim，并建议上游 `USE_Z3` 默认 OFF。

**当前状态：仅未合入（C 档）。** 三条独立证据：

```bash
# 1) 依赖仍然是分叉
cat _vendor/tilelang-v0.1.15/.gitmodules
#    → 3rdparty/tvm -> https://github.com/TileLang/tvm

# 2) Z3 仍然长在 TVM core 上，且默认打开
rg -n 'analyzer.z3_prover' _vendor/tilelang-v0.1.15/src/transform/verify_parallel_loop.cc
#    → :106  analyzer.z3_prover.GetModel(race_free)
rg -n 'set\\(USE_Z3' _vendor/tilelang-v0.1.15/CMakeLists.txt
#    → :401  set(USE_Z3 ON CACHE STRING ...)

# 3) RFC 里那条下游做法不存在
ls _vendor/tilelang-v0.1.15/src/transform/vendored
#    → No such file or directory（v0.1.14 里也没有）
```

**对读者的影响。** 如果你准备改编译器或加后端，请注意：

1. 你读的 TVM 代码是**分叉版**的；`tirx` 命名空间（`from tvm import tirx`，
   见 `tilelang/engine/lower.py:9`）不是 apache/tvm 稳定版的样子，而来自那个分叉。
2. 你在 TileLang 里用的 Z3 相关能力（昇腾 AutoSchedule 的求解、race check 的模型输出）
   目前都建立在“TileLang 打补丁的 TVM”之上；RFC 建议的迁移会把所有权从 TVM
   搬到 TileLang（`src/transform/vendored/`），那是一次大搬家。
3. RFC 还顺带把一个 Apple Metal 后端与 MLX 互操作方案塞了进来（PR 3 / PR 4），
   说明这条迁移的**影响面远大于“换一个 submodule 地址”**。

### 3.3 CuteDSL 后端（issue #1454）：已落地，但仍是实验通道

**它解决什么问题。** NVIDIA 在 CUTLASS 之上做了 CuTe DSL（Python 形式的 tile DSL）。
与其自己重新实现一遍 CuTe 的 atom，TileLang 可以选择**把 CuTe DSL 当成一个 codegen
target**：同一条 TIR pass 链，最后产出 Python 而不是 CUDA C。

**roadmap 里的条目**是：Initial Commit（#1421）、TVM FFI Support、GEMM V2
（#1855）、Blackwell Support（#1855）、More Examples、Host Codegen Integration、
PDL Support。

**当前状态：v0.1.15 已具备，且 v0.1.14 里已有。** 后端本体是完整的：

* device codegen：`src/cuda/codegen/codegen_cutedsl.cc`（与 `codegen_cuda.cc`
  并列）、`src/cuda/codegen/rt_mod_cutedsl.cc`；
* 独立 `BackendModule`：`tilelang/cuda/cutedsl_backend.py:1-22`，它**复用同一条
  `pipeline.CUDA_PIPELINE`**，只换 codegen 与 execution backend；
* JIT 通道：`tilelang/jit/adapter/cutedsl/`（adapter、libgen、wrapper）；
* 组件库：`tilelang/contrib/cutedsl/{gemm_v1,gemm_v2,gemm_tcgen05,cpasync,ldsm,
  ptx_mma,cluster,atomic,reduce,…}.py`；
* PDL：`tilelang/contrib/cutedsl/__init__.py` 与 `codegen_cutedsl.cc` 里能找到
  `launch_dependents` / `griddepcontrol`；
* 目标已被官方文档列为常见 target：`docs/get_started/targets.md` 里
  `cutedsl` 与 `cuda`/`hip`/`metal` 并列，注“Requires `nvidia-cutlass-dsl`”。

验证：

```bash
ls _vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cutedsl.cc
sed -n '1,22p' _vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py
rg -n 'cutedsl' _vendor/tilelang-v0.1.15/docs/get_started/targets.md
```

**它“实验”在哪里？** 两个地方写得很明确：

1. 依赖门禁：`tilelang/jit/adapter/cutedsl/checks.py:8-10` 要求
   `nvidia-cutlass-dsl >= 4.7.0`，并把 `4.3.4` 列入黑名单；
2. 覆盖度：`examples/conftest.py:39-46` 维护一张
   `CUTEDSL_KNOWN_FAILURES` 列表，把“DeepSeek V4 的 FP4 act quant 转换尚未
   lower”等测试自动标成 `xfail(strict=False)`。

**对读者的意义。** 学习阶段不要把它当默认路线（它需要额外的
`nvidia-cutlass-dsl` 包，且部分算子尚未 lower）；但它在**理解“TileLang 与
NVIDIA 官方 DSL 的分界线”**这件事上非常有用，第 24 章 §4.8 有更细的机制对照。

---

## 4. 昇腾前沿

昇腾侧是本章最容易出错的地方：**同一个数字 950 下面挂着四条路线、两个仓库、
七个未合入的 PR**。这一节逐条把“问题 / 改动 / 状态 / 对读者的意义”写清楚。

时间线见 [figures/02-ascend-950-timeline.md](figures/02-ascend-950-timeline.md)。

### 4.1 Ascend 950 后端：PR #3308 已进 v0.1.15，PR #3310 只在 main

#### 4.1.1 PR #3308：把 950 后端带进主干（v0.1.15 已具备）

**解决什么问题。** 在 #3308 之前，昇腾支持在两棵树里：主干只有 CUDA/ROCm/Metal…
的抽象，而 A2/A3 的实际支持在社区维护的 `tile-ai/tilelang-ascend` 适配器里。
#3308 把一条**原生 950 后端**放进了主干，用 `target="ascend"` 即可编译运行。

PR 正文列出的能力面包括：`tilelang.ascend.language` 方言；在同一 kernel 里组合
Cube `T.gemm`(AIC) 与 Vector(AIV)；`T.SimdVF` / `T.SimtVF` 两种区域；显式暴露
UB/L1/L0 的分配 API 与 `T.copy` / `T.dual_copy`；MXFP8/MXFP4 block-scaled GEMM；
fractal layout 与自动 layout 推导；`AutoSchedule`（依赖与延迟感知的调度、流水、
多缓冲）；调度感知的同步插入（含冗余同步消除与 flag 复用）；`USE_ASCEND` 构建、
Bisheng 编译、NPU runtime 加载；`tvm_ffi` / `cython` execution backend 与
`torch_npu` 当前流互操作；NPU 计时与 `msprof` profiling。

PR 正文还明确划了一条边界：**目标硬件是 Ascend 950（`dav-3510`），
A2/A3 请用社区维护的 TileLang-Ascend。** 作者自报的验证条件是在
8×Ascend950DT、CANN 9.2.0 上跑 `testing/ascend/` 与 `examples/ascend/`
（评测用的 TFLOP/s 与 GB/s 数字在该页面里是图片，本节不引用具体数值）。

**状态：v0.1.15 已具备（A 档）。** 逐条验证：

```bash
ls _vendor/tilelang-v0.1.15/tilelang/ascend/language
#   → allocate.py copy_op.py frame.py gemm_op.py kernel.py simd.py sync.py …
ls _vendor/tilelang-v0.1.15/src/ascend/codegen
#   → codegen_ascend.cc codegen_ascend.h intrin_rule_ascend.cc rt_mod_ascend.cc
head -3 _vendor/tilelang-v0.1.15/tilelang/ascend/README.md
#   → # TileLang Ascend 950 Backend
rg -n 'dav-3510' _vendor/tilelang-v0.1.15/tilelang/contrib/bisheng.py
#   → :93  os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
```

一个值得单独记下的证据：`src/backend/common/` 里**没有任何**
`TargetIsAscend` 分支（`rg -n 'TargetIsAscend' src/backend/common/` 无输出）。
这正是 #3308 一系列 commit 的目标，把后端特例关进后端自己的目录，
公共 lowering 不再知道昇腾存在。这也是 §3.1 那条解耦路线在昇腾上的落地形态。

**顺带一个“反例”证据。** `v0.1.14` 里 `tilelang/ascend/` 与 `src/ascend/`
**完全不存在**。所以“昇腾后端进主干”这件事确实是 v0.1.15 周期发生的，
说它是“v0.1.15 新增”是对的。但要说清楚：**适配器仓库（A2/A3）比它更早**，
两者不是一回事。

#### 4.1.2 PR #3310：PTO 后端（仅 main）

**解决什么问题。** 昇腾 950 有第二条 codegen 路线：不生成 Ascend C，而是生成
**PTO 指令**（PTO-DSL 的模板调用），再用 `ptoas` + Bisheng 编译。
PTO 的定位在第 25 章 §4.10 讲过：它是一套 tile 粒度的指令集。

**改动清单（20 个文件 + 2 个搬迁）。** 这是本章最干净的一条可验证结论，
因为 corpus 的 main 快照（commit `994b44e`）就是 #3310 的合并提交。

```bash
cd chapters/30-frontier/code/02-tree-diff && python3 tree_diff.py --selftest
# only in right (new on main) : 20
# only in left  (gone on main): 2
# selftest: ok (main = v0.1.15 + the PR #3310 PTO files)
```

| 类别 | 文件 |
| --- | --- |
| PTO codegen | `src/ascend/codegen/codegen_pto.{cc,h}`、`src/ascend/codegen/rt_mod_pto.cc`、`src/ascend/codegen/sfu_precision.h` |
| PTO DSL 组件 | `tilelang/contrib/ptodsl/`（`gemm.py`、`simt.py`、`simd_inst.py`、`sync.py`、`mixed_kernel.py`、`rng.py`、`dcache_bypass.py`、`common.py`、`__init__.py`） |
| PTO 运行时适配 | `tilelang/jit/adapter/pto/`（`adapter.py`、`libgen.py`、`wrapper.py`、`cython_wrapper.pyx`、`__init__.py`） |
| 被搬迁 | `src/cuda/codegen/codegen_py.{cc,h}` → `src/backend/common/codegen/codegen_py.{cc,h}` |

**状态：仅 main（B 档），不在 v0.1.15。** 验证：

```bash
ls _vendor/tilelang-v0.1.15/src/ascend/codegen/         # 没有 codegen_pto.*
rg -n 'tilelang_pto|"pto"' _vendor/tilelang-v0.1.15/tilelang/ _vendor/tilelang-v0.1.15/src/
#   → 无输出
```

**这里必须写清一个矛盾。** corpus 里的 PR 页面快照（抓取于 2026-09-30）显示
该 PR 被作者标记为 draft，正文也写着“No test results or review severity counts
were provided”；但同一个 corpus 的 main 源码快照里有
`994b44e [Ascend] Add PTO backend for Ascend950 (#3310)` 这个合并提交。
两个快照指向同一天的不同时刻。本章的处理方式是：

* **事实陈述用代码**：v0.1.15 里没有 PTO；main 快照里有 PTO。
* **PR 的评审状态标为“无法确认”**：draft 标记不等于未合入，
  合入记录也不等于今天就还在（可能被后续 PR 调整）。

**对读者的意义。** 你现在**不能**在 v0.1.15 上写 `target="pto"`。
想在 950 上用 PTO，只有三条路：等下一个 release、用主干、
或参考 `tile-ai/tilelang-ascend` 的 PTO 路线（那是另一棵树，见第 25 章 §6.3）。

#### 4.1.3 issue #3317：两条路线怎么选，**没有官方结论**

issue #3317 的标题就是“It seems there are two Ascend 950 backends, which is
suggested to use”，正文提到 PR3308 与 PR3310。corpus 抓到的页面显示它已被关闭，
但**Activity 区没有可见评论**，也就是说，抓取到的内容里没有任何维护者的选择建议。

所以：

* 不要引用它作为“官方推荐用某一条”的依据；
* 它唯一能支撑的结论是“社区已经意识到存在两条路线，且这是一个需要被回答的问题”；
* 现阶段选路线的唯一可靠依据是代码里的 target 名与构建依赖：
  `ascend` 走 Ascend C/Bisheng；`pto` 走 PTO-DSL + `ptoas`。

### 4.2 `T.PersistentKernel` + `AutoPersistent`（PR #3325）：仅未合入

**解决什么问题。** 昇腾 kernel 通常应该按设备上的**物理核数**启动，但 TileLang
程序表达的是一个**逻辑网格**（可能远大于物理核数）。今天你只能自己用
`T.Persistent` 手工做任务折叠，而这需要围绕 persistent 循环重构整个 kernel 主体。
PR #3325 的提案是：源码里照旧写自然的逻辑网格，只显式声明
`T.PersistentKernel(OUT_TILES, num_cores=28)`，由新 pass `AutoPersistent`
在 `MaterializeKernelLaunch` 之前把网格折到 `num_cores` 上，并推导 wave 循环。

**改了哪些文件。**（来自 PR 正文与 commit message）

* `tilelang.ascend.language` 新增 `T.PersistentKernel(...)`，走 dialect 自有的
  启动框架（`@kernel_launch_factory` / `launch_kernel`），把 `num_cores` 与
  wave 循环注解记录到 launch block，与 `ClusterKernel` 透传 `cluster_dims`
  是同一条机制；
* 新 pass `AutoPersistent`，在昇腾 pipeline 里位于 `MaterializeKernelLaunch` 之前；
* 前端参数校验（extent 为正、`num_cores` 必填正整数、拒绝多维网格）；
* `testing/ascend/transform/test_ascend_auto_persistent.py` 的变换级与
  整条 pipeline 的 device IR 回归测试。

实现要点是这条映射：

$$
\begin{aligned}
W &= \left\lceil\frac{N_{\text{logical}}}{N_{\text{cores}}}\right\rceil,\\
\mathit{logical\_id}
  &= \mathit{wave}\cdot N_{\text{cores}}+\mathit{physical\_block\_id}.
\end{aligned}
$$

逻辑网格不能被 `num_cores` 整除时，最后一个 wave 上加
`IfThenElse(logical_id < logical_blocks)` 边界保护；能证明整除时不生成该保护。
不含 `PersistentKernel` 框架的模块保持原样，普通 `T.Kernel` 与 `T.MixedKernel`
不会被改写。

**当前状态：仅未合入（C 档）。** 验证：

```bash
rg -n 'PersistentKernel|AutoPersistent' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
#   → 无输出
rg -n 'AutoPersistent' ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang \
                       ../tilelang-corpus/02-tilelang-core-repo/raw/src
#   → 无输出（main 快照里也没有）
# 对比：v0.1.15 里只有手写的 T.Persistent
rg -n 'def Persistent' _vendor/tilelang-v0.1.15/tilelang/language/loop.py
#   → :82
```

**性能数据（PR 作者自测，不是本章实测）。** PR 正文给出在“28 个物理 AI Cube 核
的昇腾设备”上用 `example_gemm.py` + `do_bench`/msprof 路径测得的结果，
单位为有效 TFLOPS，按 `2*M*N*K/latency_ms*1e-9` 换算。摘录
`M=8192, K=1024, N=7168`（`OUT_TILES=896=28×32`，整除、无边界保护）：

| 模式 | FP8→FP32 | BF16→FP32 | FP32→FP32 | BF16→BF16 | FP32→HF32 |
| --- | --- | --- | --- | --- | --- |
| raw（直接启动逻辑网格） | 410.7 | 250.1 | 23.0 | 275.1 | 155.4 |
| manual（手写 `T.Persistent`） | 608.1 | 343.5 | 23.5 | 351.4 | 175.1 |
| auto（`T.PersistentKernel`） | 608.7 | 343.8 | 23.5 | 351.2 | 175.2 |

PR 正文的结论是：所有测试中 auto 与 manual 的差异不超过 0.12%，且持续优于 raw。
**这是 PR 作者的测量，条件（28 核设备、msprof 口径）由作者给出，本章未复现。**

**对读者的意义。** 在 v0.1.15 上，persistent 风格只能手写 `T.Persistent`
（定义在 `tilelang/language/loop.py:82`；它要解决的问题与第 11 章的 `T.Pipelined`
不同，后者是循环内的软件流水，前者是启动层的网格折叠）。#3325 的价值在于它把“persistent”
从一种**写法**变成一种**启动注解**，并且明确拒绝了两者混用
（PR 正文：`T.Persistent` 与 `AutoPersistent` 不协调、不建议组合）。

### 4.3 `AutoSimtVF`（PR #3320）：仅未合入

**解决什么问题。** 昇腾上“一块数据在 SIMT 域里怎么算”需要显式声明：
你得用 `T.SimtVF` 把并行计算包起来（v0.1.15 里这个原语存在，
`tilelang/ascend/language/frame.py:107`）。这让 GPU 风格的 TileLang 代码无法直接
跑到昇腾，每一处 `T.Parallel` 都要人来判断该不该包。
`AutoSimtVF` 的提案就是**自动把合法的 `T.Parallel` 包成 `SIMT_VF` 区域**。

**改了哪些文件。**

* `src/ascend/transform/auto_simt_vf.cc`：pass 实现，四个模块固定顺序
  `Analyze → PlanRegions → PlanStorage → VerifyPlan → Rewrite → VerifyResult`；
* `tilelang/ascend/transform/__init__.py`：导出 `AutoSimtVF`；
* 挂载位置：`VerifyBufferInit` **之后**、`UnrollLoopSkipVF` **之前**。

Stage 1 的算法要点（来自 PR 正文）：把每个最外层 `T.Parallel` 当作 VF 种子；
静态整块 `T.fill` / `T.clear` 与变 dtype 的 copy 也能起一个 VF 区域；
GM/Cube↔UB 的搬运是区域边界；用 fragment 的定义-使用链把 copy 与 reduction
融合进对应的 VF；生命周期被单个区域包住的 `alloc_fragment` 会被搬进 VF，
跨区域存活的则新建 `shared.dyn` 状态缓冲并在两者间通信。
PR 自己划了边界：Stage 1 **只保证合法性与正确性**，不做“最赚钱的分区”搜索，
不融合标量与 SIMT 执行，也不处理 SIMD。

**当前状态：仅未合入（C 档）。** 验证：

```bash
rg -n 'AutoSimtVF' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
#   → 无输出（main 快照里也无）
# 对照：它要插在哪个 pass 前面，v0.1.15 里已经有那一步
rg -n 'UnrollLoopSkipVF|VerifyBufferInit' _vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py
#   → :76 VerifyBufferInit   :80 UnrollLoopSkipVF
```

**对读者的意义。** 这条 PR 说明一件事：**“`T.Parallel` 在昇腾上到底意味着什么”
还在协商中**。第 25 章讲过，昇腾的 `T.SimtVF` / `T.SimdVF` 是两个显式区域，
宽度还有硬约束；`AutoSimtVF` 想让编译器替你决定边界。在它合入之前，
跨硬件复用 kernel 时你必须自己处理这段差异。

### 4.4 Ascend 算子生成与调优 agent（PR #3329）：仅未合入

**解决什么问题。** 昇腾算子开发有一套固定流程：硬件识别 → 方案设计 → 算子生成 →
指定用例精度验证 → ST 测试 → 可选性能调优 → 可选代码评审。PR #3329 想把这套流程
连同“专业能力”一起放进仓库，让 agent 按流程走。

**改了哪些文件。**

* `agent/core/`：工作流定义（阶段编排、信息传递、用户确认）；
* `.agents/skills/tilelang-ascend-op/`：8 个 skill，`npu-arch`、
  `tilelang-op-design`、`tilelang-op-generate`、`tilelang-op-test`、
  `tilelang-op-profiling`、`tilelang-perf-optimize`、
  `tilelang-performance-best-practices`、`tilelang-op-review`；
* `agent/init.sh`：从同一份源文件生成 Codex / OpenCode / Claude Code 三套配置，
  避免维护三份。

PR 正文里两条约束值得记住：目标后端（PTO 或 AscendC）由启动命令里的
`TILELANG_DEFAULT_TARGET` 决定，**不写死在源码与测试里**；
任何 mode 都不得靠“改参考实现、放宽精度阈值、跳过测试”来拿到 pass。

**当前状态：仅未合入（C 档）。** 验证：

```bash
ls _vendor/tilelang-v0.1.15/agent/core        # 不存在
ls _vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend-op   # 不存在
# 但 v0.1.15 里确实有一套「给人和 agent 读」的能力文档，名字不同：
ls _vendor/tilelang-v0.1.15/.agents/skills
#   → tilelang-{semantic,tvm-ir,build,backend,submit-pr,layout,cpp-style,ascend,…}
```

注意最后一条：v0.1.15 里有一个 **`tilelang-ascend`** skill
（描述“Write, port, tune, and debug TileLang kernels for Huawei Ascend NPUs”），
#3329 要加的是 **`tilelang-ascend-op`**。名字差一个后缀，用途也不同：
前者是给“改 TileLang / 写昇腾 kernel 的人”的参考文档，后者是**算子生成流水线**。
把两者混为一谈，是本节最容易犯的错。

**对读者的意义。** 它和 §6 的 TileFoundry 指向同一个方向：**算子生产正在被
agent 化**。但本章的方法论在这里同样适用，agent 生成的 kernel 也要先按四步流程
确认 API 在哪个版本可用，否则你拿到的是一段只在他机器上能跑通的代码。

### 4.5 其余四条昇腾 PR：状态可判定，细节无法确认

任务清单里还有四条 PR（#3328、#3322、#3324、#3321），corpus **没有收录它们的页面
快照**，只有合成文档 `90-synthesis/03-ascend-route.md` 的转述。
因此对这四条，本章只能做“反向验证”：在 v0.1.15 里找它们**必然要留下的痕迹**。

| 编号 | 主题 | 反向验证（v0.1.15 里的现状） | 状态 |
| --- | --- | --- | --- |
| #3328 | L0C→f16/bf16 的 `dual_copy` on-path cast | `tilelang/ascend/language/copy_op.py:139-151` **明确拒绝**：src/dst dtype 不同且 src 在 `shared.l0c` 时直接 `raise ValueError`，注释写 dav-950 的 fixpipe 不支持 `quant_pre`，硬件可能静默算错 | 未落地；PR 细节无法确认 |
| #3322 | causal sparse attention 例子 | `ls _vendor/tilelang-v0.1.15/examples/ascend/` 只有 13 项（`example_gemm.py`、`flash_attention/`、`deepgemm/`、`example_rmsnorm.py`…），没有 causal sparse attention 例子 | 未落地；PR 细节无法确认 |
| #3324 | `torch_npu` 任务队列 | v0.1.15 里 `torch_npu` 的引用只承担互操作/加载/计时职责（`CMakeLists.txt` 的 Cython 目标、`docs/runtime_internals/stubs.md` 的加载说明、`tilelang/ascend/torch_npu_exchange.pyx`、`tilelang/jit/adapter/`、`tilelang/profiler/msprof.py` 等），没有任何任务队列 API | 未落地；PR 细节无法确认 |
| #3321 | host / device profiling 关联 | `tilelang/profiler/msprof.py` 只有 `get_msprof_cache_flush_kernel`（:14）与 `bench_with_msprof`（:40），后者内部有一个 `KernelProfile` dataclass，字段是 AIC/AIV pipe utilization；没有任何“host 时间 ↔ device kernel”的关联结构 | 未落地；PR 细节无法确认 |

```bash
# 一条命令把上面四条的反向验证跑完
sed -n '139,151p' _vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py
ls _vendor/tilelang-v0.1.15/examples/ascend/
rg -n 'task_queue' _vendor/tilelang-v0.1.15/tilelang/ascend/
rg -n 'correlat|host_time' _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py
```

**写法要求。** 对这四条，正确的结论是
“**未在 v0.1.15 / main 快照中找到对应实现；PR 的具体内容本章无法确认**”，
而不是“不存在”。#3328 尤其要注意：`copy_op.py` 里那段拒绝逻辑本身就是一个
**未解决的技术约束**（硬件 fixpipe 不支持 dav-950 上的类型转换），
PR 要做的是绕过或规避它，所以“v0.1.15 拒绝”并不能推出“PR 不可能实现”。

---

## 5. 语义与正确性 RFC

这一节的三条 RFC 都属于“讨论语义，而不是加功能”，因此它们的落地状态差异最大：
一条已经发布，两条确实还停在讨论阶段。

### 5.1 reducer 的 ownership-safe 延迟归约（issue #2897）：**主体已发布**

#### 5.1.1 它要解决什么

考虑流式 GEMV：每个 pipeline 迭代里都要往累加器里加一点东西，
如果每迭代都调一次 `T.reduce_sum`，就会在每个迭代插一次集合通信，
代价不可接受。所以要有一个“先在各线程本地累积、最后只做一次跨线程归约”的东西
这就是 reducer。

RFC 的核心论点是：**“最终结果的 replication 布局”不等于“partial 的归属”**。
旧实现把两者混在一起，产生了静默错码。RFC 给的最小复现是：

```python
BD = 8
THREADS = 128
with T.Kernel(1, threads=THREADS):
    x_frag = T.alloc_fragment((BD,), T.float32)
    T.copy(x, x_frag)
    total = T.alloc_reducer((1,), T.float32, op="sum", replication="all")
    T.fill(total, 0.0)
    for j in T.Parallel(BD):
        total[0] += x_frag[j]
    T.finalize_reducer(total)
```

在 legacy 布局下，每个逻辑 `x[j]` 会被 16 个 replica 线程各加一次，
于是 `sum([1..8])` 得到 `576` 而不是 `36`，正好是 `128/8 = 16` 倍。

#### 5.1.2 v0.1.15 里的实际状态

RFC 提出的 v2 API **已经在 v0.1.14 与 v0.1.15 里实现**：

```python
acc = T.alloc_reducer((M,), T.float32, op="sum")   # 不再需要 replication=
T.reducer_init(acc)                                 # 可选 seed：T.reducer_init(acc, init)
for k_tile in T.Pipelined(...):
    for i, j in T.Parallel(M, K_TILE):
        T.reducer_update(acc[i], A_frag[i, j] * x_frag[j])
dst = T.alloc_fragment((M,), T.float32)
T.finalize_reducer(acc, dst)                        # 独立目标，而非原地
```

代码位置：

| 组成 | 位置 |
| --- | --- |
| `alloc_reducer`（`replication` 变为 deprecated 参数） | `tilelang/language/allocate.py:296` |
| `reducer_init` / `reducer_update` / `finalize_reducer(acc, dst)` | `tilelang/language/reduce_op.py:358`、`:398`、`:434` |
| epoch 状态机校验 | `src/transform/verify_reducer_epoch.cc`（pass 注册在 `:326`） |
| legacy 语法规范化 shim | `src/transform/canonicalize_legacy_reducer.cc`（pass 注册在 `:458`） |
| 物理计划物化 | `src/transform/reducer_plan_materialize.cc` |
| 挂进各后端 pipeline | `tilelang/cuda/pipeline.py:91-92`、`tilelang/ascend/pipeline.py:74-75`，rocm / metal / cpu / webgpu 同样两行 |

```bash
rg -n 'def reducer_update|def reducer_init|def finalize_reducer' \
   _vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py
rg -n 'CanonicalizeLegacyReducer|VerifyReducerEpoch' \
   _vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py \
   _vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py
# 而且 v0.1.14 里就已经存在：
rg -n 'def reducer_update' _vendor/tilelang-v0.1.14/tilelang/language/reduce_op.py
```

`verify_reducer_epoch.cc` 的头部注释把“第一版”强制规则写得很清楚：每个
`local.reducer` 分配恰好一次 init、零到多次 update、恰好一次 finalize；
finalize 必须与 init 在同一个控制流作用域（或它的条件细化）内；
update 只能出现在 init 与 finalize 之间且必须在 `T.Parallel` 内；
reducer 在三个一等 op 之外是**不透明**的（普通读写、`fill`/`clear`/`copy`、
`access_ptr`、别名全部拒绝）。

#### 5.1.3 哪些部分确实还没落地

把 RFC 的阶段与代码对照（图见 [figures/05-reducer-rfc-phases.md](figures/05-reducer-rfc-phases.md)）：

| RFC 里的东西 | v0.1.15 |
| --- | --- |
| Phase 1：一等 update op + 状态机验证 |  已实现 |
| Phase 2：canonical ownership-safe lowering（representative predicate） | 已有承载它的 pass（`src/transform/reducer_plan_materialize.cc`，并挂进各后端 pipeline）；本次只核对到“pass 存在且被调用”，**未逐行核对其 representative predicate 的写法** |
| Phase 3：独立 finalize 目标 |  API 已存在（`finalize_reducer(acc, dst)`）；子群快路径是否实现**未在本次核对范围内** |
| 稀疏 / 分段 partial layout（RFC 用词是 later work） |  没有实现 |
| Phase 4：废弃 `replication=` 与 `T.fill(reducer)` |  只发了 `DeprecationWarning`；`examples/gemv/example_gemv.py:239,250` 仍在用 legacy 写法 |

#### 5.1.4 对读者的意义

* **不要因为 RFC 开着就不敢用 reducer。** v0.1.15 的 v2 语义是
  “一次逻辑贡献只计一次”，这正是 RFC 要求的行为。
* **也不要照抄旧教程的 legacy 写法。** `alloc_reducer(..., replication="all")`
  会走 legacy 分支并发出 `DeprecationWarning`（`tilelang/language/allocate.py:336-346`
  有明确的迁移提示），而且它正是 RFC 里那个错码例子的写法。
* **本章不引用那个错码例子来评价 v0.1.15**。它描述的是 RFC 撰写时 `main`
  （`fef9f760…`，见 RFC 正文）上的行为，不能直接套到 release 上。

### 5.2 atomic 的内存序（issue #2708）：**确实还没落地**

**它解决什么问题。** `T.atomic_add(..., memory_order="acquire")` 这类调用
在 CUDA 上被接受，但有几条**回退路径会把参数丢掉**。RFC 把三条路径列成表：

* 标量 atomic add 的 pre-Hopper 编译路径：忽略 `memory_order`；
* 标量非 fp16/bf16 add 的 Hopper 路径：直接调 `atomicAdd`；
* float32 x2/x4 add（含 `return_prev`）的 pre-SM90 回退：拆成标量 `atomicAdd`
  并忽略 `memory_order`。

RFC 要求的是“选一种行为并写下来”：要么用支持的 PTX/ fence 实现等价序，
要么在 lowering 阶段对不支持的组合报错，要么把公开契约限制为 relaxed 并诊断其它取值。
**“静默接受参数然后丢掉”不能继续是契约。**

**v0.1.15 的实情。** 打开 `src/tl_templates/cuda/atomic.h` 可以看到两套代码并存：

| 代码 | 是否处理 `memory_order` |
| --- | --- |
| `AtomicAdd` / `AtomicAddRet`（`:472`、`:508`、`:517`） | 接受 `memory_order` 参数并使用 |
| `tl_atomic_detail::AtomicAddx2Scalar` / `AtomicAddx4Scalar` / `AtomicAddx2ScalarRet` / `AtomicAddx4ScalarRet`（`:310-336`） | **不接收该参数**，内部逐个调 `atomicAdd(ref+i, val)` |
| `IsAcquireLikeMemoryOrder`（`:89-90`） | 把 `consume` 与 `acquire` 等价处理。这是 RFC 提到的“目前聚焦的 consume 修复” |

```bash
sed -n '306,337p' _vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h
sed -n '85,95p'   _vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h
rg -n 'memory_order' _vendor/tilelang-v0.1.15/tilelang/language/atomic.py | head
```

**状态：仅未合入（C 档）。** RFC 是 2026-07-19 提的，v0.1.15 里那条
`AtomicAddxNScalar` 家族依旧原样（`atomic.h` 在 v0.1.14 与 v0.1.15 之间没有变化）。

**对读者的意义。** 写原子操作时注意 `tilelang/language/atomic.py` 的文档字符串
自己就写了边界：`memory_order` 由 CUDA target 的两条路径遵守，
**HIP 与 CuteDSL codegen 目前忽略它**（见 `:45`、`:131`、`:221` 的三段说明）。
再加上本节这条回退路径的问题，你的结论应当是：

> 在 TileLang 里用非 relaxed 内存序时，**必须同时确认目标架构与所选 lowering 路径**，
> 不能只看 API 接受了这个参数。

### 5.3 cache policy（issue #3178）：**确实还没落地**

**它解决什么问题。** `T.copy` 目前没有一等的方式来指定 NVIDIA PTX 的
per-instruction cache operator（loads: `.ca/.cg/.cs/.lu/.cv`；stores:
`.wb/.cg/.cs/.wt`）。现有的 `eviction_policy` 参数是**TMA 的 L2 eviction priority**
（`evict_normal` / `evict_first` / `evict_last`），两者是完全不同的概念。
今天想表达 cache operator，只能绕开 `T.copy` 的 lowering 去用外部调用。

RFC 提议的 API 是：

```python
T.copy(src, dst,
       load_cache_policy="cg",
       store_cache_policy="wt",
       prefer_instruction="sync")
```

并明确要求：`None` 保持现有行为；非法方向组合（如 `load_cache_policy="wt"`）
在编译期拒绝；**请求的策略绝不能被静默忽略**。

**状态：仅未合入（C 档）。** 验证：

```bash
rg -n 'load_cache_policy|store_cache_policy' \
   _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src      # 无输出
# 对照：eviction_policy 确实存在，而且只服务于 TMA
rg -n 'eviction_policy' _vendor/tilelang-v0.1.15/tilelang/language/copy_op.py
#   → :140（参数）、:178-179（写进 annotation）
```

**对读者的意义。** 别把两个概念混起来：`eviction_policy` 管的是 TMA 的 L2 淘汰优先级，
不是普通 SIMT 路径上的 `ld.global`/`st.global` cache operator。
现在想控制后者，只能走 `T.__ldg` 一类低层入口（RFC 的 Alternatives 里讨论了这条路线
为什么不理想）。这也解释了 RFC 为什么坚持“必须是 `T.copy` 的一等参数”，
否则就会丢掉向量化、谓词化、尾部处理与指令选择。

---

## 6. 生态：TileOPs / TileRT / TileFoundry / TileSight

地图见 [figures/04-ecosystem-map.md](figures/04-ecosystem-map.md)。
这一节的判断全部来自各自 README 快照的原文（抓取日 2026-09-30），
**不是本章的实测**。

### 6.1 四个仓库各自的定位

| 仓库 | 定位（原文） | 快照里的动静 |
| --- | --- | --- |
| **TileOPs** | “Spec-driven LLM operators across backends — built by agents”；算子的**规格**先写，代码由规格派生并对规格判定 | 191 stars；最后 push 2026-09-30；README 首页挂着 spec 覆盖率与 benchmark 两个 badge |
| **TileRT** | “Tile-Based Runtime for Ultra-Low-Latency LLM Inference” | 1830 stars；最后 push 2026-09-24；PyPI `tilert`；有 HuggingFace 上的 `DeepSeek-V3.2-Exp-TileRT` |
| **TileFoundry** | “Hand the compiler to the agent.”，**编译器仍是工具，agent 只是拿着它的人** | 121 stars；最后 push 2026-09-29；PyPI `tilefoundry` 已到 0.0.2；News 里给出四个“整个 agent 运行过程都保留下来”的实例（Qwen3-1.7B、Qwen3.5-35B-A3B、MiniCPM3-4B、granite-4.0-h-small） |
| **TileSight** | tile 级解析性能模型：从 tile 形状、数据搬运、软件流水与硬件 profile 估算代价/访存/时间，**可在 CPU 上跑** | 10 stars；2026-08-18 创建；最后 push 2026-09-23 |
| tilelang-benchmark | 已归档，README 明确“please checkout TileOPs” | 最后 push 2026-06-10 |

### 6.2 “生态”与本章版本纪律的关系

三件事值得记住：

1. **它们都是独立的仓库，各有自己的版本节奏。** TileOPs 里的算子是用某个版本的
   TileLang 写的；在 v0.1.15 上跑之前，先按四步流程核对其中用到的 API。
2. **agent 化的生态已经出现，而且是有据可查的。** TileOPs 的 README 用
   “Kernels forged by TileFoundry”这个 badge 统计由 TileFoundry 产出的合并 PR；
   TileFoundry 的 News 把“一次 agent 运行 + 它测到的 decode 吞吐”作为交付物。
   这与 §4.4 的 PR #3329（仓库内自带的算子生成 agent）是同一趋势的两个样本。
3. **性能数字要看清口径。** TileFoundry 的 News 里提到 Mamba2/attention/MoE 混合模型
   在“单个 mega decode kernel、一步一次 launch、无 CUDA graph”下达到
   SGLang 的 97.5%（短上下文）与 83.2%（262144 上下文）。这是**上游自报、
   特定模型与特定上下文长度**下的数字，不能当作通用性能承诺。

### 6.3 一个命名坑

任务清单里出现过“TileFindry”这个拼写。corpus 与上游里都不存在这个名字；
存在的是 **`tile-ai/TileFoundry`**（121 stars，PyPI 包名 `tilefoundry`）。
本章按 `TileFoundry` 处理；在别处看到“TileFindry”应按拼写错误理解。

---

## 7. NVIDIA 侧的对照：cuTile 与 CuTe DSL

昇腾侧的每一条前沿都需要一个“对照系”，否则你无法判断某个设计是 TileLang 的选择
还是硬件的必然。NVIDIA 侧的两个新抽象正好提供这个对照。

### 7.1 cuTile Python：同一抽象层次的另一条路

`NVIDIA/cutile-python` 把自己描述为“a programming model for writing parallel
kernels for NVIDIA GPUs”。它最小的 kernel 长这样（摘自其 README）：

```python
import cuda.tile as ct

@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
```

和 TileLang 相比，它的**抽象层次几乎一样**：都是 tile 级、都在 Python 里写、
都靠编译器决定细节。区别在编译目标与生态：cuTile 生成的是 **Tile IR**，
需要 NVIDIA 驱动 r580 或更高，且其 README 写明 `tileiras` 编译器（13.2）
**只支持 Blackwell 与 Ampere/Ada**，Hopper 还在路上。

### 7.2 CuTe DSL：TileLang 把它当 target，而不是对手

CuTe DSL 是 CUTLASS 自带的 Python DSL。TileLang 对它的态度与对 cuTile 完全不同：
**不竞争，而是把它当作一个 codegen target**（§3.3）。这也是为什么
`tilelang/contrib/cutedsl/` 里会有 `gemm_v1.py` / `gemm_v2.py` /
`gemm_tcgen05.py` / `cpasync.py` 这些文件。它们是把 CUTLASS 的零散 atom
组织成 TileLang 需要的那几个入口，而不是重新实现一遍 CuTe。

### 7.3 这对读昇腾的意义

把两件事并排放：

```text
NVIDIA 侧：TileLang(CUDA codegen) ─┬─ TileLang(CuTeDSL codegen) → CUTLASS
                                   └─ 对标物：cuTile (Tile IR)
昇腾侧：  TileLang(AscendC codegen) ─┬─ TileLang(PTO codegen，仅 main)
                                     └─ 对标物：AscendNPU IR / 手写 Ascend C
```

两边的结构是**同构的**：TileLang 都在“用自己的 pass 链 + 换一个 codegen target”
的位置，而每个硬件厂商自己也有一条官方 DSL。理解了 NVIDIA 侧的这套结构，
再看昇腾侧 `ascend` / `pto` / `npuir` 三条路线，就不容易把“路线”当成“版本”。
更细的机制对照见第 24 章 §4.8 与第 25 章 §4.8–§4.10。

---

## 8. 如何持续跟踪上游

本章的判断都有保质期。这一节给一套可以长期执行的流程；
脚本版在 [`code/03-upstream-watch/upstream_watch.py`](code/03-upstream-watch/upstream_watch.py)
里，直接 `python3 upstream_watch.py` 就能打印全部命令。

### 8.1 四类要订阅的东西

| 类别 | 具体对象 | 为什么 |
| --- | --- | --- |
| Release / tag | `tile-ai/tilelang`、`tile-ai/tilelang-ascend` | 决定教材基线；每次新 tag 都要整体重审 |
| Issue / PR 标签 | `Ascend`、`roadmap`、`RFC`、`backend`、`wrong-code` | `wrong-code` 标签下的条目直接关系正确性 |
| 昇腾官方 | CANN 版本说明、Bisheng 的 arch 列表（`dav-3510` / `dav-c310` / `dav-c220` …） | arch 字符串与工具链行为不在 TileLang 仓库里 |
| 生态 | TileOPs、TileRT、TileFoundry 的 README“News”段落 | 它们会先暴露需求，再回到编译器变成 pass |

### 8.2 用 git 比较 tag 与 main

```bash
git clone https://github.com/tile-ai/tilelang.git tilelang-upstream
cd tilelang-upstream && git fetch --tags origin

git tag --sort=-creatordate | head -20                    # 最新 tag 排在前面
git log --oneline v0.1.15..origin/main -- tilelang/ascend src/ascend \
                                          testing/ascend examples/ascend
git diff --name-status v0.1.15..origin/main -- tilelang/ascend src/ascend
git log -S 'PersistentKernel' --oneline --all -- tilelang src   # 符号何时进入
```

### 8.3 一个必须知道的限制：本地 `_vendor/` 是 `--depth 1`

本书的 `_vendor/` 三个 checkout 都是 `git clone --depth 1 --branch <tag>`，
每个只有一个 commit：

```bash
python3 code/03-upstream-watch/upstream_watch.py --run
# tilelang-v0.1.14   | 7e3bbf3703e7 | v0.1.14 | 1 commit(s)
# tilelang-v0.1.15   | a35f8ddf45eb | v0.1.15 | 1 commit(s)
```

后果是**本地无法** `git diff v0.1.14..v0.1.15`，也无法用 `git log -S` 追符号。
想比较两个 tag，只有两个办法：

1. 建一个完整 clone（上面的命令）；
2. 做文件级比较。这正是 [`code/02-tree-diff/tree_diff.py`](code/02-tree-diff/tree_diff.py)
   存在的原因，也是本章 §4.1 能给出“main 比 v0.1.15 多 20 个文件”这条结论的方式。

**“我本地 diff 不出来”不等于“两者没有差异”。** 这句话值得写进工作习惯：
先 `git rev-list --count HEAD` 确认你手里的是快照还是完整历史。

### 8.4 判断“一个能力能不能用”的清单

在写任何一行依赖某个新能力的代码之前，按顺序回答：

1. **符号在 `_vendor/tilelang-v0.1.15/` 里存在吗？**（存在 → 可以用）
2. **它在 v0.1.14 里也存在吗？**（存在 → 它不是本次新增，别写“新特性”）
3. **它需要额外的运行条件吗？**（CuTe DSL 要 `nvidia-cutlass-dsl >= 4.7.0`；
   Ascend 要 CANN + Bisheng；PTO 要 `ptoas`）
4. **它在文档里被承诺的边界是什么？**（例如 `memory_order` 在 HIP/CuteDSL 上被忽略）
5. **它有没有已知的 `wrong-code` 记录？**（查 `wrong-code` 标签与 examples 里的
   `xfail` 列表）

只有前两问都答完，才轮到“它好不好用”。

---

## 9. 总表：能力 → 状态 → 验证方式

下表所有“状态”列都用 `_vendor/` 与 corpus 快照实际核对过（核对日 2026-10-01），
可以直接用第三列的命令复现。`A / B / C` 分别对应 §2.1 的三种状态。

| 能力 | 状态 | 验证方式（在本书根目录执行） |
| --- | --- | --- |
| 多后端注册表 + 每后端 pipeline（#2115） | **A**（v0.1.14 起） | `rg -n 'def register_backend' _vendor/tilelang-v0.1.15/tilelang/backend/module.py` |
| TVM 依赖仍为 `TileLang/tvm` 分叉 | **A**（= #2196 未落地） | `cat _vendor/tilelang-v0.1.15/.gitmodules` |
| `USE_Z3=ON`、TVM core 里有 `analyzer.z3_prover` | **A**（= #2196 未落地） | `rg -n 'set\(USE_Z3' _vendor/tilelang-v0.1.15/CMakeLists.txt`；`rg -n 'z3_prover' _vendor/tilelang-v0.1.15/src/transform/verify_parallel_loop.cc` |
| RFC #2196 的下游做法 `src/transform/vendored/z3_prover.cc` | **C**（不存在） | `ls _vendor/tilelang-v0.1.15/src/transform/vendored` |
| CuteDSL 后端（#1454） | **A**（v0.1.14 起） | `sed -n '1,22p' _vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py` |
| CuTeDSL 已知失败清单 | **A** | `sed -n '39,45p' _vendor/tilelang-v0.1.15/examples/conftest.py` |
| Ascend 950 AscendC 后端（#3308） | **A**（v0.1.15 新增） | `ls _vendor/tilelang-v0.1.15/tilelang/ascend/language`；`rg -n 'dav-3510' _vendor/tilelang-v0.1.15/tilelang/contrib/bisheng.py` |
| Ascend 950 PTO 后端（#3310） | **B**（仅 main 快照 `994b44e`） | `python3 chapters/30-frontier/code/02-tree-diff/tree_diff.py --selftest` |
| `T.PersistentKernel` + `AutoPersistent`（#3325） | **C** | `rg -n 'PersistentKernel' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src`（无输出） |
| `AutoSimtVF`（#3320） | **C** | `rg -n 'AutoSimtVF' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src`（无输出） |
| Ascend 算子生成 agent（#3329） | **C** | `ls _vendor/tilelang-v0.1.15/agent/core`（不存在） |
| `dual_copy` 的 L0C→UB 类型转换（#3328） | **C**（v0.1.15 明确拒绝） | `sed -n '139,151p' _vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py` |
| causal sparse attention 昇腾例子（#3322） | **C** | `ls _vendor/tilelang-v0.1.15/examples/ascend/` |
| `torch_npu` 任务队列（#3324） | **C** | `rg -n 'task_queue' _vendor/tilelang-v0.1.15/tilelang/ascend/`（无输出） |
| host/device profiling 关联（#3321） | **C** | `sed -n '38,58p' _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py` |
| reducer v2（`reducer_init` / `reducer_update` / `finalize_reducer(acc,dst)`，RFC #2897） | **A**（v0.1.14 起） | `rg -n 'def reducer_update' _vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py _vendor/tilelang-v0.1.14/tilelang/language/reduce_op.py` |
| reducer epoch 验证 + legacy 规范化 | **A** | `ls _vendor/tilelang-v0.1.15/src/transform/{verify_reducer_epoch,canonicalize_legacy_reducer}.cc` |
| reducer 的稀疏 partial layout / 废弃 `replication=` | **C**（RFC Phase 3 尾 / Phase 4） | `rg -n 'DeprecationWarning' _vendor/tilelang-v0.1.15/tilelang/language/allocate.py` |
| atomic 回退路径的 `memory_order` 语义（#2708） | **C** | `sed -n '306,337p' _vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h` |
| `T.copy` 的 `load_cache_policy` / `store_cache_policy`（#3178） | **C** | `rg -n 'load_cache_policy' _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src`（无输出） |
| 既有的 TMA `eviction_policy` | **A** | `rg -n 'eviction_policy' _vendor/tilelang-v0.1.15/tilelang/language/copy_op.py` |
| `target="pto"` 在 v0.1.15 上可用？ | **不可用**（B 档能力的直接后果） | `rg -n '"pto"' _vendor/tilelang-v0.1.15/tilelang/ _vendor/tilelang-v0.1.15/src/`（无输出） |

---

## 10. 本章小结

把本章压成七句话：

1. **三种状态 + 一种诚实结论。** v0.1.15 已具备 / 仅 main / 仅未合入 PR-RFC，
   以及资料不足时的“无法确认”。状态判定只认代码，不认标题。
2. **判定是四步流程**：release → 上一版对照 → main 快照 → PR/RFC。
   每一步都有可复现命令，`code/01-state-audit/`、`code/02-tree-diff/` 把前两步自动化了。
3. **多后端解耦（#2115）早就落地了**，而且 v0.1.14 里就有。
   它的可见形态是 `tilelang/backend/` 注册表 + 每个后端一个 `pipeline.py` + 语言方言。
4. **TVM 迁移（#2196）确实没落地**：`.gitmodules` 仍指向分叉，`USE_Z3=ON`，
   那个 vendored prover 目录不存在。这是“看起来最像架构重构、实际一点没动”的一条。
5. **CuteDSL 后端（#1454）也已经发布**：codegen、adapter、组件库、PDL 都在，
   但仍是实验通道（依赖门禁 + `xfail` 清单）。
6. **昇腾侧要分清 PR #3308 与 #3310**：前者（Ascend C 路线的 950 后端）在 v0.1.15 里，
   后者（PTO 后端）只在 main 快照里；`T.PersistentKernel`(#3325)、`AutoSimtVF`(#3320)、
   算子生成 agent(#3329) 三条都还没合入。
7. **最容易误判的是 RFC #2897（reducer）**：议题还开着，但 v2 语义、epoch 验证、
   legacy 规范化早就进了 v0.1.14 与 v0.1.15；真正没做的是稀疏 partial layout
   与废弃 legacy 写法。相比之下 #2708（atomic 内存序）与 #3178（cache policy）
   确实只剩 RFC 本身。

最后一句话：**本章的所有结论都有保质期。** 下次上游发新 tag 时，
先跑一遍 `python3 code/01-state-audit/state_audit.py`。如果它报 MISMATCH，
说明世界变了，本章需要重写，而不是把期望值改掉。

## 延伸阅读

* 三种状态与判定流程：[figures/01-three-states.md](figures/01-three-states.md)、
  [figures/03-verify-flow.md](figures/03-verify-flow.md)
* 昇腾 950 的时间线：[figures/02-ascend-950-timeline.md](figures/02-ascend-950-timeline.md)
* reducer RFC 的阶段对照：[figures/05-reducer-rfc-phases.md](figures/05-reducer-rfc-phases.md)
* 生态地图：[figures/04-ecosystem-map.md](figures/04-ecosystem-map.md)
* 可运行审计工具：[code/README.md](code/README.md)
* 引用与核验命令：[references.md](references.md)
* 上一章：第 29 章“编译器工程：写 pass、测试与回归”；
  下一章：第 31 章“贡献指南”。它会把本章的判断方法用在一个具体贡献流程上。
