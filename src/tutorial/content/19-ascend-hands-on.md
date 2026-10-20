# 第 19 章 昇腾实战：在昇腾上开发与调优算子

## 学习目标

读完本章，你应当能够：

1. 在一台昇腾机器上从零把环境装好：驱动、CANN、`torch_npu`、TileLang-Ascend，
   并说清每一层在软件栈里的位置、彼此的版本要求，以及 `set_env.sh` 里那几行环境变量
   为什么必须存在。
2. 跑通并**逐行讲清**两个最小昇腾 kernel（逐元素 add 与 GEMM）：指出
   `GM → UB → Vector → UB → GM` 与 `GM → L1 → L0A/L0B → Cube → L0C → GM`
   两条数据流，以及 Set/Wait（`barrier_all` / `set_flag` / `wait_flag`）在代码里
   到底写在哪一行、保护的是什么依赖。
3. 把同一个算子的 **Ascend C 写法**与 TileLang 写法摆在一起对照，逐条说出
   TileLang 省掉了什么（`TPipe`/`TQue` 的建队、L1→L0 的 `LoadData`、
   `SetFlag`/`WaitFlag` 的配对管理），并说得出**什么时候仍然应该直接写 Ascend C**。
4. 正确回答“用哪棵树”：适配器仓库 `tilelang-ascend`（Ascend C & PTO 路线，配套
   `tilelang.language`）与主干 `v0.1.15` 的昇腾方言（`tilelang.ascend.language`），
   两套 API **不通用**；并会用昇腾专属原语 `T.alloc_l1/l0a/l0b/l0c`、
   `T.dual_copy`、`T.SimtVF`/`T.SimdVF`、`T.Pipelined`、`T.annotate_buffer_versions`。
5. 会用 `msProf`（`msprof op` / `msprof op simulator`）与
   `do_bench(backend="msprof")` 采集一次性能，并根据 **MTE 带宽占用 / Cube(MAD) 利用率 /
   Vector 利用率** 判读瓶颈在哪一侧，据此决定是调多缓冲还是调 core ratio。
6. 会把若干算子用 ACLGraph 入图（capture + replay），并通过 `torch_tl_ascend`
   把 TileLang 算子接进 PyTorch；也能说清这条路的边界（它治的是 Host-bound，
   不是图编译）。
7. 用 `T.printf` / `T.dump_tensor` 和 **AICore exception dump** 定位运行期问题，
   并记住本章末尾 14 条**昇腾特有**的坑。

## 前置知识

* 第 6 章：NV GPU 与昇腾 NPU 的抽象对照（存储层级、执行单元、同步模型的整体图景）。
  本章把那张图落到“能执行的代码”上。
* 第 7 章：`@tilelang.jit`、`T.Kernel`、`T.copy`、`T.Parallel` 的语义，以及 JIT 与缓存。
* 第 9 章：`T.copy` / `T.gemm` / `T.reduce_*` / `T.fill` 的语义与降级行为。
* 第 11 章：`T.Pipelined` 与多缓冲的基本概念、`num_stages` 的含义。
* 第 12 章：调试与性能分析的方法论（先观察、后假设；先算 roofline 再看 profiler）。
  本章第 8 节是它在昇腾侧的动手版。
* 第 13 / 14 章：逐元素、归一化、GEMM 的算子实现。本章会直接引用它们的 GEMM 概念。

不需要你事先会写 Ascend C，也不需要你有昇腾硬件：第 6 节会手写一段 Ascend C 做对照，
而第 3 节会明确告诉你**没有 NPU 时能做到哪一步**。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守两条硬规矩：

1. **所有 kernel 示例的最高验证级别是“仅静态核对”**：代码能通过
   `python3 -m py_compile`，API 名与签名逐个在锁定版本的源码里比对过，
   **但没有任何一条性能数字、任何一个“Kernel Output Match!”来自本机实测**。
2. 任何需要真实硬件才能产生结果的步骤，一律写成
   **前提 / 命令 / 期望输出 / 验证级别** 四段式。凡是“期望输出”都以
   “在满足前提的机器上应当看到”表述，**不写成我们测过**。

本章出现的性能数字只有两类来源：**厂商公开资料 / 上游文档 / 上游样例里的数字**
（带 corpus 路径）与**本书的公式推导**（由 `code/` 里的纯 Python 程序复算）。
来源见 [`references.md`](references.md)。

本章涉及**两棵源码树**，请先认准：

| 记号 | 路径 | 是什么 |
| --- | --- | --- |
| **A 树（适配器）** | `_vendor/tilelang-ascend-v0.1.2.000/` | TileLang-Ascend 适配器仓库，tag `TileLang-ascend-v0.1.2.000-release`。走 Ascend C & PTO 两条 codegen 路线，前端用 `import tilelang.language as T` |
| **M 树（主干）** | `_vendor/tilelang-v0.1.15/` | TileLang 主干 `v0.1.15`。昇腾后端已进主干，前端用 `import tilelang.ascend.language as T` |

第 7 节会把两棵树的 API 逐项对照。**本章第 4、5 节的“第一个 kernel”用的是 A 树，
因为那两个官方 example 长在 A 树里；第 7 节的现代原语用的是 M 树的方言。**

---

## 1. 问题的由来：在昇腾上写算子，“难”在哪

### 1.1 一份 kernel 要在三层地方同时正确

在 GPU 上写 CUDA kernel，你操心的是“这段代码对不对、快不快”。在昇腾上写 Ascend C
kernel，除了这两件事，你还必须显式地操心**第三件事：数据在哪一级存储、由哪个搬运单元
搬进来、和哪条流水线同步**。原因是昇腾 AI Core 的计算单元和存储单元是被**显式分层**
暴露给程序员的：

* 做矩阵乘的 Cube 单元的输入必须是 **L0A / L0B**，输出落在 **L0C**；
* 做向量/标量运算的 Vector 单元的输入输出必须在 **UB（Unified Buffer）**；
* Cube 和 Vector 之间**不能直接交换数据**，要经过 GM/L2；
* 每一级之间的搬运由 **MTE1 / MTE2 / MTE3 / FixPipe** 这些独立的搬运单元负责；
* 这些单元各自是一条异步流水线，之间的先后关系必须由 **PipeBarrier / SetFlag / WaitFlag**
  显式保证。

于是“一份 kernel 正确”被拆成三个互相独立的义务：**算法正确**、**tiling 正确（每级存储
装得下）**、**同步正确（不读到没搬完的数据）**。本章要解决的核心痛点，就是第三条。它
最容易错、最难调、也最花时间。

### 1.2 为什么昇腾线是本教材的第一优先级

本教材的硬件优先级是 **昇腾 > NVIDIA**。原因不是性能对比，而是**可迁移性**：昇腾把
“计算单元、存储层级、搬运单元、同步”这四件事全部显式暴露，逼着你在写算子时就建立
正确的机器模型；一旦建立了这个模型，回头看 NVIDIA 的 shared memory / fragment /
`cp.async` / `mbarrier`，只是同一套抽象换了一组名字。反过来，先写 CUDA 再看昇腾，
很容易把“寄存器累加器”的直觉错误地带到 L0C 上。

### 1.3 本章路线图

本章按“装环境 → 跑最小 kernel → 对照 Ascend C → 认全原语 → 调优 → 入图部署 → 调试 →
避坑”推进，每一步都能独立执行：

```
§3 环境      §4 add（UB/Vector）      §5 GEMM（L1/L0/Cube）
   │                │                          │
   └────────► §6 Ascend C 对照 ◄───────────────┘
                     │
                     ▼
              §7 原语与「哪棵树」
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      §8 调优     §9 入图部署    §10 调试
        └────────────┴────────────┘
                     ▼
                §11 常见坑清单
```

---

## 2. 概念与模型：昇腾的机器长什么样

这一节是本章的“地图”。所有后续代码都能在这张图上找到位置。

### 2.1 AI Core：AIC（Cube）与 AIV（Vector）是两块核

昇腾的一个 **AI Core** 里有两套计算单元：**Cube（矩阵）** 与 **Vector（矢量）**，另有
负责调度与标量运算的 **Scalar**。在 A2/A3/950 这类训练与推理产品上是 **分离模式**：
Cube 和 Vector 各自带独立的 Scalar 调度单元，分别叫 **AIC**（AI Cube）与 **AIV**
（AI Vector），按 1:N 的比例组合成一个 AI Core（来源：
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md`）。

这条事实有两个直接后果，本章会反复用到：

* **同一个 kernel 里 Cube 和 Vector 可以并行**，但它们之间的数据传递要过 GM/L2（M 树
  用“Workspace 消除”把这个过程自动化，见 §7.6）。
* **Vector 侧每个核有两个子单元**，所以 A 树的 kernel 里会出现 `vid ∈ {0,1}`，
  A 树示例里用 `VEC_NUM = 2` 手动把一块 tile 劈成两半（见 §4）。

### 2.2 存储层级：GM / L1 / L0A / L0B / L0C / UB

以 Atlas A2 为例的片上存储容量（来源：catlass
`docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md`）：

| 层级 | 容量 | 谁在用 | TileLang（M 树）里的写法 |
| --- | --- | --- | --- |
| GM（HBM，全局） | 大 | 输入/输出 | `T.Tensor(...)` 参数 |
| L1（CBuf） | 512 KB | Cube 的输入中转 | `T.alloc_l1(...)`（scope `shared.l1`） |
| L0A / L0B | 各 64 KB | Cube 的左/右输入 | `T.alloc_l0a(...)` / `T.alloc_l0b(...)` |
| L0C | 128 KB | Cube 的输出/累加器 | `T.alloc_l0c(...)`（fp32 累加） |
| UB（Unified Buffer） | 192 KB | Vector 的输入输出 | `T.alloc_shared(...)`（M 树）；`T.alloc_ub(...)`（A 树） |
| BT / FB | 1 KB / 7 KB | bias / 量化·relu 参数 | 由编译器管理 |

**关键对照**：NVIDIA 的累加器在寄存器（fragment）里，昇腾的累加器在 **L0C** 里。
所以“我把 C 累加在寄存器里”这个 CUDA 直觉在昇腾上不成立；昇腾的 GEMM 是
**读 L1 → 搬进 L0A/L0B → Cube 算出 L0C → 写回 GM（或经 UB 做后处理）**。

### 2.3 搬运单元与流水线 PIPE：MTE1 / MTE2 / MTE3 / FixPipe

存储层级之间的搬运由专用单元负责（来源同上，`atlas_ascendc_10_0008`）：

| 搬运单元 | 负责的通路 |
| --- | --- |
| **MTE1** | `L1 → L0A/L0B`、`L1 → BT` |
| **MTE2** | `GM → L1`、`GM → L0A/L0B`、`GM → UB` |
| **MTE3** | `UB → GM` |
| **FixPipe** | `L0C → GM/L1`、`L1 → FP`（搬运途中可做格式/类型转换） |

此外还有 **V**（Vector 计算）、**M**（Cube 计算）、**S**（Scalar）、**FIX** 等流水。
**每一条 PIPE 都是异步的**：`GM → UB` 的 MTE2 拷贝发出后不会自动等它完成，Vector
就可能读到旧数据。这就是同步存在的全部理由，**跨 PIPE 的数据依赖必须显式同步**。

### 2.4 同步模型：PipeBarrier 与 SetFlag/WaitFlag

Ascend C 提供两种同步 API（来源同上）：

* **`PipeBarrier<PIPE_X>()`**：在**同一条序列**内约束顺序。它保证“前序指令的所有读写
  全部完成后，后序指令才开始”。注意：同一条 PIPE 内的指令虽然按顺序**发射**，但**前
  一条不一定执行完**后一条就可以开始，所以需要 barrier。
* **`SetFlag<SrcPipe, DstPipe>(id)` / `WaitFlag<SrcPipe, DstPipe>(id)`**：在**两条序列**
  之间建立一对“锁”。语义是：**SetFlag 在“前序指令的所有读写完成”后把标志位置 1；
  WaitFlag 若发现标志位为 0 就阻塞，为 1 则把标志位清 0 并放行后续指令。**

在 TileLang 里，这两者都能写：

* A 树：`T.barrier_all()`（等价于 `PIPE_ALL` 的全屏障）、`T.set_flag(src, dst, id)` /
  `T.wait_flag(src, dst, id)`、`T.set_cross_flag(...)` / `T.wait_cross_flag(...)`
  （核间）。定义见 `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:191,213,233,251,275`。
* M 树：`T.ascend_pipe_barrier("PIPE_ALL")`、`T.ascend_set_flag(...)` /
  `T.ascend_wait_flag(...)`，定义见
  `_vendor/tilelang-v0.1.15/tilelang/ascend/language/sync.py`。

但要记住 M 树最重要的一条：**默认情况下你不需要手写同步**。M 树把同步编译期自动插入
（`src/ascend/transform/insert_sync.cc`），你按业务顺序写，编译器负责 flag。

### 2.5 与 NVIDIA 的四条关键差异

| 维度 | NVIDIA（CUDA 后端） | 昇腾（Ascend 后端） |
| --- | --- | --- |
| 网格 | 1D/2D/3D block + threads | **只有 1D grid**；线程域只在 `T.SimtVF` 里 |
| 片上共享存储 | shared memory | UB（`T.alloc_shared`）+ L1（`T.alloc_l1`） |
| 矩阵乘输入 | fragment（寄存器） | **L0A/L0B**（显式分配） |
| 累加器 | fragment | **L0C**（fp32） |
| 矩阵乘原语 | `T.gemm` | A 树 `T.gemm_v0`；M 树 `T.gemm(..., unit_flag_ctrl=...)` |
| Warp 级归约 | `__shfl_sync` | `T.alloc_reducer` + `T.reducer_update/finalize`，或硬件 `warp_reduce_*` |
| 同步 | 隐式（`cp.async`/`mbarrier` 由 pass 处理） | 默认自动插入 flag；手工模式用 Set/Wait |
| 编译器 | nvcc | **毕昇（bisheng）** |

（对照事实来源：`_vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend/SKILL.md` 第 1 节，
以及第 12 章图 4 的同一张对照表。）

### 2.6 一个必须先记住的结论：两棵树

“TileLang 昇腾”在锁定版本里不是一个东西，而是**两条路线、两棵树**：

| | A 树适配器（`tilelang.language`） | M 树主干 950 后端（`tilelang.ascend.language`） |
| --- | --- | --- |
| 仓库 | `_vendor/tilelang-ascend-v0.1.2.000/` | `_vendor/tilelang-v0.1.15/` |
| codegen | `ascendc` 与 `pto` 两个 target | `ascend`（默认 `dav-3510`，即 Ascend 950） |
| 已验证设备 | A2、A3 | Ascend 950 |
| 调度 | 手写为主，`pass_configs` 开自动同步 | **AutoSchedule（Z3）默认开启** |
| 分配 | `T.alloc_L1/L0A/L0B/L0C/alloc_ub`（大写） | `T.alloc_l1/l0a/l0b/l0c`（小写） |
| 矩阵乘 | `T.gemm_v0` | `T.gemm`（带 `unit_flag_ctrl`） |
| 线程/子核 | `T.Kernel(..., is_npu=True) as (cid, vid)` | `T.Kernel(N) as bx`；子核用 `T.MixedKernel(N, sids=2) as (bx, sid)` |
| 调试 | `T.printf` / `T.dump_tensor` | `T.print`（无 `dump_tensor`） |

**两棵树的写法不通用**：把 A 树的 `T.alloc_L1` 放到 M 树会 `AttributeError`，把 M 树的
`T.alloc_l1` 放到 A 树同样不行，`target=` 也不一样。第 7 节给出一张完整对照表和选择规则。

---

## 3. 用法（一）：环境准备

### 3.1 软件栈分层与版本要求

“装昇腾环境”不是装一个包，而是铺一条链子：

```
 你的程序（PyTorch / TileLang kernel）
        │
   torch_npu  ← PyTorch 的昇腾后端
        │
 TileLang-Ascend / 主干昇腾后端   ← 前端 DSL + 编译器
        │
  毕昇编译器 bisheng + Ascend 运行时
        │
      CANN 工具包（含算子库、runtime、编译器）
        │
   昇腾驱动（npu-smi、HDC/P2P、资源管理）
        │
       NPU 硬件
```

CANN 是“昇腾异构计算架构”，对上支持 AI 框架、对下服务 AI 处理器与编程，是整个栈的
中枢（来源：`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann.md`）。
驱动是最底下的一层，负责设备管理、资源调度与 Host/Device 通信，提供 `npu-smi` 等工具
（来源：`.../cann/zh_cann_driver.md`）。毕昇是 Host-Device 异构编译器，TileLang 生成的
Ascend C 就交给它编译成 `.so`（来源：`.../cann/zh_cann_bisheng.md`）。

**版本要求（来自锁定版本的自述文件）**：

| 组件 | A 树要求 | M 树要求 |
| --- | --- | --- |
| CANN | ≥ 8.3.RC1（`README.md:79`） | 兼容版本的驱动 + CANN（含 `bisheng`、CCE 版 `ld.lld`、Ascend runtime），**不假定固定路径** |
| torch_npu | ≥ 2.6.0.RC1 | 兼容的 PyTorch + torch_npu，`torch.npu.is_available()` 为 `True` |
| Python | ≥ 3.10（安装脚本检查） | 见上游安装指南 |
| 目标设备 | A2 / A3 | Ascend 950（默认 `dav-3510`） |

> 注意：上面这些是**上游自述的版本要求**，不是我们验证过的组合。实际部署请以厂商发布说明为准。

### 3.2 第一步：驱动与 CANN

**前提**：一台带昇腾 NPU 的 Linux 机器（x86 或 aarch64），有 root 或 sudo。

**命令**（示意，随产品与安装方式不同而不同）：

```bash
# 1) 安装驱动后确认设备可见
npu-smi info

# 2) 安装 CANN 工具包后设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3) 确认编译器可用
which bisheng
```

**期望输出**：`npu-smi info` 打印出一张设备表（型号、算力、显存、温度等）；
`which bisheng` 返回 `bisheng` 的绝对路径。

**验证级别**：仅静态核对（本书作者未在有 NPU 的机器上执行过）。

> 为什么要有 `set_env.sh`：CANN 的安装是“解包 + 设环境变量”，不是写进系统路径。
> M 树自述明确要求把 CANN 环境按你的工具包说明配好再编译/运行，且**不假定任何固定
> 安装路径**（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:30-42`）。

### 3.3 第二步：torch_npu

**前提**：CANN 已配好，Python ≥ 3.8。

**命令**：

```bash
pip install torch torch_npu    # 版本必须与当前 CANN 匹配，见厂商配套说明
```

**期望输出**：

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available())"
# 应打印 True
```

**验证级别**：仅静态核对（未实测）。

> M 树的后端自述把“`torch.npu.is_available()` 返回 `True`”写成了安装完成的判据
> （`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:20-24`）。本章 `code/01-env-check`
> 就是这段检查的可执行版。

### 3.4 第三步：安装 TileLang-Ascend（两条路，先选树）

**先决定用哪棵树**（规则见 §7.6）。下面两条路分别对应 A 树和 M 树。

#### 3.4.1 A 树：适配器仓库（Ascend C & PTO 路线）

官方给了三种方式（`_vendor/tilelang-ascend-v0.1.2.000/README.md:84-133`）：

**方式一：装 wheel（推荐）**

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
pip install tilelang-*.whl
```

**方式二：从源码构建 wheel**

```bash
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
./build_wheel_ascend.sh [--enable-llvm]
pip install dist/tilelang-*.whl
```

**方式三：从源码编译安装**

```bash
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
bash install_ascend.sh          # 可选 --enable-llvm / --enable-shmem / --enable-incremental
source set_env.sh
```

`install_ascend.sh` 做了这些事（逐段读 `_vendor/tilelang-ascend-v0.1.2.000/install_ascend.sh`）：

1. 检查 Python ≥ 3.10（脚本打印与报错信息如此；注意它的版本比较写法本身有坑，
   见下面的 P2 补充）；
2. `pip install -r requirements-build.txt` 与 `requirements.txt`；
3. `git submodule update --init --recursive` 拉 TVM 子模块；
4. `bash 3rdparty/patches/apply_tvm_patches.sh` 给 TVM 打补丁（**失败即退出，不静默继续**）；
5. 默认（非增量）路径下 `rm -rf build && mkdir build`，写入 `set(USE_ASCEND ON)` 与
   `set(USE_GTEST OFF)`，`cmake ..`；带 `--enable-incremental` 则复用已有 `build/`；
6. `make -j${50% 核数}`（脚本注释说明：用满核数会把系统卡住，所以只用一半）；
7. 若带 `--enable-shmem`，额外编译并安装 `shmem` 包，最后 `source ../../install/set_env.sh`。

（脚本还支持 `--enable-llvm` 下载并解压指定版本的 LLVM、`--enable-coverage` 装 `lcov`，
这两条与前面的默认流程无关，按需启用。）

> **读脚本笔记（2026-10-01 修正）**：`install_ascend.sh` 的版本门禁是
> `python_version=$(python3 --version | grep -oP '\d+\.\d+')` 后
> `if (( major >= 3 && minor >= 10 ))`。对 3.x 系列它是正确的（3.9 被正确拒绝、
> 3.10 正确通过）；这个写法真正的问题在 **Python 4.x**：`4.0` 会被判成不满足
> （因为 `minor >= 10` 为假），届时需要把条件改成
> `major > 3 || (major == 3 && minor >= 10)`。读脚本时留意这一点即可。

`set_env.sh` 只做三件事（`_vendor/tilelang-ascend-v0.1.2.000/set_env.sh`）：

* 把仓库根目录导出为 `TL_ROOT`；
* 把 `TL_ROOT` 加到 `PYTHONPATH` 最前面（所以 `import tilelang` 拿到的是这棵树）；
* 设置 `ACL_OP_INIT_MODE=1`，注释写明是为了“使用 torch_npu 时不导入 tvm”。

**期望输出**：`echo $TL_ROOT` 指向仓库根；`python -c "import tilelang; print(tilelang.__file__)"`
指向这棵树的 `tilelang/`。

**验证级别**：仅静态核对（脚本逐行读过；未在昇腾机器上执行）。

#### 3.4.2 M 树：主干 v0.1.15 的昇腾后端

```bash
git submodule update --init --recursive
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .
```

或者可编辑安装（开发用）：

```bash
python -m pip install -r requirements-dev.txt
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -e . -v --no-build-isolation
```

自述里三点值得注意（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:44-52`）：
`USE_CUDA=OFF` 让这台机器不需要 CUDA 工具链；Linux 上昇腾默认开启；
已有 build 目录时用 `CMAKE_ARGS` 传 `-DUSE_*=...`。

**验证级别**：仅静态核对。

### 3.5 第四步：验证环境

本章 `code/01-env-check/env_check.py` 是一段可执行的环境自检，它**不导入 TileLang**
（避免在没装环境的机器上直接崩），而是逐项探测并打印结论：

```bash
python3 code/01-env-check/env_check.py
```

**期望输出**（有 NPU 的机器上）：`torch_npu: available`、`tilelang import: ok` 等；
**在无 NPU 的机器上**也能跑，会打印 `torch_npu: not installed` 之类的降级结论。

**验证级别**：CPU 逻辑验证（本机跑过，输出见该目录 README）。

### 3.6 没有 NPU 时能做什么：camodel 仿真与静态核对

如果你手边没有 A5/950 的硬件，还有一条官方路线：**A5 camodel（cycle-accurate model）**
仿真。原理是用 `libruntime_camodel.so` 替换 `libruntime.so`，对外接口完全一致，内部用
CPU 模拟 NPU 行为（来源：A 树编程指南 `docs/TileLang-Ascend Programming Guide.md:194-276`）。

**前提**：装了包含 A5 simulator 镜像（`Ascend950PR_9599`）的 CANN，CANN ≥ 9.0.0。

**命令**：

```bash
python .agents/skills/tilelang-a5-sim-convert/scripts/run_a5_sim_template.py --log-dir ./my_sim_logs
```

**期望输出**：

```
KERNEL OUTPUT MATCH!
Done!
```

**边界**：仿真**只支持 PTO 后端**（`target="pto"`）的 A5 平台；速度比真实 NPU 慢约
1000 倍以上，**只能验功能，不能测性能**；不支持 `torch.npu`，数据要用 `rtMalloc`/`rtMemcpy`。

**验证级别**：仅静态核对（未运行）。若你的机器满足前提，这是本章所有 kernel 示例最
现实的“无卡运行”方式。**但请以实际输出为准，不要相信本节给出的示例文字**。

---

## 4. 用法（二）：第一个昇腾 kernel ，逐元素 add

### 4.1 选择哪个例子

A 树的 `examples/elementwise/` 下有 3 个文件：

| 文件 | 内容 | 适合谁 |
| --- | --- | --- |
| `elementwise_add.py` | 最小：GM→UB→Add→GM，2 个 `T.barrier_all()` | **第一次跑昇腾，从这里开始** |
| `elementwise_add_pipeline.py` | 手写双缓冲 + 多类 flag | 想看“多缓冲和同步长什么样” |
| `setvalue_example.py` | `T.tile` 的写值类原语 | 之后再看 |

我们选 `elementwise_add.py`。

### 4.2 逐行解释

完整代码（`_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py`）：

```python
import argparse
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()                                  # ①

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024)
parser.add_argument("--n", type=int, default=1024)
args = parser.parse_args()
M, N = args.m, args.n

@tilelang.jit(out_idx=[-1])                                   # ②
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M                                      # ③
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):   # ④
            bx = cid // n_num                                  # ⑤
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)  # ⑥
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):                                 # ⑦
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)  # ⑧
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()                                # ⑨
                T.tile.add(c_ub, a_ub, b_ub)                   # ⑩
                T.barrier_all()                                # ⑪

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])  # ⑫

    return main

func = vec_add(M, N, 128, 256)                                 # ⑬

a = torch.randn(M, N).npu()                                    # ⑭
b = torch.randn(M, N).npu()
c = func(a, b)                                                 # ⑮
torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)     # ⑯
print("Kernel Output Match!")
```

逐条解读：

* **①** `tilelang.cache.clear_cache()`：清内核缓存。开发期每次改 kernel 都应该清，否则
  可能拿到上一次编译的 `.so`。**上线不要清**（缓存是启动开销的一部分）。
* **②** `@tilelang.jit(out_idx=[-1])`：把 Python 函数变成“调用即编译”的 JIT 入口。
  `out_idx=[-1]` 说“最后一个参数 `C` 是输出，由 runtime 分配并返回”。
* **③** `m_num * n_num`：把二维输出切成 `block_M × block_N` 的块，块数就是 grid 大小。
  这里**假设 M、N 能被 tile 整除**，例子不处理尾块（尾块见 §11 坑 P7）。
* **④** `T.Kernel(m_num * n_num, is_npu=True) as (cid, vid)`：**这是 A 树的昇腾写法**。
  `is_npu=True` 选择 NPU 后端；返回 `(cid, vid)`：`cid` 是逻辑核（块）编号，`vid ∈ {0,1}`
  是这一个逻辑核里的两个 **Vector 子单元**。对照 M 树：`T.Kernel(N) as bx`，**没有
  `vid`**，子核改用 `T.MixedKernel(N, sids=2)`。
* **⑤** `bx = cid // n_num; by = cid % n_num`：昇腾只有 1D 网格，所以二维坐标要自己从
  一维 `cid` 还原。这个“除法取模”是昇腾 kernel 的标志性写法。
* **⑥** `T.alloc_ub(...)`：在 **UB** 上申请缓冲，形状是 `(block_M // VEC_NUM, block_N)`
因为一块 tile 被两个 `vid` 各拿一半。**UB 是 Vector 的唯一输入输出区**
  （A 树里 `alloc_ub` 的 scope 是 `shared.ub`，见
  `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:156`）。
* **⑦** `with T.Scope("V")`：声明这段代码跑在 **Vector** 核上。A 树用 `T.Scope("C")` /
  `T.Scope("V")` 显式指定 Cube / Vector；M 树由编译器自动分配核。
* **⑧** `T.copy(GM_region, ub_buf)`：**MTE2** 的 `GM → UB` 搬运。`A[...]` 的起始下标
  `bx * block_M + vid * block_M // VEC_NUM` 正是“第 bx 块的第 vid 半”。
* **⑨** `T.barrier_all()`：**第一次同步**。它保证两次 `T.copy`（MTE2 搬运）**全部完成**
  后，才允许后面的 `T.tile.add`（Vector 计算）读 `a_ub` / `b_ub`。这就是 §2.4 里
  “SetFlag 等前序读写完成”的全屏障版本。**它是 MTE2 → V 的依赖**。
* **⑩** `T.tile.add(c_ub, a_ub, b_ub)`：Vector 的逐元素加（等价于 `c_ub = a_ub + b_ub`）。
  输入输出都在 UB。这是 Vector 单元的硬约束。
* **⑪** `T.barrier_all()`：**第二次同步**。保证 `T.tile.add`（V）**全部完成**后，才允许
  后面的 `T.copy` 把 `c_ub` 写回 GM。**它是 V → MTE3 的依赖**。
* **⑫** `T.copy(ub_buf, GM_region)`：**MTE3** 的 `UB → GM` 搬运。
* **⑬** `vec_add(M, N, 128, 256)`：给出 tile 尺寸（`block_M=128, block_N=256`），触发
  编译，返回可调用对象。
* **⑭** `.npu()`：把张量放到 NPU（`torch_npu` 注册的 device）。
* **⑮** `func(a, b)`：编译好的 kernel 直接当函数调，返回 `C`。
* **⑯** 与 PyTorch 参考实现 `a + b` 对拍，容差 `rtol=1e-2, atol=1e-2`（fp32 下这个容差
  很松，但逐元素加法本身没有累加误差，实际会远小于它）。

### 4.3 数据流与同步在代码里的位置

把它画成图（对应 `figures/02-first-kernel-dataflow.md`）：

```
                MTE2                      V                     MTE3
  GM(A) ──────────────► a_ub ┐
  GM(B) ──────────────► b_ub ┤
                              │  barrier_all()    ← ⑨：等两次 MTE2 全部完成
                              ▼
                            c_ub = a_ub + b_ub     ← ⑩：Vector 计算
                              │
                              │  barrier_all()    ← ⑪：等 Vector 全部完成
                              ▼
  GM(C) ◄────────────── c_ub                        ← ⑫：MTE3 写回
```

**三条 PIPE、两个 barrier、一条数据流**。这就是昇腾逐元素算子的全部骨架。注意两个
barrier 保护的是**不同方向**的依赖：⑨ 是 `MTE2 → V`（读前等写完），⑪ 是 `V → MTE3`
（写前等算完）。

### 4.4 第二个版本：手写双缓冲

当数据块很多、`GM → UB` 是瓶颈时，你会希望“搬下一块的同时算这一块”。
`elementwise_add_pipeline.py` 手工实现了这件事，用 `stages = 2` 个缓冲轮转：

```python
a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)   # 首维是版本维
...
@T.macro
def init_flag():
    T.set_flag("mte3", "mte2", 0)
    T.set_flag("mte3", "mte2", 1)

for mm in T.serial(vec_proc):
    cur, nxt = mm % stages, (mm + 1) % stages
    if mm < vec_proc - 1:
        T.wait_flag("mte3", "mte2", nxt)        # 等槽位 nxt 被上一轮消费完
        T.copy(A[... (mm + 1) * sub_M ...], a_ub[nxt, :, :])   # 预取下一块
        T.set_flag("mte2", "v", nxt)            # 通知 Vector：槽位 nxt 已就绪

    T.wait_flag("mte2", "v", cur)               # 等槽位 cur 的搬运完成
    for (i, j) in T.Parallel(sub_M // VEC_NUM, block_N):
        c_ub[cur, i, j] = a_ub[cur, i, j] + b_ub[cur, i, j]
    T.set_flag("v", "mte3", cur)                # 通知 MTE3：槽位 cur 算完了
    T.wait_flag("v", "mte3", cur)
    T.copy(c_ub[cur, :, :], C[... mm * sub_M ...])
    T.set_flag("mte3", "mte2", cur)             # 归还槽位 cur 给 MTE2
```

（完整代码见 `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add_pipeline.py`
与本章 `code/03-elementwise-pipeline/`。）

这里出现了三对 flag：**`mte3 → mte2`（归还槽位）**、**`mte2 → v`（数据就绪）**、
**`v → mte3`（计算完成）**。手工双缓冲的心智负担就在这：**每加一个缓冲版本，就要多管一组
“能写了吗 / 写完了吗”的 flag**。M 树把这套交给 `T.annotate_buffer_versions` + 自动
调度（§7.5），A 树则要你手写。这是“两棵树怎么选”最实际的判据之一。

### 4.5 期望输出与验证

**前提**：昇腾 NPU + CANN + torch_npu + A 树 TileLang-Ascend 安装完成。

**命令**：

```bash
cd examples/elementwise
python elementwise_add.py --m 1024 --n 1024
```

**期望输出**：

```
init successful!
Kernel Output Match!
```

**判据**：`torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)` 通过。
**验证级别**：仅静态核对（本书作者没有 NPU；上面两行输出是**样例自述的期望值**，
不是我们的实测结果）。

---

## 5. 用法（三）：第一个 GEMM kernel

### 5.1 为什么 GEMM 的最小例子只有几行核心代码

因为 A 树的 `T.gemm_v0` 已经把 `L1 → L0A/L0B` 的搬运、Cube 的 MAD、写回 L0C 全部包了。
你在例子里只能看到“搬进 L1”“调 gemm”两件事。**这恰恰是本章要你记住的重点：高级
原语隐藏了 L0 的细节，但你必须知道它藏在哪**。否则遇到容量/对齐问题时你无从下手。

### 5.2 逐行解释

核心部分（`_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm.py:31-53`）：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):     # ① 同一个 1D 网格
    bx = cid // n_num
    by = cid % n_num

    A_L1 = T.alloc_L1((block_M, K_L1), dtype)              # ② Cube 输入中转
    B_L1 = T.alloc_L1((K_L1, block_N), dtype)              #    注意是 K_L1，不是 K
    C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)    # ③ 累加器在 L0C，fp32

    with T.Scope("C"):                                     # ④ 这段跑在 Cube 核上
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):                         # ⑤ 沿 K 分块循环
            T.copy(A[bx * block_M, k * K_L1], A_L1)        # ⑥ GM→L1，MTE2
            T.copy(B[k * K_L1, by * block_N], B_L1)

            T.barrier_all()                                # ⑦ 等搬运完成
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))     # ⑧ L1→L0A/L0B→Cube→L0C

            T.barrier_all()                                # ⑨ 等 Cube 完成

        T.copy(C_L0, C[bx * block_M, by * block_N])        # ⑩ L0C→GM，FixPipe
```

逐条解读：

* **②** L1 只放 **`K_L1`** 列，不是整个 K。这意味着“沿 K 分块、反复复用同一块 L1 缓冲”
  是写进代码结构的，不是编译器的优化。
* **③** `T.alloc_L0C((block_M, block_N), accum_dtype)`，`accum_dtype="float"`（fp32）。
  **L0C 就是这块 tile 的累加器**，它在整个 K 循环里一直活着。
* **⑧** `T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))`：
  * 签名是 `gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False, kL0Size=128, n_actual=None)`
    （`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:418`）；
  * docstring 明确写 **A、B 必须在 L1（`T.alloc_L1`），C 必须在 L0C（`T.alloc_L0C`）**，
    `gemm_v0` 内部负责 `L1 → L0A/L0B` 的搬运；
  * `init=(k == 0)`：**第一次迭代清零累加器，之后累加**。写错这一位就会出现“结果里
    混进了上一次调用的残值”这种最难查的错误（见 §11 坑 P9）。
* **⑦ / ⑨** 两次 `barrier_all()`：⑦ 是 `MTE2 → MTE1/M`（L1 数据就绪），⑨ 是 `M → MTE2`
  （L1 可以安全地被下一轮覆盖）。注意这里用的是全屏障，代价是**没有流水重叠**。这就是
  为什么它是最小例子而不是高性能例子。

### 5.3 GM→L1→L0A/L0B→Cube→L0C→GM 的完整数据流

`T.gemm_v0` 这一行展开了是多少件事？对照 §2.2 与 §2.3：

```
  GM(A), GM(B)                                   GM(C)
     │  MTE2: DataCopy                             ▲
     ▼                                             │ FixPipe / DataCopy
   L1(512KB)                                       │
     │  MTE1: LoadData  (由 gemm_v0 内部发出)      │
     ▼                                             │
  L0A(64KB) / L0B(64KB)                            │
     │  Cube: MAD                                  │
     ▼                                             │
  L0C(128KB, fp32 累加) ───────────────────────────┘
```

对应的 TileLang / Ascend C 指令与 PIPE：

| 步骤 | PIPE | TileLang（A 树） | Ascend C |
| --- | --- | --- | --- |
| GM → L1 | MTE2 | `T.copy(A[...], A_L1)` | `DataCopy(a1Local, aGm[...], ...)` |
| L1 → L0A/L0B | MTE1 | 隐藏在 `T.gemm_v0` 内 | `LoadData(a2Local, a1Local, ...)` |
| Cube MAD | M | 隐藏在 `T.gemm_v0` 内 | `Mmad(c0Local, a2Local, b2Local, ...)` |
| L0C → GM | FixPipe | `T.copy(C_L0, C[...])` | `Fixpipe(cGm[...], c0Local, ...)` 或 `DataCopy` |

### 5.4 昇腾 GEMM 的三个硬约束

1. **A、B 在 L1，C 在 L0C**（`gemm_v0` 的 docstring 与 M 树 SKILL 的 §5.1 都写了）。
   把 C 建在 UB 或 L1 上会编译失败或静默产生错误结果。
2. **L1 输入路径的转置约定**：M 树当前的 L1 输入 lowering 要求
   `transpose_A=False`、`transpose_B=True`（`SKILL.md` §5.1），权重通常按 `[N, K]` 存；
   A 树的 `example_gemm.py` 用的是 `T.Tensor((K, N))` + 默认 `transpose_B=False`。**两棵树
   的默认布局约定不同**，跨界抄代码必错。
3. **tile 尺寸受每级容量约束**：`block_M×K_L1` 的 fp16 元素要放进 L1（512 KB），
   `block_M×block_N` 的 fp32 累加要放进 L0C（128 KB）。128×256 的 fp32 累加是 128 KB，
   **刚好等于 L0C 的容量上限**。所以官方示例用 `block_N=256` 时基本没有再往上加的余地。
   需要更大 `block_N` 就得靠 `kL0Size` 调整 L0 的双缓冲（`gemm_v0` 的 `kL0Size` 参数
   docstring 有说明：half 精度、`block_M=128, block_N=256` 时推荐 `kL0Size=64`）。

### 5.5 期望输出与验证

**前提**：同 §4.5，且 M、N 能被 128/256 整除、K 能被 64 整除。

**命令**：

```bash
cd examples/gemm
python example_gemm.py --m 1024 --n 1024 --k 1024
```

**期望输出**：

```
init successful!
Kernel Output Match!
```

**判据**：`torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)` 通过。
**验证级别**：仅静态核对（示例自述的期望值，非本书实测）。

---

## 6. Ascend C 对照：同一个算子，手写要做什么

这一节回答“TileLang 到底省了什么”。我们拿**同一个逐元素 add**和**同一个 GEMM**
分别看 Ascend C 怎么写。

### 6.1 Ascend C 的最小向量 add

下面这段来自昇腾官方文档的 Ascend C 示例（inplace 版本，
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_00020_html.md`），
稍作整理：

```cpp
namespace AscendC {
class MyKernel {
public:
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);       // ① 绑定 GM 指针
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(srcQue0, 1, BLOCK_SIZE * sizeof(half)); // ② 手工建队列（含 UB 空间）
        pipe.InitBuffer(srcQue1, 1, BLOCK_SIZE * sizeof(half));
        pipe.InitBuffer(dstQue0, 1, BLOCK_SIZE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < REPTIMES; i++) { CopyIn(i); Compute(i); CopyOut(i); }
    }

private:
    __aicore__ inline void CopyIn(int32_t i)
    {
        srcQue0.AllocTensor<half>(src0Local);                   // ③ 取出一块 UB 空间
        DataCopy(src0Local, src0Global[i*BLOCK_SIZE], BLOCK_SIZE);  // ④ GM→UB
        srcQue0.EnQue(src0Local);                               // ⑤ 入队（隐含同步）
    }
    __aicore__ inline void Compute(int32_t i)
    {
        srcQue0.DeQue<half>(src0Local);                         // ⑥ 出队（等数据到）
        dstQue0.AllocTensor<half>(dstLocal);
        Add(dstLocal, src0Local, src1Local, BLOCK_SIZE);        // ⑦ Vector 计算
        dstQue0.EnQue<half>(dstLocal);
        srcQue0.FreeTensor(src0Local);                          // ⑧ 释放
    }
    __aicore__ inline void CopyOut(int32_t i)
    {
        dstQue0.DeQue<half>(dstLocal);
        DataCopy(dstGlobal[i*BLOCK_SIZE], dstLocal, BLOCK_SIZE);// ⑨ UB→GM
        dstQue0.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;                                                 // ⑩ 全局流水对象
    TQue<QuePosition::VECIN, 0> srcQue0, srcQue1;
    TQue<QuePosition::VECOUT, 0> dstQue0;
    GlobalTensor<half> src0Global, src1Global, dstGlobal;
    LocalTensor<half> src0Local, src1Local, dstLocal;
};
}
```

把这段和 §4 的 TileLang 逐项对照：

| 你要做的事 | Ascend C | TileLang（A 树） |
| --- | --- | --- |
| 声明 GM 参数 | `SetGlobalBuffer`（①） | `T.Tensor(...)` 参数 |
| 在 UB 上申请空间 | `pipe.InitBuffer` + `AllocTensor`（②③） | `T.alloc_ub(...)` |
| GM→UB 搬运 | `DataCopy`（④） | `T.copy(...)` |
| “数据到了才计算” | `EnQue`/`DeQue`（⑤⑥） | `T.barrier_all()` |
| 计算 | `Add(...)`（⑦） | `T.tile.add(...)` |
| 释放缓冲 | `FreeTensor`（⑧） | 不需要（编译器做内存规划） |
| UB→GM 写回 | `DataCopy`（⑨） | `T.copy(...)` |
| 流水对象与队列 | 手工 `TPipe`/`TQue`（⑩） | 不需要 |
| 打核间/流水同步 | 手工 `SetFlag`/`WaitFlag` | `barrier_all`（A 树）/ 自动（M 树） |

**TileLang 省掉的正是“缓冲生命周期管理 + 队列 + 同步配对”这三件事。** 注意 `EnQue` 与
`DeQue` 不只是“入队出队”，官方文档明确写它们“实现了内存读写同步功能”（inplace 版本
仍然保留它们就是为了这个），这正是 TileLang 里 `barrier_all()` 或自动同步对应的东西。

### 6.2 Ascend C 的 GEMM：CATLASS `00_basic_matmul`

手写一个昇腾 GEMM，即使借助华为的模板库 CATLASS，也要写这么多“非算法”的东西
（`../tilelang-corpus/06-huawei-ascend/raw/catlass/examples/00_basic_matmul/basic_matmul.cpp`）：

```cpp
using ArchTag        = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;   // ① 选流水策略（pingpong）
using L1TileShape    = GemmShape<128, 256, 256>;          // ② L1 的 tile 形状
using L0TileShape    = GemmShape<128, 256, 64>;           // ③ L0 的 tile 形状

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;  // ④ swizzle
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// ⑤ 问硬件要核数
auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
```

以及 host 侧一大段 `aclrtMalloc` / `aclrtMemcpy` / `aclrtCreateStream` / workspace 计算 /
`aclrtSynchronizeStream`。这部分 TileLang 交给 `torch_npu` 与 runtime 处理。

这段代码里有三类**只有懂硬件才写得对**的决定（对应 §1.1 的三层义务）：

1. **流水策略**（`MmadAtlasA2Pingpong<true>`）：要不要 ping-pong 双缓冲。
2. **L1 / L0 的 tile 形状**（②③）：两级的 tile 是**分开指定**的，L1 是 128×256×256，
   L0 是 128×256×64。也就是说，**L1→L0 还要再按 K 切一次**。
3. **Block swizzle**（④）：`GemmIdentityBlockSwizzle<3, 0>` 决定 block 在 M/N 上的遍历
   顺序（`<offset, direction>`，与 CATLASS swizzle 文档里“M≥N 用 offset=3/direction=0”
   的判据一致）。

### 6.3 TileLang 省掉了什么（对照表）

`tilelang/ascend/README.md` 自带一张“你写什么 / Ascend C 你可以跳过什么 / 编译器做什么”
的对照表（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:144-152`），整理如下：

| 领域 | 你在 TileLang 里写 | 你可以跳过的 Ascend C | 编译器做什么 |
| --- | --- | --- | --- |
| 缓冲与数据搬运 | `T.alloc_*`、`T.copy`、`T.dual_copy` | `TPipe`/`TQue` 的建立、DMA 调用 | 规划存储；下沉拷贝与支持的 layout 转换 |
| Tiling 与 tile 算子 | 形状/dtype、tile 尺寸、块数、`T.gemm` 等 | 底层算子实现 | 把 tile 算子下沉到硬件指令 |
| SIMT 计算 | `T.SimtVF(threads=...)` 里的标量代码与 `T.Parallel` | 线程映射与 barrier 放置 | 把迭代映射到线程；为检测到的跨线程竞争插 barrier |
| SIMD 计算 | `T.SimdVF()` 里的 `T.simd.*` 与掩码 | device function 的样板代码 | 把 SIMD 算子下沉成 device function |
| 调度与流水 | 按需写 `T.Pipelined(...)` | 手工调度与缓冲槽位轮转 | 给 AIC/AIV 分配任务；调度重叠与多缓冲 |
| 同步 | **什么都不写（按顺序写操作即可）** | Set/Wait flag 与 flag ID 管理 | 推导流水与 Cube/Vector 依赖；插入成对 flag |

同一份文档还补了一句重要的**边界**：自动调度只负责“任务顺序与依赖”，**不意味着**它会把
任意 `T.Parallel` 代码自动张量化成 SIMD 指令（`.../ascend/README.md:153-155`）。

### 6.4 什么时候仍然应该写 Ascend C

TileLang 不是万能的。下面几类情况仍然值得（或必须）直接写 Ascend C：

1. **要用 TileLang 尚未封装的新硬件指令或新算子**（例如某个新代际的专用指令）。
2. **要做极端的、非常规的片上内存复用或 bank 冲突规避**，编译器的手工调度给不出你要的
   排布。
3. **需要与既有 Ascend C 算子库做零拷贝集成**（比如在一个已有 `TPipe` 环境里插一段）。
4. **目标设备/SDK 组合尚未被 TileLang 后端覆盖**（A 树自述只验证了 A2/A3；M 树只针对
   Ascend 950）。

反过来，**只要你要的是“把 tile 级的数据流写清楚、让编译器管同步”**，TileLang 就是
更省事、更不容易错的那条路。

---

## 7. 昇腾专属原语与“用哪棵树”

这一节是本章的技术核心。所有 API 名都在两棵树的源码里逐条核对过。

### 7.1 两棵树的 API 对照总表

| 能力 | A 树 `_vendor/tilelang-ascend-v0.1.2.000/` | M 树 `_vendor/tilelang-v0.1.15/` |
| --- | --- | --- |
| 前端导入 | `import tilelang.language as T` | `import tilelang.ascend.language as T` |
| JIT target | 默认；`target="ascendc"` / `target="pto"` | `target="ascend"`（默认 `dav-3510`） |
| UB 分配 | `T.alloc_ub(shape, dtype)`（scope `shared.ub`） | `T.alloc_shared(shape, dtype)`（scope `shared.dyn`，即 UB） |
| L1 / L0A / L0B / L0C | `T.alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C`（大写，`allocate.py:140-153`） | `T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c`（小写，`allocate.py:26-60`） |
| 显式矩阵乘 | `T.gemm_v0(A, B, C, transpose_A, transpose_B, init, kL0Size, n_actual)`（`ascend.py:418`） | `T.gemm(A, B, C, transpose_A, transpose_B, policy, clear_accum, unit_flag_ctrl)`（`gemm_op.py:168`） |
| 双 AIV 分裂拷贝 | 无统一原语（手写 `vid` 切分） | `T.dual_copy(src, dst, unit_flag_ctrl=..., l2_cache_ctrl=...)`（`copy_op.py:53`） |
| SIMT / SIMD 区 | 无（用 `T.Scope("V")` + `T.Parallel`） | `T.SimtVF(threads=...)` / `T.SimdVF()`（`frame.py:107,179`） |
| 执行域 | `T.Scope("C")` / `T.Scope("V")` | `T.Cube()` / `T.Vector()`；纯 kernel 自动判定 |
| 子核 ID | `T.Kernel(..., is_npu=True) as (cid, vid)` | `T.MixedKernel(N, sids=2) as (bx, sid)` |
| 流水 | 手写 flag 或 `T.Pipelined` | `T.Pipelined` + AutoSchedule |
| 多缓冲版本注解 | 无 | `T.annotate_buffer_versions({buf: count 或 (count, mode)})`（`annotations.py:19`） |
| 内存容量注解 | 无 | `T.annotate_unlimit_memory("shared", "shared.l1", ...)` |
| 设备打印 | `T.printf(...)` / `T.dump_tensor(...)`（`ascend.py:534,582`） | `T.print(...)`（`print.py`）；无 `dump_tensor` |
| 设备断言 | — | `T.device_assert(cond, msg)` |
| 同步 | `T.barrier_all` / `set_flag` / `wait_flag` / `set_cross_flag` / `wait_cross_flag` | 自动插入；手工可用 `T.ascend_pipe_barrier` / `T.ascend_set_flag` / `T.ascend_wait_flag` |
| 模式开关 | `pass_configs`（`TL_ASCEND_AUTO_SYNC` 等，见 `tilelang/transform/pass_config.py:38-62`） | `TL_ENABLE_AUTO_SCHEDULE`（默认 True）；**没有** `TL_ASCEND_*` |
| 精度/原子模式 | — | `T.set_hf32_mode(...)`、`T.set_atomic("add", dtype)` / `T.set_atomic_none()`（`mode.py`） |
| Benchmark 后端 | `do_bench(..., backend="msprof")` | `do_bench(..., backend="msprof"/"msprof_detail")` |

**这张表最重要的一行是倒数第二行**：`TL_ASCEND_AUTO_SYNC`、`TL_ASCEND_MEMORY_PLANNING`、
`TL_ASCEND_AUTO_CV_SYNC`、`TL_ASCEND_AUTO_CV_COMBINE`、`TL_ASCEND_EXCEPTION_DUMP`
这些 `PassConfigKey` **只存在于 A 树**（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:38-62`）。
在 M 树里搜 `TL_ASCEND` 只能搜到 C++ 头文件的宏保护符，搜不到这些 Python 配置键。
把 A 树的 `pass_configs` 抄到 M 树，轻则无效，重则因为未知键而报错。

### 7.2 M 树方言：`T.alloc_l1/l0a/l0b/l0c`

```python
import tilelang.ascend.language as T

A_l1  = T.alloc_l1((block_M, block_K), T.bfloat16)     # scope "shared.l1"
B_l1  = T.alloc_l1((block_N, block_K), T.bfloat16)
A_l0a = T.alloc_l0a((block_M, block_K), T.bfloat16)    # scope "shared.l0a"
B_l0b = T.alloc_l0b((block_K, block_N), T.bfloat16)    # scope "shared.l0b"
C_l0c = T.alloc_l0c((block_M, block_N), T.float32)     # scope "shared.l0c"，自动带 L0C 布局
tile  = T.alloc_shared((block_M, block_N), T.float32)  # UB（scope "shared.dyn"）
```

四个函数的 default scope 分别是 `shared.l1` / `shared.l0a` / `shared.l0b` / `shared.l0c`
（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:26,37,46,55`）。其中
`alloc_l0c` 会**自动给 buffer 打上 Ascend L0C 累加器布局**（源码里 `make_ascend_l0c_layout`），
所以你不用自己写 layout 注解；`alloc_l0a`/`alloc_l0b` 的布局**延后到消费它的 GEMM**，
由 GEMM 根据 transpose 标志推断 K-major 还是 MN-major。

> 注意：`T.alloc_shared` 在 M 树昇腾方言里被**刻意覆盖**成“分配 UB”，而不是 CUDA 的
> shared memory（`tilelang/ascend/language/__init__.py` 的注释写明“Ascend UB takes bool
> buffers in the requested scope, without CUDA's static-scope hack”）。

### 7.3 `T.dual_copy`：一次拷贝劈成两个 AIV

`T.dual_copy` 是 M 树独有的、把一块逻辑 tile 分给**两个 AIV 子核**的拷贝原语
（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53`）。规则（docstring 原文）：

* 它必须在**混合 kernel** 里用。你还要在同一个 kernel 里提供 **Cube 侧的工作**；
  `dual_copy` 本身不会把纯 Vector kernel 变成两 AIV 启动。
* 形状推断：对 rank ≥ 2 的区域，**恰有一个尾部维度是 2:1**：
  * `[M, N] → [M/2, N]`：**M-split**（`dual_dst_ctl=0b01`）；
  * `[M, N] → [M, N/2]`：**N-split**（`dual_dst_ctl=0b10`）。
* 支持的通路：`L0C→UB` 走**硬件双目的**拷贝；`GM→UB`、`UB→GM`、`UB→L1` 走
  **按 `cthread` 索引的逐 AIV 软件拷贝**（`UB→L1` 还会顺带做 ND→NZ 格式转换）。
* 静态已知的整体 extent 必须是**偶数**；奇数直接报错，不会凑合分配。

典型用法（M 树 `tilelang/ascend/README.md:94-98` 的 GEMM+ReLU）：

```python
T.dual_copy(C_l0c, C_ub)                                  # L0C → UB，M-split
with T.SimtVF(threads=128):
    for i, j in T.Parallel(block_M // 2, block_N):
        C_ub[i, j] = T.max(C_ub[i, j], 0)                  # Vector 做 ReLU
T.dual_copy(C_ub, C[m:m+block_M, n:n+block_N], l2_cache_ctrl="NOTALLOC_PW")  # UB → GM
```

### 7.4 `T.SimtVF` / `T.SimdVF`：两种 Vector 编程模型

Ascend 950 的 Vector 侧支持两种写法（`frame.py:107,179`）：

* **`T.SimtVF(threads=N)`**：SIMT 风格的线程域。在它内部用 `T.Parallel` 并行化，
  编译器把迭代映射到线程，并**自动插入 `asc_syncthreads()`** 来处理检测到的跨线程竞争。
  可以用 `T.get_thread_binding()` 取线程号。
* **`T.SimdVF()`**：**寄存器级 SIMD**，没有线程概念。向量宽度是 2048 位
  （64×float32 或 128×float16）。用 `T.simd.*` 写显式指令：

```python
with T.SimdVF():
    mask = T.simd.pset(32)                       # 32 位元素全 lane 有效
    for i in range(tile // 64):
        a_reg = T.simd.vld(a[i * 64])            # 2048-bit 向量加载
        b_reg = T.simd.vld(b[i * 64])
        res   = T.simd.vadd(a_reg, b_reg, mask)
        T.simd.vsts(c[i * 64], res, mask)
```

两者都接受 `latency=`（周期数）作为调度模型的输入；不填则由编译器估算
（`SKILL.md` §8.3）。**不要**把它和“实测周期”混为一谈。

### 7.5 `T.Pipelined` 与 `T.annotate_buffer_versions`

`T.Pipelined(iters, num_stages=k)` 表达“这个循环要做软件流水”（语义见第 11 章）。
在昇腾侧有两条必须记住的约束（A 树 `docs/tutorials/t_pipelined.md:119-180`）：

1. **`T.Pipelined` 不支持嵌套**；
2. **核内流水与核间流水不能同时开**，要混用时改用“flat pattern”：`T.Pipelined` 管核间
   （Cube↔Vector 的 workspace 交接），核内用 `T.serial` + 手写 flag 管。

`T.annotate_buffer_versions` 是 M 树 AutoSchedule 的**版本控制旋钮**
（`annotations.py:19`）。取值语义：

| 取值 | 含义 |
| --- | --- |
| `1` | 退出多缓冲资格（连显式 owner 声明也一并覆盖），保留普通依赖 |
| `≥2` | 固定物理版本数，索引模式自动 |
| `(n, mode)` | 固定版本数 + 固定索引模式；`(1, "auto")` = 保留资格但只有一个物理槽 |
| `"auto"` / `"iteration"` / `"counter"` | 只选模式，版本数交给 Z3 调度器 |

三种模式的差别（`SKILL.md` §4.2 与 `references/synchronization.md`）：

* **`iteration`**：用**扁平化后的词法循环下标**选槽位。即使某次迭代被 guard 跳过，槽位
  时钟照走。
* **`counter`**：用一个**单调计数器**，只在“一次完整的活跃 buffer epoch 完成”后推进。
  被 guard 跳过的迭代不推进计数器。
* **`auto`**：规则（无条件、单 owner）时走 `iteration` 快路径；条件执行/不规则/多 owner
  时自动改用 `counter`。

一个常见用法（来自 `_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py`）：

```python
a = T.alloc_shared((tile,), "float32")
b = T.alloc_shared((tile,), "float32")
c = T.alloc_shared((tile,), "float32")
T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})  # 固定版本数
for step in T.Pipelined(iterations, num_stages=num_stages):
    ...
```

**这份版本注解和 A 树的“手写 `buf[stages, ...]` 首维”是两件事**：M 树里编译器按注解
自己给 buffer 加版本维，你**不要**再手动加一个 stage 维，否则等于要求双缓冲两次
（`references/synchronization.md` 明确警告）。

### 7.6 选择规则：什么时候用哪棵树

```
你的目标设备是什么？
├─ Ascend 950 / 想要自动调度 / 想用 dual_copy + SimtVF 混核
│     → M 树（tilelang.ascend.language，target="ascend"）
├─ A2 / A3，或要跟现有 tilelang-ascend 算子和 Ascend C 生态对接
│     → A 树（tilelang.language，target="ascendc" 或 "pto"）
├─ 只想在无 NPU 的机器上做功能验证（PTO 后端）
│     → A 树 + target="pto" + camodel 仿真
└─ 要读某篇昇腾调优文章的代码
      → 先确认那篇文章用的是哪棵树，再照抄
```

两条**硬性纪律**：

1. **一份 kernel 内部不要混用两棵树的 API**。`T.alloc_L1`（A）和 `T.alloc_l1`（M）看着像
   同一个东西，实际来自两套命名空间。
2. **版本引用要落到具体树**。本章所有引用都标了 A 树或 M 树的路径；你自己的代码注释
   也应该这样写。

---

## 8. 调优：msProf 与瓶颈判读

### 8.1 msProf 的两种用法

A 树编程指南第 6 节（`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:2527-2563`）
把 msProf 分成两种用法：

**（a）`msprof op`：上板，看真实运行**

```bash
msprof op --kernel-name="your_kernel_func_name" python your_kernel_script.py
```

它输出：**计算内存热力图、Roofline 瓶颈分析图、Cache 热力图、通算流水图、算子代码热点图**。
适合在真实环境里快速定位“内存还是计算卡住了”。

**（b）`msprof op simulator`：仿真，看指令级细节**

```bash
msprof op simulator --soc-version=<ascend_version> --kernel-name="your_kernel_func_name" python your_kernel_script.py
```

它输出：**指令流水图、算子代码热点图、内存通路吞吐率波形图**。适合开发/调试阶段分析
指令与代码热点。

**前提**：CANN 已安装、msProf 可用、kernel 能在目标设备上跑（simulator 还要 `-g` 之类的
编译选项与环境变量）。
**期望输出**：在 msProf 的落盘目录里生成上述图表；屏幕上有采集完成提示。
**验证级别**：仅静态核对（未运行；具体输出目录结构与图形名称随 CANN 版本变化，
以你机器上的实际输出为准）。

### 8.2 TileLang 侧的入口：`do_bench(backend="msprof")`

除了直接调 `msprof`，M 树把 msProf 接进了 `do_bench`：

```python
from tilelang.profiler import do_bench

# 只取一个延迟数字（毫秒）
latency_ms = do_bench(lambda: kernel(a, b), backend="msprof",
                      _n_warmup=30, _n_repeat=100)

# 取完整的 per-kernel 画像：时长 + AIC/AIV 各 pipe 占用
profile = do_bench(lambda: kernel(a, b), backend="msprof_detail")
print(profile.dur_us, profile.tflops(2 * M * N * K), profile.gbps(bytes_moved))
```

`do_bench` 的合法 backend 是 `"event" / "cupti" / "cudagraph" / "wall" / "msprof" / "msprof_detail"`
（`_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31-66`）；在 NPU 上只能用
`"event"`、`"msprof"`、`"wall"`（源码里显式拒绝 cupti/cudagraph：`“timing requires
CUDA/HIP; use 'event', 'msprof', or 'wall' for NPU”`）。

`msprof_detail` 返回的 `KernelProfile` 字段（`tilelang/profiler/msprof.py:60-134`）正是瓶颈
判读要用的东西：

| 分组 | 字段 | 含义 |
| --- | --- | --- |
| AIC（Cube 核） | `aic_mad` | **Cube/MAD 利用率** |
| | `aic_scalar` | 标量流水占用 |
| | `aic_mte1` | L1→L0 搬运占用 |
| | `aic_mte2` | GM→L1 搬运占用 |
| | `aic_mte3` | AIC 侧 MTE3 占用 |
| | `aic_fixpipe` | FixPipe（L0C 写出）占用 |
| AIV（Vector 核） | `aiv_vec` | **Vector 利用率** |
| | `aiv_scalar` | 标量流水占用 |
| | `aiv_mte2` | GM→UB 搬运占用 |
| | `aiv_mte3` | UB→GM 搬运占用 |

注意两点（`references/profiling.md`）：**这些比值是相对各自 total_cycles 的，不要把
AIC 与 AIV 的比值加在一起当成一个统一时间轴的占比**；`do_bench(backend="msprof")`
本身不带 median/quantile 选择，要多取几次自己求中位数。

### 8.3 瓶颈判读：MTE 带宽 / Cube 利用率 / Vector

拿到 `KernelProfile` 后，按下面的顺序读（`figures/05-tuning-decision.md`）：

```
先看 dur_ns 与算术强度，是 memory-bound 还是 compute-bound？（第 2 章的账要重算一遍）
        │
   ┌────┴─────────────────────────────────────────────┐
   ▼                                                  ▼
 compute-bound（GEMM 类）                      memory-bound（逐元素/量化类）
   看 aic_mad 高不高                             看 aiv_mte2 / aiv_mte3 高不高
   ├─ aic_mad 高，dur 也高  → Cube 已经打满                 ├─ aiv_mte2 高 → GM→UB 带宽吃满
   │      → 只能靠更优算法/tile/dtype，别再叠流水            │      → 减少搬运量（融合/量化/复用）
   └─ aic_mad 低  → Cube 在等数据                           └─ aiv_mte2 低但 dur 高 → 可能在等同步
         ├─ aic_mte2 高 → GM→L1 带宽是瓶颈                      → 检查 flag 配对/多缓冲
         │      → L2 命中提示、swizzle、减少重复读
         ├─ aic_mte1 高 → L1→L0 太频繁 → 调 kL0Size / L0 tile
         └─ aic_fixpipe 高 → L0C 写回太慢 → 调 L0C 使用方式/写回策略
```

三个具体的判据（都来自 `KernelProfile` 字段，不涉及任何实测数字）：

* **Cube 忙不忙**：`aic_mad` 接近本代上限而 `dur` 仍长 → 计算是瓶颈，优化方向是算法
  （split-K、改变 tile、换 dtype），而不是调流水。
* **Cube 在等谁**：`aic_mad` 低、`aic_mte2` 高 → 等 GM→L1；`aic_mte1` 高 → L1→L0 太碎；
  `aic_fixpipe` 高 → L0C 写回。
* **Vector 侧**：`aiv_mte2`/`aiv_mte3` 高而 `aiv_vec` 低 → 纯搬运受限，应该做**算子融合**
  把多次 GM 往返合并成一次（第 13 章讲过这套账）。

**前提**：有 NPU 且有 M 树 TileLang。**期望输出**：上面各个比值。
**验证级别**：仅静态核对（`KernelProfile` 的字段来自源码；具体数值必须你自己在现场采）。

### 8.4 多缓冲的调整

多缓冲的目的是**用片上存储换流水重叠**。可调的三处：

1. **`T.Pipelined(..., num_stages=k)`**：k 越大，重叠越深，但需要的片上存储也越多。
   起始值建议 2，然后以 1 为步长试探，并**每次检查片上容量**：
   * GEMM：L1 上同时存活 `k` 份 `A_L1`/`B_L1`；
   * 向量：UB 上同时存活 `k` 份输入/输出。
2. **`T.annotate_buffer_versions({buf: (n, mode)})`**：当某个关键 buffer 的版本数需要被
   强制固定（例如它就是慢，需要保证双缓冲），用 `(2, "counter")` 之类锁死；当某个 buffer
   承载跨迭代的“进位”值（必须保留上一轮），用 `{buf: 1}` 退出多缓冲。
3. **`T.annotate_unlimit_memory("shared", "shared.l1")`**：在**调查阶段**临时移除调度器
   的容量约束，看是不是容量把版本数卡住了。**它不是扩容魔法**，取消约束后如果物理分配
   超了，kernel 会启动失败。用完要撤掉。

一个实用流程（`references/synchronization.md` 的“Tune performance with T.Stage”）：
**先保住一个正确的自动调度基线并记录延迟 → 只改一件事 → 看生成代码确认改动真的生效 →
在相同 harness 上对拍 → 只有可重复的设备时间提升才采纳**。注意：自动调度器解出来的
“更小的 II”只是个建模值，**不等于**设备上的实测性能（见 `SKILL.md` §12.3）。

### 8.5 core ratio 的调整

“core ratio”是 Cube 与 Vector 的配比（AIC:AIV）。昇腾的 AI Core 是 1:N 组合，n 由硬件
平台信息决定。TileLang 提供两个层次的调节：

1. **让编译器配比（推荐）**：M 树默认设定 `threads` 参数（只允许 1 或 2）时，
   **vid 消除 + Auto CV Ratio** 会自动把 UB 申请与搬运按核排布处理好，你的写法可以从
   “`alloc_shared((block_M // 2, block_N))` + `vid` 偏移”简化为
   “`alloc_shared((block_M, block_N))` + 不带 `vid` 的 `T.copy`”
   （A 树 README 的 “Vid reduction & Auto CV Ratio”，以及
   `docs/tutorials/vid_reduction_and_auto_cv_ratio.md`）。
2. **显式指定子核数**：M 树用 `T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid)` 显式控制每个
   block 用几个 AIV（`1` 或 `2`），并在代码里按 `sid` 切分输出。`SKILL.md` §3.2 特别提醒：
   **按 buffer 的实际 extent 去切**，一个“半宽”的本地 buffer 不能再被切一次。

**判据**：如果 `aiv_vec` 很低、`aic_mad` 也很低，而 `dur` 高，先怀疑核配比/负载不均；
如果 `aiv_vec` 高但 `aiv_mte2/3` 低，说明 Vector 已经打满，这时**再加 AIV 没用**，
该减少 Vector 工作量或改用 SIMD。

### 8.6 一个可复用的调优流程

本章 `code/08-msprof-bench/bench_msprof.py` 给了一个最小可改的骨架；`code/12-tuning-sweep/`
给了一个纯 Python 的配置扫描驱动（先把“测什么、比什么、怎么记”定下来，再上机器跑）。
原则来自 `SKILL.md` §12：

1. 先固定**可调用边界**（只测 kernel launch，还是含分配/初始化/多次 kernel）；
2. 记录**完整的实验条件**：CANN/bisheng/torch_npu 版本、设备型号、shape、dtype、stride、
   tile 尺寸、启动几何、pipeline 深度、cache 策略、pass config；
3. 先验正确性，再测性能；预热后再测；重复多次取稳定值；
4. 一次只改一个变量，A/B 交叉、必要时反向顺序；
5. 保留原始样本与结论依据。

---

## 9. 入图与部署：ACLGraph 与 PyTorch 集成

### 9.1 为什么要入图

当你要连续跑一串小算子时（例如 `RMSNorm → RoPE`），每次 launch 都要走一遍
Python + 框架 + runtime 的下发路径，**Host 侧开销会盖过设备侧计算时间**。这叫
Host-bound。ACLGraph 的解法是把“下发即执行”改成 **capture + replay 两阶段**：

* **Capture**：首次执行时，把一串 NPU 操作记录成一张静态的计算图；
* **Replay**：之后一次调用重放整张图，绕过 CPU 侧的 Python 与框架调度。

A 树编程指南明确写了它的**边界**：aclGraph“对 HostBound 场景做运行时加速，**不承载图
编译和图优化功能**”（`docs/TileLang-Ascend Programming Guide.md:2564-2587`）。

### 9.2 ACLGraph 的捕获与重放

三步走（`_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/README.md`）：

```python
# ① 编译选项：打开自动同步 / 内存规划 / 核间同步 / 核间合并
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

# ② 用 @tilelang.jit 定义算子（RMSNorm 有 out_idx；RoPE 是 in-place，没有 out_idx）
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def rms_norm_kernel(M, head_dim, block_M, eps, dtype="float16"): ...

# ③ 捕获 + 重放
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    q = tilelang_rms_norm(q, variance_epsilon)   # 只记录，不执行
    q = tilelang_apply_rope(q, sin, cos)
g.replay()                                       # 这里才真正执行；可反复调用
```

**一句话记住**：`with torch.npu.graph(g):` 里的算子**只是被记录**，`g.replay()` 才执行。

**如何确认入图生效**：看 Host 侧编译日志里有没有 `capture model` 关键字，
值为 `0`（默认，未启用）/ `1`（已启用）；需要先设

```bash
export ASCEND_HOST_LOG_FILE_NUM=1000
export ASCEND_PROCESS_LOG_PATH=logs
export ASCEND_GLOBAL_LOG_LEVEL=1
```

**前提**：昇腾 NPU + torch_npu + A 树 TileLang-Ascend。**命令**：

```bash
cd examples/aclgraph
python rms_rope_aclgraph.py
```
**期望输出**：与 PyTorch 参考实现对拍通过（`torch.testing.assert_close(..., rtol=1e-2, atol=1e-2)`）；
日志中出现 `capture model=1`。
**验证级别**：仅静态核对。

### 9.3 `torch_tl_ascend`：把 TileLang 算子接进 PyTorch

A 树的 `examples/torch_tl_ascend/` 是一个**以 Python C 扩展形式**把 TileLang-Ascend 算子
接进 PyTorch 的示例工程（目前示例算子是 `flash_attention`）。

```bash
python setup.py install        # 或 python setup.py bdist_wheel
python test_torch.py           # 测 torch.ops.tl_ascend.flash_attention
python test_source.py          # 测 packaged source 版本
```

两种调用方式：

```python
# 方式一：注册成 torch 算子的形式
import torch, torch_tl_ascend
output = torch.ops.tl_ascend.flash_attention(q, k, v)

# 方式二：直接调打包进包的 TileLang 源码算子
from torch_tl_ascend.op_source.flash_attn_bhsd import flash_attention_fwd
kernel = flash_attention_fwd(B, S, H, D)
output = kernel(q, k, v)
```

目录里还带一个 **libtorch C++ 集成示例**（`demo_libtorch/`），演示在没有 Python 的应用里
怎么调（`examples/torch_tl_ascend/README.md`）。

> 另外一条更轻的路是直接用 `torch_npu` 当“张量容器”：JIT 出来的 kernel 收 `.npu()`
> 张量、返回 `.npu()` 张量，就能和 `torch.nn.Module` 里的其它算子混排（本章所有例子都是
> 这样写的）。真正的“集成”问题出现在你需要 **autograd、自定义 backward、或者
> torch.compile 图内**的时候。那时才需要 `torch_tl_ascend` 这类桥。

### 9.4 集成检查清单

| 检查项 | 为什么 |
| --- | --- |
| 设备上下文一致 | kernel 与张量必须在同一个 `torch.npu` 设备上 |
| dtype / 布局一致 | 昇腾很多路径对 layout 有硬约束（如 `transpose_B=True`） |
| 输出语义 | in-place 算子**不能**设 `out_idx`；有输出才设 |
| 同步点 | 入图（ACLGraph）里不要夹带 host 侧同步；否则 replay 收益消失 |
| workspace | 需要多核协作的算子用 `workspace_idx=[...]` 声明自动管理的 GM workspace |
| 首次编译开销 | 第一次调用含 JIT 编译；基准测试要把它排除 |

---

## 10. 调试：`T.printf` / `T.dump_tensor` 与 AICore exception dump

### 10.1 `T.printf`

A 树的设备端打印（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534`）：

```python
def printf(format_str: str, *args)     # %d/%i 十进制、%f 浮点、%x 十六进制、%s 字符串、%p 指针
```

例：`T.printf("cid:%d vid:%d\n", cid, vid)`。

**关键前提**：要让打印在运行期真正输出，必须**在编译时开启调试打印**。A 树文档给的推荐做法
是 **kernel 级 `compile_flags`**：

```python
@tilelang.jit(target="ascendc", compile_flags=["-D_DEBUG", "--cce-enable-print"])
def debug_kernel(...): ...
```

老写法 `TL_PTO_DEBUG=1` 现在**只对 `target="pto"` 是兼容的进程级设置**，它**不会**给
AscendC 后端打开调试 flag（`docs/tutorials/print.md` 的 Note 段落）。

> 注意：一个需要你自己核对的差异：A 树 `examples/print/elementwise_print.py` 里仍然写着
> `os.environ["TL_PTO_DEBUG"] = "1"`，而 `docs/tutorials/print.md` 与编程指南都改成了
> “优先用 `compile_flags`”。**两者不一致**。以你所用版本的实际行为为准，并以文档为准
> 优先用 `compile_flags`，见 `references.md` 的“待补资料”第 3 条。

**性能代价**：设备端打印会明显拖慢算子（每次打印/断言都要额外的硬件交互），且**每个核上
`dump`/`printf` 的总空间上限是 1 MB**。**只在调试期开，上线一定关掉。**

### 10.2 `T.dump_tensor`

```python
def dump_tensor(tensor: Buffer, desc: int, dump_size: int, shape_info: tuple = ())
```

* `tensor`：`ub_buffer` / `l1_buffer` / `l0c_buffer` / `global_buffer` 都支持，**不用区分类型**，
  直接写 buffer 名即可。
* `desc`：你自己定义的附加信息（行号、编号等），**只支持 uint32**。
* `dump_size`：要 dump 的元素个数。
* `shape_info`：不填就是一维铺开；填了按形状打印，缺的显示 `-`。

输出开头会自动带一大段上下文（CANN 版本、时间戳、kernel 类型、内存位置、dtype 等）：

```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, ...
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=111, addr=0, data_type=float16, position=UB, dump_size=32
```

本章 `code/10-debug-and-dump/print_and_dump.py` 是把 A 树 `examples/print/` 收窄到最小
可读版本。**前提**：A 树环境 + 真实 NPU。**验证级别**：仅静态核对。

### 10.3 AICore exception dump

硬件异常（非法访存、MTE 错误、Cube/Vector 溢出）发生时，你通常只能看到一个错误码。
A 树的 **AI Core Exception Dump** 特性会在**异常触发瞬间**通过 CANN 的回调机制
自动把 kernel 的**输入张量**保存下来，便于事后复现（来源：
`_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/ascend_aicore_exception_dump_zh.md`）。

原理（白皮书讲得很清楚，值得记）：

* TileLang 编译时在 kernel 参数末尾附一个 **`ParamSizeInfo`** 结构，开头是 8 字节的
  **MAGIC（`0x474e414c454c4954`，即 ASCII `TILELANG`）**，后面是 kernel name 与每个
  tensor 的字节数/地址/dtype 编码；
* 异常回调按 8 字节步长在参数缓冲区里**搜 MAGIC**，定位到描述信息后构建
  `acldumpTensorInfo` 数组，调 `acldumpSaveExceptionInfo` 落盘。

用法：

```python
import os
# 必须在 import torch / ACL 初始化之前设置！
os.environ["ASCEND_DUMP_PATH"] = "/tmp/exc_dump"
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"

import torch, tilelang
from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump

@tilelang.jit(out_idx=[2], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
})
def my_kernel(...): ...

try:
    result = kernel(a, b)
except Exception as e:
    print("AI Core exception:", e)
    for t in parse_exception_dump(dump_path="/tmp/exc_dump", kernel_name="main_kernel"):
        print(t["type"], t["index"], t["dtype"], t["data"].shape)
```

要点（白皮书原文）：

* **CANN ≥ 9.3.0**；
* `ASCEND_DUMP_PATH` 与 `ASCEND_DUMP_SCENE` **必须在 ACL 初始化前设置**，否则 CANN 读不到；
* 开关是 `TL_ASCEND_EXCEPTION_DUMP`，**默认关闭**，因为开启会改变 kernel 的 ABI
  （追加参数）并改变 runtime 行为。

**前提**：A 树环境 + CANN ≥ 9.3.0 + 真实 NPU。**命令**：见 A 树
`examples/exception_dump_test/example_exception_dump.py`（本章 `code/11-exception-dump/` 是
它的注解版）。**期望输出**：异常后 dump 目录出现
`<ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/<kernel_name>.custom.<ts>`，
`parse_exception_dump` 返回可读的 numpy 数组。**验证级别**：仅静态核对。

### 10.4 编译期：看生成的 Ascend C

看不到结果的时候，先看产物。A 树支持在 JIT 模式里打印生成的 Ascend C 源码
（编程指南 §5.2）：

```python
func = vec_add(M, N, 128, 256)
print(func.get_kernel_source())
```

编译链路是：**前端 kernel → 多轮 lower 得到 TensorIR → Ascend Codegen 生成 Ascend C →
毕昇编译器（bisheng）编译成 `.so` → ctypes 载入 → 当普通 Python 函数调用**
（编程指南 §2.3）。M 树同理，可用 `kernel.get_kernel_source()` 看生成的源码，
或用 `tilelang.ascend.callback.register_ascend_postproc_callback` 在编译前改写源码做
受控实验（`SKILL.md` §11 / `references/profiling.md`）。

---

## 11. 常见坑清单（昇腾特有）

下面 14 条按“装环境 → 写代码 → 同步 → 性能”排列，每条都给出**症状 + 根因 + 怎么查**。
来源分布：P1–P3 来自本章 §3 的环境层经验；P4–P9 综合 M 树
`SKILL.md` 第 10 节“Common Pitfalls”与 A 树编程指南的同名条目；
P10–P11 来自 M 树 `references/synchronization.md`（flag 配对与缓冲所有权）；
P12–P14 来自 `SKILL.md` §6/§12 与 `references/profiling.md`。

### 环境与安装

* **P1｜`import tilelang` 拿到的不是你装的那棵树。** 症状：`AttributeError: module
  'tilelang' has no attribute 'alloc_L1'`（或反过来）。根因：A 树的 `set_env.sh` 把
  `TL_ROOT` 插到 `PYTHONPATH` 最前面；如果你的 venv 里还装了其它 TileLang，
  `tilelang.__file__` 可能指向别处。查：`python -c "import tilelang; print(tilelang.__file__)"`，
  必须指向你打算用的那棵树。
* **P2｜忘了 `source .../ascend-toolkit/set_env.sh`。** 症状：编译期找不到 `bisheng` 或
  运行时找不到 `libruntime.so`。根因：CANN 的安装不改系统路径。查：`which bisheng`、
  `echo $ASCEND_HOME_PATH`。
* **P3｜`torch.npu.is_available()` 为 False。** 症状：所有 `.npu()` 调用失败。
  根因：`torch_npu` 与 CANN 版本不配套，或没设 CANN 环境变量。查：先确认
  `npu-smi info` 能看到设备，再确认 `import torch_npu` 没报错。

### 前端与 API

* **P4｜把两棵树的 API 混用。** 症状：`AttributeError` 或静默生成错误代码。
  根因：`T.alloc_L1`（A 树）vs `T.alloc_l1`（M 树）；`T.Kernel(..., is_npu=True)`（A 树）
  vs `T.Kernel(N)`（M 树）。查：确认文件顶部的 import 是 `tilelang.language as T` 还是
  `tilelang.ascend.language as T`。
* **P5｜在 `T.Kernel` 上用 `threads=`。** 症状：`TypeError`。
  根因：M 树的昇腾 `T.Kernel` 是 1D 核网格，**不接受 `threads`**；线程域只能写在
  `T.SimtVF(threads=N)` 里。（A 树是另一种写法：`is_npu=True` + `(cid, vid)`。）
* **P6｜想用多维 grid。** 症状：`T.Kernel(2, 3)` 之类的写法无效/报错。
  根因：昇腾只支持 1D grid。正确做法：把多维压成一维，再在 kernel 内用
  `row = tile_idx // N_TILES; col = tile_idx % N_TILES` 还原。
* **P7｜尾块没处理。** 症状：非整除 shape 下结果错或越界。
  根因：A 树 `examples/elementwise`、`examples/gemm` 都**直接 `M // block_M`**，假定整除，
  不处理 partial tile。查：用不能被 tile 整除的 shape 跑一遍；A 树有
  `example_gemm_tail_block_developer.py`、M 树有 OOB padding 与 `TL_ASCEND_TAIL_MASK`。
  遇到非 32B 对齐的行，还要注意 GM→UB 的 **pad value / `data_select`** 语义。
* **P8｜GEMM 的转置/布局约定搞混。** 症状：结果全错但形状对。
  根因：M 树当前 L1 输入路径要求 `transpose_A=False`、`transpose_B=True`，权重按 `[N, K]` 存；
  A 树 `example_gemm.py` 是 `[K, N]` + 默认 `transpose_B=False`。查：抄代码前先确认
  `T.Tensor` 的 shape 与 `transpose_*` 标志是否成对。

### 同步与正确性

* **P9｜`init=` / `clear_accum` 写错。** 症状：结果里混进上一次调用的残值；第一次迭代对、
  后续全错。根因：累加器（L0C）**不会自动清零**，必须由第一次迭代清。
  A 树：`T.gemm_v0(..., init=(k == 0))`；M 树：`T.gemm(..., clear_accum=(k == 0))`。
* **P10｜手写 flag 没有成对。** 症状：偶发错误；或当前调用看起来对，**下一次调用或
  同芯片上的其它进程出错**。根因：昇腾 950 要求**严格配对**（A2 会在算子之间清理多余计数，
  950 不会）。查：对每条 `set_flag` 找出可达路径上的匹配 `wait_flag`；特别注意**零次迭代
  的循环**（内部 set 被跳过，外部 drain wait 还在）、**被 stage 移位改变的 guard**、
  **退出路径上的 final drain**。
* **P11｜`T.barrier_all()` 被当成万能同步。** 症状：加了 barrier 还是错。
  根因：`barrier_all`（`PIPE_ALL` 全屏障）只等**它之前已发出的本核异步操作**；
  它**不是 buffer ownership 的交接**，也保护不了“屏障之后又出现的新 store 复用同一物理缓冲”
  的情况。真正的 buffer 复用要 producer→consumer 与 consumer→producer **两个方向**都有
  配对的 flag。

### 性能

* **P12｜叠加了流水却没有变快。** 根因候选：
  （a）**核内/核间流水被嵌套**（`T.Pipelined` 里套 `T.Pipelined`），不支持，要用 flat pattern；
  （b）**片上容量不够**，多缓冲被容量卡住，用 `T.annotate_unlimit_memory` 调查，或减小 tile；
  （c）用户已经把 buffer 手动加了 stage 维，又开了自动多缓冲，**等于双缓冲两次**；
  （d）本来就不是瓶颈所在（例如 Cube 已满，却还在调 MTE）。
* **P13｜在 Host-bound 场景里调 kernel。** 症状：单 kernel 很快，端到端很慢。
  根因：launch 开销盖过计算。查：连续小算子的链式调用上 ACLGraph（§9）。
* **P14｜用错了 `T.Parallel` 的期望。** 症状：以为 `T.Parallel` 会自动张量化，
  结果性能不达标。根因：自动调度只管任务顺序与依赖，**不保证把任意 `T.Parallel` 变成 SIMD
  指令**。要精确控制就得写 `T.SimdVF()` + 显式 `T.simd.*`，并实测周期
  （`npusim record`，见 `references/profiling.md`）。

---

## 12. 本章小结

一页纸复习：

1. **昇腾的机器模型是显式的**：AIC(Cube) 与 AIV(Vector) 分离；存储分 GM/L1/L0A/L0B/L0C/UB；
   搬运由 MTE1/2/3 与 FixPipe 负责；跨 PIPE 依赖必须显式同步。**累加器在 L0C，不在寄存器。**
2. **同步只有两种原语**：`PipeBarrier`（同序列内保序）与 `SetFlag`/`WaitFlag`（两序列间的锁）。
   A 树要么手写 `barrier_all`/`set_flag`/`wait_flag`，要么开自动同步；**M 树默认全自动**。
3. **第一个 kernel 的骨架**：`GM → UB → Vector → UB → GM`，中间夹两个 `barrier_all`
   （一个保 MTE2→V，一个保 V→MTE3）。
4. **第一个 GEMM 的骨架**：`GM → L1 → L0A/L0B → Cube → L0C → GM`，
   `T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))` 一行把 L1→L0→Cube→L0C 全包了。
5. **Ascend C 对照**：手写要管 `TPipe`/`TQue`/`AllocTensor`/`EnQue`/`DeQue`/`DataCopy`/
   `LoadData`/`Mmad`/`Fixpipe`/`SetFlag`；TileLang 只需要 tile 级的数据流，其余交给编译器。
6. **两棵树必须分清**：A 树（`tilelang.language`，Ascend C & PTO，A2/A3）与 M 树
   （`tilelang.ascend.language`，Ascend 950，AutoSchedule）。API 名大小写都不同，
   `PassConfigKey` 也不同。**一份 kernel 只用一棵树。**
7. **专属原语**：`T.alloc_l1/l0a/l0b/l0c`（M 树）、`T.dual_copy`（劈两个 AIV）、
   `T.SimtVF`/`T.SimdVF`（两种 Vector 编程）、`T.Pipelined`（软件流水，不能嵌套）、
   `T.annotate_buffer_versions`（版本数与索引模式）。
8. **调优先分类再动手**：用 `do_bench(backend="msprof")` 拿延迟，
   `backend="msprof_detail"` 拿 `KernelProfile`；用 `aic_mad`/`aic_mte2`/`aic_mte1`/
   `aic_fixpipe`/`aiv_vec`/`aiv_mte2` 判断瓶颈在 Cube、搬运还是 Vector，再决定调多缓冲
   还是调 core ratio。
9. **Host-bound 就入图**：ACLGraph = capture + replay，治的是下发开销，不是图编译；
   `torch_tl_ascend` 是接进 PyTorch 的桥。
10. **调试三件套**：`T.printf`（要开 `compile_flags`）、`T.dump_tensor`（UB/L1/L0C/GM 都能 dump）、
    AICore exception dump（异常瞬间自动抓输入张量，需要 CANN ≥ 9.3.0 且环境变量在
    `import torch` 之前设置）。

---

## 延伸阅读

* **[R1] A 树自述与安装**：`_vendor/tilelang-ascend-v0.1.2.000/README.md`、
  `install_ascend.sh`、`set_env.sh` ，环境准备的第一手材料（§3）。
* **[R2] A 树编程指南**：`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md`
  开发环境、语法、原语、调试、性能调优、入图一站式（§3–§10）。
* **[R3] M 树昇腾后端自述**：`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md`
  950 快速开始、AutoSchedule 说明、与 Ascend C 的分工对照表（§6.3、§7）。
* **[R4] M 树昇腾使用手册（skill 文档）**：
  `_vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend/SKILL.md`
  与其 `references/synchronization.md`、`references/profiling.md`
  两棵树差异、原语语义、常见坑、调优流程的最完整单点来源（§7、§8、§11）。
* **[R5] 官方样例**：M 树 `examples/ascend/`（`example_gemm.py`、`example_vecadd.py`、
  `example_rmsnorm.py`、`example_per_token_cast_to_fp8.py`）；A 树
  `examples/{elementwise,gemm,aclgraph,torch_tl_ascend,print,exception_dump_test}/`。
* **[R6] Ascend C 与硬件**：corpus 的
  `06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/...atlas_ascendc_10_0008_html.md`
  （基本架构：AIC/AIV、存储、MTE、Set/Wait）、`..._10_00020_html.md`
  （Ascend C 向量 add 的 inplace 写法）、
  `06-huawei-ascend/raw/catlass/examples/00_basic_matmul/basic_matmul.cpp`（Ascend C GEMM）。
* **[R7] 调优与 profiling**：corpus 的 CANN `Profiling` 文档
  （`.../CANNCommunityEdition/...`）、
  A 树 `docs/tutorials/{t_pipelined,auto_tuning,print,ascend_aicore_exception_dump_zh}.md`。

完整的引用清单（含核验状态与上游链接）见 [`references.md`](references.md)；
每个代码示例的运行前提与验证级别见 [`code/README.md`](code/README.md)；
图的说明见 [`figures/README.md`](figures/README.md)。
