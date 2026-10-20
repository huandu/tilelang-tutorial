# 第 07 章 快速上手：安装与第一个 kernel

## 学习目标

读完本章，你应当能够：

1. 在三种环境下把 TileLang 装上并验证：**pip 安装 wheel**、**从源码构建**、
   **昇腾 NPU**（安装 Ascend 版并让 `import tilelang.ascend.language` 成功），
   并说清每一条路径各自的 Python / CUDA / CANN / libc 要求。
2. 从零写出第一个 TileLang kernel（逐元素向量加法），并逐行讲清
   `T.Kernel` 的网格、`T.alloc_*` 的 buffer、`T.copy` 的搬运、`T.Parallel` 的
   并行域分别在做什么。
3. 用 `kernel.get_kernel_source()` 把生成的 CUDA 打印出来，并**对照 Python 讲清
   三段关系**：网格 → `blockIdx` / `__launch_bounds__`；`T.Parallel` → 线程 +
   向量化；`T.copy` → 128-bit 访存（或 TMA / cp.async）。
4. 说清 `@tilelang.jit` 的 lazy / eager 两种用法，以及"编译一次、多次复用"在
   哪一步发生。
5. 说清编译缓存的两级结构与 key 的构成，能回答"我改了参数 / 改了 pass / 换了
   版本，为什么不命中"。
6. 写出昇腾侧同一个 `C = A + B`，并指出它与 CUDA 版的三处结构性差异。
7. 拿到一条安装或运行报错时，按本章第 8 节的清单定位到具体原因。

## 前置知识

* 第 4 章「架构总览与一次编译的全流程」：知道 `Python → IRModule → pass → 设备源码
  → host wrapper → 可调用对象` 这条链，知道后端分岔点在哪。
* 第 6 章「芯片导论：NV GPU 与昇腾 NPU 的抽象对照」：知道"线程 / warp / SM"与
  "core / Cube / Vector / UB / L1"这些词指的是什么。
* 会写 Python；用过 `pip` 或 `conda` 装过包；见过装饰器（`@decorator`）。
* **不需要**写过 CUDA，也不需要读过 TVM。本章出现的每个硬件词第一次出现时都会
  先用一句话解释。

本章只回答一个问题：**"我要写第一个能跑的 TileLang 程序，从哪一步开始？"**
语言细节在第 8 章，原语语义在第 9 章，调优在第 11 章，昇腾深入在第 19、25 章。

---

## 1. 问题的由来

### 1.1 为什么第一个动手章要从「装 + 跑」开始

TileLang 是一门**嵌入式在 Python 里的 DSL**：你写的仍然是 Python 函数，但函数体
里的 `T.Kernel` / `T.copy` / `T.gemm` 不是普通函数调用，而是会被解释成一段
IR、再被编译成设备代码。这件事意味着两件事：

* **"能导入"不等于"能编译"，"能编译"不等于"能跑"**。环境（Python 版本、libc、
  CUDA 工具链、CANN）错了，失败会发生在三个阶段中的任何一个，报错信息各不相同。
* **你写的东西和真正执行的东西不是一回事**。所以第一个 kernel 的重点不是"写得多
  漂亮"，而是**写完以后能把生成的代码调出来看**。

这一章的结构就是照着这两个事实排的：先把环境钉死（第 3 节），再写一个最小 kernel
（第 4 节），马上把生成代码调出来对照（第 5 节），最后讲清"编译一次、复用多次"
（第 6 节）。

### 1.2 本章的版本纪律

全书锁定 TileLang `v0.1.15`（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）与
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`（见
[../../VERSIONS.md](../../VERSIONS.md)）。本章严格遵守四条：

1. **安装说明只抄 v0.1.15 仓库里的**
   `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md`（与 corpus 里的站点源文件
   `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/Installation.md`
   是同一份文档），昇腾部分只抄
   `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` 与
   `_vendor/tilelang-ascend-v0.1.2.000/README.md`、`install_ascend.sh`。
2. **不把 `main` 当 v0.1.15**。corpus 里那份主仓库快照比 v0.1.15 多一个提交，
   凡涉及"有没有某功能"的判断都靠 `grep`，不靠印象。
3. **不宣称未验证的运行结果**。写这一章的本机是 macOS（Apple Silicon），
   **没有 NVIDIA GPU，没有昇腾 NPU，也没有安装 tilelang**。因此：
   * 需要加速器的示例一律标注 **仅静态核对**，并给出真机验证步骤；
   * 需要打印的生成代码，凡不是本机产生的，都注明它来自**哪一份锁定版本文件**；
   * 不出现任何性能数字（本章一个都没有）。
4. **不写没有出处的路数**。`pip install`、源码构建、昇腾安装的每条命令都能在
   锁定版本文件里找到对应段落（见 [references.md](references.md)）。

顺手说清一件事：`v0.1.15` 是 **tag**。它对应的 `VERSION` 文件内容是 `0.1.15`，
而你用 pip 装出来的版本字符串可能是 `0.1.15` 也可能带本地后缀（`+cu...git<hash>`），
取决于它是从 PyPI 装的还是本地源码构建的（`Installation.md:365-376` 讲了这套后缀是
怎么来的）。**要看内容是否一致，别只看版本字符串。**

---

## 2. 概念与模型：一个 kernel 的四个部件

先不给代码，先给一张"看代码时的坐标系"。任何一个 TileLang kernel，剥到最里面，
都是四样东西（对照图见 `figures/02-first-kernel-anatomy.md`）：

| 部件 | 你写的 | 它回答的问题 | 会被编译器兑现成 |
| --- | --- | --- | --- |
| **网格（grid）** | `with T.Kernel(...) as bx:` | "这段计算被切成多少块并行执行？" | CUDA 的 grid，`bx` → `blockIdx.x`；昇腾的核编号 |
| **buffer** | `T.alloc_shared` / `T.alloc_fragment` | "数据放在哪一级存储上？" | shared / register 数组；昇腾的 UB / L1 / L0* |
| **搬运（copy）** | `T.copy(src, dst)` | "一个 tile 怎么从一级存储到另一级？" | LDG/STG、`cp.async`、TMA；昇腾的 MTE |
| **并行域（parallel）** | `for i in T.Parallel(n):` | "这 n 个元素各做一次，具体谁做？" | 线程映射 + 向量化 |

关键在这里：**前两列是人写的，后两列是编译器补的**。你只写"要对 256 个元素各做一次
加法"，至于用 256 个线程还是 32 个线程各做 8 个、要不要凑成 128-bit 访存，编译器自己
决定。这就是 TileLang 说的"block-level，不是 thread-level"
（`_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:41` 的原文是
"the code inside `T.Kernel` operates at the **block level**, not the thread level"）。

想要 thread-level 也能写（`T.get_thread_binding()`，属 expert 层），但那是第 8 章之后
的话题。本章只用 block-level。

---

## 3. 安装：三条路径

### 3.1 环境要求（v0.1.15 的事实）

先看 NVIDIA/CUDA 侧。`Installation.md:3-10` 列出的 pip 安装前置条件是：

| 项 | 要求 |
| --- | --- |
| glibc | **2.28**（Ubuntu 20.04 或更新） |
| Python | **>= 3.10** |
| NVIDIA CUDA | **>= 10.0**（用 host 上的 CUDA 工具链），或使用 pip 提供的 CUDA 工具链（**>= 13.0**） |
| AMD ROCm | host 上有一套提供 `hipcc` 的 ROCm（Linux x86_64 的 PyPI wheel 同时含 CUDA + ROCm） |

源码构建的前置条件略有不同（`Installation.md:70-76`）：

| 项 | 要求 |
| --- | --- |
| 操作系统 | **Linux 或 Windows**（macOS 不在列；macOS 上默认只编 Metal 后端） |
| Python | **>= 3.10** |
| NVIDIA CUDA | **>= 10.0**（host 工具链），或 pip 提供的 CUDA 工具链（**>= 13.0**） |

只有两条硬门槛会卡住大多数人：**Python 版本**和**工具链**。先跑一遍体检脚本
（本章 `code/06-env-check/`）比事后看着报错猜要快得多。

### 3.2 路径①：pip 安装 wheel（最省事）

```bash
# 1) 装
pip install tilelang

# 2) 验证
python -c "import tilelang; print(tilelang.__version__)"
```

`Installation.md:15` 给的就是这一行。也可以从 Release Page 下载预先编好的 wheel
再 `pip install <wheel 文件>`（`Installation.md:19-21`），或直接从 GitHub 源码装
（`Installation.md:25-27`，`pip install git+https://github.com/tile-ai/tilelang.git`
—— 注意它装的是那个分支的源码，不是你本地的 checkout）。

**要复现本书这一版**，把版本号钉住：用与 `_vendor/tilelang-v0.1.15/VERSION` 一致的
版本字符串安装（例如 `pip install "tilelang==0.1.15"`），或从 Release Page 取版本
字符串为 `0.1.15` 的那个 wheel。**不要用没钉版本的 `pip install tilelang` 去复现
本章的结论**——那样装到的是别的版本，后面的行号和缓存 key 都可能不一样。

还有一个 **nightly 渠道**（`Installation.md:332-341`）：

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly
```

它给的是滚动构建、可能包含尚未发布的改动，稳定性低于正式发布；**本书的所有结论都
不基于 nightly**。要不要用它，取决于你是不是在追某个刚合入的修复。

### 3.3 路径②：从源码构建（要改编译器，或者要自己控制编译选项）

什么时候必须走这条？三种情况：你要**改 TileLang 本体的 C++ 或 Python**；你要
**指定编译选项**（比如关掉 CUDA、只编昇腾）；host 上没有 CUDA 工具链而你想用 pip
提供的工具链。

最直接的写法（`Installation.md:93-99`）：

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install . -v
```

`--recursive` 不能省：TileLang 依赖一份定制过的 TVM，它作为 submodule 放在
`3rdparty/`（`Installation.md:88-91` 的 Note）。`-v` 打开详细输出，编译要几分钟。

如果你**没有 host CUDA**，用 pip 提供的 CUDA 包（`Installation.md:101-124`）：

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install -r requirements-dev.txt
pip install "nvidia-cuda-nvcc>=13" "nvidia-cuda-cccl>=13" "nvidia-cuda-nvrtc>=13"
pip install . -v --no-build-isolation
```

**开发者的推荐做法**是手动 cmake + `PYTHONPATH`（`Installation.md:133-158`）：

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON            # 用 ninja 的话：cmake .. -DUSE_CUDA=ON -G Ninja
make -j                           # 或者 ninja

export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"
```

几个和本章有关的 CMake 开关（`Installation.md:160-166`）：

* `-DUSE_CUDA=ON|OFF`：编 CUDA 后端；macOS 上默认关。
* `-DUSE_ASCEND=ON|OFF`：编昇腾后端；**Linux 上默认开**。
* `-DUSE_METAL=ON`：Darwin 上默认开。
* `-DUSE_LLVM`：CPU codegen 用的 LLVM，默认关，需要 LLVM 15+。

编译缓存放在 `~/.tilelang/cache`，并且**如果系统里有 `ccache` 会自动使用**
（`Installation.md:389-391`）。这一条在第 6 节还会用到。

### 3.4 路径③：昇腾侧（CANN + Ascend 版 tilelang）

昇腾侧有**两个来源**，先分清再动手：

| | 主干 `tilelang`（v0.1.15） | 独立仓库 `tile-ai/tilelang-ascend` |
| --- | --- | --- |
| 主要后端 | CUDA（+ ROCm/Metal/CPU/WebGPU），`tilelang/ascend/` 是 Ascend 950 后端 | 昇腾为主 |
| 文档 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` | `_vendor/tilelang-ascend-v0.1.2.000/README.md` |
| 本书角色 | 第 7、25 章用它讲昇腾后端 | 第 19 章用它讲适配器路线 |

**主干路线**（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:10-51`）先说清前置条件：

* 一套可用的 **Ascend 驱动 + CANN**，其中要能找到 `bisheng`（编译器）、**支持 CCE 的
  `ld.lld`**、以及昇腾运行时库；
* 兼容的 **PyTorch 与 `torch_npu`**，并且 `import torch_npu` 后
  `torch.npu.is_available()` 返回 `True`；
* 通用 Python / 原生构建前置条件（与 3.1 节同）。

`bisheng` 要在 `PATH` 或 `BISHENG_HOME/bin` 里；昇腾运行时库要能被动态链接器找到。
后端对 Ascend 950 默认 `dav-3510`，所以这个目标上**不设** `ASCEND_NPU_ARCH` 也行
（`ascend/README.md:27-31`）。从源码构建（`ascend/README.md:32-51`）：

```bash
git submodule update --init --recursive
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .

# 可编辑开发安装：
python -m pip install -r requirements-dev.txt
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -e . -v --no-build-isolation
```

`USE_CUDA=OFF` 让 CUDA 工具链不再是必需项。注意这条命令里的 `USE_ASCEND=ON` 是
**环境变量**，与 3.3 节的 `-DUSE_ASCEND=ON` 是同一件事在两条工具链上的两种写法。

**独立仓库路线**（`_vendor/tilelang-ascend-v0.1.2.000/README.md:76-134`）提供三种装法，
并把前置条件写得更具体：

* CANN **至少 8.3.RC1**，`torch-npu` **至少 2.6.0.RC1**；先 `source` CANN 的
  `ascend-toolkit/set_env.sh`。
* **方法 1（推荐）**：装现成 wheel。设置 `ASCEND_HOME_PATH` 后
  `pip install tilelang-*.whl`。
* **方法 2**：从源码打 wheel。
  ```bash
  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  ./build_wheel_ascend.sh [--enable-llvm]
  pip install dist/tilelang-*.whl
  ```
  `build_wheel_ascend.sh:29-35` 会先检查 `ASCEND_HOME_PATH`，没设就直接退出。
* **方法 3**：`bash install_ascend.sh`，然后 `source set_env.sh`。
  `install_ascend.sh:37-45` 第一件事就是检查 `python3 --version` 是否 >= 3.10；
  脚本自己会 `git submodule update`、打 TVM 补丁、`cmake` 并 `make`
  （`install_ascend.sh:171-243`）。

`set_env.sh`（`_vendor/tilelang-ascend-v0.1.2.000/set_env.sh:1-20`）做两件事：把仓库根
加进 `PYTHONPATH`，并设 `ACL_OP_INIT_MODE=1`（避免在 `torch_npu` 场景下导入 tvm）。

两个仓库都声明自己是**针对特定昇腾代际**的：主干后端是 Ascend 950；独立仓库说
"specifically tested and validated on **A2 and A3**"（`README.md:51-52`）。
装之前先确认你的机器在哪一档。

---

## 4. 第一个 kernel：从零到能跑

本章 `code/01-first-kernel/vecadd_tilelang.py` 是这段代码的完整、可复制版本。

### 4.1 版本 A：最小可读的 kernel

选 `C = A + B`（逐元素向量加法）而不是 GEMM，是因为它把 kernel 的四个部件全占了、
又不引人任何复杂概念。11 行：

```python
import tilelang
import tilelang.language as T


def vector_add_direct(N, block_N=256, threads=256, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]

    return main
```

它和官方教程 `docs/deeplearning_operators/elementwise.md` 的第一段示例同构，只是把
一维长度写成 `N`、把 `threads` 抽成参数。

### 4.2 逐行解释

**`@T.prim_func`**：把一个 Python 函数标记成"要编译成 TIR 的 kernel"。函数体的
参数注解 `A: T.Tensor((N,), dtype)` 不是类型提示，而是**缓冲区声明**——告诉编译器
"`A` 是一个长度 `N`、元素类型 `dtype` 的一维张量"。

**`with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:`**：这是**网格**。
`T.Kernel` 在锁定版本里的 docstring 原文是 "Construct a kernel launch frame:
**a grid of tile programs**"，并且强调 "This is the target-neutral launch: the part
every backend shares"（`_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-435`）。
两点要读懂：

* 网格的尺寸是 `ceil(N / block_N)`——**这个 kernel 被切成了多少个 tile 程序**；
* `bx` 是**当前 tile 程序的编号**（从 0 到 `ceil(N/block_N)-1`）。在 CUDA 上它落到
  `blockIdx.x`；在昇腾上一维核网格落到 core 编号。**它是"第几块"，不是"第几个线程"。**

`threads=256` 是 CUDA 方言多出来的参数（`tilelang/cuda/language/kernel.py:13-36`），
只是"每块有多少线程"的一条**提示 / 上限**，不是"总线程数"。默认 128。

**`for i in T.Parallel(block_N):`**：这是**并行域**。`T.Parallel` 的 docstring 说它是
"Tools to construct nested parallel for loop … used to create element-wise tensor
expression"（`tilelang/language/loop.py:13-20`）。它的语义是："这块的 `block_N` 个元素
各做一次操作"；**`i` 不是线程号**，而是这块里的**元素下标**。谁来执行、每个线程做
几个、要不要向量化，都在 `LayoutInference` 里被推导出来
（`loop.py:24-67` 解释了布局注解的约束与理由）。

**`C[bx*block_N + i] = A[...] + B[...]`**：就是逐元素加法。`bx*block_N + i` 把
"第几块"和"块内第几个"拼成全局下标。**边界不用你管**：N 不能整除 `block_N` 时，
编译器会自己补判断（见第 5.3 节的生成结果）。

### 4.3 版本 B：加上显式 buffer 与 `T.copy`

版本 A 直接把全局内存读进来再写回去。更常见的写法是先把 tile 搬进片上 buffer：

```python
def vector_add_buffered(N, block_N=256, threads=256, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            A_local = T.alloc_fragment((block_N,), dtype)
            B_local = T.alloc_fragment((block_N,), dtype)
            C_local = T.alloc_fragment((block_N,), dtype)

            T.copy(A[bx * block_N], A_local)
            T.copy(B[bx * block_N], B_local)

            for i in T.Parallel(block_N):
                C_local[i] = A_local[i] + B_local[i]

            T.copy(C_local, C[bx * block_N])

    return main
```

新增的是**buffer** 和**搬运**两个部件：

* `T.alloc_fragment(shape, dtype)`：分配一块**寄存器级别**的 buffer（默认
  `scope="local.fragment"`，`tilelang/language/allocate.py:81`）。另一个常用的是
  `T.alloc_shared(...)`，默认落在片上共享存储（`allocate.py:49`）。**选哪个是语义，
  不是性能开关**——第 8 章会展开。
* `T.copy(src, dst)`：tile 级搬运。它的 docstring 明确说，`T.copy` 只是构造一个 tile
  级操作，"low-level loop construction and any scope-specific decisions happen
  during lowering"（`tilelang/language/copy_op.py:59-113`）——也就是说，
  **你写的是"搬这块"，生成的是 `cp.async` / TMA / 向量化 LDG-STG 之类的具体东西**。
  注意 `A[bx * block_N]` 这种写法：源只给**首地址**，形状从目标 `A_local` 推出来
  （docstring 里的 "Syntactic Sugar" 一段，`copy_op.py:96-99`）。

版本 A 和 B 的**结果一模一样**，区别只在"数据先不先落到片上"。对照读它们，是理解
`T.copy` 与 `T.Parallel` 分工的最快方式（官方 `elementwise.md` 也是按这个顺序讲的）。

### 4.4 二维版本与官方示例

把一维换成二维几乎不用改结构，只多一维网格：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
    for local_y, local_x in T.Parallel(block_M, block_N):
        y = by * block_M + local_y
        x = bx * block_N + local_x
        C[y, x] = A[y, x] + B[y, x]
```

这段取自 `elementwise.md:57-87`（官方 2-D 向量加示例）。有一个**很容易踩的坑**：
`T.Kernel` 的第一个参数对应 `bx`（`blockIdx.x`，沿 N 方向），第二个对应 `by`
（`blockIdx.y`，沿 M 方向）——**参数顺序是"N 在前、M 在后"**，和第 4 章讲 GEMM 时
的约定一致。写反了不会报错，只会算错位置。

锁定版本仓库里还有一个"编译 + 校验 + 打印源码"的完整示例：
`_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py`。它用的是
`@tilelang.jit` 的 **eager** 写法（`in_dtype` / `out_dtype` 都当参数传），并且
`main()` 里自带 `torch.testing.assert_close`——这是把"能不能跑对"写进示例里的好习惯。

### 4.5 再进一步：官方 `quickstart.py` 的 GEMM + ReLU

当你把向量加法读顺了，下一步就是
`_vendor/tilelang-v0.1.15/examples/quickstart.py` 里的 GEMM + ReLU（第 4 章正文也引过
它）。它只比版本 B 多四件事：

```python
T.clear(C_local)                                    # 累加器清零
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):   # 软件流水 + 多缓冲
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)             # tile 级矩阵乘
for i, j in T.Parallel(block_M, block_N):           # 逐元素 ReLU，直接复用 T.Parallel
    C_local[i, j] = T.max(C_local[i, j], 0)
T.copy(C_local, C[by * block_M, bx * block_N])
```

`T.Pipelined`（`tilelang/language/loop.py:112`）、`T.gemm`
（`tilelang/language/gemm_op.py:160`）、`T.clear`（`tilelang/language/fill_op.py:48`）
的细节分别属于第 9、11、14 章。这里你只需要注意到：**`T.Parallel` 既能做内存搬运的
循环，也能做这样的逐元素 epilogue（后处理）**，这是它最通用的用法。

---

## 5. 读懂生成的源码

### 5.1 把它打印出来

拿到 kernel 对象后，三种方式看源码（`tilelang/jit/kernel.py:532-600`）：

```python
src = kernel.get_kernel_source()   # 返回字符串
kernel.show_source("kernel")       # 直接打印到 stdout；还支持 "host" / "both"
kernel.export_sources(kernel_path="k.cu", host_path="host.c")   # 写到文件
```

另一种等价写法：给 `@tilelang.jit(debug_root_path="/tmp/tl-debug")`，编译时会自动把
`tilelang_jit_kernel_<name>.c` 与 `tilelang_jit_program_<name>.py`（后者的
`prim_func.script()` 文本）写下来（`tilelang/jit/__init__.py:466-505`）。

### 5.2 Python 与生成代码怎么对照

拿官方的向量加示例（`elementwise_add(1024, threads=256, dtype=bfloat16)`）当例子。
源码在 `elementwise.md:155-172`，逐条对照如下（同一张表在本章
`code/02-inspect-source/` 里是可运行的文本）：

| Python | 生成的 CUDA | 为什么 |
| --- | --- | --- |
| `with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x):` | `extern "C" __global__ void __launch_bounds__(256) main_kernel(...)` | 网格大小变成 `gridDim.x`；`threads=256` 变成 `__launch_bounds__(256)`；`b_x` 变成 `blockIdx.x` |
| `for i in T.Parallel(threads):` | `if (((int)threadIdx.x) < 32) { ... }` | `T.Parallel` 是块级并行域；编译器发现每线程能做 8 个元素，于是**只用 32 个线程**（虽然仍然 launch 了 256 个） |
| `C[b_x*threads+i] = A[...] + B[...]` | `uint4 v_ = *(uint4*)(A + ...); ... ((nv_bfloat162*)...)->x = ...` | 逐元素代码被**向量化**成 128-bit 访存（`uint4` = 16 字节 = 8 个 bfloat16）与成对加法 |
| （Python 里根本没有 `threadIdx.x`） | `((int)threadIdx.x) * 8` | TileLang**自动插入**线程索引；stride 就是向量宽度 8。循环变量 `i` 在生成代码里消失了 |

这张表的最后一行最值得盯住：**你从没写过 `threadIdx.x`，它却出现在生成代码里**。
这正是"block-level"的含义——`T.Parallel` 把你的意图翻译成线程映射，中间还顺手做了
向量化。你想验证"我的 kernel 有没有被向量化"，看这一处就够了：出现 `uint4` /
`float4` / `__half2` 这类类型，说明命中了向量化。

### 5.3 两种形态：向量化版 vs 边界判断版

同一份 Python，`N` 能不能整除 `threads` 会让生成代码长出两种形态。官方文档给了对比
（`elementwise.md:104-130`、`:180-211`）：

**能整除、且编译器选择向量化时**（`elementwise.md:155-172`）：32 个线程各处理 8 个
元素，`uint4` 一次搬 16 字节。

**不能整除时**（`elementwise.md:120-130`，示例取 `N=2047`）：

```c++
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (((i * 256) + ((int)threadIdx.x)) < 2047) {           // ← 编译器补的边界判断
      C[((i * 256) + ((int)threadIdx.x))] = (A[((i * 256) + ((int)threadIdx.x))] + B[((i * 256) + ((int)threadIdx.x))]);
    }
  }
}
```

文档在 `elementwise.md:133` 的评语是：**边界判断会挡住向量化**——除了尾部那一组，
其余线程本来都能跑更优的代码。这就是"为什么 tile 尺寸要和 shape 对齐"的第一课：
不是为了好看，是为了让编译器敢向量化。

---

## 6. JIT、编译缓存与 kernel 复用

### 6.1 两个入口

锁定版本里编译入口有两个：

* 模块级 `tilelang.compile(prim_func, out_idx=-1, target="cuda")`：适合"我已经有一个
  构造好的 `PrimFunc`"。返回一个 `JITKernel`（`tilelang/jit/__init__.py:94`）。
* 装饰器 `@tilelang.jit`：把函数包成 `JITImpl`（`tilelang/jit/__init__.py:588`），
  调用它时再决定"编译"还是"直接执行"。

第 4 节的两个版本都能配 `tilelang.compile` 用；本章 `code/01-first-kernel/` 就是这种
写法。

### 6.2 `@tilelang.jit` 的两种模式

`JITImpl` 的 docstring 写得很清楚（`tilelang/jit/__init__.py:271-330`）：模式**自动
推断**，看修饰的函数是**返回 `PrimFunc`**（lazy）还是**用 DSL 直接写 body**（eager）。

**lazy 模式**——函数自己返回 `@T.prim_func`：

```python
@tilelang.jit(out_idx=-1, target="cuda")
def build_add(N, block_N=256, threads=256, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]
    return main

kernel = build_add(1 << 20)   # 返回的是一个 JITKernel
c = kernel(a, b)              # 再手动调用
```

**eager 模式**——函数用 DSL 直接写、返回输出张量：

```python
@tilelang.jit
def add_eager(A, B, block_N=256, threads=256):
    N = T.const("N")              # 形状声明成符号
    dtype = "float16"
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        for i in T.Parallel(block_N):
            C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]
    return C

c = add_eager(a, b)           # 编译并立刻执行
```

两种模式在锁定版本里都有真实使用：官方 `examples/elementwise/example_elementwise_add.py`
与 `examples/quickstart.py` 都是 eager（`T.const` + 参数注解），而
`examples/ascend/example_vecadd.py` 是 lazy（返回 `@T.prim_func`）。**要复用、要看
源码、要接 autotuner，用 lazy；要"像调用普通函数"，用 eager。**

### 6.3 编译缓存：两级，一个 key

编译缓存分两级（`_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py`）：

| 层 | 位置 | 命中后 |
| --- | --- | --- |
| 内存级 | `KernelCache._memory_cache`（进程内字典） | 直接返回 `JITKernel` 对象，磁盘都不碰 |
| 磁盘级 | `env.TILELANG_CACHE_DIR`（默认 `~/.tilelang/cache`） | 加载已编译产物（源码 / so / cubin 等） |

key 由 `KernelCache._generate_key()` 生成（`kernel_cache.py:243-284`），内容是
`sha256(json({...}))`，进 key 的字段有：

```text
func.script(show_meta=True) 的 sha256   # 你这份 kernel 的 TIR 文本
out_idx                                 # 输出在参数里的位置
args_repr                               # 编译参数（N、block_N、threads …）
target / target_host                    # "cuda" / "ascend" / ...
execution_backend                       # "tvm_ffi" / "cython" / "nvrtc" / ...
pass_configs / compile_flags
version / platform                      # 来自 _get_base_key()，macOS 还含 torch 版本
```

从这张表可以直接推出第 8 节里几个"为什么没生效"的答案：

* **改了注释或空白** → `func.script()` 不变 → 同一 key，**命中**。
* **改了 tile 参数**（如 `block_N`）→ `args_repr` 变 → 新 key，**不命中**。
* **改了 pass 开关** → `pass_configs` 变 → 新 key，**不命中**。
* **换了 tilelang 版本** → `version` 变 → 全部不命中（这是好事，避免拿到旧产物）。

所以调试编译器时，**第一件该怀疑的事是缓存**：`pass_configs` 变了 key 确实会变，
但如果你改的是 C++ pass 本体，而 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP` 没打开，
key 里就不含库内容哈希（`env.py:375-377` 与 `kernel_cache.py:141-142`），你可能
命中一条旧产物。此时用下面任一方式强制重编：

```python
import tilelang
tilelang.disable_cache()      # 关掉缓存（env.py:559）
...
tilelang.enable_cache()       # 用完记得开回来（env.py:558）
```

或环境变量：

```bash
TILELANG_DISABLE_CACHE=1 python my_kernel.py     # env.py:372-374
TILELANG_CACHE_DIR=/tmp/tl-cache python my_kernel.py   # env.py:368，换个缓存目录
```

### 6.4 复用与"探查"

缓存带来的直接好处是**复用**：同一个 `(kernel, 参数)` 组合，第一次编译、之后是命中。
在 `JITKernel` 上还能：

* `kernel(a, b)` 直接调用（`tilelang/jit/kernel.py:206-220`，`__call__` 转发到
  `adapter.func`）；
* `kernel.get_profiler().do_bench()` 做延迟/吞吐测量（`kernel.py:516`）；
* `get_kernel_source()` / `get_host_source()` 看两份产物；
* `export_sources(...)` 落盘。

一个必须记牢的边界：**主干 v0.1.15 的 `tilelang/cache` 里没有顶层
`clear_cache()`**。`tilelang.cache` 模块只导出了一个 `cached()` 入口
（`tilelang/cache/__init__.py:75`），清缓存的能力是 `KernelCache` 这个类上的
**实例方法**（`kernel_cache.py:455`）。而独立仓库 `tilelang-ascend` **有**
`tilelang.cache.clear_cache()`（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/cache/__init__.py:70`），
它的示例里也是这么调的。**照抄独立仓库的示例到主干上会在这一行报
`AttributeError`**——这类"两个仓库同名不同面"的坑，第 19 章还会遇到。本章
`code/04-api-surface-check/` 把这条差异做成了可复现的检查。

---

## 7. 昇腾侧的第一个 kernel

### 7.1 主干写法：同一个 `C = A + B`

完整代码见 `code/05-ascend-first-kernel/vecadd_ascend.py`，结构与官方
`examples/ascend/example_vecadd.py:11-44` 同构。核心是：

```python
import tilelang
import tilelang.ascend.language as T      # ← 方言换成昇腾

@T.prim_func
def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
    with T.Kernel(num_blocks) as bx:      # ← 一维核网格，没有 threads=
        a = T.alloc_shared((tile,), "float32")    # ← alloc_shared = UB（统一缓冲）
        b = T.alloc_shared((tile,), "float32")
        c = T.alloc_shared((tile,), "float32")
        for step in T.Pipelined(iterations, num_stages=2):
            begin = (step * num_blocks + bx) * tile
            T.copy(A[begin : begin + tile], a)
            T.copy(B[begin : begin + tile], b)
            with T.SimtVF(threads=threads):           # ← 线程域单独声明
                for i in T.Parallel(tile):
                    c[i] = a[i] + b[i]
            T.copy(c, C[begin : begin + tile])
```

**三处结构性差异**，每一处都不是语法糖：

1. **方言**：`import tilelang.ascend.language as T`。主干的
   `tilelang/language/__init__.py` 是把 **CUDA 方言**再导出的
   （`tilelang/language/__init__.py:11-12` 的 `from tilelang.cuda.language import *`），
   所以昇腾必须在 import 那一行换名字。混用两个 `T` 会拿到签名不同的
   `T.Kernel`。
2. **网格是一维核网格**：昇腾的 `T.Kernel(*blocks)` **没有 `threads=` 参数**，
   而且明确只接受一个网格维度：`kernel.py:139-174` 的 docstring 写着 "a 1-D grid of
   AI cores"，多给一维会直接 `raise ValueError`。线程域改由
   `with T.SimtVF(threads=...)` 在 kernel 内部单独声明
   （`tilelang/ascend/language/frame.py:107`）。也就是说：**CUDA 上"网格 + 线程"是
   一次 launch 里的两层，昇腾上"核网格 + 线程域"是两个独立的东西。**
3. **片上存储的起点不同**：CUDA 上 `T.alloc_shared` 就是 shared memory；昇腾上
   `T.alloc_shared` 落到 **UB（Unified Buffer，统一缓冲）**，而 `T.alloc_l1` /
   `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 才是 L1 与 Cube 的各级缓冲
   （`tilelang/ascend/language/allocate.py:15,26,37,46,55`）。同一个函数名在两台机器
   上指向不同物理存储——这正是第 6 章讲的"抽象是共享的、物理是各自的"。

昇腾的 `T.Kernel` docstring 里甚至自带一个最小例子（`ascend/language/kernel.py:165-168`）：

```python
with T.Kernel(NUM_CORES) as bx:
    with T.SimtVF(threads=128):
        for i in T.Parallel(128):
            out[bx * 128 + i] = x[bx * 128 + i] * 2.0
```

比本章的版本还短，值得对着读。

一处**有意的简化**要说明：官方 `examples/ascend/example_vecadd.py` 还在每个 UB buffer
上加了一行 `T.annotate_buffer_versions({...})`。本章的示例没写它，因为按该 API 的
docstring，`num_stages > 1` 时多缓冲份数是由自动调度器算出来的，这条注解只是
「锁定份数 / 控制索引模式」的可选手段（
`_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19-38`）。
这是本章唯一一处「应当等价、但作者未在 NPU 上验证」的取舍，第 11、19、23 章会展开。

### 7.2 与独立仓库方言的差异（一句话版）

`_vendor/tilelang-ascend-v0.1.2.000` 的示例长得**不一样**：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):   # ← 独立仓库方言
    a_ub = T.alloc_ub((...), dtype)                        # ← alloc_ub
    with T.Scope("V"):
        T.barrier_all()
        T.tile.add(c_ub, a_ub, b_ub)
```

对照点（本项已做成可复现检查，见 `code/04-api-surface-check/`）：

| | 主干 `tilelang.ascend.language` | 独立仓库方言 |
| --- | --- | --- |
| 网格 | `T.Kernel(n)`，1-D 核网格 | `T.Kernel(n, is_npu=True) as (cid, vid)` |
| UB 分配 | `T.alloc_shared` | `T.alloc_ub` |
| 线程域 | `T.SimtVF` / `T.SimdVF` frame | `T.Scope("V")` + `T.barrier_all` |
| 向量运算 | `T.Parallel` / `T.simd.*` | `T.tile.add` 等 `T.tile.*` 家族 |
| 清缓存 | 顶层没有 `clear_cache` | `tilelang.cache.clear_cache()` |

**这两套不是同义词，不能互相抄。** 本章第 7 节用主干写法，因为本章的其它内容都基于
v0.1.15。要把同一段代码搬到独立仓库的方言上，以及理解两套方言背后的两条技术路线
（Ascend C & PTO / AscendNPU IR），见第 19、25 章。

---

## 8. 常见安装 / 运行错误与排查

按"先环境、再导入、再编译、再运行、再数值"的顺序排查。每条都给出**看哪一行**。

**① 装不上 / 装完导入失败**

| 现象 | 最可能的原因 | 怎么办 |
| --- | --- | --- |
| `pip install` 直接失败 | Python < 3.10（`Installation.md:8`） | 换 3.10+；`python -VV` 确认是哪个解释器在跑 |
| 安装报 libc 相关错 | glibc < 2.28（`Installation.md:7`） | 升级到 Ubuntu 20.04+；用 `ldd --version` 看 |
| 源码构建找不到 CUDA | 没装 host CUDA 且没用 pip 工具链 | 走 `Installation.md:101-124` 的 pip CUDA 路线，或 `-DUSE_CUDA=OFF`（同时也要关掉其它后端） |
| `--recursive` 忘了加，编译报缺 TVM | submodule 没拉全 | `git submodule update --init --recursive` |
| `import tilelang` 报找不到 `libtilelang.so` | 走了 `PYTHONPATH` 路线但没 build，或 `PYTHONPATH` 指错 | `Installation.md:135-158`：先在 `build/` 里 cmake + make，再设 `PYTHONPATH`；开发模式下日志会提示 "Loading tilelang libs from dev root" |

**② 导入成功但一编译就报错**

| 现象 | 最可能的原因 | 怎么办 |
| --- | --- | --- |
| 报 `target` 相关错 / 找不到目标 | 目标不匹配（例如在 macOS 上编 CUDA） | 显式给 `target=`；或先看 `tilelang.__version__` 里嵌的 SDK 是不是你想要的 |
| 报 layout 相关错 | 某个 `T.Parallel` 在全流程结束后仍没有合法布局注解 | 提示信息会带源码 span（`TILELANG_ENABLE_IR_SPAN`，`env.py:400`），直接跳到那一行看 |
| 昇腾编译报容量 / 约束不可解 | 片上存储放不下（L1/UB 是硬约束） | 减 `num_stages` 或 tile 尺寸；错误通常发生在编译期而不是运行期 |

**③ 编译过了但跑起来不对 / 看着"改了没生效"**

| 现象 | 最可能的原因 | 怎么办 |
| --- | --- | --- |
| 明明改了参数却很快 | 命中缓存，你改的字段没进 key | 见 6.3；确认改的是 `func.script()` / `args_repr` / `pass_configs` 里出现的东西 |
| 改了 C++ pass 没生效 | 缓存 key 默认不含库内容哈希 | `TILELANG_DISABLE_CACHE=1`，或打开 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP`（`env.py:375`） |
| 数值不对，且只在边界错 | tile 尺寸和 shape 不整除 + 写出了越界 | 看生成代码有没有 `if (... < N)`；第 12 章的定位方法会展开 |
| 昇腾上"编译通过但跑不出来" | 缺 `bisheng` / `ld.lld`，或 `ASCEND_HOME_PATH` 没设 | 3.4 节三条前置条件逐条查；`code/06-env-check/check_env.py --ascend` |
| `torch.npu.is_available()` 为 False | `torch_npu` 版本不匹配或 CANN 没 `source` | 先 `source .../ascend-toolkit/set_env.sh`，再确认 `torch-npu >= 2.6.0.RC1` |
| 照抄独立仓库示例报 `AttributeError: clear_cache` | 主干没有顶层 `tilelang.cache.clear_cache()` | 见 6.4；主干请用 `tilelang.disable_cache()` |

**④ 想自己动手查**

```bash
python code/06-env-check/check_env.py --ascend     # 环境体检（本机可跑）
python code/04-api-surface-check/check_api_surface.py   # API 是否真的存在（本机可跑）
```

更系统的编译错误定位、`lower_trace` / `pass_diff` 与 profiling 属第 12 章。

---

## 9. 本章小结

把本章压成一页：

```text
① 安装三条路
   wheel   : pip install tilelang                         （Python>=3.10, glibc>=2.28, CUDA>=10.0 或 pip CUDA>=13）
   源码    : git clone --recursive + pip install . -v     （或 cmake + PYTHONPATH，-DUSE_CUDA / -DUSE_ASCEND）
   昇腾    : 主干 USE_ASCEND=ON USE_CUDA=OFF pip install -v .
             独立仓库 wheel / build_wheel_ascend.sh / install_ascend.sh（要 CANN>=8.3.RC1、torch-npu>=2.6.0.RC1）

② 一个 kernel 的四个部件
   网格   T.Kernel(..., threads=)    → grid / blockIdx / __launch_bounds__
   buffer T.alloc_shared / fragment  → shared / register（昇腾：UB / L1 / L0*）
   搬运   T.copy(src, dst)           → cp.async / TMA / 向量化 LDG-STG（昇腾：MTE）
   并行域 T.Parallel(n)              → 线程映射 + 向量化（"block level, not thread level"）

③ 读生成代码
   kernel.get_kernel_source()。看三处：网格 → blockIdx/__launch_bounds__；
   向量化 → uint4 / nv_bfloat162；边界 → if (... < N)。
   能整除 + 对齐 → 向量化；不整除 → 补边界判断、丢掉向量化。

④ JIT 与缓存
   @tilelang.jit 的 lazy（返回 PrimFunc）/ eager（返回输出张量）自动推断。
   缓存两级：内存 _memory_cache + 磁盘 ~/.tilelang/cache。
   key = sha256(func.script + out_idx + args_repr + target + backend + pass_configs + version + platform)。
   改注释不变 key；改参数/pass/版本会变 key。

⑤ 昇腾的第一个 kernel（主干写法）
   换方言 tilelang.ascend.language；T.Kernel 一维、无 threads=；
   线程域用 T.SimtVF(threads=...)；alloc_shared 落到 UB。
   注意：主干没有 tilelang.cache.clear_cache()（独立仓库有）。
```

如果只能记住三句话：

1. **先跑 `check_env.py`，再动手装**——大部分失败是环境，不是代码。
2. **写完第一个 kernel 的第一件事是 `get_kernel_source()`**，把 Python 和 CUDA
   对上看。
3. **"改了没生效"优先怀疑缓存**，`TILELANG_DISABLE_CACHE=1` 是最快的排除法。

## 延伸阅读

* 官方安装指南（本章 3.1–3.3 节的一手来源）：
  `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md`。
* 官方逐元素算子教程（本章第 4、5 节的生成代码就出自这里）：
  `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md`。
* 官方语言基础（下一章的前置读物）：
  `_vendor/tilelang-v0.1.15/docs/programming_guides/language_basics.md`。
* 官方调试工具（编译失败/数值错误的下一步）：
  `_vendor/tilelang-v0.1.15/docs/tutorials/debug_tools_for_tilelang.md`。
* 昇腾 950 后端（安装、Quick Start、与 Ascend C 的分工对照）：
  `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md`。
* 昇腾适配器仓库的安装说明：
  `_vendor/tilelang-ascend-v0.1.2.000/README.md:76-134`。
* 一次编译的全流程（本章第 5、6 节的上一层）见第 4 章；语言细节见第 8 章，
  核心原语见第 9 章，流水线见第 11 章，调试与 profiling 见第 12 章，
  昇腾实战见第 19 章，昇腾后端见第 25 章。

本章所有引用的完整清单与核验状态见 [references.md](references.md)。
