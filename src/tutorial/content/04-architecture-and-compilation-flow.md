# 第 04 章 架构总览与一次编译的全流程

## 学习目标

读完本章，你应当能够：

1. 说出 v0.1.15 仓库里 16 个顶层模块各自负责什么，并在 30 秒内判断「我要改的这个东西应该落在哪个目录」。
2. 独立画出并讲清 **Python kernel → IRModule → 后端 pass pipeline → 设备源码 → host wrapper →
   可调用对象** 这条链，并指出每一跳的入口函数与文件:行号。
3. 打开 `tilelang/cuda/pipeline.py` 或 `tilelang/ascend/pipeline.py`，**按源码顺序**列出完整
   pass 序列，并指出哪些 pass 是条件执行的、条件由谁决定。
4. 用 `tilelang/backend/` 里的五类组件解释这句话：「加一个新后端不需要改 `engine/lower.py`」。
5. 严格区分 **target backend** 与 **execution backend**，并各举 v0.1.15 里的两个真实例子。
6. 拿到一条关于 TileLang 的说法时，能判断它属于「v0.1.15 里的事实」还是「仅 main / 未合入」，
   并给出可复现的核验命令。

## 前置知识

* 第 3 章「设计哲学与编程模型」：知道 tile / dataflow / schedule 三件事分别指什么，
  知道 TileLang 有 beginner / developer / expert 三层接口。
* 会写 Python、看过装饰器（`@decorator`）；知道「编译」大致是把高级代码变成低级代码。
* **不需要**读过 TVM，也不需要写过 CUDA。本章出现的 IR、pass、codegen 都会先用一句话解释再给术语。

本章只回答一个问题：**我写的那几十行 Python，到底经过了谁，最后变成了什么？**
第 20 章会把本章列出的承重 pass 逐个剖开；第 24、25 章分别展开 NVIDIA 与昇腾的代码生成。

---

## 1. 问题的由来

### 1.1 这张地图要回答的三个问题

第一次读到 TileLang 代码的人，通常会卡在三个具体的位置：

* `T.gemm(A_shared, B_shared, C_local)` 是一行 Python，它凭什么变成硬件指令？**在哪个文件里变？**
* 只用公共语言表面写的同一份 kernel，可以分别以 `target="cuda"` 和 `target="ascend"` 编译。
  **分岔点在哪一行？**
* 官方说「多后端是解耦的」「加后端不用改 engine」。**这句话在仓库里对应哪些文件？**

这三个问题没有一个是「怎么用 API」的问题，它们都是**结构问题**。结构问题的答案不该靠背，
而该靠一张能随时重画的地图。本章就是把这张地图画出来，并且把地图上的每个地名都换成
`_vendor/tilelang-v0.1.15/` 里真实存在的路径与行号。

### 1.2 本章的版本纪律

全书锁定 TileLang `v0.1.15`（commit `a35f8dd`）与 TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（见 [../../VERSIONS.md](../../VERSIONS.md)）。
本章严格遵守下面四条：

1. **源码只引 `_vendor/tilelang-v0.1.15/`**，行号用 `grep -n` / `sed -n` 现场核对，
   不凭记忆。
2. **不把 main 当成 v0.1.15**。`_vendor` 之外的任何 checkout（包括 corpus 里那份
   `02-tilelang-core-repo/` 的 main 快照）都只用于「对照」，且在正文里显式标注。
3. **不宣称未验证的运行结果**。本机是 macOS（Apple Silicon），没有 NVIDIA GPU、
   没有昇腾 NPU，也没有安装 tilelang/torch，所以任何「跑起来是什么样」的结论都要
   写成「应当是什么样」并给出验证步骤。
4. **不写没有出处的性能数字**。本章不讨论性能，只讨论结构。

顺手说清一个容易混的点：**`v0.1.15` 是 tag，不是「最新的 main」**。
在写这一章时（2026-10-01）：

```text
_vendor/tilelang-v0.1.15/            git describe --tags → v0.1.15
                                     git log -1 → a35f8dd "[Release] Bump version into 0.1.15 (#3309)"
../tilelang-corpus/02-tilelang-core-repo/raw/tilelang
                                     git log -1 → 994b44e "[Ascend] Add PTO backend for Ascend950 (#3310)"
```

也就是说，两份 checkout 只差一个已合入的 PR：**corpus 的 main 快照停在 PR #3310
（Ascend 950 的 PTO backend），而 v0.1.15 对应的是 PR #3309（bump version）**。
这个差异在下面 4.9 与 5.3 节会变成一个具体的判据。

---

## 2. 概念与模型

### 2.1 五个阶段，一条竖线

不管后端是 CUDA 还是 Ascend，一次编译都可以切成五段。请把这张竖线记住，本章后面所有内容
都是它的展开：

```text
①  前端（frontend）       Python 函数          →  TIR IRModule
②  编译期（lowering）     IRModule             →  IRModule（被 pass 反复改写）
③  代码生成（codegen）    IRModule             →  设备源码（CUDA C / Ascend C / …）
④  宿主封装（host）       host IRModule        →  host wrapper + 参数/ABI 描述
⑤  运行时（runtime）      源码 + wrapper       →  可调用对象（Adapter / torch 函数）
```

每一段都有明确的「入口函数」，而且入口函数所在的位置在两条后端上是**同一个**：

| 阶段 | 入口 | 位置 |
| --- | --- | --- |
| ① 前端 | `tilelang.language.parser` / `tilelang.language.eager` | `tilelang/language/` |
| ② 编译期 | `tilelang.engine.lower.lower_with_context` | `tilelang/engine/lower.py:144` |
| ② 中的后端分岔 | `BackendContext.lower` → `PassPipeline.lower` | `tilelang/backend/module.py:211`、`tilelang/backend/pass_pipeline/pipeline.py:22` |
| ③ 代码生成 | `tilelang.engine.lower.device_codegen` | `tilelang/engine/lower.py:99` |
| ④ 宿主封装 | `tilelang.engine.lower.host_codegen` | `tilelang/engine/lower.py:65` |
| ⑤ 运行时 | `JITKernel._create_adapter_from_artifact` | `tilelang/jit/kernel.py:324` |

这张表是本章最重要的一页：**它说明「后端差异」被压缩进了第 ② 阶段的一个函数调用里**
（`context.lower(mod)`，`tilelang/engine/lower.py:136`）。其余所有阶段都是共享的。

### 2.2 仓库地图

下图见 `figures/fig-01-repo-map.md`（Mermaid 版 + ASCII 版）。这里给一眼能看完的版本：

```text
                          tilelang/jit/  (装饰器, JITKernel, adapter)
                                 |
                                 v
tilelang/language/  -->  tilelang/engine/lower.py  -->  tilelang/backend/  (注册 + 上下文)
   | 语言表面             编译入口                       |
   |                                                  +--> tilelang/<backend>/pipeline.py
   v                                                  |
TIR IRModule <----------------------------------------+
   |
   +--> src/transform/*.cc            (pass 本体, 42 个 .cc)
   +--> src/layout/layout_inference/  (layout 求解)
   +--> src/<backend>/codegen/        (CUDA C / Ascend C codegen)
   |
   v
tilelang/jit/adapter/  (tvm_ffi / nvrtc / cython / torch / cutedsl)
   |
   v
可调用 kernel

横向支撑：
  tilelang/cache/     kernel 缓存与 CUDA / Ascend 二进制缓存
  tilelang/autotuner/ 参数搜索            tilelang/carver/  tile 配置推荐
  tilelang/tools/     lower_trace / plot_layout / pass_timing / pass_visualizer
  examples/ (54 子目录)   docs/ (37 Markdown)
```

注意这张图里有一处非常反直觉、但一定要接受的安排：**pass 的「顺序」写在后端包的 Python 里
（`tilelang/cuda/pipeline.py`、`tilelang/ascend/pipeline.py`），pass 的「行为」写在 C++ 里
（`src/transform/*.cc`）**。初学者经常在 `src/transform/` 里找「pass 的执行顺序」，那是找不到的；
反过来，在 `tilelang/cuda/pipeline.py` 里找「LayoutInference 到底做了什么」，也找不到。

### 2.3 十六个模块的职责

下表每一行的「数量」列都来自 `code/repo-map/inventory.py` 的实测输出（v0.1.15）。

| 模块 | 职责 | 实测规模 |
| --- | --- | --- |
| `tilelang/language/` | **语言表面**。`T.Kernel`、`T.alloc_*`、`T.copy`、`T.gemm`、`T.Pipelined` 等 tile op 与 frame 的定义；另有 `parser/`（AST 路线）、`eager/`（builder 路线）、`ast/`、`overrides/` 两类前端实现。 | 47 个 `.py`，6 个子目录 |
| `tilelang/engine/` | **编译入口**。`lower()` / `lower_with_context()` / `host_codegen()` / `device_codegen()` 全在这里；`semantic_check.py` 放后端无关的前端语义检查；`param.py` 定义 `CompiledArtifact`。 | 5 个 `.py` |
| `tilelang/transform/` | 通用 pass 的 **Python 包装**。每个 `def LayoutInference()` 都只做一件事：调 `_ffi_api.LayoutInference()` 把 C++ pass 取出来。另有 `simplify.py`、`pass_config.py`。 | 7 个 `.py` |
| `src/transform/` | pass 的 **C++ 本体**。`layout_inference/`、`inject_pipeline.cc`、`lower_tile_op.cc`、`make_packed_api.cc` … 都在这里。 | 42 个 `.cc`，66 个文件 |
| `tilelang/layout/` | **布局代数**。`Layout`、`Fragment`、`PartialFragment` 与 swizzle 构造函数（`make_swizzled_layout`、`make_tcgen05mma_swizzled_layout`…），以及昇腾的 `make_ascend_nz_layout` 等。 | 10 个 `.py` |
| `tilelang/backend/` | **多后端抽象**。注册表、五类组件接口、`BackendContext` 解析、目标探测。本章 2.4 与 4.4 节的主角。 | 9 个 `.py` + `pass_pipeline/` |
| `tilelang/cuda/` | **CUDA 后端垂直切片**：`language/`（CUDA 方言）、`op/`、`intrinsics/`、`pipeline.py`、`codegen.py`、`backend.py`（manifest）。 | 55 个 `.py` |
| `tilelang/ascend/` | **昇腾后端垂直切片**：`language/`（`alloc_l1/l0a/l0b/l0c`、`SimtVF`、`dual_copy`…）、`transform/`、`pipeline.py`、`codegen.py`、`backend.py`。 | 35 个 `.py` |
| `src/ascend/` | 昇腾的 **C++ 侧**：`codegen/codegen_ascend.cc`、`op/`、`layout/`、`transform/`、`runtime/`。 | 80 个 `.cc/.h` |
| `tilelang/jit/` | **装饰器与运行时装配**：`jit()`、`JITImpl`、`JITKernel`、`adapter/`（tvm_ffi / cython / nvrtc / torch / cutedsl）、`wrapper.py`（生成 host wrapper 源码）。 | 31 个 `.py` |
| `tilelang/cache/` | **两级缓存**：`KernelCache`（整个 kernel 的磁盘缓存）+ `CUDABinaryCache` / `AscendBinaryCache`（编译产物缓存）。 | 5 个 `.py` |
| `tilelang/autotuner/` | **参数搜索**：`autotune` / `AutoTuner` / `capture` / `grouped_compile`。 | 5 个 `.py` |
| `tilelang/carver/` | **tile 配置推荐**：给 GEMM / reduction / elementwise 生成并排序 tiling 候选，可转给 TVM / Triton / TileLang。 | 34 个 `.py` |
| `tilelang/tools/` | **开发者工具**：`plot_layout`、`lower_trace/`（逐 pass 对比 IR）、`pass_visualizer/`、`pass_timing.py`、`Analyzer.py`、`compile_only.py`。 | 20 个 `.py` |
| `examples/` | 官方示例。GEMM 家族、FlashAttention、DeepSeek MLA/NSA/MoE、`quickstart.py` 等。 | 54 个子目录 + 3 个顶层文件（共 57 项） |
| `docs/` | 官方文档源文件（Sphinx）。`compiler_internals/`、`runtime_internals/`、`programming_guides/`、`deeplearning_operators/`、`tutorials/`、`tools/`、`get_started/`。 | 37 个 `.md`，9 个子目录 |

两个容易被忽略的事实：

* `tilelang/language/__init__.py` 的第 11–12 行写着
  `from tilelang.cuda.language import *`，文件末尾还有 `__tilelang_dialect__ = "cuda"`
  （`tilelang/language/__init__.py:1-32`）。也就是说，**`import tilelang.language as T` 拿到的
  其实是「公共表面 + CUDA 扩展」**；昇腾要用 `from tilelang.ascend import language as T`。
  这是 v0.1.13 引入「多后端语言方言」之后的现状，不是临时 hack。
* `tilelang/transform/__init__.py` 与 `src/transform/` 是**成对**的：想在 pipeline 里用一个新的
  C++ pass，必须先在这里加一个薄包装函数，否则 Python 侧拿不到它。

### 2.4 两类「后端」

这是本章最容易混、也最有价值的一个区分（图见 `figures/fig-03-backend-abstraction.md`）：

```text
target backend（决定「生成什么」）            execution backend（决定「怎么编译和启动」）
------------------------------------------    ------------------------------------------
cuda      → tilelang/cuda/pipeline.py         tvm_ffi  / nvrtc / cython
cutedsl   → 复用 CUDA pipeline，换 codegen     cutedsl
ascend    → tilelang/ascend/pipeline.py       tvm_ffi  / cython
hip       → tilelang/rocm/pipeline.py         见 tilelang/rocm/execution_backend.py
metal / c / llvm / webgpu → 各自               见各包 execution_backend.py
```

* **target backend** 拥有：语言方言、pass pipeline、host/device codegen、目标工具链回调。
* **execution backend** 拥有：缓存编排、wrapper 构造、产物加载、参数与 stream 绑定、启动、
  运行时生命周期。

它们为什么必须分开？看一个具体的组合就明白了：**同一个 CUDA target backend，可以走
`tvm_ffi`、`nvrtc`、`cython` 三种执行路径**（`tilelang/cuda/execution_backend.py:24-33`）。
换执行后端不会换掉 pass pipeline，也不会换掉生成的 CUDA C。反过来，换 target backend
（CUDA → Ascend）会换掉 pipeline，但执行层可以继续用同一个 `tvm_ffi`。

官方把这条设计原则写在 `tilelang/backend/README.md:76-78`：
「CUDA can execute through `tvm_ffi`, `nvrtc`, or `cython`; selecting `nvrtc` does not select
a different target architecture or pass pipeline.」

---

## 3. 用法：一次编译到底发生了什么

### 3.1 最小的 kernel

用官方 `examples/quickstart.py:8-48` 的 GEMM+ReLU（本章 `code/quickstart-gemm/` 有带插桩版本）：

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
```

这 20 行里有 4 个「语义级」的声明，编译器后面所有工作都是为了兑现它们：

| 声明 | 语义 | 谁来兑现 |
| --- | --- | --- |
| `T.Kernel(..., threads=128)` | 一个 2-D 网格，每块 128 线程 | `MaterializeKernelLaunch`（`tilelang/transform/__init__.py:204`） |
| `T.alloc_shared` / `T.alloc_fragment` | 数据放在 shared / fragment（寄存器） | `LayoutInference` + `LowerTileOp` |
| `T.copy` | tile 级搬运 | `LowerTileOp` → TMA / cp.async / LDG-STG / 昇腾搬运 |
| `T.gemm` | tile 级矩阵乘 | `LowerTileOp` → mma / wgmma / tcgen05 / Cube |
| `T.Pipelined(..., num_stages=3)` | 软件流水 + 多缓冲 | `PipelinePlanning` + `InjectSoftwarePipeline`（两侧都用） |

注意最后一列：**用户写的每一个高层原语，都能对应到一个具体的 pass**。这条「原语 ↔ pass」
的对应关系是第 20 章的主线；本章只保证你能定位到它们。

### 3.2 按图走一遍

完整流程见 `figures/fig-02-compile-flow.md`。这里用文字讲一遍，每一步都给入口：

1. **装饰器阶段。** `@tilelang.jit` 调用 `jit()`（`tilelang/jit/__init__.py:588`），
   把用户函数包成 `JITImpl`（`tilelang/jit/__init__.py:270`）。此时**还没有编译**，
   只是拿到了源码与签名。
2. **取值阶段。** `matmul.compile(M=1024, ...)` 触发 `JITImpl.compile`
   （`tilelang/jit/__init__.py:466`），它先调 `get_tir()`（`:379`）把 Python 函数
   **真的执行一遍**，得到 `PrimFunc`。
3. **缓存阶段。** `compile()` 最终走 `tilelang.cache.cached()`（`tilelang/cache/__init__.py:75`），
   由 `_resolve_cache_dispatch()`（`:38`）先解析出 `BackendContext`，再在
   `KernelCache`（`tilelang/cache/kernel_cache.py:35`）里查 key。命中就直接返回旧的
   `JITKernel`，**整个 pipeline 一步都不跑**。
4. **上下文阶段。** 未命中时构造 `JITKernel`（`tilelang/jit/kernel.py:41`），
   它在 `:113` 处 `create_backend_context(target, target_host, execution_backend)`
   （`tilelang/backend/module.py:310`）——**整个编译过程只解析一次后端**。
5. **编译阶段。** `JITKernel._compile_artifact`（`tilelang/jit/kernel.py:288`）在
   `tvm.transform.PassContext(opt_level=3, config=pass_configs, ...)` 里调用
   `tilelang.engine.lower.lower_with_context`（`tilelang/engine/lower.py:144`）。
6. **语义检查。** `lower_to_host_device_ir`（`:109`）先跑
   `PreLowerSemanticCheck`（`tilelang/engine/semantic_check.py:21`）：后端无关的
   `NestedLoopChecker` / `ParallelLocalIndexChecker` / `FragmentLoopChecker`。
7. **后端 pipeline。** 紧接着 `mod = context.lower(mod)`（`tilelang/engine/lower.py:136`）
   → `BackendModule.lower` → `PassPipeline.lower` → 具体后端的 `pipeline.py`。
   **这就是唯一的后端分岔点。**
8. **host/device 切分。** pipeline 内部已经跑过 `SplitHostDevice`
   （`tilelang/transform/__init__.py:285`），回到 engine 后用
   `tirx.transform.Filter` 把 host 与 device 两个 IRModule 分开
   （`tilelang/engine/lower.py:138-139`）。
9. **设备代码生成。** `device_codegen()`（`:99`）→ `context.codegen_device()` → 后端 manifest
   里声明的 `DeviceCodegen`。CUDA 会调到 native 的 `target.build.tilelang_cuda`。
10. **宿主代码生成（可选）。** `host_codegen()`（`:65`）只在
    `enable_host_codegen=True` 或所选 execution backend 声明需要时执行。
    `tvm_ffi` 需要它（`tilelang/cuda/execution_backend.py:24-30` 里
    `enable_host_codegen=True`），`nvrtc` / `cython` 不需要。
11. **产物。** 结果打包成 `CompiledArtifact`（`tilelang/engine/param.py:154`）：
    `host_mod` / `device_mod` / `params` / `kernel_source` / `rt_mod`。
12. **适配器。** `_create_adapter_from_artifact`（`tilelang/jit/kernel.py:324`）按
    `execution_backend` 选择 `TVMFFIKernelAdapter` / `NVRTCKernelAdapter` /
    `CythonKernelAdapter` / `CuTeDSLKernelAdapter` / `MetalKernelAdapter`
    （`tilelang/jit/adapter/__init__.py:1-6`）。
13. **可调用对象。** `JITKernel.torch_function = adapter.func`（`tilelang/jit/kernel.py:149`），
    `JITKernel.__call__`（`:206`）直接转发给它。用户写的 `kernel(a, b)` 走的就是这一行。

### 3.3 我想自己看一遍

三个由浅入深的做法，从「不用装任何东西」到「要真实设备」：

```bash
# (a) 不装 tilelang，只看结构：本章自带的三个离线脚本
cd chapters/04-architecture-and-compilation-flow/code
python3 compile-path-check/check_compile_path.py   # 30 跳是否都对得上
python3 pass-order/dump_pass_order.py | head -40   # CUDA/Ascend 的真实 pass 顺序
python3 repo-map/inventory.py                      # 模块规模

# (b) 装了 tilelang 且有 GPU：编出来但不跑，只看生成的源码与回写的 TIR
python3 code/quickstart-gemm/quickstart_gemm.py --out /tmp/tl-debug
#     会在 /tmp/tl-debug 下写出 tilelang_jit_kernel_<name>.c 与
#     tilelang_jit_program_<name>.py（后者是 prim_func.script() 的文本）

# (c) 要看每个 pass 前后 IR 的变化
#     官方工具在 tilelang/tools/lower_trace/（逐 pass 对比）
#     与 tilelang/tools/pass_visualizer/（结构化浏览），
#     也可用 PassConfigKey.TL_ENABLE_DUMP_IR + TL_DUMP_IR_DIR 直接 dump
```

更具体地说，第 12 章的工具路线是：
`TILELANG_PASS_DIFF=1` / `tilelang.tools.lower_trace` 打印每次 pass 前后的 IR diff；
`PassConfigKey.TL_ENABLE_DUMP_IR` + `TL_DUMP_IR_DIR` 把 IR dump 到磁盘
（见 `code/quickstart-gemm/quickstart_gemm.py` 里对 `debug_root_path` 的用法，
以及 `tilelang/jit/kernel.py:298-301` 对 `TL_ENABLE_DUMP_IR` 的判断）。

---

## 4. 实现机制

### 4.1 前端：从装饰器到 IRModule

前端有**两条**独立路线，都在 `tilelang/language/` 之下：

| 路线 | 用户写法 | 实现位置 | 产物 |
| --- | --- | --- | --- |
| lazy / parser 路线 | `@T.prim_func` + `def main(A: T.Tensor(...))` | `tilelang/language/parser/entry.py:37` 的 `prim_func` | `PrimFunc` |
| eager / builder 路线 | 直接写 `@tilelang.jit def f(A, B): A: T.Tensor[...] ...` | `tilelang/language/eager/builder.py`（`:1760` 的 `prim_func`，内部 `mutate(func)` 改写 AST） | `PrimFunc` 或 `JITFunc` |

两条路线的共同产物是 `PrimFunc`；`tilelang/jit/__init__.py:394-415` 的
`_infer_jit_mode` / `initialize_jit_mode` 决定一个装饰过的函数走哪条路
（「函数返回 `PrimFunc`」= lazy；「函数体使用 builder 模式」= eager）。

**这一步不产生硬件信息。** 前端只建立三类事实：buffer（形状、dtype、scope）、
访问关系（谁读谁写）、并行域（网格与循环结构）。把「数据怎么放、指令怎么发」留白，
正是第 3 章讲的「dataflow 与 schedule 解耦」。

### 4.2 engine：`BackendContext` 只解析一次

`tilelang/engine/lower.py` 只有 244 行，但它是全书的枢纽。它的四个函数值得记住：

```python
def lower(func_or_mod, target="auto", target_host=None, ...) -> CompiledArtifact   # :216
def lower_with_context(func_or_mod, context, ...) -> CompiledArtifact              # :144
def lower_to_host_device_ir(func_or_mod, context, ...)                             # :109
def host_codegen(host_mod, context) -> tvm.IRModule                                # :65
def device_codegen(device_mod, context) -> tvm.IRModule                            # :99
```

`lower()` 只做一件事：`context = create_backend_context(target, target_host, "auto")`
（`:237`），然后转交 `lower_with_context`。为什么要拆成两个函数？
因为 `lower()` 是「用户入口」，而 `lower_with_context()` 是「已经知道后端的人用的入口」——
JIT 层、缓存层、autotuner 都走后者，避免重复解析（`tilelang/jit/kernel.py:113-115`
明确写了「Direct callers may omit it and let this public entry resolve one context」）。

`BackendContext` 本身是 `frozen=True, slots=True` 的 dataclass
（`tilelang/backend/module.py:188-231`），并在 `__post_init__` 里断言
`module.matches(target)`、`execution_backend` 属于该 module。也就是说：
**一个 Context 一旦构造成功，就不可能自相矛盾**。

### 4.3 后端 pipeline：顺序写在 Python 里

以 CUDA 为例，`tilelang/cuda/pipeline.py` 只有三个值得关注的东西：

```python
def CUDAPassPipelineBodyPrologue(mod, target):   # :68   —— 33 步
    ...

def CUDAPassPipelineBody(mod, target):           # :168  —— 先调 Prologue，再 40 步
    mod = CUDAPassPipelineBodyPrologue(mod, target)   # :171
    ...

CUDA_PIPELINE = PassPipeline("cuda", CUDAPassPipelineBody)   # :284
```

`PassPipeline`（`tilelang/backend/pass_pipeline/pipeline.py:11-39`）只是个薄壳：
它把执行包在 `pass_pipeline(self.name)` 的 instrumentation 会话里，
并在异常时调用 `enrich_error(exc)` 补上源码位置。**它不做任何 pass 调度**——
顺序完全由传进去的 `lower` 函数（即 `CUDAPassPipelineBody`）决定。

这句「不做任何调度」很重要，它意味着：

> 想给 CUDA 后端加一个 pass，你**不需要**学一套描述语言，只需要在
> `tilelang/cuda/pipeline.py` 里找到合适的位置插一行 `mod = XxxPass()(mod)`。

### 4.4 多后端抽象：`tilelang/backend/`

这是本章的第 4 个必答点。`tilelang/backend/` 只做三件事（细节见
`figures/fig-03-backend-abstraction.md`）：

**（a）注册表。** 后端包在 import 时构造一个 `BackendModule`（`tilelang/backend/module.py:28-99`）
并调 `register_backend()`（`:238`）。注册表有两个全局字典：`_BACKENDS`（name → manifest）
与 `_TARGET_KIND_INDEX`（target kind → name 列表）。注册时做的校验很硬：

* 每个 `target_kinds` 必须有**恰好一个** pipeline 和**恰好一个** device codegen；
* pipeline 的 `name` 必须等于 target kind；
* 必须有至少一个 execution backend；
* 如果声明了 `enable_host_codegen`，就必须提供 host codegen；
* 同一个 target kind 上多个 manifest 共存时，**必须**都提供 `supports_target` 谓词
  （`:247-251`）。

最后一条正是 CUDA / CuTeDSL 共存的方式：

```python
# tilelang/cuda/backend.py:123-143
BackendModule(name="cuda", target_kinds=("cuda",),
              supports_target=codegen.is_plain_cuda_target, ...)

# tilelang/cuda/cutedsl_backend.py:8-23
BackendModule(name="cutedsl", target_kinds=("cuda",),
              supports_target=codegen.is_cutedsl_target,
              pipelines={"cuda": pipeline.CUDA_PIPELINE}, ...)
```

两个 manifest 都占 `cuda` 这个 kind，靠 `is_plain_cuda_target` /
`is_cutedsl_target` 区分（`tilelang/cuda/codegen.py:8-13`），
而且 CuTeDSL **显式复用** CUDA 的 pipeline。这正是 README 说的
「Pipeline reuse must be explicit; it must not come from target-specific branching in the engine」
（`tilelang/backend/README.md:137-138`）。

**（b）五类组件。** 每个组件都是一个 `frozen` dataclass 或小类，只描述「怎么调」，
不描述「调什么」：

| 组件 | 类型定义 | 职责 |
| --- | --- | --- |
| `PassPipeline` | `tilelang/backend/pass_pipeline/pipeline.py:11` | 一个 `name` + 一个 `lower(mod, target)` |
| `DeviceCodegen` | `tilelang/backend/device_codegen.py:31` | `build`（带编译）与 `build_without_compile`（只出源码）两个入口 |
| `HostCodegen` / `HostCodegenHook` | `tilelang/backend/host_codegen.py:33` / `:44` | host 侧 codegen；Hook 让设备后端在 host codegen 前预处理 host IR |
| `ExecutionBackendSpec` | `tilelang/backend/execution_backend.py:18` | 名称、可用性检查、target 谓词、是否需要 host codegen / 设备编译 |
| `BackendModule` | `tilelang/backend/module.py:28` | 把上面四类组装成一个 manifest |

共享的 host codegen 只有两个，写在 `tilelang/backend/host_codegen.py:55-59`：

```python
STANDARD_HOST_CODEGENS = {
    "c":    HostCodegen("c",    build=global_func_host_codegen("target.build.tilelang_c_host")),
    "llvm": HostCodegen("llvm", build=global_func_host_codegen("target.build.llvm")),
}
```

**（c）共享的小工具。** `pass_pipeline/pipeline_utils.py` 里的
`allow_vectorize` / `should_force_let_inline` / `should_enable_race_check` /
`should_disable_shared_memory_reuse` / `LayoutVisual` 是**两个后端都在用**的开关注入点。
例如 `LayoutVisual(mod)` 在 CUDA pipeline 的 `:140` 与 Ascend pipeline 的 `:85` 各出现一次。

### 4.5 CUDA pipeline 的 73 步

下面的表来自 `code/pass-order/dump_pass_order.py` 在 v0.1.15 上的实际输出：
它逐行扫 `tilelang/cuda/pipeline.py`，只认 `mod = <pass>(...)` 语句，记录 1-based 行号与
当前 `if` 守卫。共 **73 步** = `CUDAPassPipelineBodyPrologue`（33 步）+ `Body`
中 prologue 之后的 40 步。表中「条件」列写 `（无条件）` 表示该 pass 一定会跑。

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 1 | 69 | `tirx.transform.BindTarget` | （无条件） |
| 2 | 70 | `tilelang.transform.MaterializeKernelLaunch` | （无条件） |
| 3 | 74 | `tilelang.cuda.transform.AnnotateDeviceBoundTmaCopies` | （无条件） |
| 4 | 77 | `tilelang.transform.LetInline` | `should_force_let_inline()` |
| 5 | 79 | `tilelang.transform.AddWrapperForSingleBufStore` | （无条件） |
| 6 | 81 | `tilelang.transform.LegalizeNegativeIndex` | （无条件） |
| 7 | 84 | `tilelang.transform.VerifyParallelLoop` | `should_enable_race_check()` |
| 8 | 86 | `tilelang.transform.InjectAssumes` | （无条件） |
| 9 | 88 | `tilelang.transform.Simplify` | （无条件） |
| 10 | 91 | `tilelang.transform.CanonicalizeLegacyReducer` | （无条件） |
| 11 | 92 | `tilelang.transform.VerifyReducerEpoch` | （无条件） |
| 12 | 98 | `tilelang.transform.VerifyBufferInit` | （无条件） |
| 13 | 107 | `tilelang.cuda.transform.AutoWarpSpecialization` | `allow_warp_specialized(target)` |
| 14 | 108 | `tilelang.cuda.transform.MaterializeWSSchedule` | `allow_warp_specialized(target)` |
| 15 | 109 | `tilelang.cuda.transform.ProducerConsumerWarpSpecialized` | `allow_warp_specialized(target)` |
| 16 | 114 | `tilelang.cuda.transform.LowerBlackwell2SM` | （无条件） |
| 17 | 118 | `tilelang.transform.IfStmtBinding` | （无条件） |
| 18 | 123 | `tilelang.transform.UnrollLoop` | （无条件） |
| 19 | 125 | `tilelang.transform.Simplify` | （无条件） |
| 20 | 129 | `tilelang.transform.PipelinePlanning` | （无条件） |
| 21 | 130 | `tilelang.transform.InjectSoftwarePipeline` | （无条件） |
| 22 | 131 | `tilelang.transform.Simplify` | （无条件） |
| 23 | 134 | `tilelang.transform.LayoutInference` | （无条件） |
| 24 | 138 | `tilelang.transform.ReducerPlanAndMaterialize` | （无条件） |
| 25 | 142 | `tilelang.transform.LowerTileOp` | （无条件） |
| 26 | 144 | `tilelang.transform.VerifyReducerConsumed` | （无条件） |
| 27 | 148 | `tilelang.cuda.transform.LowerL2Persistent` | （无条件） |
| 28 | 150 | `tilelang.transform.DecoupleTypeCast` | （无条件） |
| 29 | 152 | `tilelang.transform.LegalizeVectorizedLoop` | （无条件） |
| 30 | 154 | `tilelang.transform.LegalizeSafeMemoryAccess` | （无条件） |
| 31 | 156 | `tilelang.transform.LowerAccessPtr` | （无条件） |
| 32 | 162 | `tilelang.transform.Simplify` | （无条件） |
| 33 | 164 | `tilelang.transform.HoistNonRestrictParams` | （无条件） |
| 34 | 175 | `tilelang.cuda.transform.LowerSharedTmem` | （无条件） |
| 35 | 180 | `tilelang.transform.PlanAndUpdateBufferAllocationLocation` | （无条件） |
| 36 | 196 | `tilelang.cuda.transform.LowerSharedBarrier` | （无条件） |
| 37 | 201 | `tilelang.cuda.transform.FuseMBarrierArriveExpectTx` | `has_tma` |
| 38 | 203 | `tilelang.transform.HoistGlobalBufferAllocations` | （无条件） |
| 39 | 204 | `tilelang.transform.LowerOpaqueBlock` | （无条件） |
| 40 | 205 | `tilelang.transform.Simplify` | （无条件） |
| 41 | 206 | `tirx.transform.NarrowDataType` | （无条件） |
| 42 | 207 | `tilelang.transform.FlattenBuffer` | （无条件） |
| 43 | 210 | `tilelang.transform.ConfigIndexBitwidth` | （无条件） |
| 44 | 211 | `tirx.transform.Simplify` | （无条件） |
| 45 | 212 | `tilelang.transform.VectorizeLoop` | （无条件） |
| 46 | 213 | `tilelang.transform.StorageRewrite` | （无条件） |
| 47 | 214 | `tilelang.transform.LoopUnswitching` | （无条件） |
| 48 | 215 | `tilelang.transform.UnrollLoop` | （无条件） |
| 49 | 216 | `s_tir.transform.RenormalizeSplitPattern` | （无条件） |
| 50 | 217 | `tirx.transform.Simplify` | （无条件） |
| 51 | 218 | `tirx.transform.RemoveNoOp` | （无条件） |
| 52 | 219 | `s_tir.transform.HoistIfThenElse` | （无条件） |
| 53 | 221 | `tirx.transform.VerifyMemory` | （无条件） |
| 54 | 222 | `tirx.transform.AnnotateEntryFunc` | （无条件） |
| 55 | 232 | `s_tir.transform.InferFragment` | （无条件） |
| 56 | 233 | `tilelang.transform.LowerThreadAllreduce` | （无条件） |
| 57 | 236 | `tilelang.cuda.transform.LowerLDGSTG` | （无条件） |
| 58 | 237 | `tilelang.cuda.transform.LowerHopperIntrin` | （无条件） |
| 59 | 239 | `tilelang.transform.AnnotateDeviceRegions` | （无条件） |
| 60 | 240 | `tilelang.transform.SplitHostDevice` | （无条件） |
| 61 | 244 | `tilelang.cuda.transform.MarkCudaSyncCalls` | （无条件） |
| 62 | 245 | `tilelang.transform.AnnotateReadOnlyParams` | （无条件） |
| 63 | 251 | `tilelang.transform.MergeSharedMemoryAllocations` | （无条件） |
| 64 | 256 | `tilelang.cuda.transform.InjectFenceProxy` | （无条件） |
| 65 | 258 | `tilelang.transform.ThreadSync`（`"shared"`） | （无条件） |
| 66 | 259 | `tilelang.transform.ThreadSync`（`"shared.dyn"`） | （无条件） |
| 67 | 266 | `tilelang.cuda.transform.InjectTcgen05Fence` | （无条件） |
| 68 | 268 | `tilelang.transform.MergeIfStmt` | （无条件） |
| 69 | 271 | `tilelang.cuda.transform.AnnotateWarpGroupRegAlloc` | `allow_warp_specialized(pass_ctx, target)` |
| 70 | 273 | `tilelang.transform.MakePackedAPI` | （无条件） |
| 71 | 274 | `tilelang.transform.Simplify` | （无条件） |
| 72 | 275 | `tilelang.transform.LowerDeviceKernelLaunch` | （无条件） |
| 73 | 279 | `tilelang.cuda.transform.PersistThreadblock` | （无条件） |

三点读表提示：

* **第 36 步 `LowerSharedBarrier` 是无条件的**，但紧邻它的
  `if not have_mbarrier(target) and _module_has_shared_barrier(mod): raise ValueError(...)`
  （`:186-195`）会在 pre-Hopper 上直接报错。也就是说：**「pass 一定会跑」与
  「这段代码一定会成功」是两回事**。
* 第 20–25 步（`PipelinePlanning` → `InjectSoftwarePipeline` → `LayoutInference` →
  `LowerTileOp`）是 CUDA 侧最核心的六步，第 20–22 章的正文就落在这里。
* 第 60 步 `SplitHostDevice` 之后，剩下 13 步都是在**已经分开的 device IR 上**做同步、
  存储合并与启动封装。

### 4.6 Ascend pipeline 的 80 步

同样方法扫 `tilelang/ascend/pipeline.py`，得到 **80 步**（全部来自
`AscendPassPipelineBody`）。为了可读性，下面用「分段」呈现，但每一段内部严格保持源码顺序，
行号也标出来。完整可复制版本用 `python3 code/pass-order/dump_pass_order.py` 生成。

**段 A：绑定与规范化（1–13）**

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 1 | 37 | `tirx.transform.BindTarget` | （无条件） |
| 2 | 45 | `MaterializeKernelLaunch`（`lower_grid_binding=True, lower_thread_binding=False, launch_dim_tags=["cthread"]`） | （无条件） |
| 3 | 57 | `LetInline` | `should_force_let_inline(pass_ctx)` |
| 4 | 58 | `AddWrapperForSingleBufStore` | （无条件） |
| 5 | 59 | `LegalizeNegativeIndex` | （无条件） |
| 6 | 61 | `VerifyParallelLoop` | `should_enable_race_check(pass_ctx)` |
| 7 | 62 | `InjectAssumes` | （无条件） |
| 8 | 68 | `ascend_transform.RewriteDualCopy` | （无条件） |
| 9 | 72 | `ascend_transform.LegalizeSimdMerging` | （无条件） |
| 10 | 73 | `tilelang.transform.Simplify` | （无条件） |
| 11 | 74 | `CanonicalizeLegacyReducer` | （无条件） |
| 12 | 75 | `VerifyReducerEpoch` | （无条件） |
| 13 | 76 | `VerifyBufferInit` | （无条件） |

**段 B：layout 与物理存储（14–20）**

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 14 | 80 | `ascend_transform.UnrollLoopSkipVF` | （无条件） |
| 15 | 81 | `tilelang.transform.Simplify` | （无条件） |
| 16 | 83 | `ascend_transform.AscendLayoutInference` | （无条件） |
| 17 | 84 | `ReducerPlanAndMaterialize` | （无条件） |
| 18 | 86 | `ascend_transform.NormalizeAscendFractalStorage` | （无条件） |
| 19 | 91 | `ascend_transform.InsertNd2Nz` | （无条件） |
| 20 | 101 | `ascend_transform.AscendInsertOOBPadding` | （无条件） |

**段 C：自动调度（21–33，整段受 `auto_schedule_enabled` 守卫）**

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 21 | 104 | `NormalizeControlFlowForSchedule` | `auto_schedule_enabled` |
| 22 | 105 | `NormalizeConflictHints` | `auto_schedule_enabled` |
| 23 | 106 | `MaterializeScheduleUnits` | `auto_schedule_enabled` |
| 24 | 107 | `AnnotateMultiBufferEligible` | `auto_schedule_enabled` |
| 25 | 108 | `EstimateLatency` | `auto_schedule_enabled` |
| 26 | 109 | `AutoSchedule`（Z3/SMT 求解） | `auto_schedule_enabled` |
| 27 | 110 | `AssignCore` | `auto_schedule_enabled` |
| 28 | 111 | `PrepareMultiBuffer` | `auto_schedule_enabled` |
| 29 | 112 | `ResolveCore` | `auto_schedule_enabled` |
| 30 | 113 | `InsertSync` | `auto_schedule_enabled` |
| 31 | 114 | `MaterializeMultiBuffer` | `auto_schedule_enabled` |
| 32 | 115 | `LowerScheduledTIR` | `auto_schedule_enabled` |
| 33 | 116 | `RestoreWhileLoops` | `auto_schedule_enabled` |

**段 D：tile op 下沉（34–44）**

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 34 | 118 | `tilelang.transform.Simplify` | （无条件） |
| 35 | 120 | `ascend_transform.NormalizeBufferVersion` | （无条件） |
| 36 | 121 | `ascend_transform.AscendSimdVFLowerParallel` | （无条件） |
| 37 | 122 | `ascend_transform.AscendLowerTileOp` | （无条件） |
| 38 | 123 | `VerifyReducerConsumed` | （无条件） |
| 39 | 125 | `DecoupleTypeCast` | （无条件） |
| 40 | 126 | `LegalizeVectorizedLoop` | （无条件） |
| 41 | 127 | `LegalizeSafeMemoryAccess` | （无条件） |
| 42 | 128 | `LowerAccessPtr` | （无条件） |
| 43 | 129 | `tilelang.transform.Simplify` | （无条件） |
| 44 | 130 | `HoistNonRestrictParams` | （无条件） |

**段 E：扁平化与向量化（45–56）**

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 45 | 132 | `HoistGlobalBufferAllocations` | （无条件） |
| 46 | 133 | `LowerOpaqueBlock` | （无条件） |
| 47 | 134 | `tilelang.transform.Simplify` | （无条件） |
| 48 | 135 | `ascend_transform.RewriteAscendBufferVersionLayout` | （无条件） |
| 49 | 136 | `tirx.transform.NarrowDataType` | （无条件） |
| 50 | 137 | `FlattenBuffer` | （无条件） |
| 51 | 138 | `ConfigIndexBitwidth` | （无条件） |
| 52 | 139 | `tirx.transform.Simplify` | （无条件） |
| 53 | 140 | `VectorizeLoop` | （无条件） |
| 54 | 146 | `ascend_transform.RewriteFp4ToFp4x2` | （无条件） |
| 55 | 148 | `LoopUnswitching` | （无条件） |
| 56 | 149 | `UnrollLoop` | （无条件） |

**段 F：收尾、切分与同步（57–80）**

| # | 行号 | pass | 条件 |
| --- | --- | --- | --- |
| 57 | 150 | `s_tir.transform.RenormalizeSplitPattern` | （无条件） |
| 58 | 151 | `tirx.transform.Simplify` | （无条件） |
| 59 | 152 | `ascend_transform.AscendRemoveNoOp` | （无条件） |
| 60 | 153 | `s_tir.transform.HoistIfThenElse` | （无条件） |
| 61 | 154 | `tirx.transform.Simplify` | （无条件） |
| 62 | 155 | `ascend_transform.AscendRemoveNoOp` | （无条件） |
| 63 | 157 | `tirx.transform.VerifyMemory` | （无条件） |
| 64 | 158 | `tirx.transform.AnnotateEntryFunc` | （无条件） |
| 65 | 159 | `s_tir.transform.InferFragment` | （无条件） |
| 66 | 160 | `LowerThreadAllreduce` | （无条件） |
| 67 | 163 | `ascend_transform.AscendThreadSync("global")` | `allow_global_thread_synchronization(pass_ctx)` |
| 68 | 164 | `AnnotateDeviceRegions` | （无条件） |
| 69 | 165 | `ascend_transform.MarkScalarDcacheBypass` | （无条件） |
| 70 | 166 | `SplitHostDevice` | （无条件） |
| 71 | 167 | `AnnotateReadOnlyParams` | （无条件） |
| 72 | 170 | `ascend_transform.InferBufferAliases` | `not auto_schedule_enabled` |
| 73 | 171 | `ascend_transform.MergeUBAllocations` | （无条件） |
| 74 | 179 | `ascend_transform.RewriteFlagToBuf` | （无条件） |
| 75 | 181 | `ascend_transform.AscendThreadSync("shared")` | （无条件） |
| 76 | 182 | `ascend_transform.AscendThreadSync("shared.dyn")` | （无条件） |
| 77 | 183 | `MergeIfStmt` | （无条件） |
| 78 | 184 | `MakePackedAPI` | （无条件） |
| 79 | 185 | `tilelang.transform.Simplify` | （无条件） |
| 80 | 186 | `LowerDeviceKernelLaunch` | （无条件） |

注意第 72 步：**`InferBufferAliases` 只在自动调度关闭时跑**。这是个典型的
「关闭 A 就要打开 B」的例子——调度器自己会推导 buffer 别名，所以人写逻辑时需要显式补上。

### 4.7 两条 pipeline 对照

统计（脚本输出）：CUDA 73 步 / Ascend 80 步，两侧**同名**出现的 pass 有 **39** 个，
带条件守卫的分别是 CUDA **7** 步、Ascend **17** 步。阶段图见
`figures/fig-04-pipeline-compare.md`。

| 阶段 | CUDA | Ascend |
| --- | --- | --- |
| 设备绑定 | `BindTarget` | `BindTarget` |
| 网格物化 | `MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)` | `MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False, launch_dim_tags=["cthread"])` |
| 后端前处理 | `AnnotateDeviceBoundTmaCopies`、`LowerBlackwell2SM`、warp specialization 三连 | `RewriteDualCopy`、`LegalizeSimdMerging` |
| 流水 | `PipelinePlanning` + `InjectSoftwarePipeline`（通用 pass） | 由 `AutoSchedule` 段统一决定（含多缓冲） |
| 布局 | `LayoutInference`（通用） | `AscendLayoutInference` + `NormalizeAscendFractalStorage` + `InsertNd2Nz` |
| 边界 | `LegalizeSafeMemoryAccess` | 额外有 `AscendInsertOOBPadding` |
| tile op 下沉 | `LowerTileOp` | `AscendLowerTileOp` + `AscendSimdVFLowerParallel` |
| 存储合并 | `MergeSharedMemoryAllocations` | `MergeUBAllocations` |
| 同步 | `ThreadSync("shared")` ×2 + `InjectFenceProxy` + `InjectTcgen05Fence` | `AscendThreadSync("global"/"shared"/"shared.dyn")` + `RewriteFlagToBuf` |
| 收尾 | `MakePackedAPI` → `LowerDeviceKernelLaunch` → `PersistThreadblock` | `MakePackedAPI` → `LowerDeviceKernelLaunch` |

一句话总结：**两侧共享的是一条「tile op 之后」的标准 lowering 走廊（39 个同名 pass），
真正的差异集中在三处——网格物化的参数、流水/调度由谁负责、同步与片上存储怎么合并。**

### 4.8 codegen、host wrapper 与可调用对象

pipeline 结束时，`mod` 里的函数已经被 `SplitHostDevice` 分成了 host 与 device 两族。
engine 用 `tirx.transform.Filter` 把它们分开（`tilelang/engine/lower.py:138-139`），
判据是 `calling_conv == CallingConv.DEVICE_KERNEL_LAUNCH`
（`tilelang/engine/lower.py:26-44`）。

设备侧走 `context.codegen_device(..., compile_device=...)`。CUDA 的
`DeviceCodegen` 由两个全局函数支撑（`tilelang/cuda/codegen.py:16-17`）：

```text
target.build.tilelang_cuda                    # 生成并编译
target.build.tilelang_cuda_without_compile    # 只生成源码
```

Ascend 同理（`tilelang/ascend/codegen.py:5-9`）：

```text
target.build.tilelang_ascend
target.build.tilelang_ascend_without_compile
```

宿主侧的关键 pass 是 `MakePackedAPI`——它的 C++ 本体在
`src/transform/make_packed_api.cc`，文件头注释写着
「Lower PrimFunc to use the packed function API」（`:2`）。
这一步把 `PrimFunc` 改写成符合 TVM-FFI 打包调用约定的入口：新的入口形如
`(args, num_args)`（`:434-437`），用 `MakeAssertEQ` 一类断言检查参数个数与合法性
（`:421-475`），并按 `tilelang_out_idx` / `tilelang_callee_allocated_outputs`
两个属性决定输出张量由调用方还是被调方分配（`:38-42`、`:342-355`）。

最后一步是 adapter。它不属于 target backend，而属于 **execution backend**：

```text
tilelang/jit/adapter/
  tvm_ffi.py       TVMFFIKernelAdapter      # 与 tvm_ffi runtime module 对接
  nvrtc/           NVRTCKernelAdapter       # 源码 + driver 启动
  cython/          CythonKernelAdapter      # 生成 Cython wrapper
  torch/           MetalKernelAdapter       # 由框架提供 compile/launch
  cutedsl/         CuTeDSLKernelAdapter
  wrapper.py       生成 host wrapper 源码的通用实现
```

`tilelang/backend/README.md:244-270` 把这条契约写得很清楚：
target backend 需要提供「compiled 或 source-only 的 device codegen、按需的 host codegen、
source/artifact/global-symbol/argument/launch metadata、目标相关的校验与工具链回调」，
然后由 execution backend 负责利用它们完成缓存、加载、绑定与启动。

### 4.9 缓存：两级

`tilelang/cache/` 里的缓存有两层，用途完全不同：

| 层 | 类 | 键里有什么 | 命中后的效果 |
| --- | --- | --- | --- |
| kernel 级 | `KernelCache`（`tilelang/cache/kernel_cache.py:35`） | `PrimFunc` + `out_idx` + target + target_host + execution_backend + `pass_configs` + `compile_flags` | 直接返回一个 `JITKernel`，**pipeline 完全不跑** |
| 二进制级 | `CUDABinaryCache` / `AscendBinaryCache` | 源码文本 + target kind + arch + 编译选项 | 直接返回编译产物 `bytearray`，**跳过 nvcc / bisheng** |

派发逻辑在 `tilelang/cache/__init__.py:24-35`：五条通用路径
（`tvm_ffi` / `cython` / `nvrtc` / `cutedsl` / `torch`）加两条昇腾专用路径
（`AscendTVMFFIKernelCache` / `AscendCythonKernelCache`）。注意派发依据是
**resolved backend module 名**，不是 target kind（`:68-69` 的注释
「Dispatch on the resolved backend module rather than on the target kind」）。

CUDA pipeline 第 34 步是 `LowerSharedTmem`（`tilelang/cuda/pipeline.py:175`）；
它的 C++ 本体做的事是「Convert `shared.tmem` buffers to plain shared + ptx init」
（`src/cuda/transform/lower_shared_tmem.cc:3`），而 `tmem` 属于需要特定架构才有的资源
（仓库根 `README.md:123` 的原话是「TMA, WGMMA, and TMEM features require the
corresponding GPU architecture」）。架构相关的 lowering 正是二进制缓存必须把 arch 写进 key
的原因：同一份 CUDA C 在不同 `sm_*` 上要编出不同 cubin。
`tilelang/cuda/backend.py:59-120` 的 `tilelang_callback_cuda_compile`
把 `target_arch` / `target_code_list` / `compile_format` / `options` 全塞进 `make_key`。

---

## 5. 进阶与陷阱

### 5.1 pass 序列是带分支的程序，不是静态表

最容易犯的错误，是把 4.5 / 4.6 的表格背下来当成「TileLang 的 pass 顺序」。
它其实是**在默认 `pass_configs` 下、针对一个普通 kernel 的顺序**。至少有三类因素会改变它：

**（a）`pass_configs`。** 表中所有带条件列的 pass 都由
`tilelang/backend/pass_pipeline/pipeline_utils.py` 里的函数决定，而这些函数全部读同一份
`PassContext.config`：

```python
# tilelang/backend/pass_pipeline/pipeline_utils.py
def allow_vectorize(pass_ctx=None) -> bool:                 # :25
    return not pass_ctx.config.get("tirx.disable_vectorize", False)

def should_force_let_inline(pass_ctx=None) -> bool:         # :46
    return bool(pass_ctx.config.get(PassConfigKey.TL_FORCE_LET_INLINE, False))

def should_enable_race_check(pass_ctx=None) -> bool:        # :58
    default_disable = not _env_data_race_check_enabled()     # 环境变量 TILELANG_ENABLE_DATA_RACE_CHECK
    disable = pass_ctx.config.get(PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, default_disable)
    return not disable
```

**（b）目标硬件。** `allow_warp_specialized`（`tilelang/cuda/pipeline.py:27-36`）要求
「是 CUDA target」且「有 TMA」；`has_tma` 又让
`FuseMBarrierArriveExpectTx` 单独成一步（表格第 37 步）。所以同一份 kernel 在
sm_80 与 sm_90 上跑的 pass 数目并不相同。

**（c）kernel 自身的内容。** Ascend 的 13 步自动调度只在
`TL_ENABLE_AUTO_SCHEDULE` 为真时执行（`tilelang/ascend/pipeline.py:22-33`），
关闭时第 72 步的 `InferBufferAliases` 才会补位。

**可复现的结论**：任何「TileLang 的 pass 顺序是 X」的说法都必须附带
`target` + `pass_configs` + 是否开启的开关。这也是本书第 20 章会坚持给每条结论加条件的原因。

### 5.2 五个具体陷阱

1. **`import tilelang.language as T` 拿到的是 CUDA 方言。**
   `tilelang/language/__init__.py:11-12` 直接 `from tilelang.cuda.language import *`，
   并在 `:32` 声明 `__tilelang_dialect__ = "cuda"`。写昇腾 kernel 要
   `from tilelang.ascend import language as T`；混用两套 `T` 会拿到不同签名的 `T.Kernel`。
2. **不要在 `engine/lower.py` 里找后端分支。** 那里只有一次 `context.lower(mod)`
   （`:136`）。任何「按 target 分支」的逻辑都应该在对应后端的 `pipeline.py` 或
   `backend.py` 里。如果你在 engine 里加了 target 分支，等于把 `README.md:240-242`
   明令禁止的事情做了（「The engine must not contain a `target.kind.name` dispatch table
   for backend codegen.」）。
3. **改完 pass 没生效，先怀疑缓存。** `KernelCache` 命中时 pipeline 一步都不跑；
   调试编译器时应当设 `TILELANG_CACHE_DIR` 或直接 `tilelang.disable_cache()`，
   否则很容易把「缓存命中」误判成「pass 没起作用」。
4. **`SplitHostDevice` 之后 IR 是两个模块。** 在 pipeline 后半段做调试打印时，
   如果只打印一个模块，会以为「代码丢了」。engine 在 `tilelang/engine/lower.py:138-139`
   用两个 `Filter` 分开，调试时也要分开看。
5. **别把「共享前端」理解成「共享 IR」。** 前端（`tilelang/language/`）确实共享，
   但已经进到 `AscendLayoutInference` 的 IR 与进了 `LayoutInference` 的 IR
   已经是两套不同的抽象（前者带 fractal / NZ 信息，后者带 fragment / swizzle 信息）。
   第 21 章会把这两套 layout 代数并排讲。

### 5.3 v0.1.15 事实 vs 仅 main / 未合入

这一节是本章的「版本边界」。判据全部可以现场复现（命令按顺序给）：

```bash
cd _vendor/tilelang-v0.1.15
cat VERSION                       # 0.1.15
git describe --tags               # v0.1.15
git log -1 --oneline              # a35f8dd [Release] Bump version into 0.1.15 (#3309)
ls tilelang/jit/adapter/          # 没有 pto/
grep -rn "AutoSimtVF" tilelang src    # 无输出
grep -rn "PersistentKernel" tilelang src  # 无输出
```

| 能力 | v0.1.15 里有吗 | 依据 |
| --- | --- | --- |
| `tilelang/ascend/`（Ascend 950 后端）、`src/ascend/` | **有** | `ls tilelang/ascend`、`ls src/ascend`；`tilelang/ascend/backend.py:52-63` 的 manifest |
| `tilelang/backend/` 多后端抽象（`BackendModule` / `BackendContext` / `PassPipeline`） | **有** | `tilelang/backend/` 下 9 个 `.py` + `README.md` |
| CUDA / CuTeDSL 两个 manifest 共用 `cuda` target kind | **有** | `tilelang/cuda/backend.py:123-143`、`tilelang/cuda/cutedsl_backend.py:8-23` |
| Ascend 自动调度（`EstimateLatency` → `AutoSchedule` → `InsertSync` → `MaterializeMultiBuffer`） | **有** | `tilelang/ascend/pipeline.py:104-116` |
| `tilelang/jit/adapter/pto/`（PTO 执行路径） | **没有**；在 main（≥ `994b44e`） | `ls _vendor/tilelang-v0.1.15/tilelang/jit/adapter/` 无 `pto`；corpus 的 main 快照 `994b44e` 有 |
| `AutoSimtVF` pass（PR #3320） | **没有**，也未进写这份材料时的 main | `grep -rn AutoSimtVF` 在两个 checkout 都无输出 |
| `T.PersistentKernel` + `AutoPersistent`（PR #3325） | **没有**，也未进写这份材料时的 main | `grep -rn PersistentKernel` 无输出 |

两点补充：

* PR 编号只是**线索**，判据永远是「在哪个 checkout 里 grep 得到什么」。
  v0.1.15 对应的 PR 是 #3309（bump version），而 corpus 的 main 快照停在 #3310。
* 昇腾还有**另一条完整路线**：独立仓库 `tile-ai/tilelang-ascend`
  （本章锁定 `_vendor/tilelang-ascend-v0.1.2.000/`，其 `VERSION` 文件写 `0.1.4`，
  以 tag 为准）。它不是 `_vendor/tilelang-v0.1.15/tilelang/ascend/` 的同义词，
  两者的关系在第 25 章展开。

---

## 6. 本章小结

把本章压缩成一页：

```text
① 结构：16 个顶层模块，可分为「前端 / 编译入口 / 后端切片 / C++ 本体 / 运行时 / 工具」六类。
        语言表面在 tilelang/language/（默认 re-export CUDA 方言）。

② 一次编译：decorator → JITImpl → get_tir → cache → BackendContext(一次)
            → PreLowerSemanticCheck → context.lower(mod)  ← 唯一后端分岔点
            → SplitHostDevice → device/host codegen → CompiledArtifact → adapter → 可调用对象

③ 顺序在哪：tilelang/<backend>/pipeline.py（Python，可读可改）
   行为在哪：src/transform/*.cc + src/<backend>/*

④ 后端抽象：BackendModule（注册清单）+ BackendContext（一次解析、冻结）
            + PassPipeline / DeviceCodegen / HostCodegen(+Hook) / ExecutionBackendSpec
   两条硬规则：同 kind 多 manifest 必须给 supports_target 谓词；
              engine 里不许按 target.kind.name 分派。

⑤ 实测数字（脚本可复现）：CUDA 73 步 / Ascend 80 步；同名 pass 39 个；
   条件执行 CUDA 7 步 / Ascend 17 步。

⑥ 版本边界：v0.1.15 有 ascend 后端与多后端抽象；没有 pto adapter、
           没有 AutoSimtVF、没有 T.PersistentKernel。判断一律靠 grep，不靠 PR 号。
```

如果只能记住一句话：**后端差异被压缩在 `context.lower(mod)` 这一行里，
顺序写在 Python 的 pipeline 里，行为写在 C++ 的 `src/transform` 里。**

## 延伸阅读

* 官方对多后端设计的完整说明（本章 4.4 节的一手来源）：
  `_vendor/tilelang-v0.1.15/tilelang/backend/README.md`（尤其 Design Model、
  Pass Pipeline、Ownership Rules 三节）。
* 官方编译流程的六步描述：`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md:32-50`。
* 目标（Target）选择与 `auto` 探测顺序：
  `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/targets.md:10-22`。
* 昇腾 950 后端的官方说明（安装、Quick Start、与 Ascend C 的分工）：
  `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md`。
* corpus 里两份编者综合文档（二手材料，用于对照，不作为源码证据）：
  `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md`（编译期参与者与承重 pass）
  与 `../tilelang-corpus/90-synthesis/03-ascend-route.md`（昇腾路线与前沿清单）。
* 下一章：第 20 章会把本章列出的 39 个同名 pass 与两侧的特有 pass 逐个剖开；
  第 21 章讲 layout 代数；第 22 章讲流水与同步；第 24、25 章讲两条后端各自的 codegen。

本章所有引用的完整清单与核验状态见 [references.md](references.md)。
