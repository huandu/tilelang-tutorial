# 第 20 章 IR 与编译期管线精读

## 学习目标

读完本章，你应当能够：

1. 拿到任意一段 TileLang 打印出的 TIR，**分清** `IRModule` / `PrimFunc` / `Buffer` /
   `Var` / `PrimExpr` / `SBlock` / 轴（`thread_extent` 与 grid 循环）七类节点，
   并指出每个节点的职责边界。
2. 打开 `tilelang/cuda/pipeline.py` 与 `tilelang/ascend/pipeline.py`，**按九组职责**
   复述完整 pass 序列，而不是背一串名字——说得出每一组在解决什么问题。
3. 对本章精读的 11 个承重 pass（8 个通用 + 3 个昇腾特有）逐个说出：输入输出 IR 的
   差异、算法思路、源码位置、常见失败模式。
4. 用第 12 章介绍的 `lower_trace` / `pass_diff`，对一个真实 kernel 做**逐 pass 读 IR**
   的可复现流程，并知道每一步应当看到什么。
5. 判断一条 pass 相关的说法是「v0.1.15 事实」还是「仅 main / 旧名字」，并给出可复现的
   核验命令（例如 `InsertNd` 与 `InsertNd2Nz`）。

## 前置知识

* 第 4 章「架构总览与一次编译的全流程」：知道五个阶段（前端 → 编译期 → codegen →
  host 封装 → 运行时）与那个唯一的分岔点 `context.lower(mod)`。本章是它第 4 节的展开。
* 第 12 章「调试与性能分析」：本章用的工具（`TL_LOWER_TRACE`、`lower_trace()`、
  `pass_diff`、`compile_only`、布局可视化）全部在那里介绍过，本章只讲**怎么用它们读 IR**。
* 会读 Python 与基本的 C++；知道「pass 就是一次 IR → IR 的函数」。
* **不需要**读过 TVM 源码，也不需要写过 CUDA。本章出现的 `tirx` / `s_tir` 会先解释。

本章回答一个问题：**`context.lower(mod)` 这一行里，到底发生了什么。**
它不讨论性能，也不给你可以直接跑出结果的数字——它给你一张能随时重画的地图，以及
一套自己观察编译器的流程。

---

## 1. 问题的由来

### 1.1 为什么「看 IR」比「读 C++」先做

第 4 章列出了一次编译的全部阶段，也列出了 pass 的顺序。但真正改过编译器的人都知道：
**顺序表不等于理解**。三个具体问题会立刻冒出来：

* 同一份 Python kernel，为什么 CUDA 与 Ascend 的 pass 数不同（73 vs 80）？
* `LayoutInference` 到底往 IR 里写了什么？我在 `T.copy` 后面加一句会怎样？
* 报错说 `Pipeline planning error: Multiple writes to overlapping buffer regions`，
  它出现在哪一步、是不是我写的代码有问题？（见 4.3.1）

这三个问题都不是「API 怎么用」，而是**结构问题**。结构问题的答案应当来自两样东西：
**IR 本身**，以及**每个 pass 对 IR 做了什么**。

### 1.2 本章的三条纪律

1. **所有 IR 片段来自锁定版本。** 本章出现的 TIR 全部逐字来自
   `_vendor/tilelang-v0.1.15/`（`examples/eager_jit/eagerjit.en.ipynb` 的打印输出、
   `testing/python/transform/` 里的 before/after、`docs/compiler_internals/` 的
   示例）。抽取脚本在 `code/02-ir-snippet-extract/`，可现场复核。**本章不含任何
   手写伪造的 IR。**
2. **所有行号现场核对。** 引用格式统一为
   `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:129`。pass 的**顺序**在 Python
   里，pass 的**行为**在 C++ 里（`src/transform/*.cc`、`src/ascend/transform/*.cc`）。
3. **不写没验证过的运行结果。** 本机是 macOS（Apple Silicon），没有 NVIDIA GPU、
   没有昇腾 NPU，也没有安装 tilelang。因此凡涉及「跑起来是什么样」的结论，一律写成
   「应当是什么样」并给出验证步骤，并标明验证级别。

### 1.3 本章与相邻章节的分工

| 章 | 只讲什么 |
| --- | --- |
| 第 4 章 | 阶段与坐标：谁在什么位置 |
| **第 20 章（本章）** | IR 长什么样 + 每个 pass 怎么改 IR + 怎么自己看 |
| 第 21 章 | 布局代数：`LayoutInference` 的约束怎么解 |
| 第 22 章 | 流水与同步：`InjectSoftwarePipeline` / `ThreadSync` 的算法细节 |
| 第 23 章 | 昇腾自动调度：`AutoSchedule` 的 Z3 建模与求解 |
| 第 29 章 | 自己写一个 pass 并挂进 pipeline |

也就是说：本章要让 pass 的「名字」变成「你知道它会对 IR 做什么」。

---

## 2. 概念与模型

### 2.1 三种 IR 方言：`tir` / `tirx` / `s_tir`

TVM 里有三代 IR 数据结构共存，读 TileLang 源码时会同时看到：

| 方言 | 在读代码里的样子 | 特点 |
| --- | --- | --- |
| `tir` | `tvm.tir`、`T.Buffer` | 经典 TIR；`Buffer` 仍有 strides、`For` 有 `kind` |
| `tirx` | `tvm.tirx`、`T.launch_thread` | TileLang 主要使用的方言，`print(func)` 输出的就是这个 |
| `s_tir` | `tvm.s_tir`、`T.block`、`T.reads` | 带 **SBlock** 的「调度版」TIR，`reads`/`writes`/`alloc_buffers`/`annotations` 都挂在块上 |

一条可以立刻验证的规则：**TileLang 打印的 IR 里出现 `with T.block(...)`、
`T.reads()`、`T.block_attr(...)`，就是 `s_tir` 的块结构；出现 `T.launch_thread`、
`T.match_buffer`，就是 `tirx`。** 二者混在同一个函数里是正常的——
`tilelang/transform/__init__.py` 里两种包装函数并排存在，pipeline 里也是
`s_tir.transform.HoistIfThenElse` 与 `tirx.transform.Simplify` 混用。

### 2.2 七个节点，一次说清

下面七个词会贯穿全章。先给一句定义，下一节再用真实 dump 逐一对上。

| 节点 | 一句话定义 | 关键字段 |
| --- | --- | --- |
| `IRModule` | 一个「编译单元」，是 `GlobalVar → BaseFunc` 的字典；一个 kernel 通常是 `{"main": PrimFunc}` | `functions` |
| `PrimFunc` | 一个函数 = 参数 + 一个 `Stmt` 体 + 属性 | `params`、`buffer_map`、`body`、`attrs` |
| `Buffer` | 一块带形状/类型/**存储作用域**的内存对象 | `shape`、`dtype`、`data`（一个 `Var`）、`scope()` |
| `Var` | 变量：循环变量、参数句柄、buffer 的 data 指针 | `name_hint`、`dtype`、`type_annotation` |
| `PrimExpr` | 表达式树的节点基类：`Var` / `IntImm` / `Add` / `Ramp` / `Call` … | 具体子类各不同 |
| `SBlock` | 调度块：声明式地记录 `reads` / `writes` / `alloc_buffers` / `annotations`，并包住一段 `body` | 见 2.3 |
| 轴（axis） | 「谁并行」：grid 轴是 `For(kind=THREAD_BINDING)`；线程轴是 `AttrStmt("thread_extent")` | `thread_tag`（`blockIdx.x` / `threadIdx.x` / 昇腾 `cthread`） |

### 2.3 `SBlock` 是这场编译的「票据」

`SBlock` 值得单独讲，因为 TileLang 的大部分 pass 不是「改成另一个语句」，而是
**往块的注解上写/读事实**。一个块携带四类信息：

```text
with T.block("tilelang_root"):
    T.reads()            # 声明式读集合
    T.writes(...)        # 声明式写集合
    T.block_attr({...})  # 编译期元数据
    ...                  # body
```

块的**名字** `"tilelang_root"` 是一个硬编码常量：
`_vendor/tilelang-v0.1.15/src/transform/common/attr.h:17` 写着
`constexpr const char *DeviceMainBlockName = "tilelang_root";`，
`:24` 的辅助函数直接用 `node->name_hint == DeviceMainBlockName` 判断「这是不是 kernel
主块」。本章后面会多次遇到「必须挂在 `tilelang_root` 上」的约束
（例如昇腾多缓冲的版本计数）。

块上的**注解**是 pass 之间传递信息的主要通道。三个你会立刻用到的键：

| 注解键 | 谁写 | 含义 |
| --- | --- | --- |
| `layout_map` / `parallel_loop_layout` | `LayoutInference` | buffer 与并行循环的布局 |
| `software_pipeline_stage` / `software_pipeline_order` | `PipelinePlanning` | 语句属于哪个流水级、什么顺序 |
| `tl.launch_threads` | 前端 `T.Kernel(threads=...)` | 每块线程数（物化时读取） |

### 2.4 两条 pipeline 的整体形状

`CUDA_PIPELINE = PassPipeline("cuda", CUDAPassPipelineBody)`
（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:284`）与
`ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)`
（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:190`）都只是把
「一个 `IRModule` + `Target` → 一个新的 `IRModule`」的函数包起来。
`PassPipeline.lower()`（`tilelang/backend/pass_pipeline/pipeline.py:22`）做的唯一
额外事情是给这次 lowering 开一个具名的 instrumentation 作用域，好让
IR Lower Trace 能把每个 pass 归到 `pipeline_c` / `pipeline_ascend` 名下。

真实规模（`code/01-pass-order/dump_pass_order.py` 实测）：

```text
CUDA    : tilelang/cuda/pipeline.py    73 次 pass 调用
Ascend  : tilelang/ascend/pipeline.py  80 次 pass 调用
两侧同名的 pass（模块限定名去重后取交集）: 39 个
CUDA 独有 26 个，Ascend 独有 31 个（均按模块限定名去重）
```

这组数字本身没有意义；有意义的是它说明的**结构**：两条 backend 共享一条很长的
「标准 lowering 走廊」，只在两头各自换料。完整方位图见
`figures/fig-02-pipeline-phases.md` 与 `figures/fig-03-lowering-corridor.md`。

---

## 3. 用法：把一段真实 kernel 的 IR 读到骨头

### 3.1 选一个最小 kernel，拿到它的 IR

我们从 `_vendor/tilelang-v0.1.15/examples/eager_jit/eagerjit.en.ipynb` 里挑一个最小
kernel。下面是该 notebook **第 49 个 cell 的真实输出**（不是我们手写的），
也就是 `print(prim_func)` 的结果：

```python
# from tvm.script import tirx as T

@T.prim_func
def foo(x_handle: T.handle):
    x = T.match_buffer(x_handle, (2,), strides=(1,))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads()
        idx = T.Buffer((1,), "int32", scope="local.var")
        T.writes(x[T.min(1, idx[0]):T.min(1, idx[0]) + (T.max(1, idx[0]) + 1 - T.min(1, idx[0]))])
        T.block_attr({"tl.local_var_init": {idx.data: 0}})
        idx = T.alloc_buffer((1,), "int32", data=idx.data, scope="local.var")
        x[1] = T.float32(1.0)
        _tmp: T.int32 = idx[0]
        x[_tmp] = T.float32(1.0)
```

复现方式（不需要 GPU，也不需要 tilelang）：

```bash
cd chapters/20-ir-and-pass-pipeline/code/02-ir-snippet-extract
python3 extract_ir_snippets.py --only notebook
```

### 3.2 逐节点讲解

**第 1 层：函数与参数。** `@T.prim_func def foo(x_handle: T.handle)` 说明
`IRModule` 里有一个名为 `foo` 的 `PrimFunc`。它的参数是一个**句柄**（`T.handle`），
不是张量——因为 TIR 层面只知道「有个指针」，不知道它指向多大的张量。
`x = T.match_buffer(x_handle, (2,), strides=(1,))` 把句柄**绑成一块 Buffer**：
形状 `(2,)`、`strides=(1,)`（连续）。这一步在 `PrimFunc` 的 `buffer_map` 里建立
`x_handle → x` 的映射。**这就是「参数」与「Buffer」的关系。**

**第 2 层：axis（轴）。** 接下来四行是本 kernel 的**并行结构**：

```text
bx = T.launch_thread("blockIdx.x", 1)     # 网格轴：1 个 block
tx = T.launch_thread("threadIdx.x", 128)  # 线程轴：128 线程
ty = T.launch_thread("threadIdx.y", 1)
tz = T.launch_thread("threadIdx.z", 1)
```

四个 `Var` 各自绑定到一个**具名 axis**。注意这里**同时**出现了「grid 轴」与
「线程轴」——这是 `T.Kernel(...)` 语义的直接体现：TileLang 的 kernel 体是
**块级（block-level）** 的，线程轴在物化前只是一个占位 `Var`。
`threadIdx.y/z` 恒为 1，是因为这个例子只用了一维线程。

**第 3 层：主块。** `with T.block("tilelang_root"):` 是 kernel 主体。它内部：

* `T.reads()` / `T.writes(...)`：声明式读写集合。写集合用的是一段
  `min/max` 区间表达式，正说明**块级写区域可以是一个范围，而不是一个点**。
* `T.block_attr({"tl.local_var_init": {idx.data: 0}})`：一个**编译期元数据**，
  意思是「`idx` 这块 `local.var` 需要初始化成 0」。后面 `StorageRewrite` 会读它
  （见 4.3.5）。

**第 4 层：分配与作用域。**

```text
idx = T.Buffer((1,), "int32", scope="local.var")
idx = T.alloc_buffer((1,), "int32", data=idx.data, scope="local.var")
```

这两行是同一块 buffer 的「声明」与「分配」。**`scope="local.var"` 是关键信息**：
它不是 shared、不是 fragment，而是「一个标量变量」。`data=idx.data` 说明两次
出现的是同一块内存。作用域决定了这块内存在后面所有 pass 里的命运：
`local.var` 不会进 shared memory 合并，也不参与线程级同步。

**第 5 层：语句。**

```text
x[1] = T.float32(1.0)          # BufferStore：写常量下标
_tmp: T.int32 = idx[0]         # LetStmt：绑定一个模板变量
x[_tmp] = T.float32(1.0)       # BufferStore：用变量下标写
```

`x[1] = ...` 是 `BufferStore`，它的 `indices` 是一个常量 `1`；
`x[_tmp] = ...` 的 `indices` 是 `LetStmt` 引入的 `_tmp`。
**这就是 `PrimExpr` 与 `Stmt` 的分工**：`Stmt` 描述「发生了什么」，
`PrimExpr` 描述「用什么算出来」。

### 3.3 物化：axis 是怎么变成 launch 的

上面的 dump 是**物化之后**的形式（已经有 `blockIdx.x` / `threadIdx.x`）。
在物化**之前**，`T.Kernel` 只记录网格循环与线程占位符。这一点写在
`_vendor/tilelang-v0.1.15/src/transform/materialize_kernel_launch.cc` 的文件头注释里：

```text
T.Kernel traces into

  for bx in thread_binding(blockIdx.x):        # one per grid axis
    tx = tl.launch_thread_idx(0)               # x, y, z placeholders
    ty = tl.launch_thread_idx(1)
    tz = tl.launch_thread_idx(2)
    block tilelang_root { annotations: tl.launch_threads?, ... }
```

`MaterializeKernelLaunch` 就是把这个「目标无关的 launch 嵌套」翻译成后端形式：
**grid 循环**变成 `thread_extent`，**线程占位符**要么绑成 `threadIdx.*` 的
`thread_extent`，要么被丢掉。这一步的两个开关（`lower_grid_binding` /
`lower_thread_binding`）在锁定版本里的调用是：

```text
CUDA   : MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)   # pipeline.py:70
Ascend : MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False,
                                 default_threads=None,
                                 unsupported_annotations=["cluster_dims"],
                                 launch_dim_tags=["cthread"])                       # pipeline.py:45
```

一句话读法：**CUDA 保留线程域（SIMT），Ascend 丢掉线程域、只保留一维 core 网格**
（`cthread` 是昇腾的子块编号维度，由代码生成读回）。这组参数差异就是
`fig-02` 里「① 前端合法化」第一行的全部含义。

### 3.4 一个 pass 对 IR 做了什么：以 `PipelinePlanning` 为例

结构问题最终要落到「一个 pass 前后 IR 哪里变了」。最干净的样本在
`_vendor/tilelang-v0.1.15/testing/python/transform/test_tilelang_transform_pipeline_planning.py`。
它直接给出 `before` / `after` 两个 `PrimFunc` 并用结构相等断言二者一致——
也就是说，这两个函数就是 `PipelinePlanning` 的**输入/输出规格**。

输入（节选）：一个 `T.Pipelined(32, num_stages=3)` 循环里两个 `T.copy` 加一个 `T.gemm`。
输出（节选）：

```python
for ko in T.serial(
    32,
    annotations={
        "software_pipeline_async_producer_groups": [0, 0, -1],
        "software_pipeline_async_producers": [1, 1, 0],
        "software_pipeline_async_stages": [0],
        "software_pipeline_order": [0, 1, 2],
        "software_pipeline_stage": [0, 0, 2],
        "tl_pipelined_num_stages": 3,
    },
):
    T.copy(A[by * 128, ko * 32], A_shared)
    T.copy(B[ko * 32, bx * 128], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

**看清楚发生了什么**：循环体一个字都没改，循环从 `T.Pipelined` 变成 `T.serial`，
多出来的是**注解**。三个长度等于语句数的数组 `order` / `stage` /
`async_producers` 把「这三条语句分别属于哪一级、按什么顺序发射、是不是异步生产者」
编码进去。真正的循环重写留给下一步 `InjectSoftwarePipeline`。

这就是 TileLang 编译期最典型的一种 pass：**先规划、后改写**。

---

## 4. 实现机制

### 4.1 九组职责：两条 pipeline 的完整对照

本节不重复第 4 章的表格，而是按**职责**把两侧归成九组，重点讲「这组在解决什么」。
完整行号表见 `figures/fig-02-pipeline-phases.md`；下面是职责说明。

**① 前端合法化（legalize）。** 把目标无关的前端 IR 变成一个「形状正确、格式合法」
的后端 IR。三件事：绑定 target（`BindTarget`）、物化 launch
（`MaterializeKernelLaunch`）、补齐前端留下的粗糙边角
（`AddWrapperForSingleBufStore`、`LegalizeNegativeIndex`、`InjectAssumes`、
`CanonicalizeLegacyReducer`、`VerifyReducerEpoch`、`VerifyBufferInit`）。
CUDA 在这组里额外做 warp specialization 与 Blackwell 2SM 预降级；Ascend 额外做
`RewriteDualCopy`（把 L0C→UB 的双目标搬运展开）与 `LegalizeSimdMerging`
（把 MODE_MERGING 的 SIMD 赋值显式化成读-改-写）。

**② 流水（pipeline）。** 把 `T.Pipelined` 变成多缓冲的软件流水。
CUDA 用通用两件套 `PipelinePlanning` + `InjectSoftwarePipeline`；Ascend 把这一组
折进第 ③ 组的自动调度段——因为昇腾的流水要和多核分派一起解。

**③ 布局（layout）。** 推导每个 fragment / shared buffer 的物理布局，并把它写成
注解。CUDA 用 `LayoutInference`；Ascend 用 `AscendLayoutInference` 加一串物理化
前处理（`NormalizeAscendFractalStorage`、`InsertNd2Nz`、`AscendInsertOOBPadding`），
因为 Cube 的 fractal / ND→NZ 布局是硬约束。**这是两条 pipeline 的第一处真正分岔。**

**④ tile op 降级（lowering）。** 把「tile 级」概念全部消灭：
`T.copy` → 搬运指令（TMA / cp.async / LDG-STG / 昇腾 MTE），
`T.gemm` → MMA 序列（mma / wgmma / tcgen05 / Cube）。之后 IR 里不再有 tile op。
CUDA 是 `LowerTileOp`；Ascend 是 `AscendLowerTileOp` 加
`AscendSimdVFLowerParallel`（把 SIMD_VF 块内的 `T.Parallel` 降成寄存器级向量算子）。

**⑤ 内存规划（memory planning）。** 决定每块内存在物理上放哪、能不能复用。
`FlattenBuffer` 把多维 buffer 压成一维；`StorageRewrite` 分析活跃区间并复用局部存储；
CUDA 的 `MergeSharedMemoryAllocations` 与 Ascend 的 `MergeUBAllocations` 把多份
片上缓冲合并进有限的 shared / UB 空间。**这是第二处分岔**：两侧的片上内存模型不同。

**⑥ 向量化与展开（vectorize / unroll）。** 把循环变成向量访问与展开体：
`VectorizeLoop`（把可向量化的 `For` 变成 `Ramp` 访问）、`LoopUnswitching`、
`UnrollLoop`、`RenormalizeSplitPattern`、`HoistIfThenElse`。Ascend 多一个
`RewriteFp4ToFp4x2`（把 fp4 存储改成按字节打包的成对形式）与
`AscendRemoveNoOp`（基于等价分析删空语句）。

**⑦ 同步（synchronization）。** 根据读写足迹推导并插入屏障/栅栏。
CUDA 是 `ThreadSync("shared")` + `ThreadSync("shared.dyn")`，外加
`InjectFenceProxy`（Hopper 的 async proxy 栅栏）与 `InjectTcgen05Fence`
（Blackwell 的 tcgen05 栅栏）。Ascend 是 `AscendThreadSync("global"/"shared"/
"shared.dyn")`，外加 `RewriteFlagToBuf`（把同步 flag 归一到 8 槽命名空间，
放不下就溢到共享的 32 槽 get_buf/rls_buf 互斥池）。**这是第三处分岔，也是最容易出
偶发错误的地方。**

**⑧ host/device 切分。** `SplitHostDevice` 把一个 `PrimFunc` 拆成
「host 入口」与「device kernel」两族。之后 host 侧与 device 侧的分析才能分开做
（CUDA 的 `MergeSharedMemoryAllocations` 明确要求跑在它之后，因为合并后的分配点
在每个 device 函数的开头）。

**⑨ API 打包。** `MakePackedAPI` 把 host 入口改写成 TVM-FFI 的打包调用约定，
`LowerDeviceKernelLaunch` 把 device 调用变成真正的 launch，CUDA 再有
`PersistThreadblock` 把 threadblock 变成持久 threadblock。

一句总结：**四组（①–④）是抽象阶梯，三组（⑤–⑦）是物理化，两组（⑧–⑨）是接口化。**
两条 pipeline 在七组上重叠、在三处（流水 / 布局 / 同步与片上内存）分岔。

### 4.2 承重 pass 精读的通用读法

下面每个 pass 都按四问展开：**输入/输出差异 → 算法思路 → 源码位置 → 常见失败模式**。
「输入/输出差异」不是猜的：要么来自该 pass 的官方测试里的 before/after，要么来自
源码里的注释与断言，要么来自 pipeline 里紧挨它的收尾 pass。

### 4.3 通用承重 pass（8 个）

#### 4.3.1 `PipelinePlanning` —— 规划，不改写

* **输入/输出差异**：`T.Pipelined(n, num_stages=k)` 循环 → 同体 `T.serial` 循环 +
  `software_pipeline_stage/order/async_producers/async_producer_groups/async_stages`
  和 `tl_pipelined_num_stages` 注解（3.4 节的完整样例）。
* **算法思路**：`PipelineStageAnalyzer`（`src/transform/pipeline_planning.cc:447`）
  对每条语句收集读写区域，判定它是「生产者」还是「消费者」，据此算每一级的归属
  （stage）、发射顺序（order）、以及能不能走异步（async_*）。`PipelinePlanner`
  （:1002）随后把这些数组写回循环注解。配置项 `tirx.use_async_copy`（默认 `True`，
  :1360）决定是否允许异步生产者。
* **源码位置**：pass 在 `src/transform/pipeline_planning.cc:1356`；Python 包装在
  `tilelang/transform/__init__.py:19`；CUDA pipeline（`pipeline.py:129`），
  Ascend 侧**不调用**它。
* **常见失败模式**：
  * `LOG(FATAL) "Pipeline planning error: Multiple writes to overlapping buffer
    regions detected"`（:663）——同一个流水级里有两条语句写同一块 buffer 的重叠区域，
    planner 无法给出一致的版本化方案。修法是拆开写区域，或把其中一条挪到另一级。
  * 依赖传播超过最大迭代 `LOG(FATAL) "Exceeded maximum iterations ... This may
    indicate a cyclic or pathological dependency graph"`（:753）——通常说明循环体里
    存在自依赖的标量计算。

#### 4.3.2 `InjectSoftwarePipeline` —— 改写，引入多缓冲

* **输入/输出差异**：带上述注解的 `T.serial` 循环 → 展开成
  prologue / steady-state / epilogue 三段，循环体按 stage 重排，被复用的 buffer
  变成**多个物理版本**，并出现 `ptx_commit_group` / `ptx_wait_group` 一类异步等待。
* **算法思路**：`PipelineRewriter`（`src/transform/inject_pipeline.cc:1179`）的
  `BuildPipeline()`（:1208）分四步：分析每个 buffer 需要几个版本 → 生成版本化访问
  → 把循环体按 `stage/order` 重排成软件流水 → 在正确的边界插入 commit/wait。
  `PipelineBodyRewriter`（:1017）负责单条语句的版本改写。
* **源码位置**：pass 在 `src/transform/inject_pipeline.cc:4033`；Python 包装在
  `tilelang/transform/__init__.py:52`；CUDA pipeline（`pipeline.py:130`）。
* **常见失败模式**：`ICHECK "InjectSoftwarePipeline: cyclic scalar Bind dependency
  involving <var>"`（:1340）——循环体里的标量绑定互相依赖成环。修法是把这个标量
  移出流水循环，或在外面预先算好。

#### 4.3.3 `LayoutInference` —— 求解并写下布局

* **输入/输出差异**：tile op 图上只有逻辑形状 → `tilelang_root` 块上出现
  `layout_map` 注解，并行 `For` 上出现 `parallel_loop_layout`（以及可能的
  `parallel_loop_predicate` / padding guard）注解。**循环本身不展开**。
* **算法思路**：`LayoutInferencer::Substitute`（`src/transform/layout_inference/
  layout_inference.cc:1553`）先跑 `ParallelLoopFuser::Fuse`，再由
  `BufferUseDefCollector`（:237）收集所有参与布局推导的算子，然后在
  `RunInferStep`（:243）里**逐个调用算子的 `InferLayout`**（:279）做约束传播。
  拿到布局后，`LayoutInferencer`（:1551）只把结果写成注解，把「怎么用布局」留给
  `LowerTileOp`。
* **源码位置**：pass 在 `src/transform/layout_inference/layout_inference.cc:1647`；
  Python 包装在 `tilelang/transform/__init__.py:30`；CUDA pipeline
  （`pipeline.py:134`）。Ascend 用自己的 fork：`src/ascend/transform/
  layout_inference.cc:1684`（`AscendLayoutInference`）。
* **常见失败模式**：
  * `LOG(FATAL) "bad_optional_access while inferring layout for op <id> (<type>) at
    level <level>"`（:290）——某个算子在某个求解层拿不到可行布局，通常是形状/类型
    不满足该算子的布局约束（例如某条 tile 边不是向量宽度的整数倍）。
  * 布局冲突（`LayoutConflict`）——两条路径给同一块 buffer 推了互斥布局。第 21 章
    会讲怎么读这个冲突。

#### 4.3.4 `LowerTileOp` —— 消灭 tile op

* **输入/输出差异**：`tl.tileop.copy` / `tl.tileop.gemm` 等调用 → 具体指令或子图
  （`tl.ptx_cp_async`、TMA 描述符初始化、`ptx_mma` / `wgmma`、昇腾的 MTE/Cube 序列），
  并且**该函数带上 `tl.has_tma` 属性**、必要时带上 `tl.smem_alignment_map`。
* **算法思路**：`LowerTileOpPass::Substitute`（`src/transform/lower_tile_op.cc:209`）
  遍历语句，对每个 tile op 调用它自己的 lowering；`makeBufferWithLayout`（:43）
  把 fragment 的物理存储变成普通 `local` buffer（因为布局已定，寄存器槽位是确定的）。
  之后 `RemapBufferRewriter` 与 `LayoutRemapRewriter` 统一改写引用。
  `f = WithAttr(std::move(f), kHasTMA, Bool(substituter.has_tma_))` 把「这次
  lowering 真的生成了 TMA」记成属性，供后续阶段选择 pipeline（:233-235）。
* **源码位置**：pass 在 `src/transform/lower_tile_op.cc:1477`；Python 包装在
  `tilelang/transform/__init__.py:41`；CUDA pipeline（`pipeline.py:142`）。
  Ascend 用 `src/ascend/transform/lower_tile_op.cc:200`（`AscendLowerTileOp`）。
* **常见失败模式**：
  * `ICHECK "LowerTileOpPass: Require the target attribute"`（:224）——忘了
    `BindTarget`。单独调这个 pass 做实验时最容易踩。
  * `ICHECK "Padding map annotation is missing"`（:170）——带 padding 的
    昇腾路径要求先跑 `AscendInsertOOBPadding`。
  * `ramp of more than N lanes` 之类出现在 **codegen** 而不是这里（见第 12 章
    的错误清单）——因为超宽向量是 LowerTileOp 之后才被向量化成 Ramp 的。

#### 4.3.5 `StorageRewrite` —— 复用局部存储

* **输入/输出差异**：多个小分配 + 各自的生命周期 → 合并/复用的少数分配（下标被
  改写成带偏移的形式）。
* **算法思路**：三段。`LinearAccessPatternFinder`（`src/transform/storage_rewrite.cc:100`）
  线性化语句流并记录每个语句的分配信息；`StoragePlanRewriter::LivenessAnalysis` +
  `PlanMemory`（:430 起）做活跃区间分析与合并；最后 `VisitBufferAccess`（:456）
  把所有访问重写到新的分配上。**关键事实**：在 v0.1.15 里 reuse 默认关闭——
  源码注释写明「Always disable reuse currently, for shared memory reuse we depend
  on MergeSharedMemoryAllocations pass, for register reuse we depend on nvcc」
  （:1933-1935）。所以 `StorageRewrite` 实际做的是局部存储的规划与
  `local.var` 初始化（`local_var_init_map`）落地。
* **源码位置**：pass 在 `src/transform/storage_rewrite.cc:1928`（`PassStorageRewrite`）；
  Python 包装在 `tilelang/transform/__init__.py:382`；CUDA pipeline
  （`pipeline.py:213`）。**Ascend 不调用它**，取而代之的是
  `RewriteAscendBufferVersionLayout`（:135）与 `MergeUBAllocations`（:171）。
* **常见失败模式**：改完共享内存复用后结果错——先确认是不是
  `MergeSharedMemoryAllocations` 的 `disable_reuse`（CUDA）或
  `MergeUBAllocations(disable_reuse=...)`（Ascend）被打开了；这两处才是共享内存
  复用的真正开关。

#### 4.3.6 `ThreadSync` —— 插入线程级屏障

* **输入/输出差异**：共享内存上的相邻读写之间 → 出现
  `T.tvm_storage_sync("shared")` / `T.tvm_storage_sync("shared.dyn")` 调用。
* **算法思路**：`TileLangThreadSyncPlanner`（`src/transform/thread_storage_sync.cc:603`）
  是一个基于约束集的访问分析器（`ConstrVisitor`）：为每个语句收集 `AccessEntry`
  （线程集合、buffer 区间、依赖约束、是否来自 async copy、是否原子 RMW），
  用内存冲突规则推导「谁和谁之间需要屏障」，再由 `ThreadSyncInserter`（:212）把
  屏障插到正确位置。它**特意不**在原子与原子之间插屏障（否则会把原子串行化）。
* **源码位置**：pass 在 `src/transform/thread_storage_sync.cc:2091`；Python 包装在
  `tilelang/transform/__init__.py:108`；CUDA pipeline 调两次（`pipeline.py:258-259`）。
  Ascend 用 `AscendThreadSync`（`src/ascend/transform/thread_storage_sync.cc`）。
* **常见失败模式**：
  * `LOG(FATAL) "[ThreadSync] Cannot lower a required shared-memory sync inside a
    divergent region with N participating threads (not a multiple of warp size)"`
    （:327）——线程数不是 warp 的整数倍时无法安全降成一个 `bar.sync`。
  * `LOG(FATAL) "[ThreadSync] Cannot lower a required shared-memory sync inside the
    <branch> branch: its participating threads are not warp-uniform"`（:1068）——
    同步被放在一个非 warp 一致的分支里。
  * `LOG(WARNING) "[ThreadSync] Hoisting sync out of an if ... This is not a fix"`
    （:1085）——**看到这条警告不要当没事**：它明确说「这不是修复，竞态仍在」，
    正确做法是加 `T.assume` 约束条件，让屏障留在原地。

#### 4.3.7 `SplitHostDevice` —— 拆成 host 与 device

* **输入/输出差异**：一个 `PrimFunc` → 原函数（host 侧）+ 一个新的
  `<global_symbol>_kernel` 函数（device 侧）；设备函数被标记为 device kernel，
  若干属性（`kNonRestrictParams`、`cluster_dims`、`tl.smem_alignment_map`）被
  **从 host 侧搬走、挂到 device 侧**。
* **算法思路**：`HostDeviceSplitter` 顺着 `AttrStmt` 找到 `thread_extent` 的边界，
  把「设备区域」抽出来塞进新的 `IRModule`，再 `ConvertSSA()` 收尾。若函数里
  根本没有设备区域但它是 entry func 且 body 是 no-op，`ForceSplit`（定义在
  :114，调用点 :617-620）会强制拆出一个空设备 kernel——这是为了保持 ABI 一致。
* **源码位置**：pass 在 `src/transform/split_host_device.cc:631`，核心逻辑
  `::tvm::tl::SplitHostDevice` 在 :583；Python 包装在
  `tilelang/transform/__init__.py:285`；CUDA pipeline（`pipeline.py:240`），
  Ascend pipeline（`pipeline.py:166`）。
* **常见失败模式**：**拆分之后 IR 是「两个模块」**。后半段做调试打印时如果只看一个
  模块，会误判「代码丢了」。engine 层在 `tilelang/engine/lower.py:138-139` 用两次
  `Filter` 把它们分开。

#### 4.3.8 `MakePackedAPI` —— 打包成 FFI 入口

* **输入/输出差异**：`fn(A, B, C)` 形式 → TVM-FFI 打包约定，入口形如
  `(args, num_args)`；参数个数、指针种类、dtype、shape、stride、device 等检查被
  自动插入（host 侧 auto-validation）。输出张量由谁分配由 `tilelang_out_idx` /
  `tilelang_callee_allocated_outputs` 两个属性决定。
* **算法思路**：`RequiresPackedAPI`（`src/transform/make_packed_api.cc:317`）先判断
  这个函数该不该被改：已经带非默认 calling convention 的函数**不改**，源码内核
  （`kCodeBlockSource`）**不改**，没有 `global_symbol` 的内部函数**不改**。
  真正改写时先抽出 `tl.assume` 运行期检查（`AssumeRuntimeCheckExtractor`），
  再按既定模板构造新的打包参数。
* **源码位置**：pass 在 `src/transform/make_packed_api.cc:1072`，核心
  `MakePackedAPI(PrimFunc, ...)` 在 :386；Python 包装在
  `tilelang/transform/__init__.py:190`；CUDA pipeline（`pipeline.py:273`），
  Ascend pipeline（`pipeline.py:184`）。
* **常见失败模式**：`ICHECK "MakePackedAPI required the function to be annotated
  with tvm::attr::kTarget, but the function only has attributes ..."`（:401）——
  同样是没有 target 属性。手动串 pass 做实验时几乎必踩。

### 4.4 昇腾特有承重 pass（3 个）

#### 4.4.1 `AutoSchedule` —— 把调度写成约束求解问题

* **输入/输出差异**：带资源标签的任务单元（`MaterializeScheduleUnits` 的产物）→
  被 Z3 重排过的 `ScheduledTIR`（每个节点带上开始时间/顺序），并可能覆盖
  `buffer_versions`。它**不改写语句本身**，只重排与打标（改写交给
  `LowerScheduledTIR` / `MaterializeMultiBuffer`）。
* **算法思路**：`ScheduleBuilder`（`src/ascend/transform/auto_schedule.cc:158`）
  递归遍历 `IRStructure`；对直线型子列表调 `Z3SchedulePython`（:452）、对循环体调
  `Z3SchedulePythonLoop`（:680）。这两个函数都通过 FFI 找 Python 侧实现
  （`:474` 的 `tl.transform.z3_schedule_python`、`:740` 的
  `tl.transform.z3_schedule_loop_python`），实现在
  `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py`（注册点
  在 :228 与 :784，核心在 :67 与 :268）。找不到 Python 求解器时会**告警并回退**，
  而不是静默。
* **源码位置**：pass 在 `src/ascend/transform/auto_schedule.cc:1371`；pipeline
  （`tilelang/ascend/pipeline.py:109`），整段（:104-116）受
  `allow_autoschedule()` 守卫（`pipeline.py:22`，配置键
  `tl.enable_auto_schedule`，默认 `True`）。
* **常见失败模式**：`LOG(FATAL) "[ScheduleRecursive] Unknown IRStructure type"`
  （:449）——遇到了建模器不认识的 IR 节点类型，通常是某个 pass 产生了没被
  `ir_structure` 编解码覆盖的新语句。回退路径（Python 求解器缺失）只告警不报错，
  但**调度结果会退化**，此时应检查 FFI 是否注册成功。

#### 4.4.2 `InsertSync` —— 在调度结果上插同步

* **输入/输出差异**：已调度、带多缓冲时钟的 IR → 每个同步点带上 flag/事件，
  并产出 buffer 复用契约（供 `MergeUBAllocations` 读取）。
* **算法思路**：文件头注释写着它的三件事：**「Synchronization analysis, flag
  allocation, and insertion」**（`src/ascend/transform/insert_sync.cc:19-20`）。
  `InsertKernelSync` 依据任务读写足迹找同步点，给每个点分配 flag，再插入
  `set_flag`/`wait_flag` 形式的同步。它和 CUDA 的 `ThreadSync` 的根本区别是：
  CUDA 只需在线程之间插屏障，昇腾要处理**多条硬件流水**
  （MTE1/2/3、Cube、Vector、Fixpipe）之间的完成协议。
* **源码位置**：pass 在 `src/ascend/transform/insert_sync.cc:2551`；pipeline
  （`tilelang/ascend/pipeline.py:113`）。配套的 `RewriteFlagToBuf`（:179）把 flag
  归一到 8 槽命名空间/溢出的互斥池，文档见
  `tilelang/ascend/transform/__init__.py` 的 `RewriteFlagToBuf` 包装函数。
* **常见失败模式**：`ICHECK "Failed to find a sync insertion node"`（:155）与
  `ICHECK "path_end->IsControl()"`（:132）——说明依赖路径的锚点不在预期位置，
  往往由上一个 pass（`AssignCore` / `ResolveCore`）没有把核候选收窄干净导致。

#### 4.4.3 `MaterializeMultiBuffer` —— 逻辑多缓冲落成物理版本

* **输入/输出差异**：逻辑多缓冲时钟（`PrepareMultiBuffer` 建立的计数器）→
  物理 buffer 版本（`manual_buffer_versions` 元数据），并把多缓冲访问改写成
  带版本下标的访问。
* **算法思路**：文件头注释：「Lower prepared logical multi-buffer clocks to
  physical buffer versions **after synchronization insertion**」
  （`src/ascend/transform/materialize_multi_buffer.cc:19-20`）。它必须在
  `InsertSync` 之后：因为同步决定了每个版本的生命周期边界，版本数量才能定死。
  结束时有一条硬性断言 `ICHECK "Multi-buffer materialization must preserve
  tilelang_root"`（:538）——所有改写都必须在主块内部完成。
* **源码位置**：pass 在 `src/ascend/transform/materialize_multi_buffer.cc:544`；
  pipeline（`tilelang/ascend/pipeline.py:114`）。
* **常见失败模式**：`ICHECK "storage_pitch_bits_.count(info.storage)"` 一类断言
  （:123）——某个 buffer 的物理 pitch 没被登记，通常是它绕过了
  `RewriteAscendBufferVersionLayout` 的对齐步骤。修法通常是回看多缓冲的
  **来源**（`T.Pipelined` 的 `num_stages` 与用户手动标注是否冲突）。

### 4.5 codegen、host 封装与可调用对象

pipeline 走完时，`mod` 里已经是两族函数。engine 用
`tirx.transform.Filter`（`tilelang/engine/lower.py:138-139`）按
`CallingConv.DEVICE_KERNEL_LAUNCH` 把它们分开，再分别调用
`context.codegen_device(...)` 与可选的 `host_codegen(...)`。这一段与第 4 章 4.8 节
一致，本章不重复；只需记住：**你能看到的最后一份 IR，就是 `SplitHostDevice` 之后
的 host/device 两个模块。**

---

## 5. 进阶与陷阱

### 5.1 怎么自己观察：逐 pass 读 IR 的可复现步骤

把第 12 章的工具按下面的顺序用，就能把本章的每个结论亲手验一遍。

**第 0 步（不需要任何加速器）：确认顺序表。**

```bash
cd chapters/20-ir-and-pass-pipeline/code/01-pass-order
python3 dump_pass_order.py --pipeline cuda
python3 dump_pass_order.py --pipeline ascend
```

输出里的行号与守卫条件就是 `tilelang/cuda/pipeline.py`、
`tilelang/ascend/pipeline.py` 的真实内容。**先做这一步**，因为后面所有「前后 IR」
都挂在某个具体行号上。

**第 1 步：确认你手上这个 kernel 实际会跑哪些 pass。** 顺序表是「默认 `pass_configs`
下的样子」。真正决定分支的是 `tilelang/backend/pass_pipeline/pipeline_utils.py`
里那几个读 `PassContext.config` 的函数（`:25` 的 `allow_vectorize`、
`:58` 的 `should_enable_race_check` 等）。把你的 `target` 与 `pass_configs`
写下来，再对照表格里的「条件」列。

**第 2 步：看一个 pass 前后的 IR。** 两种做法：

```bash
# (a) 全量：每个 pass 都抓 before/after，写 HTML + .tir 快照
TL_LOWER_TRACE=both python3 my_kernel.py
#     报告默认落在 ./tmp/lower_trace_dir/<脚本名>/report.html

# (b) 定点：只抓你关心的几个 pass（程序 API）
python3 chapters/20-ir-and-pass-pipeline/code/03-lower-trace-walkthrough/lower_trace_walkthrough.py
```

`lower_trace()` 返回一串字典，每个含 `name` / `before_script` / `after_script` /
`insertions` / `deletions` / `changed`。**先看增删行数与 `changed`**，把「一动没动」
的 pass 划掉，再对「动了」的逐个读 diff。（更早的 `pass_diff`（环境变量
`TILELANG_PASS_DIFF`，入口 `tilelang/utils/pass_diff.py`）与它是同一类工具，
但 IR Lower Trace 是它的严格超集——多了 phase 分组、codegen 捕获与崩溃安全的
增量报告；新代码直接用 `lower_trace`。）

**第 3 步：一次只读一个 pass。** 按本章 4.3 的顺序，对每个承重 pass 问三个问题：

```text
前后 IR 差异 → 是不是本章说的那种变化？
如果没有变化   → 是这个 kernel 没触发它（正常），还是它被跳过了？
如果有变化     → 这个变化是「加注解」还是「改语句」？谁消费这个注解？
```

**第 4 步：看布局。** 打开 `TL_LAYOUT_VISUALIZATION_ENABLE`，或对 `T.Layout` /
`T.Fragment` 调 `plot_layout`（第 12 章 3.4 节）。`LayoutInference` 写的注解值
应当与你画出来的一致。

**第 5 步：看 codegen（可选）。** `compile_only` 或 `kernel.get_kernel_source()`
拿到设备源码，核对「向量宽度、swizzle、TMA/MTE 的选择」是否符合你在 IR 里读到的
决定。第 24、25 章会展开两侧的 codegen。

一个反复出现的纪律：**文本变化 ≠ 语义变化 ≠ 性能变化。** IR Lower Trace 比的是
`IRModule.script()` 的文本；某个 pass 让文本变了，不代表结果或性能会变。
每次下结论前，至少确认「谁消费了这次变化」。

### 5.2 版本差异：哪些是 v0.1.15 的，哪些名字不同

严格说，本章能给出的「版本结论」只有一类：**在锁定版本里 grep 得到什么**。
下面每一条都可复现。

**（a）昇腾侧有两条不同的路线，词汇完全不同。** 一份是进主干的
`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py`（80 次 pass 调用，含整套
`AutoSchedule` / `InsertSync` / `MaterializeMultiBuffer` 与 `InsertNd2Nz`）；
另一份是独立适配器仓库的 `_vendor/tilelang-ascend-v0.1.2.000/tilelang/engine/
phase.py`（走 `AscendLowerParallelToVector` / `CrossCorePipeline` /
`AscendSyncInsert` / `AscendStorageRewrite` / `AscendMemoryPlanning`）。
核验：

```bash
cd chapters/20-ir-and-pass-pipeline/code/04-version-diff
python3 compare_ascend_pipelines.py
```

**（b）`InsertNd` vs `InsertNd2Nz`。** 锁定版本里的真实名字是
**`InsertNd2Nz`**（`tilelang/ascend/pipeline.py:91`、
`tilelang/ascend/transform/__init__.py` 的 `def InsertNd2Nz`、
`src/ascend/transform/insert_nd2nz.cc`）。字面量 `InsertNd(`（不带 `2Nz`）
在锁定版本里**找不到**；它是早期综合文档里的旧写法，corpus 的
`90-synthesis/02-compiler-internals-map.md` 顶部也专门加了版本提示。核验：

```bash
python3 compare_ascend_pipelines.py --grep InsertNd2Nz   # 命中多个文件
python3 compare_ascend_pipelines.py --grep 'InsertNd('   # 无输出
```

**（c）几个容易被误判成「只有更新版本才有」的能力，其实是 v0.1.15 的既有事实。** 用命令能确认：

| 能力 | 在 v0.1.15 里有吗 | 核验 |
| --- | --- | --- |
| 昇腾进主干的 `tilelang/ascend/` + `src/ascend/` | **有** | `ls _vendor/tilelang-v0.1.15/tilelang/ascend` |
| 昇腾自动调度段（`AutoSchedule`…`RestoreWhileLoops`） | **有** | `sed -n '104,116p' tilelang/ascend/pipeline.py` |
| `RewriteDualCopy` / `InsertNd2Nz` / `AscendInsertOOBPadding` | **有** | 同上与 `tilelang/ascend/transform/__init__.py` |
| `StorageRewrite` 只跑在 CUDA 侧 | **是** | CUDA `pipeline.py:213`；Ascend pipeline 无此行 |

直接对着 main 快照做差集，可以一次性排除「昇腾自动调度其实是 main-only」这类猜测：

```bash
python3 compare_ascend_pipelines.py --compare-main
# 输出：only in v0.1.15: (none)   only in corpus main: (none)
```

也就是说，正文列出的 `AutoSchedule` / `InsertSync` / `MaterializeMultiBuffer` /
`InsertNd2Nz` 等 pass，在 v0.1.15 与 corpus 的 main 快照里**都在**——它们是 release
事实，不是未合入的前沿能力。反过来，**要判定「某个 pass 是哪个版本新加的」，必须
有一个更早 tag 的 checkout 才能做**；当前 `_vendor/` 只有 v0.1.15，所以本章不做这种
断言（已登记进 [references.md](references.md) 的「待补资料」）。

**（d）不要用「PR 号」当判据。** PR 只是线索；判据永远是「在哪个 checkout 里
grep 得到什么」。本章所有关于「有没有」的说法都附了可直接执行的命令。

### 5.3 五个具体陷阱

1. **把顺序表背成教条。** 4.1 节的表是「默认 `pass_configs` + 某类 kernel」的产物。
   至少三类因素会改变它：`pass_configs`（`pipeline_utils.py` 里的守卫函数）、
   目标硬件（`allow_warp_specialized` 要求 CUDA target 且有 TMA）、kernel 内容
   （昇腾的 13 步自动调度只在 `tl.enable_auto_schedule` 为真时跑）。
2. **在 `src/transform/` 里找 pass 顺序。** 顺序写在
   `tilelang/<backend>/pipeline.py`；`src/transform/*.cc` 里只有行为。
3. **改完 pass 没生效，先怀疑缓存。** `KernelCache` 命中时 pipeline **一步都不跑**。
   调试编译器时应先 `tilelang.disable_cache()` 或换 `TILELANG_CACHE_DIR`。
4. **`SplitHostDevice` 之后只看一个模块。** 见 4.3.7 的失败模式。
5. **昇腾与 CUDA 的「同步」不是同一个东西。** CUDA 的 `ThreadSync` 是线程屏障；
   昇腾的 `InsertSync`/`AscendThreadSync` 是**多流水之间的完成协议 + flag 分配**。
   看到昇腾 kernel 里没有 `bar.sync` 并不代表没同步。

---

## 6. 本章小结

把这一章压成一页：

```text
① 七个节点：IRModule → PrimFunc → (params/buffer_map, body)
             body 里是 Stmt；Stmt 里嵌 PrimExpr；Buffer/Var 是叶子。
             并行结构 = grid 的 thread_binding 循环 + thread_extent AttrStmt。
             主块名是硬编码的 "tilelang_root"（attr.h:17）。

② 三种方言：tir / tirx（launch_thread、match_buffer）/ s_tir（block、reads、writes）。

③ 两条 pipeline：CUDA 73 次、Ascend 80 次 pass 调用；39 个同名 pass 构成共享走廊。
   分岔点只有三个主题：流水（谁规划/改写）、布局（fragment vs fractal/NZ）、
   同步与片上内存（ThreadSync+MergeSharedMemory vs InsertSync+MergeUB）。

④ 承重 pass 的读法：输入/输出差异 → 算法 → 源码 → 失败模式。
   通用 8 个：PipelinePlanning / InjectSoftwarePipeline / LayoutInference /
             LowerTileOp / StorageRewrite / ThreadSync / SplitHostDevice / MakePackedAPI
   昇腾 3 个：AutoSchedule / InsertSync / MaterializeMultiBuffer

⑤ 自己看：先 dump 顺序表 → 定 pass_configs → lower_trace 抓 before/after →
   按 pass 读 diff → 布局可视化 → codegen 对照。

⑥ 版本纪律：名字与存在性一律用 grep 在 _vendor 里核验；
   昇腾有「进主干」与「适配器仓库」两条路线，词汇不同；InsertNd2Nz 才是真名。
```

如果只能记住一句话：**pass 的顺序是 Python 里的数据，pass 的行为是 C++ 里的函数，
而它们交互的唯一通道是 IR 上的注解与块结构。**

## 延伸阅读

* pass 顺序的来源：`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py`（73 步）
  与 `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py`（80 步）。逐行列表由
  `code/01-pass-order/` 生成。
* pass 框架：`_vendor/tilelang-v0.1.15/tilelang/backend/pass_pipeline/pipeline.py`
  与 `.../pipeline_utils.py`。
* 真实 IR 片段：`code/02-ir-snippet-extract/`（来源为
  `examples/eager_jit/eagerjit.en.ipynb`、
  `src/transform/materialize_kernel_launch.cc`、
  `docs/compiler_internals/inject_fence_proxy.md`）。
* 逐 pass 观察：`code/03-lower-trace-walkthrough/`（需 GPU，本机未运行）与
  `_vendor/tilelang-v0.1.15/docs/tools/lower_trace.md`。
* 版本差异：`code/04-version-diff/`；
  corpus 的 `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md`
  （二手导航材料，核对时以 `_vendor/` 为准）。
* 下一章：第 21 章展开 `LayoutInference` 的布局代数；第 22 章展开
  `InjectSoftwarePipeline` 与 `ThreadSync`；第 23 章展开昇腾的 `AutoSchedule`。

本章所有引用的完整清单与核验状态见 [references.md](references.md)。
