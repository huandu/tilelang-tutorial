# 第 25 章 昇腾后端：代码生成与硬件特性

## 学习目标

读完本章，你应当能：

1. 说清“昇腾后端”在锁定版本里其实是**四条可执行的编译路线**，并指出每一条的
   **入口、生成物、编译器**分别是什么；
2. 把 `T.alloc_shared` / `T.alloc_l1` / `T.alloc_l0a` / `T.alloc_l0b` / `T.alloc_l0c`
   逐条对应到 UB / L1 / L0A / L0B / L0C，并说出它们最终写进 CCE 源文件的
   `__ubuf__` / `__cbuf__` / `__ca__` / `__cb__` / `__cc__` 限定符；
3. 看着一段 `T.copy` 就能说出它会被降到哪一条 DMA 通路（`DMAPath`）、哪一个硬件
   流水线（MTE1/MTE2/MTE3/FIX）、以及 codegen 最终发出的 intrinsic 名；
4. 说明 `T.dual_copy`、`T.SimtVF` / `T.SimdVF`、`T.Pipelined` 这三类昇腾专属原语
   在代码生成阶段各自体现在哪里，并知道 `T.tile.*` **在 v0.1.15 里不存在**；
5. 写出 NV 与昇腾两条后端在生成物、工具链、指令层次、同步模型上的结构对照；
6. 在没有 NPU 的机器上，用“读源码 + 打印生成物清单 + 静态审计”三步，对一份
   昇腾生成代码做出可信的核查。

## 前置知识

* 第 4 章（编译全流程）与第 20 章（pass 管线）：你会用到 `MaterializeKernelLaunch`、
  `LowerTileOp`、`MakePackedAPI` 这些 pass 的名字。
* 第 6 章与第 19 章 §2：AIC/AIV、UB/L1/L0A/L0B/L0C、PipeBarrier、set/wait flag
  的基本概念。第 19 章给了机器模型与手把手实践；**本章只做代码生成**，不重复
  环境安装与调优流程。
* 第 22、23 章：自动调度、多缓冲与同步插入的算法。本章只关心它们在 codegen 里
  留下的痕迹。

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)，**没有 NVIDIA GPU，也没有昇腾 NPU**，没有安装
CANN 与 bisheng。因此本章：

* 所有“生成什么代码”的结论，都来自**逐行阅读锁定版本的 codegen 与 lowering
  源码**，而不是运行结果；
* 所有“谁编译、编译出什么”的结论，都来自**编译器调用点的源码**；
* **不含任何本机实测的性能数字**；引用到的厂商数字都注明出处与“厂商自述”；
* `code/05-emit-ascend-c/` 里有一个真实可编译的 kernel，但它的 `--emit` 模式
  **本机从未运行过**，只提供了“在昇腾机器上怎么核对”的清单。

本章的引用行号由 `code/04-cite-check/check_citations.py` 逐个核对过（见 §7.4）。

---

## 1. 问题的由来：一次 `target="ascend"` 之后到底发生了什么

第 19 章告诉你：在昇腾上写 TileLang，要先决定用哪棵树。但“用哪棵树”只是开始。
真正决定你能不能用、好不好调、出问题找谁的是**生成物**这一层：

* 编译器吐出来的是一份 **CCE/Ascend C 源文件**，还是一份 **MLIR 文本**？
* 这份源文件里出现的是 `asc_copy_gm2l1_nd2nz(...)` 这样的**内置函数调用**，
  还是 `AscendC::TPipe` / `pipe.InitBuffer(...)` 这样的 **C++ 类库调用**？
* 最终把它变成二进制的是 `bisheng` 还是 `bishengir-compile`？
* 用户写的 `T.copy` 到底变成哪一条硬件搬运指令，`T.gemm` 又变成哪一条矩阵指令？

这四个问题，就是本章的全部内容。答案在锁定版本里是**明确的、可逐行核对的**，
但它们分散在三个仓库、四种目标名、五套目录结构里。这一章把它们收拢到一张表上。

### 1.1 先固定“四条路线”这个词

锁定版本（见 [../../VERSIONS.md](../../VERSIONS.md)）里，与“昇腾代码生成”相关的
可执行路线一共**四条**：

| 代号 | 树 / 仓库 | 目标 | 生成物 | 编译器 |
| --- | --- | --- | --- | --- |
| **M1** | `_vendor/tilelang-v0.1.15/`（主干） | `target="ascend"`（别名 `"asc"`） | CCE/Ascend C 源，`asc_*` 内置函数风格（**不是** `AscendC::` 类库风格） | `bisheng --npu-arch=dav-3510` + `ld.lld` → 可执行 CCE ELF |
| **A1** | `_vendor/tilelang-ascend-v0.1.2.000/`（适配器） | `target="ascendc"` | `AscendC::TPipe` / `AscendC::TBuf` 类库风格的 C++ | `bisheng --npu-arch=dav-2201 -xasc` → `.so` |
| **A2** | 同上 | `target="pto"` | PTO-DSL C++（`pto::Tile` / `TLOAD` / `TMATMUL` 模板） | `bisheng --cce-aicore-arch=dav-c220\|dav-c310 -xcce` → `.so` |
| **L1** | corpus 快照 `tilelang-mlir-ascend` | TVM target kind `npuir` | AscendNPU IR（MLIR 文本） | `bishengir-compile --enable-hivm-compile=true …` → `.o` |

前两条（A1/A2）就是适配器仓库自述里说的 **“Ascend C & PTO”路线**，L1 是
**“AscendNPU IR”路线**；两者在同一句自述里被并列为“两条技术路线”
（`_vendor/tilelang-ascend-v0.1.2.000/README.md:12`）。M1 则是 2026-09-30 随
PR3308 进入主干的那条，它自带一套自动调度与同步插入
（`../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3308.md`）。

**这四条的 API 不通用、target 名不通用、编译器参数也不通用。** 后面每一节都会
明确写“这是哪一条”。

### 1.2 一个必须先纠正的直觉：`T.tile.*` 不在主干里

昇腾适配器的文档大量出现 `T.tile.add(...)`、`T.tile.sort32(...)` 这类写法。它来自
适配器仓库的 `tilelang/language/ascend_tile.py`（3,112 行），并在
`tilelang/language/__init__.py:86` 被注册成 `T.tile` 命名空间。

主干 `v0.1.15` 里**没有这个命名空间**：`tilelang/ascend/language/` 下不存在
`ascend_tile.py`，`tilelang/ascend/language/__init__.py:84-124` 的导出清单里也没有
任何 `tile` 别名。`code/03-api-presence-matrix/api_matrix.py` 会替你验证这一点
（输出见 §6.1）。同理，`T.alloc_ub`、`T.alloc_L1`（大写）、`T.gemm_v0`、
`T.dump_tensor` 都只存在于适配器那一侧。

---

## 2. 概念与模型：昇腾的机器长什么样（只讲影响代码生成的部分）

第 19 章 §2 已给出完整机器模型，这里只留下**会直接改变生成代码形状**的四件事。

### 2.1 AI Core 是“一 Cube + 两 Vector”的物理组

昇腾 950PR 的规格是 **32/28 个 Cube Core、64/56 个 Vector Core**
（`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md:57-58`），
也就是 1:2。这个比例不是抽象，它直接写在生成的函数签名前缀里：

```text
// _vendor/tilelang-v0.1.15/src/ascend/codegen/codegen_ascend.cc:336-351
extern "C" __global__ __mix__(1, 2)   // kMix：物理 1:2 组
extern "C" __global__ __vector__      // kVector：纯 Vector
extern "C" __global__ __cube__        // kCube：纯 Cube
```

`kernel_mode_` 由函数体预扫描得到（有没有 Cube、有没有 Vector）；`__mix__(1, 2)`
的注释写得很直接：“dav-3510's direct AIC-to-AIV data paths require the physical
1:2 group”。所以**“一个 kernel 里混用 Cube 与 Vector”在代码生成层就是换一个
函数限定符**，剩下的由 `AssignCore`（把任务分到 `kCoreCube` / `kCoreVector`，
`src/ascend/transform/assign_core.cc:64-75`、`core_mask.h:26-37`）与
`MixedKernel(sids=…)`（`tilelang/ascend/language/kernel.py:176-228`）负责。

### 2.2 五块片上存储，五个 scope，五个限定符

```text
GM (HBM)
 └─ L1      __cbuf__   scope "shared.l1"
     ├─ L0A  __ca__     scope "shared.l0a"    Cube 左矩阵
     ├─ L0B  __cb__     scope "shared.l0b"    Cube 右矩阵
     └─ L0C  __cc__     scope "shared.l0c"    Cube 累加器
 └─ UB      __ubuf__   scope "shared"/"shared.dyn"   Vector 的工作内存
```

映射写在 `CodeGenTileLangAscend::GetAscendScopeQualifier`
（`src/ascend/codegen/codegen_ascend.cc:152-164`）。用户侧的入口是
`tilelang/ascend/language/allocate.py`：`alloc_shared` 的默认 scope 是
`shared.dyn`（`:15-23`，注释明确说 Ascend 的 `alloc_shared` 就是 UB），
`alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 分别是
`shared.l1` / `shared.l0a` / `shared.l0b` / `shared.l0c`（`:26-70`）。

两个细节值得记：

* `alloc_l0c` 会**顺手打上 L0C 布局**（`make_ascend_l0c_layout`，`:65-69`）；
* `alloc_l0a` / `alloc_l0b` **不打布局**，注释说“Layout is deferred to the
  consuming gemm operation”（`:37-52`），布局由消费它的 GEMM 按 transpose
  标志推断。这就是为什么你没有“给 L0A 指定 K-major 还是 MN-major”的 API。

### 2.3 四条搬运/收尾流水线，七个搬运代号

搬运不是“一句 memcpy”，而是按**源/目的 scope 组合**选出的固定通路。枚举在
`src/ascend/op/oob_padding.h:32-44`，选择逻辑在
`src/ascend/op/oob_padding.cc:46-68`：

```text
GM  → UB   kGMToUB      GM  → L1   kGMToL1
UB  → GM   kUBToGM      UB  → L1   kUBToL1
L1  → L0A  kL1ToL0A     L1  → L0B  kL1ToL0B
L0C → UB   kL0CToUB     L0C → GM   kL0CToGM
L1  → L0A.sf / L0B.sf   （MX 尺度因子的槽位加载）
```

每一条通路归属于一条硬件流水线，定义在
`src/ascend/transform/ascend_pipe.h:52-63`：

| 位 | 名字 | 负责的搬运 |
| --- | --- | --- |
| `kMTE1` | MTE1 | L1 → L0A / L1 → L0B |
| `kMTE2` | MTE2 | GM → UB / GM → L1 |
| `kMTE3` | MTE3 | UB → GM / UB → L1 |
| `kCube` | M / MAD | 矩阵乘 |
| `kVector` | V | SimtVF / SimdVF / 向量运算 |
| `kFixpipe` | FIX | L0C → UB / L0C → GM |
| `kScalar` | S | 标量与配置操作 |

**这张表是整章的骨架**：后面 §4.3 说 `T.copy` 怎么落到这七条通路之一，§4.5 说
`T.dual_copy` 为什么走 FIX，§4.6 说 Vector 区域为什么用 V。

### 2.4 同步是“pipe 到 pipe 的事件”，不是线程屏障

因为执行单元是分离的（MTE1/2/3 搬运、M 计算、FIX 收尾、V 向量），依赖必须表达成
**流水线之间的一次通知**。codegen 把这种事件拆成三样东西：

1. `hard_event` 字符串，形如 `MTE2_FIX`；`GetPipePair` 只接受
   `{MTE1, MTE2, MTE3, M, V, S, FIX}` 组成的对，否则直接 `LOG(FATAL)`
   （`src/ascend/codegen/codegen_ascend.cc:262-272`）；
2. 事件号 `event_id`（8 槽命名空间，由 `RewriteFlagToBuf` 归一化，
   `tilelang/ascend/pipeline.py:176-179`）；
3. 最终文本：`asc_sync_notify(PIPE_src, PIPE_dst, event_t(id))` 与
   `asc_sync_wait(...)`（`codegen_ascend.cc:1349-1364`）。

跨核（Cube↔Vector）另有一套：`asc_sync_inter/subblock/block/intra_arrive|_wait`
（`codegen_ascend.cc:274-292,1366-1381`）。**用户不需要写它们**，
`InsertSync` 在 `LowerScheduledTIR` 之前插好（`tilelang/ascend/pipeline.py:113-115`）。

### 2.5 四条路线的一张总图

见 [figures/01-two-routes.md](figures/01-two-routes.md)。要点是：**M1 只进主干，
A1/A2 与 L1 各自在自己的仓库/分支里**；四者的“生成物”形态完全不同。

---

## 3. 用法：从两段真实 kernel 看生成物

### 3.1 M1：`target="ascend"` 的 GEMM + ReLU

这段改写自锁定版本的官方示例（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:66-108`，
另见 `examples/ascend/example_gemm.py`）：

```python
import tilelang
import tilelang.ascend.language as T          # ← M 树方言，不是 tilelang.language

@tilelang.jit(target="ascend")
def matmul_relu(A, B, block_M: int = 256, block_N: int = 224, block_K: int = 128):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.float32)
    num_blocks, n_tiles = 32, N // block_N

    with T.Kernel(num_blocks) as bx:                 # 一维核网格，没有 threads=
        A_l1  = T.alloc_l1((block_M, block_K), T.bfloat16)   # → __cbuf__
        B_l1  = T.alloc_l1((block_N, block_K), T.bfloat16)
        C_l0c = T.alloc_l0c((block_M, block_N), T.float32)   # → __cc__ + L0C 布局
        C_ub  = T.alloc_shared((block_M // 2, block_N), T.float32)  # → __ubuf__

        for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
            m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
            for k in T.Pipelined(K // block_K, num_stages=2):
                T.copy(A[m, k * block_K], A_l1)                       # GM→L1, MTE2
                T.copy(B[n, k * block_K], B_l1, l2_cache_ctrl="NOTALLOC_KEEP")
                T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))
            T.dual_copy(C_l0c, C_ub)                                   # L0C→UB, FIX
            with T.SimtVF(threads=128):                                # 线程域
                for i, j in T.Parallel(block_M // 2, block_N):
                    C_ub[i, j] = T.max(C_ub[i, j], 0)                  # ReLU 在 Vector
            T.dual_copy(C_ub, C[m:m+block_M, n:n+block_N], l2_cache_ctrl="NOTALLOC_PW")
    return C
```

这里有四个“读代码就能看见”的事实：

1. **没有 `threads=`**：`T.Kernel` 在昇腾方言里只接受一个 grid extent，
   传多维或 `threads=` 会被 Python 层直接拒绝
   （`tilelang/ascend/language/kernel.py:139-171`）。
2. **线程域只在 `T.SimtVF` 里出现**：pipeline 调
   `MaterializeKernelLaunch(lower_thread_binding=False, launch_dim_tags=["cthread"])`
   （`tilelang/ascend/pipeline.py:45-51`），注释写明 kernel scope 没有 threadIdx。
3. **两处 `T.dual_copy` 走的是两条不同机制**：L0C→UB 是硬件双目的拷贝，
   UB→GM 是按 `cthread` 的逐 AIV 软件拷贝（`copy_op.py:76-79`）。
4. **`C_ub` 的形状是 `block_M // 2`**：因为两个 AIV 各处理一半，这正是
   `dual_copy` 的形状契约（`copy_op.py:83-84`）。

### 3.2 A1 / A2：同一件事在适配器里的两种写法

适配器侧的等价 kernel 长相完全不同，`T.Kernel(..., is_npu=True)` 会额外绑定一个
子核号（官方示例里写成 `vid`；这里只写 Cube 侧，所以用 `_` 忽略）：

```python
import tilelang
import tilelang.language as T                     # ← A 树方言

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num, n_num = M // block_M, N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx, by = cid // n_num, cid % n_num
            A_L1  = T.alloc_L1((block_M, K_L1), dtype)      # 大写
            B_L1  = T.alloc_L1((K_L1, block_N), dtype)
            C_L0  = T.alloc_L0C((block_M, block_N), accum_dtype)
            with T.Scope("C"):                              # 显式声明这是 Cube 侧
                for k in T.serial(T.ceildiv(K, K_L1)):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()                         # 手写同步
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])
    return main
```

出处：`examples/gemm/example_gemm.py`、`examples/elementwise/elementwise_add.py`，
以及 `tilelang/language/allocate.py:140-156`（大写分配）、
`tilelang/language/ascend.py:418-533`（`gemm_v0`）、`:275-286`（`barrier_all`）。

同一段前端代码，`target="ascendc"` 与 `target="pto"` 会走出两个 codegen；
目标选择在 `tilelang/utils/target.py:99-100` 与
`tilelang/engine/lower.py:158-170`。

### 3.3 拿到生成代码的三条路

| 路线 | 怎么拿到 | 出处 |
| --- | --- | --- |
| M1 | `kernel.get_kernel_source()`，或在 `@tilelang.jit` 编译后打印 | `examples/ascend/README.md`；`code/05-emit-ascend-c/` |
| A1 / A2 | 同上；适配器编程指南 §5.2 给了完整片段 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md:2465-2490` |
| L1 | 环境变量 `TILELANG_DUMP_IR=1` 会打印 `====== TVM IR ======` 与 `====== npuir ======` | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-mlir-ascend/tilelang/engine/lower.py:269-294` |

本章的 `code/05-emit-ascend-c/print_ascend_c.py` 把第一条做成了可复现脚本：`--audit`
离线核对 API 名，`--emit` 在有卡的机器上打印并逐条对照 §4 的检查清单。

---

## 4. 实现机制

### 4.1 入口是三层的

“后端”不是单一函数，而是三个注册点：

1. **target 归一化**：`normalize_ascend_target` 把字符串 `"ascend"` 变成
   `Target({"kind": "ascend"})`，`normalize_asc_target` 接受缩写 `"asc"`
   （`tilelang/ascend/target.py:40-58`）。
2. **backend manifest**：`register_backend(BackendModule(name="ascend", …))` 把
   pipeline、device codegen、execution backend、host codegen 和
   `tilelang_callback_ascend_compile` 捆在一起
   （`tilelang/ascend/backend.py:52-62`）。
3. **device codegen 函数名**：`tilelang/ascend/codegen.py:5-9` 把它绑到
   FFI 函数 `target.build.tilelang_ascend` / `..._without_compile`。

所以“昇腾后端在哪里”的准确答案是：**入口在 `tilelang/ascend/`，实现在
`src/ascend/`，两者通过 `target.build.tilelang_ascend` 这一个 FFI 名连接。**

### 4.2 pass 管线：昇腾不是把 CUDA 的管线换掉几个 pass

对比两条 pipeline 的骨架（NV 见 `tilelang/cuda/pipeline.py:69-275`）：

```text
NV:   MaterializeKernelLaunch(lower_thread_binding=True)
      → PipelinePlanning → InjectSoftwarePipeline → LayoutInference
      → LowerTileOp → … → ThreadSync("shared") → MakePackedAPI

昇腾: MaterializeKernelLaunch(lower_thread_binding=False, launch_dim_tags=["cthread"])
      → AscendLayoutInference → NormalizeAscendFractalStorage → InsertNd2Nz
      → [AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore
         → InsertSync → MaterializeMultiBuffer → LowerScheduledTIR]
      → AscendSimdVFLowerParallel → AscendLowerTileOp
      → MergeUBAllocations → RewriteFlagToBuf → AscendThreadSync → MakePackedAPI
```

（昇腾侧的完整调用见 `tilelang/ascend/pipeline.py:36-190`，`rg -c 'mod = '` 数到
80 行；同一条命令在 NV 侧 `tilelang/cuda/pipeline.py` 数到 74 行，其中 1 行是
`mod = CUDAPassPipelineBodyPrologue(mod, target)` 这条 pipeline 主体调用而非
pass，**真正的 pass 调用是 73 次**，与第 4、20 章的表一致；方括号里的
九步只在 `TL_ENABLE_AUTO_SCHEDULE` 打开时执行，默认打开，
`pipeline.py:22-33,103-116`。）

三条差异会在第 22、23 章展开，这里只记结论：

* **流水与多缓冲由求解器负责**：`AutoSchedule` 读 `T.Pipelined(...,
  num_stages=k)` 的 `num_stages` 注解与 `T.annotate_buffer_versions` 的映射，
  经 Z3 求出一个版本数，再由 `PrepareMultiBuffer` / `MaterializeMultiBuffer`
  落成物理版本（`src/ascend/transform/auto_schedule.cc:705-740`、
  `tilelang/ascend/language/annotations.py:19-104`）。
* **同步由 `InsertSync` 生成、`RewriteFlagToBuf` 收口**：前者产出 `hard_event`，
  后者把它们压进 8 槽 flag 命名空间，超出的按背包问题溢出到共享
  `get_buf/rls_buf` 互斥池（`tilelang/ascend/pipeline.py:176-179`）。
* **UB 分配最后合并**：`MergeUBAllocations(align_bytes=32)`，对齐常量
  `_MERGE_UB_ALIGNMENT = 32`（`tilelang/ascend/pipeline.py:19,171-174`）。

### 4.3 `T.copy` 的降级：从 scope 到 intrinsic

这是全章最“可操作”的一张表。`tilelang/ascend/language/copy_op.py:230-372` 先做
Ascend 侧的参数归一化（`l2_cache_ctrl` 字符串→整数、`pad_value`/`data_select`、
`transpose`），然后把 `tl.tileop.copy` **重拼成** `tl.tileop.ascend_copy`
（`:362-371`）。这一步很关键，它是“方言 op 与通用 op 分家”的分界。

接着 `src/ascend/op/copy.cc` 按 `GetDMAPath` 选出的通路组装实参，
`src/ascend/codegen/codegen_ascend.cc` 把它们变成文本：

| 通路（源码枚举） | pipe | 生成代码 |
| --- | --- | --- |
| `kGMToUB` | MTE2 | `asc_copy_gm2ub_align(...)`（`codegen_ascend.cc:1383-1418`） |
| `kUBToGM` | MTE3 | `asc_copy_ub2gm_align(...)`（`:1420-1433`） |
| `kGMToL1` | MTE2 | `asc_copy_gm2l1_nd2nz` / `asc_copy_gm2l1_dn2nz`（`:1435-1467`） |
| `kL1ToL0A` / `kL1ToL0B` | MTE1 | `asc_copy_l12l0a` / `asc_copy_l12l0b`（`_transpose` 变体，`:1557-1577`） |
| `kL0CToUB` | FIX | `asc_copy_l0c2ub(...)`，前置 `asc_set_l0c_copy_nz_para(1,0,0)`；第 8 个实参是 `asc_dual_dst_mode`（`:1469-1492`） |
| `kL0CToGM` | FIX | `asc_copy_l0c2gm(...)`（`:1493-1515`） |
| `kUBToL1` | MTE3 | `asc_copy_ub2l1(...)`（`:1597-1609`） |

两个工程细节，读生成代码时会直接看到：

* **ND→NZ**：GM→L1 的拷贝默认走 `nd2nz`（顺带做格式转换）；`transpose=True`
  改走 `dn2nz`。UB→L1 的 ND→NZ 走另一条路：`InsertNd2Nz` 先把它改写成
  scatter + `asc_copy_ub2l1`（`tilelang/ascend/pipeline.py:88-91`、
  `codegen_ascend.cc:1610-1651`）。
* **OOB/尾部**：`AscendInsertOOBPadding` 在调度之前把越界的 DMA 范围夹住、并把
  GM→L1 的填充写成语义 `T.fill`，这样 `AutoSchedule` 能看到每次填充的确切
  写区间（`tilelang/ascend/pipeline.py:97-101`）。

`code/02-scope-dma-pipes/scope_dma_pipes.py` 会把上面三张表（scope、通路、pipe）
从源码里抽出来并做四条一致性检查。

### 4.3.1 两张“实参长什么样”的原文

与其记住 intrinsic 的名字，不如记住它**要几个实参、每个实参是什么**。下面两段是
锁定版本里 `EmitUbufToGmCopy_` 与 `EmitL1ToL0Copy_` 的原文（
`src/ascend/codegen/codegen_ascend.cc:1420-1433` 与 `:1557-1577`）：

```cpp
void CodeGenTileLangAscend::EmitUbufToGmCopy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 8)
      << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";
  // IR arg 2 is sid, which asc_copy_ub2gm_align does not take.
  EmitCApiCall_("asc_copy_ub2gm_align",
                {{op->args[0], "(__gm__ uint8_t*)(", ")"},
                 {op->args[1], "(__ubuf__ uint8_t*)(", ")"},
                 {op->args[3]}, {op->args[4]},
                 {op->args[5], "static_cast<asc_store_l2_cache_mode>(", ")"},
                 {op->args[6]}, {op->args[7]}});
}

void CodeGenTileLangAscend::EmitL1ToL0Copy_(const CallNode *op, bool is_l0a) {
  const bool is_transpose = GetIntImmArg(op, 8, intrinsic, "transpose") == 1;
  const char *function_name =
      is_transpose
          ? (is_l0a ? "asc_copy_l12l0a_transpose" : "asc_copy_l12l0b_transpose")
          : (is_l0a ? "asc_copy_l12l0a" : "asc_copy_l12l0b");
  EmitCApiCall_(function_name, { /* 8 个实参 */ });
}
```

三个可迁移的读法：

* 实参里出现 `"(__gm__ uint8_t*)("` / `"(__ubuf__ uint8_t*)("`，说明这个 intrinsic
  是**指针风格**的，参数顺序就是“目的、源”；
* `static_cast<asc_store_l2_cache_mode>(...)` 说明 `l2_cache_ctrl` 最终变成了
  C++ 枚举转换，而不是一个裸整数（对应的 Python 侧归一化在
  `copy_op.py:166-227`）；
* `is_transpose` 只影响**函数名的选择**，不影响实参个数。所以你在生成代码里看到
  `_transpose` 后缀，就知道上游某个 `T.copy` 带了 `transpose=True`。

### 4.4 `T.gemm` 的降级：只有一条路，就是 L0 MAD

`T.gemm` 在昇腾上的指令选择函数**不做选择**：

```cpp
// _vendor/tilelang-v0.1.15/src/ascend/op/gemm.cc:23-48
static String SelectInst(const GemmNode &op, int block_size, Target target) {
  (void)op; (void)block_size; (void)target;
  return kAscendMAD;                    // 常量 kAscendMAD = "ascend.mad"
}
```

`ComputeWarpPartition` 也固定返回 `{1, 1}`。因为昇腾没有 warp 概念。
所以 M 树的 GEMM 只有一种形态：**A 在 L0A、B 在 L0B、结果累加到 L0C 的 MAD**
（`asc_mmad`，`codegen_ascend.cc:1652-1667`）。区别只在参数：

* `clear_accum`：对应 `unit_flag_ctrl`（`copy_op.py:97-98` 与
  `gemm_op.py` 的注释说明 `unit_flag_ctrl` 0=手工同步、3=与 MAD 流水）；
* 精度模式：`T.set_hf32_mode(...)` 把 fp32 输入截成 HF32
  （`tilelang/ascend/language/mode.py:24-53`）；
* 量化写法：`gemm_blockscaled` + `alloc_l0a_sf` / `alloc_l0b_sf` 的 MX 尺度
  槽位（`allocate.py:73-128`），最终走 `asc_mmad_mx`（`codegen_ascend.cc:1652-1667`）。

**没有 `mma`/`wgmma`/`tcgen05` 的对应物，也没有 m×n×k 指令族**：昇腾侧只有一个
MAD 原语，M/N/K 是它的循环维度（源码里 `node_M/node_N/node_K` 必须是静态
`IntImm`，否则 `T.gemm` 直接抛错，见 `gemm_op.py:55-60`）。

### 4.5 `T.dual_copy`：一次拷贝，两种实现

`T.dual_copy` 的 docstring（`tilelang/ascend/language/copy_op.py:53-106`）把契约
写得很清楚：

* **它必须在混合 kernel 里用**：你还要在同一个 kernel 里提供 Cube 侧的工作；
  `dual_copy` 不会把纯 Vector kernel 变成两 AIV 启动。
* **形状推断**：rank ≥ 2 时恰有一个尾部维度是 2:1：
  `[M, N] → [M/2, N]` 是 M-split（`dual_dst_ctl=0b01`），
  `[M, N] → [M, N/2]` 是 N-split（`0b10`）；静态 extent 必须是偶数。
* **两条实现**：
  * `L0C→UB` 是**硬件双目的**拷贝。它不是软件劈成两条，而是在
    `asc_copy_l0c2ub` 的实参里传 `asc_dual_dst_mode`（`codegen_ascend.cc:1469-1492`）；
  * `GM→UB`、`UB→GM`、`UB→L1` 是**逐 AIV 的软件拷贝**，用 `cthread`
    （子核 ID）把区域切片，`RewriteDualCopy` 在 layout/OOB/调度之前完成这次改写
    （`tilelang/ascend/pipeline.py:63-68`、`src/ascend/transform/rewrite_dual_copy.cc`）。

还有一条**会静默出错**的限制：`dav-950` 不支持在 `cc→ub` 双目的拷贝里做类型转换。
`dual_copy` 在 Python 层主动拦截，只要 src 的 scope 是 `shared.l0c` 且
dtype 不同，就直接 `raise ValueError`，并建议你改用单独拷贝或在 VF 区域里 cast
（`copy_op.py:140-151`）。这是本章最值得记住的一个“编译期保护”。

### 4.6 `T.SimtVF` / `T.SimdVF`：两个区域，两套约束

`SIMT_VF` 与 `SIMD_VF` 在 IR 里是**不透明 SBlock**，一直活到 codegen
（`src/ascend/transform/vf_regions.h:19-24`）。它们的区别是：

| | `T.SimtVF(threads=n)` | `T.SimdVF()` |
| --- | --- | --- |
| 线程域 | 有：块体第一个 `AttrStmt(thread_extent)` 绑定 `threadIdx.x`（`vf_regions.h:38-60`） | 无 |
| 数据宽度 | 按线程数切 | **必须恰好 2048 bit** |
| 写法 | `T.Parallel` 映射到线程 | 显式 `T.simd.*` 指令 |
| 典型生成物 | `asc_syncthreads()`（`codegen_ascend.cc:3554-3557`）+ 线程映射后的向量代码 | `vector_f32` / `vector_bool` 寄存器操作（`PrintType`，`:451-537`） |

`SimdVF` 的硬约束写在 `ValidateSimdVectorType`（`:408-434`）：向量类型
`element_bits × lanes` 必须等于 2048，否则 `LOG(FATAL)`；predicate 必须是
`boolx256`。这解释了为什么官方示例里 SIMD 循环的步长是 `tile // 64`
（float32 时 2048/32 = 64，见 `examples/ascend/example_vecadd.py:31-38`）。

`VFChecker`（`tilelang/ascend/analysis/vf_checker.py`）在管线中途就检查区域内的
分配与访问合法性，位置在 `InsertNd2Nz` 之后
（`tilelang/ascend/pipeline.py:93-95`）。因为有的 UB→L1 拷贝要先被改写成
SIMD scatter 才合法。

### 4.7 `T.Pipelined` 与多缓冲：注解 → 求解 → 物理版本

用户只写 `T.Pipelined(iters, num_stages=k)`；真正决定生成代码里出现几个
buffer slot 的是三件事：

1. `AutoSchedule` 读循环上的 `num_stages` 注解（`auto_schedule.cc:705-740`），
   再结合 `T.annotate_buffer_versions({buf: …})` 的映射
   （`annotations.py:19-104`，取值语义见第 19 章 §7.5 的表）；
2. 求解器给出每个 buffer 的**版本数**与**索引模式**（`iteration` / `counter` /
   `auto`）；
3. `MaterializeMultiBuffer` 把逻辑版本落成物理槽位，`RewriteAscendBufferVersionLayout`
   再改写布局（`tilelang/ascend/pipeline.py:114,135`）。

如果用户**自己**做了 ping-pong（`buf[l % 2]`），要用
`T.annotate_manual_multi_buffer(buf, {flat: 3})` 告诉编译器“别再给我加一维”，
它只负责分析依赖并补 `set_flag`/`wait_flag`（`annotations.py:107-162`）。
**两种多缓冲不要叠加**。这正是第 19 章 §7.5 末尾的警告。

### 4.8 A 树的两条 codegen

适配器的两条路线在 `src/target/` 下各有一个 codegen：

| | `target="ascendc"` | `target="pto"` |
| --- | --- | --- |
| 目标归一化 | `lower()` 里是 `Target({"kind": "llvm", "model": "ascendc"})`（`tilelang/engine/lower.py:224`）；`determine_target` 另把 `"ascendc"` 映射成 `llvm --keys=ascend`（`tilelang/utils/target.py:99-100`） | 同上，`model` 为 `"pto"` |
| 选 codegen | `target.build.tilelang_ascend`（`tilelang/engine/lower.py:160-163`） | `target.build.tilelang_ascend_pto`（`:164-165`） |
| 头文件 | `tl_templates/ascend/common.h`、`acl/acl.h`、`runtime/rt_ffts.h`（`src/target/codegen_ascend.cc:105-109`） | `tl_templates/pto/common.h`、`<pto/pto-inst.hpp>`（`src/target/codegen_ascend_pto.cc:503-508`） |
| 函数前缀 | `extern "C" __global__ __aicore__`（`codegen_ascend.cc:100-101`） | `extern "C" __global__ AICORE`（`codegen_ascend_pto.cc:4249-4250`） |
| 分配代码 | `AscendC::TPipe pipe;` + `AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(...)`，随后 `auto v = ascend_l1.GetWithOffset<T>(size, offset)`（`codegen_ascend.cc:986,1028-1032,838-856`） | `pto::Tile<...>` 别名实例（`src/tl_templates/pto/common.h`） |
| 编译器 | `bisheng --npu-arch=dav-2201 -std=c++17 -xasc … --shared`（`libgen.py:150-183`） | `bisheng --cce-aicore-arch=dav-c220\|dav-c310 -std=gnu++17 -xcce -I3rdparty/pto-isa/include …`（`libgen.py:185-230`） |

一个容易忽略的差别：A 树的中端用的是**传统 TileLang 管线**
`PipelinePlanning` + `InjectSoftwarePipeline`（`tilelang/engine/phase.py:103-104`），
同步用 `AscendSyncInsert` / `AscendSyncInsertVS`（`:121-122`）；而 M 树把这两件
事换成了 `AutoSchedule` + `InsertSync`。这解释了为什么两棵树的 `pass_configs`
完全不同（`TL_ASCEND_*` 只在 A 树，`tilelang/transform/pass_config.py:38-62`）。

### 4.9 L1：AscendNPU IR（MLIR）路线

MLIR 路线的中端不是 TIR pass，而是 **MLIR 方言**：

* `codegen_npuir_apis` / `codegen_npuir_dev` 把 TIR 变成 `npuir` 文本
  （`tilelang/engine/lower.py:167-181`；`TILELANG_ASCEND_MODE` 选 expert/dev 两套）；
* 然后跑 MLIR pass：`canonicalize`、`adapt-triton-kernel`
  （`tilelang/tladapter/transforms/`，见 `tilelang/engine/lower.py:286-290`）；
* 最后由 **`bishengir-compile`** 编成 `.o`，关键开关是
  `--enable-hivm-compile=true`、`--enable-auto-multi-buffer=true`、
  `--enable-triton-kernel-compile=true`（`jit_npu.py:1862-1916`）。

AscendNPU IR 自己的分层方言是 **HFusion（Linalg 扩展）/ HIVM（Tile 级计算、搬运、
同步）/ HACC（Host-Device 与 launch 语义）/ Annotation、Scope**
（`../tilelang-corpus/07-tilelang-ascend-adapter/raw/huawei-techarticle-tilelang-npuir.md:136-146`）。
其中 HIVM 负责 CV 核映射（Mix Kernel 融合、核间同步、CVPipeline、AutoSubTiling），
以及核内片上内存映射与多级流水。

和 §4.2 的 M 树对照，两边的职责分配是镜像的：

| | M 树（TIR 内部） | L1（MLIR 内部） |
| --- | --- | --- |
| CV 融合 | `AssignCore` + `__mix__(1,2)` | HIVM 的 CV 核映射 |
| 多缓冲 | `PrepareMultiBuffer` / `MaterializeMultiBuffer` | `--enable-auto-multi-buffer` |
| 同步 | `InsertSync` → flag | HIVM 的 AutoSync / CVPipeline |
| 内存规划 | `MergeUBAllocations` | PlanMemory |

### 4.10 PTO 路线：tile 粒度的指令集

PTO 的定位是一套**以 tile 为粒度的指令集**：`pto.tload` / `pto.tstore` 做搬运，
`pto.tadd` / `pto.trowsum` 等做整 tile 计算，全部围绕 `!pto.tile_buf<...>`
这一个存储对象展开（`../tilelang-corpus/06-huawei-ascend/raw/PTO-Gym/docs/PTO-tile-Instruction-SPEC.md:18-60`）。
它的几个概念和 TileLang 几乎一一对应，值得对照读：

| PTO 概念 | 含义 | TileLang 对应 |
| --- | --- | --- |
| `loc`（UB / L0A / L0B / L0C / bias / scaling） | tile 的片上位置 | `T.alloc_shared/l1/l0a/l0b/l0c` |
| 物理形状 `(rows, cols)` vs 有效区 `(v_row, v_col)` | 分配多少 vs 真正有数据 | 对齐后的 allocation vs OOB/tail mask |
| `pto.tload` / `pto.tstore` | GM↔片上搬运 | `T.copy` 的 `kGMToUB` / `kUBToGM` |
| pipeline 归属：`PIPE_MTE2` / `PIPE_MTE3` / `PIPE_V` | 每条 tileop 挂在哪个硬件队列 | §2.3 的同一张表 |
| `pto.set_flag`/`wait_flag` 或 `pto.get_buf`/`rls_buf` | 跨 pipeline 排序 | `asc_sync_notify` / `asc_sync_wait` 与 buffer 版本 |

TileLang 生成 PTO 的方式**不是**发 MLIR，而是发 **PTO-DSL 的 C++ 模板调用**：
codegen 头里 `#include <pto/pto-inst.hpp>`，模板层在
`src/tl_templates/pto/common.h` 里把 `pto::Tile<...>` 落成 L1 / L0A / L0B / L0C /
UB 各自的 `TileMatL1` / `TileMatL0A` / `TileUbDataND` 别名
（`../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/README.md` 也确认了
“tilelang-ascend 用 PTO-ISA 作为 codegen 后端”这一点）。

**状态提示**：把 PTO 后端带进主干的是 PR3310。corpus 里有两份互相矛盾的快照：
PR 页面快照（2026-09-30，`../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3310.md`）
把它标为 **draft**，但同一 corpus 的 main 源码快照（commit `994b44e`）就是该 PR 的
合并提交。事实陈述以代码为准：**v0.1.15 里搜不到 `pto` 目标（见 §6.2），main 快照里有**
（详见第 30 章 §4.1.2）。

还有两层容易混淆的粒度，值得在名词上分开：

* **PTO micro instruction**：流水线同步、`pto.vlds` / `pto.vsts` / `pto.vadd`
  这类微操作。它是“一个向量循环”级别的东西；
* **PTO tile instruction**：`pto.tload` / `pto.tadd` / `pto.trowsum` 这类
  整 tile 的操作，内部展开成一串 micro instruction 加同步 flag
  （`../tilelang-corpus/06-huawei-ascend/raw/PTO-Gym/docs/PTO-tile-Instruction-SPEC.md:18-46`）。

PTO 生态里的 **`ptoas`** 是这套指令的汇编器：PTO-Gym 把“一个可运行用例”拆成
`kernel.pto`（设备侧 PTO 源）、`launch.cpp`、`main.cpp`、`golden.py`、`compare.py`
五个文件，用 `ptoas` + bisheng 编成可执行文件再比对 golden
（`../tilelang-corpus/06-huawei-ascend/raw/PTO-Gym/README.md:1-40`）。
TileLang 不写 `.pto` 汇编，而是发 `pto::` 模板的 C++，两者最终都会经过 PTO-ISA
这一层，但**中间形态不同**。

如果你要判断“某个 PTO tile instruction 在 spec 里到底怎么定义”，权威来源是
`pto-spec`（ASL 形式化定义，工作树自述为架构 `0.58.7`：
466 条 scalar instruction form、96 条 bundle/command、118 个 direct Tile operation、
905 个 ASL 单元，`../tilelang-corpus/06-huawei-ascend/raw/pto-spec/README.md:1-40`），
而不是某一篇博客。

---

## 5. 与 NV 后端的结构对照

第 24 章会展开 NV 侧；这里只给结构对照表，供你定位“同一个抽象在两边落在哪”。
NV 侧来源：`tilelang/cuda/backend.py:69-120`、
`tilelang/cuda/pipeline.py:69-275`；昇腾侧来源见 §4.2。

| 维度 | NV / CUDA（v0.1.15） | 昇腾 M1（v0.1.15） |
| --- | --- | --- |
| 执行单元 | SM（SIMT，warp=32 lane） | AIC（Cube）+ AIV（Vector），物理 1:2 组 |
| 线程模型 | `thread/block/warp` 贯穿始终 | 核内无线程；只有 `T.SimtVF` 内部有 `threadIdx.x` |
| 片上存储 | shared memory + 寄存器 | L1 / L0A / L0B / L0C / UB |
| 矩阵指令 | `mma` → `wgmma` → `tcgen05`（按 arch 选择） | 单一 `asc_mmad`（MAD），M/N/K 是循环维度 |
| 搬运 | `cp.async`、TMA | `asc_copy_gm2l1_*` / `_ub2gm_align` / `_l12l0*` |
| 收尾 | epilogue 在寄存器里做 | Fixpipe：`asc_copy_l0c2ub` / `_l0c2gm` |
| 流水/多缓冲 | `PipelinePlanning` + `InjectSoftwarePipeline` | `AutoSchedule`(Z3) + `MaterializeMultiBuffer` |
| 同步 | `ThreadSync("shared")`：warp barrier / mbarrier | `InsertSync` → `hard_event` → 8 槽 flag |
| 生成物 | CUDA C++ | CCE/Ascend C（`asc_*` 内置函数风格） |
| 编译器 | `nvcc` / NVRTC | `bisheng` + `ld.lld` |
| 产物 | cubin / fatbin | 链接好的 CCE ELF（`aibin`） |

一句话：**NV 是“一个线程块一组线程 + shared/register”；昇腾是“Cube 核 + 两个
Vector 子核 + 五块片上 buffer + 四条搬运/收尾流水线”**。这是两侧 codegen、
同步 pass、多缓冲模型完全不同的根因。见
[figures/05-nv-vs-ascend.md](figures/05-nv-vs-ascend.md)。

为什么这条差异会一路影响到代码生成？因为**同步的语义对象不同**：

* NV 侧同步的是**线程**：同一个 block 里的 128/256 个线程要看到同一份 shared
  memory，所以 `ThreadSync` 插的是 `__syncthreads()` / mbarrier 这类
  “等到所有线程到齐”的屏障；
* 昇腾侧同步的是**流水线**：MTE2 把数据放进 UB 之后，V 才能读；M 把结果写进 L0C
  之后，FIX 才能搬。这里没有“所有线程到齐”的概念，只有“MTE2 → V 这一对事件
  完成了吗”。所以 `InsertSync` 产出的是 `hard_event`（一对 pipe 名），
  codegen 发出的是 `asc_sync_notify/wait`。

把这两件事当成同一种东西，是理解昇腾自动同步时最常见的错误：昇腾的
`num_stages` 之所以能自动决定 buffer 版本数，正是因为每条依赖都能写成
“生产 pipe → 消费 pipe + 距离”，而这恰好是一个可求解的约束问题（第 23 章）。

---

## 6. 进阶与陷阱

### 6.1 两棵树的 API 不通用（已逐条核对）

`code/03-api-presence-matrix/api_matrix.py` 在锁定版本上给出下面的结果：

| API | M 树（v0.1.15） | A 树（v0.1.2.000） |
| --- | --- | --- |
| `target` | `"ascend"` / `"asc"` | `"ascendc"` / `"pto"` |
| 前端导入 | `tilelang.ascend.language` | `tilelang.language` |
| UB 分配 | `T.alloc_shared`（scope `shared.dyn`） | `T.alloc_ub`（scope `shared.ub`） |
| L1/L0 分配 | `T.alloc_l1/l0a/l0b/l0c`（小写） | `T.alloc_L1/L0A/L0B/L0C`（大写） |
| 双目的拷贝 | `T.dual_copy` | 无 |
| SIMT/SIMD 区域 | `T.SimtVF` / `T.SimdVF` | 无（用 `T.Scope("V")` + `T.Parallel`） |
| 矩阵乘 | `T.gemm` | `T.gemm_v0` |
| `T.tile.*` | **不存在** | `T.tile` = `ascend_tile`（`:86`） |
| 设备打印 | `T.print`（无 `dump_tensor`） | `T.printf` / `T.dump_tensor` |
| `pass_configs` 前缀 | `TL_ENABLE_AUTO_SCHEDULE` | `TL_ASCEND_*`（8 个键） |

把 A 树的 `T.alloc_L1` 写进 M 树的 kernel，会得到 `AttributeError`；把 M 树的
`T.dual_copy` 写进 A 树，同样是 `AttributeError`。**同一段昇腾代码在另一棵树
上不可复现**，这是排查昇腾问题时第一个要确认的事。

### 6.2 v0.1.15 里没有 PTO 目标

这条结论可以直接验证：

```bash
rg -n 'tilelang_pto|ascend_pto|"pto"' \
   _vendor/tilelang-v0.1.15/tilelang/ _vendor/tilelang-v0.1.15/src/   # 无输出
ls _vendor/tilelang-v0.1.15/src/ascend/                                # 无 pto codegen
```

PTO codegen（`codegen_ascend_pto.cc`、`rt_mod_ascend_pto.cc`、
`tl_templates/pto/`）只存在于适配器仓库。任何看到“在 v0.1.15 里用
`target="pto"`”的说法，都要先确认他指的是哪棵树。

### 6.3 issue 3317：两条路线怎么选，**没有维护者结论**

corpus 收了这个问题：*“It seems there are two Ascend 950 backends, which is
suggested to use”*（`../tilelang-corpus/08-frontier-unmerged/raw/github-issue-3317.md`）。
该 issue 已被关闭，但抓取到的页面里**只有提问正文、标签与元信息，没有维护者回答**
（帖子里的 Activity 区没有可见评论）。因此：

* 不要引用它作为“官方推荐用某一条”的依据；
* 它唯一能支撑的结论是“社区已经意识到存在两条昇腾 950 路线，并且这是一个
  需要被回答的问题”。

需要选路线时，唯一可靠的依据还是各自仓库的适用范围：
PR3308 明确写“target hardware is Ascend 950 (dav-3510)，A2/A3 请用社区维护的
TileLang-Ascend”；PR3310 的评审快照是 draft，但 main 快照已合入它（见 §4.10）
（PR3308 正文见
`../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3308.md`）。

### 6.4 三个具体的“会咬人”的点

1. **arch 字符串不通用**。M 树默认 `dav-3510`（`tilelang/contrib/bisheng.py:91-96`），
   A 树 AscendC 用 `--npu-arch=dav-2201`，A 树 PTO 用
   `--cce-aicore-arch=dav-c220`（A2/A3）或 `dav-c310`（A5）
   （`tilelang/jit/adapter/libgen.py:150-190`）。写死一个值换树就会失败。
2. **`dual_copy` 不做类型转换**。L0C→UB 的 src/dst dtype 必须一致，否则
   Python 层直接报错（`copy_op.py:140-151`）。
3. **`SimdVF` 的宽度必须整**。2048-bit 不是“建议”，是
   `ValidateSimdVectorType` 的 `LOG(FATAL)` 条件（`codegen_ascend.cc:408-434`）。

---

## 7. 调试与验证：没有 NPU 时怎么核查生成代码

### 7.1 官方给的三类调试手段

适配器编程指南 §5–§7 给了完整清单
（`../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md:2342-2526`）：

* **设备端打印**：`T.printf(fmt, *args)` 与 `T.dump_tensor(tensor, desc, dump_size, shape_info)`；
  需要编译时开调试（`compile_flags=["-D_DEBUG", "--cce-enable-print"]`，
  或 PTO 的 `TL_PTO_DEBUG=1`）。注意指南明确写了代价：**每个核上的 dump 总空间上限
  1 MB，且会明显影响性能**。
* **看生成的 Ascend C**：`func.get_kernel_source()`（指南 §5.2）。
* **性能剖析**：`msprof op`（上板）与 `msprof op simulator`（仿真）。仿真能看
  指令流水图/热点图/内存通路波形，但**仿真数字不等于上板数字**，也不要跨设备比较。

### 7.2 看生成代码时按这个顺序读

拿到一份 M1 的生成源，建议按下面的顺序核对（对应 `code/05-emit-ascend-c/` 的清单）：

1. **函数前缀**：`__mix__(1,2)` / `__vector__` / `__cube__`，确认它是不是混合核；
2. **buffer 声明**：`__ubuf__` / `__cbuf__` / `__ca__` / `__cb__` / `__cc__`
   各自出现几次，和你的 `T.alloc_*` 对不对得上；
3. **搬运**：`asc_copy_gm2l1_nd2nz`（含 ND→NZ）、`asc_copy_l12l0a/b`、
   `asc_copy_l0c2ub`、`asc_copy_ub2gm_align`；
4. **矩阵**：`asc_mmad` 的实参（M/N/K、transpose、unit flag）；
5. **同步**：`asc_sync_notify` / `asc_sync_wait` 的 pipe 对是否与 §2.3 一致；
   跨核时看 `asc_sync_inter_arrive/wait`；
6. **多缓冲**：同一个逻辑 buffer 是否出现多个 slot 偏移（版本数 ≥ `num_stages`）。

把上面六条并起来，一份 §3.1 那种 GEMM+ReLU kernel 的生成源**形态**大致是下面
这样。**注意：这不是一段实测输出**，而是从 `codegen_ascend.cc` 的发射语句
（`:336-351`、`:1383-1690`）与 `src/tl_templates/ascend/common.h:1-13` 反推的骨架；
真实函数名、实参个数与 buffer 偏移只有上机才能确认。

```text
// 由 extern "C" __global__ __mix__(1, 2) 决定：这是一个混合核
#include <tl_templates/ascend/common.h>      // c_api/asc_simd.h、simt_api/* ...（见 src/tl_templates/ascend/common.h:1-13）
#include <tl_templates/ascend/debug.h>

extern "C" __global__ __mix__(1, 2) void main_kernel(__gm__ bfloat16* A,
                                                    __gm__ bfloat16* B,
                                                    __gm__ float* C) {
  __cbuf__ bfloat16* A_l1;   // T.alloc_l1  -> scope shared.l1 -> __cbuf__
  __cbuf__ bfloat16* B_l1;
  __cc__   float*    C_l0c;  // T.alloc_l0c -> scope shared.l0c -> __cc__
  __ubuf__ float*    C_ub;   // T.alloc_shared -> scope shared.dyn -> __ubuf__

  for (/* persistent tiles */) {
    for (/* k, num_stages=2 */) {
      asc_copy_gm2l1_nd2nz(A_l1, A, /* ... */);              // MTE2
      asc_copy_gm2l1_nd2nz(B_l1, B, /* ..., l2 策略 ... */);  // MTE2
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(...));
      asc_sync_wait (PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(...));
      asc_copy_l12l0a(A_l0a, A_l1, /* ... */);               // MTE1
      asc_copy_l12l0b(B_l0b, B_l1, /* ... */);               // MTE1
      asc_mmad(C_l0c, A_l0a, B_l0b, /* M,N,K,transpose,... */);  // PIPE_M
    }
    asc_copy_l0c2ub(C_ub, C_l0c, /* ..., asc_dual_dst_mode, ... */);  // FIX
    // T.SimtVF(threads=128) 区域：ReLU + asc_syncthreads()
    asc_copy_ub2gm_align(C, C_ub, /* ..., l2 策略 ... */);           // MTE3
  }
}
```

这段骨架的价值不在于“它长得对”，而在于**它把要核对的位置都标出来了**：
如果你在真机上打印出来的代码里，某条搬运的名字、pipe 对或 buffer 限定符与这里
不符，就说明你对 §4 的某一步理解错了，或者你用的不是这一条路线。

### 7.3 没有卡时的“三级可信度”

任何关于昇腾运行结果的结论，只可能属于下面三级之一，写文档时必须标清楚：

| 级别 | 你能说什么 | 不能说什么 |
| --- | --- | --- |
| 仅静态核对 | 源码里确实这么写；API 名存在；生成物会长什么样 | “跑通了”“性能是 X” |
| CPU/仿真验证 | camodel 之类的仿真里逻辑正确 | 上板性能 |
| 真实硬件运行 | 有输入的实测结果 | 无条件的跨设备推广 |

本章属于**第一级**。`code/05-emit-ascend-c/` 的 `--emit` 也只是第一级（它打印生成物，
不运行 kernel），本机从未跑过；只有在真实 NPU 上跑出结果、并做数值比对，才进入第三级。

### 7.4 引用自查

```bash
python3 chapters/25-ascend-backend-codegen/code/04-cite-check/check_citations.py
```

它会把 `chapter.md` / `exercises.md` / `solutions.md` / `references.md` 里每个
`path:line` 逐个解析，拒绝“文件不存在”与“行号越界”。

---

## 8. 本章小结

* “昇腾后端”= **四条路线**：M1（主干 `target="ascend"`，CCE/Ascend C + bisheng
  dav-3510）、A1（适配器 `ascendc`，AscendC 类库 + bisheng dav-2201）、
  A2（适配器 `pto`，PTO-DSL + bisheng `-xcce`）、L1（MLIR 仓库 `npuir`，
  AscendNPU IR + `bishengir-compile`）。
* **存储映射**：`T.alloc_shared→UB/__ubuf__`、`alloc_l1→__cbuf__`、
  `alloc_l0a→__ca__`、`alloc_l0b→__cb__`、`alloc_l0c→__cc__`；
  `alloc_l0c` 自动带 L0C 布局，`alloc_l0a/l0b` 的布局由消费它的 GEMM 推断。
* **搬运映射**：`T.copy` 先按 scope 选 `DMAPath`（7 条 + MX 两条），再落到
  MTE1/MTE2/MTE3/FIX 之一，最后变成 `asc_copy_*` 的一个具体名字。
* **专属原语**：`T.dual_copy`（L0C→UB 硬件双目的；其它路径逐 AIV）、
  `T.SimtVF`/`T.SimdVF`（区域划分 / 2048-bit 寄存器）、`T.Pipelined` +
  `T.annotate_buffer_versions`（Z3 决定版本数）。**`T.tile.*` 与 `T.alloc_ub`
  只在适配器里。**
* **版本差异**：v0.1.15 只有一个昇腾目标 `ascend`，没有 PTO；main 源码快照已合入
  PTO（PR3310，commit `994b44e`）；issue 3317 没有维护者结论。
* 与 NV 的差别不是“换几个 pass”，而是**执行单元、线程模型、片上存储、
  矩阵指令、同步模型整条链都不同**。

## 延伸阅读

条目 id 与 [references.md](references.md) 一一对应。

* 四条路线的入口与差异：R1–R3、R4、A1、A3–A6、R45、R60、R61
* 生成物的逐行依据：R20–R32（M 树 codegen / op / pass）、A16–A18（A 树 codegen）
* 存储与搬运的机器模型：R28–R32、R37、R58
* PTO 指令集与 tile 模型：R51–R55、A17–A18
* AscendNPU IR / MLIR：R45–R50、R62
* 调试与验证：R43、R44
* 厂商与社区材料：R56–R58、R59、U1–U8
* 动手：`code/01-route-map/`、`code/02-scope-dma-pipes/`、
  `code/03-api-presence-matrix/`、`code/04-cite-check/`、`code/05-emit-ascend-c/`
