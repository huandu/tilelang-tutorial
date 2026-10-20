# 第 24 章 NVIDIA 后端：代码生成与硬件特性

## 学习目标

读完本章，你应当能够：

1. 说清一次 TileLang 编译在 NVIDIA 侧**具体分成了哪几种产物**（device 的 CUDA C / PTX / cubin、host 的 C wrapper、Python 侧的 adapter module），以及它们分别在哪个文件、哪个函数里被生成。
2. 给定一段 `T.gemm(A_shared, B_shared, C_local)`，判断它在 sm_75 / sm_80 / sm_90 / sm_100 上分别会被选成哪一种指令（MMA / WGMMA / TCGEN05 / FMA），并说出决定这件事的**谓词顺序**在哪个函数里。
3. 判断一段 `T.copy` 会不会被降成 TMA（`cp.async.bulk.tensor`）：知道默认路径下**只有 store 会自动走 TMA**，知道 load 想走 TMA 需要满足哪些显式条件。
4. 读懂 TMA 的三段式：host 侧描述符编码 → device 侧单线程发射 + mbarrier 完成通知 → 消费者等待；能指出描述符结构体在生成的 CUDA 里叫什么、由哪一行 wrapper 代码写出。
5. 说出 `T.ws(role)`、`T.tma_copy(..., barrier=...)`、`T.barrier_wait/arrive` 三者如何组合成一个 producer/consumer 流水线，以及编译器里的哪几个 pass 参与其中。
6. 描述 Blackwell 的 `tcgen05` / TMEM / block-scaled MMA 与 Hopper 的 WGMMA 在**数据落在哪里**这件事上的区别，以及 cluster / DSM 的用法与硬限制。
7. 说出 TileLang 的 CuTe DSL 后端与 NVIDIA 官方 CuTe DSL、cuTile 的关系与差别。
8. 用 `_vendor/tilelang-v0.1.14` 做对照，指出 v0.1.15 里与 NV 代码生成有关的具体变化，并说明**你是怎么查的**。

## 前置知识

* 第 4 章「架构总览与一次编译的全流程」：Python 前端 → TIR → pass pipeline → codegen 的总图。
* 第 20 章「IR 与编译期管线精读」：TIR 的结构、pass 的挂载方式、如何 dump IR。
* 第 22 章「流水线、多缓冲与同步插入」：`T.Pipelined`、多缓冲、`ThreadSync` 的插入规则。
* 第 6 章「芯片导论：NV GPU 与昇腾 NPU 的抽象对照」：SM / warp / shared memory 的基本概念。
* 第 14 章「算子实战 II：GEMM 全家族」：你需要先知道 `T.gemm` 在算子层面是干什么的。
* 术语前提：**PTX** 是 NVIDIA 的虚拟指令集，**cubin / fatbin** 是针对具体架构的机器码容器，**nvcc** 是离线编译器，**NVRTC** 是运行期编译器。本章会解释它们各自出现在哪。

本章不要求你有 NVIDIA GPU：所有结论都能用锁定版本的源码和 corpus 里的厂商文档核对。需要真卡的实验都标了「需要 GPU」。

---

## 1. 问题的由来

用户写的是这样一段东西（`_vendor/tilelang-v0.1.15/examples/quickstart.py` 风格的 GEMM+ReLU）：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), in_dtype)
    B_shared = T.alloc_shared((block_K, block_N), in_dtype)
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
```

这段代码里没有任何一行是 CUDA。它没有写 `threadIdx`，没有写 `__syncthreads()`，没有写 `cp.async.bulk.tensor`，也没有指定用 `mma.sync` 还是 `wgmma`。但最终执行的必须是三个具体的硬件事实的组合：

1. **数据怎么搬**——是用 SIMT 的 `ld.global`/`st.shared`，还是 `cp.async`，还是 TMA 的 `cp.async.bulk.tensor`；
2. **算力怎么用**——是用第 3/4 代 Tensor Core（`mma.sync` / `wgmma`），还是第 5 代（`tcgen05`），还是干脆退化成 FMA；
3. **谁去做**——是一个 block 里 256 个线程一起干，还是把其中一些 warp 专门分出来当 producer（只发拷贝）、另一些当 consumer（只算）。

NVIDIA 后端的职责，就是把上面这段 tile 级程序，**在这三个维度上做决策，并把决策翻译成合法的 CUDA C 与 PTX**。这一章讲的就是这三个决策在哪里发生、依据是什么、产物长什么样。

先给出一张全局图（`figures/01-codegen-path.md`）：TileLang 的 NVIDIA 路径不是「TIR → 一个后端」，而是「TIR → 若干个 pass → 一次 host/device 切分 → 两条并行的 codegen（device 走 CUDA C，host 走 C wrapper 或 TVM-FFI）→ 若干种运行期装载方式」。

```
                   Python frontend (tilelang/language)
                                │
                        TIR module (@T.prim_func)
                                │
        tilelang/engine/lower.py::lower(...)
                                │
        ┌───────────────────────┴───────────────────────┐
        │  backend-agnostic passes (tilelang/transform) │
        │  PipelinePlanning → InjectSoftwarePipeline    │
        │  → LayoutInference → LowerTileOp(builds cu    │
        │     tile ops) → ... → SplitHostDevice         │
        └───────────────┬───────────────┬───────────────┘
                        │               │
                 host_mod │             │ device_mod
                        │               │
   host_codegen (C/LLVM) │               │ device_codegen (CUDA / CuTeDSL)
        target.build.   │               │ target.build.tilelang_cuda
        tilelang_c_host │               │   → src/cuda/codegen/codegen_cuda.cc
                        │               │   → CUDA C source
                        │               │        │
                        │               │   nvcc (Contrib::nvcc.compile_cuda)
                        │               │        │  or NVRTC at runtime
                        │               │        ▼
                        │               │     cubin/fatbin (PTX payload optional)
                        ▼               ▼
              jit/adapter/wrapper.py: TLCUDASourceWrapper
                        │   wraps device source + host call() into one C source
                        ▼
        execution_backend:  cython  |  tvm_ffi  |  nvrtc  |  cutedsl
```

后面每一节都会把这张图里的某个方框打开。

---

## 2. 概念与模型

### 2.1 一次编译，两种产物

关键在于：**TileLang 的 device 代码和 host 代码是分开生成的，而且默认不在同一步生成。**

看 `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py` 里的 `_lower_with_context_impl`（`_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:174-213`）。它的核心顺序是：

```python
host_mod, device_mod, params, target, target_host = lower_to_host_device_ir(func_or_mod, context, runtime_only=...)
codegen_mod = device_codegen(device_mod, context) if enable_device_compile else device_codegen_without_compile(device_mod, context)
kernel_source = codegen_mod.inspect_source()
if enable_host_codegen:
    host_mod = host_codegen(host_mod, context)
```

注意两个默认值（`_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:144-170` 的 docstring 说得很直白）：`enable_host_codegen` 和 `enable_device_compile` **默认都是 False**。原因是——JIT adapter 会自己接管这两件事。

* device 默认走「只生成 CUDA C 源码、不调用 nvcc」的路径（`device_codegen_without_compile`），源码保存在 module 的 `source_map["cuda"]` 里（见 `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:151-176`）。
* host 默认由 adapter（cython / tvm_ffi / nvrtc）自己生成 wrapper，而不是走 TVM 的 `codegen_c_host`。

host/device 的切分点在 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:239-240`：

```python
mod = tilelang.transform.AnnotateDeviceRegions()(mod)
mod = tilelang.transform.SplitHostDevice()(mod)
```

切完之后，`lower_to_host_device_ir` 用两个 `Filter` 把模块一分为二（`_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:138-139`）：带 `calling_conv = kDeviceKernelLaunch` 的函数进 `device_mod`，其余进 `host_mod`。这个判据在 `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:26-51` 的 `has_device_kernel_launch` / `is_device_call_c_device` 里。

于是「设备核函数」这一段是纯粹的 `__global__`，而「怎么启动它、怎么把 `cudaStream_t` 传进来、怎么设置 dynamic shared memory 大小」全部落在 host 侧的 wrapper 上。

### 2.2 三层指令选择

「T.gemm 会变成什么指令」这个问题，其实要穿过三层：

| 层 | 位置 | 它决定什么 |
| --- | --- | --- |
| 语言层 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py` | 用户能写哪些参数（`transpose_A`、`policy`、`mbar`、`use_2cta`），以及 `T.gemm` / `T.wgmma_gemm` / `T.tcgen05_gemm` 这几个显式变体的语义 |
| 图选择层 | `_vendor/tilelang-v0.1.15/src/op/gemm.cc` → `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc` | 按 target 分派到 CUDA 实现，并在 `Gemm::SelectInst` 里按谓词顺序选出 `"cuda.mma"` / `"cuda.wgmma"` / `"cuda.tcgen05"` / `"cuda.fma"` 等 |
| 宏生成层 | `_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_*.py` + `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/*.py` | 把选中的指令展开成具体的 PTX 内联汇编序列、寄存器分配、shared memory swizzle 布局 |

三层里，**图选择层是最值得先读的**，因为它是一个几十行、可以完整读完的谓词链。

### 2.3 四代「搬运」

同一个 `T.copy`，可能落到四代不同的搬运机制上：

1. **SIMT 同步搬运**（`ld.global` + `st.shared`）：所有架构都有，`CopyInst::kNormal`。
2. **`cp.async`**（Ampere / sm_80+）：异步但仍是「每个线程负责一段」的老式异步拷贝，对应 `CopyInst::kCPAsync`。
3. **TMA**（Hopper / sm_90+）：`cp.async.bulk.tensor`，由一个线程发射、硬件按描述符做地址生成，对应 `CopyInst::kBulkLoad` / `kBulkStore` / `...1D`。
4. **DSM 集群内搬运**（Hopper+）：`cp.async.bulk.shared::cluster` 或 `map_shared_rank`，把数据直接推进**另一个 CTA 的 shared memory**。

「TMA 到底什么时候被选中」在 4.5 节；这里先记住结论：**默认的 `T.copy` 只有 global←shared 的 store 会自动走 TMA，global→shared 的 load 不会。**

### 2.4 角色划分（warp specialization）

Hopper 之后，「谁来发拷贝」这件事本身成了一个可编程的维度。CUDA 编程指南里对这件事的解释来自 Hopper 的架构文档（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md:295`）：

> The TMA operation is asynchronous and leverages the shared memory-based asynchronous barriers introduced in A100. Also, the TMA programming model is single-threaded, where a single thread in a warp is elected to issue an asynchronous TMA operation ... To further improve performance, the H100 SM adds hardware to accelerate these asynchronous-barrier wait operations.

FlashAttention-3 的论文把这件事讲成了方法论（`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608-fulltext.md:173`）：

> Hardware support for asynchrony allows for warp-specialized kernels, where the warps of a CTA are divided into producer or consumer roles that only ever issue either data movement or computation.

TileLang 把这件事暴露成了 `T.ws(role)`（2.4 节的「语言层」），并提供了一条「用户自己写角色划分」和一条「编译期自动划分」的路。细节在 4.6 节。

---

## 3. 用法

本节只讲「你写什么、会看到什么」。所有需要 GPU 的示例都在 `code/` 里给了静态核对版本。

### 3.1 打印生成的 CUDA

```python
kernel = matmul(a, b, block_M, block_N, block_K, ...)   # @tilelang.jit 产物
print(kernel.get_kernel_source())
```

`get_kernel_source()` 拿到的是 device 侧的 CUDA C。你会看到几件固定特征：

* 函数签名以 `extern "C" __global__` 开头（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:563-565`）；
* 如果线程数能推算出来，会带 `__launch_bounds__(N, min_blocks_per_sm)`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:629-646`）；
* 如果用了 TMA，会有一个 `__grid_constant__ const CUtensorMap <name>` 参数（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6562-6580` 附近）；
* 内层是 `tl::` 命名空间下的模板/内联函数调用，例如 `tl::tma_load(...)`、`tl::mbarrier_arrive_expect_tx(...)`、`tl::gemm_ss<...>(...)`。

想看到 PTX 和 cubin 需要真卡/真工具链：`_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py:271-296` 提供了 `get_ptx_from_source`，`_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py:309-366` 提供了 `get_sass_from_source`。

### 3.2 让 `T.copy` 变成（或明确不要）TMA

CUDA 方言给 `T.copy` 加了三个提示关键字（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/copy_op.py:19-70`）：

```python
T.copy(A_g, A_s)                                 # 默认：load 一般不会走 TMA
T.copy(A_g, A_s, prefer_instruction="tma")       # 明确要求 load 走 TMA（约束不满足则编译报错）
T.copy(C_s, C_g, prefer_instruction="tma")       # store 本来就默认会尝试 TMA
T.copy(A_g, A_s, disable_tma=True)               # 明确禁止这一条 copy 走 TMA
T.copy(A_g, A_s, eviction_policy="evict_last")   # L2 驱逐提示
```

需要显式异步语义、自己管同步时：

```python
T.async_copy(A_g, A_s)            # 只发 cp.async + commit_group，不插入 wait
T.ptx_wait_group(0)               # 你自己负责在消费前等

mbar = T.alloc_barrier(arrive_count=1)
T.tma_copy(A_g, A_s, barrier=mbar)   # 只发 expect_tx + tma_load，不插入 wait
T.mbarrier_wait_parity(mbar, k % 2)  # 你自己等
```

`T.tma_copy` 与 `T.copy` 的差别在 docstring 里写得很清楚（`_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:232-300`）：`T.copy` 发的是**完整同步序列**（arrive + load + wait），`T.tma_copy` 只发 producer 那半边，等待由你写。

### 3.3 手写一个 warp-specialized GEMM

官方示例 `_vendor/tilelang-v0.1.15/examples/warp_specialize/example_warp_specialize_gemm_copy_0_gemm_1.py` 是最小版本：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_K, block_N), dtype)
    C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
    data_is_ready   = T.alloc_barrier(arrive_count=128)
    compute_is_done = T.alloc_barrier(arrive_count=128)

    with T.ws(1):
        T.clear(C_local)

    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
        with T.ws(0):                                  # producer 角色
            T.barrier_wait(compute_is_done, (ko + 1) % 2)
            T.tma_copy(A[by * block_M, ko * block_K], A_shared, barrier=data_is_ready)
            T.tma_copy(B[ko * block_K, bx * block_N], B_shared, barrier=data_is_ready)
            T.barrier_arrive(data_is_ready)
        with T.ws(1):                                  # consumer 角色
            T.barrier_wait(data_is_ready, ko % 2)
            T.gemm(A_shared, B_shared, C_local)
            T.barrier_arrive(compute_is_done)

    with T.ws(1):
        T.copy(C_local, C[by * block_M, bx * block_N])
```

三个要点：

1. `T.ws(i)` 的 `i` 是 **warp group 序号**（一个 warp group = 128 线程），不是线程号。`T.ws(0)` 展开成 `tx < 128`，`T.ws(1)` 展开成 `128 <= tx < 256`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/warpgroup.py:19-59`，注释里直接写了这个映射）。
2. 两个角色之间用 **两个 mbarrier 交替握手**：`data_is_ready` 由 producer 发、consumer 等；`compute_is_done` 由 consumer 发、producer 等。
3. `T.ws` 的分支里用的是 `T.tma_copy`（而不是 `T.copy`），因为同步由 mbarrier 显式承担。

注意上面这个例子用 `num_stages=0`：多缓冲不再由 `T.Pipelined` 插入，而是由 mbarrier 的双缓冲语义自己承担（`(ko+1) % 2` 与 `ko % 2` 的奇偶翻转）。同一目录下的 `example_warp_specialize_gemm_softpipe_stage2.py` 则保留 `num_stages=2`，让软件流水与角色划分叠加。

### 3.4 Blackwell：TMEM 与 block-scaled GEMM

Blackwell（sm_100）上，第 5 代 Tensor Core 的累加器不再放在寄存器里，而放在 **Tensor Memory（TMEM）**。用法（`_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/gemm_tcgen5mma.py`）：

```python
C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
mbar   = T.alloc_barrier(1)                      # TCGEN5MMA 完成通知，arrive 数 = 1
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[bx * block_N, k * block_K], B_shared)
    T.tcgen05_gemm(A_shared, B_shared, C_tmem, trans_A, trans_B, mbar=mbar, clear_accum=k == 0)
    T.mbarrier_wait_parity(mbar, k % 2)
T.copy(C_tmem, C_local)                          # TMEM → fragment
```

如果你不想要手写的 `mbarrier_wait_parity`，用 `T.gemm(..., mbar=mbar)`，编译器会在发射之后**隐式插入**一次 `mbarrier_wait_parity`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py` 的 `gemm` docstring，第 41-76 行）。

Block-scaled MMA（MXFP8/MXFP4 这类「每 32 个元素共享一个指数」的低精度格式）在 CUDA 方言里是：

```python
T.gemm_blockscaled(A, B, C, SFA, SFB,
                   k_start=..., sf_a_granularity_k=128, sf_b_granularity_k=128,
                   mbar=mbar)          # SM100：C 在 TMEM
T.gemm_blockscaled(A, B, C, SFA, SFB, ..., sf_layout="rowmajor")
                                       # SM120：C 在 fragment
```

对应的官方示例是 `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py`（SM100，C 在 TMEM）与 `_vendor/tilelang-v0.1.15/examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`（SM120，C 在 fragment）。

### 3.5 Cluster 与 DSM

```python
with T.ClusterKernel(grid_x, grid_y, threads=128, cluster_dims=(4, 1, 1)) as (bx, by):
    rank = T.block_rank_in_cluster()          # 0..3
    T.cluster_sync()
    T.copy_cluster(A[by * BM, bx * BN], A_s, cluster_mask=0b0011)   # TMA multicast
```

`T.Kernel(..., cluster_dims=(4,1,1))` 与 `T.ClusterKernel(...)` 等价，后者只是把 `cluster_dims` 变成必填（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py` 的 `Kernel` docstring，第 26-40 行）。

`T.copy_cluster` 有两个用途（官方文档 `_vendor/tilelang-v0.1.15/docs/programming_guides/cluster_tma.md`）：

* `cluster_mask=<bitmask>`：TMA **multicast**，一个 CTA 把同一块数据广播给 mask 里的所有 CTA，省 DRAM 带宽；
* `dst_block=<rank>[, remote_barrier=...]`：**SM-to-SM**，把本 CTA 的 shared 直接推进邻居 CTA 的 shared。

### 3.6 CuTe DSL 后端

把 target 指定成带 `cutedsl` key 的 CUDA target，或把 `execution_backend` 指定成 `"cutedsl"`：

```python
kernel = tilelang.compile(func, target={"kind": "cuda", "arch": "sm_90"}, execution_backend="cutedsl")
```

它和普通 CUDA 后端的差别只在**产物形态**：device 端不再生成 CUDA C，而是生成 CuTe DSL 的 Python；`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/checks.py` 会先检查 `nvidia-cutlass-dsl>=4.7.0` 与 `cutlass.cute` 是否可导入。

---

## 4. 实现机制

### 4.1 代码生成路径：入口与结构

#### 4.1.1 Python 侧的注册表

从 v0.1.15 起，后端是**显式注册**的。`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:123-143` 用 `register_backend` 声明了 CUDA 后端的全部组件：

```python
BACKEND = register_backend(
    BackendModule(
        name="cuda",
        target_kinds=("cuda",),
        supports_target=codegen.is_plain_cuda_target,
        pipelines={"cuda": pipeline.CUDA_PIPELINE},
        device_codegens={"cuda": DeviceCodegen("cuda",
                          build=codegen.build_cuda,
                          build_without_compile=codegen.build_cuda_without_compile)},
        execution_backends=execution_backend.CUDA_EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_cuda_validate": ...,
                   "tilelang_callback_cuda_compile": ...},
    )
)
```

读这张表你就知道 NV 路径一共有哪些可选择的东西：

* `pipelines`：`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:284` 的 `CUDA_PIPELINE = PassPipeline("cuda", CUDAPassPipelineBody)`。
* `device_codegens`：`_vendor/tilelang-v0.1.15/tilelang/cuda/codegen.py:16-17` 把 TVM 的全局函数 `target.build.tilelang_cuda` / `..._without_compile` 包成 Python 回调（包装逻辑在 `_vendor/tilelang-v0.1.15/tilelang/backend/device_codegen.py:17-31`）。
* `execution_backends`：`_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:24-31`，按顺序是 `tvm_ffi`（默认，开 host codegen + device 编译）、`nvrtc`（需 `cuda-python`）、`cython`。
* `host_codegens`：`_vendor/tilelang-v0.1.15/tilelang/backend/host_codegen.py:55-59`，就是 `target.build.tilelang_c_host`（C）和 `target.build.llvm`。
* `callbacks`：`tilelang_callback_cuda_validate`（`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:30-57`，校验 `T.CUDASourceCodeKernel` 的符号名）和 `tilelang_callback_cuda_compile`（`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:59-121`，真正调 nvcc）。

#### 4.1.2 设备端：CUDA C 是怎么印出来的

真正的打印器是 C++ 的 `CodeGenTileLangCUDA`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc`，6631 行）。它的注册入口在 `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:182-184`：

```cpp
refl::GlobalDef()
    .def("target.build.tilelang_cuda", BuildTileLangCUDA)
    .def("target.build.tilelang_cuda_without_compile", BuildTileLangCUDAWithoutCompile);
```

`BuildTileLangCUDA`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:99-139`）做四件事：

1. `ValidateUniqueDeviceGlobalSymbols(mod)`：保证同一模块里没有重名的 `__global__`。
2. 调 Python 注册的 `tilelang_callback_cuda_validate`。
3. 对每个 `PrimFunc` 调 `cg.AddFunction(gvar, f)`；`AddFunction` 是打印器的主体（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6509-6628`）。
4. 调 `tilelang_callback_cuda_compile` 把源码编成 cubin/fatbin，然后用 `target::CUDAModuleCreateWithFallback` 包成 TVM CUDA module。

`AddFunction` 里几个值得记住的细节：

* `PrintFuncPrefix` 固定打印 `extern "C" __global__ `（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:563-565`）——所以生成核函数的名字就是 `global_symbol`，没有 C++ mangling。
* `PrintExtraAttrs` 从 `thread_extent` 推 `__launch_bounds__(threads, min_blocks)`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:629-646`）；`min_blocks` 来自 `T.annotate_min_blocks_per_sm(n)`。
* 参数里凡是 `storage_scope == "grid_constant"` 的，会打印成 `__grid_constant__ const <type>` ——这是 TMA 描述符参数（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6562-6580`）。
* PDL（Programmatic Dependent Launch）模式下会**抑制** `__restrict__`，因为 nvcc 在带 PDL 同步时对 kernel 参数上的 `__restrict__` 有问题（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6542-6548`）。
* 向量类型：`double4` 之外，`uint64_t/int64_t` 的 64-lane 向量在 CUDA 里没有原生类型，会退化成 `longlong4`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:986-994`）。

#### 4.1.3 编译成 cubin：nvcc 还是 NVRTC

两条路都通：

* **离线**：`tilelang_callback_cuda_compile`（`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:59-121`）拼 `-gencode arch=compute_XX,code=sm_XX`，加 `-std=c++20 -I<TILELANG_TEMPLATE_PATH> -I<CUTLASS_INCLUDE_DIR>`，然后调用 `_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py` 里的 `compile_cuda`（第 95 行起）。如果 target 的 `code` 列表多于一项，产物是 `fatbin`，否则是 `cubin`（`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:63-69`）。`--use_fast_math`、`--ptxas-options=--register-usage-level=N` 由 `pass_configs` 控制。
* **运行期**：`execution_backend="nvrtc"` 走 `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/`，用 `cuda-python` 的 NVRTC 绑定，在没有完整 CUDA toolkit 的机器上也能编（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/libgen.py:1-20` 的模块 docstring 说明了动机）。

生成 SASS/PTX 的辅助函数在 `_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py:271-366`。这些函数在本章的写作环境里**没有运行**（无 CUDA 工具链），只做了源码核对。

### 4.2 host/device 切分与 wrapper 生成

#### 4.2.1 为什么必须切

`MakePackedAPI`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:273`）把 host 侧的 `PrimFunc` 变成带 `tvm_access_ptr` 参数、带 launch 参数的 packed 函数；`LowerDeviceKernelLaunch`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:275`）把「启动」这一步变成对设备核函数的调用记录。之后 `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:138-139` 才用 `Filter` 把它们分开。

切分之后，device 模块里只剩下 `calling_conv = kDeviceKernelLaunch` 的函数。这就是为什么 `TLCUDASourceWrapper.parse_source_information` 可以断言「device module 至少一个函数、host module 恰好一个函数」——`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:466-470`。

#### 4.2.2 wrapper 干的事

`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py` 里的 `TLCUDASourceWrapper`（第 222 行起）负责把「一块 `__global__` 源码」包装成一个可以被 Python 调用的 `call()`。它的 `update_lib_code`（第 585-632 行）把三段拼起来：

```python
lib_code = self.source + self.get_cuda_host_adapter_include() + init_func + host_func
```

* `self.source` 是 device 侧 CUDA C；
* `init_func` 来自 `PREDEF_INIT_FUNC`（第 68-79 行），生成 `extern "C" TL_EXPORT int init()`，里面按需生成 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, N)`（模板见 `PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY`，第 26-33 行）；
* `host_func` 来自 `PREDEF_HOST_FUNC`（第 81-85 行），生成 `extern "C" TL_EXPORT int call(...)`。

`call()` 的正文由 `create_dispatch_func`（第 289-393 行）生成。启动部分有两种模板：

* 普通启动 → `KERNEL_LAUNCH_FUNC_CODE`（第 161-176 行）：

```c
cudaLaunchConfig_t config;
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
config.gridDim = ...; config.blockDim = ...; config.dynamicSmemBytes = ...;
cudaLaunchKernelEx(&config, kernel, args...);
```

* 带 cluster → `KERNEL_CLUSTER_LAUNCH_FUNC_CODE`（第 178-200 行）：多一条 `cudaLaunchAttributeClusterDimension`，并且**先**调 `cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`；这个 attribute 名与 `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md:73` 里厂商文档说的一致。

选择哪一个由 `get_kernel_launch_code`（第 567-571 行）根据 `self.cluster_dims[function_name]` 是否为 `None` 决定；`cluster_dims` 又是从 device 函数的 `cluster_dims` 属性读出来的（第 494-497 行）。

grid/block 的数值来自 device 函数的 `thread_extent` 属性（第 487-493 行）——这正是 `MaterializeKernelLaunch` 在 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:70` 写进去的那份信息。注意 v0.1.15 这里传了 `lower_thread_binding=True, default_threads=128`，v0.1.14 是裸调用（见 4.10 节）。

#### 4.2.3 三种 execution backend

`_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:24-31` 定义了三种：

| backend | 它拿什么 | 谁来调 `call()` |
| --- | --- | --- |
| `tvm_ffi`（默认） | TVM-FFI 的 host module（`enable_host_codegen=True`） | TVM 的 packed 调用协议 |
| `cython` | `TLCUDASourceWrapper` 生成的 C 源码 → `LibraryGenerator.compile_lib` 编成 `.so`（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/libgen.py:41-80`） | `CythonKernelWrapper`（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/adapter.py:31-35`） |
| `nvrtc` | 运行期用 NVRTC 编 | `NVRTCLibraryGenerator` 生成的 Python launcher（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/nvrtc/libgen.py:37-50`） |

三种 backend 共用同一个 `TLCUDASourceWrapper`，所以**生成的 kernel 源码是一样的**，区别只在装载与调用方式。这一点很重要：你读 `get_kernel_source()` 得到的结论对三种 backend 都成立。

### 4.3 `T.gemm` 的指令选择

#### 4.3.1 分派：先按 target 找实现

`T.gemm` 在前端是一个 `tl.tileop.gemm` 调用（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:32-76` 的 `_gemm_impl` 路径）。`LowerTileOp` 阶段，`_vendor/tilelang-v0.1.15/src/op/gemm.cc:33-48` 的 `ResolveGemmImpl` 遍历注册表找匹配目标的那一份实现；CUDA 由 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:407-422` 注册：

```cpp
RegisterGemmImpl(GemmImpl{"cuda.Gemm", MatchCudaGemmTarget,
                          cuda::Gemm::SelectInst,
                          cuda::Gemm::ComputeWarpPartition,
                          cuda::Gemm::ReuseExistingSharedLayout});
```

`GemmImpl` 把「选指令」「算 warp 划分」「是否复用已有 shared layout」三件事拆成三个函数指针，这是这段代码好读的原因。

#### 4.3.2 谓词链

`Gemm::SelectInst`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375`）的完整顺序是：

```cpp
if (op.isWgmma_)   { if (!AllowWgmma(...)) FatalWgmmaUnavailable(...); return kCudaWGMMA; }
if (op.isTcgen05_) { if (!AllowTcgen5Mma(...)) FatalTcgen5Unavailable(...); return kCudaTCGEN05; }
if (Use2CtaRequested(op)) { ... return kCudaTCGEN05; }   // 2CTA 没有单 CTA 回退
if (AllowTcgen5Mma(op, target))  return kCudaTCGEN05;
if (AllowWgmma(op, block_size, target)) return kCudaWGMMA;
if (TargetIsVolta(target)  && !AllowVoltaMma(op))  return kCudaFMA;
if (TargetIsTuring(target) && !AllowTuringMma(op)) return kCudaFMA;
return kCudaMMA;
```

翻译成人话：

1. 用户显式点名（`T.wgmma_gemm` / `T.tcgen05_gemm`）的，就必须走对应指令；约束不满足就**编译期报错**，不静默降级（`FatalWgmmaUnavailable` / `FatalTcgen5Unavailable`，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:157-181`）。
2. 标了 `use_2cta` 的，必须是 TCGEN05，且同样没有回退——注释写明了原因（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:347-353`）：「公开的 2CTA 形状约定每个 CTA 只提供 B 的一半 N 范围，回退到单 CTA 指令会变成静默越界，而不是合法降级」。
3. 否则先试 TCGEN05（`AllowTcgen5Mma`），再试 WGMMA（`AllowWgmma`）。
4. sm_70 / sm_75 上，如果连 Volta/Turing 的 MMA 形状都不满足，就退到 `cuda.fma`（纯 FMA）。
5. 其余（Ampere）走 `cuda.mma`。

#### 4.3.3 每个谓词看什么

`AllowTcgen5Mma`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:80-87`）：

```cpp
bool scope_ok = (IsSharedBuffer(op.a_) || IsTmemBuffer(op.a_)) &&
                IsSharedBuffer(op.b_) && IsTmemBuffer(op.c_);
if (!TargetIsSm100(target) || !scope_ok) return false;
DataType ab_dtype = IsTmemBuffer(op.a_) ? op.b_->dtype : op.a_->dtype;
return GetTCGEN5MMAMeta(op.m_, op.n_, op.k_, ab_dtype, op.c_->dtype).first;
```

三件事同时成立才会选 TCGEN05：目标是 sm_100；A 在 shared 或 TMEM、B 在 shared、**C 在 TMEM**；并且 `(M,N,K,dtype)` 组合在 `GetTCGEN5MMAMeta` 的白名单里。所以「我的 `T.gemm` 为什么没用 tcgen05」的第一诊断点就是 **`C` 是不是 `T.alloc_tmem` 出来的**。

`GetTCGEN5MMAMeta`（`_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h:22-177`）里，f16/bf16 → f32 的合法形状是 `atom_k=16`，`atom_n` 按 8 或 16 的步长下降，`atom_m ∈ {32,64,128,256}`；`enable_ws` 只在 `M % 64 == 0` 或 `M % 32 == 0`（且没禁用 ws）时才置位，`enable_2cta` 只在 `M % 128 == 0` 或 `M % 64 == 0` 且没禁用 2cta 时才置位——这与 PTX ISA 文档里的 shape 表完全对应（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:29795-29810`：`.ws` 形态 `32xNxK/64xNxK/128xNxK`，`N = {64,128,256}`，`K=16`）。

`AllowWgmma`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:89-97`）：

```cpp
return !pass_ctx->GetConfig(kDisableWGMMA, false) && TargetIsHopper(target) &&
       op.m_ >= 64 && num_warps % 4 == 0 && CheckWgmma(op);
```

三个条件：Hopper（sm_90x）、`M >= 64`、warp 数能被 4 整除（因为 WGMMA 是 warpgroup 级、一次 128 线程）。`CheckWgmma`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:40-78`）再看 B 是否在 shared、以及 dtype 与 K 的整除关系，例如：

* f16×f16：`K % 16 == 0`，不限制转置；
* fp8×fp8：要求 `!transA_ && transB_`（即 A 是 K-major、B 是 N-major）且 `K % 32 == 0`；
* tf32：要求 `!transA_ && transB_` 且 `K % 8 == 0`；
* int8：要求 `(!transA_) && transB_` 且 `K % 32 == 0`。

**这就是「transpose 如何影响指令」的第一处答案**：fp16/bf16 对转置不敏感，但低精度格式（fp8/tf32/int8）在 WGMMA 上只支持特定的 MN-major/K-major 组合——这与 FlashAttention-3 论文 §2.2 的表述一致（`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608-fulltext.md:186-190`：FP16 WGMMA 两种主序都接受，FP8 WGMMA 只支持 k-major）。

#### 4.3.4 warp 划分

选完指令之后，`ComputeWarpPartition`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:377-397`）决定 `m_warp × n_warp`：

* TCGEN05：`m_warp = 1, n_warp = num_warps`——因为 TCGEN05 是单线程发射的，不存在按 warp 切 M 的概念；
* WGMMA：走 `ComputeWgmmaWarpPartition`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:218-300`），硬约束 `m_warp % 4 == 0`、每 warp 至少 16 行 × 8 列；
* FMA：`1 × num_warps`；
* 其他 MMA：`ComputeDefaultWarpPartition`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:180-216`），每 warp 至少 16 行（Volta 是 16 列，其余是 8 列）。

`policy` 参数（`GemmWarpPolicy.FullRow / FullCol / Square`）就作用在这里：`FullRow` 尽量把 num_warps 全分给 M 方向，`FullCol` 全给 N 方向，`Square` 取「每 warp 的 M/N 比最接近 M/N」的那一档。找不到合法划分时 `LOG(FATAL)` 并给出具体形状与建议（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:195-212`）。

#### 4.3.5 python 侧的宏生成

选中的指令字符串决定用哪个 python 实现——`_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/__init__.py:54-63` 的注册表：

| 指令 | python 实现 | 关键行为 |
| --- | --- | --- |
| `cuda.mma` | `gemm_mma.py` → `GemmMMA` | 需要 `ldmatrix` 把 operand 搬进 fragment |
| `cuda.wgmma` | `gemm_wgmma.py` → `GemmWGMMA` | 直接从 shared 读 operand，只需要 swizzle 布局 |
| `cuda.tcgen05` | `gemm_tcgen05.py` → `GemmTCGEN5` | 累加器在 TMEM，SS/TS 两种变体 |
| `cuda.tcgen05.blockscaled` | `gemm_tcgen05_blockscaled.py` | block-scaled（v0.1.15 新增，见 4.10 节） |
| `cuda.mma.blockscaled` | `gemm_mma_sm120.py` | SM120 的 warp 级 block-scaled |
| `cuda.fma` | `gemm_fma.py` | 无 Tensor Core 的标量路径 |

`GemmWGMMA.infer_shared_layout`（`_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_wgmma.py:22-40`）是「layout 由指令决定」的典型例子：它按连续维长度选 128B/64B/32B swizzle 或线性布局——

```python
vectorized_size = 128 // self.a_dtype.bits
if continuity % (vectorized_size * 8) == 0: return make_full_bank_swizzled_layout
elif continuity % (vectorized_size * 4) == 0: return make_half_bank_swizzled_layout
elif continuity % (vectorized_size * 2) == 0: return make_quarter_bank_swizzled_layout
else: return make_linear_layout
```

**这是「transpose/layout 如何影响指令」的第二处答案**：因为 WGMMA 直接以 shared memory 为源，它的 swizzle 布局必须匹配 tensor core 的读取模式；而 MMA 路径可以复用已有的 shared layout（`ReuseExistingSharedLayout` 只对 `cuda.mma` / `cuda.mma.blockscaled` 返回 true，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:398-400`），因为它先经过寄存器。

更底层的展开在 `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/`：`wgmma_macro_generator.py` 生成 WGMMA 的 descriptor 与 atom 偏移，`tcgen05_macro_generator.py` 生成 TCGEN05 的 descriptor（含 `leading_byte_offset` / `stride_byte_offset` / `swizzle_atom_elems` 等参数，见该文件第 35-60 行的 `TCGEN05DescriptorParams`），`mma_macro_generator.py` 生成 `ldmatrix` + `mma.sync` 序列。这些文件是「宏生成层」，量很大但结构单一，读的时候抓住「descriptor 参数 → 偏移公式」这条线即可。

#### 4.3.6 一个源码层面的不一致：显式 GEMM 变体没有导出

读源码时你会注意到一件事：`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:20-30` 的 `__all__` 里列了四个显式变体——

```python
__all__ = ["gemm", "gemm_blockscaled", "gemm_sp", "make_blockscaled_gemm_layout",
           "mma_gemm_blockscaled", "tcgen05_gemm", "tcgen05_gemm_blockscaled", "wgmma_gemm"]
```

但 `_vendor/tilelang-v0.1.15/tilelang/cuda/language/__init__.py:88` 只把 `gemm, gemm_blockscaled, gemm_sp` 引进包命名空间：

```python
from .gemm_op import gemm, gemm_blockscaled, gemm_sp
```

而 `_vendor/tilelang-v0.1.15/tilelang/language/__init__.py:11` 用的是 `from tilelang.cuda.language import *`，star-import 受 `__all__` 限制，`_CUDA_API_ALL`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/__init__.py:108-196`）里也没有这几个名字。**结论：在这份锁定版本里，`T.wgmma_gemm` / `T.tcgen05_gemm` / `T.tcgen05_gemm_blockscaled` / `T.mma_gemm_blockscaled` 不在这套 facade 的导出名单上。**

与此同时，同一版本里的官方示例仍在直接调用 `T.tcgen05_gemm(...)`（例如 `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/gemm_tcgen5mma.py:48`），`gemm` 的 docstring 也在提示用 `T.tcgen05_gemm`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:39-56`）。两边对不上。

我们没有 GPU，**无法运行验证到底哪一边是真的**（写本章的环境没有安装 TileLang，也没有 CUDA）。所以这里只陈述三条可复核的源码事实（`__all__` 内容、`__init__` 的显式 import、facade 的 star-import），并给出一个保守建议：**要用显式变体，先在目标环境里确认 `T.<name>` 真的存在**（一行 `python3 -c "import tilelang.language as T; print(hasattr(T, 'tcgen05_gemm'))"` 就能验）；如果不存在，本节 `code/05-tcgen05-gemm/` 里用的是导出的 `T.gemm(..., mbar=mbar)` 同步形式，语义等价（编译器会隐式插入 `mbarrier_wait_parity`）。

### 4.4 TMA：什么时候 `T.copy` 会变成 `cp.async.bulk.tensor`

#### 4.4.1 决策函数

唯一入口是 `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-807` 的 `SelectCopyInstForLowering`，结果由 `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:791-800` 的 `Copy::Lower` 消费。把谓词按顺序展开：

1. `is_gather4` / `is_scatter4` 标注 → 强制走 gather4/scatter4 专用路径。
2. `cluster_mask != 0` → 必须是 `kBulkLoad`（multicast），否则报错并说明原因。
3. `is_tma_copy`（来自 `T.tma_copy`）→ `SelectTmaInst(allow_load=True, allow_store=True, check_last_dim=True)`；不满足就报错（**不允许回退**）。
4. `is_async_copy` / `force_cp_async` / `no_implicit_commit_wait` → 强制 `cp.async`，不满足就报错。
5. `prefer_instruction="tma"` → 尝试 TMA（load 或 store）。
6. `prefer_instruction="cp_async"` / `"sync"` → 对应路径。
7. **没有任何提示时**（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:797-803`）：

```cpp
if (!facts.disable_tma && !facts.pass_context_disables_tma) {
    CopyInst inst = SelectTmaInst(facts, /*allow_load=*/false, /*allow_store=*/true,
                                  /*check_last_dim=*/true);
    if (inst != CopyInst::kInvalid) return Supported(inst);
}
return Supported(SelectSyncLikeInst(facts));
```

注意 `allow_load=false`：**默认路径只把 shared → global 的 store 自动降成 TMA，global → shared 的 load 不会**。这是本章最容易被误解的一点。load 只有在三种情况下走 TMA：显式 `T.tma_copy`、`prefer_instruction="tma"`、或者处于 warp-specialized 的 producer 分支里（见 4.4.4）。

还有一处同样容易被误解：走到最后的兜底是 `SelectSyncLikeInst`，它的结果是 **SIMT 同步拷贝（`Normal`）**，而**不是** `cp.async`。`cp.async` 是第 4 步选出来的，前提是人给这个 copy 打了 `no_implicit_async_commit_wait` 或 `is_async_copy` 标注——而给流水里的 copy 打这个标注的是 `InjectSoftwarePipeline`（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:288-299`）。所以：**不在 `T.Pipelined` 里的一句裸 `T.copy(A_g, A_s)` 在 sm_90 上会降成同步 SIMT 拷贝**；只有在流水里、或者你显式写 `T.async_copy` 时才会变成 `cp.async`。

#### 4.4.2 约束

`CheckBulkLoad` / `CheckBulkStore`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:195-261`）要求：

* `TargetHasBulkCopy(target)`，即 `arch >= 90`（`_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:117-122`）；
* scope 必须是 global ↔ (`shared` | `shared.dyn`)；
* （check_last_dim 时）最后一维的**位数**必须是 128 的整数倍，即 16 字节对齐；
* dtype 对合法（例如 packed FP4 global → unpacked FP4 shared 只允许 load，反向 store 不支持，见 `MakeTmaUnavailableReason`，`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:531-570`）；
* `CheckGlobalStrides`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:143-193`）：最内层 stride 必须是 1，其余 stride 的字节数必须能被 16 整除，且不能超过 2^40。

1D 路径（`CheckBulkCopy1D`，`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:326-365`）额外要求：global 区域连续、shared 已有 layout 是「语义线性」的、两侧元素数相等、总位数 16 字节对齐。1D 路径不要求最后一维 16B 对齐（它用字节地址而不是描述符坐标），只要求总大小对齐。

#### 4.4.3 描述符在哪儿构建

`Copy::LowerBulk`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2108-2215`）构造 descriptor：

```cpp
TMADesc desc = plan.desc;
desc.data_type = TensorMapDataTypeForTMA(global_tensor->dtype, shared_tensor->dtype);
desc.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
desc.oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
desc.interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
Call create_descriptor = Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());
```

`create_tma_descriptor` 是一个 TIR 内建 op，它**不在 device 上执行**：`LowerHopperIntrin`（`_vendor/tilelang-v0.1.15/src/cuda/transform/lower_hopper_intrin.cc:27-44`）把这些 descriptor 初始化语句收集起来，作为 `tma_descriptor_args` 属性挂到 host 函数上：

```cpp
f = WithAttr(std::move(f), "tma_descriptor_args", substituter.init_desc_arg_map_);
```

wrapper 读这个属性（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:524-525`），并在 `generate_tma_descriptor_args`（第 415-459 行）里按 `TMA_DESC_INIT_FUNC`（第 111-131 行）生成 **host 侧** 的：

```c
CUtensorMap <name>;
...
CUresult r = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &name, type, rank, globalAddress, globalDim, globalStride + 1, boxDim,
    elementStrides, interleave, swizzle, l2Promotion, oobFill);
```

（另有 `TMA_IM2COL_DESC_INIT_FUNC`，第 133-159 行，对应 `cuTensorMapEncodeIm2col`。）所以完整链路是：

```
device IR 里的 create_tma_descriptor(TIR op)
   → LowerHopperIntrin 收集成 host 函数的 tma_descriptor_args 属性
   → wrapper 在 host 侧生成 cuTensorMapEncodeTiled(...)
   → 描述符以 __grid_constant__ const CUtensorMap 参数传给 kernel
   → device 上单线程 cp.async.bulk.tensor 使用它
```

device 侧的 PTX 在模板里：`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/copy_sm90.h:212-270` 是 1D/2D 的 `tma_load`，第 88-210 行是 multicast 变体，第 51-86 行是 `tma_store_cluster`。它们都是 `asm volatile("cp.async.bulk.tensor.Nd....")` 的内联汇编。

#### 4.4.4 mbarrier 配合

load 必须有一个 mbarrier。`LowerBulk`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2131-2155`）的处理顺序是：

1. 如果用户给了 `barrier=` 注解（`T.tma_copy(..., barrier=mbar)`）→ 用它；若它是 `shared.cluster_barrier`，标记 `is_cluster_barrier`。
2. 否则，如果 `T.tma_copy` 没给 barrier → `LOG(FATAL)`，提示「Use T.tma_copy(src, dst, barrier=mbar[idx])」。
3. 否则（普通 `T.copy` 走了 TMA load）→ 由 `lower_args.alloc_mbarrier(1, ...)` 自动分配一个 barrier，命名规则见 `MakeCopyMBarrierName`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:135-138`）。

device 侧的 mbarrier 原语在 `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/barrier.h`：`mbarrier_init`（第 11 行）、`mbarrier_arrive_expect_tx`（第 112、120 行）、`mbarrier_wait`（第 34 行）、`tma_store_arrive`（第 184 行）、`tma_store_wait`（第 194 行）。还有一个专门的融合 pass：`FuseMBarrierArriveExpectTx`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:199-202`，只在模块里真的出现了 TMA 时运行，判据是 `module_has_tma`，`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:39-46`），把 `expect_tx → tma issue → arrive` 合成一条 `arrive_and_expect_tx`。

store 不需要 barrier：它用 `tma_store_arrive` / `tma_store_wait` 这些 bulk group 原语（`T.tma_store_wait(count=0, read=True)`，`_vendor/tilelang-v0.1.15/tilelang/language/builtin.py:370-380`）。

### 4.5 warp specialization 的实现

#### 4.5.1 语言层

两个入口：

* `T.ws(*warp_group_idx)`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/warpgroup.py:19-59`）：把 warp group 序号翻译成 `tx` 的范围。`warp_group_size = 128` 是硬编码的（第 47 行），注释里写着「only available for nvidia gpus」。
* `T.annotate_ws_schedule(WSSchedule(...))`（`_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:151-165`）：挂一份**类型化**的调度，由 `WSSchedule` / `WSRole` / `WSPipeline` / `WSSync` / `WSScope` 组成（`_vendor/tilelang-v0.1.15/tilelang/language/warp_specialize.py`）。

`WSPipeline` 的 docstring（`_vendor/tilelang-v0.1.15/tilelang/language/warp_specialize.py:69-80`）说清了它的语义：

> A full/empty mbarrier pair protecting multi-versioned buffers. The producer waits for the empty barrier and signals the full barrier; the consumer waits for the full barrier and signals the empty barrier. ... Multiple buffers can share one pipeline; conversely, a buffer may be bound to several pipelines whose scopes are strictly nested — its version count is then the product of their depths (outer-major slots).

`WSRole` 还带一个 `max_nreg` 字段（`_vendor/tilelang-v0.1.15/tilelang/language/warp_specialize.py:49-66`），对应 Hopper 的 `setmaxnreg`——即 FlashAttention-3 里提到的「做 MMA 的 warp 拿更多寄存器、只发 TMA 的 warp 让出寄存器」（`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608-fulltext.md:176-178`）。

#### 4.5.2 编译期

三个 pass 按顺序跑（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:101-109`），且只在 `allow_warp_specialized(target)` 为真时才跑：

```python
def allow_warp_specialized(pass_ctx=None, target=None) -> bool:
    if (not is_cuda_target(target)) or (not have_tma(target)):
        return False
    return not pass_ctx.config.get("tl.disable_warp_specialized", False)
```

（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:27-38`。）`have_tma` 要求 `arch >= 9`（`_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py:634-650`），所以**warp specialization 在 Hopper 之前不存在**，这是编译器层面挂上去的限制，不是风格选择。

三个 pass 的分工：

* `AutoWarpSpecialization`（v0.1.15 新增）：按 `tl.enable_auto_warp_specialization` 指定的调度器（目前是 `"role_based"`）**自动推导**一份 `WSSchedule`；没设这个 config 时是 no-op（`_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:6-22`，`_vendor/tilelang-v0.1.15/src/cuda/transform/auto_schedule.cc:1-24`）。
* `MaterializeWSSchedule`：把用户或自动生成的 `WSSchedule` 物化成「角色分支 + 多版本 buffer + mbarrier + TMA + 异步 tcgen05」的显式形式（`_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:158-172`）。它比 v0.1.14 大了很多（diff 3312 行），是 v0.1.15 在 NV 侧最大的单点变化。
* `ProducerConsumerWarpSpecialized`：在 tile-op 层面（`LayoutInference` / `LowerTileOp` 之前）把可分的 pipelined loop 改写成 producer/consumer 两半（`_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:35-48`）。

分类逻辑被抽到了 `_vendor/tilelang-v0.1.15/src/cuda/transform/ws_analysis.h`（v0.1.15 新增），它把语句分成 `kTmaProducer` / `kCpAsyncProducer` / `kCpAsyncRaw` / `kSimtProducer` / `kTmaStore` / `kTcgen05Mma` / `kConsumer`（`_vendor/tilelang-v0.1.15/src/cuda/transform/ws_analysis.h:33-41`）。这个枚举就是 4.4.1 里那个「warp-specialized producer 分支」判据的来源：`ClassifyWarpSpecializedCopy`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:810-891`）对 producer 侧的 load 传 `allow_load=True`，所以**只有在这一分支里，`T.copy` 的 global→shared load 才会自动变成 TMA**。

最后还有 `AnnotateWarpGroupRegAlloc`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:268-271`），把 `set_max_nreg` 注入到 producer/consumer 两侧（`_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:143-156`）。

### 4.6 Blackwell：tcgen05、TMEM、block-scaled

#### 4.6.1 TMEM 是什么、怎么用

PTX ISA 文档的定义（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:29726-29756`）：

> On architecture `sm_100a`/`sm_100f`, the 5th generation TensorCore's Tensor Memory has a two-dimensional structure of 512 columns and 128 rows per CTA, with each cell being 32-bits in size.

以及分配规则（同文件 29758-29768）：

> The Tensor Memory is dynamically allocated. The Tensor Memory must be allocated by a single warp in a CTA ... The unit of allocation is 32 columns and the number of columns being allocated must be a power of 2. When a column is allocated, all 128 lanes of the column are allocated. All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated before the kernel exits.

TileLang 的 `T.alloc_tmem` docstring（`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:254-300`）几乎是这段的逐条翻译，并多说了几件工程约束：列数必须是 2 的幂且 ≥32；分配与释放必须由同一个 warp 完成；TMEM 基地址存在 shared memory 里，作为 TCGEN5.MMA 累加器的偏移；**只有 TCGEN5.MMA 和特定的 TMEM load/store 能访问 TMEM**，所以前后处理必须发生在数据进出 TMEM 之外。最后一条是理解 Blackwell kernel 结构的钥匙：Blackwell 上的典型形态是 shared → (tcgen05.mma) → TMEM → (TMEM load) → 寄存器 → 后处理。

编译期配套：

* `LowerSharedTmem`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:175`）把 `shared.tmem` 缓冲落实成具体的分配槽位。
* `InjectTcgen05Fence`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:266`）在**保守**的位置插入 `tcgen05.fence::before_thread_sync` / `after_thread_sync`——包住 CTA 级 shared 同步、`mbarrier_wait_parity → tcgen05/TMEM 用`、以及 `tcgen05/TMEM 用 → mbarrier_arrive` 这些线性握手点（`_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:125-139`）。它是保守的，**不消除你自己写 mbarrier 协议的责任**——`_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/README.md` 里明确写了这一点。
* 进/出 TMEM 的拷贝走独立的两条 `CopyInst`：`kTMemLoad` / `kTMemStore`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:401-414`），v0.1.15 用 `IsTmemBuffer()` 替代了字面量 scope 判断。

#### 4.6.2 2SM（`use_2cta`）

`LowerBlackwell2SM`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:110-114`）必须在 `LayoutInference` 之前跑，注释说明了原因：`use_2cta` 标记得让 `infer_layout` 看得见。`use_2cta=True` 要求 `cluster_dims` 是 `(2,1,1)` 或 `(1,2,1)`（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:100-120` 的 `gemm_blockscaled` docstring）。为什么不给回退，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:347-353` 的注释已经解释了。

#### 4.6.3 Block-scaled MMA

`T.gemm_blockscaled` 的 CUDA docstring（`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:103-135`）把两条路写得很清楚：

* **SM100**：`C` 在 TMEM → TCGEN05 `kind::mxf8f6f4.block_scale`，A/B 是 shared 里的 FP8/FP6/FP4，SFA/SFB 是已在 TMEM 里的 E8M0 scale factor；`mbar` 必填，编译器隐式插入 `mbarrier_wait_parity`。
* **SM120**：`C` 是 fragment → 同步的 `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`，操作数是 E2M1，scale 是 UE4M3，FP32 累加；A/B 与打包的 scale word 都在 shared；`sf_layout` 描述 scale 的排布；这条路径**忽略** `mbar`。

「不支持的组合直接编译失败，而不是丢 scale」——docstring 里明确写了。这一点值得记住：低精度量化最容易出的错是「scale 被悄悄忽略」，TileLang 在这里选择了显式失败。

### 4.7 cluster 与 DSM

#### 4.7.1 启动路径

`cluster_dims` 从 `T.Kernel` / `T.ClusterKernel` 记录到 launch block（`_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:397-420`，`_normalize_cluster_dims` 会把 `(1,1,1)` 归一成 `None`）。之后：

* codegen 侧：`ClusterInfoExtractor` 读出 `cluster_dims` 并记到 `this->cluster_dims`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:596-627`、`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6560-6561`），用于 threadblock swizzle 的代码生成。
* runtime 侧：`ExtractFuncInfo` 把 `cluster_dims` 变成 `kClusterDimX/Y/Z` 三个 launch param tag（`_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cuda.cc:74-82`）。
* wrapper 侧：`parse_source_information` 读 device 函数的 `cluster_dims` 属性（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:494-497`），`get_kernel_launch_code` 据此选 `KERNEL_CLUSTER_LAUNCH_FUNC_CODE`（第 567-571 行），生成带 `cudaLaunchAttributeClusterDimension` 且先设置 `cudaFuncAttributeNonPortableClusterSizeAllowed` 的启动代码。

DSM 侧的硬件事实来自 Hopper 架构文档（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md:263`）：

> With clusters, it is possible for all the threads to directly access other SM's shared memory with load, store, and atomic operations. This feature is called distributed shared memory (DSMEM) because shared memory virtual address space is logically distributed across all the blocks in the cluster.

#### 4.7.2 三种 SM-to-SM 路径

官方文档 `_vendor/tilelang-v0.1.15/docs/programming_guides/cluster_tma.md` 给了一张表，非常值得直接记住：

| 路径 | 条件 | 硬件指令 | arrive 数 |
| --- | --- | --- | --- |
| TMA 快路径 | 给了 `remote_barrier` 且区域连续 | 一条 `tl::tma_store_cluster` | 1 |
| Multi-TMA | 给了 `remote_barrier` 且 ND 区域不连续 | 每行一条 `tl::tma_store_cluster` | 行数 |
| SIMT 回退 | 没给 `remote_barrier`，或区域无法分解 | `map_shared_rank` 逐元素存 | 若给了 `remote_barrier` 则自动注入 arrive |

这里的「连续」判据是「最内维是否覆盖整个 buffer 宽度」。实现上，`Copy::LowerCluster`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:913-1040`）先看 bulk 能不能整块搬（`tma_store_cluster`，第 968 行），不能就调 `MakeTMARows`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:270-380`）逐行分解，再不能就落到 SIMT（`ClusterCopyReplacer`，`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:1050-1120`）。

#### 4.7.3 限制

* `cluster_mask` 必须是**编译期常量**，不支持动态 mask（`cluster_tma.md`「Notes」节）。
* `cluster_mask` 与 `dst_block` 在同一次 `copy_cluster` 里互斥。
* 可移植 cluster 上限是 8；B200 可通过 `cudaFuncAttributeNonPortableClusterSizeAllowed` 开到 16（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md:73`）。
* 集群场景下建议用 `cudaOccupancyMaxActiveClusters` 算占用率，而不是普通 occupancy calculator（同文件第 52 行）。
* DSM 访问应尽量合并并 32 字节对齐，避免非单位步长（同文件第 60-68 行）。

### 4.8 CuTe DSL 后端

#### 4.8.1 它在 TileLang 里怎么接

* target 判定：`_vendor/tilelang-v0.1.15/tilelang/cuda/codegen.py:8-14`，`is_cutedsl_target` 要求 `target.kind.name == "cuda"` 且 `"cutedsl" in target.keys`；`is_plain_cuda_target` 是它的补集。
* 后端注册：`_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:9-23` 是**另一个** `BackendModule`（`name="cutedsl"`），但它**复用同一个 `pipeline.CUDA_PIPELINE`**——也就是说，从 TIR 到 `LowerTileOp` 的整条 pass 链与普通 CUDA 完全一样，只有 device codegen 和 execution backend 不同。
* device codegen：`target.build.tilelang_cutedsl` / `..._without_compile`，注册在 `_vendor/tilelang-v0.1.15/src/cuda/codegen/rt_mod_cutedsl.cc:86`，打印器是 `CodeGenTileLangCuTeDSL`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cutedsl.cc:210` 起，3740 行）——它输出的是 **Python**（CuTe DSL），不是 C++。
* execution backend：`_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:35-36` 只注册了 `"cutedsl"` 一个。
* adapter：`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/`，入口 `CuTeDSLKernelAdapter`（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/adapter.py:24`），编译器是 `CuTeDSLLibraryGenerator`（`libgen.py`），它把生成的 Python 模块写进临时目录再 import（`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/libgen.py:37-60`）。
* 依赖门禁：`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cutedsl/checks.py:1-88`，要求 `nvidia-cutlass-dsl >= 4.7.0`（`4.3.4` 被显式列入黑名单），并且 `cutlass` 包下真的能找到 `cute`。

#### 4.8.2 它和官方 CuTe DSL / cuTile 的关系

`_vendor/tilelang-v0.1.15/tilelang/contrib/cutedsl/__init__.py` 的第一件事就是 `import cutlass` / `import cutlass.cute`，然后重导出 `cutlass.cute.arch` 的 `sync_threads`、`alloc_smem`、`warpgroup_reg_alloc`、`wgmma_wait_group` 等等。也就是说：

> **TileLang 的 CuTe DSL 后端 = 「TileLang 的 TIR pass 链」+「生成 CuTe DSL Python 源码」+「交给 NVIDIA 的 CUTLASS/CuTe 去编译」。**

它不是 TileLang 自己重写了一遍 CuTe，而是把 CuTe DSL 当成 **codegen target**。`_vendor/tilelang-v0.1.15/tilelang/contrib/cutedsl/` 下自己实现的 `gemm_v1.py` / `gemm_v2.py` / `gemm_tcgen05.py` / `cpasync.py` / `ldsm.py` / `ptx_mma.py` / `cluster.py` 等，是把 CUTLASS 里零散的 atom 与 helper 组织成 TileLang 需要的那几个入口。

另一条 NVIDIA 官方的路线是 **cuTile Python**，它和 CuTe DSL 并不是一回事。corpus 里 cuTile 的 README（`../tilelang-corpus/11-related-compilers/raw/github-nvidia-cutile-python.md:64-70`）写着：

> cuTile Python generates kernels based on [Tile IR] which requires NVIDIA Driver r580 or later to run. Furthermore, the `tileiras` compiler (version 13.2) only supports Blackwell GPU and Ampere/Ada GPU. Hopper GPU will be supported in the coming versions.

所以三者可以这样定位：

| 路线 | 前端抽象 | 编译目标 | 与 TileLang 的关系 |
| --- | --- | --- | --- |
| TileLang CUDA 后端 | `T.Kernel` / `T.copy` / `T.gemm` | CUDA C → nvcc → cubin | TileLang 自己的主线 |
| TileLang CuTe DSL 后端 | 同上（同一套 pass） | CuTe DSL Python → CUTLASS | TileLang 把 CUTLASS 当 codegen target |
| cuTile Python | `ct.load` / `ct.store` / tile 运算 | Tile IR → `tileiras` | 同一抽象层次的竞争路线，编译器与运行时完全独立 |

一个诚实的限制：corpus 里 `nvidia-cutedsl-docs.md` 只有 22 行，是文档站的导航壳，没有正文。所以本章关于 CuTe DSL「官方怎么描述自己」的部分只做了定位性引用，正文判断全部来自 TileLang 侧的 `checks.py` 与 `contrib/cutedsl/`。这一点记在 `references.md` 的「待补资料」里。

### 4.9 一张「用户写法 → 架构 → 生成指令」对照表

下面这张表是本章的核心速查。所有「生成指令」列的内容都能在上面的源码位置核对到；「架构限制」列来自 `_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc` 的 arch 阈值与 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc` 的谓词。

| 用户写法 | 目标架构 | 选中的指令 | 依据（v0.1.15） |
| --- | --- | --- | --- |
| `T.gemm(A_s, B_s, C_frag)`，f16→f32 | sm_70 | `mma.sync`（Volta 形态） | `AllowVoltaMma`，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:99-117` |
| 同上 | sm_75 | `mma.sync`（Turing 形态） | `AllowTuringMma`，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:131-145` |
| 同上，`M>=64`，256 线程 | sm_90/90a | `wgmma.mma_async` | `AllowWgmma` + `CheckWgmma`，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:89-97`、`40-78` |
| 同上，`M<64` 或线程数不是 4 的倍数个 warp | sm_90 | `mma.sync` | `AllowWgmma` 的 `op.m_ >= 64`、`num_warps % 4 == 0` |
| `T.gemm(A_s, B_s, C_tmem, mbar=mbar)` | sm_100 | `tcgen05.mma` | `AllowTcgen5Mma`，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:80-87` + `_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h` |
| `T.tcgen05_gemm(..., mbar=mbar)` | sm_100 | `tcgen05.mma`（不隐式等待） | `SelectInst` 第一段，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-346` |
| `T.wgmma_gemm(...)` | sm_90 | `wgmma.mma_async`（不隐式等待） | 同上，且不满足即 `FatalWgmmaUnavailable` |
| `T.tcgen05_gemm(..., use_2cta=True)` | sm_100，cluster (2,1,1) | `tcgen05.mma.cta_group::2` | `Use2CtaRequested`，`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:119-129`、`354-361` |
| `T.gemm_blockscaled(..., mbar=...)` | sm_100 | `tcgen05.mma.kind::mxf8f6f4.block_scale` | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:110-116` |
| `T.gemm_blockscaled(..., sf_layout=...)` | sm_120 | `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale` | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:118-123` |
| `T.gemm` 形状/精度都不满足 | sm_70/sm_75 | `fma`（无 Tensor Core） | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:368-373` |
| `T.copy(A_g, A_s)`（裸写，不在流水里） | sm_80+ | SIMT 同步 `Normal` | `SelectSyncLikeInst`，`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:612-626`、`:797-807` |
| `T.copy(A_g, A_s)`（在 `T.Pipelined` 里） | sm_80+ | `cp.async` + `commit_group` | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:288-299` 打标注 → `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:757-760` |
| `T.copy(A_s, C_g)` | sm_90+ | `cp.async.bulk.tensor`（TMA store） | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:797-803` |
| `T.copy(A_g, A_s, prefer_instruction="tma")` | sm_90+ | TMA load（约束不满足则报错） | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:762-780` |
| `T.tma_copy(A_g, A_s, barrier=mbar)` | sm_90+ | `cp.async.bulk.tensor`（只发 producer 半边） | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:748-756`、`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2108-2215` |
| `T.async_copy(A_g, A_s)` | sm_80+ | `cp.async` + `commit_group` | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:830-905` |
| `T.copy(A_s, B_s)`（shared→fragment） | sm_75+ | `ldmatrix` | `CheckLDSMCopy`，`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:407-410` |
| `T.copy(C_frag, B_s)` | sm_90+ | `stmatrix`（m16n8 形态要 sm_100） | `TargetHasStmatrix`，`_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:104-110` |
| `T.copy(C_tmem, C_frag)` | sm_100 | TMEM load | `CheckTMemLoad`，`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:417-420` |
| `T.copy_cluster(A_g, A_s, cluster_mask=0b11)` | sm_90+ | `cp.async.bulk.tensor...multicast::cluster` | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:718-736`、`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/copy_sm90.h:88-210` |
| `T.copy_cluster(A_s, B_s, dst_block=1, remote_barrier=b)` | sm_90+ | `cp.async.bulk.shared::cluster` | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:913-1040`、`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/copy_sm90.h:51-86` |

表里 `T.tcgen05_gemm(...)` 与 `T.wgmma_gemm(...)` 两行的「用户写法」指的是**这类语义**：
如 4.3.6 节所述，这两个名字在 v0.1.15 的 `tilelang.language` facade 上取不到，要写显式变体
请先在目标环境里确认名字存在；否则用 `T.gemm(..., mbar=mbar)` 等价的同步形式。

### 4.10 版本差异：v0.1.14 → v0.1.15（NV codegen 部分）

**方法**：`_vendor/tilelang-v0.1.14/` 与 `_vendor/tilelang-v0.1.15/` 都是 `git clone --depth 1 --branch <tag>`，工作树干净。所以直接用 `diff -rq A B` 找差异文件，再对有差异的文件跑 `diff -u`。下面是这条命令筛出来的、与 NV codegen 有关的部分。

```bash
diff -rq _vendor/tilelang-v0.1.14/tilelang/cuda _vendor/tilelang-v0.1.15/tilelang/cuda
diff -rq _vendor/tilelang-v0.1.14/src/cuda      _vendor/tilelang-v0.1.15/src/cuda
```

1. **block-scaled 从 `gemm.cc` 独立出去。** 新增 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm_blockscaled.cc`（128 行），`_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_tcgen05_blockscaled.py`，以及整个新包 `_vendor/tilelang-v0.1.15/tilelang/tileop/gemm_blockscaled/`。对照 diff 可见 v0.1.14 的 `gemm.cc` 里有一段内联的 blockscaled 分派（`if (op.sfaRegion_.defined() || ...) return kCudaMMABlockScaled;`）与 `tl.get_tcgen5_blockscaled_instr_desc` 的注册，在 v0.1.15 里都从 `gemm.cc` 删掉，移到新文件。`_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/__init__.py` 也多注册了 `cuda.tcgen05_blockscaled`。
2. **TMEM 判断从字面量 scope 改成谓词。** `AllowTcgen5Mma` 里 `op.c_.scope() == "shared.tmem"` 变成 `IsTmemBuffer(op.c_)`；`CheckTMemLoad/Store` 同样（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc` diff）。
3. **TMA 判定函数改名/拆分。** `CopyInstIsTMA(inst)` 拆成 `CopyInstIsTMALoad` 与 `CopyInstIsTMAStore`（`copy_analysis.cc` diff）；
4. **warp-specialized copy 分类泛化。** `ClassifyWarpSpecializedProducerCopy` → `ClassifyWarpSpecializedCopy`，并新增一条「`src` 在 shared、`dst` 在 global 时直接走 `SelectCopyInstForLowering`」的旁路——也就是说 store 现在也走通用判定，而不是原来的 producer-only 路径。
5. **新增自动 warp specialization。** `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py` 在 `MaterializeWSSchedule()` 之前插入 `AutoWarpSpecialization()`；新增 pass config `tl.enable_auto_warp_specialization`（`_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py` diff）；新增 `_vendor/tilelang-v0.1.15/src/cuda/transform/auto_schedule.cc`、`_vendor/tilelang-v0.1.15/src/cuda/transform/auto_schedule/`（含 `role_based_scheduler.cc`、`preprocess_ir.cc`、`memory_detector.h`）。
6. **WS 分析抽成公共头文件。** 新增 `_vendor/tilelang-v0.1.15/src/cuda/transform/ws_analysis.{h,cc}`，`TileStmtKind` 枚举（`kTmaProducer` 到 `kConsumer`）被抽出来供自动调度器复用——注释原文是「Extracted from producer_consumer_ws.cc so automatic schedulers reuse the same classification the legacy producer/consumer rewriter has been validated with」。
7. **`MaterializeWSSchedule` 大幅扩容**（`_vendor/tilelang-v0.1.15/src/cuda/transform/materialize_ws_schedule.cc` 的 diff 有 3312 行），这是 v0.1.15 NV 侧最大的单点变化。
8. **`MaterializeKernelLaunch` 参数化。** `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:70` 从 `MaterializeKernelLaunch()` 变成 `MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)`。
9. **`CodeGenTileLangCUDA` 的向量展开。** 新增 `EmitPackedX2Call` / `EmitPerLaneScalarCall`（声明见 `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.h` diff），新增 `VisitStmt_(const AssertStmtNode *)`；`t` lanes=64 的整型向量不再打印 `int8`（不存在的类型）而是 `longlong4`；fp4/fp8 的多 lane 向量改用 `tl::make_fp4_vec<>` / `tl::make_vec<>` 的逐 lane 打包构造（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:986-994`、`1132+`）。
10. **可向量化 cast 增加 fp4→fp8。** `_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc` 的 `IsCudaVectorizableCast` 新增 `float4_e2m1fn → float8(e4m3)` 一条。
11. **deprecation 文案更新。** `tl.disable_tma_lower` 的提示从「will be removed in v0.1.10」改成不带版本号；docstring 同步（`_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py` diff）。语义没变，但说明这个兼容开关至今没被删。
12. **examples**：新增 `_vendor/tilelang-v0.1.15/examples/ascend/` 与 `_vendor/tilelang-v0.1.15/examples/kpool/`（例目录数 55 → 57）；`_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/README.md`、`_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/mxfp8_illustrated.md`、`_vendor/tilelang-v0.1.15/examples/deepseek_v32/sparse_mla_fwd.py` 等有改动。
13. **纯 NV 的 `wrapper.py` 变化很少。** `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py` 的 diff 主要是新增 `TLAscendSourceWrapper`（昇腾），NV 部分只有 `_TYPE_MAP` 加了 `uint64 → uint64_t`、`get_stream_type` 的返回类型放宽成 `dict[str, str] | None`。这说明 v0.1.15 的 host wrapper 形态对 NV 而言基本冻结。

**找不到的部分也说清楚**：两个 vendor 树都是浅克隆，没有 `.git` 历史，所以无法用 `git log` 看提交信息；上面全部结论来自文件级 diff。`_vendor/tilelang-v0.1.15/tilelang/contrib/nvcc.py`、`_vendor/tilelang-v0.1.15/tilelang/contrib/nvrtc.py` 在两版之间**没有差异**（`diff -rq` 未列出），说明这两个工具封装在 v0.1.15 里没有变化。

---

## 5. 进阶与陷阱

1. **「我明明写了 `T.copy`，为什么没有 TMA」** ——先看方向：默认只有 store 自动走 TMA。load 要 TMA，请用 `T.tma_copy` 或 `prefer_instruction="tma"`，并确认最后一维是 16 字节的整数倍、global stride 是 16 字节对齐且最内层 stride 为 1。
2. **「`T.gemm` 没有用 5 代 Tensor Core」** ——先看 `C` 的作用域：TCGEN05 要求 `C` 是 `T.alloc_tmem`。`C` 在 fragment 里时，即使目标是 sm_100 也不会选 TCGEN05。
3. **显式变体不降级。** `T.wgmma_gemm` / `T.tcgen05_gemm` 在上约束不满足时是 `LOG(FATAL)`，不是回退。它们的定位是「我确知要什么，帮我到底」，而不是「尽量给我这个」。反过来，`T.copy` 的 `prefer_instruction="tma"` 也是「要么 TMA，要么报错」。
3b. **但先确认这些名字在你的环境里能从 `T` 上取到。** 4.3.6 节给出的源码事实是：在 v0.1.15 里 `T.wgmma_gemm` / `T.tcgen05_gemm` 等显式变体不在 facade 的导出名单上，而官方示例仍在调用它们。别照着示例直接抄。
4. **`use_2cta` 没有单 CTA 回退。** 因为形状约定本身就建立在「B 的 N 一半在邻居 CTA」这个前提上（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:347-353`）。
5. **TMA 描述符在 host 上编码，所以 global base 必须是 host 能看到的地址。** 如果 global buffer 的基址是在 device 函数体里算出来的（device-bound），描述符 TMA 路径不可用——`GetTmaDescriptorBaseIsDeviceBound` 与 `MakeTmaUnavailableReason`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:81-84`、`522-535`）。此时 1D bulk 路径（直接用 device 算出的地址）仍然可用，因为它不建 TensorMap。
6. **mbarrier 的 arrive 计数必须和实际到达者一致。** `T.alloc_barrier(arrive_count=...)` 里 0 是非法值（`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:194-222` 会 raise）。Multi-TMA 路径会把 `arrive_count` 自动改成行数——这是编译器替你做的事，你要知道它发生了，否则手工调 `arrive_count` 会和它打架。
7. **TMEM 必须显式释放，且必须在同一个 warp 上分配/释放。** 不写 `T.deallocate_tmem` 时 TileLang 会在分配块结束时自动释放（`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:260-282`），但如果你手工提前释放，就要自己保证「同一 warp」这条约束。
8. **warp specialization 只在 Hopper 及以后存在。** `allow_warp_specialized` 会先检查 `have_tma(target)`（`arch >= 9`），不满足直接跳过整个 WS 分支。
9. **`cluster_mask` 是编译期常量。** 想按运行期数据决定 multicast 形状，必须写多个 kernel 或者用 `dst_block`（它可以是运行期表达式）。
10. **cluster 大小不是随便开的。** 可移植上限 8，B200 可开到 16（需设 `NonPortableClusterSizeAllowed`，wrapper 已经替你设了），但更大的 cluster 会降低可并行 block 数（`blackwell-tuning-guide.md:73`）。
11. **架构条件特性会破坏前向兼容。** 用 `sm_100a` / `compute_100a` 编译出的 PTX 既不前向也不后向兼容（`blackwell-compatibility-guide.md:28`）。这意味着给 B200 编的 cubin 不能直接丢到 H100 上跑——反过来也不行。跨代部署要靠 PTX JIT 或多 `-gencode`，这也是为什么 target 的 `code` 列表支持多个条目。
12. **本章所有结论都是源码级的，不是实测的。** 本章写作环境没有 NVIDIA GPU，因此没有任何一条「这样写更快」的性能断言；凡是能看到的性能口径，都来自 corpus 里的厂商文档，且在本章里只作架构描述使用。

---

## 6. 本章小结

* TileLang 的 NV 路径是「一条 TIR pass 链 + 一次 host/device 切分 + 两条独立的 codegen」。device 出 CUDA C（或 CuTe DSL Python），host 出 wrapper 的 `call()`（或 TVM-FFI packed 模块）。
* 后端的全部可选项写在一张注册表里：`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:123-143`。三种 execution backend（`tvm_ffi` / `cython` / `nvrtc`）共用同一个 `TLCUDASourceWrapper`，所以生成的 kernel 源码相同。
* `T.gemm` 的指令选择是一个可完整读完的谓词链（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375`）：显式变体 → `use_2cta` → TCGEN05 → WGMMA → Volta/Turing 检查 → `cuda.mma`。决定 TCGEN05 的关键是 **`C` 在不在 TMEM**；决定 WGMMA 的关键是 **sm_90 + M≥64 + warp 数是 4 的倍数 + dtype/转置约束**。
* `transpose` 与 `policy` 通过两条路影响结果：低精度 dtype 在 WGMMA 上对主序有硬约束（`CheckWgmma`），而 `policy` 决定 `m_warp × n_warp` 的划分（`ComputeWarpPartition`）。layout 则由指令反推（`GemmWGMMA.infer_shared_layout` 的 swizzle 档位）。
* `T.copy` 默认**只把 store 降成 TMA**；load 想走 TMA 需要显式提示或处在 warp-specialized producer 分支。TMA 的成功条件集中在 `CheckBulkLoad/Store` 与 `CheckGlobalStrides`：架构 ≥ sm_90、global↔shared、16 字节对齐、最内层 stride 为 1。
* TMA 是「host 编描述符 + device 单线程发射 + mbarrier 完成通知」。描述符由 `LowerHopperIntrin` 提升到 host 函数属性 `tma_descriptor_args`，再由 wrapper 生成 `cuTensorMapEncodeTiled(...)`。
* warp specialization 在语言层是 `T.ws(i)` / `T.annotate_ws_schedule`，在编译期是 `AutoWarpSpecialization → MaterializeWSSchedule → ProducerConsumerWarpSpecialized`（外加 `AnnotateWarpGroupRegAlloc`），并且**只在 Hopper 及以后开启**。
* Blackwell 的新东西是 TMEM 与 tcgen05：累加器不再在寄存器里，只有 TCGEN05 与 TMEM load/store 能碰它；block-scaled MMA 在 SM100 走 TCGEN05、在 SM120 走 `mma.sync`。
* cluster 与 DSM 通过 `T.ClusterKernel(cluster_dims=...)` + `T.copy_cluster(cluster_mask=... | dst_block=...)` 使用，硬限制是「mask 编译期常量」「可移植 cluster ≤ 8」「DSM 访问要合并对齐」。
* CuTe DSL 后端与 CUDA 后端共用同一条 TIR pass 链，只是把 codegen target 换成了 CuTe DSL Python；它与 NVIDIA 官方的 cuTile（走 Tile IR + `tileiras`）是完全独立的两条路。

## 延伸阅读

* 想看 codegen 的全部细节：`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc`（6631 行）——建议从 `AddFunction`（第 6509 行）和 `PrintType`（第 797 行）两头往里读。
* 想看 TMA/cluster 的用户文档：`_vendor/tilelang-v0.1.15/docs/programming_guides/cluster_tma.md` 与 `_vendor/tilelang-v0.1.15/docs/programming_guides/instructions.md`。
* 想看 Blackwell 的 shape 白名单：`_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h` 与 `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md` 的 9.7.18 节。
* 想看 pipeline、多缓冲与同步插入的通用机制：见第 22 章。
* 想看这套机制在昇腾上的对应物：见第 25 章。
* 本章所有引用条目的出处与核验状态：见 [references.md](references.md)。
