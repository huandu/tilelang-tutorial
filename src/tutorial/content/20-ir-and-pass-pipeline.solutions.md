# 第 20 章 习题答案

每题给出**解题思路 → 参考答案 → 常见错误**。涉及的源码行号均可在
`_vendor/tilelang-v0.1.15/` 里现场核对。

---

## E1

**思路**：把 2.2 节的表重述一遍，再回 `attr.h` 找常量名。

**参考答案**：

* `IRModule`：一个编译单元，是 `GlobalVar → BaseFunc` 的字典，一个 kernel 通常是
  `{"main": PrimFunc}`。
* `PrimFunc`：一个函数，等于“参数 + 一个 `Stmt` 体 + 属性”。
* `Buffer`：一块带形状、类型、存储作用域的内存对象，由 `shape`/`dtype`/`data`/
  `scope()` 描述。
* `Var`：变量，循环变量、参数句柄、buffer 的 `data` 指针。
* `PrimExpr`：表达式树节点基类（`Var`/`IntImm`/`Add`/`Ramp`/`Call`…）。
* `SBlock`：调度块，声明式记录 `reads`/`writes`/`alloc_buffers`/`annotations`，
  并包住一段 `body`。
* 轴：grid 轴是 `For(kind=THREAD_BINDING)`；线程轴是
  `AttrStmt("thread_extent")`，靠 `thread_tag` 区分。

`"tilelang_root"` 硬编码在
`_vendor/tilelang-v0.1.15/src/transform/common/attr.h:17`
（`constexpr const char *DeviceMainBlockName = "tilelang_root";`）。
判断主块用同名文件 `:24` 的辅助函数：`node->name_hint == DeviceMainBlockName`。

**常见错误**：把 `SBlock` 说成“一个循环”。它是作用域块，可以包含循环，也可以
只是一个语句序列。

---

## E2

**思路**：直接引用两条 pipeline 里的调用。

**参考答案**：

```text
CUDA   : MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)      # pipeline.py:70
Ascend : MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False,
                                 default_threads=None,
                                 unsupported_annotations=["cluster_dims"],
                                 launch_dim_tags=["cthread"])                          # pipeline.py:45
```

1. `lower_thread_binding=False` 表示**丢弃线程占位符、忽略 `tl.launch_threads`**：
   昇腾在 kernel 作用域没有 `threadIdx`，线程域只存在于 `T.SimtVF` 内部（它自己
   再发出线程作用域）。
2. `cthread` 是昇腾的子块编号（sub-block-id）维度，由 `T.MixedKernel` 发出、由
   昇腾代码生成从 `thread_extent` 的 `AttrStmt` 上读回。它属于 launch nest，
   不属于线程域。
3. 因为“一个 program 是什么”直接决定了并行语义：CUDA 的 program 是一个 thread
   block（内部有 SIMT 线程），昇腾的 program 是一个 AI core（Core 内部由不同硬件
   流水线并行）。这决定了后续“谁并行、怎么同步”的全部选择。

**常见错误**：以为 `lower_grid_binding=False` 也适用于 Ascend。实际 Ascend 是
`lower_grid_binding=True, lower_thread_binding=False`，**保留网格、丢弃线程**。

---

## E3

**思路**：数组下标对应语句序号（0=copy A，1=copy B，2=gemm）。

**参考答案**：

1. `stage=[0,0,2]` 表示三条语句分别属于第 0、0、2 级；gemm 是 2，因为它是最远的
   消费者（`num_stages=3` 时合法 stage 是 0..2），planner 把消费者尽量放到靠后的级，
   以便它前面的两级空间专门留给搬运重叠。若写成 3 就超出了 `num_stages` 允许的范围。
2. `async_producers=[1,1,0]` 里 `1` 表示这条语句是异步生产者（两个 `T.copy`），
   `0` 表示不是（`T.gemm` 是消费者）。
3. 因为 `PipelinePlanning` 是**规划类** pass：它只把规划结果写成注解，真正的循环
   拆分与多缓冲由下一行 `InjectSoftwarePipeline` 完成（`pipeline.py:130`）。

**常见错误**：把 stage 数组读成“时钟周期”。它是**相对流水级**，不是时间。

---

## E4

**思路**：从本章 3.4 与 4.1 找例子。

**参考答案**：

* 规划类（只写注解/元数据）：`PipelinePlanning`（写 `software_pipeline_*`）、
  `LayoutInference`（写 `layout_map` / `parallel_loop_layout`）、
  `MaterializeScheduleUnits`（打资源标签）、`EstimateLatency`（写 latency/II）。
  它们把信息写在**循环或块的 `annotations` 上**，或写进 `ScheduledTIR` 的元数据。
* 改写类（真的改语句）：`InjectSoftwarePipeline`（拆循环、加版本）、
  `LowerTileOp`（消灭 tile op）、`VectorizeLoop`（生成 `Ramp`）、
  `SplitHostDevice`（拆函数）、`MakePackedAPI`（改签名）。

**常见错误**：把 `MaterializeKernelLaunch` 归到“改写类”。它确实改语句（物化
launch nest），但它属于“前端合法化”，不是“流水/布局”那一类的规划/改写配对。

---

## E5

**思路**：区分“TIR 只知道指针”与“Buffer 描述形状”。

**参考答案**：

1. 因为 TIR 层面参数只是一个**指针句柄**（`T.handle`），形状信息要靠
   `T.match_buffer` 才能附加到它上面。`buffer_map` 里存的就是 `x_handle → x`。
2. 是**声明**。证据：`T.reads()` 是空的、`T.writes(...)` 用区间表达式描述写入范围，
   它们本身不产生任何内存访问指令；它们的用途是给 layout 推导与同步分析用。
3. `local.var` 是**标量变量**，不参与 shared memory 合并、不参与线程级同步；
   `shared` 会进 shared memory 分配、被 `MergeSharedMemoryAllocations` /
   `MergeUBAllocations` 合并、并触发 `ThreadSync` / `InsertSync` 的同步分析。

**常见错误**：把 `T.reads()/T.writes()` 当成 `T.copy` 那样的搬运语义。

---

## E6

**思路**：直接跑脚本。

**参考答案**：

```text
CUDA    : 73
Ascend  : 80
同名（模块限定名去重取交集）: 39
```

`Simplify` 两侧都出现，且**都要注意模块限定名**：CUDA pipeline 里既有
`tilelang.transform.Simplify`（例如 `pipeline.py:88`）也有
`tirx.transform.Simplify`（例如 `pipeline.py:211`）。按叶子名去重会把它们算成一个，
按模块限定名去重会算成两个。这正是本章用“模块限定名”口径的原因。

**常见错误**：只数 `mod =` 行的字面次数（会把多行调用漏掉）。脚本已处理多行调用。

---

## E7

**思路**：`--grep` 是纯文本搜索。

**参考答案**：

第一条命令会命中锁定版本里多个文件（例如
`tilelang/ascend/pipeline.py`、`tilelang/ascend/transform/__init__.py`、
`src/ascend/transform/insert_nd2nz.cc`、`testing/ascend/transform/
test_ascend_insert_nd2nz.py` 等）。第二条命令（搜 `InsertNd(`）**没有任何输出**。

这说明：锁定版本里根本没有名为 `InsertNd` 的 pass；**真名是 `InsertNd2Nz`**。
`InsertNd` 只是早期综合文档里的旧写法（corpus 的
`90-synthesis/02-compiler-internals-map.md` 顶部有版本提示）。

**常见错误**：把“文档里写过”当成“代码里有”。判据永远是 grep 源码。

---

## E8

**思路**：对照 before/after 找唯一差异行。

**参考答案**：

1. after 里多了一行 `T.fence_proxy_async()`。
2. 它插在 shared 写（`smem[0] = ...`）之后、异步的 `T.ptx_wgmma_ss(...)` 之前。
   原因是 Hopper 把内存指令分成 generic proxy 与 async proxy 两条路径：当异步指令
   （`wgmma` / `tma.load` / `cp.async.bulk`）跟在 generic 流量（`ldmatrix`、
   `cp.async`、shared store）之后时，硬件要求一条 `fence.proxy.async` 来保证顺序，
   否则可能出现竞态或未定义行为。
3. pass 在 CUDA pipeline 的 `pipeline.py:256` 被调用。它对没有 TMA / async-proxy
   编程模型的目标是 no-op，pass 自己会检查 PrimFunc 的 target（文档里写得很清楚：
   “is a no-op on targets that lack the TMA / async-proxy programming model”）。

**常见错误**：以为这条 fence 是给线程同步用的。它是**proxy 状态同步**，与
`ThreadSync` 的线程屏障不是一回事。

---

## E9

**思路**：回看 4.1 的九组，找出两侧用不同 pass 解决同一问题的地方。

**参考答案**：

* **主题一：流水。** CUDA 用 `PipelinePlanning` + `InjectSoftwarePipeline`；
  Ascend 没有这两个，改由 `AutoSchedule`（+ `PrepareMultiBuffer` /
  `MaterializeMultiBuffer`）统管。**注意**：底座里 `TileLang` 的这两件套是通用的，
  在适配器仓库（`tilelang-ascend-v0.1.2.000`）里反而会用到它们，所以“谁负责流水”
  在不同昇腾路线下也不同。
* **主题二：布局。** CUDA 用 `LayoutInference`；Ascend 用 `AscendLayoutInference`
  （`src/ascend/transform/layout_inference.cc:1684`）+ `NormalizeAscendFractalStorage`
  + `InsertNd2Nz` + `AscendInsertOOBPadding`，因为 Cube 的 fractal/NZ 与 OOB
  是硬约束。
* **主题三：同步与片上内存。** CUDA 用 `ThreadSync("shared"/"shared.dyn")` +
  `InjectFenceProxy` + `MergeSharedMemoryAllocations`；Ascend 用
  `AscendThreadSync("global"/"shared"/"shared.dyn")` + `InsertSync` +
  `RewriteFlagToBuf` + `MergeUBAllocations`。

**常见错误**：把“分岔”理解成“完全没有共享”。实际共享走廊有 39 个同名 pass
（`NarrowDataType`、`FlattenBuffer`、`VectorizeLoop`、`SplitHostDevice`、
`MakePackedAPI` 等）。

---

## E10

**思路**：把 5.1 的五步流程套到这个 kernel 上。

**参考答案**（命令与预期）：

```bash
TL_LOWER_TRACE=both python3 my_kernel.py      # 全量抓 before/after
# 或对固定几个 pass：
python3 code/03-lower-trace-walkthrough/lower_trace_walkthrough.py --target cuda
```

逐 pass 预期：

* `LayoutInference`：`T.Parallel(256, 8)` 循环上出现 `parallel_loop_layout`，
  说明 8 个元素落到每个线程的 8 个寄存器槽。
* `LowerTileOp`：这个 kernel 没有 `T.copy`/`T.gemm`，所以主要变化发生在
  `T.Parallel` 的展开而不是 tile op 降级，看不到 `tl.tileop.*` 就对了。
* `VectorizeLoop`：每个线程的 8 个 fp16 元素被合并成向量访问（`Ramp`），
  生成代码里应出现一次 `uint4` 级别的 load/store。
* `ThreadSync`：这个 kernel 没有共享内存，所以**不应插入**
  `T.tvm_storage_sync("shared")`。如果出现了，先怀疑是不是上游把某个 buffer 放到了
  shared。
* `MakePackedAPI`：host 入口变成 `(args, num_args)` 形式，并插入参数/形状/stride
  检查。

**常见错误**：期望每个 pass 都改变 IR。像 `ThreadSync` 在没有 shared buffer 的
kernel 上“什么都不做”**才是正确结果**。

---

## E11

**思路**：读源码里的三处日志。

**参考答案**：

1. 因为把屏障从 `if` 里**提到外面**之后，冲突的两端都还在分支内部，屏障不再夹在
   它们之间，竞态依旧存在。所以警告明确写“This is not a fix”。
2. 正确做法是对屏障所依赖的条件加约束（例如 `T.assume` 一个参数范围），让编译器能
   证明条件是 warp 一致（或块一致）的，从而**把屏障留在分支内部**。
3. `:327` 的 `LOG(FATAL)`：某个必需 shared 同步落在发散区域，且参与线程数不是
   warp size 的整数倍，无法安全降成一个 `bar.sync`（源码注释说明这是
   “thread count is not a multiple of the warp size”的 workaround）。
   `:1068` 的 `LOG(FATAL)`：某个分支里的参与线程不是 warp 一致的，部分屏障要求
   每个未退出 warp 的所有线程都到达屏障。

**常见错误**：把这条警告当成无害信息。它意味着结果可能偶发错误。

---

## E12

**思路**：读 `storage_rewrite.cc` 里 pass 函数开头那段注释。

**参考答案**：

`src/transform/storage_rewrite.cc:1933-1935` 的注释写着：
“Always disable reuse currently, for shared memory reuse we depend on
MergeSharedMemoryAllocations pass, for register reuse we depend on nvcc or other
compiler itself.”

也就是说：`StorageRewrite` 在 v0.1.15 里 `enable_reuse = false`（:1932），
共享内存复用交给专门的合并 pass，寄存器复用交给下游编译器（nvcc 等）。

共享内存复用的真正入口：

* CUDA：`MergeSharedMemoryAllocations`（`tilelang/cuda/pipeline.py:251`），
  开关 `enable_aggressive_merge` 与 `disable_reuse`（来自
  `should_enable_aggressive_merge` / `should_disable_shared_memory_reuse`）。
* Ascend：`MergeUBAllocations(align_bytes=32, disable_reuse=...)`
  （`tilelang/ascend/pipeline.py:171`），复用契约来自 `InferBufferAliases`
  （手动调度）或 `InsertSync`（自动调度）。

**常见错误**：以为改 `StorageRewrite` 就能调共享内存复用。它管的是局部存储规划。

---

## E13

**思路**：直接由 `figures/fig-02-pipeline-phases.md` 或
`dump_pass_order.py` 的输出整理。

**参考答案**（要点）：

| 组 | CUDA 代表 | Ascend 代表 | 只有一侧有 |
| --- | --- | --- | --- |
| ① 前端合法化 | `MaterializeKernelLaunch` | `MaterializeKernelLaunch` + `RewriteDualCopy` | `AutoWarpSpecialization`（CUDA）/ `LegalizeSimdMerging`（Ascend） |
| ② 流水 | `PipelinePlanning`、`InjectSoftwarePipeline` | （折叠进③的 `AutoSchedule` 段） | 基本只有 CUDA |
| ③ 布局 | `LayoutInference` | `AscendLayoutInference`、`InsertNd2Nz` | `NormalizeAscendFractalStorage`（Ascend） |
| ④ tile op 降级 | `LowerTileOp` | `AscendLowerTileOp`、`AscendSimdVFLowerParallel` | `LowerSharedTmem`/`LowerSharedBarrier`（CUDA） |
| ⑤ 内存规划 | `StorageRewrite`、`MergeSharedMemoryAllocations` | `MergeUBAllocations`、`RewriteAscendBufferVersionLayout` | `StorageRewrite` 只有 CUDA |
| ⑥ 向量化/展开 | `VectorizeLoop`、`UnrollLoop` | 同左 + `RewriteFp4ToFp4x2`、`AscendRemoveNoOp` | `RewriteFp4ToFp4x2`（Ascend） |
| ⑦ 同步 | `ThreadSync`、`InjectFenceProxy` | `InsertSync`、`AscendThreadSync`、`RewriteFlagToBuf` | `InjectFenceProxy`（CUDA）/ `RewriteFlagToBuf`（Ascend） |
| ⑧ host/device 切分 | `SplitHostDevice` | `SplitHostDevice` | — |
| ⑨ API 打包 | `MakePackedAPI`、`PersistThreadblock` | `MakePackedAPI` | `PersistThreadblock`（CUDA） |

一句话总结：**两条 pipeline 在①④⑤⑥⑦⑧⑨上大量重叠（共享同一个降低走廊），
分歧集中在②③两处以及⑦的同步机制，也就是“谁负责流水与布局、同步用哪种协议”。**

**常见错误**：把“Ascend 不调用 `PipelinePlanning`”写成“Ascend 没有软件流水”。
昇腾的流水由自动调度段负责，只是换了实现方式。
