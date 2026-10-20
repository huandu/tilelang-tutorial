# 第 04 章 习题答案

与 [exercises.md](exercises.md) 一一对应。每题给出**解题思路**、**参考答案**与**常见错误**。
凡涉及源码的答案都给了文件:行号；凡涉及本章脚本输出的数字，都注明可以用
`code/pass-order/dump_pass_order.py` 或 `code/repo-map/inventory.py` 复现。

本章没有任何需要在本机实测才能得出的结论；涉及真实硬件的地方一律标注为“未验证”。

---

## E1 概念题 ★

**解题思路**：先背下 2.1 节的竖线，再回到源码确认每个入口的行号，最后回答“哪些是共享的”。

**参考答案**：

1. ① 前端（`tilelang/language/` 的两条前端路线）→ ② 编译期（入口
   `tilelang.engine.lower.lower_with_context`，`tilelang/engine/lower.py:144`）
   → ③ 代码生成（`device_codegen`，`tilelang/engine/lower.py:99`）
   → ④ 宿主封装（`host_codegen`，`tilelang/engine/lower.py:65`）
   → ⑤ 运行时（`JITKernel._create_adapter_from_artifact`，`tilelang/jit/kernel.py:324`）。
2. 因为第 ② 阶段里只有一行会因后端而变：`mod = context.lower(mod)`
   （`tilelang/engine/lower.py:136`）。它转发到 `BackendModule.lower`
   （`tilelang/backend/module.py:112-115`）→ `PassPipeline.lower`
   （`tilelang/backend/pass_pipeline/pipeline.py:22`）→ 具体后端的 `pipeline.py`。
   其余四个阶段是共享的，因为：① 的语言表面由 `tilelang/language/common` 提供、
   后端只做扩展；③④ 通过 `BackendContext` 里已解析好的 `DeviceCodegen` /
   `HostCodegen` 间接调用，engine 不需要知道 target 是谁；⑤ 走的是统一的
   `adapter` 接口。
3. **不是。** 反例：layout 相关 IR。CUDA 侧走 `LayoutInference`（fragment / swizzle 语义），
   昇腾侧走 `AscendLayoutInference` + `NormalizeAscendFractalStorage` + `InsertNd2Nz`
   （fractal / NZ 语义）。两者都源自同一个 `PrimFunc`，但进到 layout 之后的抽象不同
   （见本章 5.2 节第 5 条）。

**常见错误**：把“前端共享”理解成“整条编译链共享”；或者把 `context.lower(mod)` 的位置
写成 `PassPipeline.lower` 所在的文件，请注意 `tilelang/engine/lower.py:136` 才是**调用点**。

---

## E2 概念题 ★

**解题思路**：抓住“生成什么”与“怎么编译启动”这条分界。

**参考答案**：

1. target backend 拥有：语言方言、target 探测与归一化、pass pipeline、host/device codegen、
   目标工具链回调。execution backend 拥有：缓存编排、wrapper 构造、产物加载、
   参数与 stream 绑定、启动、运行时生命周期。定义见
   `tilelang/backend/README.md:61-78`。
2. target backend 例子：`cuda`（`tilelang/cuda/backend.py:123-143`）、
   `ascend`（`tilelang/ascend/backend.py:52-63`）。
   execution backend 例子：`tvm_ffi` 与 `nvrtc`（`tilelang/cuda/execution_backend.py:24-33`）、
   `cython`（`tilelang/ascend/execution_backend.py:20`）。
3. 第一句**错**。`nvrtc` 只是一种执行路径；pipeline 由 target backend 决定，换执行后端
   不会换 pipeline。`tilelang/backend/README.md:76-78` 原话：
   “selecting `nvrtc` does not select a different target architecture or pass pipeline”。
   第二句**部分错**。CuTeDSL 与 CUDA 确实是两个 manifest，但 CuTeDSL **显式复用**
   `pipeline.CUDA_PIPELINE`（`tilelang/cuda/cutedsl_backend.py:13`），
   它们的 IR lowering 是**相同**的，区别在 device codegen 与 execution backend。

**常见错误**：把 `target_kinds=("cuda",)` 出现两次当成 bug。它是设计：
靠 `supports_target` 谓词区分（`tilelang/backend/module.py:247-251`、
`tilelang/cuda/codegen.py:8-13`）。

---

## E3 概念题 ★★

**解题思路**：把 `__post_init__` 当成“manifest 的 schema 校验”来读。

**参考答案**：

1. 至少四条：
   * `pipelines` 的 key 集合必须等于 `target_kinds` 集合，且每个 pipeline 的 `name`
     必须等于 target kind（`tilelang/backend/module.py:52-57`）。防“声明了 target 却没有 pipeline”
     或“pipeline 挂错 target”。
   * `device_codegens` 的 key 集合必须等于 `target_kinds`（`:59-61`）。防“target 没有
     codegen，编译到一半才发现”。
   * `execution_backends` 不能为空、名字不能重复；若有 `enable_host_codegen` 就必须有
     `host_codegens`（`:73-80`）。防“执行策略要求 host 代码，但后端不会生成”。
   * `host_codegen_hooks` 的 key 必须是本后端拥有的 target kind（`:63-71`）。
     防 hook 挂到别人的 target 上。
2. 因为 `_resolve_backend_module`（`:300-307`）在“多个候选都能匹配”时会直接抛
   `ValueError(f"Multiple backends match target {target}: ...")`。若允许两个 manifest
   同时占一个 kind 且都没有谓词，那么每次编译都会失败；要求提供谓词等于强制后端作者
   把“我到底管哪一个变体”写清楚。
3. `register_backend` 会先写入 `_BACKENDS` 与 `_TARGET_KIND_INDEX`，再注册 FFI callback。
   若 callback 注册失败却不回滚，注册表里就会出现一个“看起来已注册、实际不完整”的后端：
   后续 `create_backend_context` 可能选中它，然后在 codegen 阶段抛出难以定位的错误。
   回滚保证“注册成功”是一个原子事实。

**常见错误**：以为这些校验只是“防御式编程”。它们其实定义了后端作者必须回答的问题清单，
是第 31 章“加后端检查清单”的雏形。

---

## E4 代码阅读题 ★★

**解题思路**：盯住源码里的注释，它们直接写了“为什么这么排”。

**参考答案**：

1. 源码在 `tilelang/cuda/pipeline.py:127-128` 写了：
   “Run pipeline planning and software-pipeline rewriting before layout inference so inferred
   layouts see the final pipelined structure directly.”
   即：流水改写会引入 buffer 版本（多缓冲），layout 推导必须看到**最终的**缓冲版本，
   否则会为“旧的、单一版本”的 buffer 推出 layout，之后又得改。
   若交换顺序，`LayoutInference` 看到的是尚未注入软件流水、也没有多缓冲的 IR，
   推出来的 fragment/shared layout 与最终代码不匹配（或在后续被强制改写）。
2. 因为 `LayoutVisual` 只做可视化：它在 `tilelang/backend/pass_pipeline/pipeline_utils.py:105-109` 里判断开关
   （`TL_LAYOUT_VISUALIZATION_ENABLE`），开启时调用
   `tilelang.analysis.LayoutVisual(formats=...)`，不返回新的 `mod`。
   把它写成 `mod = LayoutVisual(mod)` 会得到 `None`，反而破坏 pipeline。
3. 不再存在。`LowerTileOp` 把 tile 级 op 展开成具体的 TIR 语句（MMA 序列、搬运序列）。
   所以调试 tile 语义时，**这一步前后对比 IR 最有效**：之前能看到 `T.gemm` / `T.copy`，
   之后只剩语句。

**常见错误**：把 `LayoutVisual` 当成 pass 写在表格里。它确实在 pipeline 里出现，
但它不是 pass，是调试插桩点，因此本章的表格没有把它计入“73 步”。

---

## E5 代码阅读题 ★★

**解题思路**：把 13 步按“准备 → 建模 → 求解 → 落地”切段。

**参考答案**：

1. 四类工作的对应关系：
   * **任务化**：`NormalizeControlFlowForSchedule`、`NormalizeConflictHints`、
     `MaterializeScheduleUnits`（把计算/搬运拆成带资源标签的调度单元）。
   * **资源建模**：`AnnotateMultiBufferEligible`（哪些 buffer 可以多份）、
     `EstimateLatency`（每个单元的延迟估计）。
   * **求解**：`AutoSchedule`（Z3/SMT，一次性解出任务发射时刻、跨迭代重叠与 buffer 份数）。
   * **落地**：`AssignCore` / `ResolveCore`（分派 AIC/AIV 与具体核）、
     `PrepareMultiBuffer` → `InsertSync` → `MaterializeMultiBuffer`
     （把解变成真实的同步指令与 buffer 版本）、
     `LowerScheduledTIR` → `RestoreWhileLoops`（把带调度标记的 TIR 还原成可继续 lower 的形式）。
2. `EstimateLatency` 提供**代价模型**，`AnnotateMultiBufferEligible` 提供**可行域**
   （哪些 buffer 允许多份）。没有这两个输入，求解器既不知道“快不快”，
   也不知道“能选哪些”，只能瞎猜。这是约束求解类调度器的通用结构。
3. 因为自动调度器在求解过程中**自己**会推导 buffer 别名；关掉调度器之后，
   这一步就没有人做了，必须由这个 pass 补上。它说明“关掉自动调度”不是“少做一点优化”，
   而是“你要自己承担调度器原本承担的那部分建模工作”。

**常见错误**：把 `AutoSchedule` 当成唯一重要的 pass。真正的工程价值分布在
“建模是否准确”（前 4 步）与“落地是否正确”（后 7 步）两端；求解器本身只是中间一步。

---

## E6 代码阅读题 ★★

**解题思路**：从“命名空间组成”与“静态分析”两个角度理解。

**参考答案**：

1. 第一行把 CUDA 方言的全部公共符号并入 `tilelang.language`；
   第二行把 CUDA 的 `__all__` 也让渡给 `tilelang.language`，这样 `from tilelang.language import *`
   的行为由 CUDA 方言定义，而不是由这里重新发明一份；
   第三段按名字显式导入（`Kernel`、`Parallel`、`copy`、`gemm` …），
   目的是让静态类型检查器能解析到 CUDA 方言上带类型签名的符号
   （注释原话：static type checkers cannot evaluate the dynamic `__all__`）。
   `__tilelang_dialect__ = "cuda"` 则是给工具/测试的一个显式标记。
2. 昇腾 kernel 里这么写会拿到 **CUDA 方言**的 `T.Kernel`：
   它接受 `threads` / `cluster_dims` 等 CUDA 注解，而不是昇腾的
   `cthread` / `T.SimtVF` 语义。正确写法是
   `from tilelang.ascend import language as T`（或同时导入公共表面
   `tilelang.language.common` 再叠加昇腾扩展）。
3. 关系是“**公共 launch 语义 + 后端注解**”：
   `tilelang/language/kernel.py:431` 的 `Kernel` 是 target-neutral 的
   “a grid of tile programs”，docstring 明确说它“is the target-neutral launch:
   the part every backend shares”；`tilelang/cuda/language/kernel.py:13` 的 `Kernel`
   在这个基础上加了 `threads` / `prelude` / `cluster_dims`。
   `ClusterKernel` 依赖 `cluster_dims`，而 `cluster_dims` 是 SM90+ 的 CUDA 概念，
   所以它只能放在 CUDA 一侧。这也解释了昇腾 pipeline 为什么要把
   `cluster_dims` 列进 `unsupported_annotations`。

**常见错误**：以为 `tilelang.language` 是“中立的公共层”。它在 v0.1.15 里是
**CUDA 方言 + 公共表面**，中立的部分在 `tilelang/language/common.py`。

---

## E7 动手题 ★

**解题思路**：直接看脚本输出，必要时用 `--format csv` 便于筛。

**参考答案**（v0.1.15 上的实测值）：

1. CUDA 共 **73** 步；其中 `CUDAPassPipelineBodyPrologue`（`tilelang/cuda/pipeline.py:68-165`）
   贡献 **33** 步，`Body` 中 prologue 调用之后贡献 40 步。
2. Ascend 共 **80** 步，全部来自 `AscendPassPipelineBody`。
3. 两侧同名 pass 共 **39** 个。常见的有：`MaterializeKernelLaunch`、`LetInline`、
   `AddWrapperForSingleBufStore`、`LegalizeNegativeIndex`、`InjectAssumes`、`Simplify`、
   `CanonicalizeLegacyReducer`、`VerifyReducerEpoch`、`VerifyBufferInit`、
   `ReducerPlanAndMaterialize`、`DecoupleTypeCast`、`LegalizeVectorizedLoop`、
   `LegalizeSafeMemoryAccess`、`LowerAccessPtr`、`HoistGlobalBufferAllocations`、
   `LowerOpaqueBlock`、`FlattenBuffer`、`ConfigIndexBitwidth`、`VectorizeLoop`、
   `LoopUnswitching`、`UnrollLoop`、`VerifyMemory`、`AnnotateEntryFunc`、
   `InferFragment`、`LowerThreadAllreduce`、`AnnotateDeviceRegions`、`SplitHostDevice`、
   `AnnotateReadOnlyParams`、`MergeIfStmt`、`MakePackedAPI`、`LowerDeviceKernelLaunch`。
4. 带条件守卫：CUDA **7** 个，Ascend **17** 个。CUDA 的 7 个是：

| 行号 | pass | 条件 |
| --- | --- | --- |
| 77 | `tilelang.transform.LetInline` | `should_force_let_inline()` |
| 84 | `tilelang.transform.VerifyParallelLoop` | `should_enable_race_check()` |
| 107 | `tilelang.cuda.transform.AutoWarpSpecialization` | `allow_warp_specialized(target=target)` |
| 108 | `tilelang.cuda.transform.MaterializeWSSchedule` | `allow_warp_specialized(target=target)` |
| 109 | `tilelang.cuda.transform.ProducerConsumerWarpSpecialized` | `allow_warp_specialized(target=target)` |
| 201 | `tilelang.cuda.transform.FuseMBarrierArriveExpectTx` | `has_tma` |
| 271 | `tilelang.cuda.transform.AnnotateWarpGroupRegAlloc` | `allow_warp_specialized(pass_ctx=pass_ctx, target=target)` |

**常见错误**：把 73 步数成 74。多出来的那一步是
`mod = CUDAPassPipelineBodyPrologue(mod, target)` 本身，脚本刻意把它展开成
Prologue 的 33 步而不重复计数。

---

## E8 动手题 ★★

**解题思路**：把脚本当测量工具，而不是当结论。

**参考答案**：

1. 六个数字：`tilelang_top_level_dirs = 24`、`tilelang_python_files = 390`、
   `src_transform_cc_files = 42`、`examples_subdirs = 54`、
   `examples_top_level_files = 3`、`docs_markdown_files = 37`。
2. `VERSIONS.md` 数的是 `examples/` 下的**条目总数**（54 个目录 + 3 个文件 = 57），
   本章脚本把它拆成“目录数”与“顶层文件数”两项。本章采用脚本口径，理由是：
   `quickstart.py`、`conftest.py`、`pytest.ini` 不是示例目录，
   把它们计入“示例数量”会误导读者去找 `examples/quickstart` 这个不存在的目录。
3. 正常情况下输出 `compile-path hops: 30/30 verified`，退出码 0。
   若某跳失败，先用 `sed -n '<行号>p' <文件>` 核对；常见原因是 `--vendor` 指向了
   main 而不是 v0.1.15。改一行让脚本打印数量：

   ```python
   print(f"HOP count = {len(HOPS)}")
   ```

   加在 `check()` 调用之前即可。

**常见错误**：把“脚本通过”当成“章节结论被验证”。脚本只验证**位置与符号存在**，
不验证语义；语义要靠第 20–22 章的逐 pass 精读与真实运行。

---

## E9 动手题 ★★（GPU，可选）

**解题思路**：这一题的核心不是“跑通”，而是**把生成物与 pass 表格对上**。

**参考答案**（在真实 GPU 机器上应当看到的现象）：

1. 脚本以退出码 0 结束，并打印 `result matches torch.relu(a @ b)`；
   `/tmp/tl-debug/` 下出现 `tilelang_jit_kernel_*.c` 与 `tilelang_jit_program_*.py`。
2. `__global__` 对应 `LowerDeviceKernelLaunch` + `PersistThreadblock`
   （CUDA 表格第 72、73 步）；`__shared__`（或 `extern __shared__`）对应
   `MergeSharedMemoryAllocations`（第 63 步）与 `LowerSharedTmem`（第 34 步）；
   `cp.async` / TMA 相关代码对应 `LowerTileOp`（第 25 步）+ `InjectFenceProxy`（第 64 步）
   + `FuseMBarrierArriveExpectTx`（第 37 步，仅在 `has_tma` 时）。
3. `tilelang_jit_program_*.py` 是 `prim_func.script()`，也就是**前端产物的文本形式**，
   所以 `T.gemm` / `T.copy` / `T.Pipelined` **仍然在**。这说明“tile op 被展开”
   发生在 pipeline 中，而不是发生在前端；要观察展开后的形态，必须看 pipeline 中途的 IR
   （用 `lower_trace` / `TL_ENABLE_DUMP_IR`），源码文件本身看不到。
4. `block_K` 从 32 到 64：`B_shared`/`A_shared` 的字节数翻倍，`MergeSharedMemoryAllocations`
   的可用空间相应紧张，`num_stages=3` 下总 shared 需求也随之翻倍；
   如果超出设备上限，编译期就会报 shared memory 相关错误，或触发寄存器/占用率变化。

**没有 GPU 时的答案**：静态部分可以完成，用
`grep -rn "tile_op\|TileOp" src/op/` 观察 `src/op/` 里 `copy.cc`、`gemm.cc` 等
确实以 tile op 为单位实现；`LowerTileOp`（`src/transform/lower_tile_op.cc`）负责把它们展开。
**必须显式声明**：`__shared__` 大小、TMA 代码形态、正确性结果都**未验证**。

**常见错误**：在没有 GPU 的机器上把“脚本能 py_compile”写成“运行通过”。
本章 `code/README.md` 与此处都要求写明验证级别。

---

## E10 概念题 ★★

**解题思路**：从 IR 结构倒推“用户能写什么”。

**参考答案**：

1. CUDA：`lower_thread_binding=True` 会把 `threadIdx.x` 物化出来，kernel 作用域里同时
   存在 `blockIdx`（网格）与 `threadIdx`（线程），`default_threads=128` 提供默认线程数。
   昇腾：`lower_grid_binding=True` 把网格循环变成 `thread_extent` 形式的 AttrStmt，
   `launch_dim_tags=["cthread"]` 引入 `cthread`（sub-block-id 维），
   但**kernel 作用域没有 `threadIdx`**，线程域只在 `T.SimtVF` 里出现，
   由 `T.SimtVF` 自己发出线程作用域（见 `tilelang/ascend/pipeline.py:38-44` 的注释）。
2. 直接影响：CUDA 里可以用 `T.get_thread_binding()` 写 thread 级代码
   （expert 层）；昇腾里在 kernel 作用域写线程级代码没有意义，
   必须先用 `T.SimtVF`（或 `T.SimdVF`）声明一个向量计算区域，线程域才存在。
   这就是“线程域只存在于向量区域内”的含义。
3. `unsupported_annotations=["cluster_dims"]` 表示昇腾的 launch 物化不认识
   `cluster_dims`（那是 SM90+ 的 CUDA 概念）。用户若对昇腾目标写
   `T.Kernel(..., cluster_dims=2)`，该注解不会被物化，
   应当被视为不支持（具体报错形式需要真实环境验证，本章不作断言）。

**常见错误**：以为 `lower_thread_binding=False` 意味着“昇腾没有并行”。
恰恰相反：并行从“每线程”上移到“每核 + 向量区域”，粒度更粗，但并行度仍在。

---

## E11 概念题 ★★

**解题思路**：一切以 checkout 里的实际文件为准，不靠 PR 标题与发布日期。

**参考答案**：

1. 核验命令（任选三条即可）：

   ```bash
   ls _vendor/tilelang-v0.1.15/tilelang/jit/adapter/
   grep -rn "PersistentKernel" _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
   grep -rn "AutoSimtVF" _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
   ls ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/jit/adapter/
   grep -rn "PersistentKernel" ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang
   ```

2. 结论：
   * `T.PersistentKernel` / `AutoPersistent`：在 v0.1.15 **没有**，在 corpus 的 main 快照
     （`994b44e`）里 grep 也**没有**，属于未合入（PR #3325）。
   * PTO 执行路径：v0.1.15 **没有** `tilelang/jit/adapter/pto/`；
     corpus 的 main 快照**有**（它的 HEAD 正是 PR #3310“Add PTO backend for Ascend950”）。
   * 顺带：`AutoSimtVF`（PR #3320）在两个 checkout 里都没有。
3. 因为 `main` 每天都在动。如果没有版本纪律，“TileLang 支持 X”这句话就无法复核：
   同一个论断在不同日期会得到不同答案。把前沿内容集中到第 30 章，可以让全书其余 30 章
   的结论保持“对 v0.1.15 恒真”，读者也能一眼分清“现在能用”与“将来会有”。
4. 这两条都属于 5.3 节表格里“仅 main / 未合入”的行。核验路径就是第 1 问的命令，
   加上记录 checkout 的 `git log -1 --oneline` 作为时间戳。

**常见错误**：用 PR 编号或发布日期下结论。例如 PR #3308（引入 Ascend 950 后端）
与 #3309（bump 到 0.1.15）几乎同时，但只有 #3309 之内包含了 v0.1.15 的一致性。
**看文件，不看标题。**

---

## E12 动手题 ★★★

**解题思路**：照抄 `tilelang/backend/README.md` 的 Backend Package Layout，
再用 `tilelang/ascend/` 当现成范本。

**参考答案**：

1. 文件清单（参照 `tilelang/backend/README.md:325-341`）：

   ```text
   tilelang/foo/
     __init__.py            # import 时触发注册，顺序要确定
     backend.py             # BackendModule manifest + 工具链回调
     target.py              # target 解析/归一化（注册 detector 与 normalizer）
     execution_backend.py   # 兼容的 execution backend 及 auto 优先级
     language/              # foo 方言（在 common 之上扩展）
     pipeline.py            # 完整、有序的 pass 序列
     codegen.py             # host/device codegen 入口
     transform/             # 仅 foo 使用的 Python pass
     op/                    # tile op 的 foo 实现与注册
     intrinsics/            # foo 的 intrinsic 发射器
   src/foo/
     CMakeLists.txt
     codegen/               # native codegen + runtime module
     op/                    # native tile-op lowering
     transform/             # native foo-only passes
   ```

2. 必填字段：`name`、`target_kinds`、`pipelines`、`device_codegens`、
   `execution_backends`。可选：`supports_target`、`host_codegens`、
   `host_codegen_hooks`、`callbacks`。最小草稿：

   ```python
   from tilelang.backend.device_codegen import DeviceCodegen
   from tilelang.backend.execution_backend import ExecutionBackendSpec
   from tilelang.backend.module import BackendModule, register_backend
   from . import codegen, pipeline

   BACKEND = register_backend(
       BackendModule(
           name="foo",
           target_kinds=("foo",),
           pipelines={"foo": pipeline.FOO_PIPELINE},
           device_codegens={
               "foo": DeviceCodegen(
                   "foo",
                   build=codegen.build_foo,
                   build_without_compile=codegen.build_foo_without_compile,
               )
           },
           execution_backends=(ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),),
           host_codegens=STANDARD_HOST_CODEGENS,
           callbacks={"tilelang_callback_foo_compile": tilelang_callback_foo_compile},
       )
   )
   ```

   注意 `BackendModule` 会在 `__post_init__` 里检查
   “`pipelines` / `device_codegens` 的 key 集合 == `target_kinds`”，
   所以上面三个 `"foo"` 必须完全一致。
3. 复用 CUDA pipeline 的写法是直接引用同一个 `PassPipeline` 实例：
   `pipelines={"foo": tilelang.cuda.pipeline.CUDA_PIPELINE}`。
   但注意 `BackendModule.__post_init__` 要求 `pipeline.name == target_kind`
   （`tilelang/backend/module.py:55-57`），而 `CUDA_PIPELINE.name == "cuda"`，
   因此**只有 target kind 也是 `cuda` 时才可行**。v0.1.15 的先例正是 CuTeDSL：
   `tilelang/cuda/cutedsl_backend.py:11-13`，
   `target_kinds=("cuda",)` + `pipelines={"cuda": pipeline.CUDA_PIPELINE}`。
   换句话说，“复用别人的 pipeline”在 v0.1.15 里等于“共用同一个 target kind”；
   如果你的 kind 是 `foo`，就必须自己维护 `pipeline.py`。
4. 最容易违规的一条是“**Do not spread execution-mode checks through compiler passes**”
   与“**Keep host/device codegen dispatch … under target-backend ownership**”
   （`tilelang/backend/README.md:382-389`）。具体表现：为了省事，在
   `tilelang/engine/lower.py` 里加 `if context.module.name == "foo": ...`。
   正确做法是把差异放进 `tilelang/foo/pipeline.py` 与 `tilelang/foo/codegen.py`，
   让 engine 继续只有那一行 `context.lower(mod)`。

**常见错误**：忘记注册 target detector/normalizer，导致 `target="auto"` 永远选不到 `foo`；
或者把 `foo` 的 pass 直接塞进 `tilelang/transform/`，让其它后端也被迫看到它们。
