# 第 24 章 习题答案

与 `exercises.md` 一一对应。每题给出**解题思路**、**参考答案**与**常见错误**。
所有源码引用都指向 `_vendor/tilelang-v0.1.15/`。本章写作环境没有 NVIDIA GPU，
因此凡是涉及运行结果的题目，答案里都写明了「需要读者在有卡机器上自行确认」。

---

## E1 概念题 ★

**解题思路**：从 `lower()` 的默认参数入手，记住「device 只产源码、host 由 adapter 生成」
这两条默认行为。

**参考答案**：

1. 由 C++ 的 `CodeGenTileLangCUDA` 生成，文件是
   `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc`；入口函数是
   `BuildTileLangCUDA`（`src/cuda/codegen/rt_mod_cuda.cc:99-139`），每个函数主体由
   `AddFunction`（`_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc:6509` 起）打印。
2. 默认**不会**。`tilelang/engine/lower.py:144-170` 的 `lower_with_context` docstring 写明
   `enable_device_compile` 默认 False，JIT adapter 自己接管编译；默认走
   `device_codegen_without_compile`，CUDA C 源码被放在 module 的 `source_map["cuda"]`
   （`src/cuda/codegen/rt_mod_cuda.cc:151-176`）。
3. 由 `TLCUDASourceWrapper.update_lib_code`（`tilelang/jit/adapter/wrapper.py:585-632`）拼接：
   `self.source + host_adapter_include + init_func + host_func`，其中 `init_func` /
   `host_func` 分别来自 `PREDEF_INIT_FUNC`（第 68-79 行）与 `PREDEF_HOST_FUNC`
   （第 81-85 行）。cubin 的生成发生在 `tilelang_callback_cuda_compile`
   （`tilelang/cuda/backend.py:59-121`）或运行期 NVRTC。

**常见错误**：以为 `get_kernel_source()` 返回的是 PTX。它返回的是 CUDA C 源码；
PTX 要用 `tilelang/contrib/nvcc.py:271-296` 的 `get_ptx_from_source`。

---

## E2 概念题 ★

**解题思路**：读注册表 `tilelang/cuda/execution_backend.py:24-31`，再看每个 spec 的
`is_available`。

**参考答案**：

1. 依次是：`tvm_ffi`（默认，`enable_host_codegen=True` + `enable_device_compile=True`，
   无条件可用）、`nvrtc`（可用性取决于 `cuda-python` 是否可导入，见
   `_is_nvrtc_available`）、`cython`（无条件列出，实际使用需要 `tilelang_cython_wrapper`
   模块）。按 `tilelang/jit/__init__.py:97` 的签名，用户可显式传
   `execution_backend="nvrtc" | "cython" | "tvm_ffi" | "torch" | "cutedsl"`。
2. 因为三种 backend 共用同一个 `TLCUDASourceWrapper`；差别只在「生成的 C 源码怎么装载、
   `call()` 怎么被调」，device 侧源码由同一条 `CUDA_PIPELINE` + 同一个
   `CodeGenTileLangCUDA` 产出。
3. `nvrtc`。它的模块 docstring 写明动机是「不需要离线编译步骤，只依赖驱动」
   （`tilelang/jit/adapter/nvrtc/libgen.py:1-20`）；`cython` 需要 nvcc 把 C 源码编成 `.so`
   （`tilelang/jit/adapter/libgen.py:41-80`）。

**常见错误**：把 `tvm_ffi` 当成「TVM 运行 kernel」——它在 v0.1.15 里是 TileLang 自己
通过 TVM-FFI 生成的 host module
（`tilelang/cuda/execution_backend.py:24-29`，注意 `supports_callee_allocated_outputs=True`）。

---

## E3 代码阅读题 ★★

**解题思路**：注意「显式 API」与「自动选择」两类语义的差别。

**参考答案**：

1. 因为 `T.wgmma_gemm()` 是**用户点名**的接口：用户说的是「我确知要用 WGMMA」。
   如果条件不满足，静默换成 `mma.sync` 会让用户拿到一个与预期不同的实现（寄存器压力、
   layout、同步点全不一样），所以选择显式失败。错误信息由 `FatalWgmmaUnavailable`
   （`src/cuda/op/gemm.cc:157-176`）给出，并且带 `SpanHintSuffix` 指回用户源码位置。
2. 因为 2CTA 的形状约定本身就建立在「B 的 N 范围有一半在邻居 CTA」之上；回退到单 CTA
   指令会**静默越界**。理由写在 `src/cuda/op/gemm.cc:347-353` 的注释里。
3. 会继续往下走到 `AllowTcgen5Mma`（在 sm_70 上为假）与 `AllowWgmma`（sm_90 才真）之后，
   最后落到函数末尾的 `return kCudaMMA`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:375`）——因为 `TargetIsVolta(target)`
   为真但 `!AllowVoltaMma(op)` 为假，所以两个 FMA 分支都被跳过。

**常见错误**：把 `LOG(FATAL)` 当成「编译器的 bug」。它是**有意的**契约违背报错。

---

## E4 概念题 ★★

**解题思路**：定位到 `SelectCopyInstForLowering` 的后半段。

**参考答案**：

1. `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:797-803`：

```cpp
if (!facts.disable_tma && !facts.pass_context_disables_tma) {
    CopyInst inst = SelectTmaInst(facts, /*allow_load=*/false, /*allow_store=*/true,
                                  /*check_last_dim=*/true);
    if (inst != CopyInst::kInvalid) return Supported(inst);
}
```

   `allow_load=false` 就是答案。
2. 三种：(a) 显式 `T.tma_copy(src, dst, barrier=...)` → `facts.explicit_tma` 分支
   （`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:748-756`）；(b) `T.copy(..., prefer_instruction="tma")` →
   `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:762-780`；(c) 处在 warp-specialized 的 producer 分支 →
   `ClassifyWarpSpecializedCopy`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:810-891`）对 load 传
   `allow_load=True`。
3. 降成同步 SIMT 拷贝（`CopyInst::kNormal`）。因为兜底调用的是
   `SelectSyncLikeInst`（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:612-626`），它只在 LDSM/STSM/TMem/Normal 里选。
   `cp.async` 需要 `no_implicit_async_commit_wait` 或 `is_async_copy` 标注，而前者是
   `InjectSoftwarePipeline` 给流水里的 copy 打的
   （`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:288-299`）。

**常见错误**：以为 `T.copy` 的兜底一定是 `cp.async`。在 v0.1.15 里它只在流水里是。

---

## E5 代码阅读题 ★★

**解题思路**：逐条对照 `CheckBulkLoad` / `CheckBulkStore` / `CheckGlobalStrides`。

**参考答案**：

1. ✅ 可以。arch≥90、scope 正确、最后一维 64×16=1024 bits 是 128 的倍数、
   row-major stride 合法。
2. ❌ 不能。最后一维 30×16=480 bits，不是 128 的倍数
   （`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:205-215` 的 `check_last_dim` 分支）。
3. ❌ 不能。`CheckGlobalStrides` 要求最内层 stride 必须是 1
   （`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:155-165`）。
4. ❌ 不能。stride=12 元素的字节数 = 24，不能被 16 整除
   （`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:167-180`）。
5. ❌ 不能。`TargetHasBulkCopy` 要求 `arch >= 90`
   （`_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:117-122`）。

**常见错误**：只看 dtype 与 scope。真正最容易踩的是**最后一维的位数对齐**与
**stride 的字节对齐**。

---

## E6 代码阅读题 ★★

**解题思路**：`AllowWgmma` 是「架构 + 形状 + warp 数」，`CheckWgmma` 是「dtype + 转置 +
K 整除」。

**参考答案**：

1. `TargetIsHopper(target)`；`op.m_ >= 64`；`num_warps % 4 == 0`
   （`src/cuda/op/gemm.cc:89-97`）；此外还要 `CheckWgmma(op)` 为真，并且
   `kDisableWGMMA` 这个 pass config 没被设上。
2. `trans_A=False, trans_B=True` 才能被接受。fp8 的 WGMMA 只支持 k-major 操作数
   （`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:48-52`：`(!op.transA_) && op.transB_ && op.k_ % 32 == 0`）。
3. 没有要求。f16/bf16 那条分支只要求 `k % 16 == 0`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:43-46`、`58-63`）。
   原因是硬件对 fp16/bf16 的 operand 主序没有限制，而 fp8/tf32 的 WGMMA 有硬性 layout
   约束——这一点在 FlashAttention-3 论文 §2.2 有明确表述
   （`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608-fulltext.md:186-190`）。

**常见错误**：漏掉 `CheckWgmma` 里「B 必须在 shared」这个前置条件
（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:41-43`）。

---

## E7 概念题 ★

**解题思路**：policy 影响的是 `m_warp × n_warp` 的划分，不是指令本身。

**参考答案**：

1. `Gemm::ComputeWarpPartition`（`src/cuda/op/gemm.cc:377-397`）把它转发给
   `ComputeWgmmaWarpPartition`（:218-300）或 `ComputeDefaultWarpPartition`（:180-216）。
   这两个函数都要读 `policy.IsFullRow() / IsFullCol() / IsSquare()`。
2. `ComputeWarpPartition` 的前两个分支直接返回固定划分：TCGEN05 走 `m_warp=1,
   n_warp=num_warps`，FMA 同样走 `1 × num_warps`（`_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:380-390`）。原因是
   **这两类指令都不是「按 warp 切 M/N」的模型**：TCGEN05 是单线程发射、累加器在 TMEM；
   FMA 是纯标量循环，划分方式对结果没有意义。

**常见错误**：以为 policy 会影响数值结果。它只影响每个 warp 负责哪一块输出、
以及随之而来的寄存器/访存模式。

---

## E8 动手题 ★★

**解题思路**：改一个变量、观察一个结果，必须能解释「为什么」。

**参考答案**（示例，不是唯一解）：

1. 运行后应看到 `self-check: ok`；13 个场景里 store 默认走 `BulkStore`、
   裸 load 走 `Normal`、流水里的 load 走 `CPAsync`。
2. 把 `tmem->fragment f32 sm_100` 改成 sm_90：结果从 `TMemLoad` 变成 `Normal`。
   因为 `target_has_tmem(arch)` 只在 `100 <= arch <= 110` 时为真
   （对应 `_vendor/tilelang-v0.1.15/src/cuda/target_utils.cc:111-116` 的 `TargetHasTmem`
   → `TargetIsSm100`）。
3. 例如把 `g->s 2D tile, cluster_mask` 的 `last_dim_extent` 从 64 改成 30：
   `cluster_mask != 0` 分支只允许 `kBulkLoad`，而 `CheckBulkLoad` 的 `check_last_dim`
   拒绝了 30×16=480 bits。脚本会抛 `SelectionError`
   （对应 `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:737-747` 的 `Unsupported(...)`）。

**常见错误**：只报告「结果变了」而不追到具体谓词。这一题的价值全在「为什么」。

---

## E9 动手题 ★★

**解题思路**：TCGEN05 的关键是 `C` 在 TMEM；WGMMA 的关键是 sm_90 + M≥64 + warp 数是
4 的倍数。

**参考答案**（示例）：

1. `M=128, N=128, K=64, f16→f32, A=shared.dyn, B=shared.dyn`：
   * `C=shared.tmem` 且 target=sm_100a → `cuda.tcgen05`；
   * `C=local.fragment` 且 target=sm_90a（`block_size=256`，`trans_B=True`）→ `cuda.wgmma`。
2. 同上但在 sm_100a 上把 `C` 放回 `local.fragment` → 仍是 `cuda.mma`。
   挡住它的是 `AllowTcgen5Mma` 里的 `IsTmemBuffer(op.c_)`（`src/cuda/op/gemm.cc:80-87`）。
3. 可以（只要 `M % 32 == 0`）。`GetTCGEN5MMAMeta` 会返回 `atom_m = 32`，
   `enable_ws = True`（`_vendor/tilelang-v0.1.15/src/op/tcgen5_meta.h:70-80` 的
   `M % 32 == 0` 分支；`ws_valid_atom_ns = {256, 128, 64}`）。这与 PTX ISA 的
   `.ws` 形态 `32xNxK, N ∈ {64,128,256}, K=16` 对应
   （`../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:29804-29808`）。

**常见错误**：以为 sm_100 上「有 TMEM 就会用 tcgen05」。作用域不满足时不会。

---

## E10 动手题 ★★

**解题思路**：描述符的四个数组分别是 globalDim / globalStride / boxDim / elementStrides。

**参考答案**：

1. 运行脚本可知：`globalDim = [8192, 8192]`，`globalStride = [16384]`（字节），
   `boxDim = [256, 64]`，`elementStrides = [1, 1]`，`dtype = BFLOAT16`，
   `rank = 2`。
2. 因为模板传给 `cuTensorMapEncodeTiled` 的是 `globalStride + 1`
   （`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131`），即丢掉最内层
   那一项；最内层 stride 必须是 1 个元素，信息量为零。单位是**字节**，
   来自 `desc.global_stride.push_back(byte_stride)`
   （`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2076-2091`）。
3. box 最内层 = 8 时，8×16 = 128 bits，仍然能被 128 整除，所以**不会被拒绝**；
   但会命中 `TMA_MAX_BOX_DIM = 256` 之外的其它约束（这里是没超）。
   想构造拒绝，应该改成 7（7×16 = 112，不是 128 的倍数），脚本会报
   「不是 128 的倍数」。**注意**：box 只影响每次 TMA 搬运的元素数，
   上限是 256（`_vendor/tilelang-v0.1.15/src/cuda/op/tma_layout.h:40`）。

**常见错误**：把 `globalStride` 的单位当成元素。它是字节。

---

## E11 概念题 ★★

**解题思路**：`T.ws` 是 warp group 级（128 线程）的角色划分。

**参考答案**：

1. `T.ws(0)` → `tx < 128`；`T.ws(1)` → `128 <= tx < 256`
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/warpgroup.py:19-59`，
   注释里直接给了这两个展开）。
2. 因为 `arrive_count` 是**每轮需要到达的线程数**，而一个 warp group 是 128 个线程；
   producer 与 consumer 各自是独立的一轮（各自 128 线程到达），不是 256 线程一起。
   这也解释了为什么两个角色各有一个自己的 mbarrier。
3. 会漏掉反向压力：producer 可能在 consumer 还没读完上一轮数据时就覆写 shared buffer，
   这是数据竞争（读旧数据/写新数据的竞态）。用 `(ko + 1) % 2` 是因为 producer 在第 `ko`
   轮要等待的是「第 `ko+1` 轮的 buffer 空闲」，即 `compute_is_done` 在上一次翻转后的
   相位——相位与 consumer 的 `ko % 2` 相差一轮。
4. 由 `AutoWarpSpecialization` → `MaterializeWSSchedule` → `ProducerConsumerWarpSpecialized`
   三个 pass 完成（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:101-109`），
   寄存器分配由 `AnnotateWarpGroupRegAlloc` 收尾（:268-271）。前置门禁是
   `allow_warp_specialized`：必须是 CUDA target 且 `have_tma(target)` 为真
   （即 `arch >= 9`），且没有设 `tl.disable_warp_specialized`（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:27-38`）。

**常见错误**：把 `T.ws` 的编号当成线程号。它是 warp group 序号。

---

## E12 概念题 ★★

**解题思路**：TMEM 的规格与访问限制都能在 PTX ISA 文档与 `T.alloc_tmem` docstring 里找到。

**参考答案**：

1. `sm_100a`/`sm_100f` 上每个 CTA 是 **512 列 × 128 lane，每格 32 bit**；
   动态分配、以列为单位、分配单位是 32 列、列数必须是 2 的幂，且**所有 128 条 lane
   一起分配**；kernel 退出前必须显式释放
   （`../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:29726-29768`）。
   TileLang 侧的对应说明在 `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:254-300`。
2. 只有 TCGEN05.MMA 与特定的 TMEM load/store 能访问 TMEM。这意味着**前后处理必须发生
   在 TMEM 之外**：典型结构是 shared → tcgen05 → TMEM → (TMEM load) → 寄存器 → 后处理。
3. `C` 必须是 `T.alloc_tmem(...)` 出来的 buffer（scope 为 `shared.tmem`）。
   见 `AllowTcgen5Mma` 的 `IsTmemBuffer(op.c_)`（`src/cuda/op/gemm.cc:80-87`）。
4. SM100（`C` 在 TMEM）：TCGEN05 `kind::mxf8f6f4.block_scale`，`mbar` **必填**，
   编译器隐式插入 `mbarrier_wait_parity`；SM120（`C` 是 fragment）：
   `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`，**忽略 `mbar`**
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:103-135`）。

**常见错误**：以为 TMEM 是「更大的 shared memory」。它只对 TCGEN05 与 TMEM 存取指令
可见，不是一个通用 buffer。

---

## E13 概念题 ★★

**解题思路**：从 `cluster_dims` 这个 launch annotation 往下追。

**参考答案**：

1. 没有实质区别。`T.ClusterKernel` 就是 `T.Kernel` 加上必填的 `cluster_dims`
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:12-40` 的 docstring
   明确说了「T.ClusterKernel is the same launch with a required cluster_dims」）。
2. 因为可移植 cluster 上限是 8，B200 允许通过这个 attribute 显式开到 16；
   wrapper 生成的启动代码总是先设这个 attribute
   （`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:178-200`），
   厂商说明见
   `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md:73`。
3. TMA 快路径：给了 `remote_barrier` 且区域连续（最内维覆盖整个 buffer 宽度）→
   一条 `tl::tma_store_cluster`，arrive 数 1；Multi-TMA：给了 `remote_barrier` 但 ND 区域
   不连续 → 每行一条 `tma_store_cluster`，arrive 数 = 行数；SIMT 回退：没给
   `remote_barrier` 或区域无法分解 → `map_shared_rank` 逐元素存，若给了 `remote_barrier`
   则自动注入一次 arrive。三张表都在
   `_vendor/tilelang-v0.1.15/docs/programming_guides/cluster_tma.md` 里；
   实现见 `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:913-1040`。

**常见错误**：以为 `cluster_mask` 可以是运行期值。它必须是编译期常量。

---

## E14 概念题 ★★

**解题思路**：看 `cutedsl_backend.py` 复用了哪条 pipeline，再看 codegen 的注册点。

**参考答案**：

1. 复用 `CUDA_PIPELINE`。`_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:9-23`
   里 `pipelines={"cuda": pipeline.CUDA_PIPELINE}` 与普通 CUDA 后端
   （`tilelang/cuda/backend.py:128`）完全一致。
2. 生成的是 **CuTe DSL 的 Python 代码**（打印器 `CodeGenTileLangCuTeDSL`，
   `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cutedsl.cc:210` 起），
   由 `tilelang/jit/adapter/cutedsl/libgen.py` 写进临时目录并 import；
   真正把它变成 cubin 的是 NVIDIA 的 CUTLASS/CuTe（`tilelang/contrib/cutedsl/__init__.py`
   第一件事就是 `import cutlass` / `import cutlass.cute`）。
3. 不是同一件事。cuTile Python 走 **Tile IR**，需要驱动 r580+、`tileiras` 13.2，
   目前只支持 Blackwell 与 Ampere/Ada（Hopper 待支持）
   （`../tilelang-corpus/11-related-compilers/raw/github-nvidia-cutile-python.md:64-70`）；
   而 TileLang 的 CuTe DSL 后端把 CUTLASS/CuTe 当作 codegen target，
   和 cuTile 的编译器、IR、运行期都没有关系。

**常见错误**：把「CuTe DSL」与「cuTile」当成同一个东西——名字像，技术栈完全不同。

---

## E15 动手题 ★★★

**解题思路**：`diff -rq` 定位差异文件 → `diff -u` 看内容 → 逐条验证。

**参考答案**（至少三条）：

1. `_vendor/tilelang-v0.1.15/src/cuda/op/gemm_blockscaled.cc` 是**新增**文件
   （v0.1.14 没有）；对应地，v0.1.14 的 `src/cuda/op/gemm.cc` 里内联的 block-scaled
   分派与 `tl.get_tcgen5_blockscaled_instr_desc` 注册在 v0.1.15 里被删除。
2. `_vendor/tilelang-v0.1.15/src/cuda/transform/` 新增 `auto_schedule.cc`、
   `auto_schedule/`、`ws_analysis.{h,cc}`；`tilelang/cuda/pipeline.py` 在
   `MaterializeWSSchedule()` 之前插入了 `AutoWarpSpecialization()`，
   并新增 pass config `tl.enable_auto_warp_specialization`。
3. `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc` 里
   `CopyInstIsTMA` 被拆成 `CopyInstIsTMALoad` / `CopyInstIsTMAStore`，
   `ClassifyWarpSpecializedProducerCopy` 改名为 `ClassifyWarpSpecializedCopy`
   并新增「shared → global 直接走通用判定」的分支。
   （另外还有 `AllowTcgen5Mma` 里 `op.c_.scope() == "shared.tmem"` → `IsTmemBuffer(op.c_)`、
   `tilelang/cuda/pipeline.py:70` 的 `MaterializeKernelLaunch(lower_thread_binding=True,
   default_threads=128)`。）
4. `T.tcgen05_gemm` **不在**导出名单里：`tilelang/cuda/language/gemm_op.py:20-30` 的
   `__all__` 列了它，但 `tilelang/cuda/language/__init__.py:88` 只 import 了
   `gemm, gemm_blockscaled, gemm_sp`，而 `_CUDA_API_ALL`（同文件 108-196 行）也没有它；
   `tilelang/language/__init__.py:11` 用的是受 `__all__` 限制的
   `from tilelang.cuda.language import *`。所以「示例里能写」只是说明示例没有跟着
   facade 的导出名单更新——**这需要在有卡机器上实际 import 一次才能最终确认**，
   本章只陈述源码事实。

**常见错误**：用 `git log` 去查差异。两个 `_vendor/` 树都是
`git clone --depth 1 --branch <tag>`，没有提交历史，只能用文件级 `diff`。
