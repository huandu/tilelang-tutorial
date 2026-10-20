# 第 09 章 习题答案

说明：每题给出**解题思路**、**参考答案**与**常见错误**。动手题给思路与关键数据，
读者应在自己机器上复算。所有源码行号基于
[`../../VERSIONS.md`](../../VERSIONS.md) 锁定的 v0.1.15 / v0.1.2.000。

---

## E1 概念题 ★

**思路**：把「用户写什么」和「编译器决定什么」分开。

**参考答案**

1. 接口 = **一组承诺的语义 + 一组输入约束**（scope、dtype、形状、同步行为）。
   原语不承诺实现，只承诺「语句结束后 `dst` 可用」这类可观测行为。
2. 发射一条 tile op：`tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"),
   src, dst, annotations=ann)`（`tilelang/language/copy_op.py:130`）。
3. 当 `src` 和 `dst` 都是**没有 region extent 的 `BufferLoad`**，且没有任何注解时，
   直接生成 `BufferStore`，即 `dst[...] = src[...]`（同文件 `:121-128`）。

**常见错误**：以为前端会在此刻决定「用 TMA 还是 cp.async」。真正的选择发生在
`SelectCopyInstForLowering`（`src/cuda/op/copy_analysis.cc:716`）。

---

## E2 概念题 ★

**思路**：先按 scope 对筛掉不可能的形态，再按对齐/dtype 条件确认。

**参考答案**

| (src, dst) | 指令形态 | 判据 |
| --- | --- | --- |
| (global, shared)，最后一维位宽是 128 的倍数 | `cp.async.bulk.tensor`（TMA） | `CheckBulkLoad`：src=global、dst=shared/shared.dyn、`extent*bits % 128 == 0`、dtype 对、stride 合法（`copy_analysis.cc:195-225`） |
| (global, shared)，显式 `T.async_copy` | `cp.async` | `is_async_copy=true` 走 `facts.can_cp_async`，前置条件是 global→shared 且 dtype 相同（`copy_analysis.cc:427-437`） |
| (shared, fragment) | `ldmatrix` | `CheckLDSMCopy`：`TargetHasLdmatrix && IsSharedBuffer(src) && IsFragmentBuffer(dst)`（`copy_analysis.cc:407-410`） |
| (fragment, shared) | `stmatrix` | `CheckSTSMCopy`（同文件 `:412-415`） |
| (tmem, fragment) | `tcgen05.ld` | `CheckTMemLoad`：SM100 TMEM → fragment（同文件 `:417-420`） |

**常见错误**：把 (global, shared) 无条件当成 TMA。默认路径**不自动为 load 选 TMA**；
要显式 `prefer_instruction="tma"` 或 `T.tma_copy`（`copy_analysis.cc:762-777,796-807`）。

---

## E3 概念题 ★

**思路**：把 SIMT 路径、TMA 路径、昇腾 DMA 路径分三块回答。

**参考答案**

1. 读到的是 `safe_value`，**默认 0**：`PrimExpr safe_value = make_zero(src->dtype);`
   （`src/op/copy.cc:514`）。可用 `T.annotate_safe_value({A: v})` 覆盖
   （`tilelang/language/annotations.py:69-74`），它被挂到
   `CopyNode::src_oob_safe_value`（`src/op/copy.cc:94-116`）。
2. 目标越界时整条 store 被包进 `IfThenElse(dst_predicate, body)`——**不写**，
   而不是写一个替代值（`src/op/copy.cc:525`）。
3. TMA 路径没有可编程的「安全值」：越界由 descriptor 的 `oob_fill` 决定
   （`src/cuda/op/copy.h:49-60`），语义由指令定义。
4. 昇腾的 `pad_value` **不是越界填充**，是 DMA 行对齐填充：把行宽向上取整到下一个
   **32 字节**边界，用 `pad_value` 填 pad lane（`ascend/language/copy_op.py:278-287`）。
   它还会先发一条 `T.ascend_set_copy_pad_value` 语句，好让 AutoSchedule 插入
   `PIPE_S → PIPE_MTE2` 同步（同文件 `:332-351`）。

**常见错误**：把昇腾的 pad 与 CUDA 的 OOB 填 0 混为一谈；或以为 `pad_value` 会
改变越界访问的结果。

---

## E4 概念题 ★★

**思路**：按 `_gemm_dense_slots` 的推导式逐个推。

**参考答案**

设 `M, N = C.shape[-2:]`。

| 调用 | A 的形状 | B 的形状 |
| --- | --- | --- |
| `T.gemm(A, B, C)` | `[M, K]` | `[K, N]` |
| `T.gemm(A, B, C, transpose_B=True)` | `[M, K]` | `[N, K]` |
| `T.gemm(A, B, C, transpose_A=True, transpose_B=True)` | `[K, M]` | `[N, K]` |

推导式见 `tilelang/language/gemm_op.py:77-81`；校验在 `:82-88`。
`T.gemm(A, B, C, accum_dtype="float32")` 会报参数错误——v0.1.15 的 `gemm` 签名里
没有 `accum_dtype`（同文件 `:160-169`），累加类型由 C 的 dtype 决定。

**常见错误**：把 `transpose_B=True` 理解成「B 是 `[K, N]` 但会转置」。正确理解是
**B 在内存里就是 `[N, K]` 排布**，转置是逻辑上的读法。

---

## E5 概念题 ★★

**思路**：`clear_accum` 的一处细节要翻到 IR 组装函数里才能看到。

**参考答案**

1. 它把 C 的 access region 从 `rw` 改成 `w`：
   `C_arg = buffer_region_to_tile_region(C_region, "w" if clear_accum else "rw", ...)`
   （`tilelang/language/gemm_op.py:103`）。这会进入 `SetAccessRegions`，被依赖与同步
   分析使用（同仓库 `src/op/gemm.cc` 的 `GetAccessRegions`）。
2. `Square` / `FullRow` / `FullCol`（`tilelang/tileop/base.py:5-12`）。
   在 `ComputeWarpPartition` 里：tcgen05 直接 `(1, num_warps)`；wgmma 走
   `ComputeWgmmaWarpPartition(policy, M, N, num_warps)`；fma 是 `(1, num_warps)`；
   其余走 `ComputeDefaultWarpPartition(..., k_n_per_warp=8)`（Volta 是 16）
   （`src/cuda/op/gemm.cc:375-395`）。也就是说 **policy 只对 mma/wgmma 路径有实际差别**。
3. 因为 `cuda::Gemm::ReuseExistingSharedLayout` 只在 `cuda.mma` /
   `cuda.mma.blockscaled` 时返回 true（`src/cuda/op/gemm.cc:398-400`）；wgmma/tcgen05
   不复用已有 shared layout，会重新推导。

**常见错误**：以为 `policy` 是性能提示、不影响正确性——它确实不影响数值，
但会改变 fragment 的形状，所以任何手工写的 `T.Parallel` 访问模式都要跟着变。

---

## E6 概念题 ★★

**思路**：先答 `clear` 的语义，再答实现。

**参考答案**

1. `clear=True`：`out = reduce(src)`；`clear=False`：`out = out ⊕ reduce(src)`
   （sum 是加、max 取大、min 取小）。
2. 前端按 src/out 的 scope 组合分四路（`tilelang/language/reduce_op.py:103-171`）。
   以 fragment/src + shared/out 为例，`clear=False` 时会先把 `out` 拷进
   `red_frag_out`，再用 `clear=False` 归约，最后写回（同文件 `:141-158`）。
   `clear=True` 就不做这次预拷贝。
3. butterfly 从 `offset = threads/2 = 128` 开始：
   `128, 64, 32` 三级走 shared memory + `__syncthreads()`；
   `16, 8, 4, 2, 1` 五级走 `shfl_xor_sync`（`src/tl_templates/cuda/reduce.h:280-295`）。
   共 8 级，其中 3 级 shared、5 级 shuffle。
4. `batch=4` 把 `ceil(N/4)` 次 batched AllReduce 共享**同一对 barrier**，
   省掉的是 shared-memory 往返之间的 barrier 数量（`reduce_op.py:42-46`）。

**常见错误**：把「线程内归约」也算进 butterfly 级数。线程内的串行累加发生在
调用 `AllReduce` 之前，不走通信。

---

## E7 概念题 ★★

**思路**：判据是 `get_extent()`，不是「dst 有几个元素」。

**参考答案**

1. 看两边能不能推出 extent：都推不出 → 标量路径（`tl.atomic_add_elem_op`）；
   任一边能推出 → tile-region 路径（`tl.tileop.atomicadd`）
   （`tilelang/language/atomic.py:264-311`）。
2. 只有标量路径支持 `return_prev=True`；tile-region 路径抛
   `NotImplementedError("return_prev is not supported for tile-region-based atomic operations")`
   （同文件 `:304-305`）。
3. `use_tma=True` 把 tile-region 原子加降到 sm90+ 的 TMA `cp.reduce`
   （`tilelang/cuda/language/atomic.py:12-50`），dtype 只支持
   `float16/bfloat16/float32/int32/uint32/uint64`（`src/cuda/op/atomic_add.cc:48-62`）。
4. 因为它不改变某一次调用，而是**打开一个硬件写模式标志**：之后普通的
   L0C→GM / UB→GM 写回都变成原子规约，直到 `T.set_atomic_none()` 关掉
   （`tilelang/ascend/language/mode.py:63-105`）。忘记关的后果是后面的普通 store
   也会累加，数值静默出错。

**常见错误**：以为 `T.atomic_add` 与 `T.set_atomic` 是等价的两种写法。前者是
一次性原语，后者是作用域式的模式开关。

---

## E8 代码阅读题 ★★

**思路**：把每个原语套回本章 §2.2 的表。

**参考答案**

1. `T.clear(A_local)` 生成 `tl.tileop.fill`（value=0），lowering 成并行循环
   （`tilelang/language/fill_op.py:48-72`；`src/op/fill.cc:152-172`）。
   不能省，因为下面用的是 `A_local[i, j] += ...`，累加需要有初值。
2. `A[...]` 是 `BufferLoad`，它的 extent 由该 buffer 在**这个位置**推断出的区域给出；
   `A_shared` 是整 buffer，extent 就是它的 shape `(blk_m, blk_k)`。
   这里没写切片是因为 dst 侧提供了完整 extent，src 侧按 §3.1 的规则取同秩区域
   （`tilelang/language/copy_op.py:38-46,99-102`）。
3. 两种：`(blk_m,)`（压缩）或 `(blk_m, 1)`（keepdim）（`reduce_op.py:63-71`）。
4. 默认 `clear=True`（`reduce_op.py:230-232`）。若改成 `clear=False` 且 `A_powsum`
   未初始化，就是把结果加到一段未定义的值上——数值不确定。

**常见错误**：以为 `T.copy(A[i, j], A_shared)` 是「从 `A[i,j]` 开始拷一块」的
无限语义。它只在能被推断出同秩 extent 时合法，且**不支持通用广播**。

---

## E9 代码阅读题 ★★

**参考答案**

1. `selMask` 是 `uint8` 的**位串（bit-packed）**；第 i 个 bit 控制第 i 个输出元素
   （`../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/api_docs/T.tile.select.md`）。
2. 单位是**字节**。`float32` 元素 4 字节，`(64 + 8) / 4 = 18`，所以 `dst[0] = src[18]`
   （`T.tile.gather.md`）。
3. 共同点是输出都是 `[val0, idx0, val1, idx1, …]` 的**降序交错对**，且 idx 是排序前在
   aligned buffer 里的 0-based 位置。填充：float32 输入时 `actual_num` 到
   `aligned_count` 之间**原地**填 `-inf`；float16 输入不改源（`T.tile.sort.md`、
   `T.tile.topk.md`）。

**常见错误**：把 `gather` 的 offset 当元素下标用，结果差 4 倍。

---

## E10 代码阅读题 ★★★

**参考答案**

1. 源侧对应 `value = if_then_else(src_predicate, value, analyzer->Simplify(safe_value))`
   （`src/op/copy.cc:520`）——用**选择**，因为每条 lane 都要产出一个值；
   目标侧对应 `body = IfThenElse(dst_predicate, body)`（同文件 `:525`）——用
   **条件**，因为越界的 lane 干脆不写。
2. 输出里越界 lane 的取值从 `0.0` 变成 `-1.0`。对应源码里的 `src_oob_safe_value`
   （同文件 `:103,514-516`）。
3. 若 `extent` 能被证明整除/覆盖，`MakePredicate` 里的 `CanProve` 会短路，
   `cond_list` 为空、返回未定义的 `PrimExpr`，脚本也就不生成谓词
   （同文件 `:449-481`）。脚本的行为与它一致。

**常见错误**：以为越界时是「跳过」而不是「填值」。源侧越界一定会产生一个值，
否则整条向量化循环没法写。

---

## E11 动手题 ★★★

**思路**：`AllReduce` 每级把 `offset` 减半，`offset >= 32` 走 shared，否则走 shuffle。

**参考答案**

1. `threads=256, scale=1`：offset 依次为 128/64/32/16/8/4/2/1，
   前 3 级走 shared，后 5 级走 shuffle。`threads=32, scale=1`：offset 只有 16/8/4/2/1，
   **0 级 shared、5 级 shuffle**。
2. 对应 `src/tl_templates/cuda/reduce.h:280-295` 的
   `if constexpr (offset >= 32) { … shared … } else { … shfl_xor_sync … }`。
3. `threads=48` 违反 `((threads / scale) & (threads / scale - 1)) == 0`
   这条 `static_assert`（同文件 `:245-247`）：48 不是 2 的幂。原因是 butterfly 靠
   `threadIdx.x ^ offset` 精确配对，只有当归约宽度是 2 的幂时，每一级才能把参与者
   两两配对而不跨组。
4. `clear=true` 时结果等于 `reduce(src)`；`clear=false` 时结果等于
   `out_old ⊕ reduce(src)`。对应 `reduce_op.py:141-158`：
   `if not clear: copy(out, red_frag_out)`。

**常见错误**：把 `threads` 当成「归约宽度」。真正的归约宽度是 `threads / scale`。

---

## E12 动手题 ★★★

**思路**：按 §4.1 的八步判定链逐步排除。

**参考答案**（前提：A 是 fp16、行宽 64 元素 = 1024 bit，是 128 的倍数；
目标 shared buffer 是 `shared.dyn`）

| 调用 | 预测 CopyInst | 决定它的是哪一步 |
| --- | --- | --- |
| `T.copy(A_g, A_s)` | `kBulkStore` 不适用（这是 load），实际落到 `kNormal` | 第 8 步：默认只先试 TMA **store**，load 方向没有显式提示时不选 TMA，于是走同步形态（`copy_analysis.cc:796-807`） |
| `T.copy(A_g, A_s, disable_tma=True)` | `kNormal` | 第 8 步，且 `disable_tma` 直接跳过了 TMA 尝试（同文件 `:796`；`GetDisableTMA` 在 `src/cuda/op/copy.cc:149-151`） |
| `T.copy(A_g, A_s, prefer_instruction="cp_async")` | `kCPAsync` | 第 6 步：`can_cp_async` 满足（global→shared、dtype 相同）则选 cp.async（同文件 `:781-790`） |
| `T.async_copy(A_g, A_s)` | `kCPAsync` | 第 4 步：`explicit_cp_async` 优先于所有偏好设置（同文件 `:757-760`） |

会**直接编译失败**的两类：

* `T.async_copy` 在约束不满足时（不是 global→shared，或两侧 dtype 不同）——
  `Unsupported(facts.async_unavailable_reason)`（同文件 `:757-760`，原因串构造见 `:560-575`）。
* `prefer_instruction="tma"` 与 `disable_tma=True`（或 pass config
  `tl.disable_tma_lower`）冲突时——`Unsupported(...)`（同文件 `:762-777`）。

注意本表第一行：**默认的 `T.copy` load 不会自动走 TMA**。这是本章最容易误判的
一处；如果你想让它走 TMA，就显式写 `prefer_instruction="tma"`。

**常见错误**：把「能走 TMA」当成「默认走 TMA」。源码里的默认分支只对 store 尝试 TMA。

（可选硬件题：请在答案里注明实际使用的 GPU 型号、驱动、CUDA 与 TileLang 版本，
并贴出生成源码；没有硬件的读者不要填写任何「实测」结论。）
