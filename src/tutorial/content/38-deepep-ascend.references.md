# 第 38 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**

第五部分引用的 DeepSeek 仓库**不是 `_vendor/` 里的锁定源码**，而是 corpus 里
2026-10-01 抓取的快照；每个仓库的 upstream 与 commit 记在
`../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`。
DeepEP-Ascend 的快照 commit 是 `3b25377d04b2`（2026-09-29，Initial public release），
NVIDIA 侧 DeepEP 是 `93eb6eb23812`（2026-09-30）。凡引用这些仓库的文件，
行号都是**那次快照**里的行号；仓库一更新，行号就可能漂移。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章（以及整个第五部分）的示例最高验证级别是「仅静态核对」或
「CPU 逻辑验证」，**没有任何一条性能数字来自本机实测**。

---

## 一手材料：DeepEP-Ascend

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| R1 | `.../DeepEP-Ascend/README.md` | **本章主依据**：定位（`:3`、`:5`）、性能表与测量条件（`:9-21`）、Requirements（`:27-35`）、安装（`:37-63`）、Features 与 API 兼容（`:60-74`）、Ongoing（`:76-84`）、EP 用法与代码样例（`:86-196`）、Bucket / PP / Engram（`:198-226`）、环境变量（`:228-248`）、复现命令（`:250-281`） | 已核验（2026-10-01）：全文 300 行逐行读 |
| R2 | `.../DeepEP-Ascend/deep_ep/buffers/ep.py` | `EPHandle` 字段与 `topk_idx` 版本检查（`:21-98`）、`deterministic_sort`（`:100-160`）、`EPBuffer` 构造与断言（`:175-249`）、`get_buffer_size_hint`（`:251-275`）、`dispatch`（`:314-508`）、`combine`（`:521-592`）、LB 两个入口（`:594-649`） | 已核验（2026-10-01）：逐行读 |
| R3 | `.../DeepEP-Ascend/csrc/buffers/ep.hpp` | 对称内存注册与 host workspace（`:36-98`）、LB 未实现（`:75-86`）、`dispatch` 的 host 逻辑与 epilogue 拆分（`:88-409`）、`combine`（`:411-558`）、容量公式（`:560-605`） | 已核验（2026-10-01）：逐行读 |
| R4 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch.hpp` | device 侧 dispatch：UB 布局（`:26-75`）、SIMT 路由与去重（`:86-549`）、SQE 写入（`:327-429`）、本地拷贝（`:806-865`） | 已核验（2026-10-01）：逐行读 |
| R5 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/combine.hpp` 与 `combine_utils.hpp` | combine 的设备侧实现：任务与 UB 布局（`:20-55`）、metadata/WQE（`:59-151`）、主流程（`:162-...`）；FP32 环形归约（`combine_utils.hpp:14-90`） | 已核验（2026-10-01）：逐行读 |
| R6 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp`、`combine_reduce_epilogue.hpp` | 两个 epilogue kernel（被 `defer_epilogue` 单独 launch） | 已核验（2026-10-01）：读函数入口与 host 调用点 |
| R7 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/layout/ep/token.hpp` | `TokenLayout` 与 `BufferLayout`（`:9-130`）——§4.2、§3.2 的依据 | 已核验（2026-10-01）：逐行读 |
| R8 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/layout/ep/workspace.hpp`、`.../layout/common/signals.hpp` | `EPSignals` 的字段与 `peer_ptrs` / `get_sym_ptr`——§4.1、§4.6 的依据 | 已核验（2026-10-01）：逐行读 |
| R9 | `.../DeepEP-Ascend/csrc/kernels/ep/dispatch.hpp`、`combine.hpp` | host 侧 launch：拼模板参数、`jit->compile`、`jit->launch`、`num_blocks` / `num_ubuf_bytes` | 已核验（2026-10-01）：逐行读 |
| R10 | `.../DeepEP-Ascend/csrc/kernels/comm/hccl.hpp` | HCCL 通信域、netlayer 1、UBMEM 信道、Jetty、内存注册与对端地址（`:48-98`、`:124-299`） | 已核验（2026-10-01）：逐行读 |
| R11 | `.../DeepEP-Ascend/csrc/kernels/comm/symmetric.hpp`、`csrc/comm/api.hpp`、`csrc/comm/handle.hpp`、`csrc/comm/barrier.hpp` | 对称内存（`aclrtMalloc`）、`Context`（workspace / buffer 切分、物理/逻辑域）、通信域 API、barrier | 已核验（2026-10-01）：逐行读 |
| R12 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/comm/handle.hpp` | `HcommJettyInfo` / `HcommPeerInfo`（`:34-131`）、`HcommJetty` 与 SQE 模板（`:167-454`）、WRITE / WRITE_WITH_NOTIFY（`:257-258`、`:378-395`） | 已核验（2026-10-01）：逐行读 |
| R13 | `.../DeepEP-Ascend/deep_ep/include/deep_ep/common/compiled.hpp`、`common/reduction.hpp` | 常量（`kNumUbAlignmentBytes=32`、`kNumAllocationAlignmentBytes=2 MB`、`kNumMaxRanks=256`、`kNumMaxExperts=2048`、`kNumMaxVecCores=64`、`sf_pack_t=int16`）与归约状态编码 | 已核验（2026-10-01）：逐行读 |
| R14 | `.../DeepEP-Ascend/csrc/runtime/jit.hpp` | DeepJIT 的 `Config`（库根、前缀 `"EP"`、include 目录与前缀）、设备查询（AI core / vec core / UB 大小）、`set_npu_arch` / `set_barrier_in_prologue` | 已核验（2026-10-01）：逐行读 |
| R15 | `.../DeepEP-Ascend/tests/ep/test_ep.py` | 用法与口径：模式枚举（`:25-30`）、FP8 造数（`:89-100`）、cached dispatch（`:330-388`）、性能口径 `count_bytes` 与 `bench_msprof`（`:390-504`）、命令行默认值（`:553-590`） | 已核验（2026-10-01）：逐行读 |
| R16 | `.../DeepEP-Ascend/deep_ep/utils/math.py` | `per_token_cast_to_fp8`（`:30-38`，128 元素一个 fp32 scale）、`count_bytes`（`:57-64`）——payload 口径的依据 | 已核验（2026-10-01）：逐行读 |
| R17 | `.../DeepEP-Ascend/deep_ep/utils/testing.py` | `bench_msprof`（`:297-...`）：warmup / samples / flush_l2 / barrier profiling；`KernelProfile.gbps`（`:125-126`） | 已核验（2026-10-01）：读函数与参数 |
| R18 | `.../DeepEP-Ascend/deep_ep/utils/envs.py`、`deep_ep/__init__.py` | `init_dist`（HCCL backend、`WORLD_SIZE`/`RANK` 语义）、`init_jit` 在 import 时执行 | 已核验（2026-10-01）：逐行读 |

---

## 配套仓库（NVIDIA 对照、调度、均衡、JIT、算子）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| D1 | `.../DeepEP/README.md` | NVIDIA 侧对照：V2.5 拆分（`:11-20`）、接口与例子（`:103-289`）、deferred epilogues（`:283-289`）、expert load balancing 完整语义（`:330-380`） | 已核验（2026-10-01）：逐行读到 `:400` |
| D2 | `.../DeepJIT/README.md` | JIT 运行时：双后端、缓存与共享缓存（`:39-79`）、调用示例（CUDA / Ascend）、环境变量表、Ascend 后端要求（`bin/bisheng`、`bin/ld.lld`、`adv_api`）、缓存键组成 | 已核验（2026-10-01）：逐节读 |
| D3 | `.../DualPipe/README.md` | 双向流水并行：full overlap 的定位（`:3`）、气泡对比表（`:26-31`） | 已核验（2026-10-01）：逐行读（69 行） |
| D4 | `.../EPLB/README.md` | 专家负载均衡：冗余专家策略（`:5-8`）、两种策略（`:19-31`）、`rebalance_experts` 接口与例子（`:33-57`） | 已核验（2026-10-01）：逐行读（67 行） |
| D5 | `.../TileKernels/tile_kernels/moe/topk_gate_kernel.py`、`topk_gate_cuda.py`、`topk_gate_asc.py` | TileLang 侧产出 `topk_idx`：入口与 dispatch 的类型约定；CUDA/Ascend 两份实现 | 已核验（2026-10-01）：读入口与 CUDA 实现 |
| D6 | `.../TileKernels/tile_kernels/moe/moe_topk_gate_forward_kernel.py` | TileLang 侧 MoE 路由：`topk_idx` / `topk_weights` 的产出与形状约定 | 已核验（2026-10-01）：读签名与注释 |
| D7 | `.../TileKernels/tile_kernels/quant/per_token_cast_kernel.py` | TileLang 侧 per-token 量化：CUDA / Ascend 双实现，产出入 dispatch 的 fp8 + sf | 已核验（2026-10-01）：读入口与断言 |
| D8 | `.../DeepGEMM-Ascend/deep_gemm/include/deep_gemm/layout/mega_moe.hpp:26-33` | MegaMoE 的四个进度阶段（§9.2 的对照） | 已核验（2026-10-01）：读枚举；另见第 35 章 |
| D9 | `.../deepseek-article-20260930.md:9-13` | 官方发布文章：组件定位（`:12`）、「昇腾 950 的 128 卡超节点」（`:13`） | 已核验（2026-10-01）：逐字读 |
| D10 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` | 13 个仓库的 upstream、commit、日期 | 已核验（2026-10-01）：逐行读 |

---

## TileLang 侧对照（锁定源码 `_vendor/`）

本章只在 §6.4 做「TileLang JIT / ABI vs DeepJIT」的对照，以及 §9 说明分工线，
因此只需要第 26 章的结论，不新增源码引用。需要复核时回到：

* `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py`（`@tilelang.jit` 的装饰与缓存查询）；
* `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py`（磁盘缓存键与目录布局）；
* `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc`（packed host ABI）。

这三处的逐行分析见第 26 章，本章不重复。

---

## 关于性能数字的说明

**本章不提供任何本机实测数字。** 本章出现的性能相关数字只有三类：

| 数字 | 来源 | 条件 |
| --- | --- | --- |
| EP8–EP128 的 dispatch / combine GB/s | R1（`README.md:13-19`） | Ascend 950DT + CANN 9.2.0 + 手工配置的 **PoC HDK**；netlayer 1；16384 tokens/rank、hidden 7168、top-6/256 experts、32 AI cores / 64 AIVs、expert alignment 128；dispatch FP8 + row-major scale，combine BF16；10 warmup + 50 samples，刷 cache；**作者未复现** |
| 「物理 payload 带宽上限的 90–95%」 | R1（`README.md:21`） | 原文是 "roughly"；本书反推出 350–420 GB/s/rank 的**区间估计**，见 `code/04-bandwidth-normalize/` |
| 一层 MoE 的 dispatch / combine 字节数 | 本书 `code/02-comm-ledger/` 复算 | 均匀路由假设下的**上界量级**，不是实测流量 |

另外，源码里有三个**作者观察**（不是实测）：barrier 在 8 rank 时约 4 µs
（R4 `:643`）、kernel launch + `InitSocState()` 约 1.9 µs（R4 `:573`）、
VF 调用约 0.5 µs（R4 `:670`）。它们只是注释，本章引用时都写成了「源码里写的观察」。

---

## 待补资料

1. **没有 Ascend 950 UBMEM / URMA 链路的官方带宽规格。**
   §7.4 的反推（350–420 GB/s/rank）缺一个可以对照的硬件上限。
   建议来源：华为 Atlas 850E 或 Ascend 950 的产品/部署文档，
   或在真机上跑一个纯 URMA 带宽 microbenchmark。
2. **PoC HDK 与商用 Q3 HDK 的差异没有量化。**
   README 说商用版本「includes the configuration required for full-bandwidth
   operation」（`README.md:43-45`），但没有给两组数字的对比。
   2026-10-15 前后商用版本发布后，建议补一份对照。
3. **CANN 侧的头文件不在 corpus 里。**
   `hccl/hccl.h`、`hcomm/hcomm.h`、`kernel_operator.h`、`c_api/*`、`simt_api/*`
   的语义（尤其 `HcommUrmaSqeCtx` 的字段布局、`kSQDepth` 的硬件含义、
   `notifyNum = 3` 这个「magic number」）只能从调用点反推。
   §5 已明确写成「按调用点描述分工」，未对 CANN 内部实现下结论。
4. **`EP_JIT_*` 与 DeepJIT 的 `DJ_JIT_*` 的完整回落表分散在两处。**
   本章只列了 README 的六个变量；DeepJIT README 的完整表更长。
   `ASCEND_HOME_PATH` 的读取时机（import 时还是首次 launch 时）本章未核实。
5. **`deterministic_sort` 的开销没有数据。**
   源码只说它排 `T·k` 行；没有它在真实形状下的耗时。
   建议补一份「开/关 deterministic 的 dispatch 耗时对比」。
6. **combine 的「本地归约 + HBM 争抢」缺一个可分解的数字。**
   README 说 combine 更慢有这两个原因，但没有给出行内拆解。
   建议用 msProf 把 combine 拆成「URMA 写」「UB 归约」「HBM 读写」三段。
7. **Bucket 的 session 语义与 all-gather 之外的能力缺材料。**
   `README.md:220` 给了几条约束（等长 shard、不自动扩容、退出不加 barrier），
   但 reduce-scatter / all-reduce 的实现状态只有一句「still being built」。
8. **与 TileKernels / DeepGEMM-Ascend 的接口契约没有正式文档。**
   §9.1 的三张量约定（`topk_idx` / `topk_weights` / `x`）是从两边的签名
   反推出来的，上游没有一份「谁产谁消」的对接说明。
   建议向上游确认这份契约是否有意固定。
