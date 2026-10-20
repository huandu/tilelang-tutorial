# 第 34 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）。**
本章涉及的三个开源仓库（TileKernels / Engram / DeepGEMM-Ascend）取自 corpus 的
2026-09-30 快照，**不是 `_vendor/` 锁定树**；凡引用它们都写成 corpus 路径。
此外本章有**两篇论文**：Engram 论文（`paper-deepseek-engram.md`）与
mHC 论文（`paper-deepseek-mhc.md`，arXiv:2512.24880 v1，2025-12-31，
抓取日期 2026-10-01）。两者的实验数字都是**论文自报**，本章一条也没有复现。
为便于阅读，下面把
`14-deepseek-ascend-stack/raw/TileKernels/` 记作 **TK**、
`14-deepseek-ascend-stack/raw/Engram/` 记作 **EN**、
`14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/` 记作 **DG**、
`14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md` 记作 **MHC**；
表中每条引用仍写完整路径。

写作环境：macOS (Apple Silicon)、Python 3.11、torch 2.1.1；
**没有 NVIDIA GPU，也没有昇腾 NPU**。因此：

* 本章所有 kernel 的最高验证级别是 **“仅静态核对”**（源码逐行读过、行号核对过；
  **没有在本机运行过任何 kernel**）；
* 本章所有“本机数字”都来自 `code/` 下的 **CPU 复算**，标注方式见下。

引用分三类，核验状态只有四种口径：

| 口径 | 含义 |
| --- | --- |
| **已核验（静态）** | 在 corpus 快照上用 `sed -n` / `rg -n` 定位到该行并逐行读完；行号与内容一致。**不是运行验证。** |
| **已核验（本地全文）** | corpus 里保存了该来源的完整副本。|
| **本书复算** | 数字由本章 `code/` 下的脚本在本机 CPU 上算出，可复现。|
| **未核验** | 只有转述、没有原文；明确写出原因。|

---

## 一、TK：TileKernels（`14-deepseek-ascend-stack/raw/TileKernels/`）

本章的主线。上游：https://github.com/deepseek-ai/TileKernels（corpus 记录 ref
`66258df6175d`，见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`）。

### 1.1 库级说明

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| T1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:3-5` | 库的定位：用 TileLang 写、双后端、说多数算子接近硬件上限 | 已核验（静态）：逐行读 |
| T2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:9` | 2026-09-30 新增昇腾后端、运行期自动选择后端 | 已核验（静态） |
| T3 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:13-19` | 功能清单（MoE Routing / Quantization / Engram / mHC / Transform / Rand / Modeling） | 已核验（静态） |
| T4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:21-31` | 依赖与版本：Python ≥ 3.12、PyTorch ≥ 2.13、TileLang ≥ 0.1.15、CUDA ≥ 13.1、SM90/SM100、Ascend 950 + CANN ≥ 9.2.0 | 已核验（静态） |
| T5 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:54-56` | 测试入口与 `--run-benchmark` 开关 | 已核验（静态） |
| T6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:75-86` | 目录结构（engram / mhc / modeling / moe / quant / rand / transform） | 已核验（静态） |

### 1.2 Engram 算子族

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| T7 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/__init__.py:1-11` | 导出清单：fused_weight / gate_bwd / gate_fwd / grad_w_reduce / hash / sinkhorn×4 | 已核验（静态） |
| T8 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py:17-32` | 共享层：参数含义、`hc_mult = 4`、`kv` 的 `[T, hc+1, d]` 约定、`scalar = d^{-1/2}` | 已核验（静态）：逐行读 |
| T9 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py:35-61` | `is_ascend()` 分支：昇腾走 `_asc`、其余走 `_cuda` | 已核验（静态） |
| T10 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py:63-83` | 四个 `[T, hc]` 保存张量的分配与 kernel 调用 | 已核验（静态） |
| T11 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:7-12,27-29` | pass config（关 warp specialization / thread storage sync）、`threads=32`、`vec_size=8` | 已核验（静态） |
| T12 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:31-53` | `_choose_blk_d` / `_choose_num_persistent_blocks` / `v_start_phase` | 已核验（静态） |
| T13 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:68,83-95` | 网格 `(hc_mult, num_persistent_blocks)`、`x_smem` / `kv_smem`、prologue | 已核验（静态） |
| T14 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:110-126` | 双缓冲 `async_copy` + `ptx_wait_group(2)` + 三个累加器（读 `weight_fused`） | 已核验（静态）：逐行读 |
| T15 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:149-166` | `warp_reduce_sum` + `rsqrt` + signed-sqrt gate（`copysign`/`clamp`） | 已核验（静态）：逐行读 |
| T16 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:173-195` | 第二遍输出，`v` 复用 `kv_smem`，预取下一个 token | 已核验（静态） |
| T17 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:26` | `inplace` 与 `save_for_backward` 互斥的 assert | 已核验（静态） |
| T18 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:1-3,18-30` | 昇腾方言导入、`vec_size=64`、`heads_per_task=2`、`num_stages=2`、断言 | 已核验（静态）：逐行读 |
| T19 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:45-74` | `T.Kernel(num_vec_cores)`、UB 缓冲、`annotate_buffer_versions`、权重常驻 | 已核验（静态）：逐行读 |
| T20 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:76-121` | `T.Persistent([num_tokens, num_head_groups])`、image token 直通、搬 x/k | 已核验（静态） |
| T21 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:122-177` | `T.SimdVF()` 内 `vld/vcvt/vmula` 三累加、`vcadd` 归约、signed-sqrt gate | 已核验（静态）：逐行读 |
| T22 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:179-251` | 第二遍输出（`BRC_B32` 广播、`k_output_ub` 复用）、`save_for_backward` 写出 | 已核验（静态） |
| T23 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_fused_weight_kernel.py:7-16` | `fused_weight` 的契约（bf16 × bf16 → fp32） | 已核验（静态） |
| T24 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_fused_weight_cuda.py:28-38` | 逐元素实现（`T.Kernel(hc_mult, num_blk)`，`vec_size=8`） | 已核验（静态） |
| T25 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_fused_weight_asc.py:25-55` | 昇腾版（`T.Persistent([hc_mult, num_tiles])` + `S.vmul`） | 已核验（静态） |
| T26 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_kernel.py:42-85` | 反向共享层：`num_partials = num_vec_cores // hc_mult`（昇腾）/ `num_sms`（CUDA） | 已核验（静态） |
| T27 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:25-36` | docstring“8 warps per CTA, 2 warps per head”与几何量 | 已核验（静态）：逐行读 |
| T28 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:206-236` | `dldg` 的两遍计算与跨 warp shared-memory 归约 | 已核验（静态） |
| T29 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:248-253` | 穿过 sigmoid/signed-sqrt 的梯度 + `T.Select` 的 clamp 分支 | 已核验（静态）：逐行读 |
| T30 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:266-299` | `dot_x` / `dot_k` 与 `grad_x` / `grad_kv` / `grad_w_local` | 已核验（静态） |
| T31 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:332-336` | `grad_w_partial[pid_p, head, ...]`（部分和的第一维 = CTA） | 已核验（静态） |
| T32 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_grad_w_reduce_kernel.py:9-24` | `grad_w_reduce` 的契约（累加语义、fp32 输出） | 已核验（静态） |
| T33 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_grad_w_reduce_cuda.py:35-63` | 分批搬部分和 + 片上累加 + 两条链式法则 | 已核验（静态）：逐行读 |
| T34 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_kernel.py:14-27` | hash 的参数语义（乘子/素数表/offset，均为固定超参） | 已核验（静态） |
| T35 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_cuda.py:31-65` | 乘法-XOR 哈希、逐 head 取模加 offset、image token 写 `-1` | 已核验（静态）：逐行读 |
| T36 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_kernel.py:13-65` | momentum 与 step 的契约；`num_partials` 的选择 | 已核验（静态） |
| T37 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_kernel.py:68-106,109-136,139-178` | step / step_reduce / finalize 三个入口 | 已核验（静态） |
| T38 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_cuda.py:19-59` | momentum kernel（`T.ieee_fmaf`、Nesterov 前瞻） | 已核验（静态） |
| T39 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_cuda.py:68-141` | step kernel：逐行 L2、原地更新 `row_scale`、写列部分和 | 已核验（静态）：逐行读 |
| T40 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_cuda.py:150-191` | step_reduce：确定性顺序归约 | 已核验（静态） |
| T41 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_cuda.py:199-250` | finalize：`alpha / (‖·‖₂ + eps)` | 已核验（静态） |
| T42 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_asc.py:24-85` | momentum 的昇腾版（UB 多版本 + `S.vmadd/vaxpy`） | 已核验（静态） |
| T43 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_grad_w_reduce_asc.py:30-74` | 昇腾版归约（`T.Persistent([shards, hc_mult])`） | 已核验（静态） |

### 1.3 mHC 算子族

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| T44 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/__init__.py:1-17` | mHC 导出清单（expand / head_compute_mix / multilayer_recompute / norm_fn×4 / post / pre_apply_mix / pre_big_fuse / pre_split_mixes×2 / sinkhorn×2） | 已核验（静态） |
| T45 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_kernel.py:6-21` | 前向/反向入口与 `token_block_size`（CUDA 8 / 昇腾 4） | 已核验（静态）：逐行读 |
| T46 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:24-56` | Sinkhorn 前向：softmax + eps、列归一化、`repeat-1` 轮行/列交替 | 已核验（静态）：逐行读 |
| T47 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:91-92,124-153` | 反向：保存 `2 × repeat` 份中间量与行列和，逆序回放 | 已核验（静态） |
| T48 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_asc.py`（存在性） | 昇腾版 Sinkhorn（未逐行读；本章只引用其存在与 `token_block_size=4`） | 已核验（静态）：文件存在；**算法未逐行核对** |
| T49 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:24-68` | `mhc_mult3` 三段拆分、`scale`/`base` 的应用、`sigmoid`/`post_mult` | 已核验（静态）：逐行读 |
| T50 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:100-171` | 反向：`input_mixes_grad`、`mhc_scale_grad_partial`、`mhc_base_grad_partial` | 已核验（静态） |
| T51 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:176-209` | `pre_split_mixes_bwd_reduce_cuda`：单次启动归约所有参数梯度 | 已核验（静态） |
| T52 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_kernel.py:14-52,55-115` | 入口与“昇腾侧在 host 上做 `.sum(0)`、CUDA 侧用 reduce kernel”的差异 | 已核验（静态） |
| T53 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_apply_mix_cuda.py:24-53` | `o = Σ_m mix_m · x_m` 的逐元素实现与 `h_blk` 切分 | 已核验（静态） |
| T54 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_apply_mix_cuda.py:56-113` | 反向：`x_grad += mix·o_grad` 与 `mix_grad` 的归约 | 已核验（静态）：逐行读 |
| T55 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_apply_mix_asc.py:1-54` | 昇腾版（`T.SimdVF` + `vmula`/`vsts`；`mix` 的 UB padding） | 已核验（静态） |
| T56 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/post_kernel.py:17-30` | 形状断言（`[num_seqs, num_tokens, mhc, hidden]` 等） | 已核验（静态） |
| T57 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/post_cuda.py:27-60` | `new_res[o] = post[o]·out + Σ_m comb[m,o]·res[m]` 的实现与自定义 Fragment 布局 | 已核验（静态）：逐行读 |
| T58 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/post_cuda.py:66-155` | 反向四项：`d_comb` / `d_residual` / `d_post` / `d_x` | 已核验（静态） |
| T59 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/head_compute_mix_cuda.py:30-35` | `sigmoid(x·scale + base) + eps` | 已核验（静态） |
| T60 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/head_compute_mix_cuda.py:62-93` | 反向：`scale` / `base` 的部分和 | 已核验（静态） |
| T61 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/head_compute_mix_kernel.py:10-25` | 昇腾侧 `token_block_size` 的推导与“为什么不能更大”的注释 | 已核验（静态） |
| T62 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/expand_cuda.py:6-27,31-53` | expand 前向（复制 M 份）与反向（沿分支求和） | 已核验（静态）：逐行读 |
| T63 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/expand_kernel.py:5-23` | 昇腾侧 `h_blk` 的 UB 预算推导 | 已核验（静态） |
| T64 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/expand_asc.py:5-45` | 昇腾版（`T.simd.pset(16)` + 逐份 `vsts`） | 已核验（静态） |
| T65 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_cuda.py:44-135` | 融合前向：一个 token 一个 CTA，`T.get_thread_binding() < 32` 分两组 | 已核验（静态）：逐行读 |
| T66 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_kernel.py:27-53` | 昇腾侧约束（`mhc_mult == 4`、`gemm_out_mul.shape[0] == 1`、`n_splits=1`） | 已核验（静态） |
| T67 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_asc.py:43-51` | “op-for-op / bitwise identical”的融合正确性声明 | 已核验（静态）：逐行读 |
| T68 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_asc.py:122-134,236-373` | “把整条延迟路径放进一个 SimdVF phase”的注释与 Sinkhorn 复刻 | 已核验（静态）：分段读 |
| T69 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_kernel.py:36-98` | 列表 API、`_make_ptr_tables_batched`、`h_blk` 选择 | 已核验（静态）：逐行读 |
| T70 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:40-126` | kernel 内逐层循环：重算 `layer_input` 与 `residual`；bf16 往返 | 已核验（静态）：逐行读 |
| T71 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_asc.py:9-45` | 昇腾版：`TL_ENABLE_AUTO_SCHEDULE: False`、双版本 UB | 已核验（静态） |
| T72 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/norm_fn_kernel.py:21-134` | `fn_normw_merge` 与 `reduce_partials_and_rmsnorm` 的四个入口（前向/反向） | 已核验（静态） |
| T73 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/norm_fn_asc.py:1-60` | 昇腾版：`_row_reduce` / `_row_max` 的 SIMD 组合（`vcgadd`+`vselr`+`vsel`） | 已核验（静态）：分段读 |

### 1.4 PyTorch 基线（本章 CPU 复算的对照物）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| T74 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:160-217` | `engram_gate_ref`：门控的向量化基线（本章 §3.1 的恒等式来源） | 已核验（静态）：逐行读 |
| T75 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:118-157` | `engram_hash_ref`：哈希基线 | 已核验（静态） |
| T76 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:99-115` | `make_offsets`：exclusive prefix sum | 已核验（静态） |
| T77 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:6-96` | Engram Sinkhorn 的 momentum / step / finalize 三个基线 | 已核验（静态）：逐行读 |
| T78 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py:8-14` | `sinkhorn_normalize_ref`：mHC Sinkhorn 的定义 | 已核验（静态）：逐行读 |
| T79 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py:17-63` | `head_compute_mix` / `split_mixes` / `apply_mix` / `post` 四个基线 | 已核验（静态）：逐行读 |
| T80 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py:66-110` | `pre_norm_fn` 与 split-K GEMM partials 的基线 | 已核验（静态） |

### 1.5 测试（形状、容差与“测什么”的证据）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| T81 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/engram/test_engram_gate_fwd.py:18-104` | 门控前向的容差（`rtol/atol = 8e-3`）与三条分支（save/inplace/image mask） | 已核验（静态）：逐行读 |
| T82 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/engram/test_engram_gate_fwd.py:107-143` | benchmark 的口径（`count_bytes` / `time_us` / `bandwidth_gbs`） | 已核验（静态） |
| T83 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/engram/test_engram_grad_w_reduce.py:19-22` | `grad_w_reduce` 的三行基线（本章 §4.3 引用） | 已核验（静态）：逐行读 |
| T84 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/engram/test_engram_sinkhorn.py:96-144` | Engram Sinkhorn 的规模（`num_rows` 到 1e5、`head_dim` 到 5120）与 `alpha = sqrt(head_dim)` | 已核验（静态） |
| T85 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_sinkhorn.py:11-102` | mHC Sinkhorn 的 `repeat = 10`、`eps = 1e-6` 与 ±100 offset 的稳定性用例 | 已核验（静态）：逐行读 |
| T86 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_pre_big_fuse.py:61-205` | 融合 kernel 的 golden 是“四个算子串起来”；`hidden_size` 覆盖 1280–5120 | 已核验（静态）：逐行读 |
| T87 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_multilayer_recompute.py:71-125` | 重算基线与 `num_layers/num_post` 的约束 | 已核验（静态） |
| T88 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_multilayer_recompute.py:99-108` | fused vs ref 的 IO 字节记账（本章 §6.5 引用） | 已核验（静态） |

---

## 二、EN：Engram 仓库（`14-deepseek-ascend-stack/raw/Engram/`）

上游：https://github.com/deepseek-ai/Engram（corpus 记录 ref `main`）。

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| E1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/README.md:32-40` | 论文摘要与四条贡献（Sparsity Allocation / iso-parameter+iso-FLOPs / 机制分析 / 确定性寻址与卸载） | 已核验（本地全文） |
| E2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/README.md:78-90` | **“代码只是 demo”的自述**（本章 §1.3 引用）；Quick Start 与依赖 | 已核验（本地全文）：逐行读 |
| E3 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py:1-19` | demo 的三条免责声明（只是演示、未优化、mock 掉 Attention/MoE/mHC） | 已核验（本地全文）：逐行读 |
| E4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py:60-121` | `CompressedTokenizer`：NFKC + 小写 + StripAccents 的压缩查表 | 已核验（本地全文）：逐行读 |
| E5 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py:188-303` | `NgramHashMapping`：乘子生成、素数表搜索、逐 order 的 XOR 哈希 | 已核验（本地全文）：逐行读 |
| E6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py:305-324` | `MultiHeadEmbedding`：offset + 单张 `nn.Embedding` 的查表 | 已核验（本地全文） |
| E7 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py:326-378` | `Engram.forward`：门控（signed-sqrt + sigmoid）与 `value + short_conv(value)` | 已核验（本地全文）：逐行读 |
| E8 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py:396-422` | demo 的“mock hyper-connection”：`unsqueeze(2).expand(..., hc_mult, ...)` | 已核验（本地全文） |
| E9 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/Engram_paper.pdf`（存在性） | 论文原文（本 book 只使用其文本抽取，见下） | 已核验（本地全文）：文件存在 |

---

## 三、Engram 论文的文本抽取（`14-deepseek-ascend-stack/raw/paper-deepseek-engram.md`）

来源：`Engram_paper.pdf` 的 `pdftotext -layout` 抽取；上游
https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf。
**所有数字都是论文自报，本章作者未复现。**

| id | 引用位置（行号） | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| P1 | `:24-40` | 摘要：`O(1)` lookup、U 型 scaling law、27B 上的增益、host memory 卸载 | 已核验（本地全文） |
| P2 | `:53-80` | 语言建模的“两类子任务”、conditional memory vs conditional computation | 已核验（本地全文） |
| P3 | `:92-99` | 确定性 ID 支持预取；100B 表卸载开销 < 3% | 已核验（本地全文） |
| P4 | `:140-156` | Engram 的两个功能阶段（retrieval / fusion） | 已核验（本地全文） |
| P5 | `:158-166` | 词表压缩，128k 词表上缩小 23% | 已核验（本地全文） |
| P6 | `:167-176` | 多头哈希与式 (1)(2) | 已核验（本地全文） |
| P7 | `:177-208` | 上下文感知门控：式 (3)(4) | 已核验（本地全文） |
| P8 | `:209-217` | 短卷积式 (5) 与残差 | 已核验（本地全文） |
| P9 | `:212-217` | `M = 4` 的 mHC 默认骨干（本章 §6.1 引用） | 已核验（本地全文） |
| P10 | `:247-262` | 训练/推理的系统实现（All-to-All、host 预取与放置） | 已核验（本地全文） |
| P11 | `:315-351` | Decoupling compute and memory；Model placement 的取舍 | 已核验（本地全文） |
| P12 | `:352-362` | Zipfian 分布与多级 cache 层级 | 已核验（本地全文） |
| P13 | `:364-412` | Sparsity Allocation 的 formulation 与实验协议 | 已核验（本地全文） |
| P14 | `:413-430` | U 型结果、`1.7248 → 1.7109`、`ρ ≈ 75%–80%`、`20%–25%` | 已核验（本地全文）：逐行读 |
| P15 | `:431-465` | 无限内存 regime 与幂律 | 已核验（本地全文） |
| P16 | `:470-520` | 27B 规模的 Table 1（本章图 6 的数据源） | 已核验（本地全文） |
| P17 | `:670-682` | 长上下文（Multi-Query NIAH 97.0 vs 84.2 等） | 已核验（本地全文） |
| P18 | `:1127-1130` | 参考文献里的 hyper-connections（2025） | 已核验（本地全文） |

---

## 三·五、MHC：mHC 论文的文本抽取（`14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`）

来源：https://arxiv.org/abs/2512.24880（v1，2025-12-31；DeepSeek-AI）。
本地为 arXiv HTML 版正文的文本抽取（1522 行），同目录另有
`paper-deepseek-mhc.pdf`。**公式与图未保留**，所以本章引用时给的是行号 + 式号，
并用文字复述公式内容（例如把式 (6) 写成集合定义）。

本章**逐行读过**下列段落；这些行号是本章所有 mHC 结论的依据：

| id | 行号 | 内容 | 本章用在哪 |
| --- | --- | --- | --- |
| M1 | `:109-118` | 摘要：HC 破坏 identity mapping → 训练不稳定 + 访存开销；mHC 把残差空间投影到特定流形 | §1.4、§6.1 |
| M2 | `:198-207` | HC 的记号：`x_l ∈ R^{1×C}` 扩成 `x_l ∈ R^{n×C}`；`H^pre/H^post ∈ R^{1×n}`、`H^res ∈ R^{n×n}`；式 (3)(4) | §6.1、§6.2 |
| M3 | `:210-217` | 复合映射偏离单位映射 → 不稳定；mHC 用 Sinkhorn-Knopp 做“熵投影到 Birkhoff 多胞形”；`H^res x` 是凸组合 | §5.1、§6.1 |
| M4 | `:219` | 内部大规模训练只增加 **6.7%** 时间开销（`n = 4`） | §6.6、§6.7 |
| M5 | `:239-257` | §3 预备：`n` 流残差、三个映射的读/写/更新语义；映射本身的开销可忽略；Table 1 显示 `H^res` 贡献最大 | §6.1、§6.2 |
| M6 | `:282-292` | §3.1 数值不稳定：复合映射 `∏H^res`；Amax Gain Magnitude 峰值 **≈3000** | §5.1、§6.1 |
| M7 | `:294-345` | §3.2 系统开销：memory wall；Table 2 的逐 token 访存（Residual `2C/C`；HC `(5n+1)C+n²+2n` / `(3n+1)C+n²+2n`）；HC 的 I/O 约 `n` 倍、需要 gradient checkpointing、流水并行通信也涨 `n` 倍 | §6.1、§6.3、§6.5 |
| M8 | `:349-362` | §4.1 式 (6)：把 `H^res` 限制在双随机流形（Birkhoff 多胞形） | §5.1、§5.2、§6.1 |
| M9 | `:362-376` | 双随机的三条性质（谱范数 ≤1 / 乘法封闭 / 置换凸包）＋对 `H^pre`、`H^post` 的非负约束 | §5.1 |
| M10 | `:378-389` | §4.2 式 (7)：`x⃗ = vec(x)`、RMSNorm、动态 `φ` 与静态 `b` 的参数化 | §6.2 |
| M11 | `:391-399` | §4.2 式 (8)：`H^pre = σ`、`H^post = 2σ`、`H^res = Sinkhorn-Knopp` | §5.2、§6.2 |
| M12 | `:400-406` | §4.2：Sinkhorn 的起点 `M^(0)=exp(H̃)`、式 (9) `M^(t)=T_r(T_c(M^(t-1)))`、`t_max = 20` | §5.1、§5.2 |
| M13 | `:408-411` | §4.3 引言：`n = 4` 的实现只带来 **6.7%** 训练开销 | §6.6 |
| M14 | `:413-423` | §4.3.1：把“除以范数”换到矩阵乘之后；三个专用 kernel；式 (14)–(15) 的前向两遍扫描与反向两个矩阵乘各融一个 kernel | §6.2、§6.6、§7.3 |
| M15 | `:425` | §4.3.1：式 (16)–(18) 的轻量系数运算融合成一个 kernel | §6.1、§6.2 |
| M16 | `:429` | §4.3.1：式 (19) 的 Sinkhorn 迭代放一个 kernel；反向**在片上重算**中间结果并整轮回放 | §5.2 |
| M17 | `:430-499` | §4.3.1 式 (10)–(19) 的形状与 dtype（`φ: tfloat32 [nC, n²+2n]`、`x⃗: bfloat16 [1,nC]`、`α: float32` 标量、`b: float32 [1, n²+2n]`） | §6.2 |
| M18 | `:501-502` | §4.3.1 的两个“应用映射”kernel（`F_pre`、`F_post,res`）与融合收益（读 `(3n+1)C→(n+1)C`、写 `3nC→nC`）；**除式 (14)–(15) 外都用 TileLang** | §6.3、§6.6、§7.3、E11 |
| M19 | `:504-531` | §4.3.2 重计算：丢掉中间激活、反向重跑 mHC kernel；Table 3 的“存/重算”清单 | §6.5、E10 |
| M20 | `:532-539` | §4.3.2 式 (20)：`L*_r ≈ sqrt(nL/(n+2))`；重算块不能跨流水级边界 | §6.5、E10 |
| M21 | `:541-554` | §4.3.3：扩展 DualPipe 调度；`F_post,res` 走高优先级计算流；长耗时 attention 不用 persistent kernel；重算与流水通信解耦 | §6.7（详见第 38 章） |
| M22 | `:557-565` | §5.1 实验设置：`n = 4`；27B 主结果、3B/9B 算力缩放、另一个 3B 用 1T token 做 token 缩放 | §6.7 |
| M23 | `:571-685` | §5.2 主结果：Table 4 的 27B 下游 benchmark；相对 baseline 的最终 loss 降低 **0.021**；相对 HC 的 **BBH +2.1%**、**DROP +2.3%** | §6.7 |
| M24 | `:687-702` | §5.3 缩放实验（3B/9B/27B 的算力缩放曲线；3B 的 token 缩放） | §6.7 |
| M25 | `:703-715` | §5.4 稳定性：Sinkhorn 用 **20** 次迭代；单层近似双随机、复合后偏差上限约 **1.6**（对比 HC 的 ≈3000） | §5.1、§6.7 |
| M26 | `:1206-1382` | Appendix A.1 / Table 5：模型与超参。`Layer Norm ε = 1e-20`（`:1360-1363`）、expansion rate `n = 4`（`:1365-1368`）、gating factor init `α = 0.01`（`:1370-1373`）、`Sinkhorn-Knopp t_max = 20`（`:1375-1378`） | §6.7、§5.2 |

**论文与仓库实现不一致的四处**（本章 §5.2 的对照表 + §8 的 P13/P14）：

| # | 论文 | 源码（TK 的 `mhc/`） | 影响 |
| --- | --- | --- | --- |
| 1 | 起点 `M^(0) = exp(H̃)`，第一步 `T_r(T_c(·))`（M12） | 起点 `softmax(-1)`（等价于先做一次行归一化），第一步是列归一化（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:35-43`） | 迭代中间点不同，不动点相同 |
| 2 | `t_max = 20`（M12、M26） | `repeat = 10`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_sinkhorn.py:19`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_pre_big_fuse.py:24`） | “一次迭代”的定义不同，不能直接换算 |
| 3 | 无 eps | 每步 `+ eps` 且除以 `(sum + eps)`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:36-52`） | 固定点带 `O(eps)` 偏移 |
| 4 | `H^post = 2σ(·)`（M11） | 系数是参数 `mhc_post_mult_value`，仓库测试默认 **1.0**（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:62`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_pre_big_fuse.py:23`） | 用默认值等于“半强度”`post` |

（另有第 5 处不算矛盾：式 (8) 是 `H^pre = σ`，代码多加了 `mhc_pre_eps`
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:55-60`），属于数值保护。）

---

## 四、DG：DeepGEMM-Ascend（`14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/`）

本章只取三点：mHC prenorm GEMM 的性能表、“mHC kernel 由 TileLang 支撑”、
以及“什么时候下钻 MAD 原语”的动机。上游：
https://github.com/deepseek-ai/DeepGEMM-Ascend（corpus 记录 ref `8491bbb4b8c0`）。

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| D1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:5` | MAD 原语抽象：分形布局、对齐约束、地址计算、协程流水、稀疏加载 | 已核验（本地全文）：逐行读 |
| D2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:27-28` | `tilelang` 是 HC prenorm kernel 的依赖 | 已核验（本地全文） |
| D3 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:166-176` | **HC prenorm GEMM 性能表**（本章 §6.6 引用；厂商自报，条件：Ascend 950，M/N/K 见表） | 已核验（本地全文）；**作者未复现** |
| D4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:190` | “The mHC kernel is backed by Tilelang” | 已核验（本地全文） |
| D5 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:1-59` | prenorm GEMM 的 TileLang 实现：`get_config`（L0B/L0C 预算、split-K 选择） | 已核验（静态）：逐行读 |
| D6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:17` | “near-peak hardware performance”（宣传口径，未复现） | 已核验（本地全文） |

---

## 五、锁定版本源码引用（TileLang v0.1.15）

本章用到的 TileLang API 逐个在锁定版本里核对过存在性与定义位置：

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| V1 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:82` | `T.Persistent(...)` 的定义 | 已核验（2026-10-01）：`grep -n "def Persistent"` |
| V2 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:296` | `T.alloc_reducer` | 已核验（2026-10-01） |
| V3 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:358,398,434` | `T.reducer_init` / `reducer_update` / `finalize_reducer` | 已核验（2026-10-01） |
| V4 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:188` | `T.async_copy` | 已核验（2026-10-01） |
| V5 | `_vendor/tilelang-v0.1.15/tilelang/language/symbolics.py:13` | `T.dynamic` | 已核验（2026-10-01） |
| V6 | `_vendor/tilelang-v0.1.15/tilelang/language/proxy.py:265,325` | `T.ptr` 与 `T.make_tensor` | 已核验（2026-10-01） |
| V7 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads, latency)` / `T.SimdVF(latency)` | 已核验（2026-10-01） |
| V8 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19` | `T.annotate_buffer_versions(buffer_versions_map)` | 已核验（2026-10-01）：逐行读 |

“已核验（2026-10-01）”的含义：在锁定工作树上用 `rg -n` / `sed -n` 定位到定义行
并读完签名；**这不是运行验证**。本章没有在 v0.1.15 上运行任何 TileKernels kernel
（TileKernels 依赖 Python ≥ 3.12 / torch ≥ 2.13，本机是 3.11 / 2.1.1，
且没有 GPU/NPU）。

---

## 六、其它 corpus 引用

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:9-23` | https://mp.weixin.qq.com/s/X41mKH4Ds-VXUAnK6M8Eww | 2026-09-30 的官方发布清单（本次开源的六个组件） | 已核验（本地全文） |
| R2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` | — | 13 个仓库快照的 commit 与抓取日期 | 已核验（本地全文） |
| R3 | `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/examples/aot/sinkhorn_demo/pto_sinkhorn_demo_zh.md:1-20` | https://github.com/huawei-csl/pto-dsl | 第三方教程：曾是把 mHC 指向论文的**唯一线索**（现在该论文已入库，见 §三·五），并给出 Sinkhorn 的通俗解释 | 已核验（本地全文）；它指认的 arXiv:2512.24880 **已由 §三·五 的一手论文证实**，降级为旁证 |
| R4 | `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/examples/aot/sinkhorn_demo/sinkhorn_k4_builder.py`（存在性） | 同上 | PTO 版的同一条 Sinkhorn（本章只作旁证，未逐行读） | 已核验（静态）：文件存在；**未逐行核对** |
| R5 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/examples/deepseek_v4/hc_split_sinkhorn.py:1-40`（存在性） | https://github.com/tile-ai/tilelang-ascend | 适配器仓库里 hc split + sinkhorn 的昇腾版本（本章只作旁证） | 已核验（静态）：读过开头；**未逐行核对** |

---

## 七、本章的数值结论分别来自哪里

杜绝“来源不明的数字”：

| 数字 | 出现在 | 出处 | 性质 |
| --- | --- | --- | --- |
| `max abs(gate_naive − gate_fused) = 1.19e-07`、三累加器、`23040/10240/4096 B` | §2.2、§3.1、§3.5 | `code/01-engram-gate-shapes/engram_gate_shapes.py` 的实际输出 | **本书复算（CPU、torch 2.1.1、fp32）** |
| mHC Sinkhorn 的 `4.563e-01 / 5.696e-03 / 1.431e-06`、`eps` 下限、Engram L2 交替的 `1.1548` / `alpha → 2.0` | §5.2–5.4 | `code/02-sinkhorn-iterations/sinkhorn_iterations.py` 的实际输出 | **本书复算（CPU）** |
| mHC 的 `24 = 2×4 + 4×4`、`row/col sums = 0.999999`、`layer_input` / `new residual` 的形状 | §6.2–6.3 | `code/03-mhc-mix-flow/mhc_mix_flow.py` 的实际输出 | **本书复算（CPU）** |
| `16` 行/token/层、表 25.8 万 vs 1000 万槽的取行数不变、`-1` 标记 | §2.3 | `code/04-engram-hash-lookup/engram_hash_lookup.py` 的实际输出 | **本书复算（CPU）** |
| U 型曲线、`1.7248 → 1.7109`、`ρ ≈ 75%–80%`、`20%–25%`、幂律、MMLU +3.4 / BBH +5.0 / HumanEval +3.0 / MATH +2.4、NIAH 97.0 vs 84.2、`< 3%` 卸载开销 | §1.3、图 6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md`（P13–P17） | **论文自报，作者未复现** |
| 词表压缩 23% | §1.2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:158-166`（P5） | **论文自报，作者未复现** |
| HC prenorm GEMM 的 `6.7–136.7 us`、`3–82 TFLOPS`、`519–3463 GB/s` | §6.6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:166-176`（D3） | **厂商自报，作者未复现**；条件：Ascend 950，M/N/K 见表 |
| 迭代次数 `repeat = 10`、`eps = 1e-6`、`token_block_size = 8/4`、容差 `8e-3` | §5.2–5.4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_sinkhorn.py:19`、`tests/engram/test_engram_gate_fwd.py:70` | 仓库测试里的**默认值**，非实测 |
| mHC：`n = 4` 时 **6.7%** 训练开销；27B 最终 loss **降低 0.021**；相对 HC 的 **BBH +2.1% / DROP +2.3%** | §6.6、§6.7 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:219,411,683,685`（M4/M13/M23） | **论文自报，作者未复现**；条件：DeepSeek-V3 风格 MoE、27B、数据量与参数量成比例；**论文未在文本里给出加速器型号** |
| mHC：HC 的 Amax Gain 峰值 **≈3000** → mHC 复合后 **≈1.6**；Sinkhorn `t_max = 20`；`α` 初始化 **0.01**；Layer Norm ε **1e-20** | §5.1、§6.7 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:288,711`（M6/M25）与 Table 5 `:1360-1378`（M26） | **论文自报，作者未复现**；条件：27B 模型、`n = 4` |
| mHC 论文的 Table 2 与 §4.3.1 的访存记账（`(5n+1)C+n²+2n` 等；融合后读 `(3n+1)C→(n+1)C`、写 `3nC→nC`） | §6.1、§6.3 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:298-343`（M7）、`:501`（M18） | **论文的解析记账**（元素个数），不是实测 |

**没有来源的、或只有宣传口径的数字一律未采用。** 任何需要真实加速器才能得到的
数字（kernel 延迟、吞吐、利用率）在本章中均未出现，除了明确标注为
“论文自报 / 厂商自报”的引用。

---

## 八、待补资料

1. ~~mHC 没有论文~~ → **已解决（2026-10-01）**。corpus 已补入
   `14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`（arXiv:2512.24880 v1）
   与同名 PDF，`sources.yaml` 里有 `paper-deepseek-mhc` 条目（arXiv、trust: primary）。
   本章 §6 已改为“以论文 §4 为定义、以 TK 实现为参考”，证据级别从
   “源码 → 行为推断”升级为“论文 + 实现”，并列出四处不一致（见 §三·五）。
   唯一遗留的小问题：`sources.yaml` 的 note 里写“PDF 为 `arxiv-2512-24880.pdf`”，
   而磁盘上的文件名是 `paper-deepseek-mhc.pdf`，**建议后续版本把这条 note 改成实际
   文件名**，以免读者按 note 去找找不到。
2. **`engram_sinkhorn_*` 的实际调用方缺失。**
   TileKernels 仓库里没有任何代码调用这四个 kernel（只能搜到导出与测试）。
   因此“它作用在哪张矩阵上、在优化器里哪一步”本章只能从
   `momentum → step → step_reduce → finalize` 的签名与形状推断。
   建议补：一个真实的 Engram 训练脚本，或作者对这四个算子用途的说明。
3. **Engram 前向的实测基线缺失。**
   `tests/engram/test_engram_gate_fwd.py` 里有 benchmark 入口与 `count_bytes` 口径，
   但本机没有 SM90/SM100 或 Ascend 950，跑不出任何数字。
   建议补一次真机 benchmark 输出（含 `--run-benchmark` 的报告文件）。
4. **`multilayer_recompute` 的真实节省量无法核对。**
   `tests/mhc/test_multilayer_recompute.py:99-108` 给的是**记账模型**，
   没有“融合后墙钟快了多少”的真实数据。建议补一次开启/关闭重计算的
   显存与时间对照。
5. **`pre_big_fuse` 的 CUDA 与 Ascend 版本是否真的 bitwise 一致，未核验。**
   昇腾版的注释这么声明（T67），但本章没有条件在两个后端上各跑一次并对比。
   建议补一条跨后端一致性测试的记录。
6. **Engram 论文的图表缺失。** corpus 只有文本抽取，`figures/` 下的
   scaling law 图、架构图与 case study 图都在 PDF 里。图 6 因此只能用
   论文正文的数字，不能复刻原图的形状。建议补 PDF 中关键图的截图
   （带来源与许可标注）。
7. **第 33 章与本文对 `tile_kernels/` 的引用可能有重叠。** 若第 33 章已把
   `*_kernel/_cuda/_asc` 的三段式结构讲透，本章 §3.4 的对照表可与它对齐
   口径（本章只用“一份算法两个后端”的说法，未重新定义结构）。
   **（2026-10-01 更新）** 第 33 章已写完，§3.4 现在显式引用
   “第 33 章 §2–§4”并复用它的维度口径（网格/并行单位/数据放哪/归约表达/
   流水/缓冲版本/向量宽度），本条关闭。
8. **`L_r`（重算块长）由谁决定，未核验。**
   论文式 (20) 给出最优块长与“与流水级对齐”的要求（M20），但 TileKernels 的
   `multilayer_recompute` 只接受现成的列表；本章没有找到库里按式 (20) 自动选
   `L_r` 的代码，也无法确认训练脚本是否这么做。建议补：一个真实训练脚本里
   构造 `pre_mix_list` / `layer_output_list` 的片段（含 `L_r` 的来源）。
9. **论文的硬件条件缺失。** §5 的收益（0.021 loss、BBH/DROP、6.7% 开销）
   在文本抽取里没有出现任何加速器型号，也没有并行规模/序列长度的完整口径
   （只能从 §5.1 与 Table 5 读到模型规模与部分超参）。建议补：论文正文/附录里
   关于硬件与并行配置的段落（若有），以便把“论文自报”升级为“可复现的测量条件”；
   在此之前，本章**不会**把任何 mHC 数字往昇腾上迁移。

---

## 九、引用规则提醒

* 源码引用一律写成完整 corpus 路径加行号，例如
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:110-126`
  或 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:166-176`；
  锁定版本的 TileLang API 写成 `_vendor/tilelang-v0.1.15/<相对路径>`。
  （本章早期草稿里用 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/` 做过缩写，正式稿已全部展开为完整路径。）
* 论文数字一律带行号并标注“论文自报”与实验条件：
  Engram 论文写 `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:<行号>`，
  mHC 论文写 `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:<行号>`；
  **不要把 mHC 论文在未具名加速器上的收益说成昇腾上的收益**（见“待补资料”第 9 条）。
* 本章所有“某 kernel 怎么切任务 / 怎么同步”的结论都在 corpus 快照上核验过行号；
  **“已核验（静态）”≠“运行验证”**。
* 本章没有任何来自本机加速器的实测数字。
