# 第 16 章 参考资料

引用格式遵循 本书的引用格式约定（§3.3）：源码引用写成
`_vendor/<版本>/<相对路径>:行号`，资料引用写成 corpus 路径。每条都标注了核验
状态；“已核验”的源码引用由 `code/04-moe-kernel-map/kernel_map.py`、
`code/05-sparse-index-map/sparse_index_map.py`、`code/06-ascend-map/ascend_map.py`
实际打开文件逐行核对过（合计 56 处，全部命中）。

版本：TileLang `v0.1.15`（commit `a35f8dd`）、TileLang-Ascend
`v0.1.2.000-release`（commit `89a9364`）。核验日期：2026-10-01。

---

## S · 源码引用（锁定版本）

### 主干 TileLang v0.1.15

| id | 引用 | 说明 | 核验 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/tileop/base.py:5-39` | `GemmWarpPolicy`：`Square=0` / `FullRow=1` / `FullCol=2` | 已核验（2026-10-01） |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160-169` | `T.gemm` 签名（`policy` 默认 `Square`，`clear_accum` 默认 False） | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py:12-90` | 共享专家的 `moe_forward_tilelang_shared`：gate/up 双 GEMM + SiLU 融合 + down GEMM | 已核验 |
| S4 | `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py:92-230` | 路由专家的 `moe_forward_tilelang_routed`：第 120 行 grid 大小、第 157 行 `group_idx_for_bx`、第 161/189 行 `actual_rows` 掩码 | 已核验 |
| S5 | `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py:362-413` | host 侧路由：`group_padded_offsets`（第 366 行）、两条 stream（第 385 行）、`scatter_reduce`（第 406 行） | 已核验 |
| S6 | `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py:476-482` | `main()` 默认形状：`d_hidden=7168, d_expert=2048, n_routed_experts=8, n_experts_per_token=4, seq_len=8192` | 已核验 |
| S7 | `_vendor/tilelang-v0.1.15/examples/grouped_gemm/example_grouped_gemm_fwd.py:64-104` | 拼接式 grouped GEMM：`total_m_blocks`（64）、偏移表反解（85-92）、`actual_rows` 写回（104） | 已核验 |
| S8 | `_vendor/tilelang-v0.1.15/examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py:32,38,48-68` | 指针表版：`copy_num_stages = 1`（32）、`T.make_tensor`（63）；注释说明 ptr-backed 张量的多级流水尚不正确 | 已核验 |
| S9 | `_vendor/tilelang-v0.1.15/examples/grouped_gemm/example_grouped_gemm_bwd.py` | grouped GEMM 的反向（本章未展开，供延伸） | 已核验（文件存在） |
| S10 | `_vendor/tilelang-v0.1.15/examples/deepseek_mla/example_mla_decode.py:14-120` | MLA decode：第 20 行 `kv_head_num == 1` 断言、24-29 行张量签名（`dim=512`、`pe_dim=64`）、30-31 行 `glse`/`Output_partial`、65-66 行两次 `T.gemm(..., FullCol)` 累加到同一 `acc_s`、83 行 PV、88-113 行 split 合并 | 已核验 |
| S11 | `_vendor/tilelang-v0.1.15/examples/deepseek_mla/example_mla_decode.py:178-180,218-226` | `num_split > 1` 选择 `main_split`/`main_no_split`；`main()` 默认 `heads=128, kv_ctx=8192, dim=512, pe_dim=64` | 已核验 |
| S12 | `_vendor/tilelang-v0.1.15/examples/deepseek_mla/{example_mla_decode_paged.py,example_mla_decode_persistent.py,example_mla_decode_ws.py,experimental/example_mla_decode_kv_fp8.py}` | paged KV / persistent / warp-specialized / fp8 KV cache 四个变体 | 已核验（文件存在） |
| S13 | `_vendor/tilelang-v0.1.15/examples/deepseek_mla/README.md` | 官方 MLA 教程（与 corpus R1 同源）：576/512 的 head 维、`wgmma` M≥64、按 `dim` 切 `acc_o`、layout inference、`T.use_swizzle`、warp specialization、split-KV | 已核验（文件存在） |
| S14 | `_vendor/tilelang-v0.1.15/examples/deepseek_v32/README.md:134,181-213` | 第 134 行 Sparse MLA pipelined “close to 600 TFlops on H800 SXM”；第 189 行 fp8 退回 `mma.sync` 慢 1.81×；第 212-213 行 25.79 / 14.24 / 13.77 ms 与“fp8 不比 bf16 快”的结论 | 已核验 |
| S15 | `_vendor/tilelang-v0.1.15/examples/deepseek_v32/fp8_lighting_indexer.py:90-167` | lightning indexer：第 155 行相似度 GEMM、第 165-167 行 ReLU + 按头加权 + `reduce_sum` | 已核验 |
| S16 | `_vendor/tilelang-v0.1.15/examples/deepseek_v32/topk_selector.py:11-120` | `convert_to_uint16`（保序映射）、直方图 + 两遍 8-bit 基数排序、输出位置而非值 | 已核验 |
| S17 | `_vendor/tilelang-v0.1.15/examples/deepseek_v32/sparse_mla_fwd.py:30-34,118-122` | 断言 `is_causal` 与 `topk % block_I == 0`（第 33-34 行）；按 `Indices` gather KV（第 120 行） | 已核验 |
| S18 | `_vendor/tilelang-v0.1.15/examples/deepseek_v32/inference/config_671B_v3.2.json` | DeepSeek-V3.2-671B 推理配置：`n_layers=61`、`n_heads=128`、`n_routed_experts=256`、`n_activated_experts=8`、`kv_lora_rank=512`、`qk_rope_head_dim=64`、`index_n_heads=64`、`index_head_dim=128`、`index_topk=2048` | 已核验（JSON 解析后逐字段比对） |
| S19 | `_vendor/tilelang-v0.1.15/examples/deepseek_nsa/example_tilelang_nsa_fwd.py:18,29,50,80` | NSA 前向：第 18 行签名（`block_size=64`、`selected_blocks=16`）、第 29 行 `block_indices_shape`、第 50 行 `BlockIndices`、第 80 行块内掩码 `i_s <= i_t` | 已核验 |
| S20 | `_vendor/tilelang-v0.1.15/examples/dsa_hisa/README.md:1-40` | HISA 两级层级（先 pool block、再 block 内 token）与四个 kernel 的分工 | 已核验 |
| S21 | `_vendor/tilelang-v0.1.15/examples/topk/example_topk.py:18-50` | 小规模 top-k：`for k in T.serial(topk)` 反复取最大值 | 已核验 |
| S22 | `_vendor/tilelang-v0.1.15/examples/flash_decoding/example_gqa_decode_varlen_logits.py:39-83` | varlen decode：`cu_seqlens_k` 限定每个 batch 的 K 范围 | 已核验 |
| S23 | `_vendor/tilelang-v0.1.15/examples/deepseek_v4/{act_quant.py,fp8_fp4_gemm_1d1d_sm100.py,sparse_attn_fwd_sm90.py}` | V4 相关：FP8 block 量化、block-scaled GEMM、sparse attention | 已核验（文件存在） |
| S24 | `_vendor/tilelang-v0.1.15/examples/deepseek_mhc/{example_mhc_pre.py,example_mhc_post.py,example_mhc_bwd.py}` | mHC（Manifold-Constrained Hyper-Connections）的 pre/post/bwd：残差流的多路混合（`hc_mult`、`post_mix`、`comb_mix`、Sinkhorn） | 已核验（第 11 行 kernel 签名、第 31 行 `hc_mult3`） |

### 昇腾适配器 TileLang-Ascend v0.1.2.000

| id | 引用 | 说明 | 核验 |
| --- | --- | --- | --- |
| S25 | `_vendor/tilelang-ascend-v0.1.2.000/examples/fusedmoe/example_fusedmoe.py:1-33` | 策略声明（纯 GEMM kernel + PyTorch 后处理）与 `pass_configs`（`TL_ASCEND_AUTO_CV_COMBINE` 等，第 30-32 行） | 已核验 |
| S26 | `_vendor/tilelang-ascend-v0.1.2.000/examples/fusedmoe/example_fusedmoe.py:39,89,146,212,324,780,886` | kernel 家族：single / dual / grouped / grouped-dual / `T.mma` developer 版 / `*_fixed` persistent 版 | 已核验 |
| S27 | `_vendor/tilelang-ascend-v0.1.2.000/examples/fusedmoe/example_fusedmoe.py:163,228-230,961,1163,1184` | `block_metadata[total_m_blocks,3]`（163）、L1/L0C 容量注释（228-230）、作者自评的 21.28ms（961）、`_build_block_metadata`（1163）、`host_preprocess`（1184） | 已核验 |
| S28 | `_vendor/tilelang-ascend-v0.1.2.000/examples/moe_token_permute/moe_token_permute.py:17-90` | token permute/unpermute：UB 直方图与前缀和（第 79 行起）、`workspace_idx`（第 38 行）、`target="pto"` | 已核验 |
| S29 | `_vendor/tilelang-ascend-v0.1.2.000/examples/dispatch_combine/{README.md,dispatch_combine_shmem.py}` | 跨 rank dispatch/combine：`win_data`/`win_status`、`T.shmem_ub_put_nbi`（第 148 行）、`ep_receive_count`（第 68 行）、`expert_token_nums_out`（第 254 行）；README 给出 A3/CANN 8.2+/不支持 graph 模式等前提 | 已核验 |
| S30 | `_vendor/tilelang-ascend-v0.1.2.000/examples/deepseek_v4/{lightning_indexer.py:575,sparse_flash_mla/sparse_flash_mla_kernel.py:1199,sparse_flash_mla/bench_mark.md,hc_split_sinkhorn.py:27}` | V4 的 lightning indexer（L1/L0C）、sparse flash MLA（swa/hca/csa）、mHC 的 Sinkhorn | 已核验 |
| S31 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:20,71,73,582` | 官方条目：V4 kernels（20）、Dispatch & Combine（71）、LightningIndexer / TopK Selector（73、582） | 已核验 |

---

## R · Corpus 资料

| id | 引用 | 说明 | 核验 |
| --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/deepseek_mla.md` | 官方 MLA 教程全文：576 = 512 + 64、`acc_o` 过大导致寄存器溢出、按 `dim` 切 `acc_o`、warpgroup 交换 `acc_s`、layout inference、`T.use_swizzle`、split-KV、以及 “around 80 lines of Python code” | 已核验（2026-10-01，全文 195 行整篇读过） |
| R2 | 同上，AMD MI300X 一节 | MI300X 64KB shared、把 Q 改放寄存器 fragment、降低流水级数；bank conflict 策略随硬件变化 | 已核验 |
| R3 | `../tilelang-corpus/12-operator-libraries/raw/arxiv-2412-19437.md` | DeepSeek-V3 技术报告（**corpus 内只有摘要**）：671B 总参数 / 37B 激活、MLA + DeepSeekMoE、auxiliary-loss-free 负载均衡 | 已核验（文件仅 12 行，摘要 + 元数据） |
| R4 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2502-11089.md` | NSA 论文摘要：三支（压缩/选择/滑窗）+ “arithmetic intensity-balanced algorithm design” + natively trainable | 已核验（corpus 内仅摘要） |
| R5 | `../tilelang-corpus/sources/sources.yaml`（`arxiv-2412-19437`、`arxiv-2502-11089`） | 资料库对这两篇的定位与 tags（`deepseek, mla, moe` / `deepseek, sparse-attention`） | 已核验 |
| R6 | `../tilelang-corpus/sources/sources.yaml`（`arxiv-2605-02568`、`arxiv-2605-07363`、`arxiv-2506-08889`） | 稀疏注意力/索引器的同族后续工作（StreamIndex、MISA、SeerAttention-R） | 已核验（条目存在） |
| R7 | `../tilelang-corpus/sources/sources.yaml`（`arxiv-2505-04519`） | Pangu Ultra MoE：华为侧大 MoE 在昇腾上训练的公开叙述（MoE 与通信/软件栈约束） | 已核验（条目存在，正文未读） |
| R8 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/ops/sequence_modeling/mhc.py:1-30` | mHC 的定义：“The pre-layer half of Manifold-Constrained Hyper-Connections (mHC)” | 已核验 |
| R9 | `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/answers/2031020927979406721.md` | 关于 DeepSeek V4 的**知乎分析（二手，非官方）**：V4-Pro 384 个 routed experts + 1 shared、每 token 激活 6 个、与昇腾 384 超节点的耦合猜测、CSA/HCA “近处看原文，远处看索引” | 已核验（相应段落读过）；**引用时已显式标注为非官方** |
| R10 | `../tilelang-corpus/90-synthesis/01-knowledge-system.md:104,409-410,467-476` | 资料库自身的章节映射：MoE 关注 dispatch/combine、grouped GEMM、token permute；MLA/DeepSeek 系列对应 `deepseek_mla/`、`deepseek_v32/`、`deepseek_v4/`、`deepseek_mhc/` | 已核验 |

---

## L · 上游链接（未在本机核验）

| id | 链接 | 状态 |
| --- | --- | --- |
| L1 | https://arxiv.org/abs/2412.19437 （DeepSeek-V3 技术报告） | 未核验（网络未取；corpus 内只有摘要） |
| L2 | https://arxiv.org/abs/2502.11089 （NSA） | 未核验（同上） |
| L3 | https://github.com/deepseek-ai/FlashMLA | 未核验（MLA 的手写实现，官方文档引它作对照） |
| L4 | https://github.com/tile-ai/tilelang-ascend | 未核验（昇腾适配器仓库；本机以 `_vendor/` 为准） |

---

## 待补资料

1. **DeepSeek-V2 技术报告全文**（arXiv 2405.04434）。MLA 的原始定义与
   “KV cache 减少多少”的官方数字都在这篇里，但 corpus 里**没有**这份全文。
   本章因此没有引用任何 V2 的数字，只用第一方配置与官方文档做推导。建议补来源。
2. **DeepSeek-V3 / V3.2 / V4 技术报告全文**。corpus 里的 `arxiv-2412-19437`
   只有摘要；V4 的专家数、激活数目前只有二手知乎分析（R9）。本章的结构参数
   改用锁定版本里的第一方配置（S18）替代，但“V4 有多少个专家”这类事实仍缺
   权威来源。
3. **NSA 全文**。corpus 内只有摘要，本章对 NSA 三支结构的描述主要来自
   TileLang 的 `deepseek_nsa` example 与其中的 Triton 参考实现。建议补全文。
4. **昇腾 dispatch/combine 的性能口径**。`dispatch_combine/README.md` 只给
   运行前提，没有 benchmark 数字；本章的通信账是纯模型输出（`code/06`）。
   若要写“通信 vs GEMM 谁更长”，需要一份带互联带宽的真机数据。

## 与本章的引用一致性自查

* 正文出现的每条 `_vendor/...:行号` 都可以用 `code/04/`、`code/05/`、`code/06/`
  的三个脚本复跑核对（合计 56 处，0 处未命中）。
* 本章不使用任何随时间漂移的版本表述；凡涉及版本内行为的结论都钉死在
  `v0.1.15` 或 `v0.1.2.000-release` 这两个 tag 上。
* 本章**没有**本机实测的性能或正确性数字；所有数字都在正文里标了来源类别。
