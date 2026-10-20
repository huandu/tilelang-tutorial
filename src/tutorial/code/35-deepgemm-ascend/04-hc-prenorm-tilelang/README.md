# 示例 04：hc_prenorm 的 TileLang 骨架（仅静态核对）

## 目的

正文 §6 逐段读的那个 kernel。本文件从
`DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py` 提炼了一份**可读骨架**：

* `get_config()`：三个 2 的幂约束取最小，决定 K 方向切几份（原文件 `:13-57`）；
* `T.MixedKernel` + `T.alloc_shared/l1/l0a/l0b/l0c` +
  `T.annotate_buffer_versions`（原文件 `:105-131`）；
* `T.Pipelined`（隐藏维流水）+ `T.Serial`（token 维）+ `T.SimdVF`（展宽/NZ/平方和）+
  `T.dual_copy` + `T.gemm`（原文件 `:136-292`）；
* `T.PerCoreTask` 里的两段 store 与核间同步、`T.set_atomic` 的 K-split 合并
  （原文件 `:133-134`、`:294-359`）。

提炼时保留了**结构与注释**，去掉了与单卡无关的细节。**它不是可以直接跑的替代品**；
真机请调用原文件导出的 `deep_gemm.tf32_hc_prenorm_gemm`。

## 运行前提

* 昇腾 NPU + 已安装的 TileLang 主干 `v0.1.15`
  （前端是 `tilelang.ascend.language`，即第 19/25 章说的 **M 树**）；
* 已安装 `DeepGEMM-Ascend`（骨架用它的 `_C.get_num_sms()` 拿核数，
  与原文件 `:26` 一致）。

## 运行命令

```bash
python3 -m py_compile hc_prenorm_sketch.py     # 语法自检（本机可做）

# 真机上：直接调用原文件导出的算子，而不是本骨架
python3 -c "import deep_gemm, torch; \
  a = torch.randn(512, 28672, dtype=torch.bfloat16, device='npu'); \
  b = torch.randn(24, 28672, dtype=torch.float32, device='npu'); \
  d = torch.empty(512, 24, dtype=torch.float32, device='npu'); \
  s = torch.empty(512, dtype=torch.float32, device='npu'); \
  deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s); print(d.shape, s.shape)"
```

期望输出：`torch.Size([512, 24]) torch.Size([512])`。
数值判据见官方 `tests/test_hyperconnection.py:71-94`
（`d` 对 `a.float() @ b.T`、`sqr_sum` 对 `a.float().square().sum(-1)`，
`calc_diff < 1e-6`；`use_deterministic_algorithms(True)` 时要求逐位可复现）。

**本机没有 NPU，以上命令未运行。**

## 作者实际做到的验证级别

**仅静态核对**：

* `python3 -m py_compile hc_prenorm_sketch.py` 通过；
* 用到的每个原语逐个对照 M 树 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/`
  与 `.../tilelang/layout/`（`T.MixedKernel`、`T.alloc_shared`、`T.alloc_l1`、
  `T.alloc_l0a/l0b/l0c`、`T.annotate_buffer_versions`、`T.Pipelined`、
  `T.dual_copy`、`T.gemm`、`T.SimdVF`、`T.PerCoreTask`、`T.set_atomic`、
  `T.ascend_sync_inter_arrive/wait`、`make_ascend_compact_nz_layout`）；
* 结构逐段对照原文件行号（见上面的“目的”）；
* **未编译、未在 NPU 上运行**。

## 已知局限

* 骨架里 `num_store_phases` / `num_always_synced_store_phases` 的推导被简化；
  原文件 `:95-103` 有完整解释（包括“AutoSchedule 不接受 `T.PerCoreTask` 里
  条件控制流下的核间等待”这一条实现约束）。
* 骨架省略了尾块与 `aiv_id` 相关的细节；真要改这个 kernel，请以原文件为准。
* 物理缓冲区按 `block_mhc_mult3` 对齐、逻辑范围按 `mhc_mult3` 收窄——
  这个“物理大于逻辑”的约定在骨架里保留并标注（原文件 `:158-163` 有同样注释）。

