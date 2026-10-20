# 图 1：DeepSeek 昇腾全栈的组件分层

**用途**：正文 §3.1–§3.3。

```text
[fig-01] DeepSeek 昇腾全栈：组件的分层位置

┌──────────────────────────── 语言与编译 ────────────────────────────┐
│ TileLang 昇腾版      ↵ 对 Ascend C 底层指令封装，高级语言，不损失性能 │
└────────────────────────────────────────────────────────────────────┘
┌───────────────────────────── 运行时 ───────────────────────────────┐
│ DeepJIT              ↵ 运行时编译 / 缓存 / 装载 / 启动（两平台共用）│
└────────────────────────────────────────────────────────────────────┘
┌─────────────────────── 计算库与算子库 ─────────────────────────────┐
│ DeepGEMM-Ascend      ↵ GEMM / MQA logits / MegaMoE（Ascend C 为主） │
│ FlashMLA             ↵ DSA 稀疏 prefill / decode                   │
│ DeepSelect           ↵ TopK 筛选                                   │
│ TileKernels          ↵ MoE 路由 / 量化 / RoPE / Engram / mHC       │
└────────────────────────────────────────────────────────────────────┘
┌───────────────────────────── 通信 ─────────────────────────────────┐
│ DeepEP-Ascend        ↵ EP dispatch / combine，PP / Bucket / Engram │
└────────────────────────────────────────────────────────────────────┘
```

## 要点

* **只有 TileKernels 是纯 TileLang 的**（每个算子一份 `*_kernel.py` 共享层 +
  `*_cuda.py` / `*_asc.py` 两个后端）。
* **DeepGEMM-Ascend 是 Ascend C 为主、TileLang 补位**：整库只有 mHC prenorm 那一个
  算子用 `tilelang.ascend.language` 写（`deep_gemm/tilelang_ops/hc_prenorm.py`）。
* **FlashMLA / DeepSelect / DeepEP-Ascend 在这份快照里不用 TileLang**：前两者是
  Ascend C / CUDA / PTX，后者是 Ascend C + HCCL/HCOMM/UBMEM/URMA。
* **DeepJIT 是跨层的共享件**：DeepGEMM-Ascend 与 DeepEP-Ascend 的 kernel 都由它在
  运行时编译。

来源：`../../../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:9-13`；
各仓库 README（见 `../references.md`）。
