# 图 5：昇腾侧组件的硬件与版本要求

**用途**：正文 §8.1、§8.5。

```text
[fig-05] 昇腾侧组件的硬件与版本要求

  组件              昇腾硬件              工具链 / 版本           状态（本快照）
  ────────────────────────────────────────────────────────────────────────────
  TileLang 昇腾后端  Ascend 950            CANN + bisheng         随 tag 发布
  TileKernels        Ascend 950 NPU        CANN 9.2.0+            已发布（双后端）
  DeepGEMM-Ascend    Ascend 950 系列       CANN 9.20 + torch_npu  已发布
  FlashMLA（昇腾）   Ascend 950 NPU        CANN 9.2.0+            稀疏 prefill / decode
  DeepSelect（昇腾） Ascend NPU            见 README              仅 bf16 / int32
  DeepEP-Ascend      Ascend 950 (A5)       CANN + HCCL/HCOMM      EP 已发布；其余实验性
  DeepJIT            跟随使用方            bisheng + ld.lld + ACL 双后端
```

## 要点

* **硬件是 950 级别**：不要假设 A2 / A3 也能跑；DeepEP-Ascend 明确说其它代际与 CANN
  版本“has not been established by these measurements”。
* **CANN 版本以各 README 为准**：9.2.0（TileKernels、FlashMLA）与 9.20
  （DeepGEMM-Ascend）都出现在原文里，不要互相套用。
* **状态分三档**：已发布（有性能表）、实验性（接口存在但标 experimental）、
  未实现（只有 API 预留），DeepEP-Ascend 的 `Ongoing` 一节是最好的例子。
* **kernel 在运行时编译**：环境里必须一直有 CANN / Bisheng，不是装完就能删。

来源：各仓库 README 的 `Requirements` / `News` / `Ongoing`，见 `../references.md`
与正文 §8.1–§8.3。
