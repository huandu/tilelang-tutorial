# 图 3：一次训练 step 里各组件的位置

**用途**：正文 §5.1–§5.3。

```text
[fig-03] 一次训练 step 里各组件的位置

  ① MoE 路由          TileKernels（topk_gate，TileLang）
        │
  ② EP dispatch       DeepEP-Ascend（EPBuffer.dispatch）
        │
  ③ Grouped GEMM 升维 DeepGEMM-Ascend
        │
  ④ SwiGLU            TileKernels（融合量化，TileLang）
        │
  ⑤ Grouped GEMM 降维 DeepGEMM-Ascend
        │
  ⑥ combine           DeepEP-Ascend（EPBuffer.combine）
        │
  ⑦ mHC + 优化器      TileKernels（mHC）+ DeepGEMM-Ascend（HC prenorm）

  ②–⑥ 这一整段可以被 MegaMoE 融成一个 kernel；
  训练侧另有 DualPipe（流水）与 EPLB（专家负载均衡）参与组织。
```

## 要点

* **TileLang 直接覆盖 ①④⑦ 的一部分**（TileKernels 那几块），②③⑤⑥ 交给 Ascend C。
* **MegaMoE 把 ②–⑥ 融合**：DeepGEMM-Ascend 的 README 写它“fuses EP dispatch, two
  grouped GEMMs, SwiGLU, and combine”，所以这条链上的边界是可以移动的。
* **通信是异步的**：② 与 ⑥ 都是“发起 → 算别的 → `wait()`”的形式，见
  DeepEP-Ascend README 的 overlap 示例。
* **DualPipe / EPLB 不在这份发布清单里**，但 DeepEP-Ascend 为 EPLB 预留了
  `lb_prefetch_weights` / `lb_reduce_grads`（Ascend kernel 尚未实现）。

来源：`../../../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:153`、
`.../DeepEP-Ascend/README.md:81,120,175-193`、`.../TileKernels/README.md:13-19`。
