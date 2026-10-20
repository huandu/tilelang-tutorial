# 图 02 · `T.copy` 的降级判定链

```mermaid
flowchart TD
    S["T.copy / T.async_copy / T.tma_copy"] --> P1{"1. gather4 / scatter4 标记?"}
    P1 -->|是| R1["kBulkLoadGather4 / kBulkStoreScatter4<br/>(无回退)"]
    P1 -->|否| P2{"2. cluster_mask != 0?"}
    P2 -->|是| R2["必须 TMA multicast，否则报错"]
    P2 -->|否| P3{"3. T.tma_copy?"}
    P3 -->|是| R3["必须 TMA，否则报错"]
    P3 -->|否| P4{"4. T.async_copy?"}
    P4 -->|是| R4["必须 cp.async，否则报错"]
    P4 -->|否| P5{"5/6/7. prefer_instruction?"}
    P5 -->|tma| R5["TMA；与 disable_tma 冲突则报错"]
    P5 -->|cp_async| R6["cp.async；与 enable_async_copy=False 冲突则报错"]
    P5 -->|sync| R7["LDSM / STSM / TMEM / Normal"]
    P5 -->|未指定| P8{"8. 先试 TMA store"}
    P8 -->|成功| R8["kBulkStore / kBulkStore1D"]
    P8 -->|失败| R9["SelectSyncLikeInst:<br/>LDSM → STSM → TMEM → Normal"]
```

ASCII 版：

```text
  ①  gather4 / scatter4 ─────────────▶ 强制 TMA gather/scatter（无回退）
  ②  cluster_mask != 0 ─────────────▶ 必须 TMA multicast
  ③  T.tma_copy ────────────────────▶ 必须 TMA
  ④  T.async_copy ──────────────────▶ 必须 cp.async
  ⑤  prefer_instruction="tma" ──────▶ TMA（冲突即报错）
  ⑥  prefer_instruction="cp_async" ─▶ cp.async（冲突即报错）
  ⑦  prefer_instruction="sync" ─────▶ 同步形态
  ⑧  默认 ──────────────────────────▶ 先试 TMA *store*；不行再走
                                      LDSM → STSM → TMEM → Normal
```

**来源**：`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-808`
（`SelectCopyInstForLowering`）、`:593-625`（`SelectTmaInst`/`SelectSyncLikeInst`）、
`src/cuda/op/copy.h:21-33`（`CopyInst` 枚举）。

**两条容易记错的规则**：

1. 默认路径只对 **store** 方向尝试 TMA；global→shared 的 load 不写提示就不会选 TMA。
2. 显式 `T.async_copy` / `T.tma_copy` 是**硬约束**：不满足就编译失败，不会静默回退。

**边界条件**：TMA bulk load 要求最后一维位宽是 128 的整数倍
（`copy_analysis.cc:195-225`）；cp.async 要求 global→shared 且 dtype 相同
（同文件 `:427-437`）。
