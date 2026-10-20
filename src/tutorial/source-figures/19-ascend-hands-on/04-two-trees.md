# 图 4：两棵树的选择

**用途**：正文 §7.1、§7.6。

```mermaid
flowchart TD
    A[你要在昇腾上写算子] --> B{目标设备与生态?}
    B -->|Ascend 950 / 想用自动调度| M[M 树<br/>tilelang.ascend.language<br/>target=ascend]
    B -->|A2 / A3 / 要对接 Ascend C 生态| AD[A 树<br/>tilelang.language<br/>target=ascendc 或 pto]
    B -->|无 NPU，只做功能验证| SIM[A 树 + target=pto<br/>A5 camodel 仿真]
    M --> MAPI["T.alloc_l1 / l0a / l0b / l0c<br/>T.dual_copy<br/>T.SimtVF / T.SimdVF<br/>T.annotate_buffer_versions<br/>AutoSchedule 默认开"]
    AD --> AAPI["T.alloc_L1 / L0A / L0B / L0C / alloc_ub<br/>T.gemm_v0<br/>T.barrier_all / set_flag / wait_flag<br/>pass_configs: TL_ASCEND_*"]
    MAPI --> X[不要在一份 kernel 里混用两棵树的 API]
    AAPI --> X
```

## API 名称差异（最容易踩的坑）

```text
                     A 树（适配器）                M 树（主干）
  前端导入           tilelang.language             tilelang.ascend.language
  JIT target         "ascendc" / "pto"             "ascend"
  UB 分配            T.alloc_ub                    T.alloc_shared
  L1                 T.alloc_L1                    T.alloc_l1
  L0A / L0B / L0C    T.alloc_L0A/L0B/L0C           T.alloc_l0a/l0b/l0c
  子核               (cid, vid)                    T.MixedKernel(N, sids=2) -> (bx, sid)
  矩阵乘             T.gemm_v0                     T.gemm(..., unit_flag_ctrl=...)
  双 AIV 拷贝        手写 vid 切分                 T.dual_copy
  设备打印           T.printf / T.dump_tensor      T.print（无 dump_tensor）
  配置开关           TL_ASCEND_*（PassConfigKey）   TL_ENABLE_AUTO_SCHEDULE
  do_bench 后端      "msprof"                      "msprof" / "msprof_detail"
```

来源：两棵树的源码逐条比对（见 `../references.md` 的 A1–A8、S1–S12）。
