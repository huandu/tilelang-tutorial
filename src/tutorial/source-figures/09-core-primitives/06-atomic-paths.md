# 图 06 · 原子操作的两条路径

```mermaid
flowchart TD
    A["T.atomic_add(dst, value)"] --> Q{"能推出 extent 吗？"}
    Q -->|"两边都推不出（标量地址）"| S["tl.atomic_add_elem_op<br/>→ AtomicAdd / AtomicAddx2/4<br/>支持 return_prev、memory_order"]
    Q -->|"至少一边能推出（tile-region）"| R["tl.tileop.atomicadd<br/>→ per-element 原子<br/>或 use_tma=True 时的 TMA cp.reduce"]
    R --> R2["return_prev 不支持<br/>→ NotImplementedError"]
    R --> R3["use_tma dtype 白名单：<br/>float16 / bfloat16 / float32 /<br/>int32 / uint32 / uint64"]
    A --> AS["昇腾两条路：<br/>T.set_atomic('add', dtype) 打开硬件标志<br/>T.tile.atomic_add(dst_gm, src_local)"]
    AS --> AS2["set_atomic 有状态！<br/>必须配对 T.set_atomic_none()"]
```

**来源**：`_vendor/tilelang-v0.1.15/tilelang/language/atomic.py:264-311`；
`tilelang/cuda/language/atomic.py:12-50`；`src/cuda/op/atomic_add.cc:36-62`；
`tilelang/ascend/language/mode.py:63-105`；
`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py:313-341`；
`_vendor/tilelang-ascend-v0.1.2.000/src/op/ascend.cc:748-761`。

**要点**：判据是 `get_extent()` 能不能推出区域，不是“dst 有几个元素”。
`T.set_atomic` 是模式开关，`T.atomic_add` 是一次性原语，两者不等价。
