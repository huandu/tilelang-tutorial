# 图 01 · 一个原语的三段式生命周期

```mermaid
flowchart LR
    A["用户代码<br/>T.copy(A[i,j], A_s)"] --> B["前端函数<br/>tilelang/language/copy_op.py<br/>校验 + extent 归一化"]
    B --> C["TIR<br/>call_intrin(tl.tileop.copy,<br/>src_region, dst_region, annotations)"]
    C --> D["IR 节点<br/>CopyNode{src,dst,src_range,dst_range,annotations}"]
    D --> E["后端指令选择<br/>SelectCopyInstForLowering"]
    E --> F1["SIMT 循环"]
    E --> F2["cp.async"]
    E --> F3["TMA cp.async.bulk.tensor"]
    E --> F4["ldmatrix / stmatrix / tcgen05.ld"]
    D -.昇腾.-> G["tl.tileop.ascend_copy"]
    G --> H["DataCopy / ND2NZ / LoadData / Fixpipe"]
```

ASCII 版：

```text
  用户代码              前端                   TIR / IR 节点              后端
  ────────              ────                   ─────────────              ────
  T.copy(...)  ──▶  校验 + extent 归一化  ──▶  tl.tileop.copy        ──▶  选指令
                    (标量 BufferLoad 对      CopyNode{src, dst,          ├─ SIMT
                     直接退化成 BufferStore)  src_range, dst_range,      ├─ cp.async
                                              annotations}               ├─ TMA
                                                                         └─ LDSM/TMEM
                    昇腾路径：tl.tileop.ascend_copy ──▶ DataCopy/ND2NZ/LoadData/Fixpipe
```

**来源**：`_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-130`、
`src/op/copy.cc:615`、`src/cuda/op/copy_analysis.cc:716`。

**要点**：Python 只在最左边一列起作用。你越往右看，能影响的只剩
`annotations` 里的那几个键。
