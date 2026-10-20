# 图 4：CUDA 与 Ascend 两条 pipeline 的阶段对照

来源：本章作者从 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py` 与
`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py` 逐行提取（脚本见
`../code/pass-order/dump_pass_order.py`）。完整的 74 / 80 步清单在 `chapter.md`
第 4.5、4.6 节；本图只画阶段。

## 4.1 阶段对照（左 CUDA，右 Ascend）

```mermaid
flowchart TB
    subgraph S0["共同起点"]
        A0["BindTarget"]
        A1["MaterializeKernelLaunch<br/>(两侧参数不同)"]
    end

    subgraph S1["共同：规范化与小 pass 前置"]
        B1["LetInline / AddWrapperForSingleBufStore /<br/>LegalizeNegativeIndex / VerifyParallelLoop /<br/>InjectAssumes / Simplify"]
        B2["CanonicalizeLegacyReducer / VerifyReducerEpoch /<br/>VerifyBufferInit"]
    end

    subgraph S2["分岔点 1：后端专属的前处理"]
        C1["CUDA:<br/>AutoWarpSpecialization → MaterializeWSSchedule →<br/>ProducerConsumerWarpSpecialized → LowerBlackwell2SM"]
        C2["Ascend:<br/>RewriteDualCopy → LegalizeSimdMerging"]
    end

    subgraph S3["分岔点 2：流水 / 布局 / 调度"]
        D1["CUDA:<br/>IfStmtBinding → UnrollLoop →<br/>PipelinePlanning → InjectSoftwarePipeline →<br/>LayoutInference → LowerTileOp"]
        D2["Ascend:<br/>UnrollLoopSkipVF → AscendLayoutInference →<br/>NormalizeAscendFractalStorage → InsertNd2Nz →<br/>AscendInsertOOBPadding → [13 步 AutoSchedule 段，L104-116] →<br/>AscendLowerTileOp"]
    end

    subgraph S4["共同：tile op 之后的标准 lowering"]
        E1["DecoupleTypeCast / LegalizeVectorizedLoop /<br/>LegalizeSafeMemoryAccess / LowerAccessPtr /<br/>HoistGlobalBufferAllocations / LowerOpaqueBlock /<br/>FlattenBuffer / ConfigIndexBitwidth / VectorizeLoop /<br/>LoopUnswitching / UnrollLoop / VerifyMemory"]
    end

    subgraph S5["分岔点 3：同步与存储合并"]
        F1["CUDA:<br/>LowerThreadAllreduce → LowerLDGSTG → LowerHopperIntrin →<br/>MarkCudaSyncCalls → MergeSharedMemoryAllocations →<br/>ThreadSync(shared) x2 → InjectTcgen05Fence"]
        F2["Ascend:<br/>LowerThreadAllreduce → AscendThreadSync(global) →<br/>MarkScalarDcacheBypass → MergeUBAllocations →<br/>RewriteFlagToBuf → AscendThreadSync(shared) x2"]
    end

    subgraph S6["共同收尾"]
        G1["MergeIfStmt → MakePackedAPI → Simplify →<br/>LowerDeviceKernelLaunch"]
        G2["CUDA 额外: PersistThreadblock"]
    end

    A0 --> A1 --> B1 --> B2 --> C1 --> D1 --> E1 --> F1 --> G1 --> G2
    B2 --> C2 --> D2 --> E1
    F2 --> G1

    classDef common fill:#e9f7ef,stroke:#2e7d5b
    classDef nv fill:#e8f0fe,stroke:#4a6fa5
    classDef asc fill:#fff4e5,stroke:#b8860b
    class A0,A1,B1,B2,E1,G1 common
    class C1,D1,F1,G2 nv
    class C2,D2,F2 asc
```

## 4.2 用文字说清三处关键差异

```text
差异 1  MaterializeKernelLaunch 的参数（pipeline.py:70 vs :45-51）
        CUDA:   lower_thread_binding=True,  default_threads=128
        Ascend: lower_grid_binding=True, lower_thread_binding=False,
                launch_dim_tags=["cthread"], unsupported_annotations=["cluster_dims"]
        => CUDA 有 threadIdx；Ascend 的 kernel 作用域没有线程，线程只存在于 T.SimtVF 内。

差异 2  调度发生在不同的位置（cuda/pipeline.py:127-144 vs ascend/pipeline.py:83-116）
        CUDA:   PipelinePlanning + InjectSoftwarePipeline 在 LayoutInference 之前。
        Ascend: layout 先推（AscendLayoutInference），再进 16 步 AutoSchedule（含 Z3 求解）。

差异 3  同步 pass 的归属与粒度（cuda/pipeline.py:258-266 vs ascend/pipeline.py:162-182）
        CUDA:   通用 ThreadSync("shared") / ThreadSync("shared.dyn")。
        Ascend: AscendThreadSync("global"/"shared"/"shared.dyn")
                + RewriteFlagToBuf（把 flag 归一化到 8 槽命名空间，必要时溢出到共享 mutex 池）。
```

## 4.3 一句提醒

图中的 `[13 步 AutoSchedule 段]` 不是无条件执行的：`ascend/pipeline.py:103` 用
`if auto_schedule_enabled:` 包住，`ascend/pipeline.py:169` 又在关闭时补一个
`InferBufferAliases`。所以 **pass 序列是一段带分支的程序，不是一张静态表**；
本章第 5 节专门讲这件事。
