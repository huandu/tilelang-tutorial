# 图 3：共享的 lowering 走廊与两处分岔（自绘）

把两条 pipeline 的 pass 序列并排放，会看到一个“工字形”结构：前后两端各有
一段各自特有的处理，中间是一段**双方共用的、顺序一致的标准 lowering 走廊**。

```text
                CUDA 特有段                               ASCEND 特有段
      ┌───────────────────────────┐           ┌──────────────────────────────────┐
      │ WarpSpecialization 三连   │           │ RewriteDualCopy / SimdMerging     │
      │ LowerBlackwell2SM         │           │ UnrollLoopSkipVF                  │
      │ ← 流水：PipelinePlanning  │           │ AscendLayoutInference             │
      │   + InjectSoftwarePipeline│  ══════▶  │ NormalizeAscendFractalStorage     │
      │ LowerSharedTmem/Barrier   │           │ InsertNd2Nz / AscendInsertOOBPadding│
      │ MergeSharedMemoryAlloc    │           │ ← 自动调度段：AutoSchedule…       │
      │ ThreadSync / MBarrier     │           │   InsertSync / MaterializeMultiBuf │
      │ PersistThreadblock        │           │ MergeUBAllocations / FlagToBuf     │
      └───────────────────────────┘           └──────────────────────────────────┘
                    │                                        │
                    ▼                                        ▼
      ┌───────────────────────────────────────────────────────────────────────┐
      │              共享走廊（39 个同名 pass，顺序基本一致）                  │
      │  BindTarget → MaterializeKernelLaunch → 前端合法化 → (流水/布局) →     │
      │  LowerTileOp → DecoupleTypeCast → LegalizeVectorizedLoop →             │
      │  LegalizeSafeMemoryAccess → LowerAccessPtr → HoistNonRestrictParams →  │
      │  HoistGlobalBufferAllocations → LowerOpaqueBlock → FlattenBuffer →     │
      │  ConfigIndexBitwidth → VectorizeLoop → LoopUnswitching → UnrollLoop →  │
      │  RenormalizeSplitPattern → HoistIfThenElse → VerifyMemory →            │
      │  AnnotateEntryFunc → InferFragment → LowerThreadAllreduce →            │
      │  AnnotateDeviceRegions → SplitHostDevice → AnnotateReadOnlyParams →    │
      │  MergeIfStmt → MakePackedAPI → LowerDeviceKernelLaunch                 │
      └───────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
             host IRModule  +  device IRModule   →  codegen → 可调用对象
```

两个必须在脑子里记住的坐标：

* **共享走廊的中点**是 `LowerTileOp`。在它之前，IR 里还有“tile op”这种
  高层概念（`T.copy` / `T.gemm`）；在它之后，只剩 TIR 语句与硬件内建。
  调试“tile 语义”问题时，在这一步前后做 diff 信息量最大。
* **唯一的分岔点是 `context.lower(mod)`**（`tilelang/engine/lower.py:136`）。
  pipeline 之上的一切（前端、缓存、codegen 调度、adapter）都是共享的。

两侧特有的 pass 名字与行号见 `fig-02-pipeline-phases.md`。
