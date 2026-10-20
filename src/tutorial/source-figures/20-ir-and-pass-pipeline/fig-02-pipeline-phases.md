# 图 2：九组职责 · CUDA 与 Ascend 对照（自绘）

把 `tilelang/cuda/pipeline.py`（73 次 pass 调用）与
`tilelang/ascend/pipeline.py`（80 次 pass 调用）按**职责**归成九组。
括号里的行号是锁定版本 `_vendor/tilelang-v0.1.15/` 里的真实行号。

```text
组           CUDA（tilelang/cuda/pipeline.py）        ASCEND（tilelang/ascend/pipeline.py）
──────────  ────────────────────────────────────────  ──────────────────────────────────────────
① 前端        BindTarget L69                          BindTarget L37
   合法化      MaterializeKernelLaunch L70             MaterializeKernelLaunch L45
              AnnotateDeviceBoundTmaCopies L74        RewriteDualCopy L68
              AddWrapperForSingleBufStore L79         LegalizeSimdMerging L72
              LegalizeNegativeIndex L81               AddWrapperForSingleBufStore L58
              InjectAssumes L86 / Simplify L88        LegalizeNegativeIndex L59
              CanonicalizeLegacyReducer L91           InjectAssumes L62 / Simplify L73
              VerifyReducerEpoch L92                  CanonicalizeLegacyReducer L74
              VerifyBufferInit L98                    VerifyReducerEpoch L75
              [WarpSpecialization 三连 L107-109]      VerifyBufferInit L76
              LowerBlackwell2SM L114                  UnrollLoopSkipVF L80
              IfStmtBinding L118

② 流水        UnrollLoop L123 → Simplify L125         （由 AutoSchedule 段统管，
              PipelinePlanning L129                   见第 ③ 列下方）
              InjectSoftwarePipeline L130
              Simplify L131

③ 布局        LayoutInference L134                     AscendLayoutInference L83
              ReducerPlanAndMaterialize L138           ReducerPlanAndMaterialize L84
              LayoutVisual L140                        NormalizeAscendFractalStorage L86
                                                       InsertNd2Nz L91
                                                       AscendInsertOOBPadding L101
                                                       ── 自动调度段 L104-116 ──
                                                       NormalizeControlFlowForSchedule
                                                       NormalizeConflictHints
                                                       MaterializeScheduleUnits
                                                       AnnotateMultiBufferEligible
                                                       EstimateLatency → AutoSchedule
                                                       AssignCore → PrepareMultiBuffer
                                                       ResolveCore → InsertSync
                                                       MaterializeMultiBuffer
                                                       LowerScheduledTIR
                                                       RestoreWhileLoops

④ tile op     LowerTileOp L142                         NormalizeBufferVersion L120
   降级        VerifyReducerConsumed L144               AscendSimdVFLowerParallel L121
              DecoupleTypeCast L150                    AscendLowerTileOp L122
              LegalizeVectorizedLoop L152              VerifyReducerConsumed L123
              LegalizeSafeMemoryAccess L154            DecoupleTypeCast L125
              LowerAccessPtr L156                      LegalizeVectorizedLoop L126
              Simplify L162                            LegalizeSafeMemoryAccess L127
              HoistNonRestrictParams L164              LowerAccessPtr L128
              LowerSharedTmem L175                     Simplify L129
              LowerSharedBarrier L196                  HoistNonRestrictParams L130

⑤ 内存        HoistGlobalBufferAllocations L203       HoistGlobalBufferAllocations L132
   规划        LowerOpaqueBlock L204                    LowerOpaqueBlock L133
              FlattenBuffer L207                       RewriteAscendBufferVersionLayout L135
              ConfigIndexBitwidth L210                 FlattenBuffer L137
              StorageRewrite L213                      ConfigIndexBitwidth L138
              MergeSharedMemoryAllocations L251        MergeUBAllocations L171
                                                       RewriteFlagToBuf L179

⑥ 向量化/     VectorizeLoop L212                       VectorizeLoop L140
   展开        LoopUnswitching L214                     RewriteFp4ToFp4x2 L146
              UnrollLoop L215                          LoopUnswitching L148
              RenormalizeSplitPattern L216             UnrollLoop L149
              RemoveNoOp L218                          RenormalizeSplitPattern L150
              HoistIfThenElse L219                     AscendRemoveNoOp L152/L155
                                                       HoistIfThenElse L153

⑦ 同步        VerifyMemory L221                        VerifyMemory L157
              InferFragment L232                       InferFragment L159
              LowerThreadAllreduce L233                LowerThreadAllreduce L160
              MergeSharedMemoryAllocations L251        [AscendThreadSync("global") L163]
              InjectFenceProxy L256                    MarkScalarDcacheBypass L165
              ThreadSync("shared") L258                MergeUBAllocations L171
              ThreadSync("shared.dyn") L259            RewriteFlagToBuf L179
              InjectTcgen05Fence L266                  AscendThreadSync("shared") L181
                                                       AscendThreadSync("shared.dyn") L182

⑧ host/device SplitHostDevice L240                    SplitHostDevice L166
   切分        AnnotateReadOnlyParams L245             AnnotateReadOnlyParams L167
              AnnotateDeviceRegions L239               AnnotateDeviceRegions L164
                                                       MergeIfStmt L183

⑨ API 打包    MakePackedAPI L273                       MakePackedAPI L184
              LowerDeviceKernelLaunch L275             Simplify L185
              PersistThreadblock L279                  LowerDeviceKernelLaunch L186
```

**怎么读这张图**：九组里，① ② ③ ④ 是“越来越低”的抽象阶梯；⑤ ⑥ ⑦ 是
“物理化”，把逻辑 tile 落到真实的寄存器、片上存储与屏障；⑧ ⑨ 是“接口化”，
把内核切成 host 与 device，再包成可调用入口。CUDA 与 Ascend 在 ①④⑤⑥⑦⑧⑨ 上
大量重叠（共享 39 个同名 pass），真正的分歧集中在 ② 和 ③：**谁负责流水与调度**。

CUDA 用通用的 `PipelinePlanning` + `InjectSoftwarePipeline`；Ascend 用整段
`AutoSchedule`（Z3/SMT 求解），把流水、多缓冲、核间分派、同步一起解。
