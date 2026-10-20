# 图 5：NV 与昇腾后端的结构对照

来源（NV 侧）：`_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:69-120`、
`tilelang/cuda/pipeline.py:69-279`、`tilelang/contrib/nvcc.py`。
来源（昇腾侧）：`tilelang/ascend/pipeline.py:36-190`、
`tilelang/ascend/backend.py:15-62`、`tilelang/contrib/bisheng.py:88-228`。
图为**本书绘制**；它只列**结构性事实**（谁生成什么、谁编译、同步插在哪个 pass），
不比较性能。

## Mermaid

```mermaid
flowchart LR
    subgraph NV["NV / CUDA 后端"]
        N1["MaterializeKernelLaunch<br/>lower_thread_binding=True"] --> N2["PipelinePlanning<br/>+ InjectSoftwarePipeline"]
        N2 --> N3["LowerTileOp"]
        N3 --> N4["ThreadSync('shared')<br/>warp barrier / mbarrier"]
        N4 --> N5["CodeGenTileLangCUDA"]
        N5 --> N6["CUDA C++<br/>__global__ / shared / mma|wgmma|tcgen05 / tma"]
        N6 --> N7["nvcc / NVRTC"]
        N7 --> N8["cubin / fatbin"]
    end

    subgraph AS["昇腾 / Ascend 后端（M 树）"]
        A1["MaterializeKernelLaunch<br/>lower_thread_binding=False<br/>launch_dim_tags=['cthread']"] --> A2["AscendLayoutInference<br/>+ NormalizeAscendFractalStorage"]
        A2 --> A3["AutoSchedule（Z3）<br/>+ AssignCore + PrepareMultiBuffer"]
        A3 --> A4["InsertSync<br/>+ RewriteFlagToBuf"]
        A4 --> A5["AscendLowerTileOp"]
        A5 --> A6["CodeGenTileLangAscend"]
        A6 --> A7["CCE/Ascend C<br/>__global__ __mix__(1,2) / __ubuf__ / asc_*"]
        A7 --> A8["bisheng dav-3510 + ld.lld"]
        A8 --> A9["CCE ELF (aibin)"]
    end
```

## ASCII 对照表

```text
                NV / CUDA                          Ascend (M tree, v0.1.15)
--------------- ---------------------------------- -----------------------------------------
执行单元        SM (SIMT, warp=32 lane)           AIC(Cube) + AIV(Vector)，物理 1:2 group
线程模型        thread/block/warp                 核内无线程；T.SimtVF 内部才有 threadIdx
片上存储        shared memory / registers         L1 / L0A / L0B / L0C / UB
矩阵指令        mma / wgmma / tcgen05              asc_mmad（MAD），L0A×L0B→L0C
搬运            cp.async / TMA                     asc_copy_gm2l1_* / _ub2gm / _l12l0*
收尾            epilogue 在 register             Fixpipe：asc_copy_l0c2ub / _l0c2gm
流水/多缓冲     PipelinePlanning+InjectSWPipeline  AutoSchedule(Z3)+MaterializeMultiBuffer
同步            ThreadSync（warp barrier/mbarrier）InsertSync → hard_event → 8-slot flag
生成物          CUDA C++                            CCE/Ascend C（asc_* 风格）
编译器          nvcc / NVRTC                        bisheng（+ ld.lld 链接）
产物            cubin / fatbin                      linked CCE ELF (aibin)
```

一句话总结：**NV 的“一个线程块一组线程、共享内存 + 寄存器”模型，在昇腾被拆成
“Cube 核 + 两个 Vector 子核 + 五个片上 buffer + 四条搬运/收尾流水线”**；
这是两侧 codegen 与同步 pass 完全不同的根因。
