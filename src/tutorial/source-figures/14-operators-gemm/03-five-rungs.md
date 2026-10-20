# 图 3：GEMM 的五级阶梯

**用途**：正文第 2 节。每一级都把上一级暴露出来的一个瓶颈收进编译器。

```mermaid
flowchart LR
    R0["① 朴素分块<br/>直接读 global"] -->|瓶颈: 访存 O(MNK)| R1["② shared + T.gemm<br/>片上复用"]
    R1 -->|瓶颈: 搬运/计算串行| R2["③ T.Pipelined<br/>多级流水"]
    R2 -->|瓶颈: 片上带宽 / L2| R3["④ swizzle + layout 注解<br/>物理布局 + 调度"]
    R3 -->|瓶颈: 指令级细节| R4["⑤ intrinsic<br/>ldmatrix / mma / tcgen05"]

    R0 -.->|"PyTorch: a @ b"| P["正确性基准"]
    R4 -.->|"对照物: CUTLASS warp-level MMA"| C["手写基线"]
```

## 每一级「多收了什么」

```text
级   你写的                        编译器替你做的                  主要收益
①   for k in serial(K)             线程映射                        （基准）
②   T.copy + T.gemm               MMA 指令选择、fragment 切分      访存 ↓ 一个数量级
③   T.Pipelined(num_stages=3)     多缓冲、同步插入                 搬运延迟被掩盖
④   annotate_layout / use_swizzle 布局求解、rasterization          片上带宽 + L2 命中
⑤   ldmatrix / mma / stmatrix     （几乎什么都不做）                逼近手写上限
```

## 对应源码

```text
② examples/gemm/example_gemm.py (去掉 Pipelined)
③ examples/gemm/example_gemm.py
④ examples/gemm/README.md  "Enhanced GEMM Example with Annotations"
⑤ examples/gemm/example_gemm_intrinsics.py
   Blackwell 变体: examples/gemm_tcgen05/  (T.alloc_tmem + T.tcgen05_gemm + mbarrier)
```

来源：R1、S19–S21、S27（见 `../references.md` 的 id）。
图中不包含任何性能数字——「主要收益」是对机制的定性描述，具体倍数必须在
真机上按第 7 节的口径测。
