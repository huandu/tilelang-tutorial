# 图 1：数据流与调度空间的解耦

对应正文 **2.1–2.3 节**。左列是用户直接写出来的东西，右列是被解耦出去、由注解或
编译器决定的东西。

```mermaid
flowchart LR
    subgraph user["用户写下来的：数据流 dataflow"]
        direction TB
        A["申请 tile 缓冲<br/>T.alloc_shared / alloc_fragment"] --> B["搬运<br/>T.copy"]
        B --> C["计算<br/>T.gemm / T.reduce_* / T.Parallel"]
        C --> D["写回<br/>T.copy"]
    end

    subgraph sched["被解耦出去的：调度空间 scheduling space"]
        direction TB
        S1["thread binding<br/>threads=... / policy=...<br/>T.get_thread_binding()"]
        S2["layout<br/>T.annotate_layout<br/>loop_layout=..."]
        S3["tensorize<br/>T.gemm 派发到 mma/wgmma/tcgen05"]
        S4["pipeline<br/>num_stages=... / stage+order"]
    end

    B -.->|"搬运用什么机制"| S4
    C -.->|"计算分给多少线程"| S1
    C -.->|"张量核指令怎么选"| S3
    A -.->|"片上怎么摆"| S2
    S2 -.->|"LayoutInference 推导 Fragment"| A

    classDef userStyle fill:#eef6ff,stroke:#3b6ea5;
    classDef schedStyle fill:#fff4e6,stroke:#b5722a;
    class A,B,C,D userStyle;
    class S1,S2,S3,S4 schedStyle;
```

要点：

* 四个调度维度在 v0.1.15 里**都有入口，也都有缺省**（见正文 2.3 节的表）。
* 虚线表示「用户可以选择介入」；不介入时由编译器决定。
* 注意 `layout` 那条是双向的：用户可以在 `T.annotate_layout` 里写死，也可以留空让
  LayoutInference 推导（`_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:30-38`）。

事实来源：正文 2.1–2.3 节；`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md`
（论文摘要）；`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:74-84`。
