# 图 4：先建立 roofline 预期，再测量

对应正文 **3.8–3.10 节**。性能分析最常见的浪费是「先测一堆数，再想它意味什么」。
正确顺序是先算出这个算子的理论上下界，再用 profiler 回答「差在哪一项」。

```mermaid
flowchart TD
    A["算子 + 形状 + 精度"] --> B["算 FLOPs，如 GEMM: 2*M*N*K"]
    A --> C["算必须搬运的字节：每个输入存储读一次、每个输出写一次"]
    B --> D["compute_time = FLOPs / peak_flops"]
    C --> E["memory_time = bytes / bandwidth"]
    D --> F["sol_time = max(memory_time, compute_time)，较大的一项决定 bound"]
    E --> F
    F --> G["efficiency = sol_time / 实测时间"]
    G --> H{"效率离 1 远吗？"}
    H -->|"接近 1"| I["已到硬件上限：换算法 / 换精度，不要调参数"]
    H -->|"差很多"| J{"bound 是哪一类？"}
    J -->|"memory-bound"| K["先查是否多搬了字节：未融合的中间张量、重复读写、非合并访问"]
    J -->|"compute-bound"| L["先查 tensor core 利用率：tile 形状、K 维切分、occupancy、指令停顿"]
    K --> M["改一个变量 -> 重测 -> 用同一口径复算 efficiency"]
    L --> M
    M --> F

    classDef calc fill:#eef6ff,stroke:#3b6ea5;
    classDef decide fill:#fff4e6,stroke:#b5722a;
    classDef act fill:#fdeef2,stroke:#b5446e;
    class A,B,C calc;
    class D,E,F,G,H,J decide;
    class I,K,L,M act;
```

要点：

* `bytes` 用的是**算法最小搬运量**（每个输入存储读一次、每个公开输出写一次），
  不是 profiler 测到的 DRAM 流量。两者之差本身就是「多搬了」的证据。
* TileLang 自带的 `tilelang.tools.Analyzer` 就是这个流程的静态实现：它数 `T.gemm`
  的 FLOPs、数 `T.copy` 跨 global 边界的字节，返回 `total_flops` /
  `total_global_bytes` / `estimated_time`。它是**估算**，不是实测。
* 昇腾侧的 pipe 利用率（AIC 的 mad/mte1/mte2/fixpipe，AIV 的 vec/mte2/mte3）
  不能相加成一个「总利用率」——它们分母各自不同（见正文 5.2）。
