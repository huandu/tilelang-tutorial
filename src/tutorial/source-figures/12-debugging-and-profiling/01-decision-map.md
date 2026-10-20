# 图 1：问题分类 → 工具 → 定位路径

对应正文 **2.3 节**。这张图是本章的总入口：先按「症状」把手上的问题分到四类之一，
再沿着该类对应的工具链往下走；每一格右侧标注了正文小节号。

```mermaid
flowchart TD
    START(["kernel 出问题了"]) --> Q1{"哪一类？"}

    Q1 -->|"1 跑不起来 / 编译失败"| GEN["生成问题 generation"]
    Q1 -->|"2 能跑但结果错"| COR["正确性问题 correctness"]
    Q1 -->|"3 结果对但慢"| PERF["性能问题 performance"]
    Q1 -->|"4 只在某些时候错"| FLAKY["偶发 / 非确定性"]

    GEN --> G1["先读报错原文：LOG(FATAL) 的消息就是线索（4.1）"]
    G1 --> G2["compile_only：不跑 GPU 先看产出的 C/CUDA（3.2）"]
    G2 --> G3["TL_LOWER_TRACE=1：逐 pass 看 before/after（3.3）"]
    G3 --> G4["pass_visualizer：结构树里看 tile op / sync / intrinsic（3.3）"]
    G4 --> G5["plot_layout / 布局可视化：确认数据映射（3.4）"]

    COR --> C1["与 PyTorch / fp64 参考对拍，先定容差与统计口径（3.7）"]
    C1 --> C2{"差异模式？"}
    C2 -->|"整体偏移 / 少数离群"| C3["T.print / T.printf 局部打印（3.5、3.6）"]
    C2 -->|"整块错位 / 全错"| C4["回看 G2-G5 的 IR 与布局"]
    C3 --> C5["device_assert：缩小到具体张量 / 线程"]
    C4 --> C3
    C5 --> C6["AutoDD：自动缩小到最小复现"]

    FLAKY --> F1["先怀疑同步：barrier / pipeline / atomic（3.7）"]
    F1 --> F2["降为单 block、num_stages=1、关流水，把并发去掉再看"]
    F2 --> F3["固定 seed 与输入，重复跑 N 次统计复现率"]
    F3 --> F4["回到 C1 的对拍流程"]

    PERF --> P1["先算 roofline 预期：memory_time vs compute_time（3.8）"]
    P1 --> P2{"预期是 memory-bound 还是 compute-bound？"}
    P2 -->|"memory-bound"| P3["Nsight Compute：Memory Workload / SOL（3.9）"]
    P2 -->|"compute-bound"| P4["Nsight Compute：SOL / Occupancy / Stall（3.9）"]
    P3 --> P5["昇腾侧用 msProf：msprof op / do_bench(msprof)（3.10）"]
    P4 --> P5
    P5 --> P6["对照预期：差多少？差在多传的内存还是少算的 FLOPs（5.2）"]
    P6 --> P7["改一个变量，重测，记录口径"]

    classDef gen fill:#eef6ff,stroke:#3b6ea5;
    classDef cor fill:#eefaf0,stroke:#2f8f4e;
    classDef flaky fill:#fff7e6,stroke:#b5722a;
    classDef perf fill:#fdeef2,stroke:#b5446e;
    class G1,G2,G3,G4,G5 gen;
    class C1,C2,C3,C4,C5,C6 cor;
    class F1,F2,F3,F4 flaky;
    class P1,P2,P3,P4,P5,P6,P7 perf;
```

要点：

* 四类问题**不是靠猜**，而是靠「有没有报错 / 结果对不对 / 快不快 / 稳不稳」四个可观测
  信号分出来的。
* 「偶发」这一类必须先怀疑同步：绝大多数偶发结果错误来自缺 `barrier`、pipeline 的
  缓冲区版本错误，或者多个 block 对同一地址的无序写（见正文 3.7）。
* 性能类一定要**先有 roofline 预期**再上 profiler，否则你只是在看一堆数字（见正文 3.8）。
* 本图不区分 NVIDIA / 昇腾；两侧工具的对应关系见正文 5.3 节的对照表。
