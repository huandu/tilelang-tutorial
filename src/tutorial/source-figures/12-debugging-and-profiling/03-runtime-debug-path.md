# 图 3：运行期调试路径（含偶发错误）

对应正文 **3.5–3.7 节**。「能跑但结果错」这一类，标准流程是先对拍、再按差异模式分流，
最后才动用打印。**打印是核武器，不是第一手段**——它会改变 kernel 的时序与性能。

```mermaid
flowchart TD
    R0["找到一个能稳定复现的输入<br/>固定 seed、固定 shape"] --> R1{"结果对不对？<br/>与参考实现对拍"}

    R1 -->|"对"| DONE(["正确性通过"])
    R1 -->|"错"| R2{"差异形状？"}

    R2 -->|"整块错位 / 大面积错"| R3["先回编译期：<br/>IR 与 layout 是否如预期（图 2）"]
    R2 -->|"整体偏移 / 少量离群"| R4["先核对容差口径：<br/>累加类型、饱和、NaN（3.7）"]
    R2 -->|"每次不一样"| R5["进入「偶发」分支"]

    R4 --> R6["用 fp64 参考重算同一输入<br/>看误差是系统性还是随机的"]
    R6 --> R7["T.print / T.printf / T.dump_tensor<br/>局部打印中间张量"]
    R7 --> R8["T.device_assert 把范围缩到一个张量 / 一段线程"]
    R8 --> R9["AutoDD 自动缩小到最小复现"]

    R5 --> S1["假设 1：缺 barrier<br/>读到没写完的 shared"]
    R5 --> S2["假设 2：pipeline 缓冲版本错<br/>stage 数 / order 与实际不符"]
    R5 --> S3["假设 3：跨 block 无序写<br/>atomic / 归约 顺序不确定"]
    S1 --> S4["num_stages=1、关流水、降为单 block<br/>如果变对，就是同步问题"]
    S2 --> S4
    S3 --> S4
    S4 --> S5["逐步加回并发，找出第一次出错的位置"]
    S5 --> R7

    classDef ok fill:#eefaf0,stroke:#2f8f4e;
    classDef base fill:#eef6ff,stroke:#3b6ea5;
    classDef flaky fill:#fff7e6,stroke:#b5722a;
    classDef tool fill:#f6eefb,stroke:#7a4fa5;
    class DONE ok;
    class R0,R1,R2,R3,R4,R6 base;
    class R5,S1,S2,S3,S4,S5 flaky;
    class R7,R8,R9 tool;
```

要点：

* 「先在 CPU 上把参考算对，再上卡跑 kernel」永远比反过来便宜。
* 偶发错误的排查顺序是**先去掉并发，再加回并发**，而不是直接读代码找 bug。
* 昇腾侧没有 CUDA 的 device `printf` 语义；主动打印用 `T.printf` / `T.dump_tensor`
  （每个核 1 MB 上限），事后取现场用 AICore 异常 dump（见正文 3.6、4.5）。
