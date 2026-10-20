# 图 6：TileLang 与 Ascend C + MAD 的分工决策

**用途**：正文 §7.2。

**事实来源**：本章 §5.2–§5.7、§6.6 的分析，以及
`DeepGEMM-Ascend/README.md:3-11,190`；
`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:141-153`（第 19 章已核验）。

```mermaid
graph TD
    A["拿到一个昇腾算子需求"] --> B{"瓶颈在哪条流水线？"}
    B -->|"算力（Cube MAD）"| C{"形状是否已冻结、<br/>是否要榨最后 2%？"}
    B -->|"访存 / HBM 带宽"| D{"访存模式是否规整？"}
    B -->|"FixPipe / 通信"| E["先定位到具体 pipe<br/>再回到上面两条分支"]
    C -->|是| F["Ascend C + MAD 原语抽象<br/>手工 mad_m/n/k、L1/L0 stage、L2 hint、resident_*"]
    C -->|否，算法还在变| G["TileLang<br/>先跑通再优化"]
    D -->|规整| H["TileLang<br/>T.copy + 布局/多缓冲注解"]
    D -->|稀疏 / 不规则尾块 / 需 L1 bank 配平| F
    E --> C
    F --> I["用 aclnn_* 对拍 + 与硬件极限对齐"]
    G --> I
    H --> I
```

**要点**

* 主轴是**瓶颈类型**，不是「哪个更快」。
* 第二轴是「形状是否冻结」：DeepGEMM-Ascend 的配置几乎全是编译期常量，
  形状一动就要重编；TileLang 在算法迭代期的返工成本更低。
* 两条路都要做同一件事：**把结果和一份可信参考对齐**（对拍 or 手算极限）。
