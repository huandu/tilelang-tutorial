# 图 6：什么时候需要自己写 kernel

对应正文第 5 节。这是一个**从测量出发**的决策树，不是「先写 kernel 再看效果」。
树叶上的四个判据分别对应正文 5.1–5.4 的四类情况。

```mermaid
flowchart TB
    START["热点算子：profiler 里占 GPU 时间最多的几个"] --> Q0{"先试过 torch.compile /<br/>厂商库的融合写法和现成 kernel 吗？"}
    Q0 -->|没有| TRY["先试：torch.compile + 库算子<br/>再看一遍 kernel 表"]
    TRY --> Q0
    Q0 -->|试过，还是慢| Q1{"慢的原因是哪一类？"}

    Q1 -->|"相邻的逐元素/归一化算子<br/>被拆成很多个小 kernel"| K1["判据 1：融合缺失<br/>memory-bound × 多趟访存"]
    Q1 -->|"库算子覆盖不到这个融合模式<br/>（MLA/NSA/indexer/RoPE+cache 写入）"| K2["判据 2：库没有这个算子"]
    Q1 -->|"需要库不支持的低精度/布局<br/>（fp8 block scaling、int4、NZ/fractal）"| K3["判据 3：数据布局或精度不被支持"]
    Q1 -->|"单个算子已经贴着下界，但整段流水仍有串行等待"| K4["判据 4：需要跨算子的数据流重组<br/>（流水、persistent、多缓冲）"]

    K1 --> WRITE["写 kernel"]
    K2 --> WRITE
    K3 --> WRITE
    K4 --> WRITE
    WRITE --> VERIFY["用 roofline 利用率 / 库相对效率验收<br/>并做数值对比"]
    VERIFY --> BAD["反例：一个语法正确、数值正确的<br/>LayerNorm kernel 可以比 PyTorch 慢 300 倍<br/>（A100/GH200 上的 22 个 kernel 实测，见正文 5.5）"]
```

## 来源

* 决策树本身是本书的归纳，判据在正文 5.1–5.4 逐条给出证据。
* 「写得对但很慢」的反例与「roofline 利用率 / 库相对效率」两个验收判据：
  `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2607-04454.md`（摘要里写明在 NVIDIA A100 与
  GH200 上对 22 个 Triton/TileLang kernel 的实测，其中一个 LayerNorm 比 PyTorch 基线慢 300 倍以上）。
* 「先试 torch.compile」：见正文 4.3 与 `code/README.md` 里给出的观察步骤。
