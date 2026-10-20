## 图 3 · 数据管线与训练吞吐的关系

对应正文 §4。数据管线不是“训练前的准备”，它是训练循环的一部分：它供给得够快，
GPU 就不空转；它成为瓶颈，吞吐账本（§2.3）里算出的时间就永远达不到。

```mermaid
flowchart LR
    H["HF Hub<br/>（不落地）"] -->|"streaming=True<br/>逐分片拉取"| D["去重<br/>sha1 精确哈希"]
    D --> S["shuffle buffer<br/>固定容量 + 固定 seed"]
    S --> T["tokenize<br/>（Qwen2.5 BPE）"]
    T --> P["packing<br/>EOS 分隔 + 定长切块"]
    P --> L["DataLoader<br/>num_workers"]
    L --> G["GPU 前向/反向"]
    G -->|"反向依赖：喂不快 → GPU 空转"| L
```

**读图要点**

* `streaming=True` 是“不落盘”的关键：语料总量（FineWeb-Edu 完整版约 5.84 TB）
  与你机器上的磁盘大小无关，内存占用只与 buffer 有关。
* shuffle 是**带缓冲的局部 shuffle**，不是全局 shuffle；buffer 越大越接近均匀，
  但内存与首次填充延迟也越大。固定 seed 是断点续训能重放的前提。
* packing 之后每条样本都是定长 `block_size`，**没有 padding**，所以 GPU 的算力
  不会被填充 token 浪费。
* 数据管线一旦落后，表现为 GPU 利用率周期性掉到 0。这时先查 `num_workers`
  和 tokenizer 是否成了瓶颈，而不是去调 kernel。
