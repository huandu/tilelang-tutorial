# 02 · 流式数据管线

## 目的

把一条真实的预训练数据链路缩成 150 行可读代码：`load_dataset(streaming=True)` →
精确去重 → 带缓冲的 shuffle → EOS 分隔的 packing。重点是让读者能亲手验证 packing
的边界行为（跨文档拼接、丢掉尾块）与 shuffle 的确定性，而不用等一个 TB 级数据集下载完。

## 运行前提

Python 3。

* **离线模式（默认）不需要任何依赖**：内置语料 + 字节级 tokenizer。
* **流式模式**需要 `datasets`，以及网络；`--tokenizer` 需要 `transformers` 与
  tokenizer 权重（可从镜像下载）。

## 命令

```bash
# 离线自检（默认）：验证 packing 逻辑
python3 pack_tokens.py
python3 pack_tokens.py --block-size 16 --num-blocks 3 --docs 20

# 真流式读取 FineWeb-Edu 的 10B 子集
python3 pack_tokens.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
    --tokenizer Qwen/Qwen2.5-7B --block-size 2048 --take 10000
```

## 期望输出

离线模式（写作机器实测，节选）：

```
block[0] 长度=32 前 12 个 token=[91, 48, 48, 48, 48, 55, 93, 32, 84, 105, 108, 101]
block[1] 长度=32 前 12 个 token=[230, 149, 176, 230, 141, 174, 230, 144, 172, 232, 191, 144]
block[2] 长度=32 前 12 个 token=[101, 114, 110, 101, 108, 227, 128, 130, 0, 91, 48, 48]
block[3] 长度=32 前 12 个 token=[184, 170, 32, 84, 114, 97, 110, 115, 102, 111, 114, 109]
共产出 160 条序列，5120 个 token，block_size=32，tokenizer=byte
```

流式模式会打印同样的行，只是 token 变成 Qwen2.5 的真实 token id，条数取决于 `--take`。

## 正确性判据

* `block[i]` 的长度恒等于 `--block-size`（尾块被丢弃，不 padding）。
* 同一 `--seed` 两次运行的输出**逐字节相同**（shuffle 可复现）。
* `block[2]` 的中段出现 `0`（EOS），证明不同文档被拼进了同一条序列。
* token 总数 ≈ `--docs × 平均每篇 token 数`，允许尾块造成的少量损失。

## 验证级别

**CPU 逻辑验证**：离线模式已运行（上面就是真实输出）。流式模式与 `--tokenizer`
分支只做过静态核对，未在写作机器上跑过（需要网络）。
