#!/usr/bin/env python3
"""预训练数据管线的最小可运行骨架：流式读取 → 去重 → packing → shuffle。

它实现第 17 章 §4 讲的那条链路，并且**可以在没有 GPU、没有网络的机器上跑通**，
这样读者能先用小数据验证逻辑，再换成真正的 HuggingFace 数据集。

两种输入模式：

* 离线模式（默认）：用内置的小段文本做语料，走完整链路，用来验证 packing 的
  边界（EOS、跨文档拼接、丢掉尾块）以及 shuffle 的确定性。
* 流式模式：`--dataset HuggingFaceFW/fineweb-edu --config sample-10BT`，
  用 `datasets.load_dataset(..., streaming=True)` 真流式读取，不落盘、不占内存。

tokenizer 同理：`--tokenizer Qwen/Qwen2.5-7B` 用真 tokenizer（需要网络下载），
否则用字节级 fallback（词表 256），保证脚本离线可跑。

运行：

    python3 pack_tokens.py                      # 离线自检，打印统计与样例
    python3 pack_tokens.py --block-size 16 --num-blocks 4
    python3 pack_tokens.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
from collections.abc import Iterable, Iterator


EOS = 0  # 演示用的「文档结束」标记；真 tokenizer 用它自己的 eos_token_id


# --------------------------------------------------------------------------
# 语料来源
# --------------------------------------------------------------------------
_OFFLINE_DOCS = [
    "TileLang 把 tile 级的数据搬运算子写成一个 kernel。",
    "RMSNorm 是每个 Transformer 层都要跑的归一化算子。",
    "RoPE 把位置信息编码进 query 和 key 的旋转里。",
    "GQA 让多个 query 头共享同一个 KV 头，省下 KV cache 的显存。",
    "SwiGLU 是 Qwen2.5 前馈网络里用到的门控激活。",
    "混合精度的核心是：权重与激活用 bf16，累加与优化器状态用 fp32。",
    "梯度累积用小 batch 拼出等效的大 batch，代价是每步多一次前向反向。",
]


def offline_docs(n: int) -> Iterator[str]:
    """重复拼出 n 篇文档，模拟一个可流式读取的语料。"""

    for i in range(n):
        yield f"[{i:05d}] " + _OFFLINE_DOCS[i % len(_OFFLINE_DOCS)]


def hf_streaming_docs(dataset: str, config: str | None, take: int) -> Iterator[str]:
    """用 HuggingFace datasets 流式读取文本字段。

    只需要网络；不需要把数据集下载到本地。`streaming=True` 意味着迭代器每次
    只向前拉取一个分片，内存占用与语料总大小无关。
    """

    from datasets import load_dataset  # 延迟导入：离线模式下不依赖它

    ds = load_dataset(dataset, config, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= take:
            break
        text = row.get("text") or row.get("content") or ""
        if text:
            yield text


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
def byte_tokenizer(text: str) -> list[int]:
    """字节级 fallback：每个字节一个 token，词表 256，永远可离线工作。"""

    return list(text.encode("utf-8"))


def build_tokenizer(name: str | None):
    """返回 (encode, eos_id)。指定 name 时用真 tokenizer，否则用字节 fallback。"""

    if not name:
        return byte_tokenizer, EOS
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    eos = tok.eos_token_id if tok.eos_token_id is not None else EOS

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    return encode, eos


# --------------------------------------------------------------------------
# 去重 / shuffle / packing
# --------------------------------------------------------------------------
def dedup(docs: Iterable[str]) -> Iterator[str]:
    """按内容哈希做精确去重；重复文档直接丢弃。"""

    seen: set[str] = set()
    for doc in docs:
        key = hashlib.sha1(doc.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        yield doc


def shuffled(docs: Iterable[str], buffer_size: int, seed: int) -> Iterator[str]:
    """带缓冲的伪 shuffle。

    流式数据不能做全局 shuffle（那要求把所有数据读进内存），标准做法是
    维护一个固定大小的 buffer：每次从 buffer 里随机取一篇输出、再补一篇进来。
    固定 seed 保证可复现、也保证断点续训时能重放位置。
    """

    rng = random.Random(seed)
    buf: list[str] = []
    it = iter(docs)
    for doc in itertools.islice(it, buffer_size):
        buf.append(doc)
    for doc in it:
        j = rng.randrange(len(buf))
        yield buf[j]
        buf[j] = doc
    while buf:
        j = rng.randrange(len(buf))
        yield buf.pop(j)


def pack(encode, docs: Iterable[str], eos_id: int, block_size: int) -> Iterator[list[int]]:
    """把变长文档拼成定长序列。

    每篇文档编码后追加一个 EOS，然后首尾相连；每凑够 block_size 个 token 就
    产出一个训练样本。最后一个不满 block_size 的尾巴被丢弃——这是 packing
    的标准取舍：宁可丢不到一条样本，也不引入 padding（padding 会浪费算力，
    还会污染 attention 的统计）。
    """

    buf: list[int] = []
    for doc in docs:
        buf.extend(encode(doc))
        buf.append(eos_id)
        while len(buf) >= block_size:
            yield buf[:block_size]
            del buf[:block_size]


def main() -> None:
    ap = argparse.ArgumentParser(description="流式数据 → packing 骨架")
    ap.add_argument("--dataset", default=None, help="HF 数据集名；留空走离线语料")
    ap.add_argument("--config", default=None, help='数据集子集，如 sample-10BT')
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer 名；留空走字节 fallback")
    ap.add_argument("--docs", type=int, default=64, help="离线模式文档数")
    ap.add_argument("--take", type=int, default=1000, help="流式模式最多读多少篇文档")
    ap.add_argument("--block-size", type=int, default=32, help="每条训练序列的 token 数")
    ap.add_argument("--num-blocks", type=int, default=4, help="打印多少条样本")
    ap.add_argument("--shuffle-buffer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.dataset:
        raw = hf_streaming_docs(args.dataset, args.config, args.take)
    else:
        raw = offline_docs(args.docs)

    encode, eos_id = build_tokenizer(args.tokenizer)
    deduped = dedup(raw)
    shuffled_docs = shuffled(deduped, args.shuffle_buffer, args.seed)
    blocks = pack(encode, shuffled_docs, eos_id, args.block_size)

    total = 0
    for i, block in enumerate(blocks):
        total += len(block)
        if i < args.num_blocks:
            print(f"block[{i}] 长度={len(block)} 前 12 个 token={block[:12]}")
    print(f"共产出 {i + 1 if total else 0} 条序列，{total} 个 token，"
          f"block_size={args.block_size}，tokenizer={'byte' if not args.tokenizer else args.tokenizer}")
    print("注意：这些序列是「无监督语言建模」的样本——标签就是输入右移一位，"
          "所以数据管线只要产出 input_ids，labels 在训练循环里就地构造。")


if __name__ == "__main__":
    main()
