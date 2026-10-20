"""一个「没有魔法」的预训练循环：数据 → 模型 → 混合精度 → 梯度累积 → 保存/恢复。

它是第 17 章 §6 的落地版，刻意写成一个文件，方便逐行对照 nanoGPT / litgpt /
torchtitan 的结构（见 references.md R4–R6）。训练循环本身**与 TileLang 无关**：
TileLang 的职责只是把其中某几个算子换掉（§7），其余部分照旧。

三种跑法，从「本机 CPU 就能验证」到「真 7B」：

1. 冒烟模式（默认，纯 CPU，几秒到几十秒）：
       python3 train_step.py --steps 100
   用一份极小的 Qwen2 结构 + 内置语料，验证「一步能跑通」与「loss 在下降」。
2. 小规模真实训练（单卡）：
       python3 train_step.py --from-hf Qwen/Qwen2.5-0.5B --dataset roneneldan/TinyStories \
           --tokenizer Qwen/Qwen2.5-0.5B --steps 2000 --bf16 --device cuda
3. 7B 继续训练（多卡，超出单机；见正文 §10 的降级路径）。

验证级别：冒烟模式在写作机器（macOS/CPU）上跑过，属于「CPU 逻辑验证」；
`--from-hf` 与 7B 两条路径只做过静态核对，未运行。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

# 复用本章 02 示例的数据管线（流式读取 + 去重 + packing）。
sys.path.append(str(Path(__file__).resolve().parent.parent / "02-data-pipeline"))
from pack_tokens import (  # noqa: E402
    build_tokenizer,
    dedup,
    hf_streaming_docs,
    offline_docs,
    pack,
    shuffled,
)


def tiny_config() -> Qwen2Config:
    """和 Qwen2.5 同族但小到能在 CPU 上跑的结构，用来验证整条链路。"""

    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA：KV 头少于 Q 头，和 7B 的比例同向
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


def build_model(args) -> Qwen2ForCausalLM:
    if args.from_hf:
        model = Qwen2ForCausalLM.from_pretrained(args.from_hf, torch_dtype=torch.bfloat16)
    else:
        model = Qwen2ForCausalLM(tiny_config())
    return model.to(args.device)


def iteration_batches(args, encode, eos_id):
    """产出 (input_ids, labels)：labels 就是 input_ids 右移一位。"""

    if args.dataset:
        raw = hf_streaming_docs(args.dataset, args.config, args.take)
    else:
        raw = offline_docs(args.docs)
    docs = shuffled(dedup(raw), args.shuffle_buffer, args.seed)
    blocks = pack(encode, docs, eos_id, args.block_size)
    pending: list[list[int]] = []
    for block in blocks:
        pending.append(block)
        if len(pending) == args.micro_batch:
            ids = torch.tensor(pending, dtype=torch.long, device=args.device)
            yield ids, ids.clone()
            pending = []


def estimate_loss(model, batches, iters: int) -> float:
    """在 no_grad 下平均若干批的 loss。

    注意：这里拿到的仍是训练语料（冒烟模式是为了证明接口能用）。真实训练必须留出
    一份**不相交**的验证集，否则这个数只是训练 loss 的另一种写法。
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (ids, labels) in enumerate(batches):
            if i >= iters:
                break
            logits = model(ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def save_ckpt(path: Path, model, optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "step": step}, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="最小可读的预训练循环")
    ap.add_argument("--from-hf", default=None, help="从 HF 权重初始化（继续训练路线）")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer；留空用字节 fallback")
    ap.add_argument("--dataset", default=None, help="HF 数据集；留空用内置语料")
    ap.add_argument("--config", default=None)
    ap.add_argument("--take", type=int, default=2000)
    ap.add_argument("--docs", type=int, default=4000)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2, help="梯度累积步数")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shuffle-buffer", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true", help="混合精度（需要支持的设备）")
    ap.add_argument("--ckpt", default="", help="checkpoint 路径，用于保存与恢复")
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params / 1e6:.2f}M  device={args.device} bf16={args.bf16}")

    encode, eos_id = build_tokenizer(args.tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.1)

    start_step = 0
    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        start_step = int(state["step"])
        print(f"从 checkpoint 恢复：{ckpt} @ step {start_step}")

    use_amp = args.bf16 and args.device != "cpu"
    amp_ctx = torch.autocast(device_type=args.device, dtype=torch.bfloat16) if use_amp else None
    scaler = None  # bf16 与 fp32 都不需要 GradScaler；只有 fp16 才需要

    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_iter = iteration_batches(args, encode, eos_id)
    losses: list[float] = []

    for step in range(start_step, args.steps):
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            ids, labels = next(batch_iter)
            with (amp_ctx if amp_ctx is not None else torch.enable_grad()):
                logits = model(ids).logits
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                )
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum

        # 梯度裁剪放在 step 之前，且在所有累积梯度都凑齐之后。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accum_loss)

        if (step + 1) % 10 == 0 or step == start_step:
            print(f"step {step + 1:5d}  loss {accum_loss:.4f}  "
                  f"ppl {math.exp(min(accum_loss, 20)):.2f}")

    if len(losses) >= 20:
        first = sum(losses[:10]) / 10
        last = sum(losses[-10:]) / 10
        print(f"前 10 步平均 loss {first:.4f} → 后 10 步平均 loss {last:.4f} "
              f"（{'下降' if last < first else '未下降'}）")

    # 评测接口演示：换一个 seed / 数据源即得到验证集口径（真实训练务必用留出数据）。
    eval_loss = estimate_loss(model, iteration_batches(args, encode, eos_id), iters=8)
    print(f"eval loss（演示口径）{eval_loss:.4f}")

    if ckpt is not None:
        save_ckpt(ckpt, model, optimizer, args.steps)
        print(f"已保存 checkpoint：{ckpt}")


if __name__ == "__main__":
    main()
