# 06 · 预训练循环

## 目的

一个文件、没有魔法的训练循环：数据 → 前向 → 交叉熵 → 反向 → 梯度累积 → 裁剪 →
AdamW → 保存 / 恢复。它回答正文 §6 的问题，也顺带证明**训练循环本身与 TileLang
无关**——TileLang 只是其中几个算子被替换掉。

## 运行前提

* 冒烟模式（默认）：Python 3、`torch`、`transformers`；**纯 CPU 可跑**。
* 训练真实模型（`--from-hf`、`--bf16`）：GPU/NPU + 足够显存；多卡还要 `torchrun`。

## 命令

```bash
# 1) 冒烟：极小结构 + 内置语料，验证「能跑一步」与「loss 在下降」
python3 train_step.py --steps 100 --micro-batch 8 --grad-accum 2

# 2) checkpoint 保存 + 恢复
python3 train_step.py --steps 20  --ckpt /tmp/ck.pt
python3 train_step.py --steps 30  --ckpt /tmp/ck.pt   # 从 step 20 继续

# 3) 单卡真实小模型（需要网络与 GPU）
python3 train_step.py --from-hf Qwen/Qwen2.5-0.5B --tokenizer Qwen/Qwen2.5-0.5B \
    --dataset roneneldan/TinyStories --block-size 1024 --micro-batch 4 \
    --grad-accum 8 --steps 2000 --bf16 --device cuda
```

## 期望输出（冒烟模式，写作机器实测）

```
模型参数量：0.09M  device=cpu bf16=False
step     1  loss 5.5553  ppl 258.60
step    10  loss 5.3166  ppl 203.69
...
step   100  loss 3.7723  ppl 43.48
前 10 步平均 loss 5.4314 → 后 10 步平均 loss 3.8365 （下降）
eval loss（演示口径）3.7306
```

## 正确性判据

* 输出的平均 loss 随步数**单调下降**（允许抖动，但首 10 步均值 > 末 10 步均值）。
* `--ckpt` 两次运行：第二次开头打印 `从 checkpoint 恢复：... @ step 20`。
* loss 是有限值（不出现 `nan` / `inf`）；出现 `nan` 通常是 lr 过大或 dtype 选错。

## 陷阱

* `labels` 是 `input_ids` 右移一位；`cross_entropy` 前要把 `logits[:, :-1]` 与
  `labels[:, 1:]` 对齐，否则 offset 会污染 loss。
* bf16 不需要 `GradScaler`（那是 fp16 才需要的）；`--bf16` 在 CPU 上会被忽略。
* 梯度裁剪要放在**所有累积梯度凑齐之后、`optimizer.step()` 之前**。

## 验证级别

* 冒烟模式与 checkpoint 恢复：**CPU 逻辑验证**（已运行，输出见上）。
* `--from-hf`、`--bf16`、多卡路径：**仅静态核对**。
