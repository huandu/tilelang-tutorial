# 04 · 数值一致性验收

## 目的

把“换了一个算子，模型还对不对”变成一个可执行、有门限的检查，而不是靠肉眼看
loss 曲线。对应正文 §7.3 与 §9 的验收标准 M4。

## 运行前提

* NVIDIA GPU，`torch`，可用 `tilelang`。
* 依赖 `../03-tilelang-norm-torch/rmsnorm_tl.py`。若本目录单独复制，把该文件
  也拷进来即可（两个文件相加即可独立运行）。

## 命令

```bash
# 需要先让 Python 找到 03 的模块（脚本已自动加 sys.path）
python3 compare_ref.py
python3 compare_ref.py --n 3584 --e2e
```

## 期望输出

```
[PASS] forward torch.bfloat16  max_abs=... max_rel=... (rtol=0.02, atol=0.02)
[PASS] forward torch.float32   max_abs=... max_rel=... (rtol=1e-05, atol=1e-05)
[PASS] backward relL2          dx=... dw=... (门限 5e-2)
总体： 全部通过
```

## 正确性判据

* 三种检查全部 `PASS`；门限按 dtype 选（bf16 用 1e-2 量级，fp32 才收到 1e-5）。
* `--e2e` 打开后额外比较端到端 loss：同一批 token、同一权重下，
  `|loss_tilelang − loss_torch| / loss_torch < 1e-3`。

## 陷阱

* **不要用 `1e-6` 当 bf16 的门限**：bf16 只有 8 位尾数，本来就有千分级的相对误差。
* 反向梯度按“相对 L2 误差”比较比逐元素更稳：某些元素的参考值接近 0，
  逐元素相对误差会被放大。
* 别只对随机小 shape 测；要覆盖 `hidden=3584`（7B）、`hidden=896`（0.5B）等真实宽度。

## 验证级别

**仅静态核对**：`python3 -m py_compile compare_ref.py` 通过；脚本**未运行**
（写作机器无 GPU）。上面的输出是**期望**格式，具体数字要你在真机上生成。
