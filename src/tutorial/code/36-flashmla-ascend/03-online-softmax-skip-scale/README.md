# 示例 03：带 skip-scale 的 online softmax（NumPy）

## 目的

把正文 §2.2–§2.3 的递推写成一个**本机可运行、可对拍**的 NumPy 实现，
用来回答两个问题：

1. 「分块 + 两个 max + 阈值」算出来的结果，和「一次算完的 softmax」一样吗？
2. `RESCALE_THRES` 取 0（每块都 rescale）、6（报告取值）、`1e30`（永不 rescale）
   三个极端分别会发生什么？

来源：报告 `:39-73`（计算流程）与 `:57,67-68`（阈值与 `exp(6)`）。

## 关键点

* 四个状态量：`running_max`、`running_max_for_softmax`、`running_sumexp`、
  `out_accum`（全部 float64，报告里是 float32）；
* 阈值**作用在缩放后的 P 上**（`P * sm_scale`），与代码
  `config.h:43` 的注释、`kernel_body.h:1189` 的比较一致；
* skip-scale：滞后不超过阈值就**跳过** rescale，直接累加。

## 运行前提

* Python 3.8+，需要 **NumPy**；
* 可选：**PyTorch**（只在 `--check-torch` 时用）；
* **不需要** GPU / NPU。

## 运行命令

```bash
python3 online_softmax_skip_scale.py
python3 online_softmax_skip_scale.py --check-torch
python3 online_softmax_skip_scale.py --topk 1024 --block 96
```

## 期望输出（节选）

```
case: seed=0, h_q=64, topk=512, d=512, block=64, sm_scale=0.04419
比较「分块 online softmax」与「一次算完的参考实现」（float64，正常工况）：
  rescale_thres=0  (共8块)       rescale=  8  max_abs_err=3.331e-16  max_rel_err=7.296e-12
  rescale_thres=6  (报告取值)      rescale=  1  max_abs_err=2.776e-16  max_rel_err=9.375e-12
  rescale_thres=1e30 (永不)      rescale=  1  max_abs_err=2.776e-16  max_rel_err=9.375e-12

病态工况（后半段 KV 放大 40 倍；float32 累加，模拟 kernel 的精度）：
  float32, thres=6             rescale=  8  max_abs_err=3.052e-05  max_rel_err=1.214e-04
  float32, thres=1e30 (出现 inf/nan) rescale=  1  max_abs_err=nan  max_rel_err=nan

结论：
  1) 正常工况下阈值 = 6 的结果与参考实现一致，且 rescale 次数 (1/8) 少于每块都做 (8/8)；
  2) 病态工况下「永不 rescale」在 float32 下会溢出（inf/nan），
     这正是报告 :67-68 给 δ 设上界的动机——阈值不是精度问题，是溢出问题。
```

（具体误差与 rescale 次数取决于随机种子与 `--block`。）

## 作者实际做到的验证级别

**本机运行过**（macOS Apple Silicon + Python 3.11.7 + NumPy 1.24.1）。
`rescale_thres=6` 的输出与参考实现的最大绝对误差在 float64 精度内。

## 已知局限（重要）

* 它**只**验证算法等价性。**没有**验证任何硬件行为、性能、数值格式。
* 真实的 kernel 用 **float32 状态量 + bfloat16 的 S**（报告 `:71`），
  且 rescale 是**逐行**判断的；本脚本用 float64、并按「任一行超阈值就整体 rescale」
  简化。因此本脚本的误差**不能**用来推断 kernel 的数值行为。
* 它**不是**对 FlashMLA kernel 的复现，也不能用来验证该 kernel 的正确性。
