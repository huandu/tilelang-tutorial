# 03 · RMSNorm：TileLang 前向 + PyTorch autograd

## 目的

演示正文 §7 的三个要点：

1. 用 `@tilelang.jit(out_idx=[2])` 写一个能吃 `torch.Tensor` 的 RMSNorm 前向；
2. 说明 TileLang **只产出前向 kernel**，反向必须由你显式接到
   `torch.autograd.Function` 上；
3. 给出可对照的纯 PyTorch 参考实现 `rmsnorm_ref`。

## 运行前提

* NVIDIA GPU（或昇腾 NPU，把 `device="cuda"` 换成 NPU 并把 `import tilelang`
  换成 TileLang-Ascend 安装）。
* `torch >= 2.1`、`tilelang == v0.1.15`。

## 命令

```bash
python3 rmsnorm_tl.py
# 更完整的数值验收见 ../04-verify-numerics/README.md
```

## 期望输出

```
y torch.Size([256, 512]) dx torch.Size([256, 512]) dw torch.Size([512])
```

（前半句是前向输出形状，后半句证明 `loss.backward()` 已经能穿过这个自定义算子——
即 `RMSNormTileLang.backward` 被成功调用。）

## 正确性判据

* kernel 输出与 `rmsnorm_ref` 的 `torch.allclose`（bf16 用 `rtol=atol=2e-2`）。
* `RMSNormTileLang.backward` 返回的 `dx`、`dw` 与参考梯度的相对 L2 误差 < 5e-2。
* 与 `torch.nn.RMSNorm`（如该版本提供）或 HF `Qwen2RMSNorm` 的输出对齐。

## 已知差异 / 陷阱

* 本 kernel 用 `blk_m=1`，即一行一个 block，只适合教学；生产上要按 `d` 的宽度
  批量成多行一块（参考 `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py` 与
  `examples/ascend/example_rmsnorm.py` 的多行批量做法）。
* 昇腾版本要用 `T.alloc_reducer` / `T.reducer_init` / `T.finalize_reducer`，
  与 NV 侧的 `T.reduce_sum` 写法不同——见 `05-ascend-integration/`。

## 验证级别

**仅静态核对**：API 名、签名、`out_idx` 语义都与 `_vendor/tilelang-v0.1.15/`
逐项比对过；`python3 -m py_compile rmsnorm_tl.py` 通过；**未运行**（写作机器无 GPU/NPU）。
