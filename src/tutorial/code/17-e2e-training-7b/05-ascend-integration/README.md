# 05 · 昇腾上把算子接进 PyTorch

## 目的

对应正文 §8。演示昇腾上两种把 TileLang 算子接进 PyTorch 的方式：

* **路径 A**：`tilelang.compile(..., target="ascend")` 得到的 kernel 直接吃
  `.npu()` 张量（adapter 会把 ascend target 映射到 `torch.device("npu")`）。
* **路径 B**：按官方示例 `examples/torch_tl_ascend/` 把算子注册成
  `torch.ops.tl_ascend.flash_attention`，供模型代码无感调用。

## 运行前提

* 昇腾 NPU（官方在 A2 / A3 上验证过）。
* CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1。
* TileLang-Ascend `v0.1.2.000-release`（`source {CANN}/ascend-toolkit/set_env.sh`
  之后再 import）。
* 路径 B 还需要按示例的 `setup.py install` 构建 `torch_tl_ascend` 包。

## 命令

```bash
# 路径 A：Python 直接编译 + 调用
python3 invoke_ascend_op.py --path a

# 路径 B：先构建再调用（在 TileLang-Ascend 仓库里）
#   cd _vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend
#   python setup.py install
python3 invoke_ascend_op.py --path b
```

## 期望输出

```
路径 A: kernel 输出与 torch 参考一致
路径 B: torch.ops.tl_ascend.flash_attention 与参考一致
```

## 正确性判据

* 路径 A：`torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)`；
  `rstd` 与 `torch.rsqrt(x.square().mean(-1) + eps)` 对齐到 `1e-5`。
* 路径 B：`assert_close(ref, out, rtol=1e-2, atol=1e-2)`（fp16 attention 的常规门限）。
* 两条路径都要在 `torch.npu.synchronize()` **之后**再比较，否则读到的是未完成的异步结果。

## 昇腾上的现实约束（写进正文 §8）

* **无 autograd**：和 NV 侧一样，TileLang-Ascend 只给前向；反向要自己写 kernel 或
  用 PyTorch 补。整条反向链路是否留在 NPU 上，决定你能否真的用它训练。
* **算子覆盖面**：官方已验证的算子族（GEMM / attention / normalization / MoE
  dispatch-combine 等）能覆盖训练前向的大部分；但 optimizer 里那些奇怪的逐元素
  算子通常还是走 torch-npu 自带实现。
* **同步模型不同**：核间同步由编译器插入（`InsertSync`），不要在 kernel 里手写
  时序假设。

## 验证级别

**仅静态核对**：`python3 -m py_compile invoke_ascend_op.py` 通过；接口形态对照了
`_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/{README.md,test_source.py,test_torch.py}`；
**未运行**（写作机器无 NPU，也未安装 CANN）。
