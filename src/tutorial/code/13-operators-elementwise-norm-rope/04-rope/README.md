# 04 · RoPE 的两种约定与一个面向向量化的版本

## 目的

1. 先把 **interleaved** 与 **half-split (NeoX)** 两种配对约定用 PyTorch 写成可对照的
   参考实现（CPU 上就能跑）；
2. 再给两个 TileLang 版本：单行朴素版、一个 block 多行的 shared 暂存版；
3. 说明“配对方式”如何决定访存模式。

## 运行前提

* 参考实现：只需要 PyTorch（CPU 即可）
* kernel：NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang **v0.1.15**

## 运行命令

```bash
cd 04-rope
python3 -m py_compile rope_kernels.py     # 本机可做（无需 GPU）
python3 rope_kernels.py                   # 真机：跑正确性
```

参考实现在 CPU 上就能跑，用来确认你理解的是哪一种约定：

```python
import torch
from rope_kernels import torch_rope_interleaved, torch_rope_neox

x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
ones = torch.ones(1, 4)
zeros = torch.zeros(1, 4)

# c = 1, s = 0 时两种约定都必须原样返回 x
assert torch.equal(torch_rope_interleaved(x, ones, zeros), x)
assert torch.equal(torch_rope_neox(x, ones, zeros), x)

# c = 0, s = 1 时两种约定的输出必须不同
print("neox       :", torch_rope_neox(x, zeros, ones).tolist())
print("interleaved:", torch_rope_interleaved(x, zeros, ones).tolist())
```

## 预期输出

`c=1, s=0` 时两种约定都必须**原样返回 x**——这是最快的自检；如果这一步不过，
kernel 里的索引一定写错了。`c=0, s=1` 时两种约定的输出**必须不同**，
否则说明你把两个约定写成了同一个。

**正确性判据**：`rope_neox_blocked` 与 `torch_rope_neox` 在 `rtol=atol=1e-2` 内一致
（fp16 输入）；fp32 输入可以收到 1e-5。

**性能口径**：RoPE 的账本下界是“读 x 一遍 + 读 cos/sin 一遍 + 写 y 一遍”。
注意 **cos/sin 表的读法与实现强相关**：按 pair 读（每行 R/2 次）与按整行广播读
（每行 R 次）在账本上差了近一倍，虽然这部分数据总量都很小。

## 三个必须知道的坑

1. **写错约定不会报错**。两种约定都是合法的旋转，跑完也不崩，只是数值错。
   所以先用 `c=1,s=0` 自检，再和 PyTorch 对齐。
2. **部分旋转（partial RoPE）**：真实模型（MLA 的 decoupled RoPE、GLM 等）只旋转
   尾部 `rope_dim` 个通道；`dim_start = hidden_size - rope_dim` 是通用写法
   （对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rope.py:23,38`）。
3. **fp16 全程计算会多丢一次精度**：`x*c - y*s` 若两次乘、一次减都在 fp16 上做，
   会舍入两次；升到 fp32 算完再 cast 一次更稳（本文件的两个 kernel 都这么做），
   代价是 fp32 中间量占的片上空间翻倍。

## 与上游示例的关系

* interleaved 的 PyTorch 参考实现与 mask/gather 技巧来自适配器仓库
  `_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rope.py:58-100,132-150`。
  它用 `T.tile.gather` + 位运算 mask 在 UB 上做配对交换，是昇腾侧的写法；
  NVIDIA 侧不需要绕这一步。
* 生产级的 5 种频率变体（NeoX / Llama-3.1 / YaRN / LongRoPE / non-NeoX）与两种布局
  见 `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/rope.py`。
  频率表的计算在 Op 层（Python），kernel 只接收算好的 cos/sin。

## 作者验证级别

**仅静态核对**：`python3 -m py_compile` 通过；用到的 `T.*` 名字经
`../07-api-audit/api_audit.py` 对照过 `_vendor/tilelang-v0.1.15/` 的语言面。
**未在 GPU 上编译或运行过**，不含任何实测数字。
