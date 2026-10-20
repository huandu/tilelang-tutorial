# 03 · LayerNorm / RMSNorm 的三级实现

## 目的

把「先能用，再用好」讲成四个可对照的 kernel：

| kernel | x 被读几遍 | 中间量在哪 | 什么时候用 |
| --- | --- | --- | --- |
| `rms_norm_naive_row_per_block` | 2 | shared | 理解归约的最短路径 |
| `rms_norm` | 1 | fragment（寄存器） | 行放得进 fragment 的常见情形 |
| `rms_norm_splitk` | 2 | fragment（只留一段） | 行太长、或片上容量紧张 |
| `layer_norm_fwd` | 1 | fragment | 需要 mean 与 rstd 两个输出 |

## 运行前提

* NVIDIA GPU + CUDA + PyTorch(CUDA)
* TileLang **v0.1.15**
* 输入/权重用 `float32`、`float16` 或 `bfloat16`

## 运行命令

```bash
cd 03-norm
python3 -m py_compile norm_kernels.py    # 本机可做（无需 GPU）
python3 norm_kernels.py                  # 真机：跑 RMSNorm 前向正确性
```

单独拿一个 kernel 出来对照生成代码（真机上跑）：

```python
import torch
from norm_kernels import rms_norm, rms_norm_splitk, torch_reference_rms_norm

m, n = 1024, 4096
x = torch.randn(m, n, device="cuda")
for name, k in (
    ("rms_norm", rms_norm(m, n, blk_m=1, dtype="float32")),
    ("rms_norm_splitk", rms_norm_splitk(m, n, blk_m=1, blk_k=512, dtype="float32")),
):
    y = k(x)
    torch.testing.assert_close(y, torch_reference_rms_norm(x), rtol=1e-4, atol=1e-4)
    print(f"===== {name} =====")
    print(k.get_kernel_source()[:2000])
```

## 预期输出

**正确性判据**：

* `rms_norm` / `rms_norm_splitk` 与 `torch_reference_rms_norm` 在 `rtol=atol=1e-2`
  内一致（fp16/bf16 输入）；fp32 输入可以收到 1e-4。
* `layer_norm_fwd` 的 `Y` 与 `F.layer_norm` 一致；`Mean`/`Rstd` 与
  `x.mean(-1)` / `x.var(-1, unbiased=False).add(eps).rsqrt()` 一致。

**性能口径**：RMSNorm 的账本下界是「读 x + 写 y」，
有效带宽 ≈ `2 * M * N * sizeof(dtype) / latency`。把它与该卡的 peak bandwidth 比，
就知道离下界还有多远。**本机没有这些数字。**

## 三个必须知道的坑

1. **`clear` 的语义**：`T.reduce_sum(src, dst, dim=1, clear=True)`（默认）是覆盖写，
   不是累加；要"接着上次的结果累加"必须显式写 `clear=False`，而且它不是简单地
   保留 dst 当累加器——跨线程归约是 butterfly，直接累加会把同一个值计入多次，
   所以 `clear=False` 会先把 dst 拷进临时 fragment 再合并（正文 §4.2）。
2. **padding 行不能参与分母**：为了对齐把行读到 `N_padded > N` 时，多出来的列必须
   填 0，并且均值要除以 `N` 而不是 `N_padded`。TileOPs 的 RMSNorm kernel 里专门有
   一行注释讲这件事
   （`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/norm/rms_norm.py:37-40`）。
3. **layout 冲突**：`T.reduce_*` 要求 src/dst 的 fragment layout 相容，否则会在
   layout inference 阶段抛错并提示 "You may need to use a shared memory to
   transform the layout"（`_vendor/tilelang-v0.1.15/src/op/reduce.cc:221-238`）。
   遇到它先检查 dst 的形状/维度，再考虑加一层 shared。

## 与官方示例的关系

* `rms_norm` / `rms_norm_splitk` 的结构来自
  `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:6-62`，保留了同样的两趟
  split-K 写法（第二趟倒序遍历，利用 cache）。
* `layer_norm_fwd` 来自 `_vendor/tilelang-v0.1.15/examples/norm/layernorm.py:6-58`，
  只取前向；它的反传用到 `T.atomic_add` 跨 block 累加 dgamma/dbeta，
  属于第 09 章的原子操作话题。
* 官方示例里的 `tl.disable_tma_lower` 在 v0.1.15 已标记 deprecated
  （`_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:141,352-355`），推荐改为逐次
  `T.copy(..., disable_tma=True)`。

## 作者验证级别

**仅静态核对**：`python3 -m py_compile` 通过；用到的 `T.*` 名字经
`../07-api-audit/api_audit.py` 对照过 `_vendor/tilelang-v0.1.15/` 的语言面。
**未在 GPU 上编译或运行过**，不含任何实测数字。
