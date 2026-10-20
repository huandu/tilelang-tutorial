# 示例 02：调用 TileKernels 的公开 API

## 目的

展示“一份 Python API、两个后端”在**使用者视角**下是什么样：调用方完全不写
`if is_ascend()`，后端由库内部按运行期探测分派（正文 §3）。

两个脚本都刻意**只 import 外层 API**，与 `tests/**` 的写法一致——
所以它们既是使用样例，也是“同一份测试覆盖两个后端”这句话的演示。

## 文件

| 文件 | 覆盖 | 对应正文 |
| --- | --- | --- |
| `call_topk_gate.py` | 后端探测 → `moe.topk_gate` → 与 `torch/topk.py` 参考实现逐字节对拍 → 空输入短路 | §3.1、§3.2、§3.4、§2.3 |
| `call_quant_and_rope.py` | `quant.cast` / `per_token_cast` / `per_token_cast_back` / `swiglu_forward_and_per_token_cast` / `transform.apply_rotary` | §5、§6 |

## 运行前提

* 已安装 TileKernels（`pip install tile-kernels`，或把仓库根加入 `PYTHONPATH`）；
* TileLang `v0.1.15`；
* NVIDIA SM90/SM100（CUDA ≥ 13.1）**或** Ascend 950（CANN ≥ 9.2.0 + `torch_npu`）。

## 运行命令

```bash
python3 call_topk_gate.py
python3 call_quant_and_rope.py
```

## 期望输出（在满足前提的机器上）

```
is_ascend() = False
get_device() = cuda
topk_idx.shape = (512, 6), dtype = torch.int64
Kernel Output Match!
empty input short-circuits ok
```

（昇腾机器上第二行应为 `get_device() = npu`，并额外打印 `get_num_vec_cores()`。）

```
cast: (512, 2048) dtype=torch.float8_e4m3fn
per_token_cast: data=(512, 2048) sf=(512, 16)
cast_back diff = <应很小>
sf_only: (512, 16)
swiglu+cast: out=(512, 2048) sf=(512, 16)
swiglu only: (512, 2048)
rope diff = <应 <= 1e-8>
```

* **正确性判据**：`call_topk_gate.py` 的 `assert_equal` 是**逐字节**比对
  （`tile_kernels/testing/numeric.py:5-30`）；两个 diff 用
  `calc_diff`（相对误差，`:33-38`），RoPE 的目标是 `<= 1e-8`
  （与 `tests/transform/test_rope.py:45` 相同的门限）。
* **性能口径**：本示例不测性能。要测性能请用
  `python -m pytest tests/... --run-benchmark`（见 `TileKernels/README.md:47-62`）。

**本机没有 GPU / NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：两个文件都通过 `python3 -m py_compile`；函数名与参数
（`per_token_cast` / `per_token_cast_back` / `per_token_cast_with_sf_only` /
`swiglu_forward_and_per_token_cast` / `swiglu_forward` / `apply_rotary` /
`get_num_vec_cores` / `set_pdl` / `set_num_sms` / `calc_diff` / `assert_equal` /
`stable_topk` / `rotary_embedding_ref`）逐个在 corpus 的
`TileKernels/tile_kernels/` 与 `tests/` 中比对过。
**没有在任何硬件上运行过。**

## 已知局限

* `cast` 在昇腾上目前只支持 `fp32 -> bf16` 且必须开随机舍入
  （`cast_kernel.py:40-51`）；本脚本用的是 `bf16 -> e4m3`，在昇腾上会断言失败。
  这是**上游当前的实现状态**，不是本章示例的取舍。
* 量化路径的形状约束（`hidden % 64 == 0` 等）见正文 §5.3 的表。
* `positions` 用 `int32`；`int64` 需要另外一条分支（`rope_kernel.py:88-106`）。
