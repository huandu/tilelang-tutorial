# 示例 07：MoE 的 dispatch 契约与量化 GEMM

本目录有两份代码，验证级别不同：

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `moe_dispatch_contract.py` | 纯标准库 | **CPU 逻辑验证**（实际运行过，三项断言通过） |
| `skeletons.py` | TileLang（MoE + W4A8 两个骨架） | **仅静态核对**（未编译、未运行） |

## 一、`moe_dispatch_contract.py`

### 目的

核对自己对 MoE 推理里“host 侧四张表 → kernel 侧行号”这套契约的理解：

1. `group_sizes` / `group_offsets` / `group_padded_offsets` / `group_idx_for_bx`
   分别是什么；
2. 每个 `(token, expert)` 对是否恰好覆盖一次；
3. 对齐后的区间是否盖住紧凑区间；
4. 内核里 `m_start = m_start_padded - group_padded_offsets[g] + group_offsets[g]`
   这一行能不能正确还原行号。

### 运行前提

* Python 3.9+，仅标准库（`argparse`、`math`、`random`）。
* **不需要 GPU / NPU，也不需要 TileLang 与 PyTorch。**

### 运行命令

```bash
python3 moe_dispatch_contract.py
python3 moe_dispatch_contract.py --tokens 40 --experts 7 --top-k 3 --padding-m 64 --block-token 64
```

### 期望输出

默认参数下的**实际输出**节选：

```text
例子：9 个 token × top_k=2 → 18 个(token, expert) 对，4 个专家
expert | group_size | group_offset | padded_offset | padded 区间长度
     0 |          8 |            0 |             0 |             128
     1 |          4 |            8 |           128 |             128
     2 |          3 |           12 |           256 |             128
     3 |          3 |           15 |           384 |             128

  紧凑数组长度（group_sum）        : 18
  对齐后需要的 block 数（M）       : 5 = ceil(18/128) + 4
  对齐浪费的行数                   : 622
  group_idx_for_bx（前 24 个）: [0, 1, 2, 3, 3]

三项自检：
  [ok] 18 个 (token, expert) 对恰好覆盖一次
  [ok] 每个专家的对齐区间都盖住了它的紧凑区间（padding_M=128）
  [ok] 复算了 5 个 block 的 m_start / actual_rows 公式，没有越界
```

### 已知局限

* 只核对 host 侧的表与寻址公式，**不执行任何 kernel**，也不验证 GEMM 数值。
* `group_idx_for_bx` 用线性扫描复刻官方写法；真实框架里这步可以向量化。

## 二、`skeletons.py`

### 目的

两个 TileLang 骨架：

1. `moe_routed_kernel`：路由专家的 grouped GEMM（gate/up → SiLU → down →
   按路由权重写回），结构对照 `examples/fusedmoe/example_fusedmoe_tilelang.py`
   的 `moe_forward_tilelang_routed`（`:92-226`）；
2. `w4a8_dequant_gemm`：权重 int4（打包进 uint8）、激活 int8 的反量化 GEMM，
   结构对照 `examples/dequantize_gemm/example_dequant_gemm_w4a8.py:86-148`。

量化推理的三种接法都能落到这两个骨架或它们的邻居上：

* **weight-only**：解包权重，激活保持高精度；
* **weight + activation**：两侧量化，kernel 内各自反量化（骨架 2）；
* **block-scaled**：`T.gemm_blockscaled(..., SFA, SFB, k_start,
  sf_a_granularity_k, sf_b_granularity_k)`
  （`_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:275`）。

### 运行前提

* NVIDIA GPU + CUDA + PyTorch(CUDA) + **TileLang v0.1.15**。
* W4A8 骨架用 `T.int8` / `T.int32` 累加；block-scaled 那一支需要 sm100
  （见 `examples/blockscaled_gemm_sm100/`）。
* **本机不具备这些条件**，因此本文件从未被编译或运行。

### 运行命令

```bash
cd _vendor/tilelang-v0.1.15/examples/fusedmoe
python3 example_fusedmoe_tilelang.py

cd ../dequantize_gemm
python3 example_dequant_gemm_w4a8.py
```

### 期望输出

官方示例自带判据（本机未运行，只抄它们的口径）：

* fusedmoe：`torch.testing.assert_close(ref_output, tilelang_output,
  atol=1e-2, rtol=1e-2)`，然后打印 `Tilelang and Torch match`；
  性能用 `do_bench(..., backend="cupti")` 分别报 shared 与 routed 两个 kernel
  的延迟并取平均（`example_fusedmoe_tilelang.py:504-611`）；
* dequantize_gemm：`assert_allclose`，性能口径是 `2*M*N*K / latency`，
  epilogue 不计入分子。

### 作者实际做到的验证级别

**仅静态核对**（`skeletons.py`）：`py_compile` 通过；逐个在 `_vendor/` 里比对
`T.Kernel`、`T.alloc_shared/alloc_fragment`、`T.copy`、`T.gemm`、`T.Pipelined`、
`T.Parallel`、`T.annotate_layout`、`tilelang.layout.make_swizzled_layout`、
`T.const`/`T.int32`/`T.uint8`、`T.max`/`T.min`、`T.ceildiv` 的存在与签名。
**没有在 GPU 上编译或运行过。**

### 已知局限

* MoE 骨架省略了共享专家（`moe_forward_tilelang_shared`）、EP 通信、
  以及 token permute 的具体实现（只给出 permute 之后的 `X`）。
* 量化骨架的 `num_bits` 只支持 4；fp8 / mxfp4 / MXFP8 的 block scale
  需要 `T.gemm_blockscaled` 与 `uint32` 打包的 scale，本文件未展开。
* 两个骨架都没有做 autotune；`block_token` / `block_dhidden` / `block_dexpert`
  的取值需要照官方示例或自己搜。
