# 示例 03：复算 carver/roller 的 tile cost model

## 目的

`_vendor/tilelang-v0.1.15/tilelang/carver/` 是 TileLang 自带的 tile 配置推荐器。
它的 cost model 决定了哪些 tile 会被推荐、哪些会被直接丢弃。这个脚本把那套算术
用标准库重写一遍，好在没有 GPU 的机器上读懂它：

| 量 | carver 里的位置 |
| --- | --- |
| `coalesced_factor` / `coalesced_tensor_shape` | `carver/roller/policy/common.py:32-52` |
| traffic | `carver/roller/policy/default.py:398-432` |
| shared memory（best-fit，32 B 对齐） | `carver/roller/policy/default.py:452-495` + `roller/bestfit.py:22-61` |
| `reg_usage = 2 * max(prod(tile) * bits / 32)`、容量上限 | `carver/roller/policy/default.py:556-574` |
| `block_per_SM`、`num_wave` | `carver/roller/policy/default.py:568-573` |
| 排序 `priority = (traffic + 1) * num_wave` | `carver/roller/policy/default.py:111-112` |
| 设备模型（`smem_cap` / `reg_cap` / `sm_partition` / `transaction_size`） | `carver/arch/cuda.py:153-167`、`arch/arch_base.py:6-27` |

`--self-test` 会把复算的 traffic 与教科书公式
`b·M·N·K·(1/block_N + 1/block_M) + b·M·N` 逐项对比。

## 运行前提

* Python 3.9+，只用标准库。
* **不需要 GPU / NPU / TileLang / PyTorch。**

注意：**不能**在无卡机器上 `import tilelang.carver` 来对照。`CUDA.__init__` 会调用
`tvm.runtime.cuda(0)` 并在没有设备时抛
`RuntimeError("Cannot find cuda device 0.")`（`carver/arch/cuda.py:144-146`），
`carver/arch/__init__.py:47-56` 的 `auto_infer_current_arch()` 在没有 CUDA 时会退到
CPU 模型。所以这份复算是**代码阅读的产物**，只保证算术与源码一致，不保证与运行时
完全一致。

## 运行命令

```bash
python3 carver_cost_model.py --self-test --topk 8
python3 carver_cost_model.py --m 8192 --n 8192 --k 8192 --topk 12
python3 carver_cost_model.py --smem-cap 101376 --cores 108 --topk 8   # 别的卡
```

## 期望输出

正确性判据：退出码为 0，且 `--self-test` 的 `rel err` 全为 `0.00e+00`。

```text
self-test: carver-style traffic vs the closed form b*M*N*K*(1/block_N + 1/block_M) + b*M*N
 block_M block_N block_K    model MB   closed MB   rel err
      64      64      64      4328.5      4328.5  0.00e+00
     128     128      32      2181.0      2181.0  0.00e+00
     128     256      64      1644.2      1644.2  0.00e+00

candidates visited : 729   valid : 154
rejected on regs     :  122
rejected on smem     :  453
model priority values in use : 48
ties at the top     : 8

  #  block_M block_N block_K  traffic MB  smem KB blk/SM waves        prio  binding
  1      128     256      16      1644.2     12.0      1     4    6.58e+09  regs
  ...
```

三个可以直接读出来的结论：

1. **traffic 对 `block_K` 不敏感**：`128×256×16` 与 `128×256×128` 的 traffic 完全
   相同，`prio` 也完全相同。模型的这一维没有分辨率。
2. **寄存器比 shared memory 先撞墙**：122 个候选被 `reg_usage > reg_cap` 丢掉，
   其中最大的一批是 `block_K` 很大而 `M×N` 不小的形状。
3. **模型会给出大量并列**：154 个可行候选只产出 48 个不同的 `prio` 值，前 8 名并列。
   并列就是“接下来该去测量”的信号（第 11 章 3.9 节的说法，这里给出了它的量级）。

## 性能口径

**不测量任何性能。** `prio` 是无量纲的排序键，不要当成时间。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认、
`--self-test`、多组 `--m/--n/--k` 与多组 `--smem-cap/--cores`；输出与本文件一致。
没有在 GPU 上运行过 carver 本体。

## 已知局限

* **tile 传播是手写的。** carver 走的是 `PrimFuncNode.propagate`
  （`roller/node.py:332-399`）这一套通用的区域推导；本脚本对普通 GEMM 把它解析地
  写成“输出 tile 需要 A 的 `block_m` 行整条 K、B 的 `block_n` 行整条 K”。
  这一读法能精确复现教科书 traffic 公式；另一种读法（只取一个 `rstep` 切片）
  不能。要判定 carver 到底返回哪一个，需要在有卡的机器上实际跑一次——本机做不到，
  所以这里把它写成显式假设而不是结论。
* 候选枚举是一次性的因子笛卡尔积，不是 carver 的 `dfs_smem_tile` 最优先搜索
  （`roller/policy/default.py:96-134`，上限 2000 个已访问 tile）。在同样的候选集合
  上排序键一致，但访问顺序不同。
* `smem_cap` / `cores` 由命令行给出，不是从设备查询的。
* `reg_usage` 里的位宽取的是**累加器**的位宽（fp32 时 32）。carver 用的是
  `node.get_dtype()`，它在我们的复算里被取成累加 dtype；如果真实节点 dtype
  是输入 dtype（fp16 时 16），`reg_usage` 会小一倍，`block_per_SM` 的寄存器项
  会翻倍。要确认这一点同样需要在有卡的机器上跑一次 carver。
* 只建模普通 GEMM；split-K、grouped、epilogue 都不在模型里。
