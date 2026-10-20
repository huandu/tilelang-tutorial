# 示例 01：README 里那段 EP 用法的最小复刻

## 目的

把 `DeepEP-Ascend/README.md:100-196` 的用法抄成一份**带注解的走查**，
并在**不依赖设备**的层面检查源码里写死的硬约束。

## 运行前提

* Python 3.9 及以上；**不需要** NPU、不需要 `torch_npu`、不需要安装 `deep_ep`。
* 本脚本不 import `torch`。它只做文本走查与整数检查。

## 运行命令

```bash
python3 ep_minimal.py            # 走查 + 约束检查
python3 ep_minimal.py --summary  # 只看约束检查结果
```

## 期望输出

完整模式会先打印一段走查（初始化 → 容量 → dispatch → combine → 反向 → 销毁），
然后打印约束检查表。`--summary` 只打印后半部分：

```text
========================================================================
约束检查（全部来自源码断言，本机只做静态核对）
========================================================================
[OK  ] rank 数为偶数                    README.md:29；csrc/kernels/comm/hccl.hpp:85
[OK  ] 专家数能被 rank 数整除               csrc/buffers/ep.hpp:139
[OK  ] topk <= 32                   csrc/kernels/ep/combine.hpp:85；dispatch 侧 kNumTopk <= warpSize
[OK  ] hidden 是 128 的倍数             csrc/kernels/ep/combine.hpp:86（hidden % (256/2) == 0）
[OK  ] num_max_tokens_per_rank > 0  csrc/buffers/ep.hpp:585（calculate_buffer_size 的断言）

配置：ranks=8, experts=256, topk=6, hidden=7168, T=16384, fp8=True
全部通过。注意：这不代表能在真机上跑通——本机没有昇腾 NPU。
```

## 怎么把它变成能跑的代码

把走查里的片段抄进真实工程，并补上 README 里省略的部分：

1. `torch.distributed.init_process_group(backend='hccl', ...)`；
2. `torch.npu.set_device(local_rank)`；
3. `topk_idx` / `topk_weights` 从上一层的 gating 产出
   （例如 TileKernels 的 `tile_kernels.moe.topk_gate`）；
4. 多节点时设置共享的 `MASTER_ADDR` / `MASTER_PORT`。

可复现的 8 卡命令见 `DeepEP-Ascend/README.md:262-265`（bf16 与 fp8 各一条）。

## 验证级别

**CPU 逻辑验证**。作者在 macOS (Apple Silicon) / Python 3.11 上实际运行过本脚本
（含 `--summary`）。**没有任何一行 DeepEP-Ascend kernel 被运行过**——
本机没有昇腾 NPU（见 [`../../../VERSIONS.md`](../../../VERSIONS.md)）。
所有「约束」都来自对源码的静态阅读，行号可在快照 commit `3b25377d04b2` 上复核。
