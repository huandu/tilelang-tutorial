# 示例 06：decode attention 的 split-KV 骨架（TileLang）

## 目的

把 flash decoding 的两段结构写成可读的 TileLang 代码：

1. **split 段**：把 KV 方向切成 `num_splits` 段，每段独立算一份
   “已归一化的部分输出”`Output_partial` 与“本段的 lse”；
2. **combine 段**：按 `ℓ_s / Σ ℓ` 加权合并（ℓ 由 lse 还原）。

这个骨架相对官方示例做了三处简化，写在文件头注释里：只做 decode
（`seqlen_q = 1`）、KV 用 varlen 的 `cu_seqlens_k`、按比例切 split 边界。

## 运行前提

* NVIDIA GPU（官方示例用 Hopper；`example_gqa_decode.py` 对 sm89 与更高
  有不同的 heuristic 配置）+ CUDA + PyTorch(CUDA) + **TileLang v0.1.15**。
* `dim = 128`、`heads % kv_heads == 0`、`num_splits <= min_kv`。
* **本机不具备这些条件**，因此本文件从未被编译或运行。

## 运行命令

先在官方示例上确认环境可用（官方示例自带参考实现与 `do_bench` 计时）：

```bash
cd _vendor/tilelang-v0.1.15/examples/flash_decoding
python3 example_mha_inference.py
python3 example_gqa_decode.py --batch 1 --heads 32 --groups 8 --kv_seqlen 8192 --dim 128
```

再把这个骨架接到参考实现上（骨架本身只提供 kernel，不提供参考实现）：

```bash
python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('k', 'decode_attention_splitkv.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.build())
"
```

## 期望输出

在有卡机器上应当这样判定（本节未实际执行）：

* **正确性判据**：与 `example_mha_inference.py` 或 `example_gqa_decode.py`
  的 `ref_program` 对齐，`torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)`；
  官方示例用的是相似度指标 `assert_similar(..., eps=1e-2)`。
  还要与“不分 split 的单 kernel 版本”结果一致：split 只是并行化，
  不改变数学。
* **性能口径**：延迟用 `kernel.get_profiler().do_bench(warmup=500)`
  （官方示例的用法），或 `tilelang.profiler.do_bench`（默认 25 ms 预热、
  100 ms 测量、flush 256 MB L2）。吞吐换算用
  `2 * batch * heads * kv_seqlen * dim` FLOP（QK 与 PV 各一半，
  见 `example_gqa_decode.py` 的 `qk_flops` / `pv_flops`）。
* 必须一起报的变量：`batch`、`heads`、`groups`、`kv_seqlen`、`dim`、
  `num_splits`、`block_N`、`num_stages`、卡型、是否开了 warp specialization。
* 调参空间照抄官方（`example_gqa_decode.py:16-23`）：
  `block_N ∈ {64, 128}`、`block_H ∈ {64}`、`num_splits ∈ {1, 2, 4, 8}`、
  `num_stages ∈ {1, 2, 3}`。

## 作者实际做到的验证级别

**仅静态核对。** 具体做了三件事：

1. `python3 -m py_compile decode_attention_splitkv.py` 通过；
2. 逐个在 `_vendor/tilelang-v0.1.15/` 里比对 API 名：`T.Kernel`、
   `T.alloc_global/alloc_shared/alloc_fragment`、`T.copy`、`T.fill`、
   `T.clear`、`T.gemm`、`T.Pipelined`、`T.Parallel`、`T.serial`、
   `T.reduce_max`、`T.reduce_sum`、`T.exp2`、`T.log2`、`T.ceildiv`、
   `T.if_then_else`、`T.infinity`、`T.GemmWarpPolicy`；
3. 与官方示例逐行对照数据流（split 段写两样、combine 段先减最大值再加权）。

**没有在 GPU 上编译或运行过，也没有任何实测数字。**

## 已知局限

* 骨架不含 autotune 装饰器，也不含 TMA 相关 flag（官方示例对 `seqlen_q`
  方向的 TMA 用了 `disable_tma=True` 的 fallback），也不含 causal 分支。
* 没有 paged KV 的 `block_table` 间接寻址：那需要把 `T.copy` 的源地址换成
  查表结果，见第 15 章 4.6 与适配器的 `paged_flash_attn_bhsd.py`。
* `num_splits` 太小会并行度不足，太大则 combine 段的流量与启动开销占主导；
  骨架把它留给调用方，没有给出任何推荐值。

## 可复现性

把本目录整体复制到有卡机器上即可；文件不依赖其他章节的任何文件。
