# 示例 02：Qwen2.5-7B 的 KV cache 账本

## 目的

回答“KV cache 到底占多少显存、paged KV 的块怎么算、并发数从哪来”：

1. 每 token / 每层 / 每条序列的 KV 字节数（GQA 4 个 KV head）；
2. 与 MHA（28 个 KV head）对照，量化 GQA 省 7 倍的代价（算术强度）；
3. paged KV 的块账与碎片上界；
4. 80 GB 卡上“还能放几条序列”随 context 的变化。

## 运行前提

* Python 3.9+，仅标准库（`argparse`、`json`）。
* **不需要 GPU / NPU，也不需要 TileLang 与 PyTorch。**

## 运行命令

```bash
python3 kv_ledger.py
python3 kv_ledger.py --json
```

## 期望输出

下面是默认参数下的**实际输出**节选：

```text
KV cache 的单位成本（Qwen2.5-7B, bfloat16）
  形状：K 和 V 各 [n_kv_heads, head_dim] = [4, 128]
  每 token 每层：2 * 4 * 128 * 2 = 2048 B = 2.0 KiB
  每 token 全模型：2048 B * 28 层 = 57344 B = 56.0 KiB

  context |    GQA(4 KV heads) |   MHA(28 KV heads) |  GQA 占 MHA
     4096 |         224.00 MiB |           1.53 GiB |     14.3%
    32768 |           1.75 GiB |          12.25 GiB |     14.3%
   131072 |           7.00 GiB |          49.00 GiB |     14.3%

  GQA 每 token 比 MHA 省 7 倍（= n_q_heads / n_kv_heads = 7）。
  代价：decode 的算术强度也只有 7.0 FLOP/Byte（MHA 是 1.0）。

paged KV 的块账
block_size |         每层每块 |          全模型每块 |  块内浪费比例上界
        16 |    32.00 KiB |     896.00 KiB |         93.8%
       128 |   256.00 KiB |       7.00 MiB |         99.2%

concurrency 与 context 的取舍（H100 80 GB，粗略）
  权重占用（bf16，按模型卡 7.61B 参数）: 14.17 GiB
  留 10% 给运行时与激活后，KV 预算     : 57.83 GiB
  context |        每条序列 KV |       可并发序列数
     8192 |         448.00 MiB |          132
   131072 |           7.00 GiB |            8
```

## 性能口径

**本示例不测量任何性能，也不测量任何真实占用。** 全部是字节数的乘法。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过；输出与
上面一致。

## 已知局限

* 不含分配器碎片、对齐 padding、`block_table` 与序列长度数组等元数据；
  真实占用一定比账本大。
* 80 GB 卡上的并发数是“权重 + KV”的估算，忽略激活、框架常驻内存、
  CUDA context 与 TP 切分；真实可并发数要实测。
* `hidden_size=3584 / head_dim=128 / intermediate_size=18944` 三个字段是
  由模型卡的参数量反推的（见 `../01-prefill-decode-roofline/`），
  不是模型卡直接给出的字段；`../../references.md` 的“待补资料”里记了这一条。
