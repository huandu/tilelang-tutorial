# 图 4：本章那些算子落在 roofline 的哪一边

对应正文第 2.2、2.3 节。横轴是算术强度（FLOP/Byte，对数），纵轴是这台机器上
“理论最少时间”对应的算力（TFLOPS，对数）。机器的脊点（ridge point）在
`AI = 平衡点 = 162 FLOP/Byte`（H200 标定参数，来自 `code/01-op-traffic-model/op_traffic.py`）。

```text
   TFLOPS
   1000 ┤                                                  ┌───────────────────────  729 TFLOPS 天花板
        │                                                  │   （fp16 张量核，标定后）
    729 ┼─────────────────────────────────────────────┐────┤
        │                                             │    │
        │            compute-bound 区                  │    │  · GEMM 4096³（AI≈1365）
    300 ┼                                             │    │  · flash 风格 attention（AI≈2068）
        │                                             │    │
        │                                       脊点 ─┤────┤  ← AI = 162
    100 ┼                                             │  · attention prefill 落 SxS（AI≈42）
        │                                             │
     30 ┼                                             │
        │  memory-bound 区：带宽天花板                │
        │  y = AI × 4502 GB/s                         │
     10 ┼                                             │
        │  · LayerNorm 融合（AI≈2.0）                 │
      3 ┼  · decode attention（AI≈1.0）               │
        │  · AdamW（AI≈0.58）  · cos→sin 融合（AI≈0.5）│
      1 ┼  · 逐元素加（AI≈0.17）                       │
        └──┬──────┬──────┬──────┬──────┬──────┬──────┬──────→ AI (FLOP/Byte)
          0.1     0.5     1      5      50    500   2000
```

同样的信息用表列出来（`python3 code/01-op-traffic-model/op_traffic.py` 的实际输出）：

| 算子 | 算术强度 | 时间下界 | 判据 |
| --- | --- | --- | --- |
| 逐元素加 `y = x + b`（16.8M 元素，fp16） | 0.17 | 22.4 µs | memory-bound |
| `cos → sin`，eager 两个 kernel | 0.25 | 29.8 µs | memory-bound |
| `cos → sin`，融合成一个 kernel | 0.50 | 14.9 µs | memory-bound |
| LayerNorm（8192×4096，融合） | 2.00 | 29.8 µs | memory-bound |
| LayerNorm（同一形状，拆成 3 趟） | 1.00 | 59.6 µs | memory-bound |
| GEMM 4096³ | 1365.33 | 188.5 µs | compute-bound |
| attention prefill，先落 S×S 分数矩阵（S=4096，32 head） | 42.20 | 1.461 ms | memory-bound |
| attention prefill，flash 风格不落 S×S | 2068.00 | 380.6 µs | compute-bound |
| attention decode（KV 长 32768） | 1.01 | 119.2 µs | memory-bound |
| AdamW step（7B 参数，fp32 状态，bf16 梯度） | 0.58 | 40.423 ms | memory-bound |

这张图解释了本章最重要的一个判断：**同一个 attention，同时出现在脊的两侧。**
prefill 把 S×S 分数矩阵落到显存就掉进 memory-bound 区（AI≈42），不落就回到
compute-bound 区（AI≈2068）；decode 因为每读一个 KV 元素只做一次乘加，永远在
memory-bound 区（AI≈1）。

## 来源

* 机器参数：`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml`
  （厂商规格 × 仓库自带标定；标定的测量条件写在该文件的注释里）。
* 所有算子的算术强度与时间下界：本书 `code/01-op-traffic-model/op_traffic.py` 的实际输出，
  属公式推导，不是实测。
* roofline 判据的写法（`sol_time = max(bytes/带宽, flops/峰值)`）：
  `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md:10-27`。
