# 02-moe-routing · 路由与重排的可逆性

**目的**：把 MoE 的非矩阵乘部分走一遍：gating → top-k → 按专家重排 →
生成 block metadata → 专家计算 → combine（scatter-reduce）。证明「重排 + 聚合」
与「不重排的参考实现」等价，并量化不均衡与 padding 浪费。

**运行前提**：Python 3.9+，纯标准库。

**运行命令**

```bash
python3 routing_sim.py
python3 routing_sim.py --tokens 96 --experts 8 --top-k 2 --seed 7
```

**预期输出（判据）**

* `重排 vs 参考的最大误差` 应为 `0.000e+00`（参考与重排用同一个确定性函数，
  所以可精确相等）；不同实现下只要 `< 1e-12` 即通过，脚本会断言。
* `最热专家 / 平均专家` 与 `padding 后总行数`：默认参数下热点专家约为平均值的
  3 倍以上，`block_M=32` 时 padding 浪费超过 100%。
* 末尾打印 `block_metadata` 的前几行 `[expert, m_start, valid_m]`。

**作者实际做到的验证级别**：`CPU 逻辑验证`（实际运行过，round-trip 断言通过）。
