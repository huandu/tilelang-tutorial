# 第 28 章 代码示例

本章的七个示例都是**纯标准库 Python**，不需要 GPU、NPU、TileLang、PyTorch，
在本机（macOS Apple Silicon、Python 3.11）上全部实际运行过。它们不替代测量，
而是把「先算再写」这一步变成可以复算的数字。

| 目录 | 对应正文 | 它回答什么 | 需要硬件 |
| --- | --- | --- | --- |
| `01-op-ledger/` | 2.1–2.2 节 | 一个算子的 `(flops, bytes)`、算术强度、平衡点和时间下界 | 否 |
| `02-roofline-misuse/` | 2.3 节 | 八种 roofline 误读各自把答案带偏多少 | 否 |
| `03-carver-cost-model/` | 3.2 节 | carver/roller 的 cost model 逐项复算：traffic、smem、regs、waves、排序 | 否 |
| `04-tile-size-decision/` | 4 节 | 四条 tile 约束的逐条体检与「先粗算再微调」的候选表 | 否 |
| `05-fusion-split/` | 5 节 | fusion / split 的四行账本与六种真实取舍 | 否 |
| `06-case-norm-rope-quant/` | 6 节 | 完整案例：预算表 → 第一版 → 诊断 → 第二版 | 否 |
| `07-ascend-latency-model/` | 3.4、5.4 节 | 用昇腾的指令级常量判断哪条 pipe 是瓶颈、tile 最小要多大 | 否 |

## 统一约定

* **验证级别**：每个示例都是「**CPU 逻辑验证**」——脚本在本机跑过、输出与文档一致。
  没有任何一条性能数字来自本机实测（本机没有加速器）。
* **机器模型**：默认使用 `tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml`
  的**标定后**数值（带宽 4800 GB/s × 0.9380 = 4502.4 GB/s）。可用
  `--bandwidth-gbs` / `--machine` 换掉。
* **数字的类型**：全部是公式输出。凡是来自厂商文档或锁定源码的常量，都在脚本头部
  和 `../references.md` 里注明了出处。
* **不要**把这些数字当成预测。它们是下界、是判据、是相对比较。

## 运行

```bash
cd 01-op-ledger            && python3 op_ledger.py --case fused-norm-rope-quant
cd 02-roofline-misuse      && python3 roofline_misuse.py
cd 03-carver-cost-model    && python3 carver_cost_model.py --self-test --topk 8
cd 04-tile-size-decision   && python3 tile_decision.py --topk 8
cd 04-tile-size-decision   && python3 tile_decision.py --diagnose \
                                 --block-m 128 --block-n 256 --block-k 64 --stages 3 --threads 256
cd 05-fusion-split         && python3 fusion_split.py
cd 06-case-norm-rope-quant && python3 case_norm_rope_quant.py
cd 07-ascend-latency-model && python3 ascend_pipe_model.py
```

每个示例目录的 README 里都有该示例的期望输出、口径和已知局限。
