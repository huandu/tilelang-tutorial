# 示例 02：一层 MoE 的 dispatch / combine 字节账

## 目的

回答「一层 MoE 到底要搬多少字节」，并展示 EP 越大越费带宽的趋势。
口径与 DeepEP-Ascend 的测试代码一致：

* dispatch payload = `hidden(b)/token` 的字节 + SF 字节 + `topk*8`（int64 索引）
  + `topk*4`（float32 权重）——对应
  `tests/ep/test_ep.py:404-406` 的 `count_bytes(x, topk_idx, topk_weights) / num_tokens`；
* combine payload = BF16 hidden + `topk*4`；
* 每 rank 接收 token 数 ≈ `T × R·(1-(1-1/R)^k)`（均匀路由假设）。

## 运行前提

Python 3.9+，仅标准库。不需要 torch、不需要设备。

## 运行命令

```bash
python3 comm_ledger.py
python3 comm_ledger.py --layers 61        # 按 V3 量级的 61 层汇总
python3 comm_ledger.py --bf16-dispatch    # dispatch 也用 BF16，看差别
python3 comm_ledger.py --json             # 机器可读
```

## 期望输出

```text
每 token payload（本书复算，口径 = DeepEP-Ascend/tests/ep/test_ep.py:404-406）
  hidden=7168, topk=6, experts=256, T=16384, layers=1
  dispatch: 7,464 B/token (hidden 7,168 + sf 224 + idx 48 + w 24)
  combine : 14,360 B/token (bf16 hidden 14,336 + w 24)

  EP   专家/rank     期望目标rank    接收token/rank  dispatch/rank   combine/rank
   8        32        4.410          72,248       514.3 MB       989.4 MB
  16        16        5.137          84,165       599.1 MB         1.1 GB
  32         8        5.550          90,937       647.3 MB         1.2 GB
  64         4        5.770          94,543       673.0 MB         1.3 GB
 128         2        5.884          96,404       686.2 MB         1.3 GB
```

（末尾还有一段「均匀路由假设下的上界量级」提醒，见脚本输出。）

## 怎么读这张表

* **EP 越大，跨设备字节越多**：EP8 → EP128，期望目标 rank 数从 4.41 涨到 5.88，
  每 rank 的接收 token 数跟着涨；单卡的线宽不变，于是单位时间能完成的逻辑 token 数下降。
* **combine 的 payload ≈ dispatch 的 1.9 倍**（7168 fp8 vs 14336 bf16），
  这正是 §7.3 讲 combine 更慢的第一条原因。
* 表中 `dispatch/rank` 是**每 rank 的接收量**；跨 rank 合计要把所有 rank 加起来
  （脚本末尾给了一个平均合计值）。

## 已知的近似与局限

* 均匀路由假设：真实 MoE 有热点专家，去重率与接收数都会更差。
* 忽略 padding：`expert_alignment` 带来的对齐空隙不在这张账里（见 §2.2）。
* 忽略本地拷贝：落在本 rank 的 token 不占网络，本表按「跨设备」口径算，
  所以是**上界**。

## 验证级别

**CPU 逻辑验证**：作者在 macOS (Apple Silicon) / Python 3.11 上实际运行过。
**不是实测流量**，只是公式复算。
