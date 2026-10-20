# 示例 01：复现「算存比」表（每个 AI Core 多少个 q head）

## 目的

把官方技术报告里决定 **「每个 AI Core 负责多少 q head」** 的那段算术复现成
一个可运行、可断言的 Python 脚本。对应正文 §3.1，来源：

* `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md:85-111`
  （报告里的脚本与输出表）
* 判据一句话在报告 `:114`：「head ≥ 64 才是 compute bound」。

## 关键公式

| 量 | 公式 | 含义 |
| --- | --- | --- |
| `l2_bw` | `5e12 / 32 / 1650e6` ≈ **94.7 B/cycle** | 每个 AI Core 每周期能拿到的 L2 带宽 |
| `cube_cycles` | `h_q · topk · (512+512) / 4096` | 两个 GEMM（QKᵀ + PV）的 MAC 数 ÷ 4096 FMA/cycle |
| `mem_cycles` | `(h_q·512·4 + topk·512·2) / l2_bw` | Q+O（bf16）与 gathered KV（bf16）的字节数 ÷ `l2_bw` |

结论：比较 `cube_cycles` 与 `mem_cycles`，谁大谁是瓶颈。

## 运行前提

* Python 3.8+，**无第三方依赖**（只用标准库）；
* **不需要** GPU / NPU。

## 运行命令

```bash
python3 head_roofline.py
python3 head_roofline.py --assert-report   # 与报告的表逐行对照
```

## 期望输出（节选）

```
# L2 bandwidth per AI Core per cycle: 94.697 B/cycle
 16, topk= 512, cube_cycles=  2048, mem_cycles=  5883
 16, topk=1024, cube_cycles=  4096, mem_cycles= 11419
 32, topk= 512, cube_cycles=  4096, mem_cycles=  6229
 32, topk=1024, cube_cycles=  8192, mem_cycles= 11765
 64, topk= 512, cube_cycles=  8192, mem_cycles=  6921
 64, topk=1024, cube_cycles= 16384, mem_cycles= 12457
128, topk= 512, cube_cycles= 16384, mem_cycles=  8305
128, topk=1024, cube_cycles= 32768, mem_cycles= 13841

OK: output matches the report table (...)
```

（`--assert-report` 会打印最后那行 `OK`。）

## 作者实际做到的验证级别

**本机运行过**（macOS Apple Silicon + Python 3.11.7）。`--assert-report` 通过：
八行输出与报告 `:104-111` 逐行一致（忽略行首被 Markdown 修剪的空格）。
这些数字是**公式推导**，不是任何硬件上的实测吞吐。

## 已知局限

* 这个模型只算 Cube 的 MAC 与 L2 的字节（见正文 §3.1.1 对「它忽略了什么」的讨论）；
  它不包含 HBM→L2、L1→L0、L0C→UB、UB 读写、同步与发射开销。
* `5 TB/s`、`32 核`、`1650 MHz`、`4096 FMA/cycle` 都是报告给的**假设值**，
  不是本书测量的硬件参数。换平台请换这四个数。
