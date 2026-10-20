# 02-inference-latency-model：推理侧的判类账本

## 目的

把第 39 章 §3.4、§3.7 的方法做成脚本：**用两个比值给一个形状判类**。

* `算力占比 = 官方 Computaion 列 ÷ 本书复算的硬件峰值`
* `带宽占比 = 官方 Memory BW 列 ÷ 硬件模型里的 HBM 带宽`

它会打印三节：

1. **口径标定**：先用一行能自己复算的表（dense FP8 GEMM 的 `2MNK`）验一下口径，
   再演示一个对不上的例子（MQA logits），说明「口径不明就只报事实」；
2. **六行判类表**：第 35 章六张性能表里各挑一行；
3. **自定义形状**（加 `--m` 时）：算算术强度与平衡点。

## 运行前提

* Python 3.9 及以上；
* **只使用 Python 标准库**。

## 运行命令

```bash
python3 inference_latency_model.py
python3 inference_latency_model.py --m 128 --n 7168 --k 16384 --dtype fp8
```

## 预期输出

关键的两节（2026-10-01 在写作机上运行，节选）：

```text
② 先用一行能自己复算的表来标定口径（正文 §3.7）
dense FP8 GEMM M=4096, N=7168, K=16384：
  2*M*N*K = 9.621e+11 FLOP；README 的延迟 1117.6 us
  → 860.8 TFLOPS，与 README 的 Computation 列（861）一致 ✅

同样的复算套到 MQA logits（Q=4096, K=8192, heads=32, dim=128）：
  → 908 TFLOPS，比 README 的 681 高 33%
  README 没有给出这一列的公式 → 只报事实、以表格列为准（第 35 章 §5.3）

① 用两个比值给六个形状判类（正文 §3.7）
shape                                                    comp      mem   c/peak   m/peak  判类
dense FP8 GEMM  M=4096, N=7168, K=16384                   861      186   99.5%    5.3%  算力（贴着 compute roof）
M-grouped FP8xFP4  #G=8, M/group=4096, K=4096             861      180   99.5%    5.1%  算力（贴着 compute roof）
FP8 inference GEMM  M=128, N=7168, K=16384                549     2456   63.5%   70.2%  两列都过半 —— 处在平衡点附近
MQA logits prefill  Q=4096, K=8192, heads=32, dim=128      681      283   78.7%    8.1%  两列都不高 —— 去看第三条流水线（例如 FixPipe）
MQA logits decode   Q=256, K=8192, heads=32, dim=128      565     2006   65.3%   57.3%  两列都过半 —— 处在平衡点附近
HC prenorm GEMM  M=8192, N=24, K=28672                     82     3463    9.5%   98.9%  带宽（贴着 memory roof）
```

自定义形状那一节的输出（`--m 128`）：

```text
M=128, N=7168, K=16384, dtype=fp8
  算术强度         251.5 FLOP/byte
  平衡点（fp8）    247 FLOP/byte
  判类             compute-bound
                   ⚠ 落在平衡点附近：读的时间和算的时间同量级，优化要同时动两边
```

## 验证级别

**CPU 逻辑验证**：在写作环境（macOS Apple Silicon、Python 3.11）上实际运行过。
它不依赖任何硬件库，因此无法、也不需要验证任何性能结论。

## 边界与常见误用

* **分子来自官方表格、分母来自源码常量**，两者都是 **Ascend 950DT + 同一份
  仓库**——这是允许做这个除法的唯一理由。换机器、换 CANN 版本，分母要重标定。
* 判类的两个阈值（0.85 / 0.45）是**本书读表的约定**，不是硬件属性。
* 这个脚本**不预测时间**。它回答的是「该看哪条流水线」（第 28 章 §2.2 的
  正确用法；第 28 章 §2.3 列了八种误读，其中 M2/M3/M7 都和这个脚本有关）。
* 不要把 MQA logits 那一行的「两列都不高」当成「这个算子效率低」——
  它的瓶颈是第三条流水线（FixPipe），不在这个二维表里（第 35 章 §5.5）。
