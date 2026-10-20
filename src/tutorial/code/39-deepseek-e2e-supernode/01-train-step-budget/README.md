# 01-train-step-budget：训练一步的账本

## 目的

把第 39 章 §2.2、§2.8、§4 的公式复算成一个可运行的脚本：

1. **通信账**（口径 A）：每 token 每方向的逻辑 payload `k·d·b`；
2. **去重修正**：一个 token 期望覆盖多少个不同的目标 rank（`R·(1-(1-1/R)^k)`）；
3. **全集总量**（口径 C）与它相对链路带宽的时间下界；
4. **算力账**：本 rank 上这一层的专家 GEMM（`2MNK`）与算术强度；
5. **三类资源的口径**：算力 / 片内带宽 / 卡间带宽分别是哪一类来源。

它**不测任何东西**，也不预测端到端时间。它的用途是**判类**——
回答「这一层该去哪条流水线上找瓶颈」。

## 运行前提

* Python 3.9 及以上；
* **只使用 Python 标准库**——不需要 TileLang、PyTorch、NPU/GPU、CANN。

## 运行命令

```bash
# 默认参数取自 DeepEP-Ascend README 的测量条件
python3 train_step_budget.py

# 换成你自己的模型形状
python3 train_step_budget.py --tokens 8192 --hidden 7168 --topk 8 \
    --experts 256 --ffn 2048 --dispatch-bytes 1 --combine-bytes 2
```

## 默认参数的来源

| 参数 | 默认值 | 出处 |
| --- | ---: | --- |
| `--tokens` | 16384 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:11` |
| `--hidden` | 7168 | 同上 |
| `--topk` | 6 | 同上 |
| `--experts` | 256 | 同上 |
| `--dispatch-bytes` | 1（FP8） | 同上（`:21`） |
| `--combine-bytes` | 2（BF16） | 同上 |
| `--ffn` | 2048 | **演示值，请替换成你自己的模型参数**（DeepEP 的测量条件里没有这一项） |
| `--expert-align` | 128 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:131`（本脚本在输出里用文字说明，不参与计算） |

## 预期输出

分五节：通信账 / 去重修正 / 全集总量 / 算力账 / 三类资源口径。
每节都把自己的公式与出处打印在数字旁边。

实际输出（2026-10-01 在写作机上运行，节选）：

```text
① 通信账：每 token 每方向的逻辑 payload（口径 A）
dispatch  6*7168*1 = 43008 B = 42.0 KB   (FP8，1 字节)
combine   6*7168*2 + k*4 = 86040 B = 84.0 KB   (BF16，2 字节 + 权重)

② 去重修正：一个 token 期望覆盖多少个不同的目标 rank
   EP 规模 R   distinct_ranks        相对 k 的比例
         8             4.41          73.5%
        16             5.14          85.6%
        32             5.55          92.5%
        64             5.77          96.2%
       128             5.88          98.1%
EP8 → EP128：跨设备字节量上升 33%（4.41 → 5.88）

③ 全集总量（口径 C）与它相对链路的时间下界
   EP                dispatch                 combine
    8      0.518 GB /    1381 us      1.036 GB /    2986 us
  128      0.691 GB /    2159 us      1.382 GB /    4973 us

④ 算力账：本 rank 上这一层的专家 GEMM（口径：2*M*N*K）
合计      8.659e+12 FLOP = 8.66 TFLOP
算术强度  5957.8 FLOP/byte（本书复算，只看这一层）
平衡点    fp8: 247 / bf16: 124 FLOP/byte

⑤ 三类资源与它们的口径
唯一的跨类除法：3500 / 374 ≈ 9.4  → 跨卡搬一次 ≈ 本卡搬十次（量级判类）
```

## 验证级别

**CPU 逻辑验证**：本脚本在写作环境（macOS Apple Silicon、Python 3.11）上
实际运行过，上面就是真实输出。它**不需要**、也**没有**在昇腾机器上运行过。

## 读输出时必须记住的四件事

1. **第 ③ 节的两列是「纯搬运时间的下界估算」**，不含 kernel 启动、barrier、
   padding，也没有考虑重叠。用途只有判类。
2. **第 ④ 节的 `M` 是未去重的行数上界**。真实行数要乘上第 ② 节的比例，
   再按 `expert_alignment` 向上取整（口径 B）。
3. **第 ④ 节的算术强度用的是「本 rank 上这一层」的字节口径**，
   换一个「每次调用」的边界，结果就变（第 28 章 M4）。
4. **第 ⑤ 节那个除法是全书唯一允许的跨类比较**，且只用于判类——
   它的两个数一个来自源码模型常量、一个来自厂商实测，口径并不相同（正文 §4.4）。
