# 示例 01：DeepSelect 的纯 Python 复刻

## 目的

把报告
（`../../../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/docs/DeepSelect-deep-dive.zh.md:17-39`）
里的伪代码**逐行翻译成能跑的 Python**，并用 `torch.topk` 作真值对拍。它验证三件事：

1. 算法选出的 top-k 与 `torch.topk` **逐值一致**；
2. 满足 top-k 的定义性质：`min(选中) >= max(未选中)`；
3. 计数量 `W`（所有 compact 调用的输入长度之和）确实落在报告给的界
   `(1+k/B2)·L·H_m` 以内，而且远小于 `N`。

compact 步骤在报告里是设备端的 radix select；本脚本用 `torch.topk`（缺失时用
`heapq.nlargest`）代替——它只影响常数，不影响算法结构。

## 运行前提

* Python 3.9+；标准库即可运行。
* PyTorch 可选：装上就用它做 compact 与真值；没装自动回退到 `heapq`。

## 运行命令

```bash
python3 deepselect_python.py                      # N=262144, k=512, B=B2=1024, 随机输入
python3 deepselect_python.py --n 1048576 --trace  # 更大的一行 + 打印阈值/缓冲区轨迹
python3 deepselect_python.py --no-torch           # 强制 heapq 回退路径
python3 deepselect_python.py --input ascending    # 换一种输入分布
```

## 期望输出

写作机（macOS, Python 3.11.7, torch 2.1.1 CPU）上的**实际输出**：

```
compact step      : torch.topk
N=262144  k=512  B=1024  B2=1024  input=random
theoretical E[W]  : <= (1+k/B2)*L*H_m = 23,517

trial 0: W=    9,198  compactions=  5  max_buffer= 2,048  threshold=+0.9980  values_match=True  topk_property=True    20.0 ms
trial 1: W=    9,124  compactions=  5  max_buffer= 2,048  threshold=+0.9980  values_match=True  topk_property=True    14.1 ms
trial 2: W=    9,070  compactions=  5  max_buffer= 2,048  threshold=+0.9980  values_match=True  topk_property=True    14.6 ms

max observed W    : 9,198   (bound 23,517)
W / N             : 0.0351   (i.e. the extra TopK work is this fraction of one read of the input)
all trials correct: True
```

（`W` / `compactions` / `max_buffer` / 阈值是确定性的；末尾的毫秒数在每次运行之间
会变，上限在几十毫秒量级——它只是 Python 解释器的开销，不是 kernel 时间。）

`--trace` 会再打印一段采样轨迹，可以直接看到“阈值只升不降、缓冲区在 k 与 k+B2 之间
来回”：

```
scan trace (sampled every ~10% of the blocks):
    step     threshold    buffer
       0          -inf     1,024
       1     +0.975365       648
       2     +0.975365     1,281
       3     +0.991898       631
       4     +0.991898       827
       5     +0.991898     1,033
       6     +0.991898     1,256
       7     +0.991898     1,470
       8     +0.997217       550
       9     +0.997217       617
      10     +0.997217       696
      11     +0.997217       712
```

## 验证级别

**CPU 逻辑验证**：实际运行过（含 `--no-torch` 的 `heapq` 路径、`--input
ascending/staircase/descending` 几种分布）。**没有在 GPU / NPU 上运行过任何 kernel**；
脚本里的毫秒数只是 Python 解释器的开销，与 kernel 性能无关。

## 已知局限

* 用 `torch.topk` 代替 radix select 只验证**算法层面的正确性**；它无法验证设备端
  compact 步骤的数值细节（例如 CUDA 侧用 `distort` 把浮点序映射成无符号整数序）。
* 面向教学做了简化：`x` 是 Python `list`，不是张量；`B`、`B2` 由命令行给出，
  没有按 `k` 自动取 `Theta(k)`。
* 输入若有大量相等元素，`>`（严格大于）的语义会让相等的元素被过滤掉——报告也说明
  “存在相同元素时不会产生性能下降”，本脚本用连续随机数避免并列，便于对拍。
