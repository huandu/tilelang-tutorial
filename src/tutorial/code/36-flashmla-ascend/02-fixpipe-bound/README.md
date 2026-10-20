# 示例 02：复现 1C1V 的 bound 与 `B_TOPK > 73.14`

## 目的

把正文 §3.2–§3.3 用到的三笔算术复现成可执行脚本：

1. **FixPipe bound**（报告 `:142`）：解出 `B_TOPK > 73.14`；
2. **MTE2 Issue Queue bound**（报告 `:138`）：`16 × 2 = 32` 个 token；
3. **HF32 关闭后的 8192 周期**（报告 `:126-128`）。

## 关键公式

```
MMA 周期/块      = MMA_M * B_TOPK * (512+512) / 4096 = B_TOPK * 16      (MMA_M = 64)
FixPipe 周期/块  = (MMA_M*512 + MMA_M*B_TOPK) * 4 / 128 = 1024 + B_TOPK*2
FixPipe 不成为瓶颈  =>  B_TOPK*16 > 1024 + B_TOPK*2  =>  B_TOPK > 1024/14 = 512/7 ≈ 73.14

accumulator 回 L0C  = 64*64*512 MAC / 256 FMA-per-cycle = 8192 周期（关 HF32）
```

## 运行前提

* Python 3.8+，**无第三方依赖**；
* **不需要** GPU / NPU。

## 运行命令

```bash
python3 fixpipe_bound.py
python3 fixpipe_bound.py --assert-report
```

## 期望输出（节选）

```
== bound ③：FixPipe（报告 :142） ==
  条件 B_TOPK*16 > 1024 + B_TOPK*2  =>  B_TOPK > 1024/14 = 512/7 ≈ 73.14
    B_TOPK= 64: MMA=  1024, FixPipe=  1152 -> FixPipe 更慢（瓶颈）
    B_TOPK= 96: MMA=  1536, FixPipe=  1216 -> MMA 更慢（FixPipe 不是瓶颈）

== bound ①：MTE2 Issue Queue（报告 :138） ==
  用了 gather2（burst_count=2）后，一轮最多拷 16 * 2 = 32 个 token

== accumulator 回 L0C 的代价（报告 :126-128） ==
  => 2097152 / 256 = 8192 周期

OK: 关键数字与报告一致（:138, :142, :126-128）
```

## 作者实际做到的验证级别

**本机运行过**（macOS Apple Silicon + Python 3.11.7）。`--assert-report` 通过：
阈值 `512/7 ≈ 73.142857`、`B_TOPK=64` 时 MMA 1024 / FixPipe 1152、
MTE2 一轮 32 个 token、HF32 关闭 8192 周期，都与报告数字一致。

## 常见误读（正文 §3.3、习题 E6）

`B_TOPK > 73.14` 是 **1C1V** 的结论。最终代码用的是 **1C2V + `B_TOPK = 64`**
（`config.h:53-55`），两者不矛盾：报告正是用这条不等式**否掉 1C1V**。
不要把它当成对代码的硬约束。

## 已知局限

`128 B/cycle`、`256 FMA/cycle`、`4096 FMA/cycle` 都是**报告给的**假设值；
脚本只做算术，不代表这些数字在任意昇腾平台上都成立。
