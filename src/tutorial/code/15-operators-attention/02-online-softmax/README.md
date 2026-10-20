# 示例 2：online softmax 的数值验证

## 目的

正文第 2.3–2.4 节给出了递推式并做了归纳证明。本示例把它跑起来，检查四件事：

1. **逐步轨迹**：每一块的运行最大值 `m`、重标度因子 `c = exp(m_prev − m_new)`、
   分母 `ℓ`、分子 `max|O|` 各是多少，直观看到 `c` 只在最大值变大时才小于 1；
2. **与精确 softmax 逐元素一致**；
3. **结果与块划分无关**：同一行用 16/64/128/256/1024/4096 六种块大小、以及把 key
   顺序整体倒过来，结果都在浮点误差内相同（倒序会让运行最大值的**路径**完全不同）；
4. **`ℓ` 不是单调的**：故意构造「第三个块带来一个远大于历史的分数」的例子，看到
   `ℓ` 从 2 涨到 4 又掉回 2，而 `O/ℓ` 仍然精确；
5. **两个具名失效模式**：
   * 不减去最大值时 `exp` 上溢（`inf/inf = NaN`）；
   * 运行最大值仍为 `−∞` 时 `exp2(−∞ − (−∞)) = NaN`，以及 `check_inf` 如何修掉它。

## 运行前提

Python 3.9+ 与 **numpy**。不需要 GPU/NPU、不 import tilelang。

```bash
python3 -c "import numpy; print(numpy.__version__)"   # 写作机：1.24.1
```

## 运行命令

```bash
python3 online_softmax.py
```

## 期望输出（写作机实际输出，退出码 0）

```text
1. per-step trace of the recurrence (small, exact arithmetic in fp64)
 blk   m (running max)   c = exp(m_prev-m)   l (denominator)      max|O|
   0          1.500000            0.000000          2.028371    1.255805
   1          1.500000            1.000000          2.836150    2.556749
   2          1.500000            1.000000          4.285457    6.226913
exact row max (scaled): 1.500000
max abs diff  : 4.441e-16
OK: online == exact

2. result does not depend on the block schedule
  block size   #blocks      max abs diff
          16       256         4.510e-17
         ...
        4096         1         2.082e-17
reversed key order vs reference: max abs diff = 1.388e-17

3. what the running max buys: overflow of exp without it
exp(s*x) without subtracting the max : [1.01423205e+304  inf  inf  4.60460640e+299]
naive sum_j exp(sx_j) v_j / sum exp    : [nan]   <- inf/inf = NaN
row max m = 720.0
exact (subtract max)                   : [2.9999546]
online (block = 2)                     : [2.9999546]
OK: the max is a range device, not a mathematical one

4. the running denominator l is NOT monotone
 blk         m             c               l
   0       0.0     0.000e+00        2.000000
   1       0.0     1.000e+00        4.000000
   2     100.0     3.720e-44        2.000000
online O/l = [1.], exact = [1.]
OK: l is not monotone, but O/l still equals the exact softmax

5. the -inf - (-inf) hazard and what check_inf does
exp2(-inf - -inf) = nan   <- NaN, this is the failure mode
without check_inf : nan   <- NaN propagates
with    check_inf : 1.0000000000
OK: check_inf replaces a still-infinite running max with 0 ...

All checks passed.
```

第 1 段里第一块的 `c = 0.000000` 是因为初始 `m_prev = −∞`：`exp(−∞ − 1.5) = 0`，
旧的（空的）累加器被清零，这是正确的。后面两块 `c = 1`，说明运行最大值没有再变。

## 判据

* 退出码 0，且所有 `assert` 通过（脚本内部断言：
  轨迹与精确解一致、`naive` 那一支确实是 NaN、`check_inf` 那一支确实恢复出正确值）；
* 第 2 段的六行误差都在 `1e-16` 量级；
* 第 3 段会**故意**触发 `exp` 上溢（用 `np.errstate` 抑制告警，所以 stderr 是干净的）。

## 注意事项

* 这里全部是 **fp64 CPU 算术**。真实 kernel 用 fp32 累加、fp16/bf16 的 `P`，误差会更大；
  本示例只验证**公式**，不预测精度。
* 第 4 段的 `check_inf` 是**照抄**锁定版本里的写法
  （`_vendor/tilelang-v0.1.15/examples/attention_sink/example_mha_sink_fwd_bhsd.py:103-109`），
  但这里是 CPU 重演，不是运行那个 kernel。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 + numpy 1.24.1 上实际运行过，
退出码 0，`assert` 全部通过，stderr 为空。**没有运行过任何 TileLang kernel。**
