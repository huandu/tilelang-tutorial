# 04-loops · 并行与循环

对应正文第 6 节。

## 目的

1. 把 `T.serial / T.unroll / T.vectorized / T.Parallel / T.Pipelined` 的语义摆在一起；
2. 说明 `T.Parallel` 的 extents 是**逻辑迭代空间**，线程映射由布局推导决定；
3. 给出 `T.Parallel` 默认映射的**可复算模型**（lane → thread → 每线程串行）；
4. 说明“同一次数、不同 nest 形状”会改变哪个下标落到哪个线程。

## 文件

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `loop_mapping.py` | 纯标准库模型 | **CPU 逻辑验证** |
| `loops_demo.py` | TileLang 示例 | **仅静态核对** |

## 运行 `loop_mapping.py`

```bash
python3 loop_mapping.py                                   # 默认 (128, 32)，128 线程，lane 8
python3 loop_mapping.py --extents 4 4 --threads 4 --vector 2
python3 loop_mapping.py --compare                         # 不同 nest 形状的线程归属
```

前提：Python 3.9+，只用标准库。作者在 macOS (Apple Silicon) + Python 3.11 上
实际运行过。

关键输出（真实输出，节选）：

```text
$ python3 loop_mapping.py --extents 4 4 --threads 4 --vector 2
iteration point -> (thread, per-thread slot)
  index          flat  thread  slot
  (0, 0)            0       0     0
  (0, 1)            1       0     1
  (0, 2)            2       1     0
  (0, 3)            3       1     1
  (1, 0)            4       2     0
...
$ python3 loop_mapping.py --compare
  T.Parallel(8, 8)        total=64   threads=8   per-thread=8
      thread 0 owns (0, 0), (0, 1), (2, 0), (2, 1) ...
      thread 1 owns (0, 2), (0, 3), (2, 2), (2, 3) ...
  T.Parallel(2, 32)       total=64   threads=8   per-thread=8
      thread 0 owns (0, 0), (0, 1), (0, 16), (0, 17) ...
```

脚本里的公式直接照抄
`_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:248-271`。它算的是
**编译器会怎么映射**，不是硬件上的实测访存行为。

## `loops_demo.py` 的运行前提（未在本机运行）

* 运行前提：TileLang `v0.1.15` + NVIDIA GPU。
* 静态检查命令（本机实际执行过）：

  ```bash
  python3 -m py_compile loops_demo.py    # 通过
  ```

* 在有卡的机器上如何验证：编译 `pipelined_gemm`，用 `get_kernel_source()` 检查
  (a) `num_stages=3` 是否生成 3 份 shared buffer；(b) `T.Parallel(128)` 是否
  落成 `threadIdx.x` + 每个线程一条向量访存。
* 性能口径：本示例不做性能断言。
* 作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过 + 每个循环构造子的
  签名在 `_vendor/tilelang-v0.1.15/tilelang/language/loop.py`、
  `_vendor/tilelang-v0.1.15/tilelang/cuda/language/loop.py` 里核对过）。
  **未在硬件上运行**。

## 一处容易误判的地方

`loop_mapping.py` 显示的“每线程次数”对所有形状都一样（因为映射先把整个 nest
行优先展平）；变化的是**归属**。所以“nest 形状无所谓”是错的：它决定了
`thread 0` 读的是哪几个逻辑下标，也就决定了访存能不能合并。

## 原文出处

* `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13-357`
* `_vendor/tilelang-v0.1.15/tilelang/cuda/language/loop.py:18-87`
* `_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:112-350`
* `_vendor/tilelang-v0.1.15/docs/programming_guides/control_flow.md:47-140`
* `_vendor/tilelang-v0.1.15/docs/programming_guides/python_compatibility.md:21-46`
