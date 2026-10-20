# 03 · 同步点数量 vs 稳态步长

## 目的

把「漏一个同步会偶发出错，多一个同步会掉性能」的**一半**（性能那一半）变成算术：
在一个 `copy -> compute` 的 `depth` 槽环里，加不同数量的 barrier，看稳态每轮步长
怎么变。

四个变体：

| 变体 | 加了什么 | 正确吗 |
| --- | --- | --- |
| `none` | 什么都不加 | **不正确**：没有任何东西阻止 `copy[k+depth]` 覆盖 `compute[k]` 正在读的槽 |
| `war` | 每次迭代一个释放屏障（消费者读完 → 下一个写者才能写），成本 `barrier` | 正确（这是环形多缓冲的最低要求） |
| `war+raw` | 再加一个获取屏障（数据到齐 → 消费者才读） | 正确（当可见性本身需要 fence 时必需） |
| `+cb` | 再在消费者链里插一个**多余**的屏障 | 正确但慢：`barrier` 全额落在关键路径上 |
| `sync-copy` | 退化成同步拷贝（每次拷贝等上一次落地） | 正确但慢：彻底失去异步重叠 |

## 运行前提

Python 3.9+，只用标准库。不需要 GPU/NPU。

## 运行命令

```bash
python3 sync_count_model.py
python3 sync_count_model.py --latency 120 --compute 20 --barrier 8
python3 sync_count_model.py --latency 30 --compute 60 --barrier 8
```

## 期望输出（本机实际运行结果，默认参数）

```text
depth |             none |              war |          war+raw |              +cb |        sync-copy
    1 |    350cyc u= 91% |    920cyc u= 35% |   1048cyc u= 31% |   1048cyc u= 31% |   1048cyc u= 31%
    2 |    350cyc u= 91% |    476cyc u= 67% |    540cyc u= 59% |    548cyc u= 58% |    628cyc u= 51%
    3 |    350cyc u= 91% |    350cyc u= 91% |    388cyc u= 82% |    478cyc u= 67% |    628cyc u= 51%
    4 |    350cyc u= 91% |    350cyc u= 91% |    358cyc u= 89% |    478cyc u= 67% |    628cyc u= 51%
    6 |    350cyc u= 91% |    350cyc u= 91% |    358cyc u= 89% |    478cyc u= 67% |    628cyc u= 51%

Steady-state interval per iteration (last half of the loop):
  depth=3: none=  20.0  war=  20.0  war+raw=  22.6  +cb=  28.0  sync-copy=  38.0
  depth=4: none=  20.0  war=  20.0  war+raw=  20.0  +cb=  28.0  sync-copy=  38.0
```

（`u=` 是计算单元利用率 = `iterations × compute / makespan`。）

## 要读出什么

1. **必需的同步可以在深度足够时免费。** `war` 在 `depth ≥ 3` 时与 `none` 的
   makespan 完全一样：那 8 个周期的屏障被「已经在路上的那一份拷贝」盖住了。
   条件是 `latency ≳ compute + barrier`。
2. **多余的同步不会免费。** `+cb` 把屏障放进了消费者链，稳态步长从 20 涨到 28，
   正好是一个 `barrier`：这在 `depth` 加到 6 时也不会好转。
3. **深度不够时，必需品也会变成瓶颈。** `depth=1` 的 `war` 让稳态步长变成
   `latency + compute + barrier = 58`——这就是第 11 章那条
   `depth ≥ 1 + latency/compute` 判据在同步模型里的样子。
4. **彻底失去异步最贵。** `sync-copy` 在 `latency=30` 时稳态 38，在
   `--latency 120` 时稳态 128：这正是「把 `T.copy` 退化成同步拷贝 + 每语句一道
   barrier」的代价。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过。上面的数字是
**抽象周期数**，不是任何硬件上的测量值。**没有**在任何 GPU/NPU 上对照过。

## 这个模型不做什么

* 一个 `barrier` 在所有线程上只是「等到大家都到」，真实硬件上它还会随参与线程数、
  warp 分歧、以及 barrier 与访存的交互而变；这里只用一个常数 `barrier`；
* 它不区分 `__syncthreads()`、mbarrier 的 parity wait、昇腾的 SetFlag/WaitFlag：
  三种在模型里都是「一个固定成本的屏障」；
* 它不建模「漏一个同步」的**错误**行为（那需要在真实硬件上跑竞态，见正文 §5.1）。
