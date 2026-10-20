# 03 · 跨线程归约模型

## 目的

把 `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/reduce.h:240-295` 的
`tl::AllReduce` 翻译成纯 Python：给定参与归约的线程数与线程步长 `scale`，

* 逐级打印每一轮的 `offset`，以及这一轮走 **shared memory + barrier** 还是
  **`shfl_xor_sync`**；
* 复现四条 `static_assert`（线程数 > 0、scale > 0、`threads % scale == 0`、
  `threads/scale` 与 `scale` 都是 2 的幂），不满足时给出与源码同义的错误；
* 把 `clear=True` / `clear=False` 的差异也算出来。

脚本里的归约用 sum 作为例子；`max`/`min` 的通信结构与 sum 完全相同，
只是把 `+` 换成 `max`/`min`。

## 运行前提

Python 3.9+。只用标准库。**不需要 GPU / NPU / TileLang。**

## 运行命令

```bash
cd 03-thread-reduce-model
python3 thread_reduce_model.py --threads 256 --scale 1
python3 thread_reduce_model.py --threads 32  --scale 1
python3 thread_reduce_model.py --threads 48           # 应当报错
python3 thread_reduce_model.py --clear false --out-init 10
python3 thread_reduce_model.py --batch 4
```

参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--threads` | 256 | 参与归约的线程域大小（`threads`） |
| `--scale` | 1 | 相邻参与者的线程步长（`scale`） |
| `--clear` | true | `clear=True` 覆盖 out；`clear=False` 与已有 out 合并 |
| `--out-init` | 0 | `clear=False` 时 out 的已有值 |
| `--batch` | 1 | batched AllReduce 的宽度；>1 时共享 barrier，但通信结构不变 |

## 期望输出（摘要）

在 macOS（Apple Silicon）、Python 3.11 上实际运行得到的摘要：

```text
threads = 256, scale = 1, reduce width = 256 -> 8 rounds
  round  offset  mechanism   detail
      1     128  shared      red_buf[threadIdx.x] = x; __syncthreads(); x ^= 128
      2      64  shared      ...
      3      32  shared      ...
      4      16  shuffle     shfl_xor_sync(-1, x, 16)
      ...
      8       1  shuffle     shfl_xor_sync(-1, x, 1)
shared-memory rounds: 3
shuffle rounds      : 5
reduced value       : 32896.0 (= sum(1..256))
```

## 重要边界

* 这是**结构模型**，不是性能模型：不含任何延迟、带宽或 barrier 成本数字。
* `batch > 1` 在脚本里只打印“barrier 数被摊薄”的说明，不模拟真实 workspace 布局。
* 昇腾侧的 reduce 走另一套指令（`ascend_reduce` 系列），本脚本不建模。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过。
**未在加速器上运行。**
