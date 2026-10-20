# 第 15 章 代码

本章是“算子实战”章，所以代码分两类：

* **可以在这台机器上跑出结论的**：attention 的**数学**（online softmax、
  split-KV 合并、四种 mask 变体）与**访存记账**。这些用纯 CPU 实现，结论可复算。
* **只能静态阅读的**：NVIDIA 与昇腾的 TileLang kernel 本体。写作环境没有 GPU、没有
  NPU，因此这一部分只做“文件是否存在、行号是否落在范围内、API 名是否在锁定版本里”
  的核对，**不声称跑过任何 kernel**。

六个示例互相独立，可以分别复制出去运行。

## 目录

```
code/
├─ README.md                                   本文件
├─ 01-traffic-model/traffic_model.py           纯标准库；不需要 numpy/GPU
├─ 02-online-softmax/online_softmax.py         标准库 + numpy；CPU 上可运行
├─ 03-split-kv-combine/split_kv_combine.py     标准库 + numpy；CPU 上可运行
├─ 04-block-flash-attention/
│  └─ block_flash_attention.py                 标准库 + numpy；CPU 上可运行
├─ 05-official-example-map/
│  └─ map_official_examples.py                 纯标准库；只读锁定版本的源码文本
└─ 06-ascend-map/ascend_attention_map.py       纯标准库；只读锁定版本的源码文本
```

## 验证级别总表

| 示例 | 它交付什么 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- | --- |
| `01-traffic-model/traffic_model.py` | 朴素 attention 与分块融合 attention 的 **HBM 元素流量记账**：逐项列出 `S`/`P` 的读写、K/V 的重读，以及 prefill 与 decode 两种形状的对比 | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0，输出见该目录 README |
| `02-online-softmax/online_softmax.py` | online softmax 递推的**逐步轨迹**、与精确 softmax 的逐元素一致性、块大小的无关性、`ℓ` **不单调**的构造、`exp` 上溢演示、`−∞ − (−∞) = NaN` 与 `check_inf` 的作用 | Python 3.9+ 与 numpy | **CPU 逻辑验证**：实跑过，退出码 0，`assert` 全部通过 |
| `03-split-kv-combine/split_kv_combine.py` | split-KV（flash decoding）的**合并权重公式**：加权合并 vs. 平均 vs. 直接相加；以及 fp16 partial/LSE 在 split 数增加时的误差走向 | Python 3.9+ 与 numpy | **CPU 逻辑验证**：实跑过，退出码 0 |
| `04-block-flash-attention/block_flash_attention.py` | 本章的**参考实现**：把官方 kernel 的循环逐行搬到 CPU，覆盖 full / causal+`past_len` / sliding window / sink / varlen 五个变体，逐一与精确 masked softmax 对照，并开关 `check_inf` | Python 3.9+ 与 numpy | **CPU 逻辑验证**：实跑过，退出码 0，所有变体误差 < 1e-12 |
| `05-official-example-map/map_official_examples.py` | 官方 attention 例子与本章引用 API 的**行号映射**；`--strict` 时任何行号漂移都会非 0 退出 | Python 3.9+（只用标准库）；需要 `_vendor/` 在工作树上 | **CPU 逻辑验证**：实跑过 `--strict`，`drift/miss count: 0`；**未编译、未导入、未运行**任何 kernel |
| `06-ascend-map/ascend_attention_map.py` | 昇腾侧的搬运关系表（从锁定版编程指南里解析）、两棵树的昇腾 API 行号、五个昇腾 attention 例子的结构、调优文档里的瓶颈结论 | 同上 | **CPU 逻辑验证**：实跑过 `--strict`，`drift/miss count: 0`；**未在 NPU 上运行** |

## 统一说明

* 写作环境见 [../../VERSIONS.md](../../VERSIONS.md)：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此 01–04 的所有数字都是**本机 CPU 运算的结果**（数学性质与模型输出），
  05–06 的所有“行号”都是**静态阅读锁定版本源码的结果**。
* 本章正文里出现的**性能数字**只有两处，都来自锁定版本仓库自带的文档
  （昇腾 FA 的调优文档与主干仓库的昇腾 FA README），正文与 `references.md` 都标注了
  “厂商自报，作者未复现”与测量条件。
* 版本一律为 TileLang `v0.1.15`
  （commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）与 TileLang-Ascend
  `TileLang-ascend-v0.1.2.000-release`
  （commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。
* 01–04 只依赖 `numpy`；05–06 只读 `_vendor/` 下的文本文件，**不 import tilelang**，
  所以在没有安装 TileLang 的机器上也能跑。

## 建议的阅读顺序

1. 读完正文第 1、2 节后跑 `01` 与 `02`：先建立“为什么省流量”和“online softmax
   为什么是对的”这两个直觉；
2. 读完第 3 节后跑 `04`：把五个变体的数值结论与正文里的代码片段对上；
3. 读完第 5.4 节后跑 `03`：看清 split-KV 的合并公式与精度代价；
4. 最后跑 `05` 与 `06`，拿到本章所有引用的“文件:行号”清单，方便自己到锁定版本里
   继续读下去。
