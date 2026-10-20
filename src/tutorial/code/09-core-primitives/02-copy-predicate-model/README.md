# 02 · `T.copy` 越界谓词模型

## 目的

把 `_vendor/tilelang-v0.1.15/src/op/copy.cc:449-525` 的行为翻译成一段可直接运行的
纯 Python：给定源/目标 buffer 大小、拷贝偏移与 tile 长度，逐 lane 算出

* 源侧是否越界（越界时取什么值）；
* 目标侧是否越界（越界时跳不跳这次写）。

脚本还会复现源码里的「短路」：当 `CanProve` 能证明整个区域都在界内时，
`MakePredicate` 返回未定义表达式，一条谓词都不生成。

## 运行前提

Python 3.9+。只用标准库。**不需要 GPU / NPU / TileLang。**

## 运行命令

```bash
cd 02-copy-predicate-model
python3 copy_predicate_model.py                                  # tail-tile 越界场景
python3 copy_predicate_model.py --src-size 32 --tile 8 --offset 24
python3 copy_predicate_model.py --safe-value -1.0                # 改安全值
python3 copy_predicate_model.py --dst-size 4                     # 触发目标侧越界
```

参数：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--src-size` | 12 | 源 buffer 的元素数 |
| `--dst-size` | 12 | 目标 buffer 的元素数 |
| `--offset` | 8 | 从源 buffer 的哪个下标开始拷 |
| `--tile` | 8 | 拷贝多少个元素 |
| `--safe-value` | 0.0 | 源越界时使用的安全值（对应 `src_oob_safe_value`） |

## 期望输出（摘要）

在 macOS（Apple Silicon）、Python 3.11 上实际运行得到的摘要：

```text
src buffer  : 12 elements, source values = [100.0, 101.0, ..., 111.0]
dst buffer  : 12 elements
copy region : src[8:16] -> dst[0:8]   (tile = 8)

src predicate: generated (8 + 8 > 12 or negative start)
dst predicate: not generated (0 + 8 <= 12, proved safe)

 lane  src_index  in_bounds      value  stored
    0          8        yes      108.0  yes
    1          9        yes      109.0  yes
    2         10        yes      110.0  yes
    3         11        yes      111.0  yes
    4         12         no        0.0  yes     <- safe_value
    5         13         no        0.0  yes
    6         14         no        0.0  yes
    7         15         no        0.0  yes

guarded src lanes: 4 / 8
stored lanes     : 8 / 8

pseudo IR:
  value = T.if_then_else(src_pred, BufferLoad(A, [8 + i]), 0.0)
  C[0 + i] = value            # no dst guard needed
```

`--dst-size 4` 时末尾会多出一行 `if dst_pred: C[0 + i] = value`，
`stored lanes` 变成 4 / 8：目标侧越界是**不写**，而不是写替代值。

## 重要边界

* 这是 1D 的模型。真实实现按每个非 1 维度分别生成条件（`MakePredicate` 是一个循环）。
* 「`CanProve` 短路」在脚本里用一条显式的静态判断近似；真实实现用的是 TVM 的
  `arith::Analyzer`。
* 本脚本**不**模拟 TMA 的 `oob_fill`，也不模拟昇腾的 32B pad。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过。
**未与任何真实编译产物对照。**
