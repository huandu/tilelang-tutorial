# 02-dtypes · 数据类型与低精度表达

对应正文第 4 节。

## 目的

1. 把 v0.1.15 暴露的 **173 个 dtype 名字**按家族摊开，并核对回源码；
2. 说明 int4 / bool / `float4_e2m1_unpacked` 三种“没有向量形”的例外；
3. 说明 fp8 的八个拼法、fp6 的两个拼法、fp4 的两种存储模型；
4. 说明混合精度时累加 dtype 必须显式给出。

## 文件

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `dtype_families.py` | 纯标准库清单 + 比对 | **CPU 逻辑验证** |
| `fp8_fp4_exprs.py` | TileLang 示例 | **仅静态核对** |

## 运行 `dtype_families.py`

```bash
python3 dtype_families.py            # 家族清单
python3 dtype_families.py --lanes    # lane 展开计数
python3 dtype_families.py --diff     # 与 vendor 的 _all_dtypes 逐条比对
```

前提：Python 3.9+，只用标准库。作者在 macOS (Apple Silicon) + Python 3.11 上
实际运行过。

关键输出（真实输出，非示意）：

```text
$ python3 dtype_families.py --diff
vendor entries: 173; transcription entries: 173
exact match

$ python3 dtype_families.py --lanes
Lane expansion (names exported as attributes of tilelang.language)
  aliases (no lane forms)         7 names
  no-lane scalars                 3 names
  signed int                      4 names
  unsigned int                    4 names
  float                           3 names
  float8                          8 names
  float6                          2 names
  bfloat16/tfloat32/float4       16 names
  TOTAL                         173 names
```

`--diff` 会去读
`../../../_vendor/tilelang-v0.1.15/tilelang/language/dtypes.py`，从
`_all_dtypes = [...]` 里把字符串抠出来做集合比较。所以这个脚本同时是一个
**回归护栏**：如果哪天换了 pin、`_all_dtypes` 变了，`--diff` 会立刻报差异。

## `fp8_fp4_exprs.py` 的运行前提（未在本机运行）

* 运行前提：TileLang `v0.1.15`；其中 `fp8_gemm` 需要支持 fp8 的 NVIDIA GPU
  （sm89+），`fp4_storage_model()` 只查 dtype 属性，理论上不需要卡。
* 静态检查命令（本机实际执行过）：

  ```bash
  python3 -m py_compile fp8_fp4_exprs.py   # 通过
  ```

* 在有卡的机器上如何验证：

  ```python
  from fp8_fp4_exprs import fp4_storage_model
  print(fp4_storage_model())   # {'packed_bits': 4, 'unpacked_bits': 8, ...}
  ```

  对 `fp8_gemm` 则用 `torch.float8_e4m3fn` 输入与 fp32 参考做对照
  （fp8 的容差按用例自行设定，本示例不承诺具体容差）。
* 性能口径：本示例不做性能断言。
* 作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过 + 每个 dtype 符号
  与 helper 在 `_vendor/tilelang-v0.1.15/tilelang/language/dtypes.py` 与
  `fp8.py` 里核对过）。**未在硬件上运行**。

## 原文出处

* `_vendor/tilelang-v0.1.15/tilelang/language/dtypes.py:105, 120-142, 187-253,
  262-317, 319-347, 633-711, 891`
* `_vendor/tilelang-v0.1.15/tilelang/language/fp8.py:1-49`
* `_vendor/tilelang-v0.1.15/docs/programming_guides/type_system.md:7-41`
