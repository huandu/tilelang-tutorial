# 03-api-presence-matrix —— 两棵树的 API 存在性矩阵

## 目的

正文 §6.1 的版本差异表不能靠印象写。本脚本对一组 API 名逐条扫两棵树的源码
（M 树 `_vendor/tilelang-v0.1.15/`、A 树 `_vendor/tilelang-ascend-v0.1.2.000/`），
输出「在哪棵树、在哪个文件、以什么形态存在」，并**明确打印不存在**。

重点验证的四条（正文反复强调）：

* `T.dual_copy` / `T.SimtVF` / `T.SimdVF`：只存在 M 树；
* `T.alloc_ub` / `T.alloc_L1`（大写）：只存在 A 树；
* `T.tile.*`（`ascend_tile` 命名空间）：只存在 A 树；
* PTO / AscendNPU IR 目标：M 树没有，A 树 / MLIR 仓库才有。

## 运行前提

Python 3.10+，无第三方依赖。

## 运行命令

```bash
python3 api_matrix.py
```

## 期望输出

```
api                      M tree (v0.1.15)             A tree (v0.1.2.000)
-----------------------  ---------------------------  ---------------------------
dual_copy                tilelang/ascend/language/...  MISSING
...
[ok] 0 unexpected results
```

脚本对 4 条「已知答案」做断言（`T.dual_copy` 只在 M 树等），其他条目只打印事实。

## 验证级别

**CPU 逻辑验证**（本机已运行）。只读，不导入 tilelang。
