# 示例 02：昇腾最小逐元素 add（A 树）

## 目的

本章 §4 的“第一个昇腾 kernel”。用最小代码展示 `GM → UB → Vector → UB → GM`
的数据流，以及两个 `T.barrier_all()` 分别保护哪条跨 PIPE 依赖。

## 与官方样例的关系

改写自 `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py`：
保留 tile 切分、`vid` 半边、两次 barrier 的语义，去掉 CLI 之外的无关部分，
并加上“这一行在保护什么”的注释。**API 名与签名与 A 树一致，不与 M 树混用。**

写死用到的 API（逐个在 A 树里核对过）：

* `T.Kernel(..., is_npu=True)` 返回 `(cid, vid)`；
* `T.alloc_ub`（scope `shared.ub`，`tilelang/language/allocate.py:156`）；
* `T.Scope("V")`、`T.copy`、`T.barrier_all`、`T.tile.add`。

## 运行前提

* 昇腾 NPU + CANN（含 `bisheng` 与 Ascend runtime）+ `torch_npu`；
* TileLang-Ascend（A 树）：`_vendor/tilelang-ascend-v0.1.2.000`，并 `source set_env.sh`；
* M、N 分别能被 `block-m`、`block-n` 整除。

## 运行命令

```bash
python3 vec_add_adapter.py --m 1024 --n 1024 --block-m 128 --block-n 256
```

## 期望输出

```
init successful!
Kernel Output Match!
```

* **正确性判据**：`torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)` 通过。
* **性能口径**：本示例不测性能。要测请看 `../08-msprof-bench/`。

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile` 通过；`T.alloc_ub`、`T.Scope`、`T.tile.add`、
`T.barrier_all`、`is_npu=True` 逐个在 `_vendor/tilelang-ascend-v0.1.2.000/` 中比对过。

## 已知局限

* **不处理尾块**：M/N 必须整除 tile。
* 这里用全屏障换正确性，**没有流水重叠**；多缓冲版本见 `../03-elementwise-pipeline/`。
