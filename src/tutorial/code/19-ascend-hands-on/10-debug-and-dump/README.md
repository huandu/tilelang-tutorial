# 示例 10：T.printf 与 T.dump_tensor（A 树）

## 目的

本章 §10.1–§10.2。演示昇腾设备端打印：

* `T.printf(format, *args)`：打印标量（`%d`/`%f`/`%x`/`%s`/`%p`）；
* `T.dump_tensor(tensor, desc, dump_size, shape_info=())`：dump `ub_buffer` / `l1_buffer` /
  `l0c_buffer` / `global_buffer`（不用区分类型），`desc` 是 uint32 的自定义标记。

## 关键前提：必须打开调试打印

推荐做法是 **kernel 级 `compile_flags`**：

```python
@tilelang.jit(target="ascendc", compile_flags=["-D_DEBUG", "--cce-enable-print"])
```

`TL_PTO_DEBUG=1` **只对 `target="pto"` 生效**，不会给 AscendC 后端打开调试 flag
（见 `../references.md` 的「待补资料」第 3 条：A 树 `examples/print/` 里的老写法与
文档不一致）。

## 运行前提

* Ascend NPU + CANN（bisheng 支持 `--cce-enable-print`）+ `torch_npu` + A 树；
* `block_M` 能被 `2 * VEC_NUM` 整除。

## 运行命令

```bash
python3 print_and_dump.py --m 16 --n 16 --block-m 16 --block-n 16
```

## 期望输出

除了正确性输出，还会看到每核一份的 DumpTensor 头和标量打印，形如：

```
-----cid:0 vid:0-----
===========a_ub after copy (desc=222):
opType=..., DumpHead: AIV-0, CoreType=AIV, ...
CANN Version: XX.XX, TimeStamp: ...
DumpTensor: desc=222, addr=0, data_type=float16, position=UB, dump_size=8
...
Kernel Output Match!
```

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`printf`、`dump_tensor` 的签名与
`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534,582`、
`docs/tutorials/print.md`、编程指南 §5.1 逐条比对过。

## 性能与容量提醒

* 打印会明显拖慢算子；**上线务必关掉调试 flag**。
* 每个核上所有 dump/printf 使用的总空间上限是 **1 MB**。
* `T.printf`/`T.dump_tensor` 是**设备端**工具；Host 侧调试用 Python 的 `print`。
