# 示例 11：AICore exception dump（A 树）

## 目的

本章 §10.3。演示「AI Core 硬件异常发生时，自动把 kernel 的输入张量 dump 下来」。
原理（见 `../references.md` 的 A5 条）：

* 编译时在 kernel 参数末尾附一个 `ParamSizeInfo`，以 8 字节 MAGIC
  `0x474e414c454c4954`（ASCII `TILELANG`）开头；
* 异常回调在参数缓冲区里搜 MAGIC，定位后构建 `acldumpTensorInfo` 数组，
  调 `acldumpSaveExceptionInfo` 落盘。

本示例**故意写了一个越界 kernel** 来制造异常。请勿在生产配置里开这个开关。

## 运行前提

* Ascend NPU + **CANN ≥ 9.3.0**（需要 `aclrtSetExceptionInfoCallback`、
  `aclrtGetArgsFromExceptionInfo`、`acldumpSaveExceptionInfo`）+ `torch_npu` + A 树；
* `ASCEND_DUMP_PATH` / `ASCEND_DUMP_SCENE` 必须在 `import torch` 之前设置
  （本脚本通过在函数内部延迟 import 保证这一点）。

## 运行命令

```bash
python3 exception_dump.py --m 64 --n 64 --dump-dir /tmp/tilelang_exc_dump
```

## 期望输出

异常路径下：

```
AI Core exception: RuntimeError ...
  input[0] dtype=float16 shape=(4096,) min=... max=...
  input[1] dtype=float16 shape=(4096,) min=... max=...
```

dump 文件落在：

```
<ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/<kernel_name>.custom.<timestamp>
```

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`TL_ASCEND_EXCEPTION_DUMP`、
`parse_exception_dump`、`read_msaicerr_bin` 与 A 树的
`tilelang/transform/pass_config.py:62`、`tilelang/tools/ascend_exception_dump_bin.py`、
`docs/tutorials/ascend_aicore_exception_dump_zh.md` 逐条比对过。

## 已知局限

* 开启 dump 会**改变 kernel 的 ABI**（追加参数）并改变 runtime 行为，所以默认关闭；
  上线不要带这个开关。
* 不同 CANN 版本的 dump 目录结构可能不同；以你机器上 `msaicerr.py` 的输出为准。
