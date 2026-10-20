# 示例 6：昇腾侧的运行期调试

## 目的

昇腾没有 CUDA 的 device `printf` 语义可直接用：设备端打印要靠 Bisheng 编译选项打开，
而且**每个核上的 dump/print 空间上限是 1 MB**。因此昇腾侧有两条互补路径：

* **主动打印**：`T.printf(fmt, *args)` 与 `T.dump_tensor(tensor, desc, dump_size, shape_info)`，
  在 kernel 作用域打开 `-D_DEBUG --cce-enable-print`。
* **事后取现场**：AI Core 硬件异常时由 CANN 回调把输入张量落盘，再用
  `parse_exception_dump()` 读回 numpy 数组；由 `TL_ASCEND_EXCEPTION_DUMP` 开关控制，
  默认关闭。

## 运行前提

* `--static-only`：任意机器。
* 实际运行：昇腾 NPU + CANN >= 9.3.0（异常 dump 依赖
  `aclrtSetExceptionInfoCallback` / `aclrtGetArgsFromExceptionInfo` / `acldumpSaveExceptionInfo`）。
* **`ASCEND_DUMP_PATH` / `ASCEND_DUMP_SCENE` 必须在 `import torch` / ACL 初始化之前设置**，
  否则 CANN 读不到。

## 运行命令

```bash
python3 ascend_dump_demo.py --static-only          # 任何机器
python3 ascend_dump_demo.py --demo print           # 需要 NPU
python3 ascend_dump_demo.py --demo dump            # 需要 NPU
```

## 期望输出

`--static-only`（写作机实际输出，退出码 0）：

```
py_compile: ok
需要与锁定版本核对的入口：
  T.printf（昇腾适配器）                    _vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534
  T.dump_tensor（昇腾适配器）               _vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:582
  TL_ASCEND_EXCEPTION_DUMP           _vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:62
  parse_exception_dump()             _vendor/tilelang-ascend-v0.1.2.000/tilelang/tools/ascend_exception_dump_bin.py:67
  T.print（主干昇腾方言）                    _vendor/tilelang-v0.1.15/tilelang/ascend/language/print.py:31
（本机没有 NPU：这里只核对语法与入口存在性，未运行任何 kernel。）
```

`--demo print` 会在生成源码里检查设备端打印调用是否出现；`--demo dump` 只演示编译开关与
解析入口，不真正制造硬件异常。

`T.dump_tensor` 的输出形如（来自 corpus 官方 `print.md`）：

```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, ...
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=111, addr=0, data_type=float16, position=UB, dump_size=32
```

异常 dump 落盘路径（来自 corpus 白皮书）：

```
<ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/<kernel_name>.custom.<timestamp>
```

**判据**：生成源码里出现 `printf` / dump 调用；异常 dump 场景下 `parse_exception_dump()`
返回非空列表。

## 注意事项

* `TL_PTO_DEBUG=1` 是 **`target="pto"` 的旧式全局兜底**，不会打开 AscendC 的调试选项；
  AscendC 请用 kernel 作用域的 `compile_flags=["-D_DEBUG", "--cce-enable-print"]`。
* dump 依赖 CANN 版本；异常回调会改变 CANN 运行时行为，开启后 kernel 签名会多带参数，
  编译产物与不开时不同——只在排障时开。
* 每个核的 dump 空间上限 1 MB，超出的内容会被丢弃。

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过，5 个入口逐个核对；**本机没有 NPU，未运行任何 kernel**。
上面两段输出格式来自 corpus 官方文档，不是本机运行结果。
