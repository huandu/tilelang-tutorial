# 02-ir-snippet-extract — 从锁定版本里抽取真实 IR 片段

## 目的

正文里的每一段 TIR 都不是手写的：本脚本从锁定版本 `_vendor/tilelang-v0.1.15/`
里把片段**原文**抽出来，并打印来源路径与行号，方便读者复核。

抽取的三个来源：

1. `examples/eager_jit/eagerjit.en.ipynb` 的**输出单元**——它们是前端
   `print(prim_func)` 的真实 dump（含 `T.match_buffer`、`T.launch_thread`、
   `T.block("tilelang_root")`、`T.alloc_buffer(scope="local.var")`）。
2. `src/transform/materialize_kernel_launch.cc` 的文件头注释——它把
   `T.Kernel` 物化**之前**的 launch nest 写成了一段 IR 草图。
3. `docs/compiler_internals/inject_fence_proxy.md`——`InjectFenceProxy` 的
   before/after 一对 IR。

## 运行前提

* Python 3.10+（标准库 `json` / `re`）。
* checkout 内存在 `_vendor/tilelang-v0.1.15/`。
* 不需要 tilelang、torch、GPU、NPU。

## 运行命令

```bash
python3 extract_ir_snippets.py
python3 extract_ir_snippets.py --only notebook
python3 extract_ir_snippets.py --only header
python3 extract_ir_snippets.py --only fence
```

## 预期输出

每个片段前都有一行 `### ...（相对路径:行号）` 作为出处标注。抽出的文本应与
正文 3.2 / 4.2 节引用的片段逐字一致。

## 验证级别

**CPU 逻辑验证**：作者在本机跑通并核对了输出。脚本本身只读文件，不编译、不执行
任何 kernel。
