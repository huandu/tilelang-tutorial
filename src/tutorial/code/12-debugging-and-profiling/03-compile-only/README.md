# 示例 3：不跑 GPU，先看编译产物（Compile-Only）

## 目的

调试“生成错误（generation issue）”的第一原则：**先看编译器到底产出了什么**，
再去猜哪一行 TileLang 写错了。`tilelang.tools.compile_only` 把 kernel 降到源码
就停下，不做 device 编译、不启动，也不需要 GPU/CUDA driver。

`add_kernel_compile_only.py` 演示两条路径：

* 程序 API：`compile_kernel_source(func, target)`；
* 命令行：`python -m tilelang.tools.compile_only --output_file out.c <本文件>`。

## 运行前提

* `--static-only`：任意机器，只校验语法并把要核对的入口打出来。
* 打印 CPU C 源码：需要装了 tilelang 且 `--target c`（任何 wheel 都支持）。
* 打印 CUDA 源码：需要 wheel 里编进了 CUDA codegen FFI（Linux wheel 有，
  macOS Metal wheel 没有），**仍然不需要 GPU**（架构钉死在 sm_80）。

## 运行命令

```bash
# 1) 任何机器
python3 add_kernel_compile_only.py --static-only

# 2) 有 tilelang 的机器：CPU C 源码
python3 add_kernel_compile_only.py

# 3) 命令行入口（与 API 等价）
python -I -m tilelang.tools.compile_only --output_file out.c add_kernel_compile_only.py

# 4) CUDA 源码（不需要 GPU，但需要 CUDA codegen FFI）
python3 add_kernel_compile_only.py --target cuda
```

## 期望输出

`--static-only`（写作机实际输出，退出码 0）：

```
py_compile: ok
需要与锁定版本核对的入口（逐个在 _vendor/tilelang-v0.1.15/ 下确认）：
  T.empty                  tilelang/language/allocate.py:400
  T.Kernel                 tilelang/language/kernel.py:431
  T.Parallel               tilelang/language/loop.py:13
  tilelang.jit             tilelang/jit/__init__.py:575
  compile_kernel_source    tilelang/tools/compile_only.py:169
（本机未安装 tilelang：这里只核对语法与入口存在性，不生成任何源码。）
```

在有 tilelang 的机器上，`--target c` 会打印类似下面这段（来自官方 compile-only 文档，
同一段 kernel 的产物）：

```c
int32_t add_kernel(half* A, half* B, half* C) {
  for (int32_t i = 0; i < 8; ++i) {
    *(half8*)(C + (i * 8)) = (*(half8*)(A + (i * 8)) + *(half8*)(B + (i * 8)));
  }
  return 0;
}
```

**判据**：生成的源码里出现 `add_kernel` 函数、`half8` 向量化写法与 `threadIdx`（CUDA 目标）
/ 8 元素循环（C 目标），且没有报错。

## 注意事项

* compile-only **只做 lowering**，不调用 nvcc，因此拿不到 PTX/SASS。
* `--target auto` 会被拒绝：它需要探测设备；compile-only 要求显式目标。
* 输入文件会被当作普通 Python 模块导入，顶层代码会执行；示例里不要写“启动 kernel”
  这类语句，只定义 kernel。

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过，5 个入口逐个在 `_vendor/tilelang-v0.1.15/` 里核对；
**本机未安装 tilelang，未实际生成过源码**。上面那段 C 产物来自官方文档，不是本机运行结果。
