# 示例 5：用 compile_only 在无卡机器上看编译产物

## 目的

正文把「一次编译」拆成 lower → codegen → adapter 三段。本示例在**没有 GPU**的
机器上走完前两段：

1. `scale_add.get_tir(N, alpha)` —— 与 JIT wrapper 用的是同一个入口
   （`tilelang/jit/__init__.py:379`），停在 lowering 之前；
2. `tilelang.tools.compile_only.compile_kernel_source(prim_func, target=...)`
   —— 内部就是 `tilelang.lower(..., enable_device_compile=False)`
   （`tilelang/tools/compile_only.py:169-200`），不碰驱动、不产出 cubin；
3. 打印 elaborated PrimFunc 上**已经存在**的属性，并指出哪两个属性是
   到 `JITKernel` 构造时才定下来的；
4. 打印这次调用会进入 cache key 的字段（`kernel_cache.py:269-280`）。

## 运行前提

* 一台能 `pip install tilelang==0.1.15` 的机器（Linux x86_64 / aarch64 均可）；
* 默认 `--target c`（CPU C 代码生成）在任何机器上都可用；
* `--target cuda` 需要 wheel 里带 CUDA codegen FFI：Linux wheel 有，
  macOS Metal wheel **没有**，会明确报 `CUDA codegen FFI missing`；
* **不需要** GPU、CUDA 驱动、CANN。

## 运行命令

```bash
cd chapters/26-runtime-jit-abi/code/05-compile-only-inspect
python3 inspect_via_compile_only.py                 # 默认 target=c，写 out.c
python3 inspect_via_compile_only.py --target cuda   # 只生成 CUDA C，不编译成二进制
python3 -I -m tilelang.tools.compile_only --output_file out.c inspect_via_compile_only.py
```

## 期望输出

四段：TIR 文本、生成的设备源码、属性表、cache key 字段。设备源码里应能看到
`int32_t`（CPU 目标）或 `__global__`（CUDA 目标）形式的 kernel 入口，
形如官方文档给的例子：

```c
int32_t scale_add_kernel(float* A, float* B, float* Out, float alpha);
```

## 正确性判据

* 退出码 0；
* `out.c` 非空且包含一个 `*_kernel` 函数；
* 属性表里**没有** `tilelang_out_idx`（lazy 模式下它只在 derived PrimFunc 上出现），
  这一点本身就是正文 4.7 节的判据。

## 这个示例**没有**给出 host 侧的 packed 入口

`compile_only` 只调用 `enable_device_compile=False`，**不开** host codegen，
所以输出的是 device 半边。要看 `(self_handle, args, num_args, result)`，
需要在真实设备上跑起来后用 `JITKernel.get_host_source()`
（`tilelang/jit/kernel.py:545-552`）取。

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile` 通过；`compile_kernel_source` 的签名与
行为说明逐行核对 `tilelang/tools/compile_only.py:169-200`；
`JITImpl.get_tir` / `out_idx` / `execution_backend` / `pass_configs` /
`compile_flags` 属性名与 `_vendor/tilelang-v0.1.15/` 一致。
本机未安装 tilelang，**未运行**。
