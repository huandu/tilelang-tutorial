# 图 4：execution backend 与 adapter 家族

## 1. 选择顺序：谁被 `"auto"` 选中

`"auto"` 不是“最聪明的那一个”，而是**后端声明列表里第一个当前可用的**
（`tilelang/backend/module.py:167-171`）。

```text
CUDA  (tilelang/cuda/execution_backend.py:24-33)
  1. tvm_ffi   ← 永远可用，所以 "auto" 的默认结果就是它
  2. nvrtc     需要 cuda-python（is_available 探测 :6-11）
  3. cython    需要 C++ 编译器

CuTeDSL  (tilelang/cuda/execution_backend.py:35-37)
  1. cutedsl   独立 manifest，但复用 CUDA pipeline

Ascend  (tilelang/ascend/execution_backend.py:7-21)
  1. tvm_ffi   ← 同样排在第一位
  2. cython

Metal   (tilelang/metal/execution_backend.py)
  torch        → MetalKernelAdapter（唯一一个走 "torch" 名字的后端）
```

## 2. 五个 adapter 的差异

| adapter | 源码 | host 侧产物 | 设备侧产物 | 张量转换 | stream 从哪来 | 适用场景 |
| --- | --- | --- | --- | --- | --- | --- |
| `TVMFFIKernelAdapter` | `tilelang/jit/adapter/tvm_ffi.py:48` | TVM runtime module → `executable.so` | 由 codegen 回调产出的 cubin/fatbin 或 aibin，编进模块 | TVM-FFI 的 DLPack Exchange（零拷贝） | Torch 的 current work stream；Ascend 上先装 `torch_npu` 回调（`tilelang/ascend/torch_exchange.py:6-20`） | 默认路径；CUDA 与 Ascend 都是它 |
| `CythonKernelAdapter` | `tilelang/jit/adapter/cython/adapter.py:60` | 生成 C wrapper + `.pyx`，用 C++ 编译器编成 `.so` | 同左（wrapper 里用 `<<<grid, block, smem, stream>>>` 启动） | ctypes：`arr.data_ptr()` 逐个取指针 | Python 侧取到 raw stream 后当参数传进 `lib.call`（`adapter.py:431-451`） | 需要显式传 stream、或在没有 TVM 运行时的环境里跑 |
| `NVRTCKernelAdapter` | `tilelang/jit/adapter/nvrtc/adapter.py:27` | Py wrapper + launcher | 运行期用 NVRTC 编译 CUDA C | ctypes/cuda-python 混合 | 无显式 stream 时取 `torch.cuda.current_stream().cuda_stream`（`:313-317`） | 想在运行期改源码/改编译选项、避开 nvcc 的场景 |
| `MetalKernelAdapter`（注册名 `torch`） | `tilelang/jit/adapter/torch/metal.py:14` | 无 | `torch.mps.compile_shader(...)`（`:67`） | 直接把 `torch.Tensor` 传给 MPS kernel | 无显式 stream | 只有 Metal 后端用（`tilelang/jit/kernel.py:390-391` 有 `assert is_metal_target(target)`） |
| `CuTeDSLKernelAdapter` | `tilelang/jit/adapter/cutedsl/adapter.py:23` | Python 模块（`.py` 而不是 `.c`） | CuTeDSL 生成的 Python 执行器 | 通过生成的 Python 模块接收 torch 张量 | 由生成模块处理 | 走 CuTe DSL 的 NVIDIA 新路线 |

> 记法：**adapter 名字说的是“用什么把 host 和框架接起来”，不是“用什么生成设备代码”。**
> 同一个 CUDA device 源码可以被 tvm_ffi / nvrtc / cython 三种 adapter 包装，
> 生成设备代码的是 codegen，二者在 `JITKernel._compile_artifact`（device codegen）
> 与 `_create_adapter_from_artifact`（adapter）两步里分开。

## 3. “pto”不在 v0.1.15 里

| 名字 | v0.1.15 状态 | 判据 |
| --- | --- | --- |
| `tilelang/jit/adapter/pto/` | **不存在** | `ls _vendor/tilelang-v0.1.15/tilelang/jit/adapter/` → `base.py cutedsl cython kernel_cache.py libgen.py nvrtc torch tvm_ffi.py utils.py wrapper.py` |
| `pto` target / `target_is_pto` | **不存在** | `grep -rn "pto" _vendor/tilelang-v0.1.15/tilelang/ascend/target.py` 无输出 |
| `pto` adapter | **仅 main / 未进 v0.1.15** | corpus 的 main 快照 `994b44e` 里有 `tilelang/jit/adapter/pto/{__init__,adapter,libgen,wrapper}.py` 与 `cython_wrapper.pyx` |

详见正文 5.4 节。第 30 章会展开 PTO 路线本身，本章只把它当作
“adapter 家族会被继续扩充”的证据。
