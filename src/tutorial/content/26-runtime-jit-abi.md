# 第 26 章 运行时、JIT、ABI 与框架集成

## 学习目标

读完本章，你应当能够：

1. 画出 `@tilelang.jit` 的完整生命周期（装饰 → `get_tir` → 缓存查询 → lower → codegen →
   adapter 包装 → 可调用 kernel），并说出每一跳的入口函数与文件:行号。
2. 准确说出 kernel 缓存键由哪些字段组成，指出磁盘缓存的目录层级，并判断“我改了
   X”会不会失效，包括那个最容易记错的 X：**C++ pass 本体**。
3. 读懂 `MakePackedAPI` 生成的 host 入口 `(self_handle, args, num_args, result)`，
   说出它插入了哪些运行时检查、按什么顺序、报错文案长什么样。
4. 说清 `tvm_ffi` / `cython` / `nvrtc` / `torch` / `cutedsl` 五个 adapter 各自把
   host 侧接到什么产物上、stream 从哪来、什么场景该换。
5. 讲清楚 TileLang 与 PyTorch 的边界：张量怎么过去、stream 怎么对齐、
   **autograd 为什么不在 TileLang 这边**。
6. 解释“启动开销”在什么条件下成为瓶颈、CUDA Graph / ACLGraph 各自压掉的是哪一段，
   并说出 v0.1.15 里哪些相关能力还不存在。

## 前置知识

* 第 4 章“架构总览与一次编译的全流程”：知道 `frontend → lower → codegen → host →
  runtime` 五段链，知道 target backend 与 execution backend 是两个概念。
* 第 7 章“快速上手”第 6 节：知道 `@tilelang.jit` 有 lazy / eager 两种模式，
  知道缓存分内存级与磁盘级。本章把那两页展开成源码级细节。
* 第 20 章“IR 与编译期管线精读”：知道 pass pipeline 的形状，知道
  `MakePackedAPI` 与 `SplitHostDevice` 在管线里的位置。本章只讲它产出什么，
  不重复它怎么被调用。

会写 Python、用过装饰器与 `functools.cache`、知道 PyTorch 的 `Tensor` 与
`torch.cuda.Stream` 就够了。**不需要**读过 TVM runtime，也不需要写过 CUDA。

---

## 1. 问题的由来

### 1.1 三个“明明应该对”的现象

写 kernel 的人迟早会撞上这三件事，而它们的答案都在本章：

* **“我改了 `T.gemm` 的调用，为什么结果还是旧的？”**
  编译产物按 key 缓存在磁盘上。改 TIR 会换 key，但改 C++ pass 本体**默认不会**
因为 key 里默认不含 `libtilelang` 的内容哈希。
* **“我把 kernel 放在另一条 stream 上跑，对答案有时候对、有时候错。”**
  adapter 会跟着 PyTorch 的 current stream 走，但你的**比较操作**未必在同一条
  stream 上。这不是 TileLang 的 bug，是跨 stream 编程的固有约定。
* **“`kernel(a, b)` 这行 Python，凭什么能接 PyTorch 张量？”**
  因为 host 侧被 `MakePackedAPI` 改成了一个固定签名的 packed function，
  张量靠 TVM-FFI 的 DLPack 零拷贝送进去。这一步是**编译期**决定的，不是运行期猜的。

### 1.2 本章的版本纪律

全书锁定 TileLang `v0.1.15`（commit `a35f8ddf`）与 TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e`），见
[../../VERSIONS.md](../../VERSIONS.md)。本章额外遵守三条：

1. **每个“文件:行号”都可核验。** `code/01-runtime-anchor-check/verify_runtime_anchors.py`
   做两层核对：172 条锚点表逐条比对“这一行应该包含什么”，再把全章 241 处行内引用
   扫一遍（文件存在、行号在范围内、不是空行）。本机跑出 `172/172` 且扫描 0 问题。
2. **不宣称未验证的运行结果。** 本机是 macOS（Apple Silicon），没有 NVIDIA GPU、
   没有昇腾 NPU、也没有安装 `tilelang` / `torch`。凡是需要加速器才能得到的结论，
   一律写成“应当是什么样”并给出验证步骤。
3. **`main` 与 v0.1.15 分开说。** “main 有、v0.1.15 没有”的东西只出现在本章
   第 6 节“前沿”，并且给出可复现的 grep 命令。

### 1.3 与相邻章节的分工

第 20 章讲 pass pipeline 内部（每个 pass 做什么、顺序如何），本章讲
**pipeline 之外的运行时**：编译产物怎么被缓存、host 侧长什么样、
adapter 怎么把它接到框架上。第 24、25 章讲两条后端的代码生成；本章只关心
“生成完了之后”。

---

## 2. 概念与模型

### 2.1 六个阶段，一条链

完整生命周期见 `figures/fig-01-jit-lifecycle.md`。这里先给一页速记版：

```text
① 装饰      @tilelang.jit              →  JITImpl（只是包装，不编译）
② 取 TIR    JITImpl.get_tir(...)       →  PrimFunc
③ 缓存查询  JITImpl.__call__ / cached() →  JITKernel 或继续
④ lower     lower_with_context(...)    →  host_mod + device_mod + params
⑤ codegen   device_codegen / host_codegen → 设备源码 + host 源码
⑥ adapter   _create_adapter_from_artifact → ==adapter.func== 可调用对象
```

有三个容易记反的地方，先点出来：

* **装饰阶段不编译。** `@tilelang.jit` 只是把函数包成 `JITImpl`，
  顺手记下源码文本与签名（`tilelang/jit/__init__.py:638-650`）。
* **缓存查询发生在 lower 之前。** 一旦命中，④⑤⑥ 全部跳过，你拿到的是上一轮
  落盘并重新 `dlopen` 进来的库（`tilelang/cache/kernel_cache.py:641-731`）。
* **出口只有一个。** 无论走哪条路，`JITKernel` 对外都是 `adapter.func`
  （`tilelang/jit/kernel.py:149`），所有框架差异收敛在第 ⑥ 步。

### 2.2 三本缓存账

“缓存”在本章指三个不同的东西，失效条件完全不同。图见
`figures/fig-02-cache-layers.md`，速记如下：

| 层 | 位置 | key 的来源 | 典型失效原因 |
| --- | --- | --- | --- |
| ① 进程内参数缓存 | `JITImpl._kernel_cache` | `JITFunc.parse_args()` 返回的 `(p1_key, p2_key)` | 换了编译期参数或张量形状；进程退出即消失 |
| ② 磁盘 kernel 目录 | `$TILELANG_CACHE_DIR/<version>/<platform>-<machine>/kernels/<sha256>/` | `sha256(json(TIR 文本 + out_idx + target + pass_configs + …))` | 上述任一字段变化；manifest 校验失败 |
| ③ 设备二进制 | 同 `$TILELANG_CACHE_DIR` 下的 `cuda-binaries/` 与 `ascend-binaries/` | `sha256(json(设备源码 + arch + 编译选项 + …))` | 设备源码或编译选项变化 |

记住 ② 与 ③ 的分工：**② 管的是“host 侧那一整套产物”，③ 管的是“设备那一颗
binary”**。② 命中时根本不会走到 ③；③ 只在 ② 未命中、但设备源码与上一轮完全相同时
救你一次。

### 2.3 host ABI 是什么

编译前，你的 kernel 是“若干张量 + 若干标量”的函数。编译后，host 侧被改写成
一个**签名固定**的 packed function（`src/transform/make_packed_api.cc:764`）：

```text
(void* self_handle, TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)
```

“ABI”在这里不是抽象名词，它由三样具体的东西组成：

1. **参数怎么排**：所有参数打成一个 `TVMFFIAny` 数组；走 callee-allocated
   输出 ABI 时，输出张量从参数里去掉、末尾补一个 allocator anchor
   （`make_packed_api.cc:423-431`）。
2. **谁来校验**：生成的 host stub 自己做参数个数、类型码、维度、dtype、
   形状、stride、device 的检查，失败经 TVM-FFI 的 C ABI error slot 抛回 Python
   （`src/transform/tvm_ffi_binder.h:32-36`）。
3. **输出谁分配**：两条路径，编译期由 `tilelang_callee_allocated_outputs`
   属性二选一（`tilelang/jit/abi.py:16`、`:39-74`）。

细节见 `figures/fig-03-packed-abi.md` 与正文 4.8、4.9 节。

### 2.4 adapter 是“ABI ↔ 框架”的转接层

adapter 的名字（`tvm_ffi` / `cython` / `nvrtc` / `torch` / `cutedsl`）说的是
**用什么把 host 和框架接起来**，不是“用什么生成设备代码”：

* 生成设备代码的是 codegen，发生在 `JITKernel._compile_artifact`
  （`tilelang/jit/kernel.py:288`）；
* 决定怎么调用它的是 adapter，发生在 `_create_adapter_from_artifact`
  （`tilelang/jit/kernel.py:324`）。

同一个 CUDA 设备源码可以被三种 adapter 包装，产物形态、stream 传递方式、
缓存里存什么文件都不一样，见 `figures/fig-04-adapter-selection.md` 与 4.10 节。

---

## 3. 用法

### 3.1 把编译产物打出来（无卡可用）

v0.1.15 自带一个专门用来看 `lower()` 产物的工具：
`python -m tilelang.tools.compile_only`（`tilelang/tools/compile_only.py:169-200`），
它内部就是 `tilelang.lower(..., enable_device_compile=False)`，**不需要 GPU、
不需要 CUDA 驱动、不需要 cubin**。默认目标是 `c`（CPU C），任何机器都能出源码：

```bash
# 默认 target=c
python3 -I -m tilelang.tools.compile_only --output_file out.c example.py
# CUDA 源码生成也不需要 GPU，但要求 wheel 里带 CUDA codegen FFI
python3 -I -m tilelang.tools.compile_only --target cuda --output_file out.cu example.py
```

`code/05-compile-only-inspect/` 把它包成了一个能同时打印 TIR、设备源码、
PrimFunc 属性、以及这次调用会进入 cache key 的字段的脚本。

**注意它只给你设备半边。** packed host 入口要开 host codegen 才有，
真实设备上跑起来后用 `JITKernel.get_host_source()`（`tilelang/jit/kernel.py:545`）取。

### 3.2 看编译产物

`JITKernel` 上有一组只看不写的接口：

```python
kernel = add_kernel(1 << 20)
src = kernel.get_kernel_source()      # 设备源码（kernel.py:532）
host = kernel.get_host_source()       # host 源码（kernel.py:545）
kernel.show_source("both")            # 直接打印
kernel.export_sources(kernel_path="/tmp/k.cu", host_path="/tmp/k.cc")
```

还有一个开一次就够用的诊断开关：设 `TILELANG_JIT_DIAGNOSTICS=1`
（`tilelang/env.py:382`、`:459-460`），`jit_phase` 上下文管理器
（`tilelang/jit/diagnostics.py:24-43`）就会为每个阶段打印一行
`TileLang JIT phase start/done: <name> elapsed=…`。想知道“这次是编译慢还是
加载慢”，这是最省事的办法，把 `verbose=True` 传给 `@tilelang.jit` 也一样。

### 3.3 三把缓存开关

```python
import tilelang

tilelang.disable_cache()      # 关掉（env.py:558-559 暴露的顶层 API）
tilelang.enable_cache()       # 用完记得开回来
```

```bash
TILELANG_DISABLE_CACHE=1 python my_kernel.py             # env.py:444 全局禁用
TILELANG_CACHE_DIR=/tmp/tl python my_kernel.py           # env.py:368 换缓存目录
TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1 python my_kernel.py # env.py:447 把库内容哈希进 key
```

第三条是“改了 C++ pass 不生效”的正解，4.4 节会解释为什么。

---

## 4. 实现机制

### 4.1 装饰阶段：`@tilelang.jit` 做了什么

`jit()` 先收集编译参数（`out_idx`/`target`/`execution_backend`/`pass_configs`/
`debug_root_path`/`compile_flags`），返回一个闭包 `decorator`
（`tilelang/jit/__init__.py:627-650`）。真正干活的是这四行：

```python
mode = "auto"
pf: JITFunc[_P, _T] = prim_func(func, eager_jit=True)     # :640
func_source = inspect.getsource(pf.orig_func)             # :641
signature = inspect.signature(pf.orig_func)               # :642
return JITImpl(func=pf, **compile_args, func_source=..., signature=..., mode=mode)
```

要点：

* **所有函数都先被包成 `JITFunc`**，包括 lazy 风格的。区别只在后面
  `_is_lazy_style()` 怎么判定（见 4.2）。
* **源码文本与签名在装饰时被抓住**：`inspect.getsource` 依赖文件还在磁盘上，
  在交互式解释器（REPL、`exec`）里定义 kernel 可能直接抛错。
* `mode="auto"`，真正的模式要等到第一次调用、拿到实参才能定。
* `JITImpl.__post_init__` 建三个字典：`_kernel_cache`（参数键 → kernel）、
  `_call_form_cache`（原始调用形态 → kernel）、`_tuner_cache`
  （`tilelang/jit/__init__.py:357-366`）。

### 4.2 取 TIR 与模式推断

`get_tir` 很短（`tilelang/jit/__init__.py:379-392`）：先 `initialize_jit_mode()`，
再在 target 上下文里调用被装饰的函数，最后断言结果是 `PrimFunc`。

```python
self.initialize_jit_mode(*args, **kwargs)          # :383
with self._get_frontend_target_context():          # :384
    tir = self.func(*args, **kwargs)               # :388
assert isinstance(tir, PrimFunc)                   # :391
```

模式推断在 `_infer_jit_mode`（`:394-407`）：

* 如果显式写了 `mode="lazy"`/`"eager"`，照用；
* 如果 `self.func` 不是 `JITFunc`，一律 `lazy`；
* 否则问 `JITFunc._is_lazy_style(*args, **kwargs)`
  （`tilelang/language/eager/builder.py:1635-1673`）：先看函数里有没有内嵌的
  `@T.prim_func`，没有就**试着调用一次**，返回 `PrimFunc` 就算 lazy，
  抛 `JITNoBuilderError` / `EagerJITBuildError` 就算 eager。

这条“试调用”解释了 eager 模式的第一个反直觉点：**第一次调用会多跑一次函数体
用来判定模式**。也解释了为什么 eager 模式不允许 `out_idx`，
`initialize_jit_mode` 会直接报错（`:413-414`），因为 eager 用 `T.empty()`
声明输出。

参数到缓存键的转换在 `JITFunc.parse_args`（`builder.py:1692-1705`）：它返回
`((p1_key, p2_key), tensor_args)`。phase-1 键是编译期参数（`M`、`block_M`…），
phase-2 键是张量实参带来的动态形状。

### 4.3 进程内的那本账

`JITImpl.__call__` 的主干（`tilelang/jit/__init__.py:519-564`）：

```python
key, kernel_args = self.func.parse_args(*args, **kwargs)   # :550
kernel = self._kernel_cache.get(key, None)                 # :551
if kernel is None:
    kernel = self.compile(*args, **kwargs)                 # :553
    self._kernel_cache[key] = kernel                       # :554
if self.mode == "eager":
    return kernel(*kernel_args.values())                   # :562
return kernel                                              # :564
```

两点值得注意：

* **这本账的 key 是 Python 层的实参组合，不是 TIR 哈希。** 它绕过整条
  `cached()` → `KernelCache` 路径，命中时连磁盘都不碰。因此它的失效条件是
  “参数变了”或“进程重启”。
* 另有一条 `_call_form_cache` 快路径（`:49-91`、`:514-517`），只对
  “lazy 模式且没有张量参数”的 JIT 函数启用，按原始 Python 调用形态直接返回
  kernel 对象，连 `parse_args` 都省掉。

### 4.4 磁盘缓存：key 由什么组成

`KernelCache._generate_key`（`tilelang/cache/kernel_cache.py:243-284`）把下面这些
字段拼成一个 dict，`json.dumps(..., sort_keys=True)` 后取 sha256：

```python
key_data = {
    "func": sha256(func.script(show_meta=True).encode()).hexdigest(),  # :269-271
    "out_idx": (...),            # :272
    "args_repr": tuple(repr(arg) for arg in args),                     # :273
    "target": str(target),       # :274
    "target_host": ...,          # :275
    "execution_backend": ...,    # :276
    "pass_configs": ...,         # :277
    "compile_flags": ...,        # :278
    **self._get_base_key(),      # :279  → version / platform / [tilelang_lib] / [torch]
}
```

三个必须搞清楚的细节：

**其一：编译期参数是通过 TIR 文本进 key 的，不是通过 `args_repr`。**
v0.1.15 里 `tilelang.compile()` 只有两个调用点，且都以关键字参数调用：
`JITImpl.compile`（`tilelang/jit/__init__.py:466-477`）与 autotuner 的
`CompileConfig.compile_program`（`tilelang/autotuner/param.py:67-77`）；
而 `cached()` 又是 `compile()` 的唯一调用点（`:164-173`）。两者叠加的结论是：
**`args_repr` 在 v0.1.15 自带的所有路径上恒为 `()`**。`N`、
`block_N` 这类参数之所以能区分开，是因为 `get_tir()` 已经把它们烧进了 TIR，
`func.script()` 因此不同。推论：**只在 Python 侧起作用、又不体现在 TIR 上的参数，
不会进 key。**

**其二：`_get_base_key` 里有个容易被忽略的平台分支**
（`kernel_cache.py:140-150`）：

```python
base = {"version": __version__, "platform": platform.machine()}   # :141
if env.should_use_kernel_cache_lib_stamp():                        # :142
    lib_stamp = KernelCache._get_tilelang_lib_stamp()               # :143
    if lib_stamp: base["tilelang_lib"] = lib_stamp                  # :145
if sys.platform == "darwin":
    import torch; base["torch"] = torch.__version__                 # :146-149
```

`tilelang_lib` 是**库文件内容的 sha256**（`:85-137`，刻意不用 mtime，
以保证跨机器稳定），但**默认不开**：只有 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP`
为真时才加进去（`env.py:447-448`）。这就是本节开头那个问题的根因，
你改了 `src/transform/*.cc`、重编了 `libtilelang.so`，但输入 TIR 一个字节没变，
key 也不变，于是命中旧产物。macOS 上多算一个 `torch` 版本，是因为该平台的
host 编译要借用 torch 的 C++ 扩展头文件路径（`:60-73`）。

**其三：目录命名空间与 key 是两回事。**
`_get_cache_namespace()`（`:168-174`）给出 `<version>/<sys.platform>-<machine>`，
它只按版本和主机平台分桶；真正的 key 是那串 sha256，作为下一层目录名
（`:463-473`）。所以“换个版本”会把整片目录拉开，而“换个 pass 开关”只在
同一个桶里多出一个新目录。

### 4.5 磁盘缓存：目录布局与原子发布

一个一个条目长这样（`kernel_cache.py:35-58` 声明文件名，`:552-640` 负责写）：

```text
$TILELANG_CACHE_DIR/                       # env.py:368，默认 ~/.tilelang/cache
└─ <version>/                              # 例如 0.1.15
   ├─ kernels/                             # cache_root_dir
   │  └─ <sha256>/                         # 一个 kernel 配置一个目录
   │     ├─ device_kernel.cu               # 设备源码
   │     ├─ host_kernel.cu                 # wrapper/host 源码
   │     ├─ kernel_lib.so                  # 二进制（tvm_ffi 下叫 executable.so）
   │     ├─ params.json                    # KernelParam 的 JSON（engine/param.py:169）
   │     ├─ resource_usage.json            # 仅 HIP 用（kernel_cache.py:601-604）
   │     └─ manifest.json                 # 每个文件的大小 + sha256（:790-805）
   ├─ .staging/                            # 写入中的私有目录（:577-583）
   ├─ cuda-binaries/<key>/kernel.cubin     # CUDA 设备二进制缓存
   └─ ascend-binaries/<key>/kernel.aibin   # 昇腾设备二进制缓存
```

写入策略值得单独讲，因为它直接决定你会不会读到半个文件：

1. 先在 `.staging/` 下建一个带 pid 与随机后缀的私有目录（`:577-583`）；
2. 所有文件写进去，逐个 `fsync`，再写 `manifest.json`（`:585-622`）；
3. **一步 `os.rename` 把整目录挪到最终位置**（`:627-636`），
   其他进程要么看到完整目录，要么什么都看不到；
4. 抢输了（`EEXIST`/`ENOTEMPTY`）就丢掉自己的 staging，**先发布者赢**
   （`:628-634`、辅助函数 `:846-848`）。

读取端同样保守：`_load_kernel_from_disk`（`:641-731`）先查必需文件是否齐全
（`:671-675`），再用 manifest 校验大小与必需文件的 sha256（`:677-687`、
`:807-835`）；一旦发现损坏，**删掉整条目录**并当作未命中，重新编译。
这在共享网络文件系统上很关键：被截断的 `.so` 往往 `dlopen` 能过、
到第一次 launch 才炸成 `CUDA_ERROR_INVALID_IMAGE`，那时已经无法自愈
（`:678-684` 的注释就是为这件事写的）。

最后，`.staging/` 里超过一小时的残骸会在初始化时被清掉
（`_cleanup_stale_staging_dirs`，`:220-241`），这是给“进程写一半崩了”收尾的。

### 4.6 设备二进制缓存

设备编译不是在这里发生的，而是在 **codegen 的末尾**：CUDA 侧
`src/cuda/codegen/rt_mod_cuda.cc:126-140` 调用全局回调
`tilelang_callback_cuda_compile`，回调里先查缓存再决定要不要真的调编译器
（`tilelang/cuda/backend.py:59-120`）。昇腾侧同构
（`tilelang/ascend/backend.py:15-49`），只是把 `nvcc` 换成 `bisheng`。

两个子类的 key（`tilelang/cache/cuda_binary_cache.py:22-45`、
`tilelang/cache/ascend_binary_cache.py:23-42`）都建立在
`BinaryCache._finalize_key`（`tilelang/cache/binary_cache.py:112-118`）之上：

| 后端 | key 字段 | 产物 |
| --- | --- | --- |
| CUDA | `tilelang_version`、`code_hash`、`target_kind`、`target_arch`、`target_code`、`compile_format`、`options` | `kernel.cubin` 或 `.fatbin` |
| Ascend | `tilelang_version`、`code_hash`、`target_kind`、`target_arch`、`compile_format`、`options`、`linker_options` | `kernel.aibin` |

CUDA 那侧有一句很值得抄进自己项目的注释（`cuda_binary_cache.py:32-35`）：
编译选项必须进 key，因为 `--use_fast_math` 这类开关**不改变 CUDA 源码却改变
SASS**，只按源码哈希会让 fast-math 的产物满足一个 precise-math 的编译请求。

存储格式同样简单（`binary_cache.py:150-197`）：每个 key 一个**不可变**目录，
里面是 `metadata.json`（`format`/`size`/`sha256`）加一颗裸 binary；
读取时三项全对才认（`:125-147`），**发现损坏也不删**。因为 3FS 上 unlink
可能让另一个 reader 的打开句柄失效（`:28-29`、`:139-141`）。

### 4.7 lower：从 `PrimFunc` 到 host/device 两个模块

`JITKernel._compile_artifact`（`tilelang/jit/kernel.py:288-322`）做的事很少：
装好 pass 上下文、装好 instrumentation、然后调
`lower_with_context(...)`（`tilelang/engine/lower.py:144-171`）。

真正的拆分在 `lower_to_host_device_ir`（`engine/lower.py:109-141`）：

```python
PreLowerSemanticCheck(mod)                       # :134 后端无关的语义检查
mod = context.lower(mod)                         # :136 ← 唯一的后端分岔点
host_mod = tirx.transform.Filter(_is_host_call)(mod)    # :138
device_mod = tirx.transform.Filter(_is_device_call)(mod)# :139
```

`_is_device_call` 的判据是 `calling_conv == DEVICE_KERNEL_LAUNCH`
（`engine/lower.py:26-28`、`:43-44`），而 `MakePackedAPI` 在管线末尾把入口函数
改成 `kCPackedFunc`（`src/transform/make_packed_api.cc:1006`）。所以**两边的划分
是靠 calling convention 区分的，不是靠函数名**。这也解释了第 20 章 4.3.8 节里
那条 `RequiresPackedAPI` 的规则：已经带非默认 calling conv 的函数不再改写
（`make_packed_api.cc:317-340`）。

之后 `_lower_with_context_impl`（`:174-213`）按两个开关决定要不要真的生成/编译：

| 开关 | 谁传 | 作用 |
| --- | --- | --- |
| `enable_host_codegen` | `ExecutionBackendSpec.enable_host_codegen`（`tilelang/jit/kernel.py:295`） | 生成 host runtime module（`engine/lower.py:65-89`） |
| `enable_device_compile` | `ExecutionBackendSpec.enable_device_compile`（`tilelang/jit/kernel.py:296`） | 决定走 `device_codegen` 还是 `device_codegen_without_compile`（`engine/lower.py:99-106`） |

`tvm_ffi` 两个都开，`nvrtc`/`cython` 默认两个都关。因为它们的 host 侧
由 adapter 自己生成（`tilelang/cuda/execution_backend.py:24-33`）。

产物装在 `CompiledArtifact` 里（`tilelang/engine/param.py:153-166`）：
`host_mod`、`device_mod`、`params`、`kernel_source`，以及可选的 `rt_mod`。

### 4.8 host 侧 ABI：`MakePackedAPI` 生成什么

节选骨架（行号都在 `src/transform/make_packed_api.cc`）：

```cpp
PrimFunc MakePackedAPI(PrimFunc func,
                       const std::vector<int> &callee_allocated_output_indices) {   // :386
  auto global_symbol = RequiresPackedAPI(func);                                     // :388
  const bool use_callee_allocated_output_abi = !callee_allocated_output_indices.empty(); // :392
  int num_args = static_cast<int>(func_ptr->params.size());                         // :423
  if (use_callee_allocated_output_abi) { num_args -= outputs; num_args += 1; }      // :424-431
  Var v_self_handle("self_handle"), v_packed_args("args"),                          // :435-437
      v_num_packed_args("num_args"), v_result("result");
  seq_init.push_back(MakeAssertEQ(v_num_packed_args, num_args, ...));               // :473
  seq_init.push_back(MakeAssertNotNull(v_packed_args, ...));                        // :481
  ...
  binder.BindDLTensors(buffer_def, device_type, device_id, name_hint, ...);          // :777
  ...
  Stmt body = ReturnRewriter(v_result)(func_ptr->body);                             // :1011
  func = WithAttrs(std::move(func), {{kCallingConv, static_cast<int>(kCPackedFunc)}}); // :1006
  func_ptr->params = args;                                                          // :1058
}
```

插入的检查按这个顺序生效（与 `figures/fig-03-packed-abi.md` 一致）：

1. **参数个数**（`:473-477`）：`"<kernel>: num_args should be <N>"`；
2. **`args` 指针非空**（`:480-482`，仅当 `num_args > 0`）；
3. **逐个参数的 type index**（`:615-719`）。判定规则很宽：
   - handle 参数接受 `None` / `OpaquePtr` / `DLTensorPtr` / 任何静态对象
     （`:652-658`）；
   - `bool` 接受 Bool 与 Int；`int` 接受 Int 与 Bool；
     `float` 接受 Float、Int、Bool 三者（`:674-702`）。
     也就是说 **Python 侧传一个 int 给 float 标量参数是被接受的**，
     并会被 `Select` 成浮点（`:704-710`）。这是“静默”的来源之一。
4. **callee-allocated anchor 校验**（`:721-758`，仅该 ABI）：anchor 必须是
   Tensor、非空、device type 与 target 一致；
5. **DLTensor 绑定与全部形状/dtype/stride 检查**，由
   `TVMFFIABIBuilder::BindDLTensors` 生成（`:777-778` → 见 4.9）。

最后还有一层“编译期能算就算”的优化：`tl.assume` 带来的检查被抽出来
（`AssumeRuntimeCheckExtractor`，`:234-295`），`BinderAddAssert` 里若化简结果
直接为 0 就 `LOG(FATAL)` 在编译期失败（`tvm_ffi_binder.cc:56-63`）。
所以**同一个形状错误，有时是编译报错、有时是运行报错**，取决于编译器能否证明它。

### 4.9 DLTensor 绑定：形状、dtype、stride、device

`BindDLTensors`（`src/transform/tvm_ffi_binder.cc:330-1047`，函数本体很长）的生成顺序：

```text
① is_null 变量；被函数体真正用到的 buffer 必须非空       :375-396
   （没用到的输入 buffer 允许传 None，为动态/可选参数留口）
② shape 指针按 is_null 短路，避免解引用空指针            :398-424
③ ndim 断言：a_ndim == v_ndim                           :433-467
④ dtype 三元组 (code, bits, lanes) 比较                  :470-487
⑤ 逐维 shape 绑定 + 约束断言                             :592-...
⑥ stride 检查（含 RelaxedStrideCheck）                   :296-328
```

第 ④ 步的“放宽规则”是本章最值得背下来的一段，因为它决定了**什么样的
dtype 不匹配不会被拦**：

| expected | 也接受 | 源码 |
| --- | --- | --- |
| `float8_e4m3` | `e4m3fn`、`e4m3fnuz` | `:489-501` |
| `float8_e5m2` | `e5m2fnuz` | `:502-511` |
| `tfloat32` | `float32`（同存储） | `:512-520` |
| `bool` | int8 / uint8 / kDLBool(bits=1 或 8) / 任何 bits=1 的类型 | `:521-544` |
| 亚字节 dtype（如 fp4） | 任何“总位数相同”的类型 | `:545-579` |

前两条要特别注意：`e4m3` 与 `e4m3fn` 的**特殊值编码不同**（前者有 inf、
后者没有），互相顶替会改变数值语义，而 ABI 层不会报错。

报错文案本身是有用的调试信息，格式为
`kernel <kernel> input <buffer> <field> ...`（`ConstraintMessage`，`:30-54`），
例如：

```text
kernel main input A ndim mismatch, expected 2
kernel main input A dtype mismatch, expected ...
main.A is expected to have non-NULL pointer
kernel main input A violates packed ABI constraint
```

`code/03-packed-abi-sim/` 把这套规则用纯 Python 复刻了一遍，
13 个场景逐个对照上面的源码位置，在没有卡的时候，这是理解 ABI 行为最快的路径。

### 4.10 adapter 家族

选定 backend 的规则在 `BackendModule.resolve_execution_backend`
（`tilelang/backend/module.py:159-185`）：`"auto"` 取**声明列表里第一个当前可用**的
（`:167-171`），显式指定但不可用会给出可用清单（`:180-184`）。

| adapter | 源码 | host 产物 | stream 怎么传 | 缓存里存什么 |
| --- | --- | --- | --- | --- |
| `TVMFFIKernelAdapter` | `jit/adapter/tvm_ffi.py:48` | TVM runtime module → `executable.so` | **不传**；TVM-FFI 通过 PyTorch 的 DLPack Exchange API 读当前 work stream | `executable.so` + 两份源码 + `params.json`（`jit/adapter/kernel_cache.py:8`） |
| `CythonKernelAdapter` | `jit/adapter/cython/adapter.py:60` | 生成的 C wrapper + `.pyx`，编成 `.so` | **显式传参**：Python 侧取 raw stream 当最后一个实参传给 `lib.call`（`:431-451`） | `kernel_lib.so` 等（`cython/kernel_cache.py`） |
| `NVRTCKernelAdapter` | `jit/adapter/nvrtc/adapter.py:27` | Py wrapper + launcher | 缺省时取 `torch.cuda.current_stream().cuda_stream`（`:313-317`） | `kernel.cubin` **加** `kernel.py`（`nvrtc/kernel_cache.py:8-12`） |
| `MetalKernelAdapter`（注册名 `torch`） | `jit/adapter/torch/metal.py:14` | 无 | 无 | Metal 的 torch 后端**不支持缓存**，`KernelCache.cached` 会主动 `env.disable_cache()`（`kernel_cache.py:388-391`） |
| `CuTeDSLKernelAdapter` | `jit/adapter/cutedsl/adapter.py:23` | 生成 **Python** 模块而不是 C | 由生成模块处理 | `kernel.py` / `launcher.cpp` / `launcher_lib.so`（`cutedsl/kernel_cache.py:14-18`） |

三个容易踩的点：

* **注册名 `torch` ≠ torch 集成。** `tilelang/jit/kernel.py:390-391` 第一行就是
  `assert is_metal_target(target)`。想在 CUDA 上“用 torch 后端”是做不到的，
  你想要的其实是 `tvm_ffi`（它本来就走 torch 张量）。
* **每个 adapter 有自己的 cache 子类**，所以磁盘缓存里的文件扩展名、
  必需文件清单都不同（tvm_ffi 必需 `executable.so`，nvrtc 额外必需 `kernel.py`，
  CuTeDSL 额外必需 `launcher_lib.so`）。昇腾上还有自己的后缀：
  `device_kernel.asc` / `host_kernel.asc`（`tilelang/ascend/kernel_cache.py:11-20`）。
* **缓存派发按 backend module 名，不按 target kind**：
  `context.module.name == "ascend"` 走昇腾那张表，其余走通用表
  （`tilelang/cache/__init__.py:69`）。

选 adapter 的决策树（按“你遇到的第一个限制”往下走）：

```text
能装 cuda-python 吗？                → 能且想避开 nvcc：nvrtc
需要把 stream 显式当参数传吗？        → 需要：cython
目标是不是 Metal？                   → 是：torch（唯一选项，且无缓存）
目标是不是 CuTe DSL 路线？            → 是：cutedsl
以上都不是                          → tvm_ffi（默认，也是唯一支持
                                       callee-allocated 输出 ABI 的 CUDA 路径）
```

最后一条不是巧合：`supports_callee_allocated_outputs=True` 目前只声明在
CUDA 与 Ascend 的 `tvm_ffi` 条目上
（`tilelang/cuda/execution_backend.py:25-30`、
`tilelang/ascend/execution_backend.py:8-19`），换到其它 adapter 就会退回
“调用方预分配输出”那条路径，行为差别在 4.11 节展开。

### 4.11 与 PyTorch 集成

#### 张量怎么过去

`tvm_ffi` 的两条路径都在 `tilelang/jit/adapter/tvm_ffi.py`：

**默认路径（调用方预分配输出）**，`_convert_torch_func`（`:212-346`）：

1. 校验入参个数：`expected_inputs = len(self.params) - len(self.result_idx)`（`:261`），
   多了少了都直接 `ValueError`；
2. 把输入按**参数位置**填进 `tensor_list`（`:276-281`），输出位置留空；
3. 为每个输出算 dtype 与 shape（动态形状按 `dynamic_symbolic_map` 从输入张量
   反解，`:284-322`），然后 `torch.empty(*shape, dtype=..., device=...)`（`:334`）；
4. `executable(*tensor_list)`（`:339`），张量到 `DLTensor` 的转换发生在
   TVM-FFI 内部的 DLPack Exchange，**零拷贝**，Python 侧不做转换。

**callee-allocated 路径**（`:348-384`）：输出由生成的 host 函数自己分配，
Python 侧只传输入 + 一个 `allocator_anchor`；这条路径要求 torch 提供 DLPack
allocator exchange API，否则给出明确报错（`:367-371`）。对纯标量 kernel，
anchor 用一个零元素张量顶替，只为带上 allocator 与 device 状态（`:373-377`）。

#### stream 怎么对齐

流的取值集中在 `BaseKernelAdapter`
（`tilelang/jit/adapter/base.py:61-122`），两个静态方法返回的是**thunk**
（调用时才取值，因此会遵守上游的 stream 守卫）：

* `get_current_stream_functor`：NPU 优先（`torch.npu.is_available()`），
  否则 CUDA，都没有就返回 `lambda: 0`。CUDA 分支用
  `torch._C._cuda_getCurrentRawStream`；NPU 分支用
  `torch_npu._C._npu_getCurrentRawStream`，源码注释解释了为什么绕开 Python API：
  `torch.npu.current_stream()` 内部会探一次 `torch.cuda.is_available()`，
  在无 CUDA 的昇腾机器上每次约 150 µs（`base.py:80-90`、
  `tilelang/ascend/torch_exchange.py:40-55`）。
* `get_current_device_functor`：同理，用于在没给输入张量时决定输出张量放哪。

昇腾的 `tvm_ffi` 路径多一步交接：第一次遇到 NPU 张量时，
`_prepare_torch_device` 会安装一个 Cython 扩展，
**把 Torch 的 DLPack Exchange API 表打补丁**，让 TVM-FFI 提交 kernel 时
走 Torch 的当前 NPU stream（`tvm_ffi.py:40-45`、`:78-81`；
实现见 `tilelang/ascend/torch_npu_exchange.pyx:50-60`）。

一句话总结：**`tvm_ffi` 不把 stream 当参数传，而是让记录运行时自己去问；
`cython`/`nvrtc` 把 stream 当参数传。** 这决定了你换 adapter 时要不要改调用代码。

#### 动态形状怎么解

如果 kernel 里有符号维度（`T.Tensor((M, N))` 里的 `M` 是 `T.symbolic` 或
eager 的 `T.const`），适配器要在**运行时**把输出形状算出来。机制是一张
`dynamic_symbolic_map`（`tilelang/jit/adapter/tvm_ffi.py:172-210`），
每个符号变量映射到四元组 `(kind, param_index, dim_index, stride_scale)`：

| kind | 含义 | 取值来源 |
| --- | --- | --- |
| `0` | 形状 | 第 `param_index` 个张量在参数位置上的 `shape[dim_index]` |
| `1` | 步长 | 同上，但读 `stride()`（亚字节 dtype 要乘 `stride_scale`） |
| `2` | 标量参数 | 第 `param_index` 个参数本身就是标量，直接当维度用 |

两个细节值得记：

* **输入被优先访问。** 代码里专门先把非输出参数排进 `ordered` 列表
  （`:188-193`），注释解释了原因：输出在为输出分配张量时就要知道形状，
  如果某个维度同时出现在输入和输出上，它必须“归输入所有”，
  否则分配循环会读到还没填的槽位。这是“输出在参数表里排前面”的 kernel
  仍然能工作的关键。
* **亚字节 dtype 要换算。** `stride_scale = 8 // element_bits`
  （`:205-206`）：Torch 的 stride 以**存储单元**计（如 int8），
  而 kernel 期望的是**逻辑元素** stride（如 fp4 元素），两者差一个打包因子。

callee-allocated 路径不用这张表（`:127`），因为形状是由生成的 host 函数
自己算、自己申请内存的（4.8 节）。

#### autograd 的边界

必须说清楚：**v0.1.15 里 TileLang 不做自动微分。** 证据是三条可复现的检索：

```bash
grep -rn "torch.autograd\|autograd.Function\|torch.library\|register_autograd" \
     _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src     # 无输出
grep -rln "backward" _vendor/tilelang-v0.1.15/examples | head         # 一批 *_bwd_*.py
```

也就是说：

* 前向：可调用。你把 `torch.Tensor` 传进去，拿到的是 `torch.empty` 出来的张量
  （`tvm_ffi.py:334`）。**它不在任何 autograd 图里**，没有 `grad_fn`；
  输入张量的 `requires_grad` 也不会触发任何记录。
* 反向：**从哪来要你自己决定**。官方做法是再写一个反向 kernel，
  仓库里 `examples/attention_sink/example_mha_sink_bwd_bhsd.py`、
  `examples/ascend/flash_attention/example_gqa_bwd.py` 等就是成对的前向/反向实现。
* 训练时的常见组合：把 TileLang kernel 包进自己的 `torch.autograd.Function`，
  `forward` 调前向 kernel，`backward` 调反向 kernel。
  **这个包装是框架侧的事，不是 TileLang 提供的**，本章不宣称
  v0.1.15 有任何内置胶水。

### 4.12 把上面这些串起来：一次 `kernel(a, b)` 到底发生了什么

前面十一个小节是按机制切的。真正调试时你需要的是**时序**。下面按一次调用
（lazy 模式、`tvm_ffi` 后端、缓存已命中）从外到内列出经过的每一帧，
左边是 Python 帧，右边是它做了什么：

```text
kernel(a, b)
 └─ JITKernel.__call__                       tilelang/jit/kernel.py:206-222
     └─ adapter.func(...)                     （构造时赋值，kernel.py:149）
         ├─ TVMFFIKernelAdapter 的内层 func   tilelang/jit/adapter/tvm_ffi.py:258
         │   ├─ 校验入参个数                    :261
         │   ├─ 按参数位置拼出 tensor_list      :276-281
         │   ├─ 每个输出算 dtype/shape          :284-322
         │   ├─ torch.empty(...) 分配输出       :334
         │   ├─ _prepare_torch_device(...)      :336-337（昇腾才装交换表）
         │   └─ executable(*tensor_list)        :339
         │       └─ TVM-FFI 侧：
         │           ① DLPack Exchange 把 Tensor 变成 DLTensor（零拷贝）
         │           ② 问当前 work stream（不必由 Python 传）
         │           ③ 进入 MakePackedAPI 生成的 host stub
         │              ├─ num_args 断言            make_packed_api.cc:473
         │              ├─ args 非空断言            :481
         │              ├─ 逐参数 type index 检查   :615-719
         │              ├─ 每个 DLTensor：非空/ndim/dtype/shape/stride
         │              │                            tvm_ffi_binder.cc:375-620
         │              └─ 设置 device 上下文       :1021-1034
         │           ④ 真正的 kernel launch（<<<grid, block, smem, stream>>>）
         └─ 返回：单个张量 或 张量列表            tvm_ffi.py:341-344
```

注意其中有**两个** `func`：Python 侧 `adapter.func` 是包装层，
框架侧是 TVM-FFI 的 `Executable`。真正的参数校验发生在**第二个**位置，
也就是 C++ 生成的 host stub 里。这就是为什么
“Python 侧的 `assert a.shape == ...`”和“ABI 层的 shape 检查”
是两件不同的事，后者你改不了、也绕不开。

缓存命中时，`executable` 的来源是 `runtime.load_module(kernel_lib_path)`
（`tilelang/jit/adapter/tvm_ffi.py:424`），而不是本轮编译出来的模块；
这也是“改了 C++ pass 却拿到旧行为”在时序上的样子。

---

## 5. 进阶与陷阱

### 5.1 启动开销与执行模型

先建立正确的模型。CANN 官方资料把它分成两种：

> 如果 Device 上 Task 执行速度比 Host 下发慢，则 Device 上 Task 调度开销和算子执行时间
> 成为瓶颈……**Device Bound**；如果 Device 上 Task 执行速度比 Host 下发快，
> 则 Host 调度开销成为瓶颈……**Host Bound**。
> 来源：`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20240715-1.md`

同一篇资料还列出了 eager（单算子）模式下一次下发的完整 Host 流程：
Python 处理、Python 到 C++ 数据结构转换、Tiling 计算、申请算子 workspace 与输出内存、
Launch。这五步**全部发生在 device 开始执行之前**，所以小算子、大 batch 的
增量推理/微调阶段最容易变成 Host Bound。

这段描述与 TileLang 的对应关系是直接的：adapter 的
`torch.empty(...)` 输出分配（`tvm_ffi.py:334`）属于“申请输出内存”，
TVM-FFI 的 DLPack 转换与 host stub 的参数检查属于“数据结构转换”，
`executable(*tensor_list)` 属于 Launch。**本章不对这些步骤给出任何耗时数字**，
本机没有加速器，任何数字都会是没有出处的编造。

想量化自己的开销，用第 12 章介绍的 profiler：

```python
prof = kernel.get_profiler()          # tilelang/jit/kernel.py:516
lat = prof.do_bench()                 # tilelang/profiler/__init__.py:220
```

**怎么判断自己是不是 Host Bound**，按下面的顺序做，每一步都能排除一类原因：

1. **先排除编译期。** 设 `TILELANG_JIT_DIAGNOSTICS=1`
   （`tilelang/env.py:382`），看有没有 `TileLang JIT phase done: cache.compile`
   这类行。如果第一次测的是编译、后面才是稳定值，说明你测的是编译不是执行。
   正确做法是先 warmup（`do_bench` 的 `warmup` 参数就是干这个的）。
2. **再分清“kernel 自己多慢”与“外面多慢”。** 用同一个 kernel 跑两次计时：
   一次默认 `backend="event"`，一次 `backend="cudagraph"`
   （`tilelang/profiler/bench.py:39`、`:258-259`）。event 计时包含两次
   事件记录与中间的下发间隔；cudagraph 把整段录成图后回放，
   **launch 开销被压掉**。两者差距越大，说明 launch/下发占比越高，
   这就是“host bound 的量化指标”。
3. **最后定位到哪一段。** 如果差距很小，说明瓶颈在 device，去查 kernel 本身
   （第 12、28 章）；如果差距很大，看下面的 5.2 节。

同一套判断也可以放在框架侧做：用 `nsys`（NV）或 `msprof`（昇腾）看
“Launch 之间的空隙”占整段的比例。空隙越大、kernel 越短，越靠近 Host Bound。

### 5.2 用图回放压掉下放开销

两条硬件线各有一个机制，思路完全一样：**把一串下发动作录下来，之后一次提交**。

**NVIDIA：CUDA Graph。** TileLang 的 profiler 内置了这个计时后端：
把 `backend="cudagraph"` 传给 `do_bench`（`tilelang/profiler/bench.py:258-259`），
走的实现是 `bench_with_cudagraph`，其注释说明用途是
“Benchmark using CUDA graph for minimal launch overhead”
（`tilelang/profiler/torch_bench.py:144`、`:152`）。
计时用它是为了把 launch 开销从测量里剔掉；生产里用它则是为了真的把开销剔掉。

**昇腾：ACLGraph。** 暴露形式是 `torch.npu.NPUGraph`，官方示例见
`_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py:271-272`：

```python
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    q = tilelang_rms_norm(q, eps)
    q = tilelang_apply_rope(q, sin, cos)
g.replay()
```

官方 README 明确了两条语义：捕获期间的操作**只被记录、不执行**；
`replay()` 可以反复调用。`code/06-ascend-aclgraph/` 是这个示例的最小复刻。
一个容易误会的版本边界：ACLGraph 是 **CANN / `torch_npu` 的能力**，TileLang 侧
要做的是把 kernel 提交到正确的 stream（与 4.11 节的 stream 机制是同一件事）。
所以“示例只在 `tilelang-ascend` 仓库里、v0.1.15 主干里没有”说的是**示例缺失**，
不是“主干不能用”，判据见 §6.2 的两条 `ls` / `grep`，以及
`code/01-runtime-anchor-check/` 里登记的同名锚点（S118）。

想确认图模式真的生效，按 README 的做法打开 host 日志
（`ASCEND_GLOBAL_LOG_LEVEL=1` 等），搜 `capture model`，
`0` 表示未启用、`1` 表示已启用。

关于收益，corpus 里有一条带条件的记录：vLLM 后端把 `enforce_eager=False`
启用 ACLGraph 后，**在 rollout 场景**实测端到端提升 15%–20%
（`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20260326-4.md`）。
请把它当作**上层框架在特定场景下的端到端数字**来读：它不是 TileLang kernel 的
加速比，也不代表所有模型。本章不引用任何 TileLang 侧的性能数字。

### 5.3 三个必踩的坑

#### 坑 1：改了代码不生效

按“改动类型”对号入座：

| 你改了什么 | key 变不变 | 为什么 | 怎么办 |
| --- | --- | --- | --- |
| 注释、空白、变量名 | **不变** | key 用 `func.script()`，不是 Python 源码（`kernel_cache.py:269`） | 正常，确实没必要重编 |
| `block_N`、`threads` 等进 TIR 的参数 | **变** | 新 TIR 文本 → 新哈希 | 正常 |
| `pass_configs` / `target` / `out_idx` | **变** | 都在 key 字典里（`:272-278`） | 正常 |
| `src/transform/*.cc` 里的 pass 实现 | **默认不变** | `tilelang_lib` 只在 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1` 时进 key（`env.py:447`、`kernel_cache.py:142-145`） | `TILELANG_DISABLE_CACHE=1`，或打开 lib stamp |
| 只影响 Python 侧、不进 TIR 的参数 | **不变** | `args_repr` 在这条链上恒为 `()`（见 4.4） | 把参数推进 TIR，或用 `disable_cache()` |

还有一个纯 Python 层的坑：**在长驻进程里改文件不会改变已经 import 的函数对象**。
`_kernel_cache` 里存的 kernel 与函数对象的源码无关；Jupyter 里改了 cell 却还在跑
旧 kernel，八成是这一条，而不是 TileLang 的缓存。

#### 坑 2：stream 不同步

症状是“有时候对、有时候错”，而且单次运行往往看不出来。三个具体形态：

* 在 `torch.cuda.stream(side)` 里调 kernel，却在默认 stream 上做比较
  （`code/04-torch-stream-call/` 有反例）；
* 手写 `cython`/`nvrtc` 调用时传了 `stream=0`，于是 kernel 落在默认 stream，
  而数据是在别的 stream 上产生的；
* 用 `torch.cuda.current_stream()` 的**值**（而不是 thunk）缓存下来再复用，
  在 stream 被切换后就指向了旧的那条，TileLang 用 thunk 正是为了避开这一点
  （`tilelang/jit/adapter/base.py:61-69`）。

#### 坑 3：dtype / 形状的“静默”错误

ABI 检查是响亮的，但有几类不匹配它会**放你过去**（源码位置见 4.9 的表格）：

* `float8_e4m3` ↔ `e4m3fn`/`e4m3fnuz` 互相接受，但特殊值语义不同；
* 亚字节 dtype 只比总位数，fp4 可以喂进另一套 4-bit 存储约定；
* int 可以喂进 float 标量参数（`make_packed_api.cc:697-710`）；
* `bool` buffer 接受 int8/uint8 存储。

另外还有两处“用户主动关掉检查”的开关，调试期很好用、上线前必须确认：

```python
kernel(a, b, skip_tensor_validation=True)   # cython adapter，adapter.py:443-451
```

和 `tvm_ffi` 的 callee-allocated 路径对 DLPack allocator API 的硬性要求
（`tvm_ffi.py:367-371`），后者不是检查，是前置条件，不满足就直接报错。

### 5.4 跨机器复用缓存

`$TILELANG_CACHE_DIR` 是可以拷的，但要知道哪些东西能拷、哪些不能。

命名空间是 `<version>/<sys.platform>-<platform.machine()>`
（`tilelang/cache/kernel_cache.py:168-174`），所以：

* **换 tilelang 版本 → 自动隔离**。不同版本的产物落在不同目录，
  不会互相污染，代价是升级后第一次全量重编。
* **换 CPU 架构或操作系统 → 自动隔离**。`linux-x86_64` 与 `linux-aarch64`
  是两片互不可见的目录。这挡住了一类低级错误，但隔离**粒度**是粗糙的：
  同一个 `linux-x86_64` 下，用不同版本的 `nvcc` / `bisheng`、
  不同的 glibc、不同的宿主编译器产出的 `.so` 会**混在一起**
  （目标架构是进的 key 的，`target` 字段含 arch，见 4.4。但工具链版本不是）。

实践建议（按优先级）：

1. **优先共享第三层（设备二进制）**：`<version>/cuda-binaries/` 与
   `<version>/ascend-binaries/` 的 key 里显式含 `target_arch`、`options`、
   `linker_options`（4.6 节），是最安全、最可移植的一层。
2. **host 侧的 `kernels/` 目录只在同构机器之间共享**：
   同样的 CPU 架构、同样的编译器家族、同样的 CUDA/CANN 版本。
   共享前先确认 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP` 在两端取值一致，
   否则一端可能命中另一端用不同 `libtilelang` 编出来的产物。
3. **不要把 `.staging/` 拷过去**。它是写入中的目录，正常路径下最终会被
   rename 掉；残留的那些在初始化时按 3600 秒超时清理
   （`kernel_cache.py:220-241`），拷过去只会制造困惑。
4. **共享存储上不要手工删条目**。设备二进制缓存刻意“发现损坏也不删”，
   理由是 3FS 上 unlink 可能让另一个 reader 的打开句柄失效
   （`tilelang/cache/binary_cache.py:26-33`、`:139-141`）。
   要清理就整体换目录，或者离线下线整片 namespace。

---

## 6. 前沿（仅 main / 未合入，不属于 v0.1.15）

本节内容**不在** v0.1.15 里，只用于说明这套运行时接下来往哪走；正式展开见第 30 章。

### 6.1 持久化 kernel：`T.PersistentKernel`

动机与 5.1 节是同一个问题的另一面。昇腾 Kernel 通常按设备的物理核数启动，
但程序里写的往往是更大的逻辑网格；把逻辑网格折叠到物理核数、让每个核循环处理多个
wave 的任务，可以避免“启动很多块、每块干一点事”带来的调度开销。

未合入的 PR #3325 新增了 `T.PersistentKernel(num_cores=...)` 前端 API 与昇腾专用的
`AutoPersistent` pass：程序保持自然的逻辑网格，由编译器折叠到调用方给定的
`num_cores` 并推导 wave 循环；逻辑网格除不尽时给最后一个 wave 加边界保护。
版本边界要说准：**手动的 `T.Persistent` 在 v0.1.15 里就有**
（`tilelang/language/loop.py:82`），PR #3325 加的是**显式声明 + 编译器自动折叠**
这一层；`grep -rn "PersistentKernel\|AutoPersistent" _vendor/tilelang-v0.1.15/tilelang` 无输出。
PR 正文给出的对比条件是 `example_gemm.py` + `do_bench` 的 msprof 路径、
28 个物理 AI Cube 核、`OUT_TILES = 896 = 28×32`（整除）与
`992 = 28×35+12`（不整除）两组形状；结论是
`manual ≈ auto > raw`，三种模式里 auto 与手动 `T.Persistent` 的差异不超过 0.12%。

**这些数字属于未合入 PR 的原始记录，不是 v0.1.15 的实测结果，也没有在任何机器上
被本章复现。** 依据：`../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3325.md`。

### 6.2 PTO adapter：只存在于 main

本任务说明里提到 adapter 家族含 `pto`。**v0.1.15 里没有它**，可复现判据：

```bash
ls _vendor/tilelang-v0.1.15/tilelang/jit/adapter/
# __init__.py  base.py  cutedsl  cython  kernel_cache.py  libgen.py  nvrtc  torch  tvm_ffi.py  utils.py  wrapper.py
grep -rn "pto" _vendor/tilelang-v0.1.15/tilelang/ascend/target.py     # 无输出
ls ../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/jit/adapter/pto/
# __init__.py  adapter.py  cython_wrapper.pyx  libgen.py  wrapper.py   ← main 快照 994b44e 才有
```

也就是说，PTO 是“另一个 target+kernel 形态”，它带的是自己的一套
`BaseKernelAdapter` 子类与 launcher，而不是在现有 adapter 上加一个开关。
这也顺带说明：**adapter 家族的边界是“host 侧怎么接”，所以新增一条编译路线时
必须同时新增 adapter。**

---

## 7. 本章小结

一页复习卡：

```text
生命周期（图 1）
  装饰 → get_tir → 缓存查询 → lower → codegen → adapter → adapter.func
  ├─ 装饰只抓源码与签名，不编译                 jit/__init__.py:638-650
  ├─ 后端分岔只有一处：context.lower(mod)        engine/lower.py:136
  └─ 出口只有一个：JITKernel.torch_function      jit/kernel.py:149

缓存（图 2）
  ① 进程内参数缓存  JITImpl._kernel_cache        jit/__init__.py:551
  ② 磁盘 kernel 目录 sha256(TIR+编译参数+版本)   kernel_cache.py:243-284
  ③ 设备二进制      sha256(设备源码+arch+选项)   binary_cache.py:112-118
  ⚠ 改 C++ pass 默认不失效 → TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1
  ⚠ 编译期参数通过 TIR 文本进 key，不通过 args_repr

host ABI（图 3）
  MakePackedAPI 把入口改成 (self_handle, args, num_args, result)
  检查顺序：num_args → args 非空 → 类型码 → anchor → 逐个 DLTensor
  DLTensor：非空 → ndim → (code,bits,lanes) → shape 约束 → stride
  ⚠ fp8 家族、tfloat32/float32、bool/int8、亚字节总位数都会「放宽」

adapter（图 4）
  tvm_ffi（默认，DLPack 零拷贝，stream 交给运行时去问）
  cython / nvrtc（自己编 host，stream 显式传参）
  torch（其实是 Metal，唯一不支持缓存的）
  cutedsl（产物是 Python 模块）

框架集成
  张量：TVM-FFI 的 DLPack Exchange 零拷贝
  输出：默认 Python 侧 torch.empty；callee-allocated 由 host 分配
  动态形状：dynamic_symbolic_map（kind,param,dim,stride_scale），输入优先反解
  autograd：v0.1.15 不做，反向要自己写 kernel + 自己包 autograd.Function

开销
  Host Bound 时下放开销才是瓶颈；CUDA Graph / ACLGraph 压掉的是这一段
  判断：同一 kernel 比 backend="event" 与 backend="cudagraph" 的差距

可复现的三件事（本机可跑，见 code/）
  169 条锚点 + 241 处行内引用核对    code/01-runtime-anchor-check/
  缓存目录布局与 manifest 校验       code/02-cache-dir-inspect/
  packed ABI 的 13 个检查场景        code/03-packed-abi-sim/
```

如果只记三句话：

1. **缓存命中发生在 lower 之前**，所以你改了编译期东西却没重编，多半是 key 没变；
2. **host ABI 是编译期定死的 packed function**，形状/dtype 的检查在那里，不在 Python 里；
3. **adapter 只决定“怎么把 host 接给框架”**，五个 adapter 的差别就三样：
   产物形态、stream 怎么传、缓存里存什么。

## 延伸阅读

* 本章引用的全部源码锚点与 corpus 资料：见 [references.md](references.md)。
* 习题与答案：[exercises.md](exercises.md) / [solutions.md](solutions.md)。
* 可运行/可核对代码：[code/README.md](code/README.md)。
* 图：[figures/fig-01-jit-lifecycle.md](figures/fig-01-jit-lifecycle.md)、
  [figures/fig-02-cache-layers.md](figures/fig-02-cache-layers.md)、
  [figures/fig-03-packed-abi.md](figures/fig-03-packed-abi.md)、
  [figures/fig-04-adapter-selection.md](figures/fig-04-adapter-selection.md)。
* 想读 pass 顺序：见第 20 章；想读两条后端的代码生成：见第 24、25 章；
  想知道这些能力的最新动向：见第 30 章。
