# 第 26 章 习题答案

每题的结构统一为：**思路 → 参考答案 → 常见错误**。所有源码位置都与
`code/01-runtime-anchor-check/verify_runtime_anchors.py` 的锚点表一致。

---

## E1 生命周期排序（★）

**思路**：本章 2.1 节给的就是这条链，难点只在“缓存查询在 lower 之前”。

**参考答案**：`C → E → F → D → A → B`，即

```text
C 装饰 → E get_tir → F 缓存查询 → D lower → A codegen → B adapter 包装
```

缓存查询发生在 F。依据：

* `JITImpl.__call__` 先 `parse_args` 再查 `_kernel_cache`，未命中才
  `self.compile(...)`（`tilelang/jit/__init__.py:550-554`）。
* `tilelang.compile()` 直接转发给 `cached()`（`tilelang/jit/__init__.py:164-173`），
  只有 `KernelCache.cached()` 未命中时才会构造 `JITKernel`
  （`tilelang/cache/kernel_cache.py:427-437`），而 `lower` 正是发生在那里面
  （`tilelang/jit/kernel.py:288-322`）。

**追问**：F 命中磁盘缓存时，D、A、B 都**不执行**。命中路径是
`_load_kernel_from_disk` → `_build_kernel` → `JITKernel.from_database`
（`tilelang/cache/kernel_cache.py:641-731`、`:872-906`），它读取已落盘的
`device_kernel.cu` / `host_kernel.cu` / `executable.so` / `params.json`，
直接重建一个 `JITKernel` 并让 adapter 从 `.so` 反序列化
（`tilelang/jit/adapter/tvm_ffi.py:424` 的 `runtime.load_module`）。

**常见错误**：把缓存查询放在 lower 之后，理由是“得先编译才知道 key”。
实际上 key 建立在 `func.script()` 上（`kernel_cache.py:269`），只需要 TIR，
不需要 lower 的产物。

---

## E2 装饰阶段做了什么（★）

**思路**：读 `tilelang/jit/__init__.py:638-650` 的 `decorator` 闭包。

**参考答案**：

1. 两件事是 `inspect.getsource(pf.orig_func)`（`:641`）与
   `inspect.signature(pf.orig_func)`（`:642`），分别存进 `JITImpl.func_source`
   与 `JITImpl.signature`。
2. `inspect.getsource` 要求函数的源代码能被定位到，通常意味着文件还在磁盘上、
   且函数不是动态 `exec` 出来的。所以在 `python -c`、`exec()`、
   某些 Jupyter 动态定义的函数上，装饰会抛 `OSError: could not get source code`。
   脚本文件里的函数没有这个问题。
3. 不能一概而论。REPL 里**从文件导入**的函数通常仍可取到源码；真正会失败的是
   当场交互定义（或经过 `exec`）的函数体。稳妥做法是把 kernel 写在 `.py` 文件里
   再 import。这也是官方 examples 的形式。

**常见错误**：回答成“装饰阶段把它编译了，所以慢”。装饰不编译；第一次调用的开销
来自 4.2 节说的“试调用判定模式”加上真正的编译。

---

## E3 缓存 key 的三问（★★）

**思路**：把 key 的每一项对应到“改动类型”。

**参考答案**：

1. **会变**，但**不通过 `args_repr`**：参数被 `get_tir()` 烧进了 TIR
   （`T.Kernel(T.ceildiv(N, block_N))` 这类），所以 `func.script()` 变了，
   进而 `"func"` 字段变。
2. **不会变**。`func.script(show_meta=True)` 是 TIR 文本，注释与空白不进 TIR。
3. `args_repr` 恒等于空元组 `()`。依据：`tilelang.compile()` 只被
   `JITImpl.compile` 以**关键字**参数调用（`tilelang/jit/__init__.py:466-477`），
   而 `cached()` 又只被 `compile()` 调用（`:164-173`）；`*args` 一路为空。
   `KernelCache._generate_key` 里的 `args` 形参因此收不到任何位置参数。
4. 反例：某个参数**只影响 Python 侧行为、又不出现在 TIR 里**。
   例如把“是否在调用后额外做一次 `torch.clamp`”写成一个 kernel 形参，
   但它只用来在 Python 包装层里分支。此时改它既不改 TIR、又不进 `args_repr`，
   key 不变，你会继续拿到旧的 kernel 对象。

**常见错误**：认为“所有函数参数都会进 cache key”。v0.1.15 的实现里，
编译期参数是**经由 TIR 间接**进入的。

---

## E4 库内容哈希（★★）

**思路**：读 `tilelang/cache/kernel_cache.py:85-137` 与 `env.py:447-448`。

**参考答案**：

1. 它算的是 TileLang 原生库（`libtilelang.so|dylib`、`libtvm_runtime.*`、
   `libtvm_compiler.*`，macOS 还看 `libtilelang.dylib`）的 **sha256 内容哈希**，
   拼成 `name:hexdigest|name:hexdigest`。源码注释（`:95-99`）解释了为什么不用
   mtime：内容哈希在不同机器、不同安装时间上保持一致，mtime 不是。
2. 环境变量 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP`
   （`env.py:447-448`），默认关闭。
3. 每次编译都要把这些库文件**整个读一遍并哈希**（按 1 MiB 分块，`:130-132`）。
   对大体积的 `.so` 而言这是实打实的 I/O 开销，因此默认不开。
4. 正在改 C++ pass、改 codegen、或整体重编 `libtilelang` 的场景必须打开；
   否则输入 TIR 不变时 key 不变，你会拿到旧产物。这就是正文 4.4 节的
   “其二”。

**常见错误**：以为它默认开启。默认关闭这一点是文本型检查的重点
（`env.py:447` 的 `lower() in ("1","true","yes","on")`）。

---

## E5 `num_args` 的算法（★★）

**思路**：注意 `-n + 1` 里 `n` 是输出个数，那个 `1` 是 anchor。

**参考答案**：

1. `num_args = 3 - 1 + 1 = 3`。巧的是数量没变，但**内容**变了：
   三个位置从 `(A, B, Out)` 变成 `(A, B, allocator_anchor)`。
2. 多出来的 `+1` 是 allocator anchor。它至少要被断言为：
   类型码是 `kTVMFFITensor`、指针非空、DLTensor 的 `device_type` 与编译目标一致
   （`src/transform/make_packed_api.cc:732-757`）；还要从它身上取出 `device_id`
   绑定给 host 侧（`:756-757`）。
3. 因为这是 **ABI 决策**：host stub 的参数布局与 adapter 的调用约定必须完全一致，
   两边各自“按 target 现推”就会推歪。所以由执行后端在编译期声明能力
   （`ExecutionBackendSpec.supports_callee_allocated_outputs`，
   `tilelang/backend/execution_backend.py:29`），编译期把它刻成属性
   （`tilelang/jit/abi.py:16`），lower 与 adapter 都读同一个属性
   （`tilelang/jit/abi.py:10-15` 的注释就是这个意思）。

**常见错误**：把 `-n+1` 记成“输出不算参数了”。输出确实不在 `args` 里，
但换来的是一个 anchor，所以个数可能相等。

---

## E6 dtype 放宽规则（★★）

**思路**：把 4.9 节的表逐条落到源码。

**参考答案**：

1. 一共五条放宽规则，本章正文把它们归纳成“四类 + 一条特例”：
   * FP8 E4M3 家族：`e4m3` / `e4m3fn` / `e4m3fnuz` 互为可接受
     `src/transform/tvm_ffi_binder.cc:489-501`；
   * FP8 E5M2 家族：`e5m2` / `e5m2fnuz`
     `:502-511`；
   * `tfloat32` ↔ `float32`（存储相同）
     `:512-520`；
   * `bool` 接受 int8/uint8、kDLBool(bits=1/8) 以及任何 bits=1 的类型
     `:521-544`；
  * 以及亚字节 dtype（fp4 等）只比较**总位数** ，`:545-579`。
2. 最危险的是 **FP8 E4M3 家族**。`e4m3` 与 `e4m3fn` 的存储宽度相同，
   但**特殊值语义不同**：`e4m3fn` 没有 infinity 编码，NaN 的位模式也与
   `e4m3` 不同。用 ABI 层“放行”的一颗张量去喂另一种语义的 kernel，
   不会报错，只会在出现 inf/NaN 时输出不一样的东西。而这类输入在
   随机初始化阶段往往不会出现，于是问题被推迟到训练后期的某一刻。
3. 工程建议：**在 Python 侧显式断言 dtype 与形状**，不要依赖 ABI 层。
   具体做法：拿 `KernelParam` 的期望值（`tilelang/engine/param.py:13-51`）
   和实际张量比一次，或者干脆在 kernel 包装函数里写
   `assert a.dtype == torch.float8_e4m3fn`。ABI 检查是“最后一道网”，
   不是“唯一一道网”。

**常见错误**：以为 ABI 会拦住一切 dtype 错误。`tvm_ffi_binder.cc` 的注释
明确写着这些放宽是**有意**的（例如 FP8 家族在 PyTorch 侧的命名不统一）。

---

## E7 stream 的两种传法（★★）

**思路**：把“读当前 stream”和“把 stream 作为实参”区分开。

**参考答案**：

1. 读当前 stream（不传参）：`tilelang/jit/adapter/base.py:61-101` 提供 thunk，
   `TVMFFIKernelAdapter` 不出现在实参里。它靠模块开头的注释说明：
   TVM-FFI 通过 Torch 的 DLPack Exchange API 获取 active work stream
   （`tilelang/jit/adapter/tvm_ffi.py:1-6`）。
   显式传参：`CythonKernelAdapter` 的 `lambda_forward(*args, stream=-1, ...)`
   （`tilelang/jit/adapter/cython/adapter.py:443-451`）把 stream 作为关键字参数
   交给 `cython_wrapper.forward`，底层 `_forward_from_prebuild_lib` 再把它
   追加到 `lib.call` 的实参表末尾（`:431-438`）；
   `NVRTCKernelAdapter` 同形（`tilelang/jit/adapter/nvrtc/adapter.py:313-319`）。
2. `tvm_ffi` 的取值链是 `BaseKernelAdapter.get_current_stream_functor`
   → CUDA 用 `torch._C._cuda_getCurrentRawStream`，NPU 用
   `torch_npu._C._npu_getCurrentRawStream`。昇腾上多一步：第一次遇到 NPU 张量时
   `_prepare_torch_device`（`tvm_ffi.py:78-81`）调用
   `install_torch_npu_stream_exchange()`（`tilelang/ascend/torch_exchange.py:6-20`），
   它装载 Cython 扩展去**打补丁 Torch 的 DLPack Exchange API 表**
   （`tilelang/ascend/torch_npu_exchange.pyx:50-60`），让 TVM-FFI 问到的就是
   Torch 的当前 NPU stream。
3. 不用改。`tvm_ffi` 的调用体是 `func(*inputs)`（`tvm_ffi.py:258`），
   `cython` 的调用体是 `lambda_forward(*args, stream=-1, ...)`，
   两者都以 `*args` 收张量，`stream` 有默认值，不传也能工作。
   但若你**显式**传了 `stream=`，换回 `tvm_ffi` 时会直接
   `TypeError: unexpected keyword argument`，这时要删掉。

**常见错误**：认为 `tvm_ffi` 用 `torch.cuda.current_stream()` 的 Python API。
它在 CUDA 上走的是 `torch._C` 的底层入口，原因见 `base.py:91-99`；
NPU 侧的 150 µs 注释在 `base.py:80-84` 与 `torch_exchange.py:43-46`。

---

## E8 造一个“应该失效却没失效”（★★★）

**思路**：把 E3 第 3、4 问的结论变成一个具体例子。

**参考答案**：

场景：kernel 的 `T.prim_func` 固定 `block_M=64`，但在 Python 包装层有一个开关
决定“输出要不要先 zero-fill 再交给 kernel”：

```python
@tilelang.jit(out_idx=[-1])
def f(M, N, block_M=64, zero_init=False):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float16"), Out: T.Tensor((M, N), "float16")):
        with T.Kernel(...):
            ...
    return main

def run(a, zero_init=False):
    out = f(a.shape[0], a.shape[1], zero_init=zero_init)(a)
    return out
```

`zero_init` 只被 Python 层用，不进 TIR → `func.script()` 不变；
`cached()` 收到的 `args` 为空 → `args_repr` 恒为 `()`；
所以两种取值共用同一个 key，命中同一个 kernel。

两种修法：

1. **环境变量法**（应急）：`TILELANG_DISABLE_CACHE=1 python run.py`
   （`tilelang/env.py:444`），或代码里 `tilelang.disable_cache()` /
   `tilelang.enable_cache()`（`env.py:438-442`、`:558-559`）。
2. **改代码法**（正解）：让参数真正参与编译决策。可以
   （a）把它写成影响 TIR 的形式。例如用 `T.Kernel` 的 extent 或把它当
   `pass_configs` 传进去；或
   （b）在 `@tilelang.jit` 里通过 `pass_configs` 暴露，
   因为 `pass_configs` 在 key 里（`kernel_cache.py:277`）。

为什么少见但难查：这类参数通常看起来“无害”（像调试开关），
代码 review 时没人会怀疑它；出问题时症状是“改了开关没反应”，
而你会先去查 kernel 逻辑，不会先查缓存。

**常见错误**：把 `zero_init` 直接加进 `f()` 的签名就以为解决了。
只要它仍然不出现在 TIR 里，加进签名也没用。这正是关键。

---

## E9 加一条锚点（★★★）

**思路**：锚点表就是“声明 + 核对”，加一条只需两行。

**参考答案**：

1. 运行：

   ```bash
   cd chapters/26-runtime-jit-abi/code/01-runtime-anchor-check
   python3 verify_runtime_anchors.py --quiet
   # ok: 160/160 anchors verified
   ```

2. 在 `ANCHORS` 里加：

   ```python
   Anchor("A161", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 261,
          "expected_inputs = len(self.params) - len(self.result_idx)",
          "tvm_ffi adapter 的入参个数校验")
   ```

3. 把 `261` 改成 `262` 再跑，脚本会打印
   `expected '...', found 'if len(inputs) != expected_inputs:'` 并返回退出码 1。
   注意它**不会**替你判断“这句话是不是技术结论正确”，
   只判断“这一行是不是你声称的那一行”。
4. 因为 prose 里的行号会随版本漂移，而锚点表把“正文的每一处行号声明”
   变成了可执行断言。升级 `_vendor/` 时，跑一次脚本就知道正文哪些地方
   需要更新。这比人工重读 130 多处引用可靠得多。

**常见错误**：把 `needle` 写得太长（跨行匹配不上）或太短
（同一行出现多次，但 `sed -n Np` 只看一行，其实不会有歧义）。
推荐挑该行**独有**的短片段。

---

## E10 stream 竞态实验（★★★，需 CUDA）

**思路**：这是“用统计暴露非确定性”的经典实验。

**参考答案**：

1. 脚本骨架（完整版见 `code/04-torch-stream-call/`）：

   ```python
   side = torch.cuda.Stream()
   bad = 0
   for _ in range(100):
       with torch.cuda.stream(side):
           out = kernel(a, b)
       if not torch.allclose(out, ref):   # 比较在默认 stream 上
           bad += 1
   print("failures without sync:", bad)
   ```

   加同步的对照组：在 `with` 块结束后、比较之前插入 `side.synchronize()`。
   统计结果因机器、kernel 时长、随机数而异；本机没有 GPU，**本章不给任何数字**。
   要判定的是趋势：无同步组会出现非零失败次数，有同步组应当是 0。

2. 非确定性的来源：`allclose` 本身也是要下发的 device 工作，
   它落在默认 stream 上；两条 stream 之间**没有依赖边**，
   因此谁先读到数据取决于调度。若 kernel 恰好已经写完，结果正确；
   若没有，就读到未初始化或半成品的数据。两次运行的时序不同，
   结果自然不同。
3. 示例把它写成注释，是因为它是**反例**：可运行的代码只展示正确做法，
   反例留在注释里说明“为什么会错”。如果把它写成活代码，
   读者可能误以为那是推荐模式，也可能在 CI 上得到随机失败的红色。

**常见错误**：以为“不加同步只是慢一点”。不加同步是**正确性问题**，
不是性能问题。

---

## E11 ACLGraph 捕获（★★★，需昇腾 NPU）

**思路**：捕获 → 回放 → 校验，三步；重点是“捕获期间不执行”。

**参考答案**：

1. 最小实现见 `code/06-ascend-aclgraph/aclgraph_capture.py`：

   ```python
   graph = torch.npu.NPUGraph()
   with torch.npu.graph(graph):
       captured = kernel(x)
       sink = captured + out
   graph.replay()
   torch.npu.synchronize()
   assert torch.allclose(sink, out + out, atol=1e-2, rtol=1e-2)
   ```

   参考实现：`_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py:271-272`。

2. 打开 host 日志（按该目录 README）：

   ```bash
   export ASCEND_HOST_LOG_FILE_NUM=1000
   export ASCEND_PROCESS_LOG_PATH=logs
   export ASCEND_GLOBAL_LOG_LEVEL=1
   ```

   然后在日志里搜 `capture model`：`1` 表示图模式生效。
   依据：`_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/README.md:133-139`。

3. 因为捕获阶段只记录**指针与参数**，不记录数据。回放时用的是同一批
   显存地址，所以捕获后直接换一个 `torch.Tensor` 对象不会生效，
   图里仍然是旧地址。要换输入，正确做法是准备一块**固定的**输入缓冲，
   每次把新数据拷贝进去（`buf.copy_(new_input)`）再 `replay()`。
4. ACLGraph 本身的示例**不在** v0.1.15 主干里，只在昇腾适配器仓库：

   ```bash
   ls _vendor/tilelang-v0.1.15/examples/ | grep -i aclgraph          # 无输出
   grep -rli "aclgraph\|NPUGraph" _vendor/tilelang-v0.1.15/{examples,tilelang}  # 无输出
   ls _vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/          # README.md rms_rope_aclgraph.py
   ```

   需要澄清一点：ACLGraph 是 **CANN / torch_npu 的能力**，TileLang 要做的是
   把 kernel 提交到正确的 stream 上（见 E7），这本身不需要 TileLang 侧写代码。
   “主干里没有 ACLGraph 示例”说的是**示例缺失**，不是说不能用。

**常见错误**：捕获后直接换输入张量，然后发现输出没更新，误判成
“ACLGraph 不支持 TileLang kernel”。

---

## E12 autograd 的边界（★）

**思路**：先证明“没有”，再说明“怎么办”。

**参考答案**：

1. 不提供。可复现检索（在 `_vendor/tilelang-v0.1.15/` 下执行）：

   ```bash
   grep -rn "torch.autograd\|autograd.Function\|torch.library\|register_autograd" tilelang src
   # 无输出
   ```

2. 没有 `grad_fn`。输出张量由 adapter 用 `torch.empty(...)` 创建
   （`tilelang/jit/adapter/tvm_ffi.py:334`），是一个普通叶子张量；
   整条调用链里没有任何 autograd 记录点，输入张量的 `requires_grad`
   也不会被读取。
3. 反向需要自己写 kernel 并自己接进 PyTorch。可行做法：

   ```python
   class TileLangFn(torch.autograd.Function):
       @staticmethod
       def forward(ctx, a, b):
           return fwd_kernel(a, b)

       @staticmethod
       def backward(ctx, grad_out):
           return bwd_kernel_a(grad_out), bwd_kernel_b(grad_out)
   ```

   仓库里的现成例子（成对的前向/反向 kernel）：
   `_vendor/tilelang-v0.1.15/examples/attention_sink/example_mha_sink_bwd_bhsd.py`、
   `examples/attention_sink/example_gqa_sink_bwd_varlen.py`、
   `examples/ascend/flash_attention/example_gqa_bwd.py`。
   这些例子展示的是“按反向公式再写一个 kernel”，用
   `grep -rln backward _vendor/tilelang-v0.1.15/examples` 可以列出全部。

**常见错误**：回答“在 Python 里调一下 `torch.autograd.grad` 就行”。
那是让 PyTorch 对一个它看不见的 device 计算做微分，
TileLang kernel 对 PyTorch 而言是一个不透明调用。

---

## E13 缓存完整性校验（★★）

**思路**：把 `_write_manifest`、`_verify_cache_dir_integrity`、
`_get_required_files` 三块拼起来看。

**参考答案**：

1. 每个文件记两项：`size` 与 `sha256`
   （`tilelang/cache/kernel_cache.py:790-805`）。
   清单本身不记自己的哈希，也不记文件 mtime。
2. 加载时：
   * 清单里**所有**列出的文件都要大小一致（`:826-830`）；
   * 只有**必需文件**（adapter 的二进制 + `params.json`，由
     `_get_required_files` 决定，`:770-773`）才做内容 sha256 比对（`:831-832`）。
   区别对待的原因是源码文件（`device_kernel.cu` / `host_kernel.cu`）很大且
   只在需要时才读；而二进制一旦被截断就会在 launch 时炸成
   `CUDA_ERROR_INVALID_IMAGE`，代价远高于多算一次哈希。
   源码里的注释把这层意思写得很直白（`:810-814`）。
3. 校验失败时 `_load_kernel_from_disk` 会 `shutil.rmtree(cache_path)`
   并返回 `None`，也就是“当作未命中，重新编译”（`:677-687`）。
   之所以删而不是修：整个条目的内容互相关联（源码、二进制、参数必须成套），
   就地修复需要知道哪一份是对的；而重编一次的成本是确定的、可控的。
   此外写入端本来就是“staging + 原子 rename”，正常路径不会产生半成品，
   所以出现损坏通常意味着更底层的问题（崩溃、网络文件系统），
   删掉重来是更安全的默认。

**常见错误**：以为 manifest 是“每个文件都做哈希”。
源码文件只做 size 校验，这是有意的性能取舍。
