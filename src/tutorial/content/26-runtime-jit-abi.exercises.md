# 第 26 章 习题

难度标记：`★` 概念/记忆，`★★` 需要读源码，`★★★` 需要动手或推理。
「需硬件」一栏标出是否必须有 GPU / NPU 才能完成。

题目前置：只用本章与第 4、7、20 章的知识即可求解。标注「需先读」的题请在动笔前
回到对应的正文小节。

---

### E1 ★ 概念（无硬件）

把下面六步按**实际发生顺序**排好，并指出哪一步会做缓存查询：

```text
A. codegen（生成设备源码与 host 源码）
B. adapter 包装（选出 TVMFFIKernelAdapter / CythonKernelAdapter / …）
C. 装饰（把函数包成 JITImpl）
D. lower（SplitHostDevice 之后得到 host_mod 与 device_mod）
E. get_tir（得到 PrimFunc）
F. 缓存查询（进程内 / 磁盘）
```

追问：如果 F 命中了磁盘缓存，A、B、D 三步还会执行吗？为什么？

---

### E2 ★ 概念（无硬件）

`@tilelang.jit` 在装饰阶段**不编译**，但它仍然做了两件与源码有关的事。

1. 这两件事分别是什么？给出源码位置。
2. 其中一件依赖「文件还在磁盘上」。这会给你带来什么实际限制？
3. 如果我说「所以我可以在 REPL 里直接定义 kernel 并 JIT」，你同意吗？

---

### E3 ★★ 代码阅读（无硬件）

下面是从 `tilelang/cache/kernel_cache.py:269-284` 抄来的 key 构造：

```python
func_binary = func.script(show_meta=True).encode()
key_data = {
    "func": sha256(func_binary).hexdigest(),
    "out_idx": (tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]),
    "args_repr": tuple(repr(arg) for arg in args),
    "target": str(target),
    "target_host": str(target_host) if target_host else None,
    "execution_backend": execution_backend,
    "pass_configs": pass_configs,
    "compile_flags": compile_flags,
    **self._get_base_key(),
}
key_string = json.dumps(key_data, sort_keys=True)
return sha256(key_string.encode()).hexdigest()
```

请回答：

1. 把 `block_N=128` 改成 `block_N=64`，key 会不会变？**通过哪个字段**变？
2. 把 kernel 里的一句注释改掉，key 会不会变？为什么？
3. `args_repr` 这一项，在 v0.1.15 的 `@tilelang.jit` 路径上恒等于什么？请给出
   源码依据（提示：看 `tilelang/jit/__init__.py` 里 `compile()` 怎么调用 `cached()`）。
4. 由第 3 问的结果推出一个**反例**：什么样的参数改动会让缓存「应该失效却没失效」？

---

### E4 ★★ 代码阅读（无硬件）

`KernelCache._get_base_key()`（`tilelang/cache/kernel_cache.py:139-150`）里有一行：

```python
if env.should_use_kernel_cache_lib_stamp():
    lib_stamp = KernelCache._get_tilelang_lib_stamp()
    if lib_stamp:
        base["tilelang_lib"] = lib_stamp
```

1. `_get_tilelang_lib_stamp()` 算的是什么？它为什么刻意不用文件的 mtime？
2. `env.should_use_kernel_cache_lib_stamp()` 读哪个环境变量？默认值是什么？
3. 打开它之后，**每一次编译**会多付出什么代价？请从
   `tilelang/cache/kernel_cache.py:85-137` 的实现回答。
4. 什么场景下必须打开它？

---

### E5 ★★ 代码阅读（无硬件）

`src/transform/make_packed_api.cc:423-431`：

```cpp
int num_args = static_cast<int>(func_ptr->params.size());
if (use_callee_allocated_output_abi) {
  num_args -= static_cast<int>(callee_allocated_output_indices.size());
  num_args += 1;
}
```

1. 设原函数有 3 个参数，其中第 3 个是输出（`out_idx=[2]`），
   且 `use_callee_allocated_output_abi == true`。此时 `num_args` 等于几？
2. 那多出来的 `+ 1` 是什么？它至少要满足哪些条件？（见 `:721-758`）
3. 为什么这个决策必须在**编译期**定下来，而不是运行期按 target 判断？
   （提示：`tilelang/jit/abi.py:16` 的那条注释。）

---

### E6 ★★ 代码阅读（无硬件）

`src/transform/tvm_ffi_binder.cc:486-579` 里有多条「放宽」dtype 严格相等的规则。

1. 列出全部四类放宽，各给一行源码位置。
2. 哪一类最可能让你在**没有报错**的情况下算错？说明理由。
3. 由此给出一条工程建议：调用 kernel 之前应该做什么检查？

---

### E7 ★★ 概念（无硬件）

`tvm_ffi` 的 adapter **不把 stream 当参数传**，`cython` / `nvrtc` 的 adapter **会传**。

1. 两者的源码位置分别在哪？
2. `tvm_ffi` 是怎么知道该用哪条 stream 的？在昇腾上又多了哪一步？
   （提示：`tilelang/ascend/torch_exchange.py`）
3. 如果你的代码从 `tvm_ffi` 切到 `cython`，调用方式要不要改？为什么？

---

### E8 ★★★ 动手（无硬件）

构造一个「参数改了但缓存不失效」的场景，并给出两种修法。

要求：

1. 参数必须只影响 Python 侧、不影响 TIR 文本；
2. 指出这个参数**为什么**进不了 key（引用 E3 第 3 问的结论）；
3. 给出两种修法，其中一种用环境变量，另一种改代码；
4. 说明为什么这在实际项目里很少见、但一旦发生就很难查。

---

### E9 ★★★ 动手（无硬件）

`code/01-runtime-anchor-check/verify_runtime_anchors.py` 里有一张 160 条的锚点表。

1. 运行它，确认输出 `160/160`。
2. 给下面这句话补一条锚点：「tvm_ffi adapter 会校验入参个数」，
   依据是 `tilelang/jit/adapter/tvm_ffi.py` 里的 `expected_inputs` 那一行。
3. 故意把行号写错一行，确认脚本会失败并打印出「期望什么、实际是什么」。
4. 说明这张表在教材维护上的作用：为什么它比「正文里写死行号」更可靠？

---

### E10 ★★★ 动手（需 CUDA GPU）

写一个脚本，重复 100 次下面的实验，统计「对答案失败」的次数：

```python
side = torch.cuda.Stream()
with torch.cuda.stream(side):
    out = kernel(a, b)
ok = torch.allclose(out, ref)      # 比较跑在默认 stream 上，中间没有同步
```

1. 记录失败次数；再改成 `side.synchronize()` 之后比较，再记录一次。
2. 解释为什么「不加同步」的表现是**非确定性**的，而不是稳定错误。
3. 这个实验在第 26 章 `code/04-torch-stream-call/` 里以反例的形式出现。
   请说明：为什么示例把它写成注释而不是可运行代码？

---

### E11 ★★★ 动手（需昇腾 NPU）

用 `torch.npu.NPUGraph` 把两个 TileLang kernel 捕获成一张图并回放。

1. 按 `_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py`
   的写法完成捕获与 `replay()`；
2. 按该目录 README 的方法打开 host 日志，确认 `capture model` 的值为 1；
3. 解释：为什么捕获期间**不能**换输入张量？如果要换，应该怎么做？
4. 说明这个能力**不在** v0.1.15 主干里——给出核验命令与依据。

---

### E12 ★ 概念（无硬件）

1. v0.1.15 的 TileLang 是否提供 autograd 集成？给出至少一条可复现的检索命令与结果。
2. `kernel(a, b)` 返回的张量有没有 `grad_fn`？为什么？
3. 训练一个用 TileLang kernel 的模型时，反向从哪里来？请描述一种可行做法，
   并指出仓库里哪些文件是这种做法的现成例子。

---

### E13 ★★ 代码阅读（无硬件）

阅读 `tilelang/cache/kernel_cache.py:787-838`。

1. `manifest.json` 里记了每个文件的什么信息？
2. 加载时对**所有**文件和对**必需文件**分别做了什么校验？为什么两者不一样？
3. 校验失败时源码做了什么？为什么是「删掉整条目录」而不是「就地修复」？
   （提示：正文 4.5 节最后一段）
