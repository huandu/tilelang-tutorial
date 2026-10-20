# 图 3：`MakePackedAPI` 生成的 host 入口

这张图回答一个具体问题：**`kernel(a, b)` 这一行 Python，落到 host 侧到底调了什么、
它替你检查了什么。** 所有行号来自 `_vendor/tilelang-v0.1.15/`。

## 1. 入口签名变了

编译前（你写的 TIR 函数，参数就是你声明的张量/标量）：

```text
main(A: T.Tensor((M, K), fp16), B: T.Tensor((K, N), fp16), C: T.Tensor((M, N), fp16))
```

编译后（`MakePackedAPI` 改写，`src/transform/make_packed_api.cc:764`）：

```text
(void* self_handle, TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)
```

`num_args` 的算法在 `:423-431`：默认等于原函数参数个数；走
callee-allocated-output ABI 时**减掉全部输出参数、再加 1 个 allocator anchor**。
函数签名被改写为 `kCPackedFunc`、global symbol 加上
`tvm_ffi_symbol_prefix`（`:1006-1009`）。

## 2. 运行时检查的插入顺序（这才是「ABI」的实体）

```text
seq_init            ← 全部由 MakePackedAPI 生成
  ├─ ① num_args 断言                    :473-477  "num_args should be N"
  ├─ ② args 指针非空断言（num_args>0）   :480-482  "args pointer is NULL"
  ├─ ③ 逐参数 type_index 检查            :615-719
  │     handle → 接受 None / OpaquePtr / DLTensorPtr / StaticObject
  │              （:652-658，报错文案 "expected pointer or tensor handle"）
  │     bool   → 接受 Bool / Int        （:674-678）
  │     int    → 接受 Int / Bool        （:686-690）
  │     float  → 接受 Float / Int / Bool（:697-702）
  ├─ ④ callee-allocated anchor 检查（仅该 ABI）  :721-758
  │     anchor 必须是 Tensor、非空、device type 与 target 一致
  └─ ⑤ 标量/句柄参数绑定（binder.Bind）   :773-775
binder.InitNest()   ← 由 TVMFFIABIBuilder::BindDLTensors 生成
  ├─ is_null 变量 + 使用中的 buffer 必须非空   tvm_ffi_binder.cc:380-395
  ├─ shape 指针按 is_null 短路取值             :407-420
  ├─ ndim 断言（a_ndim == v_ndim）              :433-467
  ├─ dtype 三元组检查 (code, bits, lanes)       :470-487
  │     例外：float8 e4m3 / e5m2 家族互相放宽（:489-511）、
  │           tfloat32 ↔ float32（:512-520）、
  │           bool ↔ int8/uint8/kDLBool（:521-544）、
  │           亚字节 dtype 按「总位数相等」放行（:545-579）
  ├─ 逐维 shape 绑定 + 约束断言                 :592-...
  └─ stride 检查（含 relaxed stride）           RelaxedStrideCheck, :296-328
seq_check           ← 设备上下文
  └─ 设置 device_id / device_type              make_packed_api.cc:1021-1034
body
  └─ ReturnRewriter(v_result) 改写 tir.ret     :1011、:100-182
```

## 3. `const_*` 断言 vs 运行期断言

| 断言种类 | 什么时候会触发 | 生成位置 |
| --- | --- | --- |
| 编译期常量折叠后为假 | **编译期** `LOG(FATAL) "Bind have an unmet assertion"` | `tvm_ffi_binder.cc:56-63` |
| 编译期无法判定 | 运行期 `AssertStmt`，错误信息形如 `kernel <name> input <buf> ndim mismatch, expected N` | `tvm_ffi_binder.cc:30-54`（文案）、`:465-467`（ndim） |
| 参数个数 / 类型码 | 运行期 | `make_packed_api.cc:473`、`:652` |

所有运行期断言的错误都通过 TVM-FFI 的 C ABI error slot 抛回 Python，
**不回调 Python 侧 helper**（`src/transform/tvm_ffi_binder.h:32-36`）。

## 4. 输出张量由谁分配

两条路径，由 `tilelang_callee_allocated_outputs` 属性在**编译期**选定
（`tilelang/jit/abi.py:16`、`:39-74`）：

| 路径 | 谁分配输出 | 何时使用 |
| --- | --- | --- |
| 调用方预分配 | host wrapper / adapter 在 Python 侧 `torch.empty(...)` | 默认；`supports_callee_allocated_outputs=False` |
| callee 分配 | 生成的 host 函数自己构造 DLTensor 原型并交给 FFI allocator | 后端声明 `supports_callee_allocated_outputs=True`：CUDA tvm_ffi 与 Ascend tvm_ffi（`tilelang/cuda/execution_backend.py:25-30`、`tilelang/ascend/execution_backend.py:8-19`） |
