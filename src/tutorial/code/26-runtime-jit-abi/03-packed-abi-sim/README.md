# 示例 3：packed host ABI 的检查规则模拟

## 目的

`MakePackedAPI` 会把你的 kernel 入口改写成 `(self_handle, args, num_args, result)`，
并在里面插入一整套参数检查。这些检查只存在于编译产物里，没有加速器就看不着。

本示例把**规则**抽出来用纯 Python 重写，然后跑 13 个场景，逐个对照源码里的判定与报错文案：

| 场景 | 源码依据 | 期望结果 |
| --- | --- | --- |
| 参数个数不对 | `make_packed_api.cc:473-477` | `vecadd: num_args should be 2` |
| 维度数不对 | `tvm_ffi_binder.cc:461-467` | `kernel vecadd input A ndim mismatch, expected 2` |
| dtype 不对 | `tvm_ffi_binder.cc:580-587` | `kernel vecadd input A dtype mismatch, expected ...` |
| 用到的 buffer 传 NULL | `tvm_ffi_binder.cc:390-395` | `vecadd.A is expected to have non-NULL pointer` |
| 形状不满足约束 | `tvm_ffi_binder.cc:30-54` | `kernel vecadd input A violates packed ABI constraint` |
| 标量类型码不对 | `make_packed_api.cc:682-691` | `kernel vecadd scalar n expected int` |
| int 标量收到 bool | `make_packed_api.cc:686-690` | 接受 |
| fp8 e4m3fn 顶替 e4m3 | `tvm_ffi_binder.cc:489-501` | 接受 |
| fp8 e5m2 顶替 e4m3 | `tvm_ffi_binder.cc:489-501` | 拒绝 |
| bool buffer 收 int8 存储 | `tvm_ffi_binder.cc:521-544` | 接受 |
| 声明连续却传了非连续 stride | `tvm_ffi_binder.cc:296-328` | 违反约束 |
| fp4 总位数不匹配 | `tvm_ffi_binder.cc:545-579` | `is a subtype, but total bits mismatch` |

## 运行前提

* Python 3.9 以上（只用标准库）。
* **不需要** GPU / NPU / tilelang / torch。

## 运行命令

```bash
cd chapters/26-runtime-jit-abi/code/03-packed-abi-sim
python3 packed_abi_sim.py
python3 packed_abi_sim.py --trace
```

## 期望输出

```text
ok   correct call                                   -> accepted
ok   wrong arg count                                -> vecadd: num_args should be 2
...
ok   fp4 with the wrong total bit count             -> vecadd.A is a subtype, but total bits mismatch

13/13 scenarios behaved as the source says
```

## 正确性判据

* 退出码 0；
* `13/13`；
* 任何一条 `FAIL` 都表示「模拟与源码描述不符」——先回去读源码，再改这个脚本。

## 这个示例**没有**证明什么

它是模拟，不是 TileLang 的实现。它证明的是：**源码里写的规则可以被独立复现，
而且能解释你实际会看到的报错文案**。真正的检查由生成的 host stub 在设备调用时执行，
本机无法运行。真实报错文案的确切措辞可能与这里略有差异（例如 dtype 打印格式），
以源码里的 `ostringstream` 拼接为准。

## 作者实际做到的验证级别

**在本机运行过（2026-10-01）**：输出 `13/13 scenarios behaved as the source says`，退出码 0。
