# 示例 4：从 PyTorch 调用 kernel，并观察落到哪条 stream 上

## 目的

验证正文“torch 集成”一节的两条论断：

1. 默认执行后端（`tvm_ffi`）在**调用时**读 PyTorch 的 current stream，
   并把 kernel 提交到那条 stream 上（`tilelang/jit/adapter/base.py:61-101`）；
2. 跨 stream 使用时，比较结果必须在同一条 stream 上做——否则读到的是竞态，
   而且**大多数时候看起来是对的**，这正是它危险的地方。

## 运行前提

* 一块 CUDA GPU，驱动可用；
* `torch` 与 `tilelang==0.1.15` 均已安装且能 `import`；
* 昇腾机器上把脚本里的 `torch.cuda.*` 换成 `torch.npu.*`（`torch_npu` 会自动
  安装 NPU stream 交接，见 `_vendor/tilelang-v0.1.15/tilelang/ascend/torch_exchange.py:6-20`）。

## 运行命令

```bash
cd chapters/26-runtime-jit-abi/code/04-torch-stream-call
python3 torch_stream_call.py
```

## 期望输出

```text
out device : cuda:0
out dtype  : torch.float16  shape: (4096,)
matches reference: True
note: removing side.synchronize() above makes the comparison racy.
```

## 正确性判据

* 两次比较都为 `True`；
* `out.device` / `out.dtype` / `out.shape` 与 kernel 声明一致；
* “反例”注释块保持注释状态——它的价值在于说明**去掉同步会发生什么**，
  而不是作为可运行的代码。

## 反例说明了什么

```python
with torch.cuda.stream(side):
    out_side = kernel(a, b)
print(torch.allclose(out_side, ref))   # 没有 sync
```

`allclose` 自己也要在 device 上跑；它提交在**默认 stream** 上，而 kernel 在 `side` 上。
两条 stream 之间没有依赖边，`allclose` 可能读到还没写完的输出。
症状是“偶尔错、偶尔对”，单次运行几乎发现不了；只有 `N` 次重复 + 每次强制同步才能稳定复现。

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile` 通过；`tilelang.jit(out_idx=...)`、
`T.Kernel(...)`、`JITKernel.__call__` 等 API 名与 `_vendor/tilelang-v0.1.15/` 逐个比对。
本机是 macOS，无 CUDA GPU、未安装 tilelang / torch，**未运行**。上面“期望输出”
是按 API 语义写出的应然结果，不是实测记录。
