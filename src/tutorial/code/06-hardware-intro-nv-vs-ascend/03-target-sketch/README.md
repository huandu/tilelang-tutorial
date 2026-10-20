# 03 · 两台的 tile 程序骨架（静态核对）

## 目的

把本章第 3、4 节的硬件抽象“翻译”回 TileLang：同一件 `C = relu(A @ B)` 的事，在两台
机器上分别长什么样。两个文件都是**阅读材料**，逐行标注了它对应本章哪条硬件路径。

## 内容

| 文件 | 对应硬件 | 关键行 | 硬件路径 |
| --- | --- | --- | --- |
| `nv_hopper_sketch.py` | NVIDIA Hopper | `T.alloc_shared` / `T.alloc_fragment` | 寄存器 + shared memory |
| | | `T.copy(..., ...)` | global → shared（Hopper 上可降级为 TMA） |
| | | `T.gemm(...)` | 可选择 `mma` / `wgmma` 路径 |
| | | `T.Pipelined(..., num_stages=3)` | 多缓冲 + 隐式 mbarrier |
| `ascend_950_sketch.py` | 昇腾 950（Atlas 350） | `T.alloc_l1` | GM → L1（MTE2） |
| | | `T.gemm(..., C_l0c)` | L1 → L0A/L0B（MTE1）+ Cube |
| | | `T.dual_copy(C_l0c, C_ub)` | L0C → UB（Fixpipe），按 M 拆给两个 AIV |
| | | `with T.SimtVF(threads=128):` | Vector 核上的 SIMT 区域 |
| | | `T.dual_copy(C_ub, C[...])` | UB → GM（MTE3） |

## 运行前提

* `nv_hopper_sketch.py`：NVIDIA GPU + CUDA toolkit + PyTorch(CUDA) + TileLang v0.1.15。
* `ascend_950_sketch.py`：昇腾 NPU（Atlas 350 / 950 系列）+ CANN（含 `bisheng`）+
  PyTorch + `torch_npu` + TileLang v0.1.15。

两个文件都用一个 `try: import tilelang` 把 kernel 定义包起来，因此在没有对应软件栈的
机器上**仍然能通过语法检查并打印说明**；但也因此，没装 TileLang 时它们不会尝试编译
kernel。

## 运行命令

在本机（无 GPU/NPU）只能做静态检查：

```bash
python3 -m py_compile nv_hopper_sketch.py ascend_950_sketch.py
python3 nv_hopper_sketch.py      # 打印说明，不编译 kernel
python3 ascend_950_sketch.py     # 打印说明，不编译 kernel
```

在有卡机器上，需要把 kernel 从 `try/except` 里拿出来，补上数据与调用：

```bash
# NVIDIA：M=N=K=4096，fp16 输入，fp32 累加，与 torch.relu(a @ b) 对比
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# 昇腾：确认 torch_npu 可用后再跑
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"
```

### 期望输出（正确性判据）

两台机器上的判据相同：输出应与 `torch.relu(A.float() @ B.float().T)`（昇腾）或
`torch.relu(A.float() @ B.float())`（NV）在 `rtol=1e-2, atol=1e-2` 内相等——这是
bf16/fp16 混合精度的常见容差，**不是**某次真实运行的记录。性能口径必须同时说明
dtype、shape、是否含 epilogue，否则数字不可比（见本章 7.3 的提醒）。

## 作者实际做到的验证级别

**仅静态核对。**

* `python3 -m py_compile` 两个文件均通过（Python 3.11）。
* 每个 API 名在锁定版本 `_vendor/tilelang-v0.1.15/` 中核对过存在性与签名，出处写在
  各文件的 docstring 里。
* **没有**在 NVIDIA GPU 或昇腾 NPU 上编译或运行过；**没有**任何实测数值。

把“仅静态核对”当成“运行验证”是本项目反复强调的禁忌，本目录严格遵守。
