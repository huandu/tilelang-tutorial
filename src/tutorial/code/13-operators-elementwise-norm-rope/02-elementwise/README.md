# 02 · 逐元素算子的四级实现

## 目的

同一件事（`y = x + b`，外加一个 scale+cast 版本）的四级写法，用来对照
「编译产物里的访存宽度」：

1. `add_flat` —— 一个 block 处理 `threads` 个元素，每线程 1 个元素；
2. `add_thread_vec` —— 显式写出「每线程 `num_per_thread` 个元素」，相邻元素同线程；
3. `add_register_vec` —— 先用 `T.copy` 把两块数据搬进 fragment，再算，再写回；
4. `scale_cast` —— 乘标量 + dtype 转换；
5. `per_token_cast_to_fp8` —— cast 的进阶形态：分组量化（cast 之前先做一次归约）。

## 运行前提

* NVIDIA GPU（SM80+ 足够，本例不依赖 TMA/tensor core）
* CUDA + PyTorch(CUDA)
* TileLang **v0.1.15**（见 `../../../../VERSIONS.md`）

4 号、5 号只需要 `py_compile` 与 API 核对，不需要单独准备数据。

## 运行命令

```bash
cd 02-elementwise
python3 -m py_compile elementwise_variants.py     # 本机可做（无需 GPU）
python3 elementwise_variants.py                   # 真机：跑正确性
```

想看生成代码（第 12 章的方法）：

```python
import torch, tilelang
from elementwise_variants import add_flat, add_thread_vec, add_register_vec

n = 1 << 20
dtype = "float32"
for name, built in (
    ("flat", add_flat(n, threads=256, dtype=dtype)),
    ("thread_vec", add_thread_vec(n, threads=256, num_per_thread=8, dtype=dtype)),
    ("register", add_register_vec(n, threads=256, num_per_thread=8, dtype=dtype)),
):
    src = built.get_kernel_source()
    print(f"===== {name} =====")
    print(src[:2000])          # 找 uint4 / float4 之类的向量访存
```

## 预期输出

**正确性判据**：三个 add 版本与 `a + b` 在 `rtol=atol=1e-5` 内一致；
`per_token_cast_to_fp8` 与官方 `ref_program` 在 `rtol=atol=1e-2` 内一致。

**性能口径**：不要用「哪个快」下结论，先看生成代码里有没有 128-bit 访存。
建议用 `tilelang.profiler.do_bench` 记录有效带宽
（`3 * N * 4 / latency`，两个读一个写），并与该卡的 peak bandwidth 比。**本机没有这些数字。**

**期望看到的生成代码差异**（官方 elementwise 文档给的对照，方向可复用）：

* `add_flat`：`T.Parallel(threads)` 的 extent 只有 `threads`，可用并行度不足以塞满
  128-bit 向量，编译器会自己重排成「32 个线程各搬 8 个元素」的 `uint4` 形态，
  **但 launch config 仍是 256 线程**；
* `add_thread_vec`：每线程连续的 8 个元素 → 一次 `uint4` 读 + 一次 `uint4` 写；
* `add_register_vec`：多出两个寄存器数组，但访存仍是 `uint4`。

## 已知的边界与坑（写在这里，免得你踩完才发现）

* `add_register_vec` 要求 `N % (threads * num_per_thread) == 0`（代码里有断言）。
  非对齐长度请用 `add_thread_vec`，或者自己在 kernel 里写尾块分支。
* 浮点加法不满足结合律：向量化后每条 lane 计算的是同一对元素，
  所以结果与标量版**逐位相同**；但如果以后改成 `reduce_sum` 的形态，
  归约顺序会改变数值，不要用 `rtol=0` 去比。
* `scale_cast` 走的是 fp32→fp16 的跨 dtype cast。跨 dtype 的向量化宽度受限于
  源侧与目标侧 lane 数的较小值（正文 §5.1）。

## 作者验证级别

**仅静态核对**：`python3 -m py_compile` 通过；用到的 40 个 `T.*` 名字经
`../07-api-audit/api_audit.py` 对照过 `_vendor/tilelang-v0.1.15/` 的语言面。
**未在 GPU 上编译或运行过**，因此本 README 不声明任何性能数字或正确性结论。
