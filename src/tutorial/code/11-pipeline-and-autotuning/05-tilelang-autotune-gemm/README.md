# 05 · GEMM 自动调优的两条接口（TileLang）

## 目的

v0.1.15 同时提供两条自动调优入口，本示例各演示一条：

| 入口 | 代码位置 | 特点 |
| --- | --- | --- |
| `@tilelang.autotune(configs=...)` 叠在 `@tilelang.jit` 上 | `matmul_tuned` + `main_tuned` | 声明式；`configs` 可以是列表，也可以是 `(M, N, K, ...) -> list[dict]` 的函数 |
| `AutoTuner.from_kernel(...).set_compile_args(...).set_profile_args(...).run()` | `tune_once` | 程序式；能拿到 `AutotuneResult`，可保存/复用最优配置 |

`matmul_tuned` 是写成显式调用形式的同一件事：
`matmul_tuned = tilelang.autotune(configs=...)(matmul)`。`tilelang.autotune(...)` 返回的
decorator 会断言目标是一个 `JITImpl`
（`tilelang/autotuner/tuner.py:1397-1398`），所以“叠装饰器”与“显式调用”等价；
这样写的好处是同一个 `@tilelang.jit` 工厂可以同时服务两条入口——
`AutoTuner.from_kernel` 要的是**工厂本身**，不是已经叠了自动调优的那个 wrapper。

候选空间有两条来源：

| 来源 | 函数 | 说明 |
| --- | --- | --- |
| 手写笛卡尔积 + 预剪枝 | `grid_configs` | 先剔掉不整除问题尺寸、以及超 shared 预算的候选 |
| Carver 的 `MatmulTemplate` / roller | `roller_configs` | 由架构模型推荐 tile，再把 hint 映射成 kernel 参数 |

原型：`_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_autotune.py`。

## 运行前提

* NVIDIA GPU + CUDA toolkit + PyTorch（CUDA 版）+ TileLang `v0.1.15`；
* `roller_configs` 路径还需要 Carver 能识别目标架构（`tilelang.carver.arch.CUDA`）；
* 昇腾侧跑同一段逻辑需要换 dialect 与 `target="ascend"`，本示例不涉及。

## 运行命令

```bash
python3 autotune_gemm.py                    # 装饰器路径：首次调用触发调优
python3 autotune_gemm.py --programmatic     # AutoTuner 路径：打印最优配置与延迟
python3 autotune_gemm.py --topk 20          # 只影响 roller 的推荐条数
```

## 期望输出

装饰器路径：

```text
tuned kernel ready; bake the winning parameters as defaults for shipping
```

程序式路径会打印三行（`latency` / `config` / `ref_latency`），形如：

```text
best latency (ms): <float>
best config      : {"block_K": ..., "block_M": ..., "block_N": ..., "num_stages": ..., "threads": ...}
reference latency: <float>
```

**本目录不给出任何具体的延迟数值**：给不出来，因为写作机上没有 GPU。上表中的 `<float>`
是读者在自己机器上会看到的占位符，不是期望值。

正确性判据（由 `ref_prog` 与 `skip_check=False` 保证）：每个候选的输出与
`(a.float() @ b.float().T).to(fp16)` 在 `rtol=1e-2, atol=1e-2` 下一致；不一致的候选
会被判失败而不会被选为最优。

## 读结果时要注意的三件事

1. **`artifact.config` 才是可复现的产物**，`artifact.latency` 依赖机器与当前负载。
   把 config 抄回函数默认值，才是“固定最优配置”。
2. **`artifact.ref_latency` 是可选项**。没有提供 `ref_prog` 或参考实现时它会是
   `None`，不要把它当成基线。
3. **调优只在给定的候选集合里选最优**。候选集合定义得不好（见
   `../03-autotune-config-space/`），调优结果再好也只说明“在这几个里最好”。

## 作者实际做到的验证级别

**仅静态核对**：

* `python3 -m py_compile autotune_gemm.py` 通过；
* API 名与签名逐个在 `_vendor/tilelang-v0.1.15/` 中核对：
  `tilelang.autotune`（`tilelang/autotuner/tuner.py:1318`）、
  `AutoTuner`（`tilelang/autotuner/tuner.py:105`）、`AutoTuner.from_kernel`
  （同文件 `:144`）、`AutoTuner.set_compile_args`（`:156`）、
  `AutoTuner.set_profile_args`（`:211`）、`AutoTuner.run`（`:809`）、
  `set_autotune_inputs`（`tilelang/autotuner/capture.py:100`）、
  `AutotuneResult`（`tilelang/autotuner/param.py:155`）、
  `MatmulTemplate`（`tilelang/carver/template/matmul.py:10`）、
  `CUDA`（`tilelang/carver/arch/cuda.py:138`）、
  `NoRasterization`（`tilelang/carver/roller/rasterization.py:19`）。

**未在硬件上运行**，也没有任何调优产生的数字。
