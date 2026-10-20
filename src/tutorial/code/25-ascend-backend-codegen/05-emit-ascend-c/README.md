# 05-emit-ascend-c —— 打印并逐段核对生成的 Ascend C

## 目的

第 25 章讲的是“生成物”。这个示例给你一个**真实可编译的 kernel**，并用
`get_kernel_source()` 把编译器生成的设备源码打印出来，再按正文 §4.2–4.6 的清单
逐段核对：scope 限定符、DMA intrinsic 名、L0 MAD、`asc_syncthreads`、多缓冲 slot。

## 运行前提

* **仅 `--audit` 模式**：Python 3.10+，任意机器。它不导入 tilelang，只做两件事：
  `py_compile` 本脚本 + 扫描 `_vendor/` 确认 kernel 里用到的 API 名在锁定版本里存在。
* **`--emit` 模式**（真正打印 Ascend C）：需要
  1. `_vendor/tilelang-v0.1.15` 用 `USE_ASCEND=ON USE_CUDA=OFF` 构建安装（见
     `tilelang/ascend/README.md:32-51`）；
  2. CANN 工具链（`bisheng`、CCE 版 `ld.lld`、Ascend runtime 库）在 `PATH`
     / `ASCEND_HOME_PATH` 里；
  3. Ascend 950（`dav-3510`）；A 树路线需要 A2/A3。

## 运行命令

```bash
# 任意机器：只做静态审计
python3 print_ascend_c.py --audit

# 有昇腾环境的机器：M 树，打印 target="ascend" 生成物
python3 print_ascend_c.py --emit --target ascend

# A 树：AscendC 与 PTO 两条 codegen
python3 print_ascend_c.py --emit --target ascendc
python3 print_ascend_c.py --emit --target pto
```

## 期望输出（`--emit --target ascend`）

脚本会先 `print(kernel.get_kernel_source())`，然后按清单打印判定：

```
[1] kernel signature           : expect extern "C" __global__ __mix__(1, 2) / __vector__
[2] UB allocation             : expect __ubuf__ .* or T.alloc_shared -> __ubuf__
[3] GM->L1 DMA                : expect asc_copy_gm2l1_nd2nz / dn2nz
[4] L1->L0A/L0B DMA           : expect asc_copy_l12l0a / asc_copy_l12l0b
[5] Cube MAD                  : expect asc_mmad
[6] L0C->UB dual-destination  : expect asc_copy_l0c2ub with asc_dual_dst_mode
[7] UB->GM DMA                : expect asc_copy_ub2gm_align
[8] SIMT barrier              : expect asc_syncthreads
[9] multibuffer slot indexing : count distinct buffer slot offsets (>= num_stages)
```

脚本把每条“命中/未命中”都打印出来，但对**生成代码**不做自动语义判定——它只是
把你该看的地方指出来。

## 作者实际做到的验证级别

**`--audit`：CPU 逻辑验证**（本机已跑，见正文 §7）。机器上**没有昇腾 NPU，
也没有 CANN/bisheng**，所以 `--emit` 一次都没有运行过，**本目录不提供任何实测
性能或正确性结论**。

## 关于这个 kernel

它改写自锁定版本的官方示例 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:66-108`
（GEMM + ReLU）。这里把尺寸缩小、并用 `--check` 做 API 名审计，方便在没有卡的
机器上先确认“API 名没写错”，再上机看生成物。
