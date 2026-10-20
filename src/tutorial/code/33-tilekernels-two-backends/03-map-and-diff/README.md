# 示例 03：清点三层结构 + 产出后端结构对照表

## 目的

正文 §2 与 §4.4 的**数据来源**。这个脚本不 import `torch`、不 import `tilelang`、
不需要任何加速器：它只读源码文本，做三件事：

1. 扫描一个 TileKernels 源码树，列出所有 `*_kernel.py` / `*_cuda.py` / `*_asc.py`
   三元组（缺哪个也照实报告）；
2. 对每一对 `*_cuda.py` / `*_asc.py`，统计一组**结构性标记**的出现次数，
   产出对照表（正文 §4.4 那张表就是它的第一行）；
3. 在 TileLang 锁定源码树里核对本章点名过的 API 是否**真的存在**。

它是「本章的结论不靠记忆、靠可复现的机械核对」这句话的落实。

## 运行前提

* Python 3.9+，**仅标准库**；
* 一个 TileKernels 源码树（默认指向 corpus 快照，
  见 `../references.md` 的 `T1`）；
* 可选的 TileLang 锁定源码树（用于第 3 步；默认指向 `_vendor/tilelang-v0.1.15/`）。

## 运行命令

```bash
# 用默认路径（本章写作时用的 corpus 快照与 vendor 树）
python3 map_and_diff.py

# 指定别的 TileKernels 检出（例如 git clone 下来的最新 main）
python3 map_and_diff.py --root /path/to/TileKernels/tile_kernels

# 只做前两步（不核对 API）
python3 map_and_diff.py --skip-api-check
```

## 作者实际运行过的输出（`--root` 用默认值）

### 第 1 步：三层结构清单

```
扫描根目录：.../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels
== 1. 三层结构清单 ==

engram/engram_fused_weight                    ✓       ✓     ✓
engram/engram_gate_bwd                        ✓       ✓     ✓
...
moe/moe_topk_gate_forward                     ✓       ✓     ✓
moe/normalize_weight                          ✓       ✓     ✓
moe/topk_gate                                 ✓       ✓     ✓
quant/per_token_cast                          ✓       ✓     ✓
quant/swiglu_forward                          ✓       ✓     ✓
rand/randn                                    ✓       —     ✓
transform/rope                                ✓       ✓     ✓
------------------------------------------------------------------
合计 33 个算子家族；三件齐全的 32 个。
```

`rand/randn` 是唯一没有 CUDA 实现的一个——正文 §6.2 专门讲了它。

### 第 2 步：结构性对照（本章的核心样本）

```
--- moe/topk_gate ---
结构性标记                               cuda   asc
CUDA 版按 token 开网格                      1     0  <-- 只在一边
Ascend 版按 Vector 核开网格                  0     1  <-- 只在一边
显式线程数（SIMT）                            1     0  <-- 只在一边
fragment（寄存器）                          5     0  <-- 只在一边
reducer（延迟归约）                          2     0  <-- 只在一边
shared / UB                            1     3
显式缓冲版本                                 0     1  <-- 只在一边
软件流水                                   0     1  <-- 只在一边
SIMD 向量作用域                             0     2  <-- 只在一边
手写同步                                   0     1  <-- 只在一边
PDL（CUDA-only）                         1     0  <-- 只在一边
显式向量宽度                                 0    15  <-- 只在一边
位视图                                    0     1  <-- 只在一边
```

这张表把 §4.4 的每一行都变成了可复算的计数。注意 `显式向量宽度` 是 **0 : 15**
——「昇腾把宽度写出来」不是修辞，是 15 处 `vector_length` 相关引用。

### 第 3 步：API 存在性核对

```
== 3. 本章用到的 API 在锁定版本里的存在性 ==
alloc_reducer              ok  (tilelang/language/allocate.py)
reducer_init               ok  (tilelang/language/reduce_op.py)
reducer_update             ok  (tilelang/language/reduce_op.py)
finalize_reducer           ok  (tilelang/language/reduce_op.py)
pdl_sync                   ok  (tilelang/cuda/language/pdl.py)
pdl_trigger                ok  (tilelang/cuda/language/pdl.py)
view                       ok  (tilelang/language/customize.py)
SimdVF                     ok  (tilelang/ascend/language/frame.py)
SimtVF                     ok  (tilelang/ascend/language/frame.py)
Kernel                     ok  (tilelang/ascend/language/kernel.py, ...)  (+5 more)
annotate_buffer_versions   ok  (tilelang/ascend/language/annotations.py)
Pipelined                  ok  (tilelang/language/loop.py)
Persistent                 ok  (tilelang/language/loop.py)
import_source              ok  (tilelang/language/common.py)
rng_init                   ok  (tilelang/ascend/language/random.py, tilelang/cuda/language/random.py)
rng_rand_float             ok  (tilelang/cuda/language/random.py)
warp_reduce_min            ok  (tilelang/language/reduce_op.py, ...)
warp_reduce_max            ok  (tilelang/language/reduce_op.py, ...)
warp_reduce_sum            ok  (tilelang/language/reduce_op.py, ...)
get_thread_binding         ok  (tilelang/ascend/language/kernel.py, tilelang/language/kernel.py)  (+6 more)
device_assert              ok  (tilelang/ascend/debug.py, tilelang/cuda/debug.py)

21/21 命中；缺：无
```

**这一步只证明「名字存在」，不证明签名与语义**——本章正文的每条 API 结论
都是另外逐行读定义得到的（见 `../references.md` 的核验方式一栏）。

## 验证级别

**CPU 逻辑验证**：作者在 macOS (Apple Silicon)、Python 3.11 上**实际运行过**，
输出就是上面这三段（行号与语料快照 commit `66258df6175d` 对应）。

## 已知局限

* 结构性标记是**字符串计数**，不是 AST 分析：注释里出现的同名子串也会被计入。
  它用来做「一边有一边没有」的快速对照，不用于精确统计。
* 默认的 `--root` / `--vendor` 是相对于本文件的路径；把示例复制到别处时请显式传参。
* 换一个 TileKernels 版本，计数会变；对照结论只对本章锁定的快照成立。
