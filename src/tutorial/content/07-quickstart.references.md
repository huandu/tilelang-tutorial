# 第 07 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新（发布）版」之类的模糊表述。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU，
也没有安装 tilelang**。因此本章所有涉及加速器的示例最高验证级别是「仅静态核对」——
API 名与签名逐个在 `_vendor/` 中比对过，**没有任何一条运行结果或性能数字来自本机实测**。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

本章正文引用的范围较广，这里按「安装 / 前端 / 运行时 / 昇腾 / 示例与文档」分组。
「已核验」的含义是：在锁定版本工作树上用 `grep -n` / `sed -n` 定位到该行，并阅读了
完整签名或段落；行号与文件内容一致。**这不是运行验证。**

### A. 安装与环境（正文第 3 节）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:3-10` | pip 安装前置条件（glibc 2.28、Python >= 3.10、CUDA >= 10.0 或 pip CUDA >= 13.0） | 已核验（2026-10-01）：逐行阅读 |
| S2 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:70-76` | 源码构建前置条件（Linux/Windows、Python >= 3.10、CUDA） | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:88-91` | `--recursive` 与定制 TVM submodule 的说明 | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:93-99` | host CUDA 工具链下的 `pip install . -v` | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:101-124` | pip 提供 CUDA 工具链（无 host CUDA）的两种选项 | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:133-158` | `PYTHONPATH` + cmake/ninja 开发路线与验证命令 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:160-166` | CMake 选项 `USE_CUDA/USE_ROCM/USE_ASCEND/USE_METAL/USE_LLVM` | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:332-341` | nightly 渠道（滚动构建，非本书依据） | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:365-376` | 版本字符串里的 SDK/git 后缀（`NO_VERSION_LABEL`） | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:389-391` | 编译缓存默认路径 `~/.tilelang/cache` 与 `ccache` | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/VERSION` | 内容为 `0.1.15` | 已核验（2026-10-01）：`cat VERSION` |

### B. 昇腾（正文第 3.4、7 节）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:10-51` | 昇腾 950 前置条件（CANN/bisheng/ld.lld/torch_npu）与源码构建 | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:27-31` | `bisheng` 在 PATH / `BISHENG_HOME/bin`；默认 `dav-3510` | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:138-154` | 昇腾写法要点与「与 Ascend C 的分工」对照表 | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:51-52` | 「在 A2 与 A3 上验证过」 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:76-134` | 三种安装方式（wheel / 打 wheel / install_ascend.sh）与 CANN、torch-npu 版本要求 | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-ascend-v0.1.2.000/install_ascend.sh:37-45` | 脚本的 Python >= 3.10 门 | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-ascend-v0.1.2.000/install_ascend.sh:171-243` | submodule、TVM 补丁、cmake/make 流程 | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-ascend-v0.1.2.000/build_wheel_ascend.sh:29-35` | 要求先设 `ASCEND_HOME_PATH` | 已核验（2026-10-01） |
| S20 | `_vendor/tilelang-ascend-v0.1.2.000/set_env.sh:1-20` | 设 `PYTHONPATH` 与 `ACL_OP_INIT_MODE=1` | 已核验（2026-10-01） |
| S21 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/cache/__init__.py:70` | 独立仓库**有**顶层 `clear_cache()`（与主干不同） | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-ascend-v0.1.2.000/VERSION` | 内容为 `0.1.4`（以 tag 为准） | 已核验（2026-10-01）：`cat VERSION` |
| S23 | `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py:31-46` | 独立仓库方言：`is_npu=True`、`alloc_ub`、`T.Scope("V")`、`barrier_all` | 已核验（2026-10-01） |
| S23a | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19-38` | `T.annotate_buffer_versions` 的语义：`num_stages > 1` 时多缓冲份数由自动调度器计算，该注解用于锁定份数 / 索引模式 | 已核验（2026-10-01）：逐行阅读 docstring；由此确定本章昇腾示例省略它是「语义应当等价、未在 NPU 上验证」的取舍 |

### C. 前端：网格、buffer、copy、parallel（正文第 2、4 节）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S24 | `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455` | `T.Kernel` 的 "grid of tile programs" / "target-neutral launch" 语义与参数说明 | 已核验（2026-10-01）：`grep -n "^def Kernel"` → 431 |
| S25 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-52` | CUDA 方言的 `threads=`（默认 128）、`T.get_thread_binding()` 与 docstring 示例 | 已核验（2026-10-01） |
| S26 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13-80` | `T.Parallel` 是块级并行域；布局注解约束由 `ParallelLoopLayoutValidator` 检查 | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:112` | `T.Pipelined(...)` 签名 | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-132` | `T.copy` 是 tile op；形状推断；「首地址」语法糖 | 已核验（2026-10-01）：docstring "Syntactic Sugar" 在 `:99` |
| S29 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49` | `T.alloc_shared(shape, dtype, scope="shared.dyn")` | 已核验（2026-10-01） |
| S30 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:81` | `T.alloc_fragment(shape, dtype, scope="local.fragment")` | 已核验（2026-10-01） |
| S31 | `_vendor/tilelang-v0.1.15/tilelang/language/fill_op.py:48` | `T.clear(buffer)` | 已核验（2026-10-01） |
| S32 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160` | `T.gemm(...)` 签名 | 已核验（2026-10-01） |
| S33 | `_vendor/tilelang-v0.1.15/tilelang/language/tir/entry.py:10` | `@T.prim_func` 的定义 | 已核验（2026-10-01） |
| S34 | `_vendor/tilelang-v0.1.15/tilelang/language/symbolics.py:13` | `T.dynamic(name, dtype)` | 已核验（2026-10-01） |
| S35 | `_vendor/tilelang-v0.1.15/tilelang/language/__init__.py:1-32` | `tilelang.language` 是 CUDA 方言的 re-export，`__tilelang_dialect__ = "cuda"` | 已核验（2026-10-01） |
| S36 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-174` | 昇腾 `T.Kernel`：1-D 核网格、无 `threads=`、多于 1 维直接报错 | 已核验（2026-10-01）：`raise` 在 `:172-173` |
| S37 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:165-168` | docstring 自带的昇腾最小示例 | 已核验（2026-10-01） |
| S38 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads=128, latency=0)`、`T.SimdVF(latency=0)` | 已核验（2026-10-01） |
| S39 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15,26,37,46,55` | `alloc_shared`（UB）/`alloc_l1`/`alloc_l0a`/`alloc_l0b`/`alloc_l0c` 的默认 scope | 已核验（2026-10-01） |
| S40 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53` | `T.dual_copy` 定义（L0C→UB 专用搬运） | 已核验（2026-10-01） |
| S41 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/__init__.py:126` | `__tilelang_dialect__ = "ascend"` | 已核验（2026-10-01） |

### D. 运行时、JIT 与缓存（正文第 5、6 节）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S42 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:94` | 模块级 `tilelang.compile(...)` | 已核验（2026-10-01） |
| S43 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:270-330` | `JITImpl` 与 lazy / eager 两种模式的说明 | 已核验（2026-10-01）：class 定义在 `:270` |
| S44 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:466-505` | `JITImpl.compile` 与 `debug_root_path` 落盘行为 | 已核验（2026-10-01） |
| S45 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:519-560` | `JITImpl.__call__` 与 mode 自动推断、kernel 复用 | 已核验（2026-10-01）：`def __call__` 在 `:519` |
| S46 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:571-635` | `@tilelang.jit` 装饰器签名（`out_idx`/`target`/`execution_backend`/`pass_configs` …） | 已核验（2026-10-01） |
| S47 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:206-220` | `JITKernel.__call__` 转发到 `adapter.func` | 已核验（2026-10-01） |
| S48 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:516` | `JITKernel.get_profiler()` | 已核验（2026-10-01） |
| S49 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532-600` | `get_kernel_source` / `get_host_source` / `show_source` / `export_sources` | 已核验（2026-10-01）：四者分别在 `:532`/`:545`/`:557`/`:592` |
| S50 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:243-284` | `_generate_key`：进 key 的字段与 sha256 构造 | 已核验（2026-10-01） |
| S51 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:141-142` | key 里是否含库内容哈希由 `should_use_kernel_cache_lib_stamp()` 决定 | 已核验（2026-10-01） |
| S52 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:455` | `KernelCache.clear_cache()` 是**实例方法** | 已核验（2026-10-01） |
| S53 | `_vendor/tilelang-v0.1.15/tilelang/cache/__init__.py:24-97` | 两条 dispatch map（普通 / 昇腾）与模块级 `cached()`；**无顶层 `clear_cache`** | 已核验（2026-10-01）：`grep -n "^def clear_cache\\("` 无输出 |
| S54 | `_vendor/tilelang-v0.1.15/tilelang/env.py:368` | `TILELANG_CACHE_DIR` 默认 `~/.tilelang/cache` | 已核验（2026-10-01） |
| S55 | `_vendor/tilelang-v0.1.15/tilelang/env.py:372-377` | `TILELANG_DISABLE_CACHE`、`TILELANG_KERNEL_CACHE_USE_LIB_STAMP` | 已核验（2026-10-01） |
| S56 | `_vendor/tilelang-v0.1.15/tilelang/env.py:400` | `TILELANG_ENABLE_IR_SPAN`（诊断时回报源码 span） | 已核验（2026-10-01） |
| S57 | `_vendor/tilelang-v0.1.15/tilelang/env.py:438-441,558-561` | `enable_cache` / `disable_cache` 的实现与顶层导出 | 已核验（2026-10-01） |

### E. 官方示例与文档（正文第 4、5 节）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S58 | `_vendor/tilelang-v0.1.15/examples/quickstart.py:1-86` | GEMM + ReLU 的完整示例（含 `get_kernel_source()` 与 `do_bench`） | 已核验（2026-10-01）：逐行阅读；**未运行**（无 GPU） |
| S59 | `_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py:11-63` | 官方 1-D 逐元素加示例（eager 写法 + `assert_close`） | 已核验（2026-10-01）；未运行 |
| S60 | `_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py:11-44` | 官方昇腾 vecadd（SIMT / SIMD 两种模式） | 已核验（2026-10-01）；未在 NPU 上运行 |
| S61 | `_vendor/tilelang-v0.1.15/examples/ascend/README.md:19-25` | 昇腾示例的性能口径（带宽 GB/s 的计法） | 已核验（2026-10-01） |
| S62 | `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:26-41` | 最小向量加示例与「block level, not thread level」的官方表述 | 已核验（2026-10-01） |
| S63 | `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:57-87` | 二维向量加示例（`bx`/`by` 参数顺序） | 已核验（2026-10-01） |
| S64 | `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:104-133` | 边界处理与「边界判断挡住向量化」的评语 | 已核验（2026-10-01） |
| S65 | `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:155-172` | 向量化版生成 CUDA（`uint4` + `nv_bfloat162`） | 已核验（2026-10-01）：逐行比对 |
| S66 | `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:176-211` | `num_per_thread` 建议与对应的生成 CUDA | 已核验（2026-10-01） |
| S67 | `_vendor/tilelang-v0.1.15/docs/programming_guides/language_basics.md` | `@T.prim_func`、`T.Tensor`、动态维度（第 8 章前置） | 已核验（2026-10-01）：文件存在，正文与示例逐段阅读 |
| S68 | `_vendor/tilelang-v0.1.15/docs/tutorials/debug_tools_for_tilelang.md` | 编译失败 / 数值错误的排查路线（第 12 章前置） | 已核验（2026-10-01） |

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/Installation.md` | https://tilelang.com/get_started/Installation.html | 官方安装指南的站点源文件；与 S1–S10 引用的仓库内文档**内容相同** | 已核验（2026-10-01）：与 `_vendor/.../docs/get_started/Installation.md` 做 diff，差异只出现在 4 个含 em-dash（`—`）的行上（corpus 侧是重复编码），正文其余部分逐行一致 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md` | https://tilelang.com/get_started/overview.html | 三层编程接口与六步编译流程（第 4 章已引；本章第 2 节的心智模型与之对齐） | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/targets.md` | https://tilelang.com/get_started/targets.html | `target` 的取值与 `auto` 探测（本章 3.x、6.x 的 `target=` 参数背景） | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/elementwise.md` | https://tilelang.com/deeplearning_operators/elementwise.html | 逐元素算子教程的站点源文件；与 S62–S66 引用的仓库内文档**内容相同** | 已核验（2026-10-01）：与 `_vendor/.../docs/deeplearning_operators/elementwise.md` 做 diff，差异均为 em-dash（`—`）编码，正文与示例代码一致 |
| R5 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/language_basics.md` | https://tilelang.com/programming_guides/language_basics.html | 语言基础（第 8 章前置） | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md` | https://tilelang.com/tutorials/debug_tools_for_tilelang.html | 调试工具（第 12 章前置） | 已核验（2026-10-01） |
| R7 | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/docs/get_started/Installation.md` | https://github.com/tile-ai/tilelang | 主仓库快照里的同类文档，用于对照「corpus 的 main 快照比 v0.1.15 多一个提交」这一事实 | 已核验（2026-10-01）：文件存在；**仅用于对照，不作为本章依据** |
| R8 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/README.md` | https://github.com/tile-ai/tilelang-ascend | 昇腾适配器仓库的安装说明（与 S15–S16 引用的 `_vendor` 快照同源） | 已核验（2026-10-01） |
| R9 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md` | https://github.com/tile-ai/tilelang-ascend/blob/master/docs/TileLang-Ascend%20Programming%20Guide.md | 昇腾编程指南；本章只在 7.2 节做定位性引用，细节留到第 19 章 | 已核验（2026-10-01） |

---

## 上游链接核验说明

* 上表「已核验」的链接，核验方式是：确认对应 corpus 文件在本地存在，且其头部记录了
  抓取日期与原始 URL。本章写作时**未重新发起网络请求**（corpus 抓取日期 2026-09-30，
  本章写作日期 2026-10-01，相差 1 天）。
* `R1`、`R4` 与 `_vendor` 内的同名文档做了逐行 `diff`，确认是同一份内容；两者可互相
  印证，因此本章把「安装说明来自 v0.1.15」这条判断建立在两份文件一致的基础上。
* `R7` 是 main 快照，**不是** v0.1.15；本章只在「版本边界」语境下引用它，不作为
  API 是否存在判据。

---

## 关于性能数字

**本章不出现任何性能数字。** 所有涉及"快慢"的表述都是定性的（例如"边界判断会挡住
向量化"），其依据是官方文档的原文评语，而非本机或第三方的测量数据。第 11、12、19 章
涉及实测时会各自给出测量口径。

---

## 待补资料

1. **PyPI 上 `0.1.15` 是否存在，未在本机核实。** 本章 3.2 节只给出「把版本号钉住」
   的方法（`pip install "tilelang==0.1.15"`）与「从 Release Page 取版本字符串为
   `0.1.15` 的 wheel」这条替代路径，没有断言 PyPI 上一定有该版本。建议后续版本补一条
   对 `https://pypi.org/project/tilelang/#history`（或对应 Release 页）的抓取，以确认
   可安装的具体版本清单。
2. **`pip install git+https://github.com/tile-ai/tilelang.git` 装的是哪个分支，
   未核实。** v0.1.15 的 `Installation.md:25-27` 给了这条命令，但没有写分支；本章据此
   提醒读者"它装的是那个分支的源码，不是你本地的 checkout"，但没给出该分支名。
3. **昇腾 wheel 的发布位置与版本列表未在 corpus 中。** 本章 3.4 节只说"下载现成
   wheel"，未给出具体下载地址（`tilelang-ascend/README.md:89-99` 也没写死 URL）。
   第 19 章若需要精确到文件，建议补抓昇腾侧的 Release 页。
4. **昇腾 950 的 `dav-3510` 与 A2/A3 的对应关系未核实。** 主干文档写"默认 `dav-3510`
   对应 Ascend 950"（S13），独立仓库文档写"在 A2/A3 上验证"（S15），两者未在同一份
   材料里被明确对应。本章只分别引用，未做合并结论。
