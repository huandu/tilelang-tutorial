# 02-cpp-pass-template — C++ pass 的完整模板与构建步骤

## 目的

演示本章 3.3 节：在 C++ 里写一个新 pass 需要改的**全部**位置，以及怎么把它编译
出来、跑起来。模板本身是一个“报告型”pass：遍历 IR，数出所有
`For(kind=kParallel)`，打成一条警告，然后原样返回 func。

它是幂等的，所以可以先用它把“编辑 → 重编 → 挂 pipeline → 跑测试”这条链路走通，
再往里面加真正改 IR 的算法。

## 文件

| 文件 | 放到哪里 | 作用 |
| --- | --- | --- |
| `count_parallel_loops.cc` | `src/transform/count_parallel_loops.cc` | pass 本体（visitor + pass func + 注册宏） |
| `python_wrapper_snippet.py` | 参考用 | 三处改动：`transform/__init__.py` 包装、pipeline 调用、可选 pass config |
| `test_count_parallel_loops.py` | `testing/python/transform/` | pytest 骨架（含一个 `requires_cuda` 端到端用例） |

## 运行前提

* 一份**可写**的 TileLang 源码 checkout（不是 `_vendor/`），已按
  `CONTRIBUTING.md:64-93` 配好开发环境：`uv venv` / `requirements-dev.txt` /
  `pre-commit install`。
* 能编译 C++ 扩展：`cmake`、`ninja`（或 make）、一个 C++17 编译器。
* 只有端到端那个用例需要 NVIDIA GPU + CUDA 工具链；前两个断言不需要。
* 昇腾侧的同学不需要 CUDA，但需要 CANN 与 `torch_npu`（见 `04` 的昇腾说明）。

## 构建与验证步骤

```bash
# --- 0) 准备一棵可写的源码树 ---
git clone --recurse-submodules git@github.com:<your-name>/tilelang.git
cd tilelang
uv venv --seed .venv && source .venv/bin/activate
uv pip install --requirements requirements-dev.txt
pre-commit install --install-hooks

# --- 1) 落地三处改动 ---
cp <THIS_DIR>/count_parallel_loops.cc src/transform/count_parallel_loops.cc
#   然后按 python_wrapper_snippet.py 的三段注释，分别改
#   tilelang/transform/__init__.py、tilelang/cuda/pipeline.py、
#   （可选）tilelang/transform/pass_config.py
cp <THIS_DIR>/test_count_parallel_loops.py testing/python/transform/

# --- 2) 编译扩展 ---
python3 -m pip install --no-build-isolation --verbose --editable .

# --- 3) 冒烟：确认新 pass 能被找到（不需要 GPU） ---
python3 -c "import tilelang; print(tilelang.transform.CountParallelLoops())"

# --- 4) 跑单元测试（前两条不需要 GPU；整份文件需要 CUDA） ---
python3 -m pytest testing/python/transform/test_count_parallel_loops.py -q

# --- 5) 格式与风格 ---
bash format.sh --files src/transform/count_parallel_loops.cc
python3 maint/scripts/audit_cpp_api_style.py     # 建议性的 API/头文件审计
```

## 预期输出

* 第 3 步：打印一个 `tvm.transform.Pass` 对象（形如
  `<tvm.transform.Pass object at 0x...>`）。这一步失败通常只有两种原因：
  注册名与 Python 包装不一致（`AttributeError: ... has no attribute
  'CountParallelLoops'`），或者新 `.cc` 没进构建（CMake 用
  `file(GLOB ...)`，见 `CMakeLists.txt:374-389`，**新增文件后要重新 configure**）。
* 第 4 步：`3 passed`（无 CUDA 时 2 passed + 1 skipped）。
* 若报错文本里出现 `Parallel loops: N parallel loop(s)`，说明 pass 真跑起来了。

## 在这台机器上达到的验证级别

**仅静态核对。** 作者本机没有 CUDA、没有 CANN，也没有编译过 TileLang
（`command -v nvcc` 无输出；`import tilelang` 直接 `ModuleNotFoundError`）。
本目录的结论只有一类证据：**逐行对照锁定版本的现有 pass**。

具体对照关系（都能在 `_vendor/tilelang-v0.1.15/` 里现场核对）：

| 模板里的部分 | 对照的上游代码 |
| --- | --- |
| `struct ... : public StmtExprVisitor` + `VisitStmt_(const ForNode*)` | `src/transform/verify_parallel_loop.cc:28-56` |
| `SpanHintSuffix(...)` + `#include "span_utils.h"` | `src/transform/verify_buffer_init.cc:22,383` |
| `CreatePrimFuncPass(pass_func, 0, "tl.X", {})` | `src/transform/verify_buffer_init.cc:416`、`src/transform/thread_storage_sync.cc:2104` |
| `TVM_FFI_STATIC_INIT_BLOCK` + `refl::GlobalDef().def("tl.transform.X", X)` | `src/transform/verify_buffer_init.cc:419-422`、`src/ascend/transform/insert_nd2nz.cc:488-490` |
| Python 薄包装 | `tilelang/transform/__init__.py:86-94` |
| pytest 的 `capfd` 断言 | `testing/python/transform/test_tilelang_transform_verify_parallel_loop.py:23-37` |

**“能编译”这件事本机没有验证过**，请把它当作待验证项：在按上面步骤跑通、
看到第 3/4 步的真实输出之前，不要引用任何“已编译通过”的说法。
