# 第 29 章 代码目录

本章的示例都围绕「**改**编译器」展开。与第 20 章不同，这里的代码是要被拷进
TileLang 源码树、编译、挂进 pipeline、然后被测试的。

**本机（macOS / Apple Silicon，无 NVIDIA GPU、无昇腾 NPU，未安装 tilelang/tvm）
没有任何一个示例被真正编译或运行过。** 每个示例的 README 都写清了自己的验证级别，
请照实引用，不要把「仅静态核对」当成「运行验证」。

## 验证级别总表

| 目录 | 作用 | 需要硬件 | 需要能编译 TileLang | 验证级别 |
| --- | --- | --- | --- | --- |
| `01-python-pass/` | 一个最小可用的 Python pass + 它的单元测试 | 否 | 否（纯 Python） | **仅静态核对**（`py_compile` 通过；API 名逐个在 v0.1.15 里核对过） |
| `02-cpp-pass-template/` | C++ pass 模板、注册、挂载、pytest 骨架 + 构建步骤 | 否 | **是**（要重装 tilelang） | **仅静态核对**（未编译；步骤写成可复现命令） |
| `03-verify-pass-template/` | 以 `verify_buffer_init` / `verify_parallel_loop` 为模板的 verify pass 骨架 + 模板抽取脚本 | 否 | 抽取脚本不需要 | **抽取脚本：CPU 逻辑验证**；C++ 骨架：**仅静态核对** |
| `04-regression-judge/` | 按 v0.1.15 的阈值/最小样本数规则判定一次性能对比是否算回归 | 否 | 否 | **CPU 逻辑验证**（已在本机运行） |

## 定位 `_vendor/`

所有脚本都按「脚本所在目录向上数 4 层」定位书根，再拼 `_vendor/tilelang-v0.1.15/`。
只要章节目录还在
`.workspace/research/tilelang-book/chapters/29-compiler-engineering/` 下，
脚本无需传参就能找到锁定版本的源码。

## 运行顺序建议

```bash
# 0) 先确认本机有没有编译环境（大多数机器上会没有任何输出）
command -v nvcc cmake ninja g++

# 1) 纯 Python 示例：只做语法检查（顶层 import tvm，所以没有 tvm 时跑不了）
python3 -m py_compile 01-python-pass/hoist_constant_broadcast.py
python3 -m py_compile 01-python-pass/test_hoist_constant_broadcast.py

# 2) 抽取 verify pass 的模板骨架（只用标准库，任何机器都能跑）
python3 03-verify-pass-template/extract_verify_skeleton.py

# 3) 回归判定器：跑一遍，对比它给出的结论与正文 6.3 节
python3 04-regression-judge/judge_regression.py --self-test

# 4) 只有在有编译环境的机器上，才去走 02-cpp-pass-template/README.md 的构建与测试
```

## 纪律

* 不修改 `_vendor/`，不修改 corpus。这里的脚本对源码只读。
* 把示例拷进 TileLang 源码树时注意：`_vendor/` 是只读的锁定版本，本章的 C++/Python
  片段是**给你抄进自己的 fork** 的，不是给 `_vendor/` 打补丁。
* `02-cpp-pass-template/` 的测试里有一个 `@tilelang.testing.requires_cuda` 的端到端
  用例，它的前提与命令写在那个目录的 README 里。
