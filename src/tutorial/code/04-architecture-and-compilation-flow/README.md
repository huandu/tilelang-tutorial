# 第 04 章代码目录

本章是关于「结构」的章，不需要 GPU/NPU 就能验证主要结论。所有示例都以
`_vendor/tilelang-v0.1.15/` 为唯一事实来源，**不 import tilelang**，因此在一台没有
编译过 TileLang、也没有加速器的机器上也能跑。

| 示例 | 作用 | 前置 | 运行命令 | 作者验证级别 |
| --- | --- | --- | --- | --- |
| `pass-order/` | 从两条 pipeline 源码里提取**真实 pass 顺序**与条件守卫 | Python 3.8+ | `python3 pass-order/dump_pass_order.py` | **在本机运行过**（macOS arm64, Python 3.11） |
| `repo-map/` | 统计仓库模块清单，生成正文里的目录/数量表 | Python 3.8+ | `python3 repo-map/inventory.py` | **在本机运行过** |
| `compile-path-check/` | 逐跳校验「装饰器 → 可调用 kernel」路径上的 30 个文件:行号 | Python 3.8+ | `python3 compile-path-check/check_compile_path.py` | **在本机运行过** |
| `quickstart-gemm/` | 用官方 GEMM+ReLU 走一遍完整编译流程 | CUDA GPU + 已安装的 TileLang；`debug_root_path` 需要可写目录 | `python3 quickstart-gemm/quickstart_gemm.py --out /tmp/tl-debug` | **仅静态核对**（`python3 -m py_compile` 通过 + API 名与 v0.1.15 逐个比对）；本机无 GPU，未运行 |

## 一键跑完三个离线示例

```bash
cd chapters/04-architecture-and-compilation-flow/code
python3 pass-order/dump_pass_order.py | head -20
python3 repo-map/inventory.py
python3 compile-path-check/check_compile_path.py
```

三个脚本都以 `_vendor/tilelang-v0.1.15/` 为默认输入，可以用 `--vendor <path>`
指向别的 checkout。注意：指向 `main` 时结果会不一样，正文第 5 节说明了这一点。

## 关于「验证级别」的诚实说明

* `pass-order`、`repo-map`、`compile-path-check` 只读源码、不算数、不联网，属于
  **可复现的静态证据**。它们证明的是「源码里确实这么写」，不是「kernel 跑起来是对的」。
* `quickstart-gemm` 依赖真实 CUDA GPU。本机（macOS、无 NVIDIA GPU、无昇腾 NPU）
  只做到了 `py_compile` 与 API 名称/签名比对，**没有跑过**。`code/quickstart-gemm/README.md`
  写了在有卡机器上应该看到什么。
