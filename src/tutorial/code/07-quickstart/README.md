# 第 07 章 代码目录

本章是全书第一个动手章，所以 `code/` 里的东西分两类：

* **需要真实加速器才能"跑"的**（01、03、05）——只能在本机上做语法核对与 API 核对，
  真机验证步骤写在各自 README 里；
* **在本机就能跑的纯 Python 工具**（02、04、06）——作者的 macOS 机器上已经跑过，
  用来把本章的若干结论变成可复现的检查。

## 示例一览

| 目录 | 内容 | 是否需要加速器 | 验证级别 |
| --- | --- | --- | --- |
| `01-first-kernel/` | 一维向量加法（`T.Kernel` + `T.Parallel`；以及显式 buffer + `T.copy` 两种写法） | 需要 NVIDIA GPU | 仅静态核对 |
| `02-inspect-source/` | 把「Python ↔ 生成 CUDA」对照关系与官方生成结果打印出来 | 不需要 | 在本机运行过 |
| `03-jit-and-cache/` | `@tilelang.jit` 的 lazy / eager 两种用法与编译缓存 | 需要 NVIDIA GPU | 仅静态核对 |
| `04-api-surface-check/` | 离线核对本章用到的 API 是否真的存在于锁定版本源码 | 不需要 | 在本机运行过 |
| `05-ascend-first-kernel/` | 昇腾侧同一个 `C = A + B`（对照） | 需要昇腾 NPU | 仅静态核对 |
| `06-env-check/` | 安装前的环境体检 | 不需要 | 在本机运行过 |

## 怎么最快地"看到东西"

如果手上暂时没有 GPU，先跑这三个（都只需要一台能跑 Python 的机器）：

```bash
cd 02-inspect-source && python inspect_source.py      # 看 Python ↔ CUDA 对照
cd ../04-api-surface-check && python check_api_surface.py   # 验证 API 确实存在
cd ../06-env-check && python check_env.py             # 体检本机环境
```

拿到 GPU 机器后，再从 `01-first-kernel/` 开始按 README 走。

## 验证级别的含义

* **仅静态核对**：作者本机是 macOS（Apple Silicon），**既没有 NVIDIA GPU，也没有昇腾
  NPU，也没有安装 tilelang**。这类示例只做到「`python3 -m py_compile` 通过」+「每个
  API 名/签名在 `_vendor/` 里核对过」，**没有运行过**。
* **在本机运行过**：这类示例是纯 Python，不 import tilelang，作者在本机跑过，输出
  可复现。

代码里不出现任何真实机器上的私有路径、token 或 IP。

## 与本章正文的对应关系

| 正文小节 | 代码 |
| --- | --- |
| 3.x 安装 | `06-env-check/` |
| 4.x 第一个 kernel | `01-first-kernel/` |
| 5.x 读懂生成的源码 | `02-inspect-source/` |
| 6.x JIT、缓存与复用 | `03-jit-and-cache/` |
| 7.x 昇腾侧的第一个 kernel | `05-ascend-first-kernel/` |
| 全文 API 出处核对 | `04-api-surface-check/` |
