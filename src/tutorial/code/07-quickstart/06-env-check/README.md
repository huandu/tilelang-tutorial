# 06-env-check · 安装前的环境体检

## 目的

在真机上动手装之前，先把 `docs/get_started/Installation.md` 里那几条环境要求
（Python >= 3.10、glibc >= 2.28、CUDA 工具链）和昇腾侧的几条要求
（`ASCEND_HOME_PATH`、`bisheng`、`ld.lld`、`torch_npu`）逐条查一遍。

脚本只报告事实，不自动安装、不改任何文件。

## 运行前提

只有 Python 3.10+。**不需要 tilelang，不需要 GPU。**

## 运行命令

```bash
python check_env.py                 # 通用 + CUDA
python check_env.py --ascend        # 额外体检昇腾环境
```

## 预期输出

```text
平台：Linux-5.15.0-... / x86_64

== 通用环境（对照 docs/get_started/Installation.md:7-9） ==
  [ok  ] Python >= 3.10                 3.11.7  (必需)
  [ok  ] glibc >= 2.28                  2.35  (必需)

== CUDA 环境 ==
  [ok  ] nvcc 可用                      /usr/local/cuda/bin/nvcc  ...release 12.4...

== Python 依赖 ==
  [ok  ] import torch                   2.4.0+cu124
  [ok  ] torch.cuda.is_available()      True
  [FAIL] import tilelang                ModuleNotFoundError: No module named 'tilelang'
```

**正确性判据**：脚本退出码为 0（它只报告，不判定通过与否）。你要看的是有没有 `FAIL`
落在**必需**项上：装 tilelang 之前 `import tilelang` 报 FAIL 是正常的。

## 验证级别

**在本机运行过（纯 Python，只读环境）。** 脚本不 import tilelang，因此在本章作者的
macOS 机器上也能跑；`--ascend` 分支在非昇腾机器上会如实报缺。

## 在真实机器上的验证步骤

1. 装 tilelang **之前**跑一次，记下哪些是你环境已有的。
2. 按本章 3.2 / 3.3 / 3.4 节安装。
3. 再跑一次，确认 `import tilelang` 从 FAIL 变成 ok，并打印出版本号
   （形如 `0.1.15`，或带 `+cu...git<hash>` 一类的后缀）。
