# 示例 01：昇腾环境自检

## 目的

在装机过程中反复确认“卡在哪一层”：驱动 / CANN / torch_npu / TileLang。
它只探测“装没装、能不能 import、版本与路径”，**不判定环境是否达标**。

## 运行前提

* Python 3.9+；**不需要** NPU、不需要 torch、不需要 TileLang。

## 运行命令

```bash
python3 env_check.py
```

## 期望输出

在写作机（macOS, Python 3.11，无 NPU）上的**实际输出**形如：

```
[ok  ] os                              Darwin arm64 python=3.11.x
[MISS] binary:npu-smi                  not found on PATH
[MISS] binary:bisheng                  not found on PATH
[MISS] env:ASCEND_HOME_PATH            <unset>
[MISS] env:ASCEND_TOOLKIT_HOME         <unset>
[MISS] env:TL_ROOT                     <unset>
[MISS] env:ACL_OP_INIT_MODE            <unset>
[ok  ] import:torch                    version=... path=...
[MISS] import:torch_npu                ModuleNotFoundError: No module named 'torch_npu'
[MISS] import:tilelang                 ModuleNotFoundError: No module named 'tilelang'
...
结论： 这台机器还没有昇腾环境：先按 §3.2–§3.4 铺驱动/CANN/torch_npu/TileLang。
```

在真实的昇腾机器上，你应当看到 `binary:bisheng`、`env:ASCEND_HOME_PATH`、
`import:tilelang`、`torch.npu.is_available` 全部为 `ok`（或至少前几项为 `ok`）。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，输出即上面那种
“大量 MISS”的降级形态。**没有在昇腾机器上运行过。**

## 已知局限

* 不检查 CANN 版本号，也不检查 `torch_npu` 与 CANN 的配套关系——这两件事以厂商说明为准。
* `import:tilelang` 只能告诉你装没装，不能告诉你装的是 A 树还是 M 树；要区分请分别看
  `tilelang.language`（A 树有）与 `tilelang.ascend.language`（M 树有）两项。
