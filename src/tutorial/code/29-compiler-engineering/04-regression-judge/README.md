# 04-regression-judge — 一次性能对比算不算回归

## 目的

本章 6.3 节说：**「跑出两个数字」不等于「知道有没有回归」**。仓库里真正的判定
逻辑在 `maint/scripts/test_perf_regression.py:24-32,243-257`，规则是
「至少 `MIN_COUNT` 条同时低于 `THRESHOLD` 才算回归」。

本脚本把那条规则单独抽出来，做成一个可以在**任何机器**（没有 GPU/NPU 也行）
验证的判定器：输入两个 JSON，输出每条的 speedup 与最终结论。

## 文件

| 文件 | 作用 |
| --- | --- |
| `judge_regression.py` | 判定器；`--self-test` 用三组构造数据演示规则 |

## 运行前提

* Python 3.10+ 与标准库。**不需要** GPU/NPU，**不需要** tilelang。

## 运行命令

```bash
python3 judge_regression.py --self-test
python3 judge_regression.py --old baseline_result.json --new current_result.json
python3 judge_regression.py --old baseline_result.json --new current_result.json \
    --fail-on-regression          # 判为回归时退出码 1，供 CI 用
```

`--old` / `--new` 的输入来自
`maint/scripts/run_current_regression.py`（由
`maint/scripts/run_perf_regression.sh` 编排，对比基线通常取 `origin/main`）。

## 预期输出（`--self-test`，本机已运行）

```text
场景 A：40 条里 1 条 0.85x
threshold=0.9  min_count=2  cases=40

**不判为回归**：1 条低于 0.9x，未达到 2 条的门槛。

场景 B：40 条里 2 条 0.88x
threshold=0.9  min_count=2  cases=40

**回归：2/40 条低于 0.9x**
  - case_07: 0.8800x
  - case_19: 0.8800x

场景 C：一侧缺 case
threshold=0.9  min_count=2  cases=1

| case | old | new | speedup |
| --- | --- | --- | --- |
| a | 1 | 0.5 | 2.0000x |

**不判为回归**：0 条低于 0.9x，未达到 2 条的门槛。
```

三个场景各回答一个问题：

* A：单条掉到 0.85x 是噪声，不判回归（这正是上游注释里 7 份假警报的形态）；
* B：两条同时掉，判回归（对应 6 份真回归的形态）；
* C：只在一边出现的 case 不参与比较——**「少了结果」和「结果变差」是两回事**，
  `compare_perf_regression.py` 对这两种情况分别有 `--fail-on-missing` 与
  `--fail-on-regression` 两个不同的开关。

## 在这台机器上达到的验证级别

**CPU 逻辑验证**：作者在本机（macOS / Apple Silicon，无 GPU/NPU）运行过
`--self-test`，上面的输出是真实输出。

但要注意边界：本目录**没有**验证过
`maint/scripts/run_perf_regression.sh` 本身能否跑通（它需要构建两个 venv、两个
wheel，并在真实加速器上跑 30+ 个 benchmark）。判定规则照抄自上游；整条流程属于
**待验证**。
