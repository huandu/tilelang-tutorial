# 图 4：判定一次改动是否造成性能回归，以及定位到 pass 的流程（自绘）

这张图的每一步都对应一个**可执行的命令**或一个**已存在的文件**，不是流程图装饰。

```text
   ① 判定：一次对比算不算回归
   ─────────────────────────────────────────────────────────────────
   maint/scripts/run_perf_regression.sh
     ├─ 造两个 venv + 两个 wheel（current 与 origin/main 基线）
     ├─ maint/scripts/run_current_regression.py   →  *_result.json
     └─ maint/scripts/compare_perf_regression.py  →  md + png + json
   判定规则（maint/scripts/test_perf_regression.py:24-32,243-257）：
     speedup = old_latency / new_latency
     regressed ⟺ 至少有 MIN_COUNT(默认 2) 条 speedup < THRESHOLD(默认 0.90)
     ↑ 单条 0.044x 这种极端值也判「不是回归」，见 code/04-regression-judge/
                                    │
                       判为回归？ ───┴─── 否 → 记录数字，收工
                                    │ 是
                                    ▼
   ② 确认不是噪声：重跑
   ─────────────────────────────────────────────────────────────────
   REFRESH=1 bash maint/scripts/run_perf_regression.sh -k <算子/用例>
   重跑后仍低于阈值 → 不是抖动，继续往下。
                                    │
                                    ▼
   ③ 定位到「编译期变慢」还是「kernel 变慢」
   ─────────────────────────────────────────────────────────────────
   TILELANG_PASS_PROFILE=1 python3 <你的用例>.py
     └─ tilelang/tools/pass_timing.py：每个 pass 的 inclusive / self 用时
        （入口：tilelang/tools/pass_timing.py:334 create_pass_timing_tool）
        · 只有编译变慢 → 看它改了哪个 pass 的复杂度
        · 只有 kernel 变慢 → 编译产物变了，看 ④
                                    │
                                    ▼
   ④ 看「哪些 pass 改了 IR」
   ─────────────────────────────────────────────────────────────────
   TL_LOWER_TRACE=terminal python3 <你的用例>.py
     └─ tilelang/tools/lower_trace/：逐 pass 的 before/after IR diff
        （程序 API：tilelang/tools/lower_trace/__init__.py:49 lower_trace()）
        旧工具 TILELANG_PASS_DIFF 仍在，但官方文档已标注被 lower_trace 取代
        （docs/tools/pass_diff.md 顶部 admonition）。
                                    │
                                    ▼
   ⑤ 定位到具体 pass → 回到图 3 的四层测试补一条基线
```

## 三个容易搞混的点

* **“编译慢”和“跑得慢”是两件事。** `pass_timing` 量的是前者，
  benchmark 量的是后者。一个 pass 让编译变慢 3 倍但产物一样，kernel 性能不变。
* **`pass_timing` 是 inclusive + self 两个数。** self 时间才是这个 pass 自己花掉的；
  inclusive 包含它调用的子 pass。看到“某个外层 pass 很慢”先看 self。
* **回归判定需要样本量。** 上游那句注释说得很直白：单条掉到 0.044x 也**不算**
  回归，只有 ≥2 条同时掉才算。改动越小，越要按这个规则来，否则噪声会把你带偏。

来源：作者自绘。行号与文件名取自 `_vendor/tilelang-v0.1.15/`（2026-10-01 核对）；
`run_perf_regression.sh` 的环境变量与默认值见该脚本 `:15-32`。
