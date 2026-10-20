# 图 6：ACLGraph 的 capture 与 replay

**用途**：正文 §9.2。来源：A 树 `examples/aclgraph/README.md` 与
`docs/TileLang-Ascend Programming Guide.md:2564-2587`。

```text
  普通 eager：每次调用都走一遍下发路径

    op1 ──▶ launch ──▶ 执行
    op2 ──▶ launch ──▶ 执行
    op3 ──▶ launch ──▶ 执行
    （Host 侧 Python/框架/runtime 开销累加；小算子时盖过设备计算）


  ACLGraph：capture 一次，replay 多次

    ── capture（只记录，不执行）─────────────────────────────
       g = torch.npu.NPUGraph()
       with torch.npu.graph(g):
           q = rms_norm(q, eps)     # 记录
           q = apply_rope(q, sin, cos)
    ──────────────────────────────────────────────────────

    ── replay（一次调用重放整张图）──────────────────────────
       g.replay()   →  [ op1 → op2 ] 一次下发
       g.replay()   →  [ op1 → op2 ]
       ...
    ──────────────────────────────────────────────────────
```

## 边界

* ACLGraph **只降低下发开销（Host-bound）**，不做图编译、不做图优化。
* 确认生效：Host 日志里搜 `capture model`，`0`=关闭、`1`=开启。
* 需要 `TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_MEMORY_PLANNING` /
  `TL_ASCEND_AUTO_CV_SYNC` / `TL_ASCEND_AUTO_CV_COMBINE` 四个 A 树 pass config。
* 图里不要夹带 host 侧同步，否则 replay 的收益消失。
