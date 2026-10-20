# 第 30 章 习题

说明：

* 难度：`★` 基础 / `★★` 需要综合本章内容 / `★★★` 动手或推理链较长。
* “需要硬件”：本章**没有任何一题需要 GPU 或 NPU**。所有题目都能在一台只装了
  Python 与 `git` 的机器上完成，因为判断依据全部是源码与资料快照。
* 所有源码引用都指向 `_vendor/tilelang-v0.1.15/`（对照时用 `_vendor/tilelang-v0.1.14/`），
  资料引用都指向 `../tilelang-corpus/`。
* 动手题请在本章目录下执行，路径相对本目录；`code/` 里的三个脚本可以直接跑。

---

## E1 概念题 ★ 不需要硬件

请写出本章的四种结论档位，并各给一句话定义：

1. v0.1.15 已具备；
2. 仅 main；
3. 仅未合入 PR/RFC；
4. 无法确认。

然后回答：**第 4 档为什么不是“兜底话术”？** 请举本章出现过的一个具体案例说明。

---

## E2 概念题 ★ 不需要硬件

本章给出的四步判定流程是：

1. 在 `_vendor/tilelang-v0.1.15/` 里搜符号；
2. 在 `_vendor/tilelang-v0.1.14/` 里搜同一个符号；
3. 在 corpus 的 main 快照里搜；
4. 都搜不到 → 只可能是未合入的 PR/RFC。

请分别回答：

1. 第 2 步**不能**回答什么问题？（提示：考虑 `reducer v2`）
2. 第 3 步命中之后，为什么仍然不能在代码里依赖它？
3. 第 4 步为什么必须注明“快照日期”，而不能直接写成“这个功能不存在”？

---

## E3 概念题 ★ 不需要硬件

运行 `python3 code/03-upstream-watch/upstream_watch.py --run`，观察 `_vendor/` 三个
checkout 的 `history` 一列。回答：

1. 每个 checkout 有几个 commit？为什么？
2. 因此在 `_vendor/tilelang-v0.1.15/` 里执行 `git diff v0.1.14..v0.1.15` 会怎样？
3. 想比较两个 tag 的内容，本章推荐的两条替代路径分别是什么？
4. “我本地 diff 不出来”这句话，为什么**不能**被理解成“两者没有差异”？

---

## E4 代码阅读题 ★★ 不需要硬件

阅读下面三条证据：

```bash
cat _vendor/tilelang-v0.1.15/.gitmodules
rg -n 'set\(USE_Z3' _vendor/tilelang-v0.1.15/CMakeLists.txt
rg -n 'z3_prover' _vendor/tilelang-v0.1.15/src/transform/verify_parallel_loop.cc
```

回答：

1. `3rdparty/tvm` 指向哪个仓库？这与 RFC #2196 的诉求是什么关系？
2. `USE_Z3` 的默认值是什么？RFC 的作者建议上游用什么默认值？为什么？
3. `verify_parallel_loop.cc:106` 那一行说明 Z3 prover 的所有权在谁手里？
   RFC 的下游分支主张把它搬到哪？
4. 综合以上，issue #2196 属于本章的哪一档？请写出你的判定依据。

---

## E5 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:1-22` 与
`_vendor/tilelang-v0.1.15/examples/conftest.py:39-45`，回答：

1. `cutedsl` 这个 `BackendModule` 复用了哪一条 pipeline？这说明了什么设计取向？
2. `CUTEDSL_KNOWN_FAILURES` 里列了几条测试？它们被标成什么标记？该标记的
   `strict` 参数是什么值，含义是什么？
3. 结合上面两点：issue #1454 属于哪一档？为什么“roadmap 上还列着几条”
   不能推出“这个后端还没实现”？

---

## E6 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:139-151`，回答：

1. 这段代码在什么条件下抛异常？异常消息给出的替代做法是什么？
2. 注释里提到的 `dav-950` 与 `quant_pre` / `fixpipe` 是什么关系？
3. 由这段代码可以推出 §4.5 中 PR #3328 的什么状态？为什么**不能**进一步推出
   “#3328 不可能实现”？

---

## E7 概念题 ★★ 不需要硬件

RFC #2897 的开头描述了一个最小错码例子：`BD=8`、`threads=128` 时，
`sum([1..8])` 会返回 `576` 而不是 `36`。

1. `576 / 36` 等于多少？这个倍数与 `threads`、`BD` 的关系是什么？
2. 这个例子里的 API 写法（`alloc_reducer(..., replication="all")` +
   `T.fill` + `acc[i] += ` + `T.finalize_reducer(acc)`）在本章里被称为什么写法？
3. 为什么**不能**用这个例子来判断 v0.1.15 里的 `T.reducer_update` 也会算错？
   （提示：读出 RFC 正文里被检查的 commit）

---

## E8 代码阅读题 ★★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/src/transform/verify_reducer_epoch.cc` 文件头部的
注释块（约 1–30 行），列出它强制执行的规则，至少写四条。
然后回答：

1. 这个 pass 在 `tilelang/cuda/pipeline.py` 的第几行被挂上？
2. 它和 `CanonicalizeLegacyReducer` 的先后顺序是什么？为什么是这个顺序？

---

## E9 概念题 ★★ 不需要硬件

关于 `T.atomic_add(..., memory_order=...)`：

1. RFC #2708 列出了几条“接受参数但丢掉”的路径？分别是什么？
2. 在 `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/atomic.h` 里，
   `AtomicAdd` / `AtomicAddRet` 与 `AtomicAddx2Scalar` / `AtomicAddx4Scalar`
   在处理 `memory_order` 上有什么差别？请给出这些符号所在的行号范围。
3. `tilelang/language/atomic.py` 的文档字符串如何描述 HIP 与 CuteDSL 的行为？
   请引用其中一句。

---

## E10 动手题 ★★★ 不需要硬件

`code/01-state-audit/state_audit.py` 里每条能力由若干“探针”定义。
请你**新增一条能力**，用于判定

> `T.launch_dependents`（CuTeDSL 的 PDL 支持）是否已在 v0.1.15 里存在。

要求：

1. 至少写两个探针（一个路径探针或一个文本探针）；
2. 在 `v0.1.14`、`v0.1.15`、`main` 三棵树上给出你**预期**的结果；
3. 运行脚本，让 `self-check: ok`；
4. 用 `rg` 手工复核一次，并写下你实际看到的结果与预期是否一致。

提示：本章 §3.3 已经给出这条能力的落点线索，但你需要自己把它翻译成探针。

---

## E11 动手题 ★★ 不需要硬件

用 `code/02-tree-diff/tree_diff.py` 比较 `v0.1.14` 与 `v0.1.15` 的 `examples/` 目录：

```bash
python3 code/02-tree-diff/tree_diff.py \
    --left  _vendor/tilelang-v0.1.14 \
    --right _vendor/tilelang-v0.1.15
```

（脚本的 `--left/--right` 接受任意两棵树；默认是 v0.1.15 与 main 快照。）

回答：

1. 只在 `right`（v0.1.15）里出现的 `examples/` 路径有哪些？
2. 只在 `left`（v0.1.14）里出现的 `examples/` 路径有哪些？
3. 结合本章 §4.1，解释为什么 `examples/ascend/` 会出现在第 1 问的答案里，
   而 `examples/ascend/flash_attention/example_gqa.py` 之类的内容属于哪一档能力。

---

## E12 动手题 ★★★ 不需要硬件

corpus 没有收录 PR #3321（host/device profiling 关联）的页面快照。
请你设计一份**一页以内的状态判定报告**，用于把这条 PR 的判定落到纸面上：

1. 需要先做哪些**反向验证**（在 v0.1.15 里找什么）？写出命令。
2. 报告里必须出现哪几项信息，才能让别人复核你的结论？
3. 如果反向验证通过（即 v0.1.15 里确实没有该能力），最终结论应该怎么写？
   请给出你的措辞，注意不要写成“这个功能不存在”。
4. 如果将来 corpus 补上了 #3321 的页面快照，你的报告还需要补哪一项？
