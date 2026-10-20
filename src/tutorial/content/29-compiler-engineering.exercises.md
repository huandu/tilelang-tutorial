# 第 29 章 习题

难度：`★` 入门 / `★★` 需要读一点源码 / `★★★` 需要动手或综合。
「需要硬件」一栏标明是否必须在有 GPU/NPU 的机器上做。本章大部分习题只需要
**读源码**与**静态推理**；只有 E5、E9 的完整版需要能跑 TileLang 的环境。

所有源码引用都可在 `_vendor/tilelang-v0.1.15/`（必要时
`_vendor/tilelang-ascend-v0.1.2.000/`）里现场核对。

---

### E1 概念题 ★

下面五个 pass，各自属于 C++ 侧还是 Python 侧？给出你的**判断依据**（不要只给答案），
并说明如果把它们互相搬家会失去什么：

1. `HoistBroadcastValues`
2. `LayoutInference`
3. `AddWrapperForSingleBufStore`
4. `VFChecker`
5. `z3_schedule_python`（注意：它出现在
   `tilelang/ascend/transform/z3_scheduler.py` 里）

*需要硬件：否*

---

### E2 概念题 ★★

某人把一个 C++ pass 的注册名从 `"tl.transform.Foo"` 改成了
`"tl.transform.Bar"`，同时把 Python 薄包装写成
`def Bar(): return _ffi_api.Bar()`，但忘了改
`CreatePrimFuncPass(pass_func, 0, "tl.Foo", {})`。

请回答：

1. 编译会报错吗？运行时的哪一步会报错？
2. 有哪些地方表现**正常**？为什么这件事比「直接报 AttributeError」更麻烦？
3. 在本章 2.3 节列的「三处一致」里，这一处对应的是哪一处，为什么它偏偏最不容易
   被发现？
4. 昇腾适配器仓库（`_vendor/tilelang-ascend-v0.1.2.000/`）的
   `tilelang/transform/_ffi_api.py` 与 v0.1.15 的写法有什么不同？如果直接把适配器
   里的一段 pass 代码拷进 v0.1.15，C++ 注册部分需要改什么？

*需要硬件：否*

---

### E3 代码阅读题 ★★

读 `_vendor/tilelang-v0.1.15/src/transform/verify_buffer_init.cc:315-321`：

```cpp
void VisitStmt_(const BufferStoreNode *op) final {
  const Object *saved = active_store_;
  active_store_ = op;
  StmtExprVisitor::VisitStmt_(op);
  active_store_ = saved;
  written_.insert(op->buffer->data);
}
```

请回答：

1. 为什么 `written_.insert(...)` 在 `VisitStmt_` **之后**而不是之前？这个顺序
   保证了哪一条语义？
2. `active_store_` 的作用是什么？结合 `VisitExpr_(const BufferLoadNode *op)`
   （`:323-328`）说明它解决了哪个具体问题。
3. 如果把这个函数的顺序反过来（先 insert 再 visit），
   `test_reports_a_store_that_reads_its_own_destination`
   （`testing/python/transform/test_tilelang_transform_verify_buffer_init.py:183-201`）
   会变成通过还是失败？
4. 文件头注释（`:11-15`）说这个分析「deliberately imprecise in one direction」，
   指的是哪个方向？为什么这个取舍能让检查默认打开？

*需要硬件：否*

---

### E4 代码阅读题 ★★

读 `_vendor/tilelang-v0.1.15/tilelang/transform/add_bufstore_wrapper.py:104-159`
（`pre_visit` / `post_visit` / `ir_transform` 三段）。

请回答：

1. 为什么 `tile_operation_depth` 必须在 `post_visit` 里 `-= 1`，不能只在
   `pre_visit` 里 `+= 1`？
2. `thread_binding_vars` 是干什么用的？为什么命中它就不做改写？
3. 这个 pass 的最后一行是
   `return prim_func_pass(pass_fn, opt_level=0, name="tl.AddWrapperForSingleBufStore")`，
   而 `hoist_broadcast_values.py:99` 只写了
   `return prim_func_pass(pass_fn, opt_level=0)`。这个差别会在**哪个工具的输出里**
   被看到？请说出工具名与它会显示什么。
4. 为什么这个 pass 必须挂在几乎所有 pipeline 的**最前面**
   （`cuda/pipeline.py:79`、`ascend/pipeline.py:58`）？

*需要硬件：否*

---

### E5 动手题 ★★

仿照 `code/01-python-pass/hoist_constant_broadcast.py`，写一个 Python pass
`CountSerialLoops()`，统计每个 `PrimFunc` 里 `ForKind.SERIAL` 的循环数量，并在
`name="tl.CountSerialLoops"` 下注册（Python 侧直接 export 到
`tilelang.transform`，不需要 FFI）。

要求：

1. 先用 `python3 -m py_compile` 过一遍语法；
2. 写至少两条测试：一条命中（有 serial 循环）、一条反向（只有 parallel 循环时
   不报）；
3. 在注释里用一行说明：为什么这个 pass 写在 Python 侧是合理的。

*需要硬件：否（静态检查）/ 是（若要用 pytest 真正运行）*

---

### E6 动手题 ★★★

`code/03-verify-pass-template/verify_local_alloc_limit.cc` 检查的是
「on-chip 分配元素数超限」。请把它改造成检查**另一件事**：
**「同一块 `local.fragment` 被写入之后、在被任何 pass 读之前，必须至少被写过两次」**
（即：单写未读的 fragment 是可疑的）。

要求：

1. 先不要写代码，先写出**判定规则的一句话描述**与**精度边界**（什么情况你会故意
   不报）；
2. 列出至少 5 条边界反向用例，说明每一条为什么必须存在（可参考本章 5.5 节的表）；
3. 再写出 C++ 骨架的关键部分（visitor 重载签名 + `Report` 结构 +
   `EmitReport` 的措辞）。不需要完整可编译。

*需要硬件：否*

---

### E7 概念题 ★★

你写了一个新 pass，它要读 `T.copy` 写入 shared buffer 的 `src`/`dst` region，
用来推断是否存在冗余拷贝。

请回答：

1. 它必须放在哪个 pass 之前？为什么？
2. 如果它同时需要 layout 信息（例如需要知道 `shared` buffer 的 swizzle），
   这个顺序约束会不会冲突？冲突时你会怎么办？
3. 该 pass 需要被挂到 CUDA 与 Ascend 两条 pipeline 上，还是只挂一条？说出判断
   依据，并列出 v0.1.15 里所有后端 pipeline 文件。
4. 用一句话说明：为什么 TileLang 用「命令式 pipeline + 硬编码顺序」，而不是
   像 LLVM 那样做 pass 依赖解析？请从**可读性**与**可调试性**两个角度各给一条
   理由（提示：想想 `TL_LOWER_TRACE` 的报告是按什么顺序排的）。

*需要硬件：否*

---

### E8 概念题 ★★

一次性能对比的结果是：40 个 case 里，`case_A` 的 speedup 是 0.85x，其余 39 个都是
1.00x。

请回答：

1. 按 v0.1.15 的判定规则（`maint/scripts/test_perf_regression.py:24-32,243-257`），
   这算不算回归？给出判定过程。
2. 把 `case_A` 之外再加 `case_B`（0.88x），结论变了吗？
3. 上游那段注释用什么数据支持了这个门槛？为什么它特意提到「最低 0.044x」这个
   极端值？
4. 如果这次对比里有一个 case 只在当前版本出现、基线里没有，脚本会把它算进
   「回归」吗？对应的开关叫什么？
5. 运行 `python3 code/04-regression-judge/judge_regression.py --self-test`，
   把你看到的输出与你的第 1、2 问答案对照。

*需要硬件：否*

---

### E9 动手题 ★★★

场景：你把某个 pass 加进了 pipeline，编译出来的 kernel 数值正确，但 E2E 延迟从
1.00 ms 掉到 1.14 ms；同一批 40 个 case 里有 6 个都掉了 10% 以上。

请：

1. 写出**判定这一步**要跑的命令（含环境变量），并说明为什么不能只看你手上那一个
   数字；
2. 写出**区分「编译变慢」与「产物变慢」**的命令；
3. 假设 `TILELANG_PASS_PROFILE=1` 显示某个外层 pass 的 inclusive 时间涨了 3 倍，
   但 self 时间没变，你的下一步是什么？
4. 假设 `TL_LOWER_TRACE=terminal` 显示你的 pass 之后 `VectorizeLoop` 的 diff
   也变大了，这说明了什么？你会怎么继续缩小范围？
5. 假如最终发现是「你的 pass 把循环切小了，导致向量化掉到 4 字节」，请说明你会用
   四层测试里的哪一层来把这个回归**固定住**，并给出这条测试的形态（一句话即可）。

*需要硬件：是（要真正跑 benchmark）*

---

### E10 评审题 ★★

你收到一个 PR：它新增了一个 C++ pass `src/transform/verify_no_warp_divergence.cc`，
只挂进了 `tilelang/cuda/pipeline.py`，带了一个 IR 基线测试和一个 `requires_cuda`
的端到端用例，并且在 `src/op/builtin.h` 里加了一个新 config key。

请列出至少 6 条你会提出的评审问题，并说明每条属于本章 8.4 节的哪一类关注点。
至少覆盖：后端覆盖、位置与顺序、开关的四处一致性、测试的边界、命名与风格、
以及「这个检查应该默认开还是默认关」。

*需要硬件：否*

---

### E11 昇腾题 ★★

你要为昇腾加一条新的检查：**「`T.copy` 的 GM→L1 搬运必须显式处理 OOB 尾部」**。

请回答：

1. 这条检查应该放在主干的 `src/ascend/transform/` 还是适配器仓库的
   `src/transform/`？给出判断依据（提示：先看主干里有没有做类似事情的 pass）。
2. 你看中了 `AscendInsertOOBPadding` 的注释（
   `tilelang/ascend/transform/__init__.py:154-163`）里关于「必须跑在
   `InsertNd2Nz` 之后、`AutoSchedule` 之前」的说明。请解释这两条顺序约束各自的
   原因。
3. 昇腾侧的单测入口脚本是哪一个？它跑哪两个目录？它与 CUDA 侧入口的两点行为差异
   是什么？
4. `_vendor/tilelang-ascend-v0.1.2.000/AGENTS.md:59-75` 的四条通用原则中，哪一条
   与本章「所有结论必须标验证级别」是同一件事？再指出 `:79-113` 里哪一条会直接
   影响你**怎么查 API**。

*需要硬件：否*
