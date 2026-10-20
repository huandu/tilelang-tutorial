# 第 30 章 习题答案

与 `exercises.md` 一一对应。每题给出**解题思路**、**参考答案**与**常见错误**。
所有源码引用都指向 `_vendor/tilelang-v0.1.15/`（对照用 `_vendor/tilelang-v0.1.14/`），
资料引用都指向 `../tilelang-corpus/`。

本章写作环境没有 GPU 与 NPU，因此答案里不出现任何运行结果；
凡是涉及「在真机上会怎样」的问题，答案都只给判断方法。

---

## E1 概念题 ★

**解题思路**：回到 §2.1 的表格，逐档写定义；第 4 档的关键是「矛盾」二字。

**参考答案**：

1. **v0.1.15 已具备**——符号能在 `_vendor/tilelang-v0.1.15/` 的 `tilelang/` 或
   `src/` 里搜到，可以在代码里依赖。
2. **仅 main**——不在 v0.1.15，但在 corpus 的 main 快照
   （`02-tilelang-core-repo/raw/tilelang`，commit `994b44e`）里能搜到；
   下一个 release 会带上，现在还依赖不了。
3. **仅未合入 PR/RFC**——两棵树里都搜不到，只在 PR/RFC 页面（或合成文档的转述）里
   出现；只能当方向。
4. **无法确认**——资料未覆盖，或页面状态与代码快照互相矛盾，不能下断定。

第 4 档不是兜底话术，因为前 3 档都要求「有证据支撑一个确定的结论」。
本章的实例是 **PR #3310（PTO 后端）**：它的页面快照（2026-09-30）显示被标记为
draft，而同一 corpus 的 main 源码快照里有
`994b44e [Ascend] Add PTO backend for Ascend950 (#3310)`。两个快照指向同一天的不同
时刻，谁也无法只凭它们断定「今天合入了没有」。这时唯一诚实的写法是：
以代码为准陈述「v0.1.15 里没有、main 快照里有」，并把 PR 的评审状态标为无法确认。

**常见错误**：把「无法确认」当成「未合入」的同义词，直接写成「#3310 没有合入」。
这会与 main 快照里的合并提交冲突。

---

## E2 概念题 ★

**解题思路**：逐条想「这一步的输出能排除什么、不能排除什么」。

**参考答案**：

1. 第 2 步**不能**回答「这个能力是否曾经在更早的版本里存在过」，也**不能**回答
   「它的行为在 v0.1.14 与 v0.1.15 之间有没有变」。它只能回答「是不是这两个版本都有」。
   以 `reducer v2` 为例：`T.reducer_update` 在 v0.1.14 与 v0.1.15 里**都**存在
   （`tilelang/language/reduce_op.py`），所以它是「v0.1.14 之前就有的能力」，
   把它写成「v0.1.15 新增」是错的。两个版本的文件并非完全相同——
   `reduce_op.py` 与 `reducer_plan_materialize.cc` 在 v0.1.15 里有改动——
   所以第 2 步只能回答「有无」，回答不了「同不同」。
2. 因为 main 快照是**那一天**的主干，而 release tag 是**另一个 commit**；
   主干上的东西可能被回滚、可能被后续 PR 重写，也可能在下一个 release 之前
   被判定为不可用。依赖它 = 依赖一个尚未发布的契约。
3. 因为「搜不到」这个事实是相对于**某两棵树**而言的，而不是相对于「今天的上游」。
   本章 §4.5 的四条 PR 就属于这种情况：corpus 只收录了转述，没有页面快照，
   所以只能说「未在本章的 v0.1.15 / main 快照中找到」。

**常见错误**：把第 4 步写成「不存在」。四个字之差，把「我没找到」说成了「它没有」。

---

## E3 概念题 ★

**解题思路**：先跑脚本拿到事实，再回答「为什么」。

**参考答案**：

1. 每个 checkout **只有 1 个 commit**。因为它们是按 tag 做的
   `git clone --depth 1 --branch <tag>`（见 [VERSIONS.md](../../VERSIONS.md) 的说明
   与 `_vendor/` 的实际状态），只拉取了 tag 指向的那一个 commit，没有历史。
2. 会失败：两个 tag 之间没有共同祖先，git 无法计算差异
   （典型报错是 `fatal: bad revision` 或 `Invalid symmetric difference expression`）。
3. 两条替代路径：① 建一个完整 clone 再 `git diff`（见 §8.2）；
   ② 做**文件级比较**，即 `code/02-tree-diff/tree_diff.py`
   （可以用 `--left/--right` 指定任意两棵树）。
4. 因为「diff 不出来」是**工具能力的限制**，不是**对象状态的描述**。
   把工具限制当成事实结论，是本章 §2.4 说的「误判」之一。
   正确的做法是先 `git rev-list --count HEAD` 确认自己手里是快照还是完整历史。

**常见错误**：在 `_vendor/` 里反复尝试 `git diff v0.1.14 v0.1.15`，然后得出
「两个版本差不多」的结论。

---

## E4 代码阅读题 ★★

**解题思路**：三条命令分别对应「依赖指向」「构建开关」「运行时所有权」。

**参考答案**：

1. `.gitmodules` 里写着
   `3rdparty/tvm -> https://github.com/TileLang/tvm`，即**仍然是 TileLang 自己的
   TVM 分叉**。RFC #2196 的诉求正是把它换成 `apache/tvm` + `apache/tvm-ffi`，
   因此这条证据说明**迁移没有发生**。
2. `USE_Z3` 的默认值是 `ON`（`CMakeLists.txt:401`：
   `set(USE_Z3 ON CACHE STRING ...)`）。RFC 作者的建议相反：上游应当
   **默认 OFF**，并且让「没有 Z3」成为一条真正可编译的路径，理由是
   「avoid adding a mandatory solver dependency to normal TileLang builds」。
3. `verify_parallel_loop.cc:106` 写的是 `analyzer.z3_prover.GetModel(race_free)`，
   说明 Z3 prover 是挂在 **TVM 的 `arith::Analyzer`** 上的，所有权在 TVM core。
   RFC 的下游分支主张把它搬到 **TileLang 自己的 `src/transform/vendored/`**，
   让 apache/tvm 保持未打补丁的状态，同时给 `tvm::arith::Z3Prover(analyzer)`
   留一个兼容 shim。
4. **仅未合入（C 档）**。依据：依赖仍指向分叉；`USE_Z3` 仍默认 ON 且 prover 属于
   TVM core；RFC 建议的 `src/transform/vendored/` 目录在两棵树里都不存在。

**常见错误**：看到 RFC 里写着「we have a downstream branch running on
apache/tvm」就以为已经迁移了。那句话说的是**提问者自己的分支**，不是 TileLang 主干。

---

## E5 代码阅读题 ★★

**解题思路**：一个看「它怎么接进来」，一个看「它被接受的程度」。

**参考答案**：

1. 它复用 `pipeline.CUDA_PIPELINE`（`cutedsl_backend.py` 的
   `pipelines={"cuda": pipeline.CUDA_PIPELINE}`）。说明 TileLang 把 CuTe DSL 当成
   **codegen target**：从 TIR 到 `LowerTileOp` 的整条 pass 链不变，
   只有 device codegen 与 execution backend 不同。这也解释了为什么
   `src/cuda/codegen/codegen_cutedsl.cc` 与 `codegen_cuda.cc` 并排放在同一个目录。
2. 列了 **2 条**（`minference/test_vs_sparse_attn.py::test_vs_sparse_attn` 与
   `deepseek_v4/test_tilelang_example_deepseek_v4.py::test_example_act_quant`），
   被标成 `pytest.mark.xfail`，且 `strict=False`——意思是**预期失败，但如果它
   意外通过也不报错**（作者的注释是「so unexpected passes are reported」）。
3. issue #1454 属于 **A 档（v0.1.15 已具备，且 v0.1.14 起）**。
   roadmap 条目只是路线图：其中 Initial Commit、GEMM V2、Blackwell Support
   等早已完成；「还列着几条」既不代表「未开始」，也不代表「未完成」。
   后端的成熟度要用另外的证据衡量——比如依赖门禁
   （`tilelang/jit/adapter/cutedsl/checks.py` 要求 `nvidia-cutlass-dsl >= 4.7.0`）
   与这张 `xfail` 清单。

**常见错误**：把 `xfail(strict=False)` 读成「这些测试被跳过」。
它是**会运行、预期失败**，且允许意外通过。

---

## E6 代码阅读题 ★★

**解题思路**：先读条件与异常消息，再读注释里的硬件理由。

**参考答案**：

1. 当 **src 的 dtype 与 dst 的 dtype 不同，且 src buffer 的作用域是
   `shared.l0c`** 时抛 `ValueError`。消息给的两条替代做法是：用一次单独的
   copy，或者**在 VF block 里做 cast**。
2. 注释的意思是：dav-950 的 fixpipe（`cc → ub`，即 L0C → UB）双目标搬运
   **不支持 `quant_pre`（类型转换）**；如果 src/dst dtype 不同，
   硬件可能**静默产生错误结果**，所以 Python 层直接拒绝，而不是让它错下去。
3. 可以推出：**v0.1.15 里没有这个能力**（它被显式拒绝），因此 #3328 属于
   C 档。但不能推出「#3328 不可能实现」——恰恰相反，这段拒绝逻辑说明存在一个
   **具体的技术约束**（硬件不支持），而 PR 要做的正是绕过或规避它
   （例如改用别的搬运路径、或用 VF 里的显式 cast 配合 dual_copy）。

**常见错误**：把「当前拒绝」理解为「设计上反对」。这里是硬件约束 + 快速失败，
不是设计立场。

---

## E7 概念题 ★★

**解题思路**：先算倍数，再对照 RFC 与本章 §5.1 的两处提醒。

**参考答案**：

1. `576 / 36 = 16`。倍数恰好是 `threads / BD = 128 / 8 = 16`：
   每个逻辑 `x[j]` 被 16 个 replica 线程各加了一次，所以结果被放大了 16 倍。
2. 被称为 **legacy（v1）写法**：`alloc_reducer(..., replication="all")` 选择的是
   基于 fragment 的旧实现，配套 `T.clear`/`T.fill` 与原地 `finalize_reducer(acc)`。
3. 因为 RFC 正文明确写了它检查的是 `main` 的
   commit `fef9f760920fa5e81596e08696ee3c6e8e1c18ce`，描述的是**当时那个 commit**
   的行为。而 v2 API（`reducer_init` / `reducer_update` /
   `finalize_reducer(acc, dst)`）的语义是「每个逻辑贡献只计一次」，
   与 legacy 的 replica 语义完全不同——两者共享 `alloc_reducer` 这个名字，
   但走的是两条实现路径（`replication` 为 `None` 走 v2，非 `None` 走 legacy）。
   用 legacy 的错码例子去评价 v2，属于张冠李戴。

**常见错误**：认为「RFC 开着 = 问题还在」。RFC 是设计讨论，
它的 Phase 1–3 已经落地；真正还开着的部分是稀疏 partial layout 与废弃 legacy 写法。

---

## E8 代码阅读题 ★★★

**解题思路**：这段注释本身就是规格说明，逐条摘出来即可；顺序问题看 pipeline。

**参考答案**：
规则（任选四条即可，下面是注释里的条目）：

1. 每个 `local.reducer` 分配**恰好一次** `reducer_init`、零到多次
   `reducer_update`、**恰好一次** `finalize`（静态上一个分配对应一个 epoch 站点）；
2. `finalize` 必须在 `init` 所在的控制流作用域内（同一串嵌套的串行循环与条件分支），
   或在其条件细化中；`init`/`finalize` 都不得出现在 `T.Parallel` 循环内部；
3. `update` 只能出现在 `init` 与 `finalize` 之间，且**只能在 `T.Parallel` 循环内**；
4. reducer 在三个一等 op 之外是**不透明的**：普通 load/store、`fill`/`clear`/`copy`、
   `access_ptr` 与别名一律拒绝；
5. `finalize` 的目标必须是普通 buffer，不能是某个 reducer；
6. 声明的 seed 必须与 reducer 的 dtype 一致。

顺序与位置：

1. 它们在 `tilelang/cuda/pipeline.py:91-92`
   （`CanonicalizeLegacyReducer` 在 `:91`，`VerifyReducerEpoch` 在 `:92`）；
   昇腾 pipeline 在 `tilelang/ascend/pipeline.py:74-75`，顺序相同。
2. `CanonicalizeLegacyReducer` 先、`VerifyReducerEpoch` 后。因为前者的职责是把
   legacy 语法**改写成 v2 形式**，后者只对 v2 形式做校验。
   先规范化的好处是：legacy 代码一旦被转换，就与原生 v2 代码接受同一套校验，
   **不需要为两种语法各维护一套规则**；这也是 `canonicalize_legacy_reducer.cc`
   头部注释所说的「It is a deprecation shim」。

**常见错误**：以为 `VerifyReducerEpoch` 会给 legacy 代码开绿灯。它只认 v2；
legacy 必须先被规范化——而且规范化用的是**严格白名单**，
任何不认识的写法都会在编译期报错，而不是被静默接受。

---

## E9 概念题 ★★

**解题思路**：把 RFC 的三条路径与头文件里的两族实现对上号。

**参考答案**：

1. 三条：① 标量 atomic add 的 pre-Hopper 编译路径；② 标量非 fp16/bf16 add 的
   Hopper 路径（直接调 `atomicAdd`）；③ float32 x2/x4 add（含 `return_prev`）
   的 pre-SM90 回退（拆成标量 `atomicAdd`）。
2. `AtomicAdd`（`:472` 与 `:508`）与 `AtomicAddRet`（`:517`）都**接收**
   `memory_order` 参数并在实现里使用；而
   `tl_atomic_detail::AtomicAddx2Scalar`（`:310`）、`AtomicAddx4Scalar`（`:316`）、
   `AtomicAddx2ScalarRet`（`:323`）、`AtomicAddx4ScalarRet`（`:331`）的签名里
   **根本没有这个参数**，内部逐个调用标量 `atomicAdd(ref+i, val)`。
   也就是说，走到这条回退路径时，调用方请求的内存序被丢掉了。
3. 文档字符串写明 HIP 与 CuteDSL 会忽略它，例如 `atomic_add` 的说明里有一句：
   「HIP and CuteDSL codegen currently ignore it.」
   （同一句话也出现在 `atomic_max` 与 `atomic_min` 的说明里。）

**常见错误**：只读 `tilelang/language/atomic.py` 的签名（那里确实接受
`memory_order`）就下结论。参数被接受 ≠ 参数被实现；要看 C++/CUDA 模板层的两族实现。

---

## E10 动手题 ★★★

**解题思路**：把一个「能力」翻译成若干「探针」（路径存在 / 文本出现），
并给出你在三棵树上的预期，让脚本替你核对。

**参考答案（示例实现）**：

在 `CAPABILITIES` 列表末尾加一条：

```python
Capability(
    "cutedsl-pdl",
    "CuTeDSL 的 PDL 支持（launch_dependents / griddepcontrol）",
    [
        Probe("text", "launch_dependents"),
        Probe("text", "griddepcontrol"),
    ],
    expect={TAG_PREVIOUS: True, TAG_RELEASE: True},
),
```

预期：两棵 release 树上都命中（本章 §3.3 核对过
`tilelang/contrib/cutedsl/__init__.py` 与 `src/cuda/codegen/codegen_cutedsl.cc`
里都有这两个符号），因此状态是「v0.1.15 已具备」，且不是 v0.1.15 新增。

手工复核：

```bash
rg -l 'launch_dependents|griddepcontrol' \
   _vendor/tilelang-v0.1.14/tilelang _vendor/tilelang-v0.1.14/src \
   _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src
python3 code/01-state-audit/state_audit.py     # 结尾应为 self-check: ok
```

写完再想一步：探针是不是太宽松了？「出现 `griddepcontrol` 字符串」可能只是注释。
更严的做法是再加一个路径探针，或把 scope 收到具体文件
（`Probe("text", "launch_dependents", scope="tilelang/contrib/cutedsl/__init__.py")`）。

**常见错误**：只写一个文本探针，于是任何提到该词的文档都会让能力「命中」。
一条能力至少两个探针，是本章 §2.3 强调的原则。

---

## E11 动手题 ★★

**解题思路**：先跑脚本，再把结果与 §4.1 的结论对上。

**参考答案**：

1. 只在 v0.1.15 里出现的 `examples/` 路径主要是 `examples/ascend/` 下的内容：
   `examples/ascend/README.md`、`example_gemm.py`、`example_gemm_splitk.py`、
   `example_per_token_cast_to_fp8.py`、`example_rmsnorm.py`、`example_vecadd.py`、
   `flash_attention/`（含 `example_mha.py`、`example_gqa.py`、`example_gqa_bwd.py`…）、
   `deepgemm/`，以及对应的 `test_*.py`。
   （具体清单以你运行时脚本的输出为准；脚本默认每节最多列 40 条。）
2. 只在 v0.1.14 里出现的 `examples/` 路径通常为 0 条——v0.1.15 是在它之上加东西，
   没有整目录删除。如果脚本报出非空结果，应当逐条去看是什么，
   不要默认它是噪音。
3. 因为 `examples/ascend/` 是随 PR #3308 一起进主干的，而 #3308 的内容已在
   v0.1.15 里（见 §4.1.1），v0.1.14 里完全没有昇腾后端。
   `examples/ascend/flash_attention/example_gqa.py` 这类文件属于
   **「v0.1.15 已具备」**档——它们是 950 后端自带的例子；
   但它们**跑起来需要昇腾 950 硬件 + CANN + Bisheng**，
   所以是「能用」而不是「你现在这台机器上能跑」。

**常见错误**：把第 3 问答成「属于未合入」。例子随后端一起合入了；
真正未合入的是 §4.5 那四条 PR 对应的能力。

---

## E12 动手题 ★★★

**解题思路**：题目要的是一份可复核的报告，而不是一个新的结论。
核心是「反向验证 + 信息完备 + 措辞准确」。

**参考答案**：

1. 反向验证 = 在当前能拿到的最强证据里找这条能力的**必然痕迹**，例如：

   ```bash
   rg -n 'msprof'  _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py
   rg -n '^def |^class ' _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py
   rg -n -i 'correlat|host_time|device_time|kernel_name' \
      _vendor/tilelang-v0.1.15/tilelang/profiler/
   ```

   如果「host 时间 ↔ device kernel」的关联结构确实存在，它一定会在这两个文件里
   留下字段或函数名；搜不到，就是「未在本章快照中找到」。
2. 报告至少要有五项，缺一项别人就无法复核：
   ① 被判定能力的名称与来源 PR/issue 编号；
   ② 用到的每一棵树的**路径与 commit**（`_vendor/tilelang-v0.1.15` 是 `a35f8ddf`，
   main 快照是 `994b44e`）；
   ③ 判定日期；
   ④ 逐条验证命令与它们的原始输出（哪怕是「无输出」）；
   ⑤ 结论措辞与不确定项。
3. 措辞建议：

   > 未在本章的 `_vendor/tilelang-v0.1.15`（`a35f8ddf`，2026-09-30）与
   > main 快照（`994b44e`）中找到与「host/device profiling 关联」对应的实现；
   > 该 PR 的具体内容本章无法确认（corpus 未收录其页面快照）。

   要点是「在哪些树上没找到」+「对 PR 本身不下判断」，而不是
   「这个功能不存在」。
4. 需要补的一项：**PR 的页面快照**（标题、作者、状态、改动文件清单、
   作者自报的测试条件与结果）。有了它，才能把「PR 要做什么」从转述升级为
   直接证据，并把结论从「无法确认细节」推进到「未合入，内容为……」。

**常见错误**：把「搜不到」直接写成结论，或在报告里省略 commit 与日期，
导致报告过期后无法判断哪一部分失效了。
