# 第 32 章 习题参考答案

说明：每题给「解题思路 + 参考答案 + 常见错误」。引用正文时用「§N.M」，引用源码时给
`../tilelang-corpus/...` 或 `_vendor/...` 的路径与行号。本章没有需要真机的题，所以答案
都可以在写作机上核对。

---

## E1

**思路**：§2.1 的表格已经把「组件 → 一句话定位」列好了，归层是 §3.1 的四层。

**参考答案**

1. 分层：
   * **语言/编译**：TileLang 昇腾版。
   * **计算库**：DeepGEMM-Ascend（GEMM / MQA logits / MegaMoE）、FlashMLA（稀疏注意力）、
     DeepSelect（TopK 筛选）。
   * **算子库**：TileKernels（MoE 路由、量化、RoPE、Engram、mHC 等「常规向量计算和
     访存算子」）。
   * **通信库**：DeepEP-Ascend。
   （DeepJIT 不属于这四层中的任何一层，它是跨层的运行时，见 §3.2 的表格。）
2. 「常规」这个定语排除了 GEMM、Attention、TopK 这类**极限性能算子**。文章把这三类
   分别交给了 DeepGEMM、FlashMLA、DeepSelect（
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:12`）。
   TileKernels 的职责是「数据处理所需的常规向量计算和访存算子」。
3. §1.2 说这次发布是「高级语言编译工具、计算库、分布式通信库」三层一起给。只有算子库
   的话，用户还是缺两样东西：写算子要用什么语言（TileLang）、多卡训推要怎么通信
   （DeepEP-Ascend）。三层齐了才叫「全栈」。

**常见错误**：把 DeepJIT 也归进「计算库」。它不是算子库，是运行时（§3.3）。

---

## E2

**思路**：§2.2 已把这句话拆成三点；§2.3 已点出两个过度推论。

**参考答案**

1. 三个目标：
   * **封装 Ascend C 底层指令**——用户不直接管理 `TPipe`/`TQue`、`LoadData`、
     `SetFlag`/`WaitFlag` 这些搬运与同步细节。
   * **提供高级语言的编程方式**——用 tile 级的数据流描述算法（`T.copy` / `T.gemm` /
     `T.Pipelined`）。
   * **不损失硬件性能**——封装不能以性能为代价。
2. 不能当性能承诺，因为它说的是**设计目标**，不是「所有算子达到峰值」。核验真实性能
   要用各仓库 README 里的性能表，并且必须带上测量条件（硬件型号、CANN 版本、形状、
   口径），见 §6.2 与 §8。
3. 两个过度推论：
   * 「PyTorch 里所有算子都有昇腾实现」——原文限定在「训练中用到的、且本来就用
     TileLang 写的」算子。
   * 「DeepSeek 全部算子都用 TileLang 写 / 昇腾侧是同一份代码」——原文只承诺「有对应的
     高性能实现」，不承诺同一份代码；实际上昇腾侧有大量 Ascend C 手写实现（§4.4）。

**常见错误**：把「对应的高性能实现」理解成「TileLang 一条路走到底」。第 35–38 章
会反复展示反例。

---

## E3

**思路**：§4.2 明确列了三层：API、心智模型、实现技巧。

**参考答案**

1. 三层可迁移性：
   * **API 可迁移（最强）**：DeepGEMM → DeepGEMM-Ascend 是同一个包名与同一套 API
     （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:3,9`）；
     TileKernels / DeepSelect 甚至是一份 Python API、运行期选后端
     （`.../TileKernels/README.md:9`、`.../DeepSelect/deep_select/interface.py:10-23`）。
   * **心智模型可迁移（中等）**：第 14–16 章学的 GEMM / Attention / MoE 算法不变，
     变的是硬件约束。
   * **实现技巧不可迁移（最弱）**：稀疏加载、协程流水等是平台特有的。
2. 例：DeepGEMM-Ascend「makes extensive use of Ascend-specific optimization techniques,
   such as sparse data loading and coroutine-based pipelining」
   （`.../DeepGEMM-Ascend/README.md:5`）。这些技巧在 CUDA 上没有对应物。
3. 错在把「接口对应」当成「行为等价」。§4.3 给了三类反例：格式（scaling factor 打包）、
   子集（融合 kernel 仅 CUDA、昇腾仅 bf16/int32）、语义（DeepEP-Ascend 的 stream 行为与
   支持模式是昇腾特有的）。调参经验属于「实现技巧」层，**恰恰是三层里最不可迁移的**。

**常见错误**：只举 API 层的例子就结束，忽略了后两层。

---

## E4

**思路**：三类差异都在 §4.3，且都有 README 原文。

**参考答案**

1. **格式差异**：
   * 昇腾：「each pair of UE8M0 scaling factors along the K dimension is packed into an
     `int16`, and the packed values are stored in MN-major order」
     （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:55-56`）。
   * NVIDIA SM100：「packs 4 UE8M0 into a single `torch.int`」
     （`.../DeepGEMM/README.md:64-67`）。
   * 两者不能互换；昇腾侧提供 `transform_sf_into_required_layout` 做转换
     （`.../DeepGEMM-Ascend/README.md:67-68`）。
2. **子集差异**：
   * FlashMLA 的融合算子「This kernel currently supports CUDA only; Ascend is not
     supported.」（`.../FlashMLA/README.md:43`）；昇腾侧这一版给的是稀疏 prefill / decode。
   * DeepSelect 昇腾侧只支持 `torch.bfloat16`（`.../DeepSelect/README.md:37`）与
     `torch.int32` 输出（`:90`），`sorted` 在昇腾上只支持 `False`
     （`.../DeepSelect/deep_select/interface.py:54`）。
3. **语义差异**：DeepEP-Ascend 提醒「supported modes and stream behavior are specific to
   Ascend」（`.../DeepEP-Ascend/README.md:62`），并且 EP 有额外约束
   `do_expand=True`、`allow_multiple_reduction=True`、`do_handle_copy` 必须保持 `False`
   （`:72`）。

**常见错误**：把「昇腾侧没有 FP32 的 TopK」写成「DeepSelect 不支持 FP32」——CUDA 侧
是支持的（`.../DeepSelect/README.md:37`）。差异是**平台相关**，不是组件整体的能力。

---

## E5

**思路**：这段代码是「一份算法、两个后端」的入口，看的是**运行期选择**。

**参考答案**

1. **每次调用**（严格说是首次拿到 kernel 时）做选择；判据是 `is_ascend()`
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/config.py:24-31`
   用 `os.path.exists('/dev/davinci_manager')` 判断，并带 `lru_cache`）。
2. 因为昇腾与 CUDA 的后端模块在不同的构建里才存在。如果写在文件顶部，在没有装对应
   后端的机器上，`import` 阶段就会 `ImportError`，连另一条分支都走不到。写在 `if` 里
   可以让「同一份 Python 源码」在两种环境下都能导入。
3. `get_num_vec_cores()` 反映昇腾的 Vector 核数（AIV 数 = AI Core 数 × 2，
   `tile_kernels/config.py:70-71`），因为 top-k gate 是向量/归约型算子；`use_pdl` 反映
   CUDA 的 Programmatic Dependent Launch（把前一个 kernel 的尾部与后一个 kernel 的
   头部重叠），昇腾侧没有这个概念。这就是 §4.3 说的「实现技巧不可迁移」。
4. 对应 §4.2 的第一层：**调用方只需要 `topk_gate(scores, k)`，看不见设备**。

**常见错误**：以为后端选择发生在编译期。这段代码是纯 Python，运行时才决定。

---

## E6

**思路**：先认树，再列原语，最后回答「为什么只有这一个算子用 TileLang」。

**参考答案**

1. `from tilelang.ascend import language as T`
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:7`），
   另有 `from tilelang.layout import make_ascend_compact_nz_layout`（:8）。这属于**主干
   昇腾后端**（M 树），不是适配器仓库。
2. 昇腾专属原语（举例）：
   * `T.MixedKernel(num_aic_cores) as (core_id, aiv_id)`（:105）：混合 Cube/Vector 核。
   * `T.alloc_l1 / T.alloc_l0a / T.alloc_l0b / T.alloc_l0c`（:116-120）：显式分配 L1 与
     L0A/L0B/L0C。
   * `T.annotate_buffer_versions({...})`（:122-131）：给 buffer 指定版本数（多缓冲）。
   * `T.SimdVF(latency=640)`（:195）与 `T.simd.*`（:196-266）：手写 SIMD 向量指令。
   * `T.dual_copy(...)`（:268-271）：把一次拷贝劈给两个 AIV。
   * `T.gemm(..., transpose_B=True, clear_accum=..., unit_flag_ctrl=...)`（:281-292）。
   * `T.PerCoreTask()` + `T.Task()`（:294-356）：按核组织任务与核间同步。
3. 注释说的是：AutoSchedule（自动调度）在 `T.PerCoreTask` 里**拒绝条件控制流下的核间
   等待**，所以作者把「可选的末尾 barrier」编码成一个 0/1 次的统一循环来绕开（:96-103）。
   这说明用 TileLang 写不等于不用理解调度——自动调度有它的边界，越界时要手工改写成它
   能接受的形状。
4. 判据：
   * **用 TileLang**：算子的形状与数据流能用 tile 表达，且瓶颈不在「必须手控 MAD 原语
     与稀疏加载」上。mHC prenorm 是一个 `D = A @ Bᵀ` 加上沿行的平方和归约（:369），
     属于 TileLang 的舒适区。
   * **不用 TileLang**：要压到接近硬件峰值、需要控制分形布局、对齐、地址计算、稀疏
     加载与协程流水（`.../DeepGEMM-Ascend/README.md:5`）。

**常见错误**：只看文件名就断言「DeepGEMM-Ascend 是 TileLang 库」。整库只有
`tilelang_ops/hc_prenorm.py` 一个文件是 TileLang。

---

## E7

**思路**：DeepSelect 的平台差异写在文档字符串里，不是单独的 `Ongoing` 节。

**参考答案**

1. `_IS_ASCEND = os.path.exists("/dev/davinci_manager")`
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/deep_select/interface.py:10`），
   与 TileKernels 的判据**相同**（都在探测同一个设备文件）。
2. 至少四条（`interface.py:52-68`）：
   * `input` 的 dtype：Ascend only supports `torch.bfloat16`（子集差异）。
   * `sorted`：只有 CUDA 的 fp32 变体支持；Ascend only supports `False`（子集差异）。
   * `indices_type`：Ascend only supports `torch.int32`（子集差异）。
   * NaN 回退：CUDA 只写 `output_idx[batch_idx][0]`，昇腾整行写 `0x3F3F3F3F` 且短行也检查
     （语义差异）。
   * `begin` / `hint` 目前不支持。
3. 例如 `indices_type`：签名默认 `torch.int64`，在 CUDA 上传 `torch.int64` 是合法的，
   在昇腾上会失败——**同一个参数、同一个缺省值，两个平台的行为不同**。
4. 补充：「一一对应」不仅要看接口名，还要看**每个参数的平台支持矩阵**。参数级的差异
   比函数级的差异更隐蔽。

**常见错误**：以为平台差异一定写在 `Ongoing` 一节。DeepSelect 就没有这一节。

---

## E8

**思路**：三种状态在 §8.3 的表格里。

**参考答案**

1. 判据：
   * **已发布**：README 有性能表；接口在 API 列表里且没有实验性标注。
   * **实验性**：接口存在，但被显式标为 experimental / ongoing。
   * **未实现**：只有 API 预留（或根本不在表里），对应 README 明说 not implemented。
2. 例子（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:76-84`）：
   * 已发布：EP dispatch / combine（有带宽表，`:13-21`）。
   * 实验性：PP 与 Engram「remain experimental」。
   * 未实现：`lb_prefetch_weights` / `lb_reduce_grads` 的 Ascend kernel「are not
     implemented yet」；另外 reduce-scatter / all-reduce 也还在做。
3. 处理动作（至少两条）：
   * **不要写进生产路径**，先把这条依赖从设计里去掉或换成已实现的替代方案；
   * **把「什么时候能回来」写进跟踪清单**——记录快照 commit，下次升级时重读 `Ongoing`；
   * 如果要提前用，至少在测试里做成可跳过的分支，并明确失败时的降级路径。
4. 不能。DeepSelect 就没有 `Ongoing` 一节，但它的平台限制写在文档字符串里（E7）。
   正确的结论是「这份快照里没写」，而不是「没有限制」。

**常见错误**：把 `BucketBuffer.all_gather` 当成「Bucket 已经齐全」——`Ongoing` 明确说
reduce-scatter 与 all-reduce 还在做。

---

## E9

**思路**：这是一道「用工具复现核验流程」的题，答案取决于脚本实际输出。

**参考答案**

1. 命令与预期形态（实际输出以运行结果为准）：

   ```bash
   python3 code/01-verify-checklist/extract_sections.py \
       ../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md
   ```

   脚本会按小节名（`Requirements`、`News`、`Ongoing` …）打印标题与正文；找不到的小节
   会打印 `(未找到)`。
2. 结论形态：「DeepEP-Ascend 在本快照上要求 Ascend 950 (A5) 且需要 UBMEM 连接与
   UBC_CTP/URMA 通道；验证过的组合是 Ascend 950DT + CANN 9.2.0 + Python 3.12 + PyTorch
   2.13.0+cpu + torch_npu 2.13.0rc1；其它代际与 CANN 版本尚未由它的测量确认。」
   证据：`README.md:28-29`（硬件/通道）、`:31,34`（版本与「has not been established」）、
   `:36`（工具链要在运行时可用）。
3. 两次输出的结构差异通常在于「哪些小节存在」。例如 TileKernels 有 `News` 与
   `Requirements`，但没有 `Ongoing`；脚本必须如实打印 `(未找到)`，否则读者会误以为
   「没有输出 = 没有问题」。

**常见错误**：把脚本输出当成权威结论。脚本只是**抽取**，判断仍然要人来做。

---

## E10

**思路**：这是开放题，评分的标准是「每一问都有出处、存疑处有交代」。

**参考答案（示例：以 FlashMLA 为例）**

1. 组件：FlashMLA（昇腾侧）；快照 commit `2e5429fc5653`（2026-09-30，
   「Open-source release for Ascend 950（2026.09.30）」），见 `PROVENANCE.md`。
2. 六问：
   * 属于哪一段：§6.1 的 ③（稀疏注意力）。
   * 硬件/工具链：Ascend 950 NPU、CANN 9.2.0 或更高、`torch_npu`、PyTorch 2.0+
     （`README.md:83-88`）。
   * 最近更新：2026-09-30 发布昇腾稀疏 prefill/decode（`README.md:13`）。
   * 实验性/未实现：昇腾侧不含融合 norm-RoPE-attn-RoPE-cast（`:43`），
     decode 只支持 FP8/FP4 KV cache（`:136`）。
   * commit 是否一致：是（同上）。
   * 性能条件：Ascend 950、prefill 410 TFlops（95%）、decode 360 TFlops（83%）
     （`:13,53,63`），口径是「up to」与相对理论峰值。
3. 结论句示例：「FlashMLA 在 2026-09-30 快照上为 Ascend 950 提供了 DSA 稀疏 prefill
   与 decode 算子（README 自述 up to 410 / 360 TFlops），但这一版不含融合 kernel，
   且 decode 只支持 FP8/FP4 KV cache。」
4. 存疑处：README 提到的模型页无法在本机核实（正文 §10.2）；「理论硬件峰值」的具体
   数值 README 没有直接给出，只有百分比。

**常见错误**：把「up to 410 TFlops」写成「FlashMLA 昇腾侧是 410 TFlops」。必须保留
「up to」与测量条件。

---

## E11

**思路**：公式是定性的量级估计，不是精确模型；重点是「卡数 ↑ ⇒ 比值 ↑」。

**参考答案**

1. 各项：`tokens` 是这一步要处理的 token 数；`topk` 是每个 token 路由到的专家数；
   `hidden` 是隐藏维；`b` 是每个元素字节数（BF16 为 2、FP8 为 1）。
2. 两个角度：
   * 卡越多，专家被分散到越多设备上，一个 token 的 top-k 专家落在本卡之外的概率越大，
     需要跨卡搬运的比例越高。
   * 卡越多，每张卡分摊到的本地 token 越少，本地计算量下降，而通信量（近似）不随卡数
     下降，于是「通信/计算」的比值上升。
3. 那句话没有给出：**拓扑**（几级交换、什么组网、是否全互联）、**带宽/延迟数值**、
   **具体硬件型号与规模细节**（例如超节点里每张卡的角色）。本章的处理方式是：只引用
   这句话本身，把可核到的部署形态交给 DeepEP-Ascend 的 README（§7.3），并明确标注
   那是 PoC 配置、不是公开发行版。

**常见错误**：从这句话直接推出「128 卡全互联、单跳带宽 X」。原文没有这些信息。

---

## E12

**思路**：把 §6.1 的表格按推理顺序走一遍，再叠上 §8 的核验清单。

**参考答案**

1. 顺序（MoE 推理，DSA 稀疏注意力）：
   * 输入与常规算子（归一化 / RoPE / 量化 / 采样）：**TileKernels**（§6.1 第 ⑤ 段的
     「常规算子」）。
   * 索引打分：**DeepGEMM-Ascend** 的 MQA logits（第 ① 段）。
   * TopK 筛选：**DeepSelect**（第 ② 段）。
   * 稀疏注意力：**FlashMLA**（第 ③ 段）。
   * 专家并行通信：**DeepEP-Ascend**（第 ④ 段）。
2. 每个组件第一件要确认的事：
   * TileKernels：`Requirements` 里的 Ascend 950 与 CANN 9.2.0（`README.md:29-31`）。
   * DeepGEMM-Ascend：`Requirements` 里的 CANN 9.20 与 `torch_npu`（`:23-26`），
     以及 MQA logits 是 FIX-pipe bound（`:142`）。
   * DeepSelect：输入 dtype 与 `indices_type` 的昇腾限制（`:37`、`interface.py:52-68`）。
   * FlashMLA：decode 只支持 FP8/FP4 KV cache（`:136`）。
   * DeepEP-Ascend：`Ongoing` 里哪些是实验性 / 未实现（`:76-84`）。
3. 障碍：想要 `torch.float32` 输入或 `sorted=True`，都是 **DeepSelect** 的限制——
   昇腾实现只支持 `torch.bfloat16`（`.../DeepSelect/README.md:37`），`sorted` 在 CUDA 的
   fp32 变体上才支持、昇腾只支持 `False`（`interface.py:54`）。另外 `indices_type` 在
   昇腾上只能 `torch.int32`（`:59`）。
4. 总结句示例：「只凭这份快照，我能确定**有哪些组件、它们的接口形状、以及各自的
   硬件/CANN 要求**；我不能确定**真实端到端性能、超节点的具体拓扑、以及 PoC 配置
   之外的可获得性**——这些都超出 README 的范围，需要真机与最新的厂商资料。」

**常见错误**：把 DeepSelect 的 `sorted` 限制说成「DeepSelect 不支持排序」——CUDA 侧
支持，昇腾侧不支持。
