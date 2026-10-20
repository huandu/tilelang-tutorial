# 图 2：NVIDIA 侧与昇腾侧的一一对应

**用途**：正文 §4.1–§4.3。

```text
[fig-02] NVIDIA 侧与昇腾侧的一一对应

      能力            NVIDIA 侧                 昇腾侧
  ┌────────────┬─────────────────────┬──────────────────────────┐
  │ GEMM       │ DeepGEMM            │ DeepGEMM-Ascend（API 兼容）│
  │ 通信       │ DeepEP              │ DeepEP-Ascend（API 对齐）  │
  │ 注意力     │ FlashMLA            │ FlashMLA（同仓库双平台）   │
  │ 算子库     │ TileKernels(CUDA)   │ TileKernels(Ascend)       │
  │ TopK       │ DeepSelect(CUDA)    │ DeepSelect(Ascend)        │
  │ 语言/编译  │ TileLang 主干       │ TileLang 昇腾后端          │
  │ 运行时 JIT │ DeepJIT(CUDA)       │ DeepJIT(Ascend)           │
  └────────────┴─────────────────────┴──────────────────────────┘
```

## 要点

* **接口可迁移**：DeepGEMM → DeepGEMM-Ascend 是同一个包名与同一套 API；DeepEP →
  DeepEP-Ascend 的公开 buffer API 对齐；TileKernels / DeepSelect 是**一份 Python API**，
  运行期按设备自动选后端。
* **实现不可迁移**：稀疏加载、协程流水、地址计算、同步插入都是各自平台特有的。
* **哪里并不对应**（正文 §4.3 展开）：
  * scaling factor 打包格式不同——昇腾把**一对** UE8M0 打包进 `int16` 且按 MN-major 存，
    NV 侧 SM100 把**四个** UE8M0 打包进一个 `torch.int`；
  * 功能子集不同——FlashMLA 的融合 norm-RoPE-attn-RoPE-cast 只支持 CUDA，
    DeepSelect 的昇腾实现只支持 bf16 / int32；
  * 语义不同——DeepEP-Ascend 的 stream 行为与支持模式是昇腾特有的。

来源：`../../../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:3,55-56`、
`.../DeepGEMM/README.md:64-67`、`.../DeepEP-Ascend/README.md:3,62`、
`.../FlashMLA/README.md:43,101`、`.../TileKernels/README.md:9`、
`.../DeepSelect/README.md:37`、`.../DeepJIT/README.md:11`。
