# 图 1：一次编译的产物路径（NVIDIA）

```mermaid
flowchart TD
    A["Python frontend<br/>tilelang/language/*"] --> B["TIR module<br/>@T.prim_func"]
    B --> C["tilelang/engine/lower.py::lower<br/>create_backend_context → 选后端"]
    C --> D["CUDA_PIPELINE<br/>tilelang/cuda/pipeline.py"]
    D --> D1["PipelinePlanning → InjectSoftwarePipeline"]
    D1 --> D2["LayoutInference → LowerTileOp<br/>（这里做 T.gemm / T.copy 的指令选择）"]
    D2 --> D3["LowerHopperIntrin<br/>（描述符提升为 host 属性）"]
    D3 --> D4["AnnotateDeviceRegions → SplitHostDevice"]
    D4 --> H["host_mod"]
    D4 --> V["device_mod"]

    V --> V1["CodeGenTileLangCUDA<br/>src/cuda/codegen/codegen_cuda.cc"]
    V1 --> V2["CUDA C 源码<br/>kernel_source / source_map['cuda']"]
    V2 --> V3a["nvcc / NVRTC<br/>tilelang_callback_cuda_compile"]
    V3a --> V4["cubin 或 fatbin"]

    H --> H1["TLCUDASourceWrapper<br/>tilelang/jit/adapter/wrapper.py"]
    H1 --> H2["extern \"C\" TL_EXPORT int call(...)"]
    H1 --> H3["cuTensorMapEncodeTiled(...)<br/>（若有 TMA）"]

    V2 --> W["source + init() + call()"]
    H2 --> W
    H3 --> W
    W --> E1["execution_backend = cython"]
    W --> E2["execution_backend = tvm_ffi"]
    V2 --> E3["execution_backend = nvrtc<br/>运行期编译"]
    V2 --> E4["execution_backend = cutedsl<br/>改生成 CuTe DSL Python"]

    classDef codegen fill:#e8f0ff,stroke:#3b5bdb
    class V1,H1 codegen
```

要点：

* device 与 host 是**两棵独立的产物树**，在 `SplitHostDevice` 处分开
  （`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:239-240`）。
* 默认情况下 device 只产源码不编译、host 由 adapter 自己生成 wrapper
  （`_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:144-170`）。
* 三种 execution backend 共用同一个 wrapper，所以 kernel 源码相同。
