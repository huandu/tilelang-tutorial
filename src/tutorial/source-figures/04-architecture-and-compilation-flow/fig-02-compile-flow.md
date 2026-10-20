# 图 2：一次编译的全流程（Python kernel → 可调用对象）

来源：本章作者根据 `_vendor/tilelang-v0.1.15/` 源码走读绘制。每个方框后面标注
了实际实现所在的“文件:行号”，行号核验日期 2026-10-01。

```mermaid
flowchart TB
    A["1. 用户写 kernel<br/>@tilelang.jit"]
    B["2. 装饰器：包成 JITImpl<br/>jit/__init__.py:588-652"]
    C["3. 取值取 shape：get_tir()<br/>把 Python 函数跑成 PrimFunc<br/>jit/__init__.py:379-392"]
    D["4. 缓存查询<br/>cache/__init__.py:75-98<br/>未命中才继续"]
    E["5. 创建 BackendContext（只一次）<br/>backend/module.py:310-335"]
    F["6. 前端语义检查<br/>engine/semantic_check.py:21-31"]
    G["7. 后端 pass pipeline<br/>cuda/pipeline.py:284 或 ascend/pipeline.py:190"]
    H["8. host / device 切分<br/>engine/lower.py:109-141"]
    I["9. device codegen<br/>engine/lower.py:99-106<br/>CUDA C / Ascend C"]
    J["10. host codegen（可选）<br/>engine/lower.py:65-89"]
    K["11. CompiledArtifact<br/>engine/param.py:153-166"]
    L["12. adapter 构造<br/>jit/kernel.py:324-424"]
    M["13. 可调用对象<br/>JITKernel.__call__ → torch_function"]

    A --> B --> C --> D --> E --> F --> G --> H
    H --> I --> K
    H --> J --> K
    K --> L --> M
    D -. "命中则直接返回 JITKernel" .-> M

    classDef phase fill:#eef3ff,stroke:#4a6fa5
    class A,B,C,D,E,F,G,H,I,J,K,L,M phase
```

## 关键分界线

```text
        Python 侧（可读、可调试）                   C++ 侧（性能、IR 变换）
  +---------------------------------------+  +--------------------------------------+
  | decorator -> PrimFunc -> BackendContext|  | pass 本体（src/transform/*.cc）       |
  |  -> pipeline 的 Python 调度            |->| codegen（src/cuda, src/ascend）      |
  +---------------------------------------+  +--------------------------------------+
                    ^                                        ^
        「顺序」写在这里（backend/<x>/pipeline.py）   「做什么」写在这里（src/transform/）
```

这条分界线解释了两件事：

1. 想改 **pass 的顺序、开关、目标相关裁剪** → 改 `tilelang/<backend>/pipeline.py`。
2. 想改 **某个 pass 的行为** → 改 `src/transform/`（或对应后端的 `src/<backend>/transform/`）。

## 三条容易被忽略的支路

```text
  adapter 选择（jit/kernel.py:338-424）
      tvm_ffi | cython | nvrtc | torch(Metal) | cutedsl
                ^ 由 BackendContext.execution_backend 决定，不由 target 决定

  host codegen 开关（engine/lower.py:144-213）
      enable_host_codegen=False（默认）时只产出 device 源码 + IR，交给 adapter 自己编译

  缓存短路（cache/__init__.py:75-98）
      kernel_cache 命中时整个 pipeline 都不会跑；这也是「改了 pass 但结果没变」的常见原因
```
