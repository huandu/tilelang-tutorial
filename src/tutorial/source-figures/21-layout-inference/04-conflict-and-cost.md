# 图 4：冲突的三条出路与代价择优

来源：`_vendor/tilelang-v0.1.15/src/transform/layout_inference/layout_inference.cc:437-468`
（三条出路）、`src/layout/utils.cc:443-480`（包含检查）、
`src/layout/gemm_layouts.cc:999-1023`（swizzle 合并）、
`src/transform/layout_inference/layout_cost_model.h:1-90`（代价）。图为**本书绘制**。

## Mermaid

```mermaid
flowchart TD
    X["同一 buffer 收到两个不同布局"] --> C1{"新布局包含旧布局？<br/>ProveFragmentContains"}
    C1 -->|是| OK1["升级到更宽的布局"]
    C1 -->|否| C2{"两者都是 swizzle？"}
    C2 -->|是| OK2["合并：取更小粒度"]
    C2 -->|否| C3{"在 kFree 阶段？"}
    C3 -->|是| OK3["逐根重推整条链路<br/>用代价模型择优"]
    C3 -->|否| FAIL["LOG(FATAL)<br/>Get different layout for ..."]
```

## 代价比较（字典序）

```text
   AttemptCost { mem, regs }
   BetterThan(other):
       mem  != other.mem  ->  mem  < other.mem
       mem  == other.mem  ->  regs < other.regs

   register-count（默认）: mem = 寄存器数组溢出字节;  regs = fragment 槽位总数
   io-aware    （可选）  : mem 再叠加 触及 global 的语句按
                           max(带宽字节, 发射等价字节)
```

配置键：`tl.layout_cost_model`（`tilelang/transform/pass_config.py:122`）。
切换模型会改变同一次编译的推导结果——可作为「转换代价被忽略」的实验。
