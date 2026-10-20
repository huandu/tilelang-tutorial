# 图 6 · 昇腾 MTE/Cube/Vector 流水与多缓冲同步

正文位置：第 4.3、4.4 节。

```mermaid
sequenceDiagram
    participant GM
    participant MTE2
    participant L1 as L1 Buffer (slot 0/1)
    participant MTE1
    participant Cube as Cube (AIC)
    participant L0C
    participant Fix as Fixpipe
    participant UB as Unified Buffer (slot 0/1)
    participant Vec as Vector (AIV)
    participant MTE3
    Note over L1,UB: 多缓冲：每个 buffer 有多个物理槽位
    MTE2->>L1: 搬入 tile k（写 slot k%2）
    Note over MTE2,L1: SetFlag(MTE2→MTE1)：“数据就绪”
    MTE1->>L1: WaitFlag(MTE2→MTE1)
    MTE1->>Cube: L1 → L0A/L0B
    Cube->>L0C: 矩阵累加
    Note over MTE1,L1: SetFlag(MTE1→MTE2)：“缓冲可归还”
    MTE2->>L1: WaitFlag(MTE1→MTE2) 后才覆盖该槽
    Fix->>UB: L0C → UB
    Vec->>UB: 读 UB 计算
    Vec->>MTE3: 结果
    MTE3->>GM: UB → GM
    Note over Vec,MTE2: 跨核：Cube 与 Vector 之间的生产—消费也要成对同步
```

## 这张图想说什么

* 每一次「覆盖同一块 buffer」都需要一个**反向**的归还信号。正反两个方向是**独立的
  event id 空间**（TileLang-Ascend 编程指南：「event ID 按有向 pipe pair 分配」）。
* 多缓冲版本数、片上容量和 Set/Wait 必须**一起**算，不能分开。
* 在 TileLang 里你不用手写这些 flag：`InsertSync` 把它们插进去
  （`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:113`）。

## 事实来源

* MTE 路径与 SetFlag/WaitFlag 语义：
  `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md`。
* 指令队列与 double buffer：
  `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20240819-1.md`（厂商示意模型）。
* 手工同步的完整生命周期（含反向归还、`MTE2 → MTE1` 与 `MTE1 → MTE2` 的例子）：
  `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md`
  「同步原语」一节。
* `AutoSchedule → PrepareMultiBuffer → InsertSync → MaterializeMultiBuffer` 的顺序：
  `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:107-115`。

## 注意

图中是**结构示意**，不是某次运行的 trace，也不含任何延迟/带宽数字。真实时间线必须用
msProf 或 camodel 在设备上取证（见第 12、19 章）。

## 许可

自绘（Mermaid），随本书正文。
