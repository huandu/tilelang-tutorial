// 第 19 章 §6.1：同一个逐元素 add 的 Ascend C 写法（参考，非可编译工程）。
//
// 来源：昇腾官方文档
//   ../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/
//   canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_00020_html.md
// 这是「如何使用 Tensor 原地操作提升算子性能」文档里的示例，用于回答
// “TileLang 省掉了什么”。它在真正的 Ascend C 工程里还需要算子原型定义、
// tiling、编译与运行框架；本文件只截取最核心的 kernel 结构。
//
// 验证级别：仅静态核对（未编译、未运行）。
// 编译：真正的构建见 CANN 的 Ascend C 算子工程模板（CMake + <<<>>> 直调或
//       自定义算子包），本文件不能单独用 g++ 编译。

namespace AscendC {
class MyKernel {
public:
    __aicore__ inline MyKernel() {}

    __aicore__ inline void Init(__gm__ uint8_t* src0Gm,
                               __gm__ uint8_t* src1Gm,
                               __gm__ uint8_t* dstGm)
    {
        // ① 把裸指针绑定成 GlobalTensor。TileLang 里对应 T.Tensor(...) 参数。
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);

        // ② 手工建队列并为它分配 UB 空间。
        //    TileLang 里对应 T.alloc_ub(...) + 编译器的内存规划。
        pipe.InitBuffer(srcQue0, 1, BLOCK_SIZE * sizeof(half));
        pipe.InitBuffer(srcQue1, 1, BLOCK_SIZE * sizeof(half));
        pipe.InitBuffer(dstQue0, 1, BLOCK_SIZE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // ③ 显式的 CopyIn / Compute / CopyOut 三段循环。
        //    TileLang 里只有一段数据流描述，分段由编译器决定。
        for (int i = 0; i < REPTIMES; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t i)
    {
        srcQue0.AllocTensor<half>(src0Local);
        srcQue1.AllocTensor<half>(src1Local);
        // ④ MTE2：GM -> UB。
        DataCopy(src0Local, src0Global[i * BLOCK_SIZE], BLOCK_SIZE);
        DataCopy(src1Local, src1Global[i * BLOCK_SIZE], BLOCK_SIZE);
        // ⑤ EnQue 隐含「同步」：保证后续 DeQue 读到的是搬完的数据。
        //    TileLang 里对应 T.barrier_all()（A 树）或自动插入的同步（M 树）。
        srcQue0.EnQue(src0Local);
        srcQue1.EnQue(src1Local);
    }

    __aicore__ inline void Compute(int32_t i)
    {
        srcQue0.DeQue<half>(src0Local);
        srcQue1.DeQue<half>(src1Local);
        dstQue0.AllocTensor<half>(dstLocal);
        // ⑥ Vector 计算：注意输入输出都必须在 UB。
        Add(dstLocal, src0Local, src1Local, BLOCK_SIZE);
        dstQue0.EnQue<half>(dstLocal);
        // ⑦ 释放缓冲，交还给队列复用。
        srcQue0.FreeTensor(src0Local);
        srcQue1.FreeTensor(src1Local);
    }

    __aicore__ inline void CopyOut(int32_t i)
    {
        dstQue0.DeQue<half>(dstLocal);
        // ⑧ MTE3：UB -> GM。
        DataCopy(dstGlobal[i * BLOCK_SIZE], dstLocal, BLOCK_SIZE);
        dstQue0.FreeTensor(dstLocal);
    }

private:
    // ⑨ 整个算子要自己持有的对象：流水、队列、张量句柄。
    //    TileLang 里这些全部由编译器生成，用户不写。
    TPipe pipe;
    TQue<QuePosition::VECIN, 0> srcQue0, srcQue1;
    TQue<QuePosition::VECOUT, 0> dstQue0;
    GlobalTensor<half> src0Global, src1Global, dstGlobal;
    LocalTensor<half> src0Local, src1Local, dstLocal;
};
}  // namespace AscendC
