// Excerpt of an official Ascend C kernel (LeakyReLU), trimmed for reading.
//
// Source (verified, read-only snapshot):
//   ../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/
//       zh_developer_techArticles_20230830-1.md:350-440
//
// It is an ELEMENTWISE operator, not the chapter's GEMM+ReLU; the corpus does
// not contain a full Ascend C GEMM+epilogue listing (see references.md, gap 3).
// It is included because it shows the Ascend C programming model that
// TileLang@Ascend removes:
//   * CopyIn / Compute / CopyOut split;
//   * TPipe + TQue (EnQue/DeQue) for inter-stage sync and multi-buffering;
//   * BUFFER_NUM written by the user.
//
// Verification level: static only; never compiled or run.

__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                            uint32_t tileNum, float scalar) {
    this->blockLength = totalLength / GetBlockNum();
    this->tileNum = tileNum;
    this->scalar = static_cast<half>(scalar);
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
    // Each core binds its own slice of global memory.
    xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * get_block_idx(),
                        this->blockLength);
    yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * get_block_idx(),
                        this->blockLength);
    // The user declares the queue depth explicitly (multi-buffering).
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
}

__aicore__ inline void Process() {
    // Loop count doubles because of the double buffer -- user's job here.
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__aicore__ inline void CopyIn(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    DataCopy(xLocal, xGm[progress * tileLength], tileLength);
    inQueueX.EnQue(xLocal);
}

__aicore__ inline void Compute(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    LeakyRelu(yLocal, xLocal, scalar, tileLength);
    outQueueY.EnQue<half>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

__aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<half> yLocal = outQueueY.DeQue<half>();
    DataCopy(yGm[progress * tileLength], yLocal, tileLength);
    outQueueY.FreeTensor(yLocal);
}
