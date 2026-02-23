const std = @import("std");

// ==========================================
// 1. 底层 PTX 寄存器读取 (获取线程坐标)
// ==========================================
// 我们通过 Zig 的内联汇编 (Inline Assembly) 直接读取 NVIDIA GPU 的特殊硬件寄存器

inline fn threadIdx_x() u32 {
    return asm ("mov.u32 %[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
}

inline fn blockIdx_x() u32 {
    return asm ("mov.u32 %[r], %ctaid.x;"
        : [r] "=r" (-> u32),
    );
}

inline fn blockDim_x() u32 {
    return asm ("mov.u32 %[r], %ntid.x;"
        : [r] "=r" (-> u32),
    );
}

// ==========================================
// 2. CUDA Kernel 本体
// ==========================================
// 关键点 1: 必须使用 export 暴露出符号
// 关键点 2: 必须使用 callconv(.PtxKernel) 告诉 LLVM 这是 GPU 入口函数

export fn matmul_kernel(
    xout: [*]f32,
    x: [*]const f32,
    w: [*]const f32,
    n: u32,
    d: u32,
) callconv(.nvptx_kernel) void {
    // 计算当前线程的全局一维索引
    const i = blockIdx_x() * blockDim_x() + threadIdx_x();

    // 边界检查：防止线程总数大于 n 时发生内存越界
    if (i < n) {
        var val: f32 = 0.0;

        // 当前线程负责的权重矩阵的起始位置 (第 i 行)
        const row_offset = i * d;

        // 执行点积运算：dot(w[i, :], x)
        // 这里的循环会在单个 CUDA Core 上顺序执行
        var j: u32 = 0;
        while (j < d) : (j += 1) {
            // 在实际的高性能版本中，这里可以使用 shared memory 或 warp reduction 优化
            // 但作为基线版本，这种 1-thread-per-row 的写法最安全
            val += w[row_offset + j] * x[j];
        }

        // 将结果写回全局显存
        xout[i] = val;
    }
}
