const std = @import("std");
const matrix = @import("matrix.zig");

// 引入底层的 CUDA Driver API
const cuda = @cImport({
    @cInclude("cuda.h");
});

// 辅助函数：捕获并打印 CUDA 错误
fn checkCuda(err: cuda.CUresult) !void {
    if (err != cuda.CUDA_SUCCESS) {
        var err_str: [*c]const u8 = null;
        _ = cuda.cuGetErrorString(err, &err_str);
        if (err_str != null) {
            std.debug.print("CUDA Error: {s}\n", .{err_str});
        } else {
            std.debug.print("CUDA Error Code: {d}\n", .{err});
        }
        return error.CudaExecutionFailed;
    }
}

test "test" {
    const allocator = std.testing.allocator;

    // 1. 定义测试矩阵的维度
    const n: u32 = 128;
    const d: u32 = 256;

    // 2. 在 Host (CPU) 上分配内存
    const x = try allocator.alloc(f32, d);
    defer allocator.free(x);
    const w = try allocator.alloc(f32, n * d);
    defer allocator.free(w);

    const xout_cpu = try allocator.alloc(f32, n);
    defer allocator.free(xout_cpu);
    const xout_gpu = try allocator.alloc(f32, n);
    defer allocator.free(xout_gpu);

    // 3. 填充随机初始数据
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (x) |*val| val.* = random.float(f32) * 2.0 - 1.0;
    for (w) |*val| val.* = random.float(f32) * 2.0 - 1.0;
    @memset(xout_cpu, 0.0);
    @memset(xout_gpu, 0.0);

    // ==========================================
    // 4. CUDA 初始化与上下文准备
    // ==========================================
    try checkCuda(cuda.cuInit(0));
    var device: cuda.CUdevice = 0;
    try checkCuda(cuda.cuDeviceGet(&device, 0));

    var ctx: cuda.CUcontext = undefined;
    try checkCuda(cuda.cuCtxCreate(&ctx, null, 0, device));
    defer _ = cuda.cuCtxDestroy(ctx);

    // 5. 加载我们在 build.zig 中刚刚编译出来的 PTX 模块
    var module: cuda.CUmodule = undefined;
    try checkCuda(cuda.cuModuleLoad(&module, "kernel.ptx"));

    // 获取 Kernel 函数指针
    var kernel: cuda.CUfunction = undefined;
    try checkCuda(cuda.cuModuleGetFunction(&kernel, module, "matmul_kernel"));

    // ==========================================
    // 6. Device (GPU) 显存分配
    // ==========================================
    var d_x: cuda.CUdeviceptr = 0;
    var d_w: cuda.CUdeviceptr = 0;
    var d_xout: cuda.CUdeviceptr = 0;

    try checkCuda(cuda.cuMemAlloc_v2(&d_x, d * @sizeOf(f32)));
    defer _ = cuda.cuMemFree_v2(d_x);
    try checkCuda(cuda.cuMemAlloc_v2(&d_w, n * d * @sizeOf(f32)));
    defer _ = cuda.cuMemFree_v2(d_w);
    try checkCuda(cuda.cuMemAlloc_v2(&d_xout, n * @sizeOf(f32)));
    defer _ = cuda.cuMemFree_v2(d_xout); // Added missing defer

    // ==========================================
    // 7. Host to Device 数据拷贝
    // ==========================================
    try checkCuda(cuda.cuMemcpyHtoD_v2(d_x, x.ptr, d * @sizeOf(f32)));
    try checkCuda(cuda.cuMemcpyHtoD_v2(d_w, w.ptr, n * d * @sizeOf(f32)));

    // ==========================================
    // 8. 设置参数并启动 Kernel
    // ==========================================
    // Zig requires variables to take pointers for the anyopaque array
    var n_arg = n;
    var d_arg = d;

    var args = [_]?*anyopaque{
        @ptrCast(&d_xout),
        @ptrCast(&d_x),
        @ptrCast(&d_w),
        @ptrCast(&n_arg),
        @ptrCast(&d_arg),
    };

    // Calculate grid and block dimensions
    const block_size = 256;
    const grid_size = (n + block_size - 1) / block_size;

    try checkCuda(cuda.cuLaunchKernel(kernel, grid_size, 1, 1, // gridDimX, Y, Z
        block_size, 1, 1, // blockDimX, Y, Z
        0, null, // sharedMemBytes, hStream
        &args[0], // kernelParams
        null // extra
    ));

    // Wait for the GPU to finish
    try checkCuda(cuda.cuCtxSynchronize());

    // ==========================================
    // 9. Device to Host 数据拷贝
    // ==========================================
    try checkCuda(cuda.cuMemcpyDtoH_v2(xout_gpu.ptr, d_xout, n * @sizeOf(f32)));

    // ==========================================
    // 10. CPU 端验证结果
    // ==========================================
    matrix.matmul(xout_cpu, x, w, n, d);

    // 比较 GPU 和 CPU 的计算结果
    var passed = true;
    for (0..n) |i| {
        const diff = @abs(xout_cpu[i] - xout_gpu[i]);
        if (diff > 1e-5) {
            std.debug.print("Mismatch at index {d}: CPU={d}, GPU={d}\n", .{ i, xout_cpu[i], xout_gpu[i] });
            passed = false;
            break;
        }
    }

    if (passed) {
        std.debug.print("Matrix multiplication test passed successfully!\n", .{});
    } else {
        return error.TestFailed;
    }
}
