const std = @import("std");
const debug = std.debug.print;

pub fn matmul(xout: []f32, x: []f32, w: []f32, n: usize, d: usize) void {
    // 编译期确定向量宽度：ARM NEON=4, AVX2=8, AVX-512=16
    const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
    const Vec = @Vector(vec_len, f32);

    // 展开因子：打破数据依赖链，提升 ILP（Instruction Level Parallelism）
    // batch 也是编译期常量
    const unroll = 4;
    const batch = vec_len * unroll;

    for (0..n) |i| {
        var sum0: Vec = @splat(0.0);
        var sum1: Vec = @splat(0.0);
        var sum2: Vec = @splat(0.0);
        var sum3: Vec = @splat(0.0);

        var val: f32 = 0.0;
        const w_row = w[i * d .. (i + 1) * d];
        var j: usize = 0;

        // 主循环：每次处理 unroll * vec_len 个元素
        // 偏移量全部由编译期常量 vec_len 推导，自动适配目标平台
        while (j + batch <= d) : (j += batch) {
            const x0: Vec = x[j + 0 * vec_len ..][0..vec_len].*;
            const x1: Vec = x[j + 1 * vec_len ..][0..vec_len].*;
            const x2: Vec = x[j + 2 * vec_len ..][0..vec_len].*;
            const x3: Vec = x[j + 3 * vec_len ..][0..vec_len].*;

            const w0: Vec = w_row[j + 0 * vec_len ..][0..vec_len].*;
            const w1: Vec = w_row[j + 1 * vec_len ..][0..vec_len].*;
            const w2: Vec = w_row[j + 2 * vec_len ..][0..vec_len].*;
            const w3: Vec = w_row[j + 3 * vec_len ..][0..vec_len].*;

            // ==========================================
            // 优化：使用 @mulAdd 强制编译器生成 FMA 指令
            // ==========================================
            sum0 = @mulAdd(Vec, x0, w0, sum0);
            sum1 = @mulAdd(Vec, x1, w1, sum1);
            sum2 = @mulAdd(Vec, x2, w2, sum2);
            sum3 = @mulAdd(Vec, x3, w3, sum3);
        }

        // 合并 4 个累加器，再 reduce 成标量
        sum0 += sum1;
        sum2 += sum3;
        sum0 += sum2;
        val = @reduce(.Add, sum0);

        // 中间尾部：仍有整向量可处理（< batch 但 >= vec_len 个元素）
        var tail_sum: Vec = @splat(0.0);
        while (j + vec_len <= d) : (j += vec_len) {
            const xv: Vec = x[j..][0..vec_len].*;
            const wv: Vec = w_row[j..][0..vec_len].*;
            // 优化：对剩余的整向量块同样使用 @mulAdd 累加到一个本地向量寄存器上
            tail_sum = @mulAdd(Vec, xv, wv, tail_sum);
        }
        // 将中间尾部的累加器归约到标量结果中
        val += @reduce(.Add, tail_sum);

        // 最终尾部：不足一个向量的剩余元素（< vec_len 个）
        while (j < d) : (j += 1) {
            // 优化：即使是标量也显式使用 @mulAdd，底层会映射为标量 FMA (如 vfmadd132ss)
            val = @mulAdd(f32, w_row[j], x[j], val);
        }

        xout[i] = val;
    }
}

pub fn softmax(x: []f32) void {
    if (x.len == 0) return;

    const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
    const Vec = @Vector(vec_len, f32);

    var j: usize = 0;

    // ==========================================
    // 阶段 1：寻找最大值 (Find Max)
    // ==========================================
    // 初始化为负无穷大
    var v_max: Vec = @splat(-std.math.inf(f32));

    while (j + vec_len <= x.len) : (j += vec_len) {
        const vx: Vec = x[j..][0..vec_len].*;
        // 使用 @select 进行向量化的逐元素比较，提取最大值
        v_max = @select(f32, vx > v_max, vx, v_max);
    }

    // 将向量中的元素归约出一个标量最大值
    var max_val = @reduce(.Max, v_max);

    // 尾部处理
    while (j < x.len) : (j += 1) {
        if (x[j] > max_val) max_val = x[j];
    }

    // ==========================================
    // 阶段 2：计算指数并求和 (Exp and Sum)
    // ==========================================
    j = 0;
    var v_sum: Vec = @splat(0.0);
    const v_max_splat: Vec = @splat(max_val);

    while (j + vec_len <= x.len) : (j += vec_len) {
        const vx: Vec = x[j..][0..vec_len].*;

        // 向量化减法与指数运算 (@exp 内部会自动映射到 LLVM 的向量化指令)
        const v_exp = @exp(vx - v_max_splat);

        // 写回并累加
        x[j..][0..vec_len].* = v_exp;
        v_sum += v_exp;
    }

    var sum = @reduce(.Add, v_sum);

    // 尾部处理
    while (j < x.len) : (j += 1) {
        const exp_val = std.math.exp(x[j] - max_val);
        x[j] = exp_val;
        sum += exp_val;
    }

    // ==========================================
    // 阶段 3：归一化 (Normalize)
    // ==========================================
    j = 0;
    // 性能关键：将除法转换为乘法
    const inv_sum = 1.0 / sum;
    const v_inv_sum: Vec = @splat(inv_sum);

    while (j + vec_len <= x.len) : (j += vec_len) {
        const vx: Vec = x[j..][0..vec_len].*;
        // 向量乘法显著快于向量除法
        x[j..][0..vec_len].* = vx * v_inv_sum;
    }

    // 尾部处理
    while (j < x.len) : (j += 1) {
        x[j] *= inv_sum;
    }
}

// RoPE Relative Positional Encoding
pub fn rope(
    query: []f32,
    k_target: []f32,
    dim: usize,
    head_size: usize,
    pos: usize,
    kv_dim: usize,
    frequency_base: f32,
) void {
    var i: usize = 0;
    while (i < dim) : (i += 2) {
        const head_dim = i % head_size;
        const freq = 1.0 / std.math.pow(
            f32,
            frequency_base,
            @as(f32, @floatFromInt(head_dim)) / @as(f32, @floatFromInt(head_size)),
        );
        const val = @as(f32, @floatFromInt(pos)) * freq;
        const fcr = std.math.cos(val);
        const fci = std.math.sin(val);

        // How many vectors to rotate? (query is always rotated, key depends on kv_dim)
        const rotn: usize = if (i < kv_dim) 2 else 1;

        for (0..rotn) |v_idx| {
            // v_idx 0 = query, v_idx 1 = key (inside cache)
            const vec = if (v_idx == 0) query else k_target;

            const v0 = vec[i];
            const v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

pub fn root_mean_square_normalization(
    out: []f32,
    x: []const f32,
    weight: []const f32,
    epsilon: f32,
) void {
    // 1. 自动获取当前架构的最佳向量长度（如果不支持 SIMD 则退化为 8）
    const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
    const Vec = @Vector(vec_len, f32);

    var ss: f32 = 0.0;
    var j: usize = 0;

    // ==========================================
    // 阶段 1：计算平方和 (Sum of Squares)
    // ==========================================
    var vec_ss: Vec = @splat(0.0);

    // 主循环：每次并行计算 vec_len 个平方和
    while (j + vec_len <= x.len) : (j += vec_len) {
        // 从切片加载数据为向量
        const vx: Vec = x[j..][0..vec_len].*;
        // SIMD 乘法与累加
        vec_ss += vx * vx;
    }

    // 将 SIMD 向量内的所有元素相加（归约）成一个标量
    ss = @reduce(.Add, vec_ss);

    // 尾部处理：处理末尾凑不够一个 vec_len 的剩余元素
    while (j < x.len) : (j += 1) {
        ss += x[j] * x[j];
    }

    // ==========================================
    // 阶段 2：计算归一化标量 (计算方差倒数)
    // ==========================================
    ss /= @as(f32, @floatFromInt(x.len));
    ss += epsilon; // 防止除零的 epsilon
    ss = 1.0 / std.math.sqrt(ss);

    // ==========================================
    // 阶段 3：归一化并应用缩放权重 (Normalize and Scale)
    // ==========================================
    // 把标量 ss 广播 (splat) 到整个向量中，避免在循环里重复标量乘法
    const v_ss: Vec = @splat(ss);
    j = 0; // 重置索引

    // 主循环：每次并行处理 vec_len 个元素的缩放
    while (j + vec_len <= x.len) : (j += vec_len) {
        const vx: Vec = x[j..][0..vec_len].*;
        const vw: Vec = weight[j..][0..vec_len].*;

        // 核心 SIMD 运算：o = weight * (ss * x)
        const v_out = vw * (v_ss * vx);

        // 将结果写回输出数组 o
        out[j..][0..vec_len].* = v_out;
    }

    // 尾部处理
    while (j < x.len) : (j += 1) {
        out[j] = weight[j] * (ss * x[j]);
    }
}
