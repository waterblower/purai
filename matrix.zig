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

            sum0 += x0 * w0;
            sum1 += x1 * w1;
            sum2 += x2 * w2;
            sum3 += x3 * w3;
        }

        // 合并 4 个累加器，再 reduce 成标量
        sum0 += sum1;
        sum2 += sum3;
        sum0 += sum2;
        val = @reduce(.Add, sum0);

        // 中间尾部：仍有整向量可处理（< batch 但 >= vec_len 个元素）
        while (j + vec_len <= d) : (j += vec_len) {
            const xv: Vec = x[j..][0..vec_len].*;
            const wv: Vec = w_row[j..][0..vec_len].*;
            val += @reduce(.Add, xv * wv);
        }

        // 最终尾部：不足一个向量的剩余元素（< vec_len 个）
        while (j < d) : (j += 1) {
            val += w_row[j] * x[j];
        }

        xout[i] = val;
    }
}
