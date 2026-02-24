const gguf = @import("./gguf.zig");
const std = @import("std");
const debug = std.debug.print;

test "serialize" {
    const allocator = std.testing.allocator;

    const path1 = "models/llama-2-7b-chat.Q2_K.gguf";

    const model = try gguf.Read(allocator, path1);
    defer model.deinit();

    debug("{f}\n", .{model});

    try model.serialize("models/test-out.gguf");
    try expectFilesEqual(path1, "models/test-out.gguf");
}

test "quantize_to_Q4_0" {
    const allocator = std.testing.allocator;

    const path1 = "./models/GLM-OCR-f16.gguf";
    const path2 = "./models/GLM-OCR-Q4_0.gguf";

    const model = try gguf.Read(allocator, path1);
    defer model.deinit();

    const model2 = try model.quantize_to_Q4_0(allocator);
    defer model2.deinit();
    try model2.serialize(path2);
}

test "compare" {
    const allocator = std.testing.allocator;

    const path1 = "./models/Pure-Q4_0.gguf";
    const path2 = "./models/GLM-OCR-Q4_0.gguf";

    const model1 = try gguf.Read(allocator, path1);
    defer model1.deinit();

    const model2 = try gguf.Read(allocator, path2);
    defer model2.deinit();

    // 设定浮点数的最大容忍误差为 1e-5 (即 0.00001)
    const is_identical = isEquivalentModel(model1, model2, 1e-5);

    if (is_identical) {
        std.debug.print("两个模型描述了完全相同的网络架构。\n", .{});
    } else {
        std.debug.print("发现差异，请查看上方的具体 Mismatch 报错。\n", .{});
    }
}

/// 逐字节比较两个文件是否完全一致
fn expectFilesEqual(path1: []const u8, path2: []const u8) !void {
    const file1 = try std.fs.cwd().openFile(path1, .{});
    defer file1.close();

    const file2 = try std.fs.cwd().openFile(path2, .{});
    defer file2.close();

    const stat1 = try file1.stat();
    const stat2 = try file2.stat();

    // 1. 第一道防线：文件大小必须一致
    try std.testing.expectEqual(stat1.size, stat2.size);

    // 2. 第二道防线：分块（Chunk）读取并对比
    // 使用一页大小的栈内存，零堆分配，极致速度
    var buf1: [std.heap.pageSize()]u8 = undefined;
    var buf2: [std.heap.pageSize()]u8 = undefined;

    var total_bytes_read: usize = 0;
    while (true) {
        const n1 = try file1.read(&buf1);
        const n2 = try file2.read(&buf2);

        // 读取的字节数必须相同
        try std.testing.expectEqual(n1, n2);

        if (n1 == 0) break; // 遇到 EOF (文件末尾)，对比结束

        // 核心断言：对比当前数据块是否完全一致
        // 使用 expectEqualSlices 极其重要！如果失败，它会精准打印出是哪一个 byte 不一样
        try std.testing.expectEqual(buf1, buf2);

        total_bytes_read += n1;
    }

    // 3. 最终确认：实际读取的字节数等于文件大小
    try std.testing.expectEqual(stat1.size, total_bytes_read);
}

/// 比较两个 GgufContext 是否描述了同一个模型架构
/// epsilon: 浮点数比较的容差范围 (例如 1e-5)
pub fn isEquivalentModel(ctx_a: *const gguf.GgufContext, ctx_b: *const gguf.GgufContext, epsilon: f64) bool {
    // 1. 检查数量是否一致
    if (ctx_a.kv_map.count() != ctx_b.kv_map.count()) {
        std.debug.print("Mismatch: KV count differs ({d} vs {d})\n", .{ ctx_a.kv_map.count(), ctx_b.kv_map.count() });
        return false;
    }
    if (ctx_a.tensor_map.count() != ctx_b.tensor_map.count()) {
        std.debug.print("Mismatch: Tensor count differs ({d} vs {d})\n", .{ ctx_a.tensor_map.count(), ctx_b.tensor_map.count() });
        return false;
    }

    // 2. 逐一比对 Metadata KV Map
    var kv_it = ctx_a.kv_map.iterator();
    while (kv_it.next()) |entry_a| {
        const key = entry_a.key_ptr.*;
        const val_a = entry_a.value_ptr.*;

        const val_b_ptr = ctx_b.kv_map.getPtr(key) orelse {
            std.debug.print("Mismatch: Missing metadata key in ctx_b: {s}\n", .{key});
            return false;
        };
        const val_b = val_b_ptr.*;

        if (!compareGgufValues(val_a, val_b, epsilon)) {
            std.debug.print("Mismatch: Metadata value differs for key: {s}\n", .{key});
            return false;
        }
    }

    // 3. 逐一比对 Tensor Map (名称、类型、维度)
    var tensor_it = ctx_a.tensor_map.iterator();
    while (tensor_it.next()) |entry_a| {
        const name = entry_a.key_ptr.*;
        const info_a = entry_a.value_ptr.*;

        const info_b_ptr = ctx_b.tensor_map.getPtr(name) orelse {
            std.debug.print("Mismatch: Missing tensor in ctx_b: {s}\n", .{name});
            return false;
        };
        const info_b = info_b_ptr.*;

        if (info_a.type != info_b.type) {
            std.debug.print("Mismatch: Tensor type differs for {s} ({s} vs {s})\n", .{ name, @tagName(info_a.type), @tagName(info_b.type) });
            return false;
        }

        if (info_a.n_dims != info_b.n_dims) {
            std.debug.print("Mismatch: Tensor dimension count differs for {s} ({d} vs {d})\n", .{ name, info_a.n_dims, info_b.n_dims });
            return false;
        }

        // 仅比对有效的维度范围
        for (0..info_a.n_dims) |d| {
            if (info_a.dims[d] != info_b.dims[d]) {
                std.debug.print("Mismatch: Tensor dimensions differ for {s} at dim {d}\n", .{ name, d });
                return false;
            }
        }
    }

    return true;
}

/// 内部辅助函数：比对单个 GgufValue
fn compareGgufValues(a: gguf.GgufValue, b: gguf.GgufValue, epsilon: f64) bool {
    const tag_a = std.meta.activeTag(a);
    const tag_b = std.meta.activeTag(b);

    if (tag_a != tag_b) return false;

    return switch (a) {
        .UINT8 => |v| v == b.UINT8,
        .INT8 => |v| v == b.INT8,
        .UINT16 => |v| v == b.UINT16,
        .INT16 => |v| v == b.INT16,
        .UINT32 => |v| v == b.UINT32,
        .INT32 => |v| v == b.INT32,
        .UINT64 => |v| v == b.UINT64,
        .INT64 => |v| v == b.INT64,
        .BOOL => |v| v == b.BOOL,
        .STRING => |v| std.mem.eql(u8, v, b.STRING),
        .FLOAT32 => |v| @abs(v - b.FLOAT32) <= @as(f32, @floatCast(epsilon)),
        .FLOAT64 => |v| @abs(v - b.FLOAT64) <= epsilon,
        .ARRAY => |arr_a| {
            const arr_b = b.ARRAY;
            if (arr_a.type != arr_b.type or arr_a.len != arr_b.len) return false;

            // 针对浮点数数组，需要按浮点数解析并带容差比对
            if (arr_a.type == .FLOAT32 or arr_a.type == .FLOAT64) {
                return compareFloatArrays(arr_a, arr_b, epsilon);
            } else {
                // 对于其他类型数组（如整数、字符串的内存布局），直接使用二进制比对
                return std.mem.eql(u8, arr_a.data, arr_b.data);
            }
        },
    };
}

/// 内部辅助函数：专门用于比对浮点数数组
fn compareFloatArrays(arr_a: anytype, arr_b: anytype, epsilon: f64) bool {
    var i: usize = 0;
    if (arr_a.type == .FLOAT32) {
        while (i < arr_a.len) : (i += 1) {
            const f_a = @as(f32, @bitCast(std.mem.readInt(u32, arr_a.data[i * 4 ..][0..4], .little)));
            const f_b = @as(f32, @bitCast(std.mem.readInt(u32, arr_b.data[i * 4 ..][0..4], .little)));
            if (@abs(f_a - f_b) > @as(f32, @floatCast(epsilon))) return false;
        }
    } else if (arr_a.type == .FLOAT64) {
        while (i < arr_a.len) : (i += 1) {
            const f_a = @as(f64, @bitCast(std.mem.readInt(u64, arr_a.data[i * 8 ..][0..8], .little)));
            const f_b = @as(f64, @bitCast(std.mem.readInt(u64, arr_b.data[i * 8 ..][0..8], .little)));
            if (@abs(f_a - f_b) > epsilon) return false;
        }
    }
    return true;
}
