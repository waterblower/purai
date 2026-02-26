// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
// https://huggingface.co/docs/hub/en/gguf
const std = @import("std");
const Allocator = std.mem.Allocator;
const debug = std.debug.print;
const mem = std.mem;

// GGUF Magic Number: "GGUF"
const GGUF_MAGIC = 0x46554747;

const GGML_MAX_DIMS = 8;

pub const GgufType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    MXFP4 = 39, // MXFP4 (1 block)
    COUNT = 40,
};

pub const GgufMetadataValueType = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// 简单的 Tagged Union 来存储元数据值
pub const Value = union(GgufMetadataValueType) {
    UINT8: u8,
    INT8: i8,
    UINT16: u16,
    INT16: i16,
    UINT32: u32,
    INT32: i32,
    FLOAT32: f32,
    BOOL: bool,
    STRING: []const u8,
    // 数组比较复杂，我们暂存其原始数据的切片和类型，用到时再解析
    ARRAY: struct { len: u64, type: GgufMetadataValueType, data: []const u8 },
    UINT64: u64,
    INT64: i64,
    FLOAT64: f64,
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: [GGML_MAX_DIMS]u64,
    n_dims: u32,
    type: GgufType,
    offset: u64, // 相对 tensor_data_base 的偏移
    data: [*]const u8, // 指向数据的直接指针
};

pub fn Read(allocator: Allocator, path: []const u8) !*GgufContext {
    // 1. 读取整个文件
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const file_size = stat.size;

    // 分配对齐内存
    const buffer = try allocator.alignedAlloc(u8, mem.Alignment.fromByteUnits(4096), file_size);
    errdefer allocator.free(buffer);

    const bytes_read = try file.readAll(buffer);
    if (bytes_read != file_size) return error.IncompleteRead;

    // 2. 创建上下文
    const ctx = try allocator.create(GgufContext);
    ctx.* = .{
        .allocator = allocator,
        .data = buffer,
        .version = 0,
        .tensor_count = 0,
        .kv_count = 0,
        .kv_map = std.StringArrayHashMap(Value).init(allocator),
        .tensor_map = std.StringArrayHashMap(TensorInfo).init(allocator),
        .tensor_data_base = undefined,
    };

    const self = ctx;
    var cursor: usize = 0;
    // Header
    const magic = std.mem.readInt(u32, self.data[0..4], .little);
    if (magic != GGUF_MAGIC) return error.InvalidMagic;
    cursor += 4;

    self.version = std.mem.readInt(u32, self.data[cursor..][0..4], .little);
    cursor += 4;

    self.tensor_count = std.mem.readInt(u64, self.data[cursor..][0..8], .little);
    cursor += 8;

    self.kv_count = std.mem.readInt(u64, self.data[cursor..][0..8], .little);
    cursor += 8;

    // Metadata KV Pairs
    var i: usize = 0;
    while (i < self.kv_count) : (i += 1) {
        const key = try readString(self.data, &cursor);
        const val_type = @as(GgufMetadataValueType, @enumFromInt(std.mem.readInt(u32, self.data[cursor..][0..4], .little)));
        cursor += 4;

        const val = try readValue(self.data, &cursor, val_type);
        try self.kv_map.put(key, val);
    }

    // Tensor Infos
    i = 0;
    while (i < self.tensor_count) : (i += 1) {
        const name = try readString(self.data, &cursor);
        const n_dims = std.mem.readInt(u32, self.data[cursor..][0..4], .little);
        cursor += 4;

        // 1 is multiplicative identity
        var dims: [GGML_MAX_DIMS]u64 = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
        var d: usize = 0;
        while (d < n_dims) : (d += 1) {
            dims[d] = std.mem.readInt(u64, self.data[cursor..][0..8], .little);
            cursor += 8;
        }

        const type_val = @as(GgufType, @enumFromInt(std.mem.readInt(u32, self.data[cursor..][0..4], .little)));
        cursor += 4;

        const offset = std.mem.readInt(u64, self.data[cursor..][0..8], .little);
        cursor += 8;

        try self.tensor_map.put(name, .{
            .name = name,
            .dims = dims,
            .n_dims = n_dims,
            .type = type_val,
            .offset = offset,
            .data = undefined, // Will be set after alignment
        });
    }

    // Alignment padding
    // GGUF spec: tensor data is aligned to ALIGNMENT (default 32)
    // 这个 alignment 值通常在 metadata "general.alignment" 中，找不到则默认 32
    var alignment: usize = 32;
    if (self.kv_map.get("general.alignment")) |val| {
        switch (val) {
            .UINT32 => |v| alignment = v,
            .UINT64 => |v| alignment = @intCast(v),
            .INT32 => |v| alignment = @intCast(v),
            else => {},
        }
    }

    // Align cursor
    const padding = (alignment - (cursor % alignment)) % alignment;
    cursor += padding;

    self.tensor_data_base = self.data[cursor..].ptr;

    // Fixup tensor data pointers
    var it = self.tensor_map.iterator();
    while (it.next()) |entry| {
        const info = entry.value_ptr;
        info.data = self.tensor_data_base + info.offset;
    }
    return ctx;
}

pub const GgufContext = struct {
    allocator: Allocator,
    data: []align(4096) u8, // 整个文件的内存映射/缓冲区

    version: u32,
    tensor_count: u64,
    kv_count: u64,

    // 核心查找表
    kv_map: std.StringArrayHashMap(Value),
    tensor_map: std.StringArrayHashMap(TensorInfo),

    // 张量数据块的起始位置（绝对指针）
    tensor_data_base: [*]const u8,

    pub fn getU32(model: *const GgufContext, key: []const u8) !u32 {
        const val = model.kv_map.get(key) orelse return error.MissingMetadata;
        return switch (val) {
            .UINT32 => |v| v,
            .UINT64 => |v| @intCast(v),
            else => return error.TypeMismatch,
        };
    }

    pub fn getF32(model: *const GgufContext, key: []const u8) !f32 {
        const val = model.kv_map.get(key) orelse return error.MissingMetadata;
        return switch (val) {
            .FLOAT32 => |v| v,
            else => return error.TypeMismatch,
        };
    }

    // 辅助函数：安全抓取张量
    pub fn getTensor(model: *const GgufContext, name: []const u8) !TensorInfo {
        return model.tensor_map.get(name) orelse {
            std.log.err("Missing critical tensor: {s}\n", .{name});
            return error.MissingTensor;
        };
    }

    pub fn deinit(self: *GgufContext) void {
        self.kv_map.deinit();
        self.tensor_map.deinit();
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }

    pub fn serialize(self: *const GgufContext, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // 准备用于存放转换后字节的栈内存数组
        var buf8: [1]u8 = undefined;
        var buf32: [4]u8 = undefined;
        var buf64: [8]u8 = undefined;

        // 1. Write Header
        std.mem.writeInt(u32, &buf32, GGUF_MAGIC, .little);
        try file.writeAll(&buf32);

        std.mem.writeInt(u32, &buf32, self.version, .little);
        try file.writeAll(&buf32);

        std.mem.writeInt(u64, &buf64, self.tensor_count, .little);
        try file.writeAll(&buf64);

        std.mem.writeInt(u64, &buf64, self.kv_count, .little);
        try file.writeAll(&buf64);

        // 2. Write Metadata KV Pairs
        var kv_it = self.kv_map.iterator();
        while (kv_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const val = entry.value_ptr.*;

            try writeString(file, key);

            const val_type = std.meta.activeTag(val);
            std.mem.writeInt(u32, &buf32, @intFromEnum(val_type), .little);
            try file.writeAll(&buf32);

            try writeValue(file, val);
        }

        // 3. Write Tensor Infos
        var tensor_it = self.tensor_map.iterator();
        while (tensor_it.next()) |entry| {
            const info = entry.value_ptr.*;

            try writeString(file, info.name);

            std.mem.writeInt(u32, &buf32, info.n_dims, .little);
            try file.writeAll(&buf32);

            var d: usize = 0;
            while (d < info.n_dims) : (d += 1) {
                std.mem.writeInt(u64, &buf64, info.dims[d], .little);
                try file.writeAll(&buf64);
            }

            std.mem.writeInt(u32, &buf32, @intFromEnum(info.type), .little);
            try file.writeAll(&buf32);

            std.mem.writeInt(u64, &buf64, info.offset, .little);
            try file.writeAll(&buf64);
        }

        // 4. Alignment padding
        var alignment: usize = 32;
        if (self.kv_map.get("general.alignment")) |val| {
            switch (val) {
                .UINT32 => |v| alignment = v,
                .UINT64 => |v| alignment = @intCast(v),
                .INT32 => |v| alignment = @intCast(v),
                else => {},
            }
        }

        const current_pos = try file.getPos();
        const padding = (alignment - (current_pos % alignment)) % alignment;

        buf8[0] = 0; // 填充字节为 0
        var p: usize = 0;
        while (p < padding) : (p += 1) {
            try file.writeAll(&buf8);
        }

        // 5. Write Tensor Data Block
        const data_start_offset = @intFromPtr(self.tensor_data_base) - @intFromPtr(self.data.ptr);
        const tensor_data_slice = self.data[data_start_offset..];
        try file.writeAll(tensor_data_slice);
    }

    pub fn format(
        self: *const GgufContext,
        writer: anytype,
    ) !void {

        // A pre-filled buffer of spaces we can slice for padding
        const padding = " " ** 256;

        // 1. Header Summary
        try writer.print("GGUF Context [v{d}]\n", .{self.version});
        try writer.print("==============================================================\n", .{});
        const size_mb = @as(f64, @floatFromInt(self.data.len)) / (1024.0 * 1024.0);
        try writer.print("File Size: {d:.2} MB | Tensors: {d} | Metadata: {d}\n", .{ size_mb, self.tensor_count, self.kv_count });

        // 2. Metadata Section (KV Pairs)
        try writer.print("\n[Metadata]\n", .{});
        var kv_it = self.kv_map.iterator();
        while (kv_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const val = entry.value_ptr.*;

            // Key column (padding 35 chars)
            try writer.print("  {s:<35} = ", .{key});

            // Value formatting logic
            switch (val) {
                .STRING => |s| {
                    if (s.len > 50) {
                        try writer.print("\"{s}...\" (len={d})", .{ s[0..47], s.len });
                    } else {
                        try writer.print("\"{s}\"", .{s});
                    }
                },
                .ARRAY => |arr| {
                    // 核心需求：只打印维度/长度，不打印内容
                    try writer.print("[Array<{s}>, len={d}]", .{ @tagName(arr.type), arr.len });
                },
                .FLOAT32 => |v| try writer.print("{d:.6}", .{v}),
                .FLOAT64 => |v| try writer.print("{d:.6}", .{v}),
                // 使用内联 switch 处理其他标量类型
                else => |v| {
                    switch (val) {
                        .UINT8, .INT8, .UINT16, .INT16, .UINT32, .INT32, .UINT64, .INT64, .BOOL => try writer.print("{any}", .{v}),
                        else => unreachable,
                    }
                },
            }
            try writer.writeAll("\n");
        }
        // --- 2. Tensor Section: Dynamic Calculation ---
        try writer.print("\n[Tensors]\n", .{});

        // PASS 1: Calculate Widths
        var max_name_len: usize = 4;
        var max_type_len: usize = 4;

        var calc_it = self.tensor_map.iterator();
        while (calc_it.next()) |entry| {
            const t = entry.value_ptr;
            if (t.name.len > max_name_len) max_name_len = t.name.len;
            const type_str = @tagName(t.type);
            if (type_str.len > max_type_len) max_type_len = type_str.len;
        }

        max_name_len += 2;
        max_type_len += 2;

        // PASS 2: Print Headers
        // Usage: padding[0 .. needed_spaces]
        try writer.print("  {s}", .{"NAME"});
        try writer.writeAll(padding[0 .. max_name_len - 4]);

        try writer.print("{s}", .{"TYPE"});
        try writer.writeAll(padding[0 .. max_type_len - 4]);

        try writer.print("{s}\n", .{"SHAPE"});

        // Print Divider
        try writer.print("  ", .{});
        // Dynamically print dashes using the same slicing trick if needed,
        // or just a loop for dashes since they aren't in our 'padding' constant.
        for (0..(max_name_len - 1)) |_| try writer.writeAll("-");
        try writer.print(" ", .{});
        for (0..(max_type_len - 1)) |_| try writer.writeAll("-");
        try writer.print(" ---------------\n", .{});

        // PASS 3: Print Data
        var print_it = self.tensor_map.iterator();
        while (print_it.next()) |entry| {
            const t = entry.value_ptr;
            const type_str = @tagName(t.type);

            try writer.print("  {s}", .{t.name});
            try writer.writeAll(padding[0 .. max_name_len - t.name.len]);

            try writer.print("{s}", .{type_str});
            try writer.writeAll(padding[0 .. max_type_len - type_str.len]);

            try writer.print("[", .{});
            for (0..t.n_dims) |i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{t.dims[i]});
            }
            try writer.print("]\n", .{});
        }
        try writer.print("\n", .{});
    }

    pub fn quantize_to_Q4_0(old_ctx: *const GgufContext, allocator: std.mem.Allocator) !*GgufContext {
        // 1. 获取张量对齐参数 (Alignment)，通常为 32 字节
        var alignment: usize = 32;
        if (old_ctx.kv_map.get("general.alignment")) |val| {
            switch (val) {
                .UINT32 => |v| alignment = v,
                .UINT64 => |v| alignment = @intCast(v),
                .INT32 => |v| alignment = @intCast(v),
                else => {},
            }
        }

        // 2. 预扫描：计算量化后所有张量所需的总字节数
        var new_total_size: usize = 0;
        var it = old_ctx.tensor_map.iterator();
        while (it.next()) |entry| {
            const info = entry.value_ptr.*;
            var elements: usize = 1;
            for (0..info.n_dims) |d| elements *= info.dims[d];

            // 计算对齐填充
            const padding = (alignment - (new_total_size % alignment)) % alignment;
            new_total_size += padding;

            // 仅对元素数量能被32整除，且维度大于1的矩阵（通常是权重而非偏置）进行量化
            if ((info.type == .F32 or info.type == .F16) and
                info.n_dims > 1 and
                elements % 32 == 0)
            {
                new_total_size += (elements / 32) * 18; // Q4_0 的体积
            } else {
                // 保留原格式的体积 (假设当前仅有 F32 或 F16)
                if (info.type == .F32) {
                    new_total_size += elements * 4;
                } else if (info.type == .F16) {
                    new_total_size += elements * 2;
                } else {
                    std.debug.print(
                        "failed to quantize: requantizing from type {any} is disabled\n",
                        .{info.type},
                    );
                    return error.UnsupportedTypeForCopy;
                }
            }
        }

        // 3. 分配新的张量数据缓冲区
        const new_data = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), new_total_size);
        errdefer allocator.free(new_data);

        // 4. 初始化新的 GgufContext
        const new_ctx = try allocator.create(GgufContext);
        new_ctx.* = .{
            .allocator = allocator,
            .data = new_data, // 注意：这里的 data 仅存放张量二进制块，序列化函数支持这种模式
            .version = old_ctx.version,
            .tensor_count = old_ctx.tensor_count,
            .kv_count = old_ctx.kv_count,
            .kv_map = try old_ctx.kv_map.clone(), // 浅拷贝键值对
            .tensor_map = std.StringArrayHashMap(TensorInfo).init(allocator),
            .tensor_data_base = new_data.ptr,
        };
        errdefer new_ctx.deinit();

        // 更新文件类型元数据 (2 = MOSTLY_Q4_0)
        if (new_ctx.kv_map.contains("general.file_type")) {
            try new_ctx.kv_map.put("general.file_type", .{ .UINT32 = 2 });
        }

        // 5. 执行数据映射与物理量化操作
        var current_offset: usize = 0;
        it = old_ctx.tensor_map.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            const old_info = entry.value_ptr.*;

            var elements: usize = 1;
            for (0..old_info.n_dims) |d| elements *= old_info.dims[d];

            // 重新计算并应用新文件中的物理对齐
            const padding = (alignment - (current_offset % alignment)) % alignment;
            // 填充区清零确保文件干净
            @memset(new_data[current_offset .. current_offset + padding], 0);
            current_offset += padding;

            var new_info = old_info;
            new_info.offset = current_offset;
            new_info.data = new_data.ptr + current_offset; // 指向新分配的内存地址

            // 同时处理 F32 和 F16 的量化
            if ((old_info.type == .F32 or old_info.type == .F16) and
                old_info.n_dims > 1 and
                elements % 32 == 0)
            {
                new_info.type = .Q4_0;
                const num_blocks = elements / 32;
                const new_byte_size = num_blocks * 18;

                const dst_q4: [*]BlockQ4_0 = @ptrCast(@alignCast(new_data.ptr + current_offset));

                if (old_info.type == .F32) {
                    // --- 原始数据是 F32 的路径 ---
                    const src_f32: [*]const f32 = @ptrCast(@alignCast(old_info.data));
                    for (0..num_blocks) |b| {
                        const src_block = src_f32[b * 32 ..][0..32];
                        dst_q4[b] = quantizeBlockQ4_0(src_block); // 调用 f32 原生版本
                    }
                } else if (old_info.type == .F16) {
                    // debug("old_info.type == .F16\n", .{});
                    // --- 原始数据是 F16 的路径 ---
                    const src_f16: [*]const f16 = @ptrCast(@alignCast(old_info.data));
                    for (0..num_blocks) |b| {
                        const src_block = src_f16[b * 32 ..][0..32];
                        dst_q4[b] = quantize_block_Q4_0_f16(src_block); // 调用 f16 适配器版本
                    }
                }
                current_offset += new_byte_size;
            } else {
                // 不满足量化条件（如 1D 偏置项），原样拷贝二进制数据
                var byte_size: usize = 0;
                if (old_info.type == .F32) byte_size = elements * 4 else if (old_info.type == .F16) byte_size = elements * 2;

                @memcpy(new_data[current_offset .. current_offset + byte_size], old_info.data[0..byte_size]);
                current_offset += byte_size;
            }

            try new_ctx.tensor_map.put(name, new_info);
        }

        return new_ctx;
    }
};

fn readString(data: []const u8, cursor: *usize) ![]const u8 {
    const len = std.mem.readInt(u64, data[cursor.*..][0..8], .little);
    cursor.* += 8;
    const str = data[cursor.* .. cursor.* + len];
    cursor.* += len;
    return str;
}

fn readValue(data: []const u8, cursor: *usize, type_val: GgufMetadataValueType) !Value {
    switch (type_val) {
        .UINT8 => {
            defer cursor.* += 1;
            return .{ .UINT8 = data[cursor.*] };
        },
        .INT8 => {
            defer cursor.* += 1;
            return .{ .INT8 = @as(i8, @bitCast(data[cursor.*])) };
        },
        .UINT16 => {
            defer cursor.* += 2;
            return .{ .UINT16 = std.mem.readInt(u16, data[cursor.*..][0..2], .little) };
        },
        .INT16 => {
            defer cursor.* += 2;
            return .{ .INT16 = std.mem.readInt(i16, data[cursor.*..][0..2], .little) };
        },
        .UINT32 => {
            defer cursor.* += 4;
            return .{ .UINT32 = std.mem.readInt(u32, data[cursor.*..][0..4], .little) };
        },
        .INT32 => {
            defer cursor.* += 4;
            return .{ .INT32 = std.mem.readInt(i32, data[cursor.*..][0..4], .little) };
        },
        .FLOAT32 => {
            defer cursor.* += 4;
            return .{ .FLOAT32 = @as(f32, @bitCast(std.mem.readInt(u32, data[cursor.*..][0..4], .little))) };
        },
        .UINT64 => {
            defer cursor.* += 8;
            return .{ .UINT64 = std.mem.readInt(u64, data[cursor.*..][0..8], .little) };
        },
        .INT64 => {
            defer cursor.* += 8;
            return .{ .INT64 = std.mem.readInt(i64, data[cursor.*..][0..8], .little) };
        },
        .FLOAT64 => {
            defer cursor.* += 8;
            return .{ .FLOAT64 = @as(f64, @bitCast(std.mem.readInt(u64, data[cursor.*..][0..8], .little))) };
        },
        .BOOL => {
            defer cursor.* += 1;
            return .{ .BOOL = data[cursor.*] != 0 };
        },
        .STRING => {
            return .{ .STRING = try readString(data, cursor) };
        },
        .ARRAY => {
            const type_sub = @as(GgufMetadataValueType, @enumFromInt(std.mem.readInt(u32, data[cursor.*..][0..4], .little)));
            cursor.* += 4;
            const len = std.mem.readInt(u64, data[cursor.*..][0..8], .little);
            cursor.* += 8;
            // 暂时不递归解析数组内容，只记录位置，让调用者按需解析
            // 计算数组字节大小比较麻烦，需要预扫描。
            // 简化起见：我们这里做一个Hack，如果你需要读取数组，需要在此处完善
            // 但对于读取 Config，通常不需要解析很长的数组。
            // *简单实现*：仅支持基础定长类型的数组跳过/读取
            const start = cursor.*;
            try skipArray(data, cursor, type_sub, len);
            return .{
                .ARRAY = .{
                    .type = type_sub,
                    .len = len,
                    .data = data[start..cursor.*],
                },
            };
        },
    }
}

fn skipArray(data: []const u8, cursor: *usize, type_val: GgufMetadataValueType, len: u64) !void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        switch (type_val) {
            .STRING => {
                const str_len = std.mem.readInt(u64, data[cursor.*..][0..8], .little);
                cursor.* += 8 + str_len;
            },
            .ARRAY => return error.NestedArraysNotSupported,
            else => {
                // Fixed size types
                const size: usize = switch (type_val) {
                    .UINT8, .INT8, .BOOL => 1,
                    .UINT16, .INT16 => 2,
                    .UINT32, .INT32, .FLOAT32 => 4,
                    .UINT64, .INT64, .FLOAT64 => 8,
                    else => unreachable,
                };
                cursor.* += size;
            },
        }
    }
}

fn writeString(file: std.fs.File, str: []const u8) !void {
    var buf64: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf64, str.len, .little);
    try file.writeAll(&buf64);
    try file.writeAll(str);
}

fn writeValue(file: std.fs.File, val: Value) !void {
    var buf8: [1]u8 = undefined;
    var buf16: [2]u8 = undefined;
    var buf32: [4]u8 = undefined;
    var buf64: [8]u8 = undefined;

    switch (val) {
        .UINT8 => |v| {
            buf8[0] = v;
            try file.writeAll(&buf8);
        },
        .INT8 => |v| {
            buf8[0] = @bitCast(v);
            try file.writeAll(&buf8);
        },
        .UINT16 => |v| {
            std.mem.writeInt(u16, &buf16, v, .little);
            try file.writeAll(&buf16);
        },
        .INT16 => |v| {
            std.mem.writeInt(i16, &buf16, v, .little);
            try file.writeAll(&buf16);
        },
        .UINT32 => |v| {
            std.mem.writeInt(u32, &buf32, v, .little);
            try file.writeAll(&buf32);
        },
        .INT32 => |v| {
            std.mem.writeInt(i32, &buf32, v, .little);
            try file.writeAll(&buf32);
        },
        .FLOAT32 => |v| {
            std.mem.writeInt(u32, &buf32, @bitCast(v), .little);
            try file.writeAll(&buf32);
        },
        .UINT64 => |v| {
            std.mem.writeInt(u64, &buf64, v, .little);
            try file.writeAll(&buf64);
        },
        .INT64 => |v| {
            std.mem.writeInt(i64, &buf64, v, .little);
            try file.writeAll(&buf64);
        },
        .FLOAT64 => |v| {
            std.mem.writeInt(u64, &buf64, @bitCast(v), .little);
            try file.writeAll(&buf64);
        },
        .BOOL => |v| {
            buf8[0] = if (v) 1 else 0;
            try file.writeAll(&buf8);
        },
        .STRING => |s| try writeString(file, s),
        .ARRAY => |arr| {
            std.mem.writeInt(u32, &buf32, @intFromEnum(arr.type), .little);
            try file.writeAll(&buf32);
            std.mem.writeInt(u64, &buf64, arr.len, .little);
            try file.writeAll(&buf64);
            try file.writeAll(arr.data);
        },
    }
}

// 1. 基础常量与数据结构定义
pub const QK4_0 = 32;

pub const BlockQ4_0 = extern struct {
    d: f16, // 缩放因子 (2 字节)
    qs: [QK4_0 / 2]u8, // 32 个 4-bit 整数打包为 16 字节
};

// 2. 核心量化函数 (纯函数，返回值形式)
pub fn quantizeBlockQ4_0(src: *const [QK4_0]f32) BlockQ4_0 {
    var dst: BlockQ4_0 = undefined;

    // 寻找绝对值最大值
    var max_abs: f32 = 0.0;
    for (src) |val| {
        const abs_val = @abs(val);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    // 计算缩放因子并存储
    const d: f32 = max_abs / 8.0;
    const id: f32 = if (d != 0.0) 1.0 / d else 0.0;
    dst.d = @floatCast(d);

    // 循环量化并交织打包
    for (0..16) |i| {
        const x0 = src[i] * id;
        const x1 = src[i + 16] * id;

        const xi0: i8 = @intFromFloat(@round(x0));
        const xi1: i8 = @intFromFloat(@round(x1));

        const c0: i8 = @max(-8, @min(7, xi0));
        const c1: i8 = @max(-8, @min(7, xi1));

        const q0: u8 = @intCast(c0 + 8); // [-8,7] -> [0,15]
        const q1: u8 = @intCast(c1 + 8);
        dst.qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }

    return dst;
}

// --- 3. F16 适配器函数 (接收 f16，零时转换后量化) ---
pub fn quantize_block_Q4_0_f16(src_f16: *const [QK4_0]f16) BlockQ4_0 {
    // 在极速的栈内存上开辟一块 32 个 f32 的空间 (仅占用 128 字节)
    var temp_f32: [QK4_0]f32 = undefined;

    // 利用 Zig 的 @floatCast 触发底层的硬件 SIMD 转换指令
    for (0..QK4_0) |i| {
        temp_f32[i] = @floatCast(src_f16[i]);
    }

    // 转换完毕，直接复用核心函数的数学逻辑
    return quantizeBlockQ4_0(&temp_f32);
}

pub const QK_K = 256;

// Q4_K 物理内存块 (144 bytes)
pub const BlockQ4_K = extern struct {
    d: f16, //           Super-block scale (全局缩放)
    dmin: f16, //        Super-block min (全局偏移)
    scales: [12]u8, //   极其复杂的 6-bit 缩放和偏移量打包
    qs: [QK_K / 2]u8, // 128 bytes: 256 个 4-bit 的核心权重 (高低位拆分)
};

/// 纯 Zig 实现的 Q4_K 标量反量化内核
/// blocks:  从 GGUF 中切出来的物理结构体切片
/// out_f32: 提前申请好的、用于存放解压后浮点数的内存
pub fn dequantize_row_q4_K(blocks: []const BlockQ4_K, out_f32: []f32) void {
    // 确保输出数组的大小完美匹配 (每个 block 解压出 256 个 f32)
    std.debug.assert(out_f32.len == blocks.len * QK_K);

    var y_idx: usize = 0;

    for (blocks) |*block| {
        // 1. 将全局范围的 f16 转换为 f32 (Zig 0.11+ 原生支持 f16 硬件指令/软实现)
        const d_f32 = @as(f32, block.d);
        const min_f32 = @as(f32, block.dmin);

        // 2. 地狱级位运算：从 12 个字节中强行解包出 16 个 6-bit 整数
        var sc: [8]u8 = undefined;
        var m: [8]u8 = undefined;

        // --- 解包 Scales (前 4 个在低位，后 4 个跨字节拼接) ---
        sc[0] = block.scales[0] & 63;
        sc[1] = block.scales[1] & 63;
        sc[2] = block.scales[2] & 63;
        sc[3] = block.scales[3] & 63;
        sc[4] = (block.scales[0] >> 6) | ((block.scales[4] & 0x0F) << 2);
        sc[5] = (block.scales[1] >> 6) | ((block.scales[4] & 0xF0) >> 2);
        sc[6] = (block.scales[2] >> 6) | ((block.scales[5] & 0x0F) << 2);
        sc[7] = (block.scales[3] >> 6) | ((block.scales[5] & 0xF0) >> 2);

        // --- 解包 Mins (逻辑同上) ---
        m[0] = block.scales[6] & 63;
        m[1] = block.scales[7] & 63;
        m[2] = block.scales[8] & 63;
        m[3] = block.scales[9] & 63;
        m[4] = (block.scales[6] >> 6) | ((block.scales[10] & 0x0F) << 2);
        m[5] = (block.scales[7] >> 6) | ((block.scales[10] & 0xF0) >> 2);
        m[6] = (block.scales[8] >> 6) | ((block.scales[11] & 0x0F) << 2);
        m[7] = (block.scales[9] >> 6) | ((block.scales[11] & 0xF0) >> 2);

        // 3. 解压 256 个实际的权重值
        var q_idx: usize = 0;
        var j: usize = 0;

        // 每次循环处理 2 个子块 (共 64 个权重)
        while (j < 4) : (j += 1) {
            // 将子块的 6-bit 缩放系数与全局缩放相乘
            const d1 = d_f32 * @as(f32, @floatFromInt(sc[2 * j]));
            const d2 = d_f32 * @as(f32, @floatFromInt(sc[2 * j + 1]));
            const m1 = min_f32 * @as(f32, @floatFromInt(m[2 * j]));
            const m2 = min_f32 * @as(f32, @floatFromInt(m[2 * j + 1]));

            // 处理 32 个物理字节，解压出 64 个 f32 浮点数
            var l: usize = 0;
            while (l < 32) : (l += 1) {
                const q_val = block.qs[q_idx + l];

                // 偶数子块使用字节的 低 4 位 (0~15)
                out_f32[y_idx + l] = d1 * @as(f32, @floatFromInt(q_val & 0x0F)) - m1;

                // 奇数子块使用字节的 高 4 位 (0~15)
                out_f32[y_idx + l + 32] = d2 * @as(f32, @floatFromInt(q_val >> 4)) - m2;
            }

            y_idx += 64; // f32 数组前进 64
            q_idx += 32; // qs 字节数组前进 32
        }
    }
}

// Q6_K 物理内存块 (210 bytes)
pub const BlockQ6_K = extern struct {
    ql: [QK_K / 2]u8, //     128 bytes: 权重的低 4 bits
    qh: [QK_K / 4]u8, //      64 bytes: 权重的高 2 bits
    scales: [QK_K / 16]i8, // 16 bytes: 每个子块的 8-bit scale
    d: f16, //                 2 bytes: Super-block scale
};

pub fn quantize(
    a: Allocator,
    model_path: []const u8,
    output_path: []const u8,
) !void {
    const model = try Read(a, model_path);
    defer model.deinit();

    const quantized_model = try model.quantize_to_Q4_0(a);
    defer quantized_model.deinit();

    try quantized_model.serialize(output_path);
}

pub fn print(
    a: Allocator,
    model_path: []const u8,
) !void {
    const model = try Read(a, model_path);
    defer model.deinit();

    debug("{f}\n", .{model});
}

/// GGUF 字符串数组的零拷贝迭代器
pub const StringIterator = struct {
    raw_data: []const u8,
    offset: usize = 0,
    count: u64,
    current_index: u64 = 0,

    /// 初始化迭代器并进行类型安全检查
    pub fn init(value: Value) !StringIterator {
        if (value != .ARRAY) return error.NotArray;
        const array = value.ARRAY;

        if (array.type != .STRING) return error.NotStringArray;

        return StringIterator{
            .raw_data = array.data,
            .count = array.len,
        };
    }

    /// 获取下一个字符串切片。如果遍历结束，返回 null。
    pub fn next(self: *@This()) !?[]const u8 {
        // 如果已经遍历完所有元素，正常退出
        if (self.current_index >= self.count) return null;

        // 1. 读取 8 字节的长度前缀 (小端序)
        if (self.offset + 8 > self.raw_data.len) return error.BufferTooSmall;
        const str_len = std.mem.readInt(u64, self.raw_data[self.offset .. self.offset + 8][0..8], .little);
        self.offset += 8;

        // 2. 根据读取的长度切分字符串
        if (self.offset + str_len > self.raw_data.len) return error.BufferTooSmall;
        const str = self.raw_data[self.offset .. self.offset + str_len];
        self.offset += str_len;

        self.current_index += 1;
        return str;
    }
};
