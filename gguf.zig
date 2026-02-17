const std = @import("std");
const Allocator = std.mem.Allocator;
const debug = std.debug;
const mem = std.mem;

// GGUF Magic Number: "GGUF"
const GGUF_MAGIC = 0x46554747;

pub const GgufType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // ... Q5, Q8 略 ...
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
    I8 = 16,
    I16 = 17,
    I32 = 18,
    COUNT,
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
pub const GgufValue = union(GgufMetadataValueType) {
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

pub const GgufTensorInfo = struct {
    name: []const u8,
    dims: [4]u64,
    n_dims: u32,
    type: GgufType,
    offset: u64, // 相对 tensor_data_base 的偏移
    data: [*]const u8, // 指向数据的直接指针
};

pub const GgufContext = struct {
    allocator: Allocator,
    data: []align(4096) u8, // 整个文件的内存映射/缓冲区

    version: u32,
    tensor_count: u64,
    kv_count: u64,

    // 核心查找表
    kv_map: std.StringHashMap(GgufValue),
    tensor_map: std.StringHashMap(GgufTensorInfo),

    // 张量数据块的起始位置（绝对指针）
    tensor_data_base: [*]const u8,

    pub fn init(allocator: Allocator, path: []const u8) !*GgufContext {
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
            .kv_map = std.StringHashMap(GgufValue).init(allocator),
            .tensor_map = std.StringHashMap(GgufTensorInfo).init(allocator),
            .tensor_data_base = undefined,
        };

        try ctx.parse();
        return ctx;
    }

    pub fn deinit(self: *GgufContext) void {
        self.kv_map.deinit();
        self.tensor_map.deinit();
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }

    // ------------------------------------------------------------------------
    // Parsing Logic
    // ------------------------------------------------------------------------

    fn parse(self: *GgufContext) !void {
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

            var dims: [4]u64 = .{ 1, 1, 1, 1 };
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
    }

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------

    fn readString(data: []const u8, cursor: *usize) ![]const u8 {
        const len = std.mem.readInt(u64, data[cursor.*..][0..8], .little);
        cursor.* += 8;
        const str = data[cursor.* .. cursor.* + len];
        cursor.* += len;
        return str;
    }

    fn readValue(data: []const u8, cursor: *usize, type_val: GgufMetadataValueType) !GgufValue {
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
                return .{ .ARRAY = .{ .type = type_sub, .len = len, .data = data[start..cursor.*] } };
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
};
