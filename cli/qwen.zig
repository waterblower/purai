const std = @import("std");
const gguf = @import("gguf");

pub const Tokenizer = struct {
    a: std.mem.Allocator,
    // ID -> 字符串的直接映射表
    id_to_string: [][]const u8,

    pub fn init(allocator: std.mem.Allocator, model: *const gguf.GgufContext) !Tokenizer {
        // 从 GGUF 的键值对中安全提取词表数组
        const tokens_val = model.kv_map.get("tokenizer.ggml.tokens") orelse {
            return error.MissingTokens;
        };
        if (tokens_val != .ARRAY) return error.InvalidTokensFormat;

        const gguf_tokens = tokens_val.ARRAY;

        // 分配我们自己的查询数组
        var id_to_string = try allocator.alloc([]const u8, gguf_tokens.len);

        // 使用迭代器逐个填充数组
        var iter = try gguf.StringIterator.init(tokens_val);
        var i: usize = 0;
        while (try iter.next()) |str| {
            id_to_string[i] = str;
            i += 1;
        }

        return Tokenizer{
            .a = allocator,
            .id_to_string = id_to_string,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.a.free(self.id_to_string);
    }

    // 根据输出的 Token ID 返回对应的字符串
    pub fn decode(self: *const Tokenizer, id: u32) ![]const u8 {
        if (id >= self.id_to_string.len) return error.UnkonwnTokenID;
        return self.id_to_string[id];
    }
};
