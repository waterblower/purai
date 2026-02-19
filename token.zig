// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
const std = @import("std");
const asBytes = @import("std").mem.asBytes;
const debug = std.debug.print;

pub const Tokenizer = struct {
    a: std.mem.Allocator,
    vocab: [][]u8, // char** vocab
    vocab_scores: []f32, // float* vocab_scores
    sorted_vocab: []TokenIndex, // TokenIndex *sorted_vocab
    vocab_size: i32, // int vocab_size
    max_token_length: u32, // unsigned int max_token_length
    // 含义：单字节 Token 的字符串缓存。
    // 大小：512 字节（256 个 ASCII/二进制值 + 每个都要跟一个 \0 结束符）
    // 作用：为了处理 Raw Byte Fallback
    //   如果遇到词表里没有的生僻字符（比如 Emoji 或罕见汉字）
    //   Llama 的 Tokenizer 会回退到 Byte-level
    //   把它们拆成一个个单字节的 Token（例如 <0xE8>）
    byte_pieces: [512]u8, // stores all single-byte strings

    pub fn init(
        a: std.mem.Allocator,
        tokenizer_path: []const u8,
        vocab_size: i32,
    ) !*Tokenizer {
        var t = try a.create(Tokenizer);
        t.vocab_size = vocab_size;
        t.a = a;

        // 1. Initialize byte_pieces (single byte strings)
        // C: t->byte_pieces[i * 2] = (unsigned char)i;
        for (0..256) |i| {
            t.byte_pieces[i * 2] = @intCast(i);
            t.byte_pieces[i * 2 + 1] = 0; // Null terminator
        }

        // 2. Allocate main arrays
        t.vocab = try a.alloc([]u8, @as(usize, @intCast(vocab_size)));
        t.vocab_scores = try a.alloc(f32, @intCast(vocab_size));
        // In C, sorted_vocab is NULL. In Zig, we set it to an empty slice.
        // Logic later will check if it needs to be populated.
        t.sorted_vocab = &.{};

        // 3. Open file
        var file = try std.fs.cwd().openFile(
            tokenizer_path,
            .{ .mode = .read_only },
        );
        defer file.close();

        // 4. Read max_token_length
        // C: fread(&t->max_token_length, sizeof(int), 1, file)
        const bytes_read = try file.read(asBytes(&t.max_token_length));
        debug("bytes read: {d}\n", .{bytes_read});

        // 5. Read Vocab Loop
        for (0..@intCast(vocab_size)) |i| {
            // Read Score (float)
            // We read 4 bytes as u32, then cast the bits to f32
            var br = try file.read(asBytes(&t.vocab_scores[i]));
            if (br < @sizeOf(@TypeOf(t.vocab_scores[i]))) {
                debug("bytes read vocab_scores[{d}]: {d}\n", .{ i, br });
                return error.IncompleteRead;
            }

            // Read Len (int)
            // const len_i32 = try reader.readInt(i32, .little);
            // const len = @as(usize, @intCast(len_i32));
            var len_32: i32 = 0;
            br = try file.read(asBytes(&len_32));
            if (br < @sizeOf(@TypeOf(len_32))) {
                debug("len {d} : {d}\n", .{ len_32, br });
                return error.IncompleteRead;
            }

            const len = @as(usize, @intCast(len_32));

            // Allocate string buffer (len + 1 for null terminator)
            // C: malloc(len + 1)
            const str_buf = try a.alloc(u8, len + 1);

            // Read string bytes
            // C: fread(t->vocab[i], len, 1, file)
            _ = try file.read(str_buf[0..len]);

            // Add null terminator
            // C: t->vocab[i][len] = '\0';
            str_buf[len] = 0;

            // Assign to vocab array
            t.vocab[i] = str_buf[0..len];
        }
        return t;
    }

    pub fn deinit(self: *Tokenizer) void {
        var a = self.a;
        // 1. 释放每个词汇字符串
        for (self.vocab) |s| {
            // [修复] 重建原始切片：s.ptr 还是那个指针，
            // 但长度需要 + 1 才能匹配 alloc 的大小
            const original_slice = s.ptr[0 .. s.len + 1];
            a.free(original_slice);
        }
        // 2. 释放数组
        a.free(self.vocab);
        a.free(self.vocab_scores);
        // 3. 释放懒加载的排序数组
        if (self.sorted_vocab.len > 0) {
            a.free(self.sorted_vocab);
        }
        // 4. 释放结构体本身
        a.destroy(self);
    }
};

pub const TokenIndex = struct {
    str: []u8,
    id: i32,
};
