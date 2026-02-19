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

    pub fn decode(tokenizer: *Tokenizer, prev_token: i32, token: i32) ![]const u8 {
        // 获取当前 token 对应的字符串切片
        var piece: []const u8 = tokenizer.vocab[@intCast(token)];

        // [修复 1] 剔除末尾可能存在的 Null Terminator (\0)
        // 因为我们在 init 里分配了 len + 1，所以 piece 可能包含 \0
        if (piece.len > 0 and piece[piece.len - 1] == 0) {
            piece = piece[0 .. piece.len - 1];
        }

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if (prev_token == 1 and piece.len > 0 and piece[0] == ' ') {
            // Zig 切片操作：跳过第一个字符
            piece = piece[1..];
        }

        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        // Pattern check: length 6, starts with "<0x", ends with ">"
        if (piece.len == 6 and piece[0] == '<' and piece[1] == '0' and piece[2] == 'x' and piece[5] == '>') {
            // 尝试解析中间的两个十六进制字符 (piece[3..5])
            if (std.fmt.parseInt(u8, piece[3..5], 16)) |byte_val| {
                // 解析成功，返回 byte_pieces 中对应的字节
                // C: t->byte_pieces + byte_val * 2
                const offset = @as(usize, byte_val) * 2;

                // 返回对应的切片
                // byte_pieces[offset] 是那个字节，byte_pieces[offset+1] 是 \0
                // 我们只需要返回包含那个字节的切片
                return tokenizer.byte_pieces[offset .. offset + 1];
            } else |_| {
                // 解析失败（不是合法的十六进制），不做处理，直接返回原始 piece
            }
        }

        return piece;
    }

    pub fn encode(
        t: *Tokenizer,
        a: std.mem.Allocator,
        text: []const u8,
        bos: bool, // beginning of stream, the C version uses int8_t
        eos: bool, // end of stream
        tokens: []i32,
    ) !usize {
        // if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
        // The C version has a NULL check
        // however, the null type safety in Zig saves us the trouble to runtime checking
        // and the code logic becomes cleaner

        // 1. Lazy malloc and sort the vocabulary (if needed)
        if (t.sorted_vocab.len == 0) {
            t.sorted_vocab = try a.alloc(TokenIndex, @intCast(t.vocab_size));
            for (0..@intCast(t.vocab_size)) |i| {
                t.sorted_vocab[i] = .{ .str = t.vocab[i], .id = @as(i32, @intCast(i)) };
            }
            std.mem.sort(TokenIndex, t.sorted_vocab, {}, compare_tokens);
        }

        // 2. Create temporary buffer
        // C: malloc(t->max_token_length*2 +1 +2)
        const buf_len = t.max_token_length * 2 + 3;
        const str_buffer = try a.alloc(u8, buf_len);
        defer a.free(str_buffer);

        var n_tokens: usize = 0;

        // 3. Add optional BOS (=1) token
        if (bos) {
            tokens[n_tokens] = 1;
            n_tokens += 1;
            debug("n_tokens: {d} {d} \n", .{ n_tokens, tokens[n_tokens - 1] });
        }

        // 4. Add Dummy Prefix (if text is not empty)
        // C assumes text != "" checked by caller or implicit, here we check text.len
        if (text.len > 0) {
            const dummy_prefix = str_lookup(" ", t.sorted_vocab);
            if (dummy_prefix != -1) {
                tokens[n_tokens] = dummy_prefix;
                n_tokens += 1;
            }
        }

        // 5. UTF-8 Processing Loop
        var str_len: usize = 0;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];

            // Reset buffer if current byte is NOT a continuation byte
            // 0xC0 = 11000000, 0x80 = 10000000
            if ((c & 0xC0) != 0x80) {
                str_len = 0;
            }

            // Append byte to buffer
            str_buffer[str_len] = c;
            str_len += 1;

            // Check next byte (bounds checked)
            if (i + 1 < text.len) {
                const next_c = text[i + 1];
                // If next is continuation byte and buffer not full, continue accumulating
                if ((next_c & 0xC0) == 0x80 and str_len < 4) {
                    continue;
                }
            }

            // Full codepoint read. Lookup.
            // We use the slice str_buffer[0..str_len]
            const id = str_lookup(str_buffer[0..str_len], t.sorted_vocab);

            if (id != -1) {
                tokens[n_tokens] = id;
                n_tokens += 1;
            } else {
                // Byte fallback: encode each byte as a token
                for (0..str_len) |j| {
                    // +3 offset for <unk>, <s>, </s>
                    tokens[n_tokens] = @as(i32, @intCast(str_buffer[j])) + 3;
                    n_tokens += 1;
                }
            }
            str_len = 0;
        }

        // 6. Merge Loop (BPE)
        while (true) {
            var best_score: f32 = -1e10;
            var best_id: i32 = -1;
            var best_idx: ?usize = null;

            if (n_tokens < 2) break;

            for (0..(n_tokens - 1)) |j| {
                // C: sprintf(str_buffer, "%s%s", vocab[i], vocab[i+1])
                const tok_str1 = t.vocab[@intCast(tokens[j])];
                const tok_str2 = t.vocab[@intCast(tokens[j + 1])];

                // Print into buffer, get the resulting slice
                // Note: bufPrint returns the slice of written bytes
                const merged_str = try std.fmt.bufPrint(str_buffer, "{s}{s}", .{ tok_str1, tok_str2 });

                const id = str_lookup(merged_str, t.sorted_vocab);
                if (id != -1) {
                    const score = t.vocab_scores[@intCast(id)];
                    if (score > best_score) {
                        best_score = score;
                        best_id = id;
                        best_idx = j;
                    }
                }
            }

            if (best_idx == null) {
                break; // No more merges possible
            }

            // Merge the pair
            const idx = best_idx.?;
            tokens[idx] = best_id;

            // Shift remaining tokens back
            // C: for (int i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1];
            // Zig: move memory
            const tail_len = n_tokens - 1 - (idx + 1);
            if (tail_len > 0) {
                // Using a loop for clarity, though std.mem.copy works too
                for (0..tail_len) |offset| {
                    tokens[idx + 1 + offset] = tokens[idx + 2 + offset];
                }
            }
            n_tokens -= 1;
        }

        // 7. Add optional EOS (=2) token
        if (eos) {
            tokens[n_tokens] = 2;
            n_tokens += 1;
        }

        return n_tokens;
    }
};

pub const TokenIndex = struct {
    str: []u8,
    id: i32,
};

fn compare_tokens(_: void, a: TokenIndex, b: TokenIndex) bool {
    return std.mem.order(u8, a.str, b.str) == .lt;
}

fn str_lookup(str: []const u8, sorted_vocab: []TokenIndex) i32 {
    // 二分查找 (Binary Search)
    var low: usize = 0;
    var high: usize = sorted_vocab.len;

    while (low < high) {
        const mid = low + (high - low) / 2;
        const cmp = std.mem.order(u8, str, sorted_vocab[mid].str);

        switch (cmp) {
            .eq => return sorted_vocab[mid].id,
            .lt => high = mid,
            .gt => low = mid + 1,
        }
    }
    return -1; // Not found
}
