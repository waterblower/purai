const std = @import("std");
const gguf = @import("gguf");

pub const Tokenizer = struct {
    a: std.mem.Allocator,
    id_to_string: [][]const u8,
    string_to_id: std.StringHashMap(u32),
    merges: std.StringHashMap(u32),

    pub fn init(allocator: std.mem.Allocator, model: *const gguf.GgufContext) !Tokenizer {
        const tokens_val = model.kv_map.get("tokenizer.ggml.tokens") orelse return error.MissingTokens;
        if (tokens_val != .ARRAY) return error.InvalidTokensFormat;
        const gguf_tokens = tokens_val.ARRAY;

        var id_to_string = try allocator.alloc([]const u8, gguf_tokens.len);
        var string_to_id = std.StringHashMap(u32).init(allocator);
        try string_to_id.ensureTotalCapacity(@as(u32, @intCast(gguf_tokens.len)));

        var iter = try gguf.StringIterator.init(tokens_val);
        var i: usize = 0;
        while (try iter.next()) |str| {
            id_to_string[i] = str;
            try string_to_id.put(str, @as(u32, @intCast(i)));
            i += 1;
        }

        var merges = std.StringHashMap(u32).init(allocator);
        if (model.kv_map.get("tokenizer.ggml.merges")) |merges_val| {
            if (merges_val == .ARRAY) {
                var merge_iter = try gguf.StringIterator.init(merges_val);
                var rank: u32 = 0;
                while (try merge_iter.next()) |merge_str| {
                    try merges.put(merge_str, rank);
                    rank += 1;
                }
            }
        }

        return Tokenizer{
            .a = allocator,
            .id_to_string = id_to_string,
            .string_to_id = string_to_id,
            .merges = merges,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit();
        self.string_to_id.deinit();
        self.a.free(self.id_to_string);
    }

    pub fn decode(self: *const Tokenizer, id: u32) ![]const u8 {
        if (id >= self.id_to_string.len) return error.UnknownTokenID;
        return self.id_to_string[id];
    }

    // A helper struct representing a piece of text during the BPE merge process.
    const BpeNode = struct {
        start: usize,
        end: usize,
        prev: ?usize,
        next: ?usize,
    };

    /// Encodes text into BPE token IDs using Qwen2/GPT-2 logic.
    pub fn encode(self: *const Tokenizer, text: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).empty;
        errdefer tokens.deinit(self.a);

        var pos: usize = 0;

        // Step 1: Pre-tokenization loop (Chunking)
        while (pos < text.len) {
            const chunk_end = getNextChunkBoundary(text, pos);
            const chunk = text[pos..chunk_end];
            pos = chunk_end;

            // Step 2: Apply BPE to the chunk
            try self.bpeEncodeChunk(chunk, &tokens);
        }

        return tokens.toOwnedSlice(self.a);
    }

    /// Internal BPE algorithm applied to a small text chunk.
    fn bpeEncodeChunk(self: *const Tokenizer, chunk: []const u8, tokens: *std.ArrayList(u32)) !void {
        if (chunk.len == 0) return;

        // Initialize a doubly linked list of bytes for this chunk
        var nodes = try self.a.alloc(BpeNode, chunk.len);
        defer self.a.free(nodes);

        for (0..chunk.len) |i| {
            nodes[i] = .{
                .start = i,
                .end = i + 1,
                .prev = if (i > 0) i - 1 else null,
                .next = if (i < chunk.len - 1) i + 1 else null,
            };
        }

        const head: ?usize = 0;
        var merge_key_buf: [256]u8 = undefined;

        // Step 3: Iteratively find and apply the best merges
        while (true) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_node_idx: ?usize = null;

            var curr_idx = head;
            while (curr_idx) |idx| {
                const node = nodes[idx];
                const next_idx = node.next orelse break;
                const next_node = nodes[next_idx];

                const str1 = chunk[node.start..node.end];
                const str2 = chunk[next_node.start..next_node.end];

                // Check if this pair exists in the merges dictionary
                // GGUF BPE merges are space-separated (e.g., "e r")
                if (std.fmt.bufPrint(&merge_key_buf, "{s} {s}", .{ str1, str2 })) |merge_key| {
                    if (self.merges.get(merge_key)) |rank| {
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_node_idx = idx;
                        }
                    }
                } else |_| {
                    // Buffer too small for this pair, skip it
                }
                curr_idx = node.next;
            }

            // If no valid merges were found, we are done with this chunk
            if (best_node_idx) |idx| {
                var node = &nodes[idx];
                const next_idx = node.next.?;
                const next_node = &nodes[next_idx];

                // Merge: extend current node's boundary to include the next node
                node.end = next_node.end;

                // Update linked list pointers to bypass `next_node`
                node.next = next_node.next;
                if (next_node.next) |nn_idx| {
                    nodes[nn_idx].prev = idx;
                }
            } else {
                break;
            }
        }

        // Step 4: Map the final merged sub-strings to Token IDs
        var curr_idx = head;
        while (curr_idx) |idx| {
            const node = nodes[idx];
            const token_str = chunk[node.start..node.end];

            if (self.string_to_id.get(token_str)) |id| {
                try tokens.append(self.a, id);
            } else {
                // Byte fallback: If a chunk somehow doesn't exist, map its raw bytes.
                // This shouldn't happen in a well-formed BPE model, but handles edge cases safely.
                for (token_str) |b| {
                    const byte_str = &[_]u8{b};
                    if (self.string_to_id.get(byte_str)) |byte_id| {
                        try tokens.append(self.a, byte_id);
                    }
                }
            }
            curr_idx = node.next;
        }
    }

    /// A simplified pre-tokenizer that splits text into logical groups (Alphanumeric, Whitespace, Punctuation)
    /// to prevent the BPE O(N^2) algorithm from bogging down on massive strings.
    fn getNextChunkBoundary(text: []const u8, start_pos: usize) usize {
        if (start_pos >= text.len) return text.len;
        const c = text[start_pos];
        var pos = start_pos + 1;

        if (std.ascii.isWhitespace(c)) {
            while (pos < text.len and std.ascii.isWhitespace(text[pos])) : (pos += 1) {}
        } else if (std.ascii.isAlphanumeric(c)) {
            while (pos < text.len and std.ascii.isAlphanumeric(text[pos])) : (pos += 1) {}
        } else {
            // Group punctuation and multi-byte UTF-8 sequences
            while (pos < text.len and !std.ascii.isWhitespace(text[pos]) and !std.ascii.isAlphanumeric(text[pos])) : (pos += 1) {}
        }
        return pos;
    }
};
