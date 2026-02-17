const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const debug = std.debug.print;
const asBytes = std.mem.asBytes;
const eql = std.mem.eql;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const gguf = @import("./gguf.zig");

pub fn main() !void {
    // 1. Setup Allocator
    const a = gpa.allocator();
    defer _ = gpa.deinit();

    // 2. Parse Args
    const unparsed_args = try std.process.argsAlloc(a);
    defer std.process.argsFree(a, unparsed_args);
    var args = try parseArgs(unparsed_args);

    // Load GGUF
    const model = try gguf.GgufContext.init(a, args.gguf_path);
    defer model.deinit();
    debug("{f}\n", .{model});

    // 3. Build Transformer
    // Note: Assuming build_transformer signature from previous context
    debug("model path: {s}\n", .{args.checkpoint_path});
    var t = try Transformer.build(a, args.checkpoint_path);
    defer t.deinit();

    if (args.steps == 0 or args.steps > t.config.seq_len) {
        args.steps = t.config.seq_len;
    }
    debug("steps: {d}\n", .{args.steps});

    // // 4. Build Tokenizer
    var tokenizer = try Tokenizer.init(
        a,
        args.tokenizer_path,
        t.config.vocab_size,
    );
    defer tokenizer.deinit();

    // // 5. Build Sampler
    var sampler = try Sampler.init(
        a,
        @intCast(t.config.vocab_size),
        args.temperature,
        args.topp,
        args.rng_seed,
    );
    defer sampler.deinit();

    // // 6. Run Mode
    if (eql(u8, args.mode, "generate")) {
        try generate(
            a,
            t,
            tokenizer,
            sampler,
            args.prompt,
            args.steps,
        );
    } else if (eql(u8, args.mode, "chat")) {
        try chat(
            a,
            t,
            tokenizer,
            sampler,
            args.prompt,
            args.system_prompt,
            args.steps,
        );
    } else {
        debug("unknown mode: {s}\n", .{args.mode});
        error_usage();
    }
}

// 1. 定义配置结构体
pub const CliArgs = struct {
    checkpoint_path: []const u8 = "./models/stories110M.bin",
    gguf_path: []const u8 = "./models/test.gguf",
    tokenizer_path: []const u8 = "./models/tokenizer.bin",
    temperature: f32 = 1.0,
    topp: f32 = 0.9,
    steps: i32 = 256,
    prompt: []const u8 = "",
    rng_seed: u64 = 0,
    mode: []const u8 = "generate",
    system_prompt: ?[]const u8 = null,
};

// 2. 解析函数
fn parseArgs(args: [][:0]u8) !CliArgs {
    debug("args: {d}\n", .{args.len});
    // Basic arg validation
    if (args.len < 2) {
        error_usage();
    }

    // Initialize with defaults and the mandatory checkpoint path
    var config = CliArgs{};

    // "Poor man's argparse" loop
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        debug("arg: {s}\n", .{args[i]});
        // Validation: Must have arg after flag
        if (i + 1 >= args.len) error_usage();

        const flag = args[i];
        const val = args[i + 1];

        // Validation: Flag must start with dash and be length 2 (-x)
        if (flag.len != 2 or flag[0] != '-') error_usage();

        switch (flag[1]) {
            't' => config.temperature = try std.fmt.parseFloat(f32, val),
            'p' => config.topp = try std.fmt.parseFloat(f32, val),
            's' => config.rng_seed = try std.fmt.parseInt(u64, val, 10),
            'n' => config.steps = try std.fmt.parseInt(i32, val, 10),
            'i' => config.prompt = val,
            'z' => config.tokenizer_path = val,
            'm' => config.mode = val,
            'y' => config.system_prompt = val,
            else => error_usage(),
        }
    }

    // Parameter overrides / Validation logic
    if (config.rng_seed == 0) config.rng_seed = @intCast(std.time.timestamp());
    if (config.temperature < 0.0) config.temperature = 0.0;
    if (config.topp < 0.0 or config.topp > 1.0) config.topp = 0.9;
    if (config.steps < 0) config.steps = 0;

    return config;
}

fn error_usage() noreturn {
    std.debug.print(
        \\Usage:   run <checkpoint> [options]
        \\Example: run model.bin -n 256 -i "Once upon a time"
        \\Options:
        \\  -t <float>  temperature in [0,inf], default 1.0
        \\  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
        \\  -s <int>    random seed, default time(NULL)
        \\  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
        \\  -i <string> input prompt
        \\  -z <string> optional path to custom tokenizer
        \\  -m <string> mode: generate|chat, default: generate
        \\  -y <string> (optional) system prompt in chat mode
        \\
    , .{});
    std.process.exit(1);
}

fn read_checkpoint(a: std.mem.Allocator, checkpoint_path: []const u8) !*Transformer {
    // 1. Allocate the Transformer struct
    const t = try a.create(Transformer);

    // 2. Open file
    // C: fopen(checkpoint, "rb"); if (!file) { ... exit ... }
    var fd = try std.fs.cwd().openFile(
        checkpoint_path,
        .{ .mode = .read_only },
    );
    defer fd.close();

    // 3. Read Config
    // C: fread(config, sizeof(Config), 1, file)
    const n_read = try fd.read(std.mem.asBytes(&t.config));
    if (n_read != @sizeOf(Config)) {
        return error.ReadIncorrectSize;
    }
    debug("{any}\n", .{t.config});

    // 4. Shared weights logic (Vocab size hack)
    const shared_weights = t.config.vocab_size > 0;
    if (t.config.vocab_size < 0) {
        t.config.vocab_size = -t.config.vocab_size;
    }

    // 5. Get file size
    // C: fseek/ftell
    const stat = try fd.stat();
    t.file_size = stat.size;

    try fd.seekTo(0);

    // This was missing. We ask for 'file_size' bytes aligned to 4096.
    t.data = try a.alignedAlloc(
        u8,
        std.mem.Alignment.fromByteUnits(4096),
        t.file_size,
    );
    // Safety: If readAll fails, free this memory
    errdefer a.free(t.data);

    // 6. Memory Map
    // C: mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    const bytes_read = try fd.readAll(t.data);
    if (bytes_read != t.file_size) {
        return error.IncompleteRead;
    }

    // 7. Calculate weights pointer
    // 跳过开头的 Config 结构体大小（28字节）
    const weights_start = t.data.ptr + @sizeOf(Config);

    // 强转为 f32 指针。
    // 注意：因为你用了 alignedAlloc(..., 4096)，这里的对齐是安全的。
    const weights_ptr = @as([*]f32, @ptrCast(@alignCast(weights_start)));

    // 8. Call unimplemented function
    memory_map_weights(&t.weights, &t.config, weights_ptr, shared_weights);

    return t;
}

fn malloc_run_state(a: std.mem.Allocator, p: *const Config) !*RunState {
    var s = try a.create(RunState);
    s.a = a;

    // 1. Prepare dimensions as usize to prevent overflow during multiplication
    const dim: usize = @intCast(p.dim);
    const hidden_dim: usize = @intCast(p.hidden_dim);
    const n_layers: usize = @intCast(p.n_layers);
    const n_heads: usize = @intCast(p.n_heads);
    const n_kv_heads: usize = @intCast(p.n_kv_heads);
    const vocab_size: usize = @intCast(p.vocab_size);
    const seq_len: usize = @intCast(p.seq_len);

    const kv_dim = (dim * n_kv_heads) / n_heads;

    // 2. Allocate and Zero (equivalent to calloc)

    s.x = try a.alloc(f32, dim);
    @memset(s.x, 0);

    s.xb = try a.alloc(f32, dim);
    @memset(s.xb, 0);

    s.xb2 = try a.alloc(f32, dim);
    @memset(s.xb2, 0);

    s.hb = try a.alloc(f32, hidden_dim);
    @memset(s.hb, 0);

    s.hb2 = try a.alloc(f32, hidden_dim);
    @memset(s.hb2, 0);

    s.q = try a.alloc(f32, dim);
    @memset(s.q, 0);

    // KV Caches
    const cache_size = n_layers * seq_len * kv_dim;
    s.key_cache = try a.alloc(f32, cache_size);
    @memset(s.key_cache, 0);

    s.value_cache = try a.alloc(f32, cache_size);
    @memset(s.value_cache, 0);

    s.att = try a.alloc(f32, n_heads * seq_len);
    @memset(s.att, 0);

    s.logits = try a.alloc(f32, vocab_size);
    @memset(s.logits, 0);

    // Note: s.k and s.v are not allocated here.
    // They are just view pointers that will point inside key_cache/value_cache during inference.
    return s;
}

fn memory_map_weights(w: *TransformerWeights, config: *const Config, ptr: [*]f32, shared_weights: bool) void {
    // We use u64 for intermediate calculations to prevent overflow on large models
    const dim: u64 = @intCast(config.dim);
    const hidden_dim: u64 = @intCast(config.hidden_dim);
    const n_layers: u64 = @intCast(config.n_layers);
    const n_heads: u64 = @intCast(config.n_heads);
    const n_kv_heads: u64 = @intCast(config.n_kv_heads);
    const vocab_size: u64 = @intCast(config.vocab_size);
    const seq_len: u64 = @intCast(config.seq_len);

    const head_size = dim / n_heads;

    // Use a mutable pointer cursor
    var p_ptr = ptr;

    w.token_embedding_table = p_ptr;
    p_ptr += vocab_size * dim;

    w.rms_att_weight = p_ptr;
    p_ptr += n_layers * dim;

    w.wq = p_ptr;
    p_ptr += n_layers * dim * (n_heads * head_size);

    w.wk = p_ptr;
    p_ptr += n_layers * dim * (n_kv_heads * head_size);

    w.wv = p_ptr;
    p_ptr += n_layers * dim * (n_kv_heads * head_size);

    w.wo = p_ptr;
    p_ptr += n_layers * (n_heads * head_size) * dim;

    w.rms_ffn_weight = p_ptr;
    p_ptr += n_layers * dim;

    w.w1 = p_ptr;
    p_ptr += n_layers * dim * hidden_dim;

    w.w2 = p_ptr;
    p_ptr += n_layers * hidden_dim * dim;

    w.w3 = p_ptr;
    p_ptr += n_layers * dim * hidden_dim;

    w.rms_final_weight = p_ptr;
    p_ptr += dim;

    // Skip RoPE frequencies (freq_cis_real and freq_cis_imag)
    p_ptr += seq_len * head_size / 2;
    p_ptr += seq_len * head_size / 2;

    w.wcls = if (shared_weights) w.token_embedding_table else p_ptr;
}

const Transformer = struct {
    a: std.mem.Allocator,
    config: Config, //              the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: *RunState, //             buffers for the "wave" of activations in the forward pass

    file_size: u64, //         File sizes are unsigned 64-bit integers in std.fs
    data: []align(4096) u8, // We keep the original mmap result (bytes) for safe unmapping (munmap).
    //                         We cast this to floats strictly when assigning into 'weights'.

    fn build(a: Allocator, checkpoint_path: []const u8) !*Transformer {
        // read in the Config and the Weights from the checkpoint
        const t = try read_checkpoint(a, checkpoint_path);
        // allocate the RunState buffers
        const state = try malloc_run_state(a, &t.config);
        t.state = state;
        t.a = a;
        return t;
    }

    pub fn deinit(self: *Transformer) void {
        // 1. Free RunState buffers
        // Note: s.k and s.v are pointers into key_cache/value_cache, so we don't free them directly.
        self.state.deinit();

        // 2. Free the main model data buffer
        // Since we switched to readAll + alignedAlloc, we use allocator.free.
        // If you revert to mmap later, change this to std.posix.munmap(self.data);
        if (self.data.len > 0) {
            self.a.free(self.data);
        }
        self.a.destroy(self);
    }
};

// extern is important here because we want to fix the fields ordering
// for binary file reading
const Config = extern struct {
    dim: i32, //        transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, //   number of layers
    n_heads: i32, //    number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, //    max sequence length
};

const TransformerWeights = struct {
    // token embedding table
    token_embedding_table: [*]f32, // (vocab_size, dim)

    // weights for rmsnorms
    rms_att_weight: [*]f32, // (layer, dim) rmsnorm weights
    rms_ffn_weight: [*]f32, // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size
    wq: [*]f32, // (layer, dim, n_heads * head_size)
    wk: [*]f32, // (layer, dim, n_kv_heads * head_size)
    wv: [*]f32, // (layer, dim, n_kv_heads * head_size)
    wo: [*]f32, // (layer, n_heads * head_size, dim)

    // weights for ffn
    w1: [*]f32, // (layer, hidden_dim, dim)
    w2: [*]f32, // (layer, dim, hidden_dim)
    w3: [*]f32, // (layer, hidden_dim, dim)

    // final rmsnorm
    rms_final_weight: [*]f32, // (dim,)

    // (optional) classifier weights for the logits, on the last layer
    wcls: [*]f32,
};

const RunState = struct {
    a: std.mem.Allocator,
    // current wave of activations
    x: []f32, //   activation at current time stamp (dim,)
    xb: []f32, //  same, but inside a residual branch (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim,)
    hb: []f32, //  buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []f32, //   query (dim,)
    k: []f32, //   key (dim,)
    v: []f32, //   value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits

    // kv cache
    key_cache: []f32, //   (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)

    pub fn deinit(self: *RunState) void {
        var a = self.a;
        a.free(self.x);
        a.free(self.xb);
        a.free(self.xb2);
        a.free(self.hb);
        a.free(self.hb2);
        a.free(self.q);
        // k and v are skipped because they point into key_cache/value_cache
        a.free(self.att);
        a.free(self.logits);
        a.free(self.key_cache);
        a.free(self.value_cache);
        a.destroy(self);
    }
};

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

const Tokenizer = struct {
    a: std.mem.Allocator,
    vocab: [][]u8, // char** vocab
    vocab_scores: []f32, // float* vocab_scores
    sorted_vocab: []TokenIndex, // TokenIndex *sorted_vocab
    vocab_size: i32, // int vocab_size
    max_token_length: u32, // unsigned int max_token_length
    byte_pieces: [512]u8, // stores all single-byte strings

    fn init(
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

const TokenIndex = struct {
    str: []u8, // char *str
    id: i32, // int id
};

const Sampler = struct {
    vocab_size: usize,
    probindex: []ProbIndex,
    temperature: f32,
    topp: f32,
    rng_state: u64,
    a: Allocator,

    fn init(
        a: Allocator,
        vocab_size: usize,
        temperature: f32,
        topp: f32,
        rng_seed: u64,
    ) !*Sampler {
        var sampler = try a.create(Sampler);
        sampler.a = a;
        sampler.vocab_size = vocab_size;
        sampler.temperature = temperature;
        sampler.topp = topp;
        sampler.rng_state = rng_seed;
        sampler.probindex = try a.alloc(ProbIndex, vocab_size);
        return sampler;
    }

    fn deinit(self: *Sampler) void {
        self.a.free(self.probindex);
        self.a.destroy(self);
    }
};

const ProbIndex = struct {
    prob: f32,
    index: i32,
};

fn generate(
    a: std.mem.Allocator,
    transformer: *Transformer,
    tokenizer: *Tokenizer,
    sampler: *Sampler,
    prompt: []const u8,
    steps: i32,
) !void {
    // 1. Encode prompt
    // C: malloc((strlen(prompt)+3) * sizeof(int))
    // Zig: 我们根据 prompt 长度估算需要的 token 空间 (+3 for safety: BOS, EOS, null terminator)
    const prompt_tokens = try a.alloc(i32, prompt.len + 3);
    defer a.free(prompt_tokens);

    // 调用 encode (假设你会在别处实现这个函数)
    // 1 = BOS (True), 0 = EOS (False)
    const num_prompt_tokens = try encode(
        a,
        tokenizer,
        prompt,
        true,
        false,
        prompt_tokens,
    );
    if (num_prompt_tokens < 1) {
        debug("something is wrong, expected at least 1 prompt token\n", .{});
        return error.PromptEncodingFailed;
    }

    // 2. Main Loop
    var start: i64 = 0; // used to time our code
    var next: i32 = 0; // will store the next token
    var token: i32 = prompt_tokens[0]; // kick off with the first token
    var pos: usize = 0;
    const max_steps = @as(usize, @intCast(steps));

    while (pos < max_steps) {
        // forward the transformer to get logits for the next token
        // (假设 forward 返回 []f32)
        const logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) break;

        // print the token as string, decode it with the Tokenizer object
        const piece = try decode(tokenizer, token, next);

        // safe_printf logic: Zig 的 print 默认处理 utf8，如果 piece 包含非打印字符可能需要过滤，
        // 但通常直接输出即可。
        debug("{s}", .{piece});

        // 刷新缓冲区确保实时显示 (Zig 的 stdout 默认可能有缓冲)
        // 这里的 flush 取决于你具体的 stdout 实现，简单起见可以忽略，或使用 std.io.bufferedWriter 的 flush

        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = std.time.milliTimestamp();
        }
    }
    debug("\n", .{});

    // report achieved tok/s
    if (pos > 1) {
        const end = std.time.milliTimestamp();
        // 计算耗时 (秒)
        const duration_s = @as(f64, @floatFromInt(end - start)) / 1000.0;
        const tok_per_s = @as(f64, @floatFromInt(pos - 1)) / duration_s;
        debug("achieved tok/s: {d:.4}\n", .{tok_per_s});
    }
}

// -------------------------------------------------------------------------
// 依赖函数原型 (你需要实现这些函数，或者将它们放在文件的适当位置)
// -------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Main Encode Function
// ----------------------------------------------------------------------------

fn encode(
    allocator: std.mem.Allocator,
    t: *Tokenizer,
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
        t.sorted_vocab = try allocator.alloc(TokenIndex, @intCast(t.vocab_size));
        for (0..@intCast(t.vocab_size)) |i| {
            t.sorted_vocab[i] = .{ .str = t.vocab[i], .id = @as(i32, @intCast(i)) };
        }
        std.mem.sort(TokenIndex, t.sorted_vocab, {}, compare_tokens);
    }

    // 2. Create temporary buffer
    // C: malloc(t->max_token_length*2 +1 +2)
    const buf_len = t.max_token_length * 2 + 3;
    const str_buffer = try allocator.alloc(u8, buf_len);
    defer allocator.free(str_buffer);

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

fn forward(transformer: *Transformer, token: i32, pos: usize) []f32 {
    const config = &transformer.config;
    const w = &transformer.weights;
    const s = transformer.state;

    // Dimensions
    const dim: usize = @intCast(config.dim);
    const hidden_dim: usize = @intCast(config.hidden_dim);
    const head_size = dim / @as(usize, @intCast(config.n_heads));
    const n_heads: usize = @intCast(config.n_heads);
    const n_kv_heads: usize = @intCast(config.n_kv_heads);
    const kv_dim = (dim * n_kv_heads) / n_heads;
    const kv_mul = n_heads / n_kv_heads;

    // 1. Copy embedding into x
    const token_idx = @as(usize, @intCast(token));
    // [FIX] Slice the pointer explicitly: [start .. end]
    const content_row = w.token_embedding_table[token_idx * dim .. (token_idx + 1) * dim];
    @memcpy(s.x, content_row);

    const seq_len: usize = @intCast(config.seq_len);

    // 2. Forward all layers
    for (0..@intCast(config.n_layers)) |l| {
        // --- Attention Block ---

        // RMSNorm
        // [FIX] Explicit slicing for weights
        rmsnorm(s.xb, s.x, w.rms_att_weight[l * dim .. (l + 1) * dim]);

        // Key/Value Cache Offsets
        const loff = l * seq_len * kv_dim;
        const k_cache_offset = loff + pos * kv_dim;
        const v_cache_offset = loff + pos * kv_dim;

        // Target slices in the cache
        const k_target = s.key_cache[k_cache_offset .. k_cache_offset + kv_dim];
        const v_target = s.value_cache[v_cache_offset .. v_cache_offset + kv_dim];

        // QKV Matmuls
        // [FIX] Explicit slicing for all matmul weights
        // wq: (dim, dim)
        matmul(s.q, s.xb, w.wq[l * dim * dim .. (l + 1) * dim * dim], dim, dim);
        // wk: (kv_dim, dim)
        matmul(k_target, s.xb, w.wk[l * dim * kv_dim .. (l + 1) * dim * kv_dim], kv_dim, dim);
        // wv: (kv_dim, dim)
        matmul(v_target, s.xb, w.wv[l * dim * kv_dim .. (l + 1) * dim * kv_dim], kv_dim, dim);

        // RoPE Relative Positional Encoding
        var i: usize = 0;
        while (i < dim) : (i += 2) {
            const head_dim = i % head_size;
            const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(head_dim)) / @as(f32, @floatFromInt(head_size)));
            const val = @as(f32, @floatFromInt(pos)) * freq;
            const fcr = std.math.cos(val);
            const fci = std.math.sin(val);

            // How many vectors to rotate? (query is always rotated, key depends on kv_dim)
            const rotn: usize = if (i < kv_dim) 2 else 1;

            for (0..rotn) |v_idx| {
                // v_idx 0 = query, v_idx 1 = key (inside cache)
                const vec = if (v_idx == 0) s.q else k_target;

                const v0 = vec[i];
                const v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Multihead Attention
        multihead_attention(
            pos,
            n_heads,
            head_size,
            loff,
            kv_dim,
            kv_mul,
            s,
            config,
        );

        // Output Projection (Wo)
        // [FIX] Explicit slicing: [l*dim*dim .. (l+1)*dim*dim]
        matmul(s.xb2, s.xb, w.wo[l * dim * dim .. (l + 1) * dim * dim], dim, dim);

        // Residual Connection 1
        for (0..dim) |idx| {
            s.x[idx] += s.xb2[idx];
        }

        // --- Feed Forward Block ---

        // FFN RMSNorm
        // [FIX] Explicit slicing
        rmsnorm(s.xb, s.x, w.rms_ffn_weight[l * dim .. (l + 1) * dim]);

        // Calculate offsets for FFN weights
        // w1, w2, w3 all have size (dim * hidden_dim) per layer
        const ffn_offset = l * dim * hidden_dim;
        const ffn_size = dim * hidden_dim;

        // W1 (Gate) & W3 (Value)
        // [FIX] Explicit slicing using calculated offsets
        matmul(s.hb, s.xb, w.w1[ffn_offset .. ffn_offset + ffn_size], hidden_dim, dim);
        matmul(s.hb2, s.xb, w.w3[ffn_offset .. ffn_offset + ffn_size], hidden_dim, dim);

        // SwiGLU Activation
        for (0..hidden_dim) |idx| {
            var val = s.hb[idx];
            // silu(x) = x * sigmoid(x)
            val *= (1.0 / (1.0 + std.math.exp(-val)));
            // elementwise multiply with w3 output
            val *= s.hb2[idx];
            s.hb[idx] = val;
        }

        // W2 (Projection)
        // [FIX] Explicit slicing
        matmul(s.xb, s.hb, w.w2[ffn_offset .. ffn_offset + ffn_size], dim, hidden_dim);

        // Residual Connection 2
        for (0..dim) |idx| {
            s.x[idx] += s.xb[idx];
        }
    }

    // 3. Final RMSNorm
    // [FIX] Explicit slicing
    rmsnorm(s.x, s.x, w.rms_final_weight[0..dim]);

    // 4. Classifier
    // [FIX] Explicit slicing: wcls size is vocab_size * dim
    const vocab_size = @as(usize, @intCast(config.vocab_size));
    matmul(s.logits, s.x, w.wcls[0 .. vocab_size * dim], vocab_size, dim);

    return s.logits;
}

fn multihead_attention(
    pos: usize,
    n_heads: usize,
    head_size: usize,
    loff: usize,
    kv_dim: usize,
    kv_mul: usize,
    run_state: *const RunState,
    config: *const Config,
) void {
    // 优化 3: 预计算缩放因子
    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_size)));

    // SIMD 设置
    const vec_len = 8;
    const Vec = @Vector(vec_len, f32);

    for (0..n_heads) |h| {
        const q = run_state.q[h * head_size .. (h + 1) * head_size];
        const att = run_state.att[h * @as(usize, @intCast(config.seq_len)) ..];

        // GQA (Grouped Query Attention) offset calculation
        // 移出内层循环，减少整数乘法
        const h_kv = h / kv_mul;
        const head_offset_in_kv = h_kv * head_size;

        for (0..pos + 1) |t| {
            // 优化 4: 简化指针算术
            const k_offset = loff + t * kv_dim + head_offset_in_kv;
            const k = run_state.key_cache[k_offset .. k_offset + head_size];

            // 优化 1: SIMD Dot Product
            var vec_sum: Vec = @splat(0.0);
            var idx: usize = 0;
            while (idx + vec_len <= head_size) : (idx += vec_len) {
                const vq: Vec = q[idx..][0..vec_len].*;
                const vk: Vec = k[idx..][0..vec_len].*;
                vec_sum += vq * vk;
            }
            var score = @reduce(.Add, vec_sum);
            // 处理尾部 (如果 head_size 不是 8 的倍数)
            while (idx < head_size) : (idx += 1) {
                score += q[idx] * k[idx];
            }

            att[t] = score * scale; // 改除为乘
        }

        softmax(att[0 .. pos + 1]);

        const xb_head = run_state.xb[h * head_size .. (h + 1) * head_size];
        @memset(xb_head, 0);

        for (0..pos + 1) |t| {
            const v_offset = loff + t * kv_dim + head_offset_in_kv;
            const v = run_state.value_cache[v_offset .. v_offset + head_size];
            const a_weight = att[t];

            // 优化 1: SIMD Weighted Accumulation
            // 这里的逻辑是: vec_xb += scalar_weight * vec_v
            const vec_weight: Vec = @splat(a_weight);
            var idx: usize = 0;
            while (idx + vec_len <= head_size) : (idx += vec_len) {
                const vv: Vec = v[idx..][0..vec_len].*;
                // 注意：这里需要先把 xb_head 读取为向量，加完再存回去
                var vxb: Vec = xb_head[idx..][0..vec_len].*;
                vxb += vec_weight * vv;
                xb_head[idx..][0..vec_len].* = vxb;
            }
            // 处理尾部
            while (idx < head_size) : (idx += 1) {
                xb_head[idx] += a_weight * v[idx];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Random Number Generation Helpers (Xorshift)
// ----------------------------------------------------------------------------

fn random_u32(state: *u64) u32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @as(u32, @truncate((x *% 0x2545F4914F6CDD1D) >> 32));
}

fn random_f32(state: *u64) f32 {
    // random float32 in [0, 1)
    return @as(f32, @floatFromInt(random_u32(state) >> 8)) / 16777216.0;
}

// ----------------------------------------------------------------------------
// Sampling Helper Functions
// ----------------------------------------------------------------------------

fn sample_argmax(probabilities: []f32) i32 {
    // return the index that has the highest probability
    var max_i: usize = 0;
    var max_p: f32 = probabilities[0];

    for (1..probabilities.len) |i| {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return @as(i32, @intCast(max_i));
}

fn sample_mult(probabilities: []f32, coin: f32) i32 {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1)
    var cdf: f32 = 0.0;
    for (0..probabilities.len) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return @as(i32, @intCast(i));
        }
    }
    return @as(i32, @intCast(probabilities.len - 1)); // in case of rounding errors
}

fn compare_probindex(_: void, a: ProbIndex, b: ProbIndex) bool {
    // sort descending by probability
    return a.prob > b.prob;
}

fn sample_topp(probabilities: []f32, topp: f32, probindex: []ProbIndex, coin: f32) i32 {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

    // 1. cutoff the least likely tokens
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const cutoff = (1.0 - topp) / @as(f32, @floatFromInt(probabilities.len - 1));
    var n0: usize = 0;

    for (0..probabilities.len) |i| {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = @as(i32, @intCast(i));
            probindex[n0].prob = probabilities[i];
            n0 += 1;
        }
    }

    // sort the candidate tokens
    const candidates = probindex[0..n0];
    std.mem.sort(ProbIndex, candidates, {}, compare_probindex);

    // 2. truncate the list where the cumulative probability exceeds topp
    var cumulative_prob: f32 = 0.0;
    var last_idx: usize = n0 - 1; // in case of rounding errors consider all elements

    for (0..n0) |i| {
        cumulative_prob += candidates[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // 3. sample from the truncated list
    const r = coin * cumulative_prob;
    var cdf: f32 = 0.0;

    for (0..last_idx + 1) |i| {
        cdf += candidates[i].prob;
        if (r < cdf) {
            return candidates[i].index;
        }
    }

    return candidates[last_idx].index; // should not be reached
}

// ----------------------------------------------------------------------------
// Main Sample Function
// ----------------------------------------------------------------------------

fn sample(sampler: *Sampler, logits: []f32) i32 {
    // sample the token given the logits and some hyperparameters
    var next: i32 = 0;

    if (sampler.temperature == 0.0) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits);
    } else {
        // apply the temperature to the logits
        for (0..logits.len) |q| {
            logits[q] /= sampler.temperature;
        }

        // apply softmax to the logits to get the probabilities for next token
        softmax(logits);

        // flip a (float) coin (this is our source of entropy for sampling)
        const coin = random_f32(&sampler.rng_state);

        // we sample from this distribution to get the next token
        if (sampler.topp <= 0.0 or sampler.topp >= 1.0) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler.topp, sampler.probindex, coin);
        }
    }

    return next;
}

fn decode(tokenizer: *Tokenizer, prev_token: i32, token: i32) ![]const u8 {
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

fn rmsnorm(o: []f32, x: []f32, weight: []f32) void {
    // calculate sum of squares
    var ss: f32 = 0.0;
    for (x) |val| {
        ss += val * val;
    }
    ss /= @as(f32, @floatFromInt(x.len));
    ss += 1e-5;
    ss = 1.0 / std.math.sqrt(ss);

    // normalize and scale
    for (0..x.len) |j| {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn softmax(x: []f32) void {
    // find max value (for numerical stability)
    var max_val = x[0];
    for (x[1..]) |val| {
        if (val > max_val) max_val = val;
    }

    // exp and sum
    var sum: f32 = 0.0;
    for (0..x.len) |i| {
        x[i] = std.math.exp(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (0..x.len) |i| {
        x[i] /= sum;
    }
}

fn matmul(xout: []f32, x: []f32, w: []f32, n: usize, d: usize) void {
    // 1. 定义向量宽度 (AVX2 = 8 floats)
    const vec_len = 8;
    const Vec = @Vector(vec_len, f32);

    // 2. 针对每一行输出
    for (0..n) |i| {
        // 准备 4 个累加器，打破数据依赖链
        var sum0: Vec = @splat(0.0);
        var sum1: Vec = @splat(0);
        var sum2: Vec = @splat(0);
        var sum3: Vec = @splat(0);

        var val: f32 = 0.0;

        // 获取当前行的权重切片
        const w_row = w[i * d .. (i + 1) * d];
        var j: usize = 0;

        // 3. 主循环：一次处理 4 * 8 = 32 个元素
        // 这允许 CPU 并行执行 4 个 FMA 指令
        while (j + 32 <= d) : (j += 32) {
            // 加载 X (激活值)
            const x0: Vec = x[j + 0 ..][0..vec_len].*;
            const x1: Vec = x[j + 8 ..][0..vec_len].*;
            const x2: Vec = x[j + 16 ..][0..vec_len].*;
            const x3: Vec = x[j + 24 ..][0..vec_len].*;

            // 加载 W (权重)
            const w0: Vec = w_row[j + 0 ..][0..vec_len].*;
            const w1: Vec = w_row[j + 8 ..][0..vec_len].*;
            const w2: Vec = w_row[j + 16 ..][0..vec_len].*;
            const w3: Vec = w_row[j + 24 ..][0..vec_len].*;

            // 并行累加 (Instruction Level Parallelism)
            sum0 += x0 * w0;
            sum1 += x1 * w1;
            sum2 += x2 * w2;
            sum3 += x3 * w3;
        }

        // 4. 合并 4 个累加器
        sum0 += sum1;
        sum2 += sum3;
        sum0 += sum2;

        // 归约 (Reduce) 成单个标量
        val = @reduce(.Add, sum0);

        // 5. 处理剩余的尾部 (Tail Loop)
        while (j < d) : (j += 1) {
            val += w_row[j] * x[j];
        }

        xout[i] = val;
    }
}

fn chat(
    a: std.mem.Allocator,
    transformer: *Transformer,
    tokenizer: *Tokenizer,
    sampler: *Sampler,
    cli_user_prompt: ?[]const u8,
    cli_system_prompt: ?[]const u8,
    steps: i32,
) !void {
    // buffers for reading the system prompt and user prompt from stdin
    var system_prompt_buf: [512]u8 = undefined;
    var user_prompt_buf: [512]u8 = undefined;
    var rendered_prompt_buf: [1152]u8 = undefined;

    // token buffer
    const prompt_tokens = try a.alloc(i32, 1152);
    defer a.free(prompt_tokens);

    var num_prompt_tokens: usize = 0;
    var user_idx: usize = 0;

    // start the main loop
    var user_turn = true; // user starts
    var next: i32 = 0; // will store the next token
    var token: i32 = 0; // stores the current token
    var pos: usize = 0;

    const max_steps = @as(usize, @intCast(steps));

    while (pos < max_steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            var system_prompt: []const u8 = "";
            var user_prompt: []const u8 = "";

            // get the (optional) system prompt at position 0
            if (pos == 0) {
                if (cli_system_prompt) |sp| {
                    system_prompt = sp;
                } else {
                    system_prompt = try read_stdin(
                        "Enter system prompt (optional): ",
                        &system_prompt_buf,
                    );
                }
            }

            // get the user prompt
            if (pos == 0 and cli_user_prompt != null) {
                user_prompt = cli_user_prompt.?;
            } else {
                user_prompt = try read_stdin("User: ", &user_prompt_buf);
            }

            // render user/system prompts into the Llama 2 Chat schema
            var rendered: []const u8 = "";
            if (pos == 0 and system_prompt.len > 0) {
                rendered = try std.fmt.bufPrint(
                    &rendered_prompt_buf,
                    "[INST] <<SYS>>\n{s}\n<</SYS>>\n\n{s} [/INST]",
                    .{ system_prompt, user_prompt },
                );
            } else {
                rendered = try std.fmt.bufPrint(
                    &rendered_prompt_buf,
                    "[INST] {s} [/INST]",
                    .{user_prompt},
                );
            }

            // encode the rendered prompt into tokens
            num_prompt_tokens = try encode(a, tokenizer, rendered, true, false, prompt_tokens);
            user_idx = 0;
            user_turn = false;
            debug("Assistant: ", .{});
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx];
            user_idx += 1;
        } else {
            token = next;
        }

        // EOS (=2) token ends the Assistant turn
        if (token == 2) {
            user_turn = true;
        }

        // forward the transformer to get logits for the next token
        const logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos += 1;

        if (user_idx >= num_prompt_tokens and next != 2) {
            // the Assistant is responding, so print its output
            const piece = try decode(tokenizer, token, next);
            debug("{s}", .{piece});
            // flush output (optional, depends on stdout buffering)
        }

        if (next == 2) {
            debug("\n", .{});
        }
    }
    debug("\n", .{});
}

fn read_stdin(guide: []const u8, buffer: []u8) ![]const u8 {
    // 1. Get stdout and stdin handles (New 0.15 syntax)
    const stdout_file = std.fs.File.stdout();
    const stdin_file = std.fs.File.stdin();

    // 2. Print the guide (prompt)
    // Using writeAll directly on the file handle ensures it prints immediately (unbuffered).
    try stdout_file.writeAll(guide);

    // 3. Create a reader (New 0.15 syntax)
    // We must provide a backing buffer for the reader state.
    var stdin_reader_state: [4096]u8 = undefined;
    const stdin_reader = stdin_file.reader(&stdin_reader_state);

    // Access the generic Reader interface
    var reader = stdin_reader.interface;

    // 4. Read until delimiter
    // This is now a method on the Reader interface.
    const line = reader.takeDelimiterExclusive('\n') catch |err| {
        switch (err) {
            error.ReadFailed => {
                return error.ReadFailed;
            },
            error.EndOfStream => {
                return error.EndOfStream;
            },
            error.StreamTooLong => {
                return error.StreamTooLong;
            },
        }
    };

    // 5. Trim carriage return (\r) for Windows compatibility
    // Note: takeDelimiterExclusive returns a slice of the internal buffer.
    // We need to copy it if we want it to persist, but for this function returning a slice is tricky
    // because the slice points to 'stdin_reader_state' which is on the stack!
    if (line.len > 0) {
        // CRITICAL FIX: We must copy the result into the caller-provided 'buffer'
        if (line.len > buffer.len) {
            return error.StreamTooLong;
        }
        @memcpy(buffer[0..line.len], line);

        // Toss the delimiter from the stream
        // so it doesn't get read next time
        return std.mem.trimRight(
            u8,
            buffer[0..line.len],
            "\r",
        );
    }

    // EOF reached
    return "";
}
