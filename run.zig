const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const debug = std.debug.print;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

pub fn main() !void {
    // 1. Setup Allocator
    const a = gpa.allocator();

    // 2. Parse Args
    const args = try std.process.argsAlloc(a);
    defer std.process.argsFree(a, args);

    // Default parameters
    var checkpoint_path: ?[]const u8 = null;
    var tokenizer_path: []const u8 = "tokenizer.bin";
    var temperature: f32 = 1.0;
    var topp: f32 = 0.9;
    var steps: i32 = 256;
    var prompt: []const u8 = ""; // Default to empty string (safe for Zig)
    var rng_seed: u64 = 0;
    var mode: []const u8 = "generate";
    var system_prompt: ?[]const u8 = null;

    // Basic arg validation
    if (args.len < 2) {
        error_usage();
    }
    checkpoint_path = args[1];

    // "Poor man's argparse" loop
    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        // Validation: Must have arg after flag
        if (i + 1 >= args.len) error_usage();

        const flag = args[i];
        const val = args[i + 1];

        // Validation: Flag must start with dash and be length 2 (-x)
        if (flag.len != 2 or flag[0] != '-') error_usage();

        switch (flag[1]) {
            't' => temperature = try std.fmt.parseFloat(f32, val),
            'p' => topp = try std.fmt.parseFloat(f32, val),
            's' => rng_seed = try std.fmt.parseInt(u64, val, 10),
            'n' => steps = try std.fmt.parseInt(i32, val, 10),
            'i' => prompt = val,
            'z' => tokenizer_path = val,
            'm' => mode = val,
            'y' => system_prompt = val,
            else => error_usage(),
        }
    }

    // Parameter overrides
    if (rng_seed == 0) rng_seed = @intCast(std.time.timestamp());
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 or topp > 1.0) topp = 0.9;
    if (steps < 0) steps = 0;

    // 3. Build Transformer
    // Note: Assuming build_transformer signature from previous context
    var t = try build_transformer(a, checkpoint_path.?);
    defer t.deinit(a);

    if (steps == 0 or steps > t.config.seq_len) steps = t.config.seq_len;

    // 4. Build Tokenizer
    var tokenizer = try Tokenizer.init(
        a,
        tokenizer_path,
        t.config.vocab_size,
    );
    defer tokenizer.deinit(a);

    // 5. Build Sampler
    // var sampler = try Sampler.init(a, t.config.vocab_size, temperature, topp, rng_seed);
    // defer sampler.deinit(a);

    // 6. Run Mode
    // if (std.mem.eql(u8, mode, "generate")) {
    //     try generate(&t, &tokenizer, &sampler, prompt, steps);
    // } else if (std.mem.eql(u8, mode, "chat")) {
    //     try chat(&t, &tokenizer, &sampler, prompt, system_prompt, steps);
    // } else {
    //     std.debug.print("unknown mode: {s}\n", .{mode});
    //     error_usage();
    // }
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

fn build_transformer(a: Allocator, checkpoint_path: []const u8) !*Transformer {
    // read in the Config and the Weights from the checkpoint
    const t = try read_checkpoint(a, checkpoint_path);
    // allocate the RunState buffers
    const state = try malloc_run_state(a, &t.config);
    t.state = state.*;
    return t;
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
    config: Config, //              the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState, //             buffers for the "wave" of activations in the forward pass

    file_size: u64, //         File sizes are unsigned 64-bit integers in std.fs
    data: []align(4096) u8, // We keep the original mmap result (bytes) for safe unmapping (munmap).
    //                         We cast this to floats strictly when assigning into 'weights'.

    pub fn deinit(self: *Transformer, a: std.mem.Allocator) void {
        // 1. Free RunState buffers
        // Note: s.k and s.v are pointers into key_cache/value_cache, so we don't free them directly.
        self.state.deinit(a);

        // 2. Free the main model data buffer
        // Since we switched to readAll + alignedAlloc, we use allocator.free.
        // If you revert to mmap later, change this to std.posix.munmap(self.data);
        if (self.data.len > 0) {
            a.free(self.data);
        }
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

    pub fn deinit(self: *RunState, a: std.mem.Allocator) void {
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
    }
};

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

const Tokenizer = struct {
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
        const bytes_read = try file.read(
            std.mem.asBytes(&t.max_token_length),
        );
        debug("bytes read: {d}\n", .{bytes_read});

        // 5. Read Vocab Loop
        // for (0..vocab_size) |i| {
        //     // Read Score (float)
        //     // We read 4 bytes as u32, then cast the bits to f32
        //     const score_bits = try reader.readInt(u32, .little);
        //     t.vocab_scores[i] = @bitCast(score_bits);

        //     // Read Len (int)
        //     const len_i32 = try reader.readInt(i32, .little);
        //     const len = @as(usize, @intCast(len_i32));

        //     // Allocate string buffer (len + 1 for null terminator)
        //     // C: malloc(len + 1)
        //     const str_buf = try a.alloc(u8, len + 1);

        //     // Read string bytes
        //     // C: fread(t->vocab[i], len, 1, file)
        //     try reader.readNoEof(str_buf[0..len]);

        //     // Add null terminator
        //     // C: t->vocab[i][len] = '\0';
        //     str_buf[len] = 0;

        //     // Assign to vocab array
        //     t.vocab[i] = str_buf;
        // }
        return t;
    }

    pub fn deinit(_: *Tokenizer, _: Allocator) void {
        return;
    }
};

const TokenIndex = struct {
    str: []u8, // char *str
    id: i32, // int id
};
