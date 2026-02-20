const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const debug = std.debug.print;
const asBytes = std.mem.asBytes;
const eql = std.mem.eql;
const Tokenizer = @import("./token.zig").Tokenizer;
const TokenIndex = @import("./token.zig").TokenIndex;
const matrix = @import("./matrix.zig");
const matmul = matrix.matmul;
const softmax = matrix.softmax;
const rope = matrix.rope;
const root_mean_square_normalization = matrix.root_mean_square_normalization;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const gguf = @import("./gguf.zig");

pub fn main() !void {
    // 1. Setup Allocator
    const a = gpa.allocator();
    defer _ = gpa.deinit();

    // 2. Parse Args
    var args = try parseArgs(a);

    // Load GGUF
    // const model = try gguf.GgufContext.init(a, args.gguf_path);
    // defer model.deinit();
    // debug("{f}\n", .{model});

    // 3. Build Transformer
    // Note: Assuming build_transformer signature from previous context
    debug("model path: {s}\n", .{args.checkpoint_path});
    var t = try Transformer.build(a, args.checkpoint_path);
    defer t.deinit();

    if (args.steps == 0 or args.steps > t.config.seq_len) {
        args.steps = t.config.seq_len;
    }
    debug("steps: {d}\n", .{args.steps});

    // 4. Build Tokenizer
    var tokenizer = try Tokenizer.init(
        a,
        args.tokenizer_path,
        t.config.vocab_size,
    );
    defer tokenizer.deinit();

    // 5. Build Sampler
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
    checkpoint_path: []const u8 = "./models/stories15M.bin",
    gguf_path: []const u8 = "./models/llama2.gguf",
    tokenizer_path: []const u8 = "./models/tokenizer.bin",
    temperature: f32 = 1.0,
    topp: f32 = 0.9,
    steps: i32 = 256,
    prompt: []const u8 = "",
    rng_seed: u64 = 0,
    mode: []const u8 = "generate",
    system_prompt: ?[]const u8 = null,
};

// work in progress
const Arg = struct {
    name: []const u8,
    short: ?u8,
    default: union(enum) {
        bool,
        i64,
        string: []const u8,
    },
};

// 2. 解析函数
fn parseArgs(a: Allocator) !CliArgs {
    const unparsed_args = try std.process.argsAlloc(a);
    defer std.process.argsFree(a, unparsed_args);

    for (unparsed_args) |arg| {
        std.debug.print("{s} ", .{arg});
    }
    std.debug.print("\n", .{});

    // Basic arg validation
    if (unparsed_args.len < 2) {
        error_usage();
    }

    // Initialize with defaults and the mandatory checkpoint path
    var args = CliArgs{};

    // "Poor man's argparse" loop
    var i: usize = 1;
    while (i < unparsed_args.len) : (i += 2) {
        // Validation: Must have arg after flag
        if (i + 1 >= unparsed_args.len) error_usage();

        const flag = unparsed_args[i];
        const val = unparsed_args[i + 1];

        // Validation: Flag must start with dash and be length 2 (-x)
        if (flag.len != 2 or flag[0] != '-') error_usage();

        switch (flag[1]) {
            't' => args.temperature = try std.fmt.parseFloat(f32, val),
            'p' => args.topp = try std.fmt.parseFloat(f32, val),
            's' => args.rng_seed = try std.fmt.parseInt(u64, val, 10),
            'n' => args.steps = try std.fmt.parseInt(i32, val, 10),
            'i' => args.prompt = val,
            'z' => args.tokenizer_path = val,
            'm' => args.mode = val,
            'y' => args.system_prompt = val,
            else => error_usage(),
        }
    }

    // Parameter overrides / Validation logic
    if (args.rng_seed == 0) args.rng_seed = @intCast(std.time.timestamp());
    if (args.temperature < 0.0) args.temperature = 0.0;
    if (args.topp < 0.0 or args.topp > 1.0) args.topp = 0.9;
    if (args.steps < 0) args.steps = 0;

    return args;
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
    var fd = try std.fs.cwd().openFile(
        checkpoint_path,
        .{ .mode = .read_only },
    );
    defer fd.close();

    // 3. Read Config
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
    const stat = try fd.stat();
    t.file_size = stat.size;

    try fd.seekTo(0);

    // 6. Memory Map
    if (comptime builtin.os.tag == .windows) {
        // This was missing. We ask for 'file_size' bytes aligned to 4096.
        t.data = try a.alignedAlloc(
            u8,
            std.mem.Alignment.fromByteUnits(4096),
            t.file_size,
        );
        // Safety: If readAll fails, free this memory
        errdefer a.free(t.data);
        const bytes_read = try fd.readAll(t.data);
        if (bytes_read != t.file_size) {
            return error.IncompleteRead;
        }
    } else {
        t.data = try std.posix.mmap(
            null,
            t.file_size,
            std.posix.PROT.READ,
            std.posix.MAP{ .TYPE = .PRIVATE },
            fd.handle,
            0,
        );
        errdefer std.posix.munmap(t.data);
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
        const state = try RunState.init(a, &t.config);
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
        if (comptime builtin.os.tag == .windows) {
            if (self.data.len > 0) {
                self.a.free(self.data);
            }
        } else {
            std.posix.munmap(self.data);
        }

        self.a.destroy(self);
    }
};

// extern is important here because we want to fix the fields ordering
// for binary file reading
const Config = extern struct {
    // transformer dimension
    // 主干道的宽度 (The Residual Stream)
    // 这是模型中每一个 Token（词）被表示成的向量长度。
    dim: i32,
    // 这是 Feed-Forward Network (FFN/MLP) 内部的隐藏层维度。
    // 数据流进入 FFN 时，会被暂时“投影”到一个更高的维度（通常是 dim 的 4 倍或者更多）
    // 在这里进行非线性变换（SwiGLU / ReLU），提取更复杂的特征。
    // 然后再“投影”回 dim，为了能加回到主干道上。
    hidden_dim: i32,
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
    // k: []f32, //   key (dim,)
    // v: []f32, //   value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits

    // kv cache
    key_cache: []f32, //   (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)

    fn init(a: std.mem.Allocator, p: *const Config) !*RunState {
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

        return s;
    }

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
    const num_prompt_tokens = try tokenizer.encode(
        a,
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
    var timer = try std.time.Timer.start();
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
        const piece = try tokenizer.decode(token, next);

        // safe_printf logic: Zig 的 print 默认处理 utf8，如果 piece 包含非打印字符可能需要过滤，
        // 但通常直接输出即可。
        debug("{s}", .{piece});

        // 刷新缓冲区确保实时显示 (Zig 的 stdout 默认可能有缓冲)
        // 这里的 flush 取决于你具体的 stdout 实现，简单起见可以忽略，或使用 std.io.bufferedWriter 的 flush

        token = next;
    }
    debug("\n", .{});

    // report achieved tok/s
    if (pos > 1) {
        // 计算耗时 (秒)
        const duration = @as(f64, @floatFromInt(timer.read())) / std.time.ns_per_s;
        const tokens = @as(f64, @floatFromInt(pos - 1));
        const tokens_per_sec = tokens / duration;
        debug("achieved tok/s: {d:.4}\n", .{tokens_per_sec});
    }
}

fn forward(transformer: *Transformer, token: i32, pos: usize) []f32 {
    const config = &transformer.config;
    const weights = &transformer.weights;
    const run_state = transformer.state;

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
    const content_row = weights.token_embedding_table[token_idx * dim .. (token_idx + 1) * dim];
    @memcpy(run_state.x, content_row);

    const seq_len: usize = @intCast(config.seq_len);

    // 2. Forward all layers
    for (0..@intCast(config.n_layers)) |layer_index| {
        // --- Attention Block ---

        // RMSNorm
        // [FIX] Explicit slicing for weights
        root_mean_square_normalization(
            run_state.xb,
            run_state.x,
            weights.rms_att_weight[layer_index * dim .. (layer_index + 1) * dim],
        );

        // Key/Value Cache Offsets
        const loff = layer_index * seq_len * kv_dim;
        const k_cache_offset = loff + pos * kv_dim;
        const v_cache_offset = loff + pos * kv_dim;

        // Target slices in the cache
        const k_target = run_state.key_cache[k_cache_offset .. k_cache_offset + kv_dim];
        const v_target = run_state.value_cache[v_cache_offset .. v_cache_offset + kv_dim];

        // QKV Matmuls
        // wq: (dim, dim)
        matmul(
            run_state.q,
            run_state.xb,
            weights.wq[layer_index * dim * dim .. (layer_index + 1) * dim * dim],
            dim,
            dim,
        );
        // wk: (kv_dim, dim)
        matmul(
            k_target,
            run_state.xb,
            weights.wk[layer_index * dim * kv_dim .. (layer_index + 1) * dim * kv_dim],
            kv_dim,
            dim,
        );
        // wv: (kv_dim, dim)
        matmul(
            v_target,
            run_state.xb,
            weights.wv[layer_index * dim * kv_dim .. (layer_index + 1) * dim * kv_dim],
            kv_dim,
            dim,
        );

        rope(run_state.q, k_target, dim, head_size, pos, kv_dim);

        // Multihead Attention
        multihead_attention(
            pos,
            n_heads,
            head_size,
            loff,
            kv_dim,
            kv_mul,
            run_state,
            config,
        );

        // Output Projection (Wo)
        // [FIX] Explicit slicing: [l*dim*dim .. (l+1)*dim*dim]
        matmul(run_state.xb2, run_state.xb, weights.wo[layer_index * dim * dim .. (layer_index + 1) * dim * dim], dim, dim);

        // Residual Connection 1
        for (0..dim) |idx| {
            run_state.x[idx] += run_state.xb2[idx];
        }

        // --- Feed Forward Block ---

        // FFN RMSNorm
        root_mean_square_normalization(run_state.xb, run_state.x, weights.rms_ffn_weight[layer_index * dim .. (layer_index + 1) * dim]);

        // Calculate offsets for FFN weights
        // w1, w2, w3 all have size (dim * hidden_dim) per layer
        const ffn_offset = layer_index * dim * hidden_dim;
        const ffn_size = dim * hidden_dim;

        // W1 (Gate) & W3 (Value)
        // [FIX] Explicit slicing using calculated offsets
        matmul(run_state.hb, run_state.xb, weights.w1[ffn_offset .. ffn_offset + ffn_size], hidden_dim, dim);
        matmul(run_state.hb2, run_state.xb, weights.w3[ffn_offset .. ffn_offset + ffn_size], hidden_dim, dim);

        // SwiGLU Activation
        for (0..hidden_dim) |idx| {
            var val = run_state.hb[idx];
            // silu(x) = x * sigmoid(x)
            val *= (1.0 / (1.0 + std.math.exp(-val)));
            // elementwise multiply with w3 output
            val *= run_state.hb2[idx];
            run_state.hb[idx] = val;
        }

        // W2 (Projection)
        // [FIX] Explicit slicing
        matmul(run_state.xb, run_state.hb, weights.w2[ffn_offset .. ffn_offset + ffn_size], dim, hidden_dim);

        // Residual Connection 2
        for (0..dim) |idx| {
            run_state.x[idx] += run_state.xb[idx];
        }
    }

    // 3. Final RMSNorm
    // [FIX] Explicit slicing
    root_mean_square_normalization(run_state.x, run_state.x, weights.rms_final_weight[0..dim]);

    // 4. Classifier
    // [FIX] Explicit slicing: wcls size is vocab_size * dim
    const vocab_size = @as(usize, @intCast(config.vocab_size));
    matmul(run_state.logits, run_state.x, weights.wcls[0 .. vocab_size * dim], vocab_size, dim);

    return run_state.logits;
}

fn multihead_attention(
    pos: usize, //       当前正在生成的 Token 在序列中的索引
    n_heads: usize,
    head_size: usize, // 单个注意力头的向量维度大小
    // 当前层（Layer）在全局 KV Cache 中的起始内存偏移量。
    // 为了避免碎片化，整个模型所有层（例如 32 层）的 KV Cache 通常被分配在一个巨大的连续一维数组中。
    layer_offset: usize,
    kv_dim: usize,
    // KV 广播倍数 / GQA 分组比例
    // 这是 Grouped Query Attention (GQA) 的关键参数。它表示多少个 Query Head 共享同一个 KV Head。
    // 如果 Query 有 32 个头，KV 只有 8 个头，那么 kv_mul = 4。
    // 这意味着 Head 0, 1, 2, 3 都去读 KV Head 0 的数据。
    kv_mul: usize,
    run_state: *const RunState,
    config: *const Config,
) void {
    // 优化 3: 预计算缩放因子
    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_size)));

    // SIMD 设置
    const vec_len = std.simd.suggestVectorLength(f32) orelse 8;
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
            const k_offset = layer_offset + t * kv_dim + head_offset_in_kv;
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
            const v_offset = layer_offset + t * kv_dim + head_offset_in_kv;
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

fn sample_argmax(probabilities: []f32) usize {
    // return the index that has the highest probability
    var max_i: usize = 0;
    var max_p: f32 = probabilities[0];
    for (1..probabilities.len) |i| {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

fn sample_mult(probabilities: []f32, coin: f32) usize {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1)
    var cdf: f32 = 0.0;
    for (0..probabilities.len) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return probabilities.len - 1; // in case of rounding errors
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
        next = @intCast(sample_argmax(logits));
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
            next = @intCast(sample_mult(logits, coin));
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler.topp, sampler.probindex, coin);
        }
    }

    return next;
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
            num_prompt_tokens = try tokenizer.encode(a, rendered, true, false, prompt_tokens);
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
            const piece = try tokenizer.decode(token, next);
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
