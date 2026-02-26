const std = @import("std");
const string = []const u8;
const gguf = @import("gguf");
const qwen = @import("qwen.zig");

pub fn run_model(a: std.mem.Allocator, gguf_path: []const u8) !void {
    const model = try gguf.Read(a, gguf_path);
    defer model.deinit();

    const config = try loadQwenConfig(model);

    std.debug.print("\n=== Model Blueprint ===\n", .{});
    std.debug.print("Layers (Blocks): {d}\n", .{config.block_count});
    std.debug.print("Hidden Dim: {d}\n", .{config.embedding_length});
    std.debug.print("FFN Dim: {d}\n", .{config.feed_forward_length});
    std.debug.print("Vocab Size: {d}\n", .{config.vocab_size});
    std.debug.print(
        "GQA Ratio: {d} Q heads per KV head\n",
        .{config.@"attention.head_count" / config.@"attention.head_count_kv"},
    );
    std.debug.print("=======================\n", .{});
    std.debug.print("{any}\n", .{config});

    // 第二步：装填权重
    var weights = try Qwen3_Global_Weights.Load(a, model, &config);
    defer weights.deinit();

    std.debug.print(
        "Successfully bound {d} global tensors and {d} transformer blocks.\n",
        .{ 3, weights.blocks.len },
    );

    var tokenizer = try qwen.Tokenizer.init(a, model);
    defer tokenizer.deinit();
    std.debug.print("Tokenizer loaded. Vocab size: {d}\n", .{tokenizer.id_to_string.len});

    // 测试一下！Qwen 的特殊 Token
    // 1. 测试常规基础词汇 (比如 " the" 或常见的中文切词)
    std.debug.print("ID 1083 string: {s}\n", .{try tokenizer.decode(151643)});

    // 2. 用十六进制透视 151643 和 151644
    for ([_]u32{ 151643, 151644 }) |id| {
        const str = try tokenizer.decode(id);
        std.debug.print("ID {d} (len={d}) Hex: ", .{ id, str.len });
        for (str) |b| {
            std.debug.print("{x:0>2} ", .{b});
        }
        std.debug.print("\n", .{});
    }

    // 第三步：申请大内存块与 FBA
    // const max_seq_len: usize = 1024 * 32;
    // var state = try RunState.init(a, &config, max_seq_len);
    // defer state.deinit(a);

    // std.debug.print("Memory allocated successfully. Total size: {d} MB\n", .{state.raw_memory.len / 1024 / 1024});

    // // 模拟前向传播（Forward Pass）循环
    // for (0..3) |token_index| {
    //     // 1. 获取 FBA 的接口
    //     const fba_allocator = state.fba.allocator();

    //     // 2. 瞬时切分计算所需的变量
    //     const transient = try TransientState.init(fba_allocator, &config, max_seq_len);

    //     // （这里将是你未来执行 matmul 等算子的地方）
    //     _ = transient;
    //     std.debug.print("Forward pass for token {d} prepared.\n", .{token_index});

    //     // 3. 计算结束，游标瞬间清零。为下一个 Token 计算释放出全部 Transient 内存
    //     state.fba.reset();
    // }
}

pub fn loadQwenConfig(model: *const gguf.GgufContext) !Qwen3_Config {
    // 1. 校验架构
    const arch_val = model.kv_map.get("general.architecture") orelse {
        return error.MissingArchitecture;
    };
    if (arch_val == .STRING and !std.mem.eql(u8, arch_val.STRING, "qwen3")) {
        std.log.err("Expected qwen3, but got {s}\n", .{arch_val.STRING});
        return error.WrongGeneralArchitecture;
    }

    // 2. 特殊处理：从 tokenizer 数组提取词表大小 (151936)
    var vocab_size: usize = 0;
    if (model.kv_map.get("tokenizer.ggml.tokens")) |tokens_val| {
        if (tokens_val == .ARRAY) {
            vocab_size = tokens_val.ARRAY.len;
        }
    } else {
        return error.MissingVocabSize;
    }

    // 3. 提取带 qwen3 前缀的超参数
    return Qwen3_Config{
        .context_length = try model.getU32("qwen3.context_length"),
        .embedding_length = try model.getU32("qwen3.embedding_length"),
        .feed_forward_length = try model.getU32("qwen3.feed_forward_length"),
        .block_count = try model.getU32("qwen3.block_count"),
        .@"attention.head_count" = try model.getU32("qwen3.attention.head_count"),
        .@"attention.head_count_kv" = try model.getU32("qwen3.attention.head_count_kv"),
        .vocab_size = vocab_size,
        .norm_rms_epsilon = try model.getF32("qwen3.attention.layer_norm_rms_epsilon"),
        .rope_freq_base = try model.getF32("qwen3.rope.freq_base"),
    };
}

pub const Qwen3_Config = struct {
    context_length: usize,
    embedding_length: usize, //           number of dimensions
    feed_forward_length: usize, //        number of hidden dimensions
    block_count: usize, //                number of layers (aka blocks)
    @"attention.head_count": usize, //    number of Query heads
    @"attention.head_count_kv": usize, // number of Key & Value heads
    vocab_size: usize,
    norm_rms_epsilon: f32,
    rope_freq_base: f32,
};

pub const Qwen3_Global_Weights = struct {
    a: std.mem.Allocator,
    token_embd: gguf.TensorInfo,
    // 当激活值穿透了全部 Blocks 之后，它的数值可能又变得很大或者方差很乱。
    // 在进行最终的概率预测前，必须做最后一次全局的 RMSNorm 归一化。
    output_norm: gguf.TensorInfo,
    // 它是 token_embedding 的逆向操作
    // 经过 output_norm 洗礼低维度向量，会去乘以这个巨大的矩阵，将其重新映射回高纬度的词表空间
    // 最终你会得到一个长度和Tokenizer维度一样的数组（Logits）
    // 里面数值最大的那个索引，就是模型输出的下一个字。
    output: gguf.TensorInfo,

    // 这是一个切片，长度将等于 config.block_count
    blocks: []Qwen3_BlockWeights,

    pub fn Load(
        a: std.mem.Allocator,
        model: *const gguf.GgufContext,
        config: *const Qwen3_Config,
    ) !Qwen3_Global_Weights {
        var weights: Qwen3_Global_Weights = undefined;

        // 1. 抓取全局独立张量
        weights.a = a;
        weights.token_embd = try model.getTensor("token_embd.weight");
        weights.output_norm = try model.getTensor("output_norm.weight");
        weights.output = try model.getTensor("output.weight");

        // 2. 为所有 Blocks 动态分配内存
        weights.blocks = try a.alloc(Qwen3_BlockWeights, config.block_count);
        errdefer a.free(weights.blocks);

        // 3. 遍历每一层，动态拼接字符串进行抓取
        for (0..config.block_count) |i| {
            // 分配一个小缓冲区来格式化张量名称，比如 "blk.0.attn_q.weight"
            var name_buf: [64]u8 = undefined;

            // 宏定义一个局部抓取函数，减少重复的格式化代码
            const fetch = struct {
                fn call(
                    m: *const gguf.GgufContext,
                    buf: []u8,
                    layer: usize,
                    suffix: []const u8,
                ) !gguf.TensorInfo {
                    const name = try std.fmt.bufPrint(buf, "blk.{d}.{s}.weight", .{ layer, suffix });
                    return m.getTensor(name);
                }
            }.call;

            weights.blocks[i] = .{
                .attn_norm = try fetch(model, &name_buf, i, "attn_norm"),
                .attn_q = try fetch(model, &name_buf, i, "attn_q"),
                .attn_k = try fetch(model, &name_buf, i, "attn_k"),
                .attn_v = try fetch(model, &name_buf, i, "attn_v"),
                .attn_output = try fetch(model, &name_buf, i, "attn_output"),
                .attn_q_norm = try fetch(model, &name_buf, i, "attn_q_norm"),
                .attn_k_norm = try fetch(model, &name_buf, i, "attn_k_norm"),
                .ffn_norm = try fetch(model, &name_buf, i, "ffn_norm"),
                .ffn_gate = try fetch(model, &name_buf, i, "ffn_gate"),
                .ffn_up = try fetch(model, &name_buf, i, "ffn_up"),
                .ffn_down = try fetch(model, &name_buf, i, "ffn_down"),
            };
        }

        return weights;
    }

    pub fn deinit(self: *Qwen3_Global_Weights) void {
        self.a.free(self.blocks);
    }
};

// 参考 https://lmstudio.ai/models/deepseek/deepseek-r1-0528-qwen3-8b
// blk.0.attn_k.weight        Q4_K  [4096, 1024]
// blk.0.attn_k_norm.weight   F32   [128]
// blk.0.attn_norm.weight     F32   [4096]
// blk.0.attn_output.weight   Q4_K  [4096, 4096]
// blk.0.attn_q.weight        Q4_K  [4096, 4096]
// blk.0.attn_q_norm.weight   F32   [128]
// blk.0.attn_v.weight        Q6_K  [4096, 1024]
// blk.0.ffn_down.weight      Q6_K  [12288, 4096]
// blk.0.ffn_gate.weight      Q4_K  [4096, 12288]
// blk.0.ffn_norm.weight      F32   [4096]
// blk.0.ffn_up.weight        Q4_K  [4096, 12288]
pub const Qwen3_BlockWeights = struct {
    attn_norm: gguf.TensorInfo, //   the input tensor, the output of the previous layer,
    //                               if this is the 1st block, it's the output of the embedding layer

    attn_output: gguf.TensorInfo,

    // [说明](./doc.md)
    attn_k: gguf.TensorInfo,
    attn_k_norm: gguf.TensorInfo,

    attn_q: gguf.TensorInfo, //    RMSNorm(attn_q, attn_q_norm);
    attn_q_norm: gguf.TensorInfo,

    attn_v: gguf.TensorInfo,

    ffn_norm: gguf.TensorInfo,
    ffn_gate: gguf.TensorInfo,
    ffn_up: gguf.TensorInfo,
    ffn_down: gguf.TensorInfo,
};

pub const RunState = struct {
    // 物理层面上唯一的一块真实内存
    raw_memory: []align(std.heap.pageSize()) u8,

    // 持久化状态 (Persistent State) - 生命周期贯穿整个对话
    key_cache: []f32,
    value_cache: []f32,

    // 瞬态内存分配器 (Transient Allocator) - 用于前向传播中的临时张量
    fba: std.heap.FixedBufferAllocator,

    pub fn init(a: std.mem.Allocator, config: *const Qwen3_Config, max_seq_len: usize) !RunState {
        const head_dim = config.embedding_length / config.@"attention.head_count";
        const kv_dim = config.@"attention.head_count_kv" * head_dim;

        // 1. 计算 KV Cache 需要的精确字节数
        const kv_cache_elements = config.block_count * max_seq_len * kv_dim;
        const kv_cache_bytes = kv_cache_elements * @sizeOf(f32);

        // 2. 估算单次 Forward 需要的瞬态内存峰值
        // x, q, k, v, hb, att, logits 等总和大概在 2MB 左右。分配 8MB 确保绝对安全。
        const transient_bytes = 8 * 1024 * 1024;

        // 3. 一次性申请对齐的连续大内存块 (64字节对齐，有利于 AVX/SIMD)
        const total_bytes = (kv_cache_bytes * 2) + transient_bytes;
        const raw_memory = try a.alignedAlloc(
            u8,
            std.mem.Alignment.fromByteUnits(std.heap.pageSize()),
            total_bytes,
        );

        // 4. 从大内存块头部切出 KV Cache，并将 []u8 强转为 []f32
        var offset: usize = 0;

        const key_slice_bytes = raw_memory[offset .. offset + kv_cache_bytes];
        const aligned_key_slice_bytes: []align(@alignOf(f32)) u8 = @alignCast(key_slice_bytes);
        const key_cache = std.mem.bytesAsSlice(f32, aligned_key_slice_bytes);
        offset += kv_cache_bytes;

        const value_slice_bytes = raw_memory[offset .. offset + kv_cache_bytes];
        const value_slice_bytes_bytes: []align(@alignOf(f32)) u8 = @alignCast(value_slice_bytes);
        const value_cache = std.mem.bytesAsSlice(f32, value_slice_bytes_bytes);
        offset += kv_cache_bytes;

        // 5. 将剩下的内存全部移交给 FixedBufferAllocator
        const fba_buffer = raw_memory[offset..];
        const fba = std.heap.FixedBufferAllocator.init(fba_buffer);

        return RunState{
            .raw_memory = raw_memory,
            .key_cache = key_cache,
            .value_cache = value_cache,
            .fba = fba,
        };
    }

    pub fn deinit(self: *RunState, a: std.mem.Allocator) void {
        // 整个释放过程只有这一行，彻底杜绝内存碎片
        a.free(self.raw_memory);
    }
};

pub const TransientState = struct {
    x: []f32,
    xb: []f32,
    xb2: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    hb: []f32,
    hb2: []f32,
    att: []f32,
    logits: []f32,

    pub fn init(allocator: std.mem.Allocator, config: *const Qwen3_Config, max_seq_len: usize) !TransientState {
        const dim = config.embedding_length;
        const hidden_dim = config.feed_forward_length;
        const head_dim = dim / config.@"attention.head_count";
        const kv_dim = config.@"attention.head_count_kv" * head_dim;

        return TransientState{
            .x = try allocator.alloc(f32, dim),
            .xb = try allocator.alloc(f32, dim),
            .xb2 = try allocator.alloc(f32, dim),
            .q = try allocator.alloc(f32, dim),
            .k = try allocator.alloc(f32, kv_dim),
            .v = try allocator.alloc(f32, kv_dim),
            .hb = try allocator.alloc(f32, hidden_dim),
            .hb2 = try allocator.alloc(f32, hidden_dim),
            .att = try allocator.alloc(f32, config.@"attention.head_count" * max_seq_len),
            .logits = try allocator.alloc(f32, config.vocab_size),
        };
    }
};
