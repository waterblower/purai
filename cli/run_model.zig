const std = @import("std");
const string = []const u8;
const gguf = @import("gguf");

pub fn run_model(a: std.mem.Allocator, gguf_path: []const u8) !void {
    const model = try gguf.Read(a, gguf_path);
    defer model.deinit();

    const config = try loadQwenConfig(model);

    std.debug.print("\n=== Model Blueprint ===\n", .{});
    std.debug.print("Layers (Blocks): {d}\n", .{config.block_count});
    std.debug.print("Hidden Dim: {d}\n", .{config.embedding_length});
    std.debug.print("FFN Dim: {d}\n", .{config.feed_forward_length});
    std.debug.print("Vocab Size: {d}\n", .{config.vocab_size});
    std.debug.print("GQA Ratio: {d} Q heads per KV head\n", .{config.@"attention.head_count" / config.@"attention.head_count_kv"});
    std.debug.print("=======================\n", .{});
    std.debug.print("{any}\n", .{config});

    // 接下来这这里做什么？
}

pub fn loadQwenConfig(model: *const gguf.GgufContext) !Qwen3Config {
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
    return Qwen3Config{
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

pub const Qwen3Config = struct {
    embedding_length: usize, //           number of dimensions
    feed_forward_length: usize, //        number of hidden dimensions
    block_count: usize, //                number of layers (aka blocks)
    @"attention.head_count": usize, //    number of Query heads
    @"attention.head_count_kv": usize, // number of Key & Value heads
    vocab_size: usize,
    norm_rms_epsilon: f32,
    rope_freq_base: f32,
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
