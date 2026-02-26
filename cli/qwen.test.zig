const std = @import("std");
const qwen = @import("qwen.zig");
const gguf = @import("gguf");
const testing = std.testing;

test "QwenTokenizer special tokens and Chinese characters" {
    const allocator = testing.allocator;

    // 1. 指定你的 GGUF 模型文件的实际路径
    // 注意：请将此路径替换为你本地的真实路径
    const model_path = "./models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf";

    // 打开并读取模型
    const model = try gguf.Read(allocator, model_path);
    defer model.deinit();

    // 2. 初始化 Tokenizer
    var tokenizer = try qwen.Tokenizer.init(allocator, model);
    defer tokenizer.deinit();

    // 3. 测试特殊标识符 (Special Tokens)
    // 使用 expectEqualStrings 进行精确的字节级断言
    try testing.expectEqualStrings("<｜begin▁of▁sentence｜>", try tokenizer.decode(151643));
    try testing.expectEqualStrings("<|im_start|>", try tokenizer.decode(151644));

    for (151600..tokenizer.id_to_string.len) |i| {
        std.debug.print("{d}: {s}\n", .{ i, try tokenizer.decode(@intCast(i)) });
    }
}
