const gguf = @import("./gguf.zig");
const std = @import("std");

test "always succeeds" {
    const allocator = std.testing.allocator;
    const model = try gguf.Read(allocator, "./models/test.gguf");
    defer model.deinit();
}
