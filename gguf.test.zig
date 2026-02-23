const gguf = @import("./gguf.zig");
const std = @import("std");
const debug = std.debug.print;

test "serialize" {
    const allocator = std.testing.allocator;
    const model = try gguf.Read(allocator, "./models/test.gguf");
    defer model.deinit();

    try model.serialize("models/test-out.gguf");
    try expectFilesEqual("./models/test.gguf", "models/test-out.gguf");
}

test "quantize_to_Q4_0" {
    const allocator = std.testing.allocator;

    const path1 = "./models/GLM-OCR-f16.gguf";
    const path2 = "./models/GLM-OCR-Q4_0.gguf";

    const model = try gguf.Read(allocator, path1);
    defer model.deinit();

    const model2 = try model.quantize_to_Q4_0(allocator);
    defer model2.deinit();
    try model2.serialize(path2);
}

/// 逐字节比较两个文件是否完全一致
fn expectFilesEqual(path1: []const u8, path2: []const u8) !void {
    const file1 = try std.fs.cwd().openFile(path1, .{});
    defer file1.close();

    const file2 = try std.fs.cwd().openFile(path2, .{});
    defer file2.close();

    const stat1 = try file1.stat();
    const stat2 = try file2.stat();

    // 1. 第一道防线：文件大小必须一致
    try std.testing.expectEqual(stat1.size, stat2.size);

    // 2. 第二道防线：分块（Chunk）读取并对比
    // 使用一页大小的栈内存，零堆分配，极致速度
    var buf1: [std.heap.pageSize()]u8 = undefined;
    var buf2: [std.heap.pageSize()]u8 = undefined;

    var total_bytes_read: usize = 0;
    while (true) {
        const n1 = try file1.read(&buf1);
        const n2 = try file2.read(&buf2);

        // 读取的字节数必须相同
        try std.testing.expectEqual(n1, n2);

        if (n1 == 0) break; // 遇到 EOF (文件末尾)，对比结束

        // 核心断言：对比当前数据块是否完全一致
        // 使用 expectEqualSlices 极其重要！如果失败，它会精准打印出是哪一个 byte 不一样
        try std.testing.expectEqual(buf1, buf2);

        total_bytes_read += n1;
    }

    // 3. 最终确认：实际读取的字节数等于文件大小
    try std.testing.expectEqual(stat1.size, total_bytes_read);
}
