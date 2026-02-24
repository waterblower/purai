const std = @import("std");

// 这是一个“类型工厂”函数，它在编译期执行，返回一个自定义的 struct 类型
pub fn Matrix(comptime shape: []const usize) type {
    // 1. 在编译期计算一维扁平数组的总长度
    comptime var total_len: usize = 1;
    for (shape) |dim| {
        total_len *= dim;
    }

    // 2. 动态生成并返回一个全新的 struct 类型
    return struct {
        // 把 shape 存下来作为类型的关联常量，方便以后查阅
        pub const shape_info = shape;

        // 这里的 total_len 是在编译期就算好的常量
        data: [total_len]f32,

        // 我们甚至可以顺手写一个初始化所有元素为 0 的便捷函数
        pub fn initZeros() @This() {
            return .{
                // 使用 ** 运算符在编译期重复初始化数组
                .data = [_]f32{0.0} ** total_len,
            };
        }

        // 获取当前张量的元素总数
        pub fn len(_: @This()) usize {
            return total_len;
        }
    };
}

pub fn main() !void {
    // 实例化一个 2x3x4 的三维张量类型
    const MyTensorType = Matrix(&.{ 2, 3, 4 });

    // 创建这个张量的实例并初始化为 0
    var tensor = MyTensorType.initZeros();

    // 修改第一个元素
    tensor.data[0] = 3.14;

    std.debug.print("Tensor shape: {any}\n", .{MyTensorType.shape_info});
    std.debug.print("Tensor flat length: {d}\n", .{tensor.len()});
    std.debug.print("First element: {d}\n", .{tensor.data[0]});

    // 验证底层扁平数组的长度确实是 2 * 3 * 4 = 24
    std.debug.assert(tensor.data.len == 24);
}
