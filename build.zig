const std = @import("std");

pub fn build(b: *std.Build) void {
    // 1. 目标架构 (Target)
    // 默认行为就是 'native'，利用当前 CPU 的所有特性 (AVX/SIMD)。
    // 这对矩阵乘法性能至关重要。
    const target = b.standardTargetOptions(.{});

    // 2. 优化等级 (Optimize)
    // 将默认优化模式设置为 .ReleaseFast (相当于 -O3 -fno-sanitize)
    // 这意味着你只需运行 'zig build' 而不需要加参数，就能得到最快版本。
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // 3. 定义可执行文件
    const exe = b.addExecutable(
        .{
            .name = "run",
            .root_module = b.createModule(.{
                .root_source_file = b.path("run.zig"),
                .target = target,
                .optimize = optimize,
            }),
        },
    );

    // 4. 激进优化选项
    // 去除符号表 (减小体积)
    exe.root_module.strip = true;

    // 如果你有链接其他 C 库，开启 LTO (Link Time Optimization) 会有帮助
    // 但对于纯 Zig 单文件，这个选项影响不大，开着也无妨
    // exe.want_lto = true;

    // 安装到 zig-out/bin
    b.installArtifact(exe);

    // 5. 添加 'run' 命令支持 (zig build run -- args)
    const run_cmd = b.addRunArtifact(exe);

    // 确保运行前先编译
    run_cmd.step.dependOn(b.getInstallStep());

    // 允许传递参数
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
