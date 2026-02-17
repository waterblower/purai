const std = @import("std");

pub fn build(b: *std.Build) void {
    llama2(b);
}

fn llama2(b: *std.Build) void {
    // 1. 目标架构 (Target)
    // 默认行为就是 'native'，利用当前 CPU 的所有特性 (AVX/SIMD)。
    // 这对矩阵乘法性能至关重要。
    const target = b.standardTargetOptions(.{});

    // 2. 优化等级 (Optimize)
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
