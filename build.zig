const std = @import("std");
const Build = std.Build;
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(
        .{ .default_target = .{} },
    );
    const optimize = b.standardOptimizeOption(.{});

    // Build Internal Libraries
    const gguf = b.createModule(.{
        .root_source_file = b.path("lib/gguf.zig"),
        .target = target,
    });
    main_cli(b, target, optimize, gguf);
    llama2(b, target, optimize);
    gpu_test_step(b, target);
    qwen_test_step(b, target, optimize, gguf);
}

fn main_cli(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    gguf: *std.Build.Module,
) void {
    const exe = b.addExecutable(.{
        .name = "ai",
        .root_module = b.createModule(.{
            .root_source_file = b.path("cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("gguf", gguf);

    // Prepare Artifacts
    const install_artifact = b.addInstallArtifact(exe, .{});
    const run_artifact = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_artifact.addArgs(args);
    }

    // The Run Step
    const run_step = b.step("ai", "Build and run the main AI cli");
    run_step.dependOn(&install_artifact.step);
    run_step.dependOn(&run_artifact.step);
}

fn llama2(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const exe = b.addExecutable(.{
        .name = "run",
        .root_module = b.createModule(.{
            .root_source_file = b.path("run.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

// ==========================================
// Qwen Tokenizer 单元测试
// ==========================================
fn qwen_test_step(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    gguf: *std.Build.Module,
) void {
    const qwen_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("cli/qwen.test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    qwen_test.root_module.addImport("gguf", gguf);

    const run_qwen_test = b.addRunArtifact(qwen_test);

    const qwen_step = b.step("test-qwen", "Run Qwen tokenizer tests");
    qwen_step.dependOn(&run_qwen_test.step);
}

// ==========================================
// 全新的纯 Zig GPU 构建与测试管线
// ==========================================
fn gpu_test_step(b: *std.Build, target: std.Build.ResolvedTarget) void {

    // 1. 核心操作：调用 Zig 原生编译器生成 PTX
    // 我们强制利用 Zig 的底层的 LLVM NVPTX 后端生成 NVIDIA 汇编
    // 修改点：使用 -femit-asm=kernel.ptx 来指定输出文件名，并用 -fno-emit-bin 阻止生成 .o 文件
    const build_ptx = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-obj",
        "-O",
        "ReleaseFast",
        "-target",
        "nvptx64-cuda",
        "-mcpu=sm_86",
        "-femit-asm=kernel.ptx", // 关键修改 1：输出汇编（即 PTX）并命名为 kernel.ptx
        "-fno-emit-bin", // 关键修改 2：不需要生成底层的 .o 二进制对象文件
        "kernel.zig",
    });
    // 2. 编译测试主机程序 (kernel.test.zig)
    const gpu_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("kernel.test.zig"),
            .target = target,
            .optimize = null,
        }),
    });

    // 3. 链接 C 标准库和 CUDA Driver API
    gpu_test.linkLibC();
    gpu_test.linkSystemLibrary("cuda");

    if (builtin.os.tag == .windows) {
        // === 【Windows 专属自动寻址配置】 ===
        // 自动读取 Windows 的 CUDA_PATH 环境变量（如 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x）
        if (std.process.getEnvVarOwned(b.allocator, "CUDA_PATH")) |cuda_path| {
            defer b.allocator.free(cuda_path);

            // 拼接 include 和 lib 目录路径
            const include_dir = b.pathJoin(&.{ cuda_path, "include" });
            const lib_dir = b.pathJoin(&.{ cuda_path, "lib", "x64" });

            gpu_test.addIncludePath(.{ .cwd_relative = include_dir });
            gpu_test.addLibraryPath(.{ .cwd_relative = lib_dir });
        } else |_| {
            std.debug.print("⚠️ Warning: can't find cuda\n", .{});
        }
    }

    // 4. 依赖绑定：必须先生成 kernel.ptx，再运行主机测试！
    gpu_test.step.dependOn(&build_ptx.step);

    const run_gpu_test = b.addRunArtifact(gpu_test);

    // 5. 注册指令：zig build gpu
    const gpu_step = b.step("gpu", "Build PTX and run GPU tests");
    gpu_step.dependOn(&run_gpu_test.step);
}
