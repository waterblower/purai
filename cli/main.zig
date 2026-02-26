const builtin = @import("builtin");
const std = @import("std");
const log = std.log;
const mem = std.mem;
const eq = std.mem.eql;
const gguf = @import("gguf");
const run_model = @import("./run_model.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

pub fn main() !void {
    if (builtin.os.tag == .windows) {
        // 65001 是 Windows 中 UTF-8 的代码页 (Code Page) 标识
        _ = std.os.windows.kernel32.SetConsoleOutputCP(65001);
    }

    const a = gpa.allocator();
    defer _ = gpa.deinit();

    const args = try std.process.argsAlloc(a);
    defer std.process.argsFree(a, args);

    var result = try parseCommandArgs(a, args[1..]);
    defer result.deinit();

    if (result.is("quantize")) {
        try gguf.quantize(a, result.flags.get("m").?, result.flags.get("o").?);
    } else if (result.is("print")) {
        try gguf.print(a, result.flags.get("m").?);
    } else if (result.is("run")) {
        try run_model.run_model(a, result.flags.get("m").?);
    } else {
        // print the help message
    }
}

/// The struct representing our parsed output
pub const ParsedCommand = struct {
    command: []const u8,
    flags: std.StringHashMap([]const u8),

    pub fn is(self: ParsedCommand, cmd: []const u8) bool {
        return mem.eql(u8, self.command, cmd);
    }

    pub fn deinit(self: *ParsedCommand) void {
        self.flags.deinit();
    }
};

/// Parses an array of string arguments into a ParsedCommand struct
pub fn parseCommandArgs(
    allocator: std.mem.Allocator,
    args: []const []const u8,
) !ParsedCommand {
    if (args.len == 0) {
        return error.EmptyArguments;
    }

    // Initialize the map. If parsing fails later, errdefer cleans it up automatically.
    var flags = std.StringHashMap([]const u8).init(allocator);
    errdefer flags.deinit();

    // First element is always the command
    const command = args[0];

    // Iterate through the rest of the array in pairs
    var i: usize = 1;
    while (i < args.len) {
        const current_arg = args[i];

        // Check if the current argument is a flag (starts with '-')
        if (std.mem.startsWith(u8, current_arg, "-")) {
            // Strip any leading dashes to turn "-m" or "--m" into "m"
            var key = current_arg;
            while (std.mem.startsWith(u8, key, "-")) {
                key = key[1..];
            }

            // Ensure there is a corresponding value argument
            if (i + 1 >= args.len) {
                return error.MissingFlagValue;
            }

            const value = args[i + 1];

            // Insert into our hashmap
            try flags.put(key, value);

            // Jump ahead by 2 to skip the value we just consumed
            i += 2;
        } else {
            // If it doesn't start with '-', it breaks the expected `-flag value` pattern
            log.debug("{s}", .{current_arg});
            return error.UnexpectedArgument;
        }
    }

    return ParsedCommand{
        .command = command,
        .flags = flags,
    };
}
