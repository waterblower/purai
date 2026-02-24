const std = @import("std");
const log = std.log;
const eq = std.mem.eql;
const gguf = @import("gguf");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

pub fn main() !void {
    const a = gpa.allocator();
    // Example usage of your API
    const result = try parse_args(a);

    if (eq(u8, "quantize", result.command)) {
        var iter = result.flags.iterator();
        while (iter.next()) |entry| {
            log.info("{s}: {s}", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        const model = try gguf.Read(a, result.flags.get("m").?);
        defer model.deinit();

        const quantized_model = try model.quantize_to_Q4_0(a);
        defer quantized_model.deinit();
        try quantized_model.serialize(result.flags.get("o").?);
    }
}

fn parse_args(a: std.mem.Allocator) !ParsedCommand {
    // Get raw OS args
    const args = try std.process.argsAlloc(a);
    // args.deinit();

    return try parseCommandArgs(a, args[1..]);
}

/// The struct representing our parsed output
pub const ParsedCommand = struct {
    command: []const u8,
    flags: std.StringHashMap([]const u8),

    /// Helper to clean up the hash map's memory when we're done
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
