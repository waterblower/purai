platform "roc-ai-platform"
	requires {
		main! : List(Str) => Try({}, [Exit(I32), ..])
	}
	exposes [Stdin, Stdout, Stderr, File]
	packages { roc: "nightly-2026-09-24-f45bfbe" }
	provides { "roc_main": main_for_host! }
	hosted {
		"roc_stdout_write": Host.stdout_write!,
		"roc_stdout_line": Host.stdout_line!,
		"roc_stdout_write_bytes": Host.stdout_write_bytes!,
		"roc_stdout_flush": Host.stdout_flush!,
		"roc_stderr_write": Host.stderr_write!,
		"roc_stderr_line": Host.stderr_line!,
		"roc_stderr_write_bytes": Host.stderr_write_bytes!,
		"roc_stderr_flush": Host.stderr_flush!,
		"roc_stdin_line": Host.stdin_line!,
		"roc_stdin_read_text": Host.stdin_read_text!,
		"roc_stdin_read_bytes": Host.stdin_read_bytes!,
		"roc_file_read_text": Host.file_read_text!,
		"roc_file_read_bytes": Host.file_read_bytes!,
		"roc_file_write_text": Host.file_write_text!,
		"roc_file_append_text": Host.file_append_text!,
		"roc_file_write_bytes": Host.file_write_bytes!,
		"roc_file_append_bytes": Host.file_append_bytes!,
		"roc_file_exists": Host.file_exists!,
		"roc_file_remove": Host.file_remove!,
		"roc_file_create_dir_all": Host.file_create_dir_all!,
	}
	targets: {
		inputs_dir: "targets/",
		arm64mac: { inputs: ["libhost.a", app] },
		x64mac: { inputs: ["libhost.a", app] },
	}

import Host
import Stdin
import Stdout
import Stderr
import File

main_for_host! : List(Str) => I32
main_for_host! = |args| {
	match main!(args) {
		Ok({}) => 0
		Err(Exit(code)) => code
		Err(other) => {
			_ = Stderr.line!("${Str.inspect(other)}")
			1
		}
	}
}
