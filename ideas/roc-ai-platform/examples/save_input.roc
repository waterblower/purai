app [main!] { pf: platform "../platform/main.roc", roc: "nightly-2026-09-24-f45bfbe" }

import pf.File
import pf.Stdin
import pf.Stderr

# Save stdin as UTF-8 text; useful for small dataset preparation programs.
main! = |args| {
	match args {
		[_, destination] => {
			contents = Stdin.read_text!({})?
			File.write_text!(destination, contents)?
			Ok({})
		}
		_ => {
			Stderr.line!("Usage: save_input <destination>")?
			Err(Exit(2))
		}
	}
}
