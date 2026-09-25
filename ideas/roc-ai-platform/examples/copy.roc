app [main!] { pf: platform "../platform/main.roc", roc: "nightly-2026-09-24-f45bfbe" }

import pf.File
import pf.Stderr

# Copies any file, including binary data. The destination is overwritten.
main! = |args| {
	match args {
		[_, source, destination] => {
			contents = File.read_bytes!(source)?
			File.write_bytes!(destination, contents)?
			Ok({})
		}
		_ => {
			Stderr.line!("Usage: copy <source> <destination>")?
			Err(Exit(2))
		}
	}
}
