app [main!] { pf: platform "../platform/main.roc", roc: "nightly-2026-09-24-f45bfbe" }

import pf.File
import pf.Stderr
import pf.Stdout

# args includes the executable name. Read a UTF-8 file and preserve its contents.
main! = |args| {
	match args {
		[_, path] => {
			contents = File.read_text!(path)?
			Stdout.write!(contents)?
			Ok({})
		}
		_ => {
			Stderr.line!("Usage: cat <path>")?
			Err(Exit(2))
		}
	}
}
