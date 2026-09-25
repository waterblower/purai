app [main!] { pf: platform "../platform/main.roc", roc: "nightly-2026-09-24-f45bfbe" }

import pf.Stdin
import pf.Stdout

# Echo lines, including blank lines. Normalize LF/CRLF to LF.
main! = |_args| {
	var $reading = True
	while $reading {
		line = Stdin.line!({})?
		if line.eof {
			$reading = False
		} else {
			Stdout.line!(line.text)?
		}
	}
	Ok({})
}
