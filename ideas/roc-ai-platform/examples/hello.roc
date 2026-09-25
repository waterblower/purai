app [main!] { pf: platform "../platform/main.roc", roc: "nightly-2026-09-24-f45bfbe" }

import pf.Stdout

main! = |_args| {
	Stdout.line!("Hello from Roc!")?
	Ok({})
}
