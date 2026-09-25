app [main!] { pf: platform "../platform/main.roc", roc: "nightly-2026-09-24-f45bfbe" }

import pf.File
import pf.Stdin
import pf.Stdout
import pf.Stderr

main! = |args| {
	match args {
		[_, "streams"] => {
			Stdout.write!("out")?
			Stdout.line!("put")?
			Stdout.write_bytes!([0, 255, 10])?
			Stdout.flush!({})?
			Stderr.write!("err")?
			Stderr.line!("or")?
			Stderr.write_bytes!([0, 254, 10])?
			Stderr.flush!({})?
			Ok({})
		}
		[_, "bytes"] => {
			data = Stdin.read_bytes!({})?
			Stdout.write_bytes!(data)?
			Ok({})
		}
		[_, "text"] => {
			data = Stdin.read_text!({})?
			Stdout.write!(data)?
			Ok({})
		}
		[_, "mixed-input"] => {
			first = Stdin.line!({})?
			Stdout.line!(first.text)?
			rest = Stdin.read_bytes!({})?
			Stdout.write_bytes!(rest)?
			Ok({})
		}
		[_, "write", path, value] => File.write_text!(path, value)
		[_, "append", path, value] => File.append_text!(path, value)
		[_, "append-bytes", path] => File.append_bytes!(path, [0, 255, 10])
		[_, "mkdir", path] => File.create_dir_all!(path)
		[_, "remove", path] => File.remove!(path)
		[_, "exists", path] => {
			found = File.exists!(path)?
			Stdout.line!(
				if found {
					"yes"
				} else {
					"no"
				},
			)?
			Ok({})
		}
		[_, "recover", path] => {
			match File.read_text!(path) {
				Ok(_) => Err(Exit(3))
				Err(IoErr(_)) => Stdout.line!("recovered")
			}
		}
		[_, "repeat", path] => {
			var $index = 0
			while $index < 300 {
				contents = File.read_bytes!(path)?
				File.write_bytes!("copy-one.bin", contents)?
				File.write_bytes!("copy-two.bin", contents)?
				$index = $index + 1
			}
			Ok({})
		}
		[_, "exit"] => Err(Exit(7))
		_ => Err(Exit(2))
	}
}
