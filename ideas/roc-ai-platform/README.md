# Roc AI platform

A small Roc platform with a Rust host for writing command-line programs and
collecting tested Roc examples for later AI model training. It provides synchronous
standard I/O and file I/O, with no third-party Rust dependencies.

Compiler: **`nightly-2026-09-24-f45bfbe`**, pinned in `roc-version` and every Roc
application header. This uses Roc's new compiler and direct-symbol host ABI.
Older alpha4 programs use a different API and syntax.

## Build and run

Requires Rust/Cargo, macOS command-line developer tools, and Python 3 for tests.
The current build targets are native Apple Silicon and Intel macOS. Apple Silicon
has been tested end to end; Intel is configured but untested. Linux and Windows
host/linker packaging are not implemented yet.

```sh
cd roc-ai-platform
./scripts/install-roc.sh  # downloads a checksum-pinned compiler into .tools/
./build.sh
./scripts/roc.sh build examples/hello.roc --output=target/hello
./target/hello
```

`scripts/roc.sh` uses `$ROC` if set, then `roc` on PATH, then the project-local
compiler. It checks the exact version and keeps the default compiler cache inside
`.tools/`. Set `ROC=/absolute/path/to/roc` to choose a compiler explicitly.
After setup, builds and tests need no network access.

```sh
./scripts/roc.sh build examples/echo.roc --output=target/echo
printf 'first\n\nlast\n' | ./target/echo

./scripts/roc.sh build examples/save_input.roc --output=target/save-input
printf 'training example\n' | ./target/save-input target/sample.txt

./scripts/roc.sh build examples/cat.roc --output=target/cat
./target/cat target/sample.txt

./scripts/roc.sh build examples/copy.roc --output=target/copy
./target/copy target/sample.txt target/copied.txt
```

These commands build standalone executables. Using `roc build` also avoids the
filesystem watcher required by this nightly's interactive `roc run` command.

## Write a Roc program

Place this in `examples/my_program.roc`:

```roc
app [main!] { roc: "nightly-2026-09-24-f45bfbe", pf: platform "../platform/main.roc" }

import pf.File
import pf.Stdin
import pf.Stdout

main! = |_args| {
    input = Stdin.read_text!({})?
    File.write_text!("output.txt", input)?
    Stdout.line!("Saved output.txt")?
    Ok({})
}
```

The platform path is relative to the `.roc` source. File operation paths are
relative to the **process working directory**. `main!` receives command-line
arguments including the executable name at index zero.

Every I/O operation returns `Try(value, [IoErr(Str)])`. Use `?` to propagate an
error, or `match` to recover. Diagnostics include the operation, path where
applicable, Rust error kind, and OS message. Diagnostic text is for people;
OS-dependent wording is not a stable format to parse.

`Ok({})` exits with status 0. `Err(Exit(code))` sets an explicit status (use
0–255 for portable shell behavior). Other uncaught errors are printed to stderr
and exit with status 1. A failure to flush output also changes a successful exit
to 1. Broken pipes return I/O errors instead of terminating through SIGPIPE.

## Standard I/O

| Function | Success value / behavior |
| --- | --- |
| `Stdin.line!({})` | `{ text: Str, eof: Bool }`; removes one LF or CRLF terminator |
| `Stdin.read_text!({})` | `Str`; reads remaining stdin to EOF, validating UTF-8 |
| `Stdin.read_bytes!({})` | `List(U8)`; reads remaining stdin to EOF unchanged |
| `Stdout.write!(text)` / `Stderr.write!(text)` | `{}`; writes text unchanged |
| `Stdout.line!(text)` / `Stderr.line!(text)` | `{}`; writes text followed by LF |
| `Stdout.write_bytes!(bytes)` / `Stderr.write_bytes!(bytes)` | `{}`; writes raw bytes |
| `Stdout.flush!({})` / `Stderr.flush!({})` | `{}`; flushes buffered output |

An empty input line returns `{ text: "", eof: False }`; EOF returns
`{ text: "", eof: True }`. An unterminated final line has `eof: False`; the
following read reports EOF. Other whitespace, including a lone trailing CR,
is preserved. Invalid UTF-8 is an error for text operations. For interactive
prompts, flush stdout before reading stdin.

## File I/O

| Function | Success value / behavior |
| --- | --- |
| `File.read_text!(path)` | `Str`; reads a whole UTF-8 file |
| `File.read_bytes!(path)` | `List(U8)`; reads a whole binary file |
| `File.write_text!(path, text)` | `{}`; creates or truncates a file |
| `File.write_bytes!(path, bytes)` | `{}`; creates or truncates a binary file |
| `File.append_text!(path, text)` | `{}`; appends, creating a missing file |
| `File.append_bytes!(path, bytes)` | `{}`; appends bytes, creating a missing file |
| `File.exists!(path)` | `Bool`; follows symlinks and propagates lookup errors |
| `File.remove!(path)` | `{}`; removes a file or symlink, errors if absent |
| `File.create_dir_all!(path)` | `{}`; creates missing parent directories |

Writes do not add newlines or create parent directories. They are not atomic
replacements and do not call `fsync`; an error can leave a partially written file.
Whole-file and read-to-EOF operations hold contents in memory. Paths are UTF-8
strings; non-UTF-8 command-line arguments are converted lossily to Roc strings.
File operations have the process's normal filesystem access.

## Tests and training examples

```sh
./test.sh
```

Runs Rust formatting, Clippy, three Rust unit tests, Roc formatting, and twelve
end-to-end tests that compile and execute all five examples plus `tests/io.roc`.
Checks include blank-line/EOF behavior, UTF-8 errors, binary round trips, append
and truncation, missing files, directories, exit codes, broken pipes, and repeated
host calls with shared Roc values. Test files live in temporary directories under
`target/integration` and are removed afterward.

For a future training corpus, keep each program with its input, expected stdout,
stderr, exit status, and any file fixtures. Use the pinned compiler and isolated
working directories, as the integration runner does. `examples/` is a small set
of working seed programs; this project does not yet generate or train a model.

## Internals and ABI changes

- `platform/`: public Roc modules, internal hosted signatures, and entrypoint.
- `src/effects.rs`: Rust I/O logic; `src/host.rs`: C ABI boundary and allocation.
- `src/roc_platform_abi.rs`: compiler-generated layouts, ownership helpers, and
  allocation helpers. Do not hand-edit.
- `scripts/RustGlue.roc`: the matching official Roc Rust glue generator.

After changing hosted signatures, run `./scripts/regenerate-glue.sh`, update the
Rust host to match, rebuild, and run `./test.sh`. Hosted arguments transfer an
owned Roc reference to Rust; the host uses generated `RocOwned` guards to release
arguments on success and error paths. Return values transfer ownership back to Roc.

The implementation follows [Roc's platform architecture](https://www.roc-lang.org/docs/main/langref/platforms/)
and the [Rust platform template](https://github.com/lukewilliamboswell/roc-platform-template-rust).
See `THIRD_PARTY_NOTICES.md` for exact source revisions and license notices.
