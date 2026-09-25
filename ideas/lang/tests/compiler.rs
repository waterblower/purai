use lang::{check, compile};

#[test]
fn rejects_invalid_programs_with_source_locations() {
    for (source, expected) in [
        ("", "missing `fn main()`"),
        ("fn main() { missing() }", "unknown function"),
        ("fn main() {} fn main() {}", "duplicate function"),
        ("fn print() {} fn main() {}", "reserved builtin"),
        ("fn main() { println() }", "one `str` argument"),
        ("fn main() { println(unknown) }", "unknown string binding"),
        ("fn main() { let x = x }", "unknown string binding"),
        ("fn main() { let x = \"value\" x() }", "not callable"),
        ("fn main() { main(\"bad\") }", "expects no arguments"),
        ("fn main() { println(\"oops) }", "unterminated string"),
        ("fn main() { println(\"\\q\") }", "unsupported escape"),
        ("fn main() {", "expected `}`"),
        ("fn main() {} stray", "expected `fn`"),
        ("fn main(x) {}", "parameters are not implemented"),
        ("fn main() { println(\"你好\") 💥 }", "unexpected character"),
    ] {
        let error = compile(source).unwrap_err();
        assert!(error.message.contains(expected), "{source:?}: {error:?}");
        let rendered = error.render("test.lang", source);
        assert!(rendered.starts_with("test.lang:"));
        assert!(rendered.contains('^'));
    }
}

#[test]
fn supports_bindings_shadowing_comments_and_forward_calls() {
    check(
        r#"
        // binding initializers resolve against the preceding scope
        fn main() {
            let a = "first";
            let b = a
            let a = "second"
            println(b)
            println(a);
            later()
        }
        fn later() { print("") }
    "#,
    )
    .unwrap();
}

#[test]
fn emits_deterministic_arm64_macho() {
    let source = include_str!("../examples/hello.lang");
    let bytes = compile(source).unwrap();
    assert_eq!(&bytes[..4], &0xfeedfacf_u32.to_le_bytes());
    assert_eq!(&bytes[4..8], &0x0100000c_u32.to_le_bytes());
    assert_eq!(bytes, compile(source).unwrap());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::*;
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        path::PathBuf,
        process::{Command, Stdio},
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lang-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn executable(&self, source: &str) -> PathBuf {
            let output = self.0.join("program");
            fs::write(&output, compile(source).unwrap()).unwrap();
            fs::set_permissions(&output, fs::Permissions::from_mode(0o755)).unwrap();
            output
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn assert_output(source: &str, expected: &[u8]) {
        let temp = Temp::new();
        let output = Command::new(temp.executable(source))
            .env("PATH", "")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{:?}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected);
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn hello_world_runs() {
        assert_output(include_str!("../examples/hello.lang"), b"Hello, world!\n");
    }

    #[test]
    fn nested_calls_preserve_return_addresses_and_utf8() {
        assert_output(
            r#"
            fn main() { first() println("done") }
            fn first() { second() second() }
            fn second() {
                let text = "你好\tworld\n"
                print("") print(text) print("\0\"\\")
            }
        "#,
            "你好\tworld\n\0\"\\你好\tworld\n\0\"\\done\n".as_bytes(),
        );
    }

    #[test]
    fn long_strings_cross_code_signing_and_vm_pages() {
        let text = "こんにちは!".repeat(20_000);
        // >64 KiB also exercises MOVK, multi-page hashes, ADRP relocations,
        // and printing through a pipe larger than its usual capacity.
        assert_output(
            &format!("fn main() {{ print(\"{text}\") println(\"end\") }}"),
            format!("{text}end\n").as_bytes(),
        );
    }

    #[test]
    fn write_failure_has_nonzero_exit_status() {
        let temp = Temp::new();
        let status = Command::new(temp.executable("fn main() { println(\"fail\") }"))
            .stdout(Stdio::from(fs::File::open("/dev/null").unwrap()))
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(1));
    }

    #[test]
    fn cli_build_and_rebuild_work_without_path_tools() {
        let temp = Temp::new();
        let source = temp.0.join("input.lang");
        let output = temp.0.join("output");
        for expected in ["first build", "second build"] {
            fs::write(&source, format!("fn main() {{ println(\"{expected}\") }}")).unwrap();
            let built = Command::new(env!("CARGO_BIN_EXE_lang"))
                .env("PATH", "")
                .arg("build")
                .arg(&source)
                .arg("-o")
                .arg(&output)
                .output()
                .unwrap();
            assert!(
                built.status.success(),
                "{}",
                String::from_utf8_lossy(&built.stderr)
            );
            let result = Command::new(&output).env("PATH", "").output().unwrap();
            assert!(result.status.success());
            assert_eq!(result.stdout, format!("{expected}\n").as_bytes());
        }
    }

    #[test]
    fn cli_rejects_errors_before_creating_output_and_preserves_source() {
        let temp = Temp::new();
        let source = temp.0.join("input.lang");
        let output = temp.0.join("output");
        fs::write(&source, "fn main() { nope() }").unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_lang"))
            .arg("build")
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(!output.exists());
        assert!(String::from_utf8_lossy(&result.stderr).contains("unknown function"));
        let valid = "fn main() {}";
        fs::write(&source, valid).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_lang"))
            .arg("build")
            .arg(&source)
            .arg("-o")
            .arg(&source)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert_eq!(fs::read_to_string(&source).unwrap(), valid);
    }
}
