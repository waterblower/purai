use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::{self, Command},
};

const HELP: &str = "lang 0.1.0 — native ARM64 macOS compiler\n\nUsage:\n  lang check SOURCE\n  lang build SOURCE [-o OUTPUT]\n  lang run SOURCE [-o OUTPUT]\n\nDefault output: build/<source stem>\nNo external assembler, linker, or signing command is used.\n";

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os().skip(1);
    let Some(command) = args.next() else {
        print!("{HELP}");
        return Ok(());
    };
    if command == "--help" || command == "-h" {
        print!("{HELP}");
        return Ok(());
    }
    if command == "--version" {
        println!("lang {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if command != "build" && command != "check" && command != "run" {
        return Err(format!(
            "unknown command `{}`\n{HELP}",
            command.to_string_lossy()
        ));
    }
    let input = PathBuf::from(args.next().ok_or("expected a source file")?);
    let mut output = None;
    while let Some(arg) = args.next() {
        if arg != "-o" || output.is_some() || command == "check" {
            return Err(format!("unexpected argument `{}`", arg.to_string_lossy()));
        }
        output = Some(PathBuf::from(
            args.next().ok_or("expected a path after -o")?,
        ));
    }
    let source = fs::read_to_string(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    if command == "check" {
        lang::check(&source).map_err(|e| e.render(&input.display().to_string(), &source))?;
        println!("Checked {}", input.display());
        return Ok(());
    }
    if command == "run" && !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("running generated programs requires ARM64 macOS".into());
    }
    let output =
        output.unwrap_or_else(|| Path::new("build").join(input.file_stem().unwrap_or_default()));
    if fs::canonicalize(&output).ok() == fs::canonicalize(&input).ok() {
        return Err("output must not overwrite the source file".into());
    }
    let image =
        lang::compile(&source).map_err(|e| e.render(&input.display().to_string(), &source))?;
    write_executable(&output, &image).map_err(|e| format!("{}: {e}", output.display()))?;
    eprintln!(
        "Built {} ({} bytes, arm64-macos)",
        output.display(),
        image.len()
    );
    if command == "run" {
        let executable = fs::canonicalize(&output).map_err(|e| e.to_string())?;
        let status = Command::new(executable)
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("program failed: {status}"));
        }
    }
    Ok(())
}

fn write_executable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    // Install a new inode atomically: rewriting an already executed signed
    // inode can interact with the kernel's cached code-signing page hashes.
    let temporary = parent.join(format!(
        ".lang-{}-{}.tmp",
        process::id(),
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| {
        file.write_all(bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o755))?;
        }
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
