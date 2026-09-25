//! Direct C symbols consumed by Roc. Each hosted argument transfers one owned
//! reference to Rust; RocOwned releases it once on both success and error paths.
//! Returned strings/lists transfer their new reference to Roc. No mutable global
//! allocator context is needed: the generated default allocator is stateless.

use crate::effects;
use crate::roc_platform_abi::*;
use std::ffi::c_void;
use std::io::{self, Write};
use std::mem::ManuallyDrop;

type Bytes = RocListWith<u8, false>;
type OwnedStr = RocOwned<RocStr, RocStrRelease>;
type OwnedBytes = RocOwned<Bytes, RocListSpineRelease>;

fn host() -> RocHost {
    make_roc_host(std::ptr::null_mut())
}

fn error(context: &str, err: io::Error) -> RocStr {
    RocStr::from_str(&format!("{context}: {:?}: {err}", err.kind()), &host())
}

fn unit(result: io::Result<()>, context: &str) -> HostStdoutWriteResult {
    match result {
        Ok(()) => HostStdoutWriteResult {
            payload: HostStdoutWriteResultPayload { ok: [] },
            tag: HostStdoutWriteResultTag::Ok,
        },
        Err(err) => HostStdoutWriteResult {
            payload: HostStdoutWriteResultPayload {
                err: ManuallyDrop::new(error(context, err)),
            },
            tag: HostStdoutWriteResultTag::Err,
        },
    }
}

fn text(result: io::Result<String>, context: &str) -> HostStdinReadTextResult {
    match result {
        Ok(value) => HostStdinReadTextResult {
            payload: HostStdinReadTextResultPayload {
                ok: ManuallyDrop::new(RocStr::from_str(&value, &host())),
            },
            tag: HostStdinReadTextResultTag::Ok,
        },
        Err(err) => HostStdinReadTextResult {
            payload: HostStdinReadTextResultPayload {
                err: ManuallyDrop::new(error(context, err)),
            },
            tag: HostStdinReadTextResultTag::Err,
        },
    }
}

fn bytes(result: io::Result<Vec<u8>>, context: &str) -> HostStdinReadBytesResult {
    match result {
        Ok(value) => HostStdinReadBytesResult {
            // u8 elements own no references; copying them creates an independent list.
            payload: HostStdinReadBytesResultPayload {
                ok: ManuallyDrop::new(unsafe { Bytes::from_slice(&value, &host()) }),
            },
            tag: HostStdinReadBytesResultTag::Ok,
        },
        Err(err) => HostStdinReadBytesResult {
            payload: HostStdinReadBytesResultPayload {
                err: ManuallyDrop::new(error(context, err)),
            },
            tag: HostStdinReadBytesResultTag::Err,
        },
    }
}

macro_rules! stream_effects {
    ($stream:ident, $write:ident, $line:ident, $bytes:ident, $flush:ident) => {
        #[no_mangle]
        unsafe extern "C" fn $write(value: RocStr) -> HostStdoutWriteResult {
            let value = unsafe { OwnedStr::from_raw(value) };
            unit(
                effects::write(&mut io::$stream().lock(), value.as_slice(), false),
                stringify!($write),
            )
        }
        #[no_mangle]
        unsafe extern "C" fn $line(value: RocStr) -> HostStdoutWriteResult {
            let value = unsafe { OwnedStr::from_raw(value) };
            unit(
                effects::write(&mut io::$stream().lock(), value.as_slice(), true),
                stringify!($line),
            )
        }
        #[no_mangle]
        unsafe extern "C" fn $bytes(value: Bytes) -> HostStdoutWriteResult {
            let value = unsafe { OwnedBytes::from_raw(value) };
            unit(
                effects::write(&mut io::$stream().lock(), value.as_slice(), false),
                stringify!($bytes),
            )
        }
        #[no_mangle]
        extern "C" fn $flush() -> HostStdoutWriteResult {
            unit(io::$stream().lock().flush(), stringify!($flush))
        }
    };
}

stream_effects!(
    stdout,
    roc_stdout_write,
    roc_stdout_line,
    roc_stdout_write_bytes,
    roc_stdout_flush
);
stream_effects!(
    stderr,
    roc_stderr_write,
    roc_stderr_line,
    roc_stderr_write_bytes,
    roc_stderr_flush
);

#[no_mangle]
extern "C" fn roc_stdin_line() -> HostStdinLineResult {
    match effects::read_line(&mut io::stdin().lock()) {
        Ok((value, eof)) => HostStdinLineResult {
            payload: HostStdinLineResultPayload {
                ok: ManuallyDrop::new(HostStdinLineOk {
                    text: RocStr::from_str(&value, &host()),
                    eof,
                }),
            },
            tag: HostStdinLineResultTag::Ok,
        },
        Err(err) => HostStdinLineResult {
            payload: HostStdinLineResultPayload {
                err: ManuallyDrop::new(error("stdin.line", err)),
            },
            tag: HostStdinLineResultTag::Err,
        },
    }
}

#[no_mangle]
extern "C" fn roc_stdin_read_text() -> HostStdinReadTextResult {
    text(
        effects::read_text(&mut io::stdin().lock()),
        "stdin.read_text",
    )
}

#[no_mangle]
extern "C" fn roc_stdin_read_bytes() -> HostStdinReadBytesResult {
    bytes(
        effects::read_bytes(&mut io::stdin().lock()),
        "stdin.read_bytes",
    )
}

#[no_mangle]
unsafe extern "C" fn roc_file_read_text(path: RocStr) -> HostFileReadTextResult {
    let path = unsafe { OwnedStr::from_raw(path) };
    text(
        std::fs::read_to_string(path.as_str()),
        &format!("file.read_text({:?})", path.as_str()),
    )
}

#[no_mangle]
unsafe extern "C" fn roc_file_read_bytes(path: RocStr) -> HostFileReadBytesResult {
    let path = unsafe { OwnedStr::from_raw(path) };
    bytes(
        std::fs::read(path.as_str()),
        &format!("file.read_bytes({:?})", path.as_str()),
    )
}

macro_rules! file_write {
    ($name:ident, $value:ty, $owner:ty, $append:expr) => {
        #[no_mangle]
        unsafe extern "C" fn $name(path: RocStr, value: $value) -> HostStdoutWriteResult {
            let path = unsafe { OwnedStr::from_raw(path) };
            let value = unsafe { <$owner>::from_raw(value) };
            unit(
                effects::write_file(path.as_str(), value.as_slice(), $append),
                &format!("{}({:?})", stringify!($name), path.as_str()),
            )
        }
    };
}

file_write!(roc_file_write_text, RocStr, OwnedStr, false);
file_write!(roc_file_append_text, RocStr, OwnedStr, true);
file_write!(roc_file_write_bytes, Bytes, OwnedBytes, false);
file_write!(roc_file_append_bytes, Bytes, OwnedBytes, true);

#[no_mangle]
unsafe extern "C" fn roc_file_exists(path: RocStr) -> HostFileExistsResult {
    let path = unsafe { OwnedStr::from_raw(path) };
    match std::path::Path::new(path.as_str()).try_exists() {
        Ok(value) => HostFileExistsResult {
            payload: HostFileExistsResultPayload {
                ok: ManuallyDrop::new(value),
            },
            tag: HostFileExistsResultTag::Ok,
        },
        Err(err) => HostFileExistsResult {
            payload: HostFileExistsResultPayload {
                err: ManuallyDrop::new(error(&format!("file.exists({:?})", path.as_str()), err)),
            },
            tag: HostFileExistsResultTag::Err,
        },
    }
}

#[no_mangle]
unsafe extern "C" fn roc_file_remove(path: RocStr) -> HostFileRemoveResult {
    let path = unsafe { OwnedStr::from_raw(path) };
    unit(
        std::fs::remove_file(path.as_str()),
        &format!("file.remove({:?})", path.as_str()),
    )
}

#[no_mangle]
unsafe extern "C" fn roc_file_create_dir_all(path: RocStr) -> HostFileCreateDirAllResult {
    let path = unsafe { OwnedStr::from_raw(path) };
    unit(
        std::fs::create_dir_all(path.as_str()),
        &format!("file.create_dir_all({:?})", path.as_str()),
    )
}

#[no_mangle]
extern "C" fn roc_alloc(length: usize, alignment: usize) -> *mut c_void {
    DefaultAllocators::roc_alloc(std::ptr::null_mut(), length, alignment)
}

#[no_mangle]
extern "C" fn roc_dealloc(ptr: *mut c_void, alignment: usize) {
    DefaultAllocators::roc_dealloc(std::ptr::null_mut(), ptr, alignment);
}

#[no_mangle]
extern "C" fn roc_realloc(ptr: *mut c_void, length: usize, alignment: usize) -> *mut c_void {
    DefaultAllocators::roc_realloc(std::ptr::null_mut(), ptr, length, alignment)
}

#[no_mangle]
unsafe extern "C" fn roc_dbg(ptr: *const u8, len: usize) {
    let _ = io::stderr()
        .lock()
        .write_all(unsafe { std::slice::from_raw_parts(ptr, len) });
    let _ = io::stderr().lock().write_all(b"\n");
}

#[no_mangle]
unsafe extern "C" fn roc_expect_failed(ptr: *const u8, len: usize) {
    unsafe { roc_dbg(ptr, len) };
    std::process::exit(1);
}

#[no_mangle]
unsafe extern "C" fn roc_crashed(ptr: *const u8, len: usize) {
    unsafe { roc_dbg(ptr, len) };
    std::process::exit(1);
}

#[cfg(not(test))]
#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const i8) -> i32 {
    // We provide C main, so Rust's usual startup code does not ignore SIGPIPE.
    // Ignore it explicitly: writes to closed pipes should return IoErr(BrokenPipe).
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn signal(number: i32, handler: usize) -> usize;
        }
        // SIGPIPE=13 and SIG_IGN=1 on both supported macOS architectures.
        if unsafe { signal(13, 1) } == usize::MAX {
            let _ = writeln!(io::stderr(), "Could not initialize SIGPIPE handling");
            return 1;
        }
    }
    let context = host();
    let args: Vec<_> = std::env::args_os().collect();
    // Every element is initialized before ownership is transferred to Roc.
    let list = unsafe { RocList::<RocStr>::allocate(args.len(), &context) };
    for (index, arg) in args.iter().enumerate() {
        unsafe {
            list.elements
                .add(index)
                .write(RocStr::from_str(&arg.to_string_lossy(), &context))
        };
    }
    let exit = unsafe { roc_main(list) };
    // C main does not run Rust's runtime cleanup, so explicitly flush stdout.
    let flushed = io::stdout()
        .lock()
        .flush()
        .and_then(|()| io::stderr().lock().flush());
    if exit == 0 && flushed.is_err() {
        1
    } else {
        exit
    }
}
