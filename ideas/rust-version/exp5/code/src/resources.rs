//! OS measurements of this process (not children or a logical VM estimate).
use serde::Serialize;
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Usage {
    pub peak_rss_bytes: u64,
    pub user_seconds: f64,
    pub system_seconds: f64,
}
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub fn usage() -> Option<Usage> {
    let mut raw = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes a libc::rusage into properly allocated storage;
    // the buffer is inspected only after a successful return.
    let value = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, raw.as_mut_ptr()) != 0 {
            return None;
        }
        raw.assume_init()
    };
    let multiplier = if cfg!(target_os = "macos") { 1 } else { 1024 };
    Some(Usage {
        peak_rss_bytes: u64::try_from(value.ru_maxrss)
            .ok()?
            .checked_mul(multiplier)?,
        user_seconds: value.ru_utime.tv_sec as f64 + value.ru_utime.tv_usec as f64 / 1e6,
        system_seconds: value.ru_stime.tv_sec as f64 + value.ru_stime.tv_usec as f64 / 1e6,
    })
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn usage() -> Option<Usage> {
    None
}
