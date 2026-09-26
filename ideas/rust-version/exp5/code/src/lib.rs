pub mod data;
pub mod ir;
pub mod model;
pub mod proposal;
pub mod resources;
pub mod sha256;
pub mod train;
pub mod vm;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn fingerprint(bytes: &[u8]) -> String {
    sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn canonical_id(value: &impl serde::Serialize) -> String {
    fingerprint(&serde_json::to_vec(value).expect("serializable IR"))
}
