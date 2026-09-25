pub mod baseline;
pub mod data;
pub mod graph;
pub mod math;
pub mod model_io;
pub mod predict;
pub mod train;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
