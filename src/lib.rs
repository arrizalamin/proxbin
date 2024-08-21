//! proxbin
//!
//! proxbin is a Hierarchical Navigable Small World (HNSW) implementation for
//! fast approximate nearest neighbor search on binary vectors.

mod hnsw;
pub mod metric;

pub use hnsw::params::HNSWParams;
pub use hnsw::{BinaryVector, HNSWData, HNSW};
