//! HNSWNode structure.

use std::hash::Hash;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A binary vector of length N.
pub type BinaryVector<const N: usize> = [u8; N];

/// A node in the HNSW graph.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct HNSWNode<K: Eq + Hash, const N: usize> {
    #[cfg_attr(feature = "serde", serde(with = "serde_bytes"))]
    pub vector: BinaryVector<N>,
    pub connections: Vec<Vec<K>>,
    pub deleted: bool,
}
