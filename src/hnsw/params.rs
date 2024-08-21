//! Contains the HNSWParams structure for configuring the HNSW index.

use crate::metric::{Hamming, Metric};

/// Parameters for configuring the HNSW index.
pub struct HNSWParams<'a> {
    pub max_connections: usize,
    pub level_0_max_connections_multiplier: f32,
    pub ef_construction: usize,
    pub level_multiplier: f64,
    pub seed: u64,
    pub metric: &'a dyn Metric,
}

impl<'a> HNSWParams<'a> {
    pub fn with_max_connections(self, max_connections: usize) -> Self {
        HNSWParams {
            max_connections,
            ..self
        }
    }

    pub fn with_level_0_max_connections_multiplier(
        self,
        level_0_max_connections_multiplier: f32,
    ) -> Self {
        HNSWParams {
            level_0_max_connections_multiplier,
            ..self
        }
    }

    pub fn max_connections(&self, level: usize) -> usize {
        if level == 0 {
            (self.max_connections as f32 * self.level_0_max_connections_multiplier) as usize
        } else {
            self.max_connections
        }
    }

    pub fn with_ef_construction(self, ef_construction: usize) -> Self {
        HNSWParams {
            ef_construction,
            ..self
        }
    }

    pub fn with_level_multiplier(self, level_multiplier: f64) -> Self {
        HNSWParams {
            level_multiplier,
            ..self
        }
    }

    pub fn with_seed(self, seed: u64) -> Self {
        HNSWParams { seed, ..self }
    }

    pub fn with_metric(self, metric: &'a dyn Metric) -> Self {
        HNSWParams { metric, ..self }
    }
}

impl<'a> Default for HNSWParams<'a> {
    fn default() -> Self {
        HNSWParams {
            max_connections: 16,
            level_0_max_connections_multiplier: 2.0,
            ef_construction: 200,
            level_multiplier: 1.0 / std::f64::consts::LOG2_E,
            seed: 42,
            metric: &Hamming,
        }
    }
}
