//! Main HNSW implementation.

mod node;
pub mod params;

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    hash::Hash,
    usize, vec,
};

use anyhow::{anyhow, Result};
use rand::Rng;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use self::{node::HNSWNode, params::HNSWParams};
pub use node::BinaryVector;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HNSWData<K: Eq + Hash, const N: usize> {
    nodes: HashMap<K, HNSWNode<K, N>>,
    entry_point: K,
}

/// Hierarchical Navigable Small World (HNSW) index for binary vectors.
pub struct HNSW<'a, K: Eq + Hash, const N: usize> {
    pub data: HNSWData<K, N>,
    params: HNSWParams<'a>,
    rng: rand::rngs::StdRng,
    pub deleted_count: usize,
    max_level: usize,
}

impl<'a, K, const N: usize> HNSW<'a, K, N>
where
    K: Clone + Ord + Eq + Hash + Default,
{
    /// Creates a new HNSW index with default parameters.
    pub fn new() -> Self {
        let params = HNSWParams::default();
        HNSW {
            data: HNSWData {
                nodes: HashMap::new(),
                entry_point: K::default(),
            },
            rng: rand::SeedableRng::seed_from_u64(params.seed),
            params,
            deleted_count: 0,
            max_level: 0,
        }
    }

    /// Creates a new HNSW index with custom parameters.
    pub fn new_with_params(params: HNSWParams<'a>) -> Self {
        HNSW {
            data: HNSWData {
                nodes: HashMap::new(),
                entry_point: K::default(),
            },
            rng: rand::SeedableRng::seed_from_u64(params.seed),
            params,
            deleted_count: 0,
            max_level: 0,
        }
    }

    /// Creates a new HNSW index with initial data.
    pub fn new_with_data(data: HNSWData<K, N>, params: HNSWParams<'a>) -> Self {
        HNSW {
            data,
            rng: rand::SeedableRng::seed_from_u64(params.seed),
            params,
            deleted_count: 0,
            max_level: 0,
        }
    }

    /// Returns an iterator over the nodes in the index.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &HNSWNode<K, N>)> {
        self.data.nodes.iter()
    }

    /// Returns the number of nodes in the index.
    pub fn size(&self) -> usize {
        self.data.nodes.len() - self.deleted_count
    }

    /// Cleans deleted nodes from the index and reindexes the remaining nodes.
    pub fn reindex(&mut self) {
        let data = self
            .data
            .nodes
            .drain()
            .filter(|(_, node)| !node.deleted)
            .map(|(key, node)| (key, node.vector))
            .collect::<HashMap<K, BinaryVector<N>>>();

        self.data.nodes = HashMap::new();
        self.data.entry_point = K::default();
        self.deleted_count = 0;
        self.max_level = 0;

        data.into_iter()
            .for_each(|(key, vector)| self.insert(key, vector));
    }

    /// Inserts a new vector to the index.
    pub fn insert(&mut self, key: K, vector: BinaryVector<N>) {
        let node_level = self.random_level();

        let new_node = HNSWNode {
            vector,
            connections: vec![Vec::new(); node_level + 1],
            deleted: false,
        };

        self.data.nodes.insert(key.clone(), new_node);

        if self.data.nodes.len() == 1 {
            self.data.entry_point = key.clone();
            self.max_level = node_level;
            return;
        }

        let mut entry_point = self.data.entry_point.clone();
        let mut entry_point_dist = self
            .params
            .metric
            .distance(&self.data.nodes[&entry_point].vector, &vector);

        for level in (0..=node_level).rev() {
            let (closest, closest_dist, candidates) =
                self.find_closest_and_candidates(&vector, &entry_point, level);

            if closest_dist < entry_point_dist {
                entry_point = closest;
                entry_point_dist = closest_dist;
            }

            let mut all_candidates = candidates;
            all_candidates.push(Reverse((entry_point.clone(), entry_point_dist)));

            self.connect_new_node(&key, all_candidates, level);
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.data.entry_point = key;
        }
    }

    fn find_closest_and_candidates(
        &self,
        vector: &BinaryVector<N>,
        entry_point: &K,
        level: usize,
    ) -> (K, usize, BinaryHeap<Reverse<(K, usize)>>) {
        let mut current_best = (
            entry_point.clone(),
            self.params
                .metric
                .distance(&self.data.nodes[entry_point].vector, vector),
        );
        let mut candidates = BinaryHeap::new();
        candidates.push(Reverse(current_best.clone()));
        let mut visited = HashSet::new();
        visited.insert(entry_point.clone());

        while let Some(Reverse((current_node, current_dist))) = candidates.pop() {
            if current_dist > current_best.1 {
                break;
            }

            if level < self.data.nodes[&current_node].connections.len() {
                for neighbor in &self.data.nodes[&current_node].connections[level] {
                    if visited.insert(neighbor.clone()) {
                        let dist = self
                            .params
                            .metric
                            .distance(&self.data.nodes[neighbor].vector, vector);
                        candidates.push(Reverse((neighbor.clone(), dist)));
                        if dist < current_best.1 {
                            current_best = (neighbor.clone(), dist);
                        }
                    }
                }
            }

            if candidates.len() > self.params.ef_construction {
                candidates = candidates
                    .into_iter()
                    .take(self.params.ef_construction)
                    .collect();
            }
        }

        (current_best.0, current_best.1, candidates)
    }

    fn connect_new_node(
        &mut self,
        new_key: &K,
        candidates: BinaryHeap<Reverse<(K, usize)>>,
        level: usize,
    ) {
        let new_connections: Vec<_> = candidates
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((node, _))| node)
            .take(self.params.max_connections(level))
            .collect();

        // Add connections to the new node
        self.data.nodes.get_mut(new_key).unwrap().connections[level] = new_connections.clone();

        for conn in &new_connections {
            self.update_reverse_connections(conn, new_key, level);
        }
    }

    fn update_reverse_connections(&mut self, conn: &K, new_key: &K, level: usize) {
        let conn_node = self.data.nodes.get_mut(conn).unwrap();
        if conn_node.connections.len() <= level {
            conn_node.connections.resize(level + 1, Vec::new());
        }
        if !conn_node.connections[level].contains(new_key) {
            if conn_node.connections[level].len() < self.params.max_connections(level) {
                conn_node.connections[level].push(new_key.clone());
            } else {
                self.prune_connections(conn, new_key, level);
            }
        }
    }

    fn prune_connections(&mut self, node_key: &K, new_key: &K, level: usize) {
        let node_vector = &self.data.nodes[node_key].vector;
        let mut connections = self.data.nodes[node_key].connections[level].clone();
        connections.push(new_key.clone());

        // Sort connections and keep the closest ones
        connections.sort_by_key(|conn_key| {
            self.params
                .metric
                .distance(&self.data.nodes[conn_key].vector, node_vector)
        });
        connections.truncate(self.params.max_connections(level));

        self.data.nodes.get_mut(node_key).unwrap().connections[level] = connections;
    }

    fn random_level(&mut self) -> usize {
        let mut level = 0;
        while self.rng.gen::<f64>() < self.params.level_multiplier {
            level += 1;
        }
        level
    }

    /// Removes a vector from the index by its ID and returns the removed vector.
    pub fn remove(&mut self, key: &K) -> Result<BinaryVector<N>> {
        if let Some(node) = self.data.nodes.get_mut(key) {
            if node.deleted {
                return Err(anyhow!("Node already deleted"));
            }
            node.deleted = true;
            self.deleted_count += 1;
            Ok(node.vector.clone())
        } else {
            Err(anyhow!("Invalid key: not found"))
        }
    }

    /// Searches for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &BinaryVector<N>, k: usize) -> Vec<(K, BinaryVector<N>, usize)> {
        if self.data.nodes.is_empty() {
            return Vec::new();
        }

        let (entry_point, entry_point_dist) = self.find_entry_point(query);

        self.search_level_zero(query, k, &entry_point, entry_point_dist)
    }

    fn find_entry_point(&self, query: &BinaryVector<N>) -> (K, usize) {
        let mut entry_point = self.data.entry_point.clone();
        let mut entry_point_dist = self
            .params
            .metric
            .distance(&self.data.nodes[&entry_point].vector, query);
        let mut current_level = self.max_level; // Use max_level instead of node's max_level

        while current_level > 0 {
            let new_entry = self.search_level(query, &entry_point, entry_point_dist, current_level);
            entry_point = new_entry.0;
            entry_point_dist = new_entry.1;
            current_level -= 1;
        }

        (entry_point, entry_point_dist)
    }

    fn search_level(
        &'a self,
        query: &BinaryVector<N>,
        mut entry_point: &'a K,
        mut entry_point_dist: usize,
        level: usize,
    ) -> (K, usize) {
        let mut changed = true;
        while changed {
            changed = false;
            for neighbor in &self.data.nodes[entry_point].connections[level] {
                let dist = self
                    .params
                    .metric
                    .distance(&self.data.nodes[neighbor].vector, query);
                if dist < entry_point_dist {
                    entry_point = neighbor;
                    entry_point_dist = dist;
                    changed = true;
                }
            }
        }
        (entry_point.clone(), entry_point_dist)
    }

    fn search_level_zero(
        &self,
        query: &BinaryVector<N>,
        k: usize,
        entry_point: &K,
        entry_point_dist: usize,
    ) -> Vec<(K, BinaryVector<N>, usize)> {
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();
        candidates.push(Reverse((entry_point_dist, entry_point.clone())));
        visited.insert(entry_point.clone());

        let mut results = Vec::new();

        while results.len() < k && !candidates.is_empty() {
            if let Some(Reverse((dist, node_key))) = candidates.pop() {
                let node = &self.data.nodes[&node_key];
                if !node.deleted {
                    results.push((node_key.clone(), node.vector, dist));
                }

                for neighbor in &node.connections[0] {
                    if visited.insert(neighbor.clone()) {
                        let neighbor_dist = self
                            .params
                            .metric
                            .distance(&self.data.nodes[neighbor].vector, query);
                        candidates.push(Reverse((neighbor_dist, neighbor.clone())));
                    }
                }
            }
        }

        results.sort_by_key(|&(_, _, dist)| dist);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_simple_add_and_search() {
        let mut hnsw = HNSW::<usize, 2>::new_with_params(
            HNSWParams::default()
                .with_max_connections(5)
                .with_ef_construction(10)
                .with_level_multiplier(0.5)
                .with_seed(42),
        );

        let v1 = [0b00000000, 0b11111111];
        let v2 = [0b11111111, 0b00000000];
        let v3 = [0b01010101, 0b01010101];

        hnsw.insert(0, v1);
        hnsw.insert(1, v2);
        hnsw.insert(2, v3);

        let query = [0b00000000, 0b11111111];
        let results = hnsw.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (0, v1, 0));
        assert_eq!(results[1], (2, v3, 8));
    }

    #[test]
    fn test_remove() {
        let mut hnsw = HNSW::new();
        let v1 = [0b00000000, 0b11111111];
        let v2 = [0b11111111, 0b00000000];
        let v3 = [0b01010101, 0b01010101];

        hnsw.insert(0, v1);
        hnsw.insert(1, v2);
        hnsw.insert(2, v3);

        let removed_vector = hnsw.remove(&1).unwrap();
        assert_eq!(removed_vector, v2);

        // Test that the node was removed
        let results = hnsw.search(&v2, 10);
        assert!(!results.contains(&(1, v2, 0)));

        // Test that removing a node that doesn't exist returns Error
        assert!(hnsw.remove(&100).is_err());
    }

    #[test]
    fn test_remove_empty() {
        let mut hnsw: HNSW<usize, 2> = HNSW::new();
        assert!(hnsw.remove(&0).is_err());
    }

    #[test]
    fn test_remove_single_node() {
        let mut hnsw: HNSW<usize, 2> = HNSW::new();
        let vector = [0b00000000, 0b11111111];
        hnsw.insert(0, vector.clone());
        let removed_vector = hnsw.remove(&0).unwrap();
        assert_eq!(removed_vector, vector);
        assert_eq!(hnsw.size(), 0);
    }

    #[test]
    fn test_reindex() {
        let mut hnsw: HNSW<usize, 2> = HNSW::new();
        let v1 = [0b00000000, 0b11111111];
        let v2 = [0b11111111, 0b00000000];
        let v3 = [0b01010101, 0b01010101];
        let v4 = [0b10101010, 0b10101010];

        hnsw.insert(0, v1);
        hnsw.insert(1, v2);
        hnsw.insert(2, v3);
        hnsw.insert(3, v4);

        assert!(hnsw.remove(&1).is_ok());
        assert!(hnsw.remove(&3).is_ok());
        assert_eq!(hnsw.size(), 2);
        assert_eq!(hnsw.data.nodes.len(), 4);

        hnsw.reindex();

        assert_eq!(hnsw.data.nodes.len(), 2);
        assert!(!hnsw.data.nodes.contains_key(&1));
        assert!(!hnsw.data.nodes.contains_key(&3));
        assert!(hnsw.data.nodes.contains_key(&0));
        assert!(hnsw.data.nodes.contains_key(&2));

        // Check that the entry point is still valid
        assert!(hnsw.data.nodes.contains_key(&hnsw.data.entry_point));
    }

    #[test]
    fn test_complex_add_and_search() {
        let mut index = HNSW::<usize, 4>::new_with_params(
            HNSWParams::default()
                .with_max_connections(10)
                .with_ef_construction(20)
                .with_level_multiplier(0.4)
                .with_seed(42),
        );
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(42);

        // Add 100 random vectors
        for i in 0..100 {
            let vector: BinaryVector<4> = rng.gen();
            index.insert(i, vector);
        }

        let known_vector = [0b01010101, 0b10101010, 0b00001111, 0b11110000];
        index.insert(100, known_vector);

        let results = index.search(&known_vector, 5);

        assert_eq!(results[0], (100, known_vector, 0));

        for i in 1..results.len() {
            assert!(results[i].2 >= results[i - 1].2);
        }

        let random_query: BinaryVector<4> = rng.gen();
        let random_results = index.search(&random_query, 10);

        for i in 1..random_results.len() {
            assert!(random_results[i].2 >= random_results[i - 1].2);
        }
    }
}
