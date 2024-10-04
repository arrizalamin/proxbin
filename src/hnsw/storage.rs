use std::collections::hash_map::IntoIter;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use anyhow::Result;

use super::node::HNSWNode;

pub trait Storage<K: Eq + Hash, const N: usize>:
    for<'a> IndexMut<&'a K, Output = HNSWNode<K, N>>
    + IntoIterator<Item = (K, HNSWNode<K, N>)>
    + Default
    + Debug
{
    fn insert(&mut self, key: K, node: HNSWNode<K, N>) -> Result<()>;

    fn remove(&mut self, key: &K) -> Result<Option<HNSWNode<K, N>>>;

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a HNSWNode<K, N>)>
    where
        K: 'a;

    fn contains_key(&self, key: &K) -> bool;

    fn get_mut(&mut self, key: &K) -> Option<&mut HNSWNode<K, N>> {
        if self.contains_key(key) {
            Some(self.index_mut(key))
        } else {
            None
        }
    }

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn clear(&mut self) -> Result<()>;
}

#[derive(Default, Debug)]
pub struct InMemoryStorage<K: Eq + Hash, const N: usize> {
    nodes: HashMap<K, HNSWNode<K, N>>,
}

impl<K: Eq + Hash + Clone, const N: usize> IntoIterator for InMemoryStorage<K, N> {
    type Item = (K, HNSWNode<K, N>);
    type IntoIter = IntoIter<K, HNSWNode<K, N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

impl<K: Eq + Hash + Clone, const N: usize> Index<&K> for InMemoryStorage<K, N> {
    type Output = HNSWNode<K, N>;

    fn index(self: &Self, index: &K) -> &Self::Output {
        self.nodes.get(&index).unwrap()
    }
}

impl<K: Eq + Hash + Clone, const N: usize> IndexMut<&K> for InMemoryStorage<K, N> {
    fn index_mut(&mut self, index: &K) -> &mut Self::Output {
        self.nodes.get_mut(&index).unwrap()
    }
}

impl<K: Eq + Hash + Clone + Default + Debug, const N: usize> Storage<K, N>
    for InMemoryStorage<K, N>
{
    fn insert(&mut self, key: K, node: HNSWNode<K, N>) -> Result<()> {
        self.nodes.insert(key, node);
        Ok(())
    }

    fn remove(&mut self, key: &K) -> Result<Option<HNSWNode<K, N>>> {
        Ok(self.nodes.remove(key))
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a HNSWNode<K, N>)>
    where
        K: 'a,
    {
        self.nodes.iter()
    }

    fn contains_key(&self, key: &K) -> bool {
        self.nodes.contains_key(key)
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn clear(&mut self) -> Result<()> {
        self.nodes.clear();
        Ok(())
    }
}
