use std::sync::atomic::{AtomicU32, Ordering};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use proxbin::{BinaryVector, HNSW};
use rand::Rng;

fn bench_insert(c: &mut Criterion) {
    let mut index = HNSW::<u32, 32>::new();
    let mut rng = rand::thread_rng();

    static COUNTER: AtomicU32 = AtomicU32::new(0);
    c.bench_function("insert vector", |b| {
        b.iter(|| {
            let vector: BinaryVector<32> = rng.gen();
            let key = COUNTER.fetch_add(1, Ordering::Relaxed);
            index.insert(black_box(key), black_box(vector));
        })
    });
}

fn bench_search(c: &mut Criterion) {
    let mut index = HNSW::<u32, 32>::new();
    let mut rng = rand::thread_rng();

    for i in 0..10_000 {
        let vector: BinaryVector<32> = rng.gen();
        index.insert(i, vector);
    }

    c.bench_function("search", |b| {
        b.iter(|| {
            let query: BinaryVector<32> = rng.gen();
            index.search(black_box(&query), black_box(10));
        })
    });
}

criterion_group!(benches, bench_insert, bench_search);
criterion_main!(benches);
