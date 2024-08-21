# proxbin

proxbin is a Hierarchical Navigable Small World (HNSW) implementation for fast approximate nearest neighbor search on binary vectors.

## Usage

```rust
use proxbin::{HNSW, BinaryVector, HNSWParams};

fn main() {
    // Create a new HNSW index for 2 bytes binary vectors
    let mut hnsw = HNSW::<u8, 2>::new();

    // Insert some vectors to the index
    let v1 = [0b00000000, 0b11111111];
    let v2 = [0b11111111, 0b00000000];
    let v3 = [0b01010101, 0b01010101];

    index.insert(1, v1);
    index.insert(2, v2);
    index.insert(3, v3);

    // Search for the nearest neighbor
    let query = [0b000000001, 0b11111111];
    let results = index.search(&query, 2);
}
```

## Limitation

- Only supports binary vectors.
- Only fixed vector size.
