pub trait Metric {
    fn distance(&self, a: &[u8], b: &[u8]) -> usize;
}

/// Calculates the Hamming distance between two binary vectors.
pub struct Hamming;

impl Metric for Hamming {
    fn distance(&self, a: &[u8], b: &[u8]) -> usize {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones() as usize)
            .sum()
    }
}

/// Calculates the Ochiai/cosine distance between two binary vectors.
pub struct Cosine;

impl Metric for Cosine {
    fn distance(&self, a: &[u8], b: &[u8]) -> usize {
        assert_eq!(a.len(), b.len());

        let mut a_and_b = 0;
        let mut a_ones = 0;
        let mut b_ones = 0;

        for (x, y) in a.iter().zip(b.iter()) {
            a_and_b += (x & y).count_ones();
            a_ones += x.count_ones();
            b_ones += y.count_ones();
        }

        if a_ones == 0 && b_ones == 0 {
            return 0; // Both vectors are all zeros, consider them identical
        }

        let cosine_similarity = a_and_b as f64 / (a_ones as f64 * b_ones as f64).sqrt();

        ((1.0 - cosine_similarity) * (a.len() * 8) as f64).round() as usize
    }
}

/// Calculates the Jaccard/Tanimoto distance between two binary vectors.
pub struct Jaccard;

impl Metric for Jaccard {
    fn distance(&self, a: &[u8], b: &[u8]) -> usize {
        assert_eq!(a.len(), b.len());

        let mut a_and_b = 0;
        let mut a_or_b = 0;

        for (x, y) in a.iter().zip(b.iter()) {
            a_and_b += (x & y).count_ones() as usize;
            a_or_b += (x | y).count_ones() as usize;
        }

        if a_or_b == 0 {
            return 0;
        }

        let jaccard_similarity = a_and_b as f64 / a_or_b as f64;

        ((1.0 - jaccard_similarity) * (a.len() * 8) as f64).round() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let metric = Hamming;

        let a = [0b00000000, 0b11111111];
        let b = [0b11111111, 0b00000000];
        assert_eq!(metric.distance(&a, &b), 16);

        let c = [0b01010101, 0b01010101];
        let d = [0b10101010, 0b10101010];
        assert_eq!(metric.distance(&c, &d), 16);

        let e = [0b00000000, 0b00000000];
        let f = [0b00000000, 0b00000000];
        assert_eq!(metric.distance(&e, &f), 0);

        let e = [0b01100000, 0b00011000];
        let f = [0b00110000, 0b00110000];
        assert_eq!(metric.distance(&e, &f), 4);
    }

    #[test]
    fn test_cosine_distance() {
        let metric = Cosine;

        let a = [0b00000000, 0b11111111];
        let b = [0b11111111, 0b00000000];
        assert_eq!(metric.distance(&a, &b), 16);

        let c = [0b01010101, 0b01010101];
        let d = [0b10101010, 0b10101010];
        assert_eq!(metric.distance(&c, &d), 16);

        let e = [0b00000000, 0b00000000];
        let f = [0b00000000, 0b00000000];
        assert_eq!(metric.distance(&e, &f), 0);

        let e = [0b01100000, 0b00011000];
        let f = [0b00110000, 0b00110000];
        assert_eq!(metric.distance(&e, &f), 8);
    }

    #[test]
    fn test_jaccard_distance() {
        let metric = Jaccard;

        let a = [0b00000000, 0b11111111];
        let b = [0b11111111, 0b00000000];
        assert_eq!(metric.distance(&a, &b), 16);

        let c = [0b01010101, 0b01010101];
        let d = [0b10101010, 0b10101010];
        assert_eq!(metric.distance(&c, &d), 16);

        let e = [0b00000000, 0b00000000];
        let f = [0b00000000, 0b00000000];
        assert_eq!(metric.distance(&e, &f), 0);

        let e = [0b01100000, 0b00011000];
        let f = [0b00110000, 0b00110000];
        assert_eq!(metric.distance(&e, &f), 11);
    }
}
