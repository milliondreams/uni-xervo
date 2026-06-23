//! Host-side similarity scoring for sparse and multi-vector embeddings.
//!
//! uni-xervo emits sparse and multi-vector embeddings but deliberately does not
//! own an index — native multi-vector / inverted-index storage is a downstream
//! concern. These pure, dependency-free helpers close that gap for the common
//! case of reranking a small top-k candidate set in the host process:
//!
//! - [`max_sim`] / [`colbert_rerank`] score per-token (ColBERT) vectors via late
//!   interaction (sum of per-query-token maxima).
//! - [`sparse_dot`] scores two learned-sparse vectors.
//!
//! They make these embeddings useful immediately, without waiting on a native
//! index. For large corpora, prefer a real index over scoring every candidate
//! here.

use std::collections::HashMap;

/// Dot product of two equal-length vectors, truncating to the shorter length.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the MaxSim late-interaction score between query and document tokens.
///
/// Late interaction (ColBERT) scores a query against a document as the sum, over
/// each query token, of that token's maximum similarity to any document token.
/// Similarity is the dot product, so pre-normalized (unit-norm) per-token
/// vectors — uni-xervo's default — make it cosine similarity.
///
/// All vectors are expected to share the same dimensionality; mismatched
/// vectors are scored over their shared prefix. Returns `0.0` when either side
/// has no tokens.
///
/// # Examples
/// ```
/// use uni_xervo::score::max_sim;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
/// // token 0 best-matches doc[0] (1.0); token 1 best-matches doc[1] (0.5).
/// assert!((max_sim(&query, &doc) - 1.5).abs() < 1e-6);
/// ```
pub fn max_sim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Rank `docs` against `query` by [`max_sim`], returning one score per document.
///
/// Scores are returned in `docs` order — the caller sorts and truncates to its
/// top-k. Each document is its own list of per-token vectors.
///
/// # Examples
/// ```
/// use uni_xervo::score::colbert_rerank;
///
/// let query = vec![vec![1.0, 0.0]];
/// let doc_a = vec![vec![1.0, 0.0]];
/// let doc_b = vec![vec![0.0, 1.0]];
/// let scores = colbert_rerank(&query, &[&doc_a, &doc_b]);
/// assert!(scores[0] > scores[1]);
/// ```
pub fn colbert_rerank(query: &[Vec<f32>], docs: &[&[Vec<f32>]]) -> Vec<f32> {
    docs.iter().map(|doc| max_sim(query, doc)).collect()
}

/// Compute the dot product of two learned-sparse vectors.
///
/// Each input is a list of `(term_id, weight)` pairs. The result sums
/// `weight_a * weight_b` over terms present in both. Inputs need not be sorted;
/// duplicate term ids within a vector are summed. The two vectors must come from
/// the same model — term ids are not comparable across models.
///
/// # Examples
/// ```
/// use uni_xervo::score::sparse_dot;
///
/// let a = vec![(1u32, 0.5f32), (4, 2.0)];
/// let b = vec![(4u32, 1.0f32), (9, 3.0)];
/// // only term 4 overlaps: 2.0 * 1.0 = 2.0
/// assert!((sparse_dot(&a, &b) - 2.0).abs() < 1e-6);
/// ```
pub fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    // Index the smaller side to keep the map compact.
    let (index_src, scan_src) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    let mut index: HashMap<u32, f32> = HashMap::with_capacity(index_src.len());
    for &(term, weight) in index_src {
        *index.entry(term).or_insert(0.0) += weight;
    }

    scan_src
        .iter()
        .filter_map(|&(term, weight)| index.get(&term).map(|w| w * weight))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_sim_sums_per_query_token_maxima() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        assert!((max_sim(&query, &doc) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn max_sim_empty_sides_score_zero() {
        let q = vec![vec![1.0, 0.0]];
        assert_eq!(max_sim(&[], &q), 0.0);
        assert_eq!(max_sim(&q, &[]), 0.0);
    }

    #[test]
    fn colbert_rerank_orders_relevant_doc_higher() {
        let query = vec![vec![1.0, 0.0]];
        let relevant = vec![vec![1.0, 0.0]];
        let irrelevant = vec![vec![0.0, 1.0]];
        let scores = colbert_rerank(&query, &[&relevant, &irrelevant]);
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn sparse_dot_sums_overlapping_terms() {
        let a = vec![(1u32, 0.5f32), (4, 2.0)];
        let b = vec![(4u32, 1.0f32), (9, 3.0)];
        assert!((sparse_dot(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_dot_disjoint_is_zero() {
        let a = vec![(1u32, 1.0f32)];
        let b = vec![(2u32, 1.0f32)];
        assert_eq!(sparse_dot(&a, &b), 0.0);
    }

    #[test]
    fn sparse_dot_is_symmetric() {
        let a = vec![(1u32, 0.5f32), (4, 2.0), (7, 1.5)];
        let b = vec![(4u32, 1.0f32), (7, 2.0), (9, 3.0)];
        assert!((sparse_dot(&a, &b) - sparse_dot(&b, &a)).abs() < 1e-6);
    }

    #[test]
    fn sparse_dot_sums_duplicate_terms() {
        let a = vec![(4u32, 1.0f32), (4, 1.0)]; // duplicate term id -> weight 2.0
        let b = vec![(4u32, 3.0f32)];
        assert!((sparse_dot(&a, &b) - 6.0).abs() < 1e-6);
    }
}
