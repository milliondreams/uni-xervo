//! Learned-sparse text embedding types and the [`SparseEmbeddingModel`] trait.
//!
//! Sparse embeddings represent text as a small set of weighted vocabulary terms
//! rather than a single dense vector. They power hybrid first-stage retrieval:
//! a learned-sparse model (SPLADE family, the BGE-M3 sparse head) keeps the
//! inverted-index compatibility of BM25 while learning term weights — and term
//! expansion, for SPLADE — that beat BM25 in and out of domain.
//!
//! This trait sits alongside [`EmbeddingModel`](crate::traits::EmbeddingModel):
//! it is a separate, narrow capability so a provider opts into it only when it
//! can produce term-weight vectors. The downstream inverted index and impact
//! scoring are deliberately out of scope — this is the *producer* side only.

use crate::error::Result;
use crate::traits::TokenUsage;
use async_trait::async_trait;
use std::any::Any;

/// A single learned-sparse vector as `(term_id, weight)` pairs.
///
/// Each `term_id` indexes into the model's term space (see
/// [`SparseEmbeddingModel::vocab_size`]); `weight` is the non-negative learned
/// importance of that term. Entries are sparse: only non-zero terms appear.
/// Term ids are only comparable within the *same* model — two models with
/// different vocabularies produce incompatible sparse vectors.
pub type SparseVector = Vec<(u32, f32)>;

/// Result of a sparse-embedding call, carrying one [`SparseVector`] per input.
///
/// Mirrors [`EmbedResult`](crate::traits::EmbedResult): `usage` is `Some` when a
/// remote provider reports token counts and `None` for local providers where the
/// concept does not apply.
#[derive(Debug, Clone)]
pub struct SparseEmbedResult {
    /// One sparse vector per input, in input order.
    pub vectors: Vec<SparseVector>,
    /// Token usage reported by the provider, if any.
    pub usage: Option<TokenUsage>,
}

/// A model that produces learned-sparse term-weight vectors from text.
///
/// Use cases include SPLADE-style retrieval and the BGE-M3 sparse head for
/// hybrid dense + sparse search. One [`SparseVector`] is returned per input
/// text. For scoring two sparse vectors, see
/// [`sparse_dot`](crate::score::sparse_dot).
#[async_trait]
pub trait SparseEmbeddingModel: Send + Sync + Any {
    /// Embed a batch of text strings into learned-sparse vectors.
    ///
    /// Returns one [`SparseVector`] per input, in input order.
    ///
    /// # Errors
    /// Returns an error if tokenization fails, the model session errors, or the
    /// upstream API rejects the request.
    async fn embed(&self, texts: Vec<&str>) -> Result<SparseEmbedResult>;

    /// The size of the term space the produced `term_id`s index into.
    ///
    /// For SPLADE this is the model's full vocabulary (term expansion); for the
    /// BGE-M3 lexical head it is the tokenizer vocabulary the input tokens map to.
    fn vocab_size(&self) -> u32;

    /// The underlying model identifier (HuggingFace repo ID or remote API model name).
    fn model_id(&self) -> &str;

    /// Optional warmup hook (e.g. load weights into memory on first access).
    /// The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Dyn-safety smoke check: fails to compile if the trait becomes non-object-safe.
    #[test]
    fn sparse_embedding_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn SparseEmbeddingModel>) {}
    }
}
