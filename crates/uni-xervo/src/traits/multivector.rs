//! Multi-vector (late-interaction / ColBERT) embedding types and the
//! [`MultiVectorEmbeddingModel`] trait.
//!
//! Where [`EmbeddingModel`](crate::traits::EmbeddingModel) pools a sequence into
//! one dense vector, a multi-vector model emits *one vector per token*. Scoring
//! is then done with MaxSim (sum over query tokens of the max similarity to any
//! document token) — the late-interaction approach popularized by ColBERT and
//! the strongest known method for layout-rich document retrieval (ColPali /
//! ColQwen2).
//!
//! This trait is a separate, narrow capability so a provider opts into it only
//! when it can skip pooling and surface per-token vectors. The native
//! multi-vector index is deliberately out of scope — this is the *producer*
//! side, immediately usable via host-side MaxSim (see
//! [`max_sim`](crate::score::max_sim)).

use crate::error::Result;
use crate::traits::TokenUsage;
use async_trait::async_trait;
use std::any::Any;

/// Result of a multi-vector embedding call: per-token vectors for each input.
///
/// `vectors` is indexed input → token → dimension. It is *ragged*: padding and
/// (optionally) special tokens are stripped, so the token count varies per
/// input. Each per-token vector has [`MultiVectorEmbeddingModel::dimensions`]
/// elements. `usage` follows the [`EmbedResult`](crate::traits::EmbedResult)
/// convention — `Some` for remote providers, `None` for local ones.
#[derive(Debug, Clone)]
pub struct MultiVectorEmbedResult {
    /// Per input, per retained token, one dense vector.
    pub vectors: Vec<Vec<Vec<f32>>>,
    /// Token usage reported by the provider, if any.
    pub usage: Option<TokenUsage>,
}

/// A model that produces per-token (multi-vector / ColBERT) embeddings from text.
///
/// Use cases include late-interaction reranking with BGE-M3's ColBERT head and
/// `answerai-colbert-small`. Each input yields a variable-length list of
/// per-token vectors; score them against a query with
/// [`max_sim`](crate::score::max_sim).
#[async_trait]
pub trait MultiVectorEmbeddingModel: Send + Sync + Any {
    /// Embed a batch of text strings into per-token vectors.
    ///
    /// Returns one ragged list of per-token vectors per input, in input order.
    ///
    /// # Errors
    /// Returns an error if tokenization fails, the model session errors, or the
    /// upstream API rejects the request.
    async fn embed(&self, texts: Vec<&str>) -> Result<MultiVectorEmbedResult>;

    /// The dimensionality of each per-token vector produced by this model.
    fn dimensions(&self) -> u32;

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
    fn multi_vector_embedding_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn MultiVectorEmbeddingModel>) {}
    }
}
