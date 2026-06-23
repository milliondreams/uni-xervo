//! Coherence checks for the unified `ModelInfo` surface, the `embedder()`
//! alias, and the `prelude`.
//!
//! Cheap, mock-backed: confirms `model_id()` (and `active_execution_providers`)
//! are reachable through the typed handles via the `ModelInfo` supertrait, that
//! `embedder()` aliases `embedding()`, and that `use uni_xervo::prelude::*`
//! brings the common names into scope.

use uni_xervo::prelude::*;

mod common;
use common::mock_support::{MockProvider, make_spec, runtime_with_embed};

#[tokio::test]
async fn model_info_reachable_through_embedding_handle_and_alias() {
    let runtime = runtime_with_embed().await.unwrap();

    // `model_id()` comes from the ModelInfo supertrait on EmbeddingModel.
    let handle = runtime.embedding("embed/test").await.unwrap();
    assert!(!handle.model_id().is_empty());
    assert!(handle.active_execution_providers().is_empty()); // mock: none

    // `embedder()` is an alias for `embedding()` — same model behind it.
    let aliased = runtime.embedder("embed/test").await.unwrap();
    assert_eq!(handle.model_id(), aliased.model_id());
}

#[tokio::test]
async fn model_info_reachable_through_reranker_and_generator() {
    // Reranker — previously had no model_id; now gains it via ModelInfo.
    let rerank_rt = ModelRuntime::builder()
        .register_provider(MockProvider::rerank_only())
        .catalog(vec![make_spec(
            "rerank/test",
            ModelTask::Rerank,
            "mock/rerank",
            "test-model",
        )])
        .build()
        .await
        .unwrap();
    let reranker = rerank_rt.reranker("rerank/test").await.unwrap();
    assert!(!reranker.model_id().is_empty());

    // Generator — likewise gains model_id via ModelInfo.
    let gen_rt = ModelRuntime::builder()
        .register_provider(MockProvider::generate_only())
        .catalog(vec![make_spec(
            "generate/test",
            ModelTask::Generate,
            "mock/generate",
            "test-model",
        )])
        .build()
        .await
        .unwrap();
    let generator = gen_rt.generator("generate/test").await.unwrap();
    assert!(!generator.model_id().is_empty());
}

/// Compile-only: every name a typical caller needs resolves through the prelude.
/// Never called; its purpose is to fail compilation if the prelude regresses.
#[allow(dead_code)]
fn prelude_brings_common_names_into_scope() {
    fn _assert_named<T>() {}
    _assert_named::<ModelRuntime>();
    _assert_named::<ModelAliasSpec>();
    _assert_named::<EmbedResult>();
    _assert_named::<SparseEmbedResult>();
    _assert_named::<MultiVectorEmbedResult>();
    _assert_named::<TranscribeResult>();
    // Score helpers resolve through the prelude.
    let _ = max_sim(&[], &[]);
    let _ = sparse_dot(&[], &[]);
    // Trait objects compile (ModelInfo + the task traits are in scope).
    fn _accept(_: &dyn EmbeddingModel, _: &dyn RerankerModel, _: &dyn ModelInfo) {}
}
