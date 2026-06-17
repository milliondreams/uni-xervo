//! Resolver and cache tests for the multimodal trait surface.
//!
//! Mirrors the parity tests in `handle_cache_test.rs` for the seven new
//! resolver methods on [`ModelRuntime`]: per-alias caching via `Arc::ptr_eq`,
//! `AliasNotFound` on missing aliases, and `ProviderCapabilityMissing` on
//! handle/trait downcast mismatch.

use uni_xervo::api::ModelTask;
use uni_xervo::error::RuntimeError;
mod common;
use common::mock_support::{
    MockProvider, make_spec, runtime_with_audio_embedder, runtime_with_document_extractor,
    runtime_with_embed, runtime_with_image_embedder, runtime_with_multimodal_embedder,
    runtime_with_nlp_model, runtime_with_ocr_model, runtime_with_transcriber,
};
use uni_xervo::runtime::ModelRuntime;

// `unwrap_err` requires the Ok variant to be Debug — Arc<dyn TraitName> is
// not, so we extract errors via pattern matching throughout this file.

// ── Happy path: repeated calls return the same Arc ──────────────────────

#[tokio::test]
async fn image_embedder_cache_returns_same_arc() {
    let runtime = runtime_with_image_embedder().await.unwrap();
    let h1 = runtime.image_embedder("embed_image/test").await.unwrap();
    let h2 = runtime.image_embedder("embed_image/test").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn audio_embedder_cache_returns_same_arc() {
    let runtime = runtime_with_audio_embedder().await.unwrap();
    let h1 = runtime.audio_embedder("embed_audio/test").await.unwrap();
    let h2 = runtime.audio_embedder("embed_audio/test").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn multimodal_embedder_cache_returns_same_arc() {
    let runtime = runtime_with_multimodal_embedder().await.unwrap();
    let h1 = runtime
        .multimodal_embedder("embed_multimodal/test")
        .await
        .unwrap();
    let h2 = runtime
        .multimodal_embedder("embed_multimodal/test")
        .await
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn nlp_model_cache_returns_same_arc() {
    let runtime = runtime_with_nlp_model().await.unwrap();
    let h1 = runtime.nlp_model("nlp/test").await.unwrap();
    let h2 = runtime.nlp_model("nlp/test").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn document_extractor_cache_returns_same_arc() {
    let runtime = runtime_with_document_extractor().await.unwrap();
    let h1 = runtime
        .document_extractor("document_extract/test")
        .await
        .unwrap();
    let h2 = runtime
        .document_extractor("document_extract/test")
        .await
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn transcriber_cache_returns_same_arc() {
    let runtime = runtime_with_transcriber().await.unwrap();
    let h1 = runtime.transcriber("transcribe/test").await.unwrap();
    let h2 = runtime.transcriber("transcribe/test").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

#[tokio::test]
async fn ocr_model_cache_returns_same_arc() {
    let runtime = runtime_with_ocr_model().await.unwrap();
    let h1 = runtime.ocr_model("ocr/test").await.unwrap();
    let h2 = runtime.ocr_model("ocr/test").await.unwrap();
    assert!(std::sync::Arc::ptr_eq(&h1, &h2));
}

// ── AliasNotFound paths ─────────────────────────────────────────────────

#[tokio::test]
async fn image_embedder_alias_not_found() {
    let runtime = runtime_with_image_embedder().await.unwrap();
    let res = runtime.image_embedder("missing/alias").await;
    assert!(matches!(res, Err(RuntimeError::AliasNotFound { .. })));
}

#[tokio::test]
async fn nlp_model_alias_not_found() {
    let runtime = runtime_with_nlp_model().await.unwrap();
    let res = runtime.nlp_model("missing/alias").await;
    assert!(matches!(res, Err(RuntimeError::AliasNotFound { .. })));
}

#[tokio::test]
async fn transcriber_alias_not_found() {
    let runtime = runtime_with_transcriber().await.unwrap();
    let res = runtime.transcriber("missing/alias").await;
    assert!(matches!(res, Err(RuntimeError::AliasNotFound { .. })));
}

// ── Capability mismatch paths ───────────────────────────────────────────
// Build a runtime where an alias declares one task but the resolver asks
// for a different trait; downcast must fail with ProviderCapabilityMissing.

#[tokio::test]
async fn image_embedder_capability_mismatch_when_alias_is_text_embed() {
    let runtime = runtime_with_embed().await.unwrap();
    let res = runtime.image_embedder("embed/test").await;
    assert!(matches!(
        res,
        Err(RuntimeError::ProviderCapabilityMissing { ref capability, .. })
            if capability == "ImageEmbeddingModel"
    ));
}

#[tokio::test]
async fn nlp_model_capability_mismatch_when_alias_is_text_embed() {
    let runtime = runtime_with_embed().await.unwrap();
    let res = runtime.nlp_model("embed/test").await;
    assert!(matches!(
        res,
        Err(RuntimeError::ProviderCapabilityMissing { ref capability, .. })
            if capability == "NlpModel"
    ));
}

#[tokio::test]
async fn ocr_model_capability_mismatch_when_alias_is_text_embed() {
    let runtime = runtime_with_embed().await.unwrap();
    let res = runtime.ocr_model("embed/test").await;
    assert!(matches!(
        res,
        Err(RuntimeError::ProviderCapabilityMissing { ref capability, .. })
            if capability == "OcrModel"
    ));
}

// ── Each resolver targets the correct slot ──────────────────────────────
// A runtime hosting multiple multimodal aliases must route each to its
// own cache slot; cross-resolver lookups must miss cleanly.

#[tokio::test]
async fn multi_alias_runtime_routes_to_correct_resolver() {
    let nlp_spec = make_spec("nlp/test", ModelTask::Nlp, "mock/nlp", "test-model");
    let ocr_spec = make_spec("ocr/test", ModelTask::Ocr, "mock/ocr", "test-model");
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::nlp_only())
        .register_provider(MockProvider::ocr_only())
        .catalog(vec![nlp_spec, ocr_spec])
        .build()
        .await
        .unwrap();

    runtime.nlp_model("nlp/test").await.unwrap();
    runtime.ocr_model("ocr/test").await.unwrap();

    // Asking nlp_model for the ocr alias must fail with capability mismatch.
    let res = runtime.nlp_model("ocr/test").await;
    assert!(matches!(
        res,
        Err(RuntimeError::ProviderCapabilityMissing { ref capability, .. })
            if capability == "NlpModel"
    ));
}
