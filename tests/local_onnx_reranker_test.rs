#![cfg(any(feature = "provider-onnx", feature = "provider-onnx-dynamic"))]
//! Provider-level tests for `LocalOnnxRerankerProvider`.
//!
//! These tests cover the provider shape (id, capabilities, task gating)
//! without needing a real ONNX model on disk. End-to-end tests that
//! actually load a model from HuggingFace live in
//! `local_onnx_reranker_hf_e2e_test.rs` and are gated by `EXPENSIVE_TESTS=1`.

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxRerankerProvider;
use uni_xervo::traits::ModelProvider;

fn spec(alias: &str, task: ModelTask, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: "local/onnx-reranker".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }
}

#[test]
fn provider_id_is_stable() {
    let provider = LocalOnnxRerankerProvider;
    assert_eq!(provider.provider_id(), "local/onnx-reranker");
}

#[test]
fn capabilities_advertise_only_rerank() {
    let provider = LocalOnnxRerankerProvider;
    let caps = provider.capabilities();
    assert_eq!(caps.supported_tasks, vec![ModelTask::Rerank]);
}

#[tokio::test]
async fn load_rejects_non_rerank_task() {
    let provider = LocalOnnxRerankerProvider;
    let bad = spec(
        "oops/embed",
        ModelTask::Embed,
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    );

    let err = provider
        .load(&bad)
        .await
        .expect_err("loading an Embed-task spec into the reranker provider must fail");
    assert!(
        matches!(err, RuntimeError::CapabilityMismatch(_)),
        "expected CapabilityMismatch, got {err:?}"
    );
}

#[tokio::test]
async fn health_is_healthy() {
    use uni_xervo::traits::ProviderHealth;
    let provider = LocalOnnxRerankerProvider;
    let health = provider.health().await;
    assert!(
        matches!(health, ProviderHealth::Healthy),
        "expected Healthy, got {health:?}"
    );
}
