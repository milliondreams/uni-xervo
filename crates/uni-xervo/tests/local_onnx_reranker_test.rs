#![cfg(any(feature = "provider-onnx", feature = "provider-onnx-dynamic"))]
//! Provider-level tests for the rerank task on `LocalOnnxProvider`.
//!
//! These tests cover the provider shape (id, capabilities, task gating)
//! without needing a real ONNX model on disk. End-to-end tests that
//! actually load a model from HuggingFace live in
//! `local_onnx_reranker_hf_e2e_test.rs` and are gated by `EXPENSIVE_TESTS=1`.

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::traits::ModelProvider;

fn spec(alias: &str, task: ModelTask, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: "local/onnx".to_string(),
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
    let provider = LocalOnnxProvider::new();
    assert_eq!(provider.provider_id(), "local/onnx");
}

#[test]
fn capabilities_advertise_raw_and_rerank() {
    let provider = LocalOnnxProvider::new();
    let caps = provider.capabilities();
    assert!(
        caps.supported_tasks.contains(&ModelTask::Rerank),
        "expected Rerank in capabilities, got {:?}",
        caps.supported_tasks
    );
    assert!(
        caps.supported_tasks.contains(&ModelTask::Raw),
        "expected Raw in capabilities, got {:?}",
        caps.supported_tasks
    );
}

#[tokio::test]
async fn load_rejects_unsupported_task() {
    let provider = LocalOnnxProvider::new();
    let bad = spec(
        "oops/generate",
        ModelTask::Generate,
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    );

    let err = provider
        .load(&bad)
        .await
        .expect_err("loading a Generate-task spec into the ONNX provider must fail");
    assert!(
        matches!(err, RuntimeError::CapabilityMismatch(_)),
        "expected CapabilityMismatch, got {err:?}"
    );
}

#[tokio::test]
async fn health_is_healthy() {
    use uni_xervo::traits::ProviderHealth;
    let provider = LocalOnnxProvider::new();
    let health = provider.health().await;
    assert!(
        matches!(health, ProviderHealth::Healthy),
        "expected Healthy, got {health:?}"
    );
}
