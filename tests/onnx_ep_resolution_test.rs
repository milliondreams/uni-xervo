#![cfg(any(feature = "provider-onnx", feature = "provider-onnx-dynamic"))]
//! Unit tests for the runtime EP-resolution layer.
//!
//! These tests don't require a real ONNX model, an ORT runtime DLL, or any
//! network access — they only exercise the parsing and feature-aware default
//! logic in `crate::provider::onnx_ep`. The deeper integration tests that
//! load real models live in `local_onnx_*_hf_e2e_test.rs`.

// onnx_ep is `pub(crate)`, so we test through the surface of the providers
// that consume it: catalog spec parsing → load (which fails fast on missing
// runtime, but the EP list is constructed *before* the load attempt).

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxRerankerProvider;
use uni_xervo::traits::ModelProvider;

fn rerank_spec_with_eps(eps: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "rerank/test".to_string(),
        task: ModelTask::Rerank,
        provider_id: "local/onnx-reranker".to_string(),
        model_id: "cross-encoder/ms-marco-MiniLM-L6-v2".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "execution_providers": eps }),
    }
}

/// Loading with a CUDA-only EP list when `gpu-cuda` isn't enabled at compile
/// time should surface a clear `Config` error before any model download.
#[cfg(not(feature = "gpu-cuda"))]
#[tokio::test]
async fn cuda_only_eps_fail_when_cuda_feature_disabled() {
    let provider = LocalOnnxRerankerProvider;
    let spec = rerank_spec_with_eps(serde_json::json!(["cuda"]));

    let err = provider
        .load(&spec)
        .await
        .expect_err("loading with cuda-only EPs must fail when gpu-cuda is off");

    // Should be a Config error from `feature_not_enabled`, not an HF download
    // failure. This proves the EP-list validation runs before any I/O.
    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("CUDA") && msg.contains("gpu-cuda"),
        "error should explain CUDA needs gpu-cuda, got: {msg}"
    );
}

/// CoreML-only EPs without `gpu-coreml` should fail with the same Config
/// error pattern. Verifies the guard is feature-uniform across EPs.
#[cfg(not(feature = "gpu-coreml"))]
#[tokio::test]
async fn coreml_only_eps_fail_when_coreml_feature_disabled() {
    let provider = LocalOnnxRerankerProvider;
    let spec = rerank_spec_with_eps(serde_json::json!(["coreml"]));

    let err = provider
        .load(&spec)
        .await
        .expect_err("loading with coreml-only EPs must fail when gpu-coreml is off");

    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );
}

/// String form of `execution_providers` (single EP name as a JSON string,
/// not array) should parse — we documented this in `parse_execution_providers_option`.
#[cfg(not(feature = "gpu-cuda"))]
#[tokio::test]
async fn execution_providers_accepts_string_form() {
    let provider = LocalOnnxRerankerProvider;
    let spec = rerank_spec_with_eps(serde_json::json!("cuda"));

    // Same as the array case: should fail with Config error before any I/O,
    // proving the string was parsed into the same internal representation.
    let err = provider
        .load(&spec)
        .await
        .expect_err("string-form cuda EP should also fail without gpu-cuda");
    assert!(
        matches!(err, RuntimeError::Config(_)),
        "expected RuntimeError::Config, got {err:?}"
    );
}
