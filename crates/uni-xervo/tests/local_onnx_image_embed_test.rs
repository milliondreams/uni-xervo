//! Fixture-level tests for `local/onnx` × `EmbedImage`.
//!
//! Validates catalog-registration (options validation, capability advertising,
//! resolver wiring) without downloading SigLIP-2 weights. Live model loads
//! would land in `onnx_models_expensive_test.rs` once a stable HF repo with
//! ONNX-exported SigLIP-2 is picked.

#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn image_embed_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::EmbedImage,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

#[tokio::test]
async fn local_onnx_advertises_embed_image_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let tasks = LocalOnnxProvider::new().capabilities().supported_tasks;
    assert!(tasks.contains(&ModelTask::EmbedImage));
}

#[tokio::test]
async fn image_embed_alias_registers_with_full_options() {
    let opts = serde_json::json!({
        "onnx_path": "onnx/model.onnx",
        "image_size": 384,
        "dimensions": 1152,
        "normalization": "siglip",
        "pool": "none",
        "normalize": true
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![image_embed_spec("embed/siglip", "some/repo", opts)])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "valid options should pass validation: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn image_embed_alias_rejects_unknown_normalization() {
    let opts = serde_json::json!({
        "onnx_path": "onnx/model.onnx",
        "image_size": 384,
        "dimensions": 1152,
        "normalization": "bogus"
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![image_embed_spec("embed/bad", "some/repo", opts)])
        .build()
        .await;
    assert!(result.is_err(), "unknown normalization must be rejected");
}

#[tokio::test]
async fn image_embed_alias_rejects_unknown_pool() {
    let opts = serde_json::json!({
        "onnx_path": "onnx/model.onnx",
        "image_size": 384,
        "dimensions": 1152,
        "pool": "max"
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![image_embed_spec("embed/bad", "some/repo", opts)])
        .build()
        .await;
    assert!(result.is_err(), "unknown pool kind must be rejected");
}

#[tokio::test]
async fn image_embed_alias_rejects_zero_dimensions() {
    let opts = serde_json::json!({
        "onnx_path": "onnx/model.onnx",
        "image_size": 384,
        "dimensions": 0
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![image_embed_spec("embed/bad", "some/repo", opts)])
        .build()
        .await;
    assert!(result.is_err(), "dimensions=0 must be rejected");
}
