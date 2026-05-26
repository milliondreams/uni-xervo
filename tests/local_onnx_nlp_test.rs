//! Fixture-level tests for `local/onnx` × `Nlp` task wiring.
//!
//! These tests do not download model weights — they validate the
//! catalog-registration path (options validation, provider capability) and
//! the resolver/handle-cache wiring through the public `ModelRuntime`
//! surface.
//!
//! Live model-load + inference coverage sits in
//! [`onnx_models_expensive_test.rs`] under
//! `kniv_deberta_nlp_cascade_end_to_end`, gated on `EXPENSIVE_TESTS=1`.

#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn nlp_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Nlp,
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
async fn local_onnx_advertises_nlp_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let provider = LocalOnnxProvider::new();
    let tasks = provider.capabilities().supported_tasks;
    assert!(
        tasks.contains(&ModelTask::Nlp),
        "local/onnx must advertise Nlp capability"
    );
}

#[tokio::test]
async fn nlp_alias_registration_passes_validation_with_defaults() {
    // No options supplied — provider should accept (defaults will resolve
    // canonical paths on the kniv repo at load time).
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(
            "nlp/kniv",
            "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
            serde_json::Value::Null,
        )])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "Lazy-warmup nlp alias should build without downloading weights: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn nlp_alias_registration_accepts_custom_options() {
    let opts = serde_json::json!({
        "onnx_path": "onnx/cascade-int8.onnx",
        "tokenizer_path": "tokenizer.json",
        "label_maps_path": "label_maps.json",
        "max_seq_len": 128
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(
            "nlp/kniv-int8",
            "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
            opts,
        )])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "custom options should validate: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn nlp_alias_rejects_unknown_option_key() {
    let opts = serde_json::json!({ "not_a_real_option": "x" });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(
            "nlp/bad",
            "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
            opts,
        )])
        .build()
        .await;
    assert!(
        result.is_err(),
        "unknown option key must fail options validation"
    );
}

#[tokio::test]
async fn nlp_alias_rejects_zero_max_seq_len() {
    let opts = serde_json::json!({ "max_seq_len": 0 });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![nlp_spec(
            "nlp/zero",
            "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
            opts,
        )])
        .build()
        .await;
    assert!(result.is_err(), "max_seq_len=0 must be rejected");
}
