//! Catalog-time option validation for the `embed_sparse` / `embed_multi_vector`
//! tasks on `local/onnx`.
//!
//! These exercise [`validate_provider_options`] via builder construction without
//! loading any model (validation runs before download), so they are cheap and
//! always-on. Real model loads live in the `*_expensive_test.rs` files.

#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn spec(alias: &str, task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task,
        provider_id: "local/onnx".to_string(),
        model_id: "some/model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

async fn build(spec: ModelAliasSpec) -> uni_xervo::error::Result<std::sync::Arc<ModelRuntime>> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await
}

#[tokio::test]
async fn sparse_accepts_valid_options() {
    let res = build(spec(
        "embed_sparse/splade",
        ModelTask::EmbedSparse,
        serde_json::json!({
            "sparse_method": "mlm",
            "tokenizer_path": "tokenizer.json",
            "max_seq_len": 256,
            "top_k": 128
        }),
    ))
    .await;
    assert!(
        res.is_ok(),
        "valid sparse options rejected: {:?}",
        res.err()
    );
}

#[tokio::test]
async fn sparse_rejects_unknown_sparse_method() {
    let res = build(spec(
        "embed_sparse/bad",
        ModelTask::EmbedSparse,
        serde_json::json!({ "sparse_method": "bm25" }),
    ))
    .await;
    assert!(res.is_err());
}

#[tokio::test]
async fn sparse_rejects_unknown_key() {
    let res = build(spec(
        "embed_sparse/bad",
        ModelTask::EmbedSparse,
        serde_json::json!({ "pooling": "cls" }),
    ))
    .await;
    assert!(res.is_err(), "pooling is not a valid sparse key");
}

#[tokio::test]
async fn multi_vector_accepts_valid_options() {
    let res = build(spec(
        "embed_multi_vector/colbert",
        ModelTask::EmbedMultiVector,
        serde_json::json!({
            "dimensions": 96,
            "normalize": true,
            "drop_special_tokens": false,
            "output_index": 0,
            "max_seq_len": 300
        }),
    ))
    .await;
    assert!(
        res.is_ok(),
        "valid multi-vector options rejected: {:?}",
        res.err()
    );
}

#[tokio::test]
async fn multi_vector_rejects_non_boolean_normalize() {
    let res = build(spec(
        "embed_multi_vector/bad",
        ModelTask::EmbedMultiVector,
        serde_json::json!({ "normalize": "yes" }),
    ))
    .await;
    assert!(res.is_err());
}

#[tokio::test]
async fn multi_vector_rejects_unknown_key() {
    let res = build(spec(
        "embed_multi_vector/bad",
        ModelTask::EmbedMultiVector,
        serde_json::json!({ "sparse_method": "mlm" }),
    ))
    .await;
    assert!(
        res.is_err(),
        "sparse_method is not a valid multi-vector key"
    );
}
