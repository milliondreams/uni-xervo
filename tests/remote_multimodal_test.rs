//! Fixture-level tests for remote multimodal embedders.
//!
//! Validates catalog-registration, capability advertising, and resolver
//! wiring for `remote/cohere` and `remote/gemini` × `EmbedMultimodal`.
//! Live API tests sit behind `EXPENSIVE_TESTS=1` + API keys in env.

#![cfg(all(feature = "provider-cohere", feature = "provider-gemini"))]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::{RemoteCohereProvider, RemoteGeminiProvider};
use uni_xervo::runtime::ModelRuntime;

fn multimodal_spec(alias: &str, provider_id: &str, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::EmbedMultimodal,
        provider_id: provider_id.to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Object(serde_json::Map::new()),
    }
}

#[tokio::test]
async fn cohere_advertises_embed_multimodal_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let tasks = RemoteCohereProvider::new().capabilities().supported_tasks;
    assert!(tasks.contains(&ModelTask::EmbedMultimodal));
}

#[tokio::test]
async fn gemini_advertises_embed_multimodal_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let tasks = RemoteGeminiProvider::new().capabilities().supported_tasks;
    assert!(tasks.contains(&ModelTask::EmbedMultimodal));
}

#[tokio::test]
async fn cohere_multimodal_alias_registers_with_defaults() {
    let mut spec = multimodal_spec("embed/cohere-mm", "remote/cohere", "embed-v4.0");
    spec.options = serde_json::json!({ "api_key_env": "CO_API_KEY_UNUSED" });
    let result = ModelRuntime::builder()
        .register_provider(RemoteCohereProvider::new())
        .catalog(vec![spec])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "cohere multimodal alias should validate: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn gemini_multimodal_alias_registers_with_defaults() {
    let mut spec = multimodal_spec("embed/gemini-mm", "remote/gemini", "gemini-embedding-001");
    spec.options = serde_json::json!({ "api_key_env": "GEMINI_API_KEY_UNUSED" });
    let result = ModelRuntime::builder()
        .register_provider(RemoteGeminiProvider::new())
        .catalog(vec![spec])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "gemini multimodal alias should validate: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}
