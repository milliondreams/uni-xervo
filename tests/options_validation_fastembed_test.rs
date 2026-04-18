#![cfg(feature = "provider-fastembed")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalFastEmbedProvider;
use uni_xervo::runtime::ModelRuntime;

fn fastembed_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "embed/fastembed".to_string(),
        task: ModelTask::Embed,
        provider_id: "local/fastembed".to_string(),
        model_id: "all-MiniLM-L6-v2".to_string(),
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
async fn builder_accepts_valid_fastembed_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalFastEmbedProvider::new())
        .catalog(vec![fastembed_spec(serde_json::json!({
            "cache_dir": "/tmp/models",
            "execution_providers": ["cpu", "cuda"]
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_fastembed_execution_provider() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalFastEmbedProvider::new())
        .catalog(vec![fastembed_spec(serde_json::json!({
            "execution_providers": ["coreml"]
        }))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Execution provider 'coreml'")
    );
}

#[tokio::test]
async fn builder_rejects_non_array_fastembed_execution_providers() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalFastEmbedProvider::new())
        .catalog(vec![fastembed_spec(serde_json::json!({
            "execution_providers": "cuda"
        }))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be an array of strings")
    );
}
