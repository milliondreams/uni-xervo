#![cfg(feature = "provider-openai")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::RemoteOpenAIProvider;
use uni_xervo::runtime::ModelRuntime;

fn openai_spec(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
        provider_id: "remote/openai".to_string(),
        model_id: "text-embedding-3-small".to_string(),
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
async fn builder_rejects_unknown_openai_option_key() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({"unknown": true}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Unknown option")
    );
}

#[tokio::test]
async fn builder_accepts_base_url_string() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({"base_url": "http://localhost:8000/v1"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_non_string_base_url() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({"base_url": 42}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a string")
    );
}

#[tokio::test]
async fn builder_rejects_empty_base_url() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({"base_url": ""}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(runtime.err().unwrap().to_string().contains("non-empty URL"));
}

#[tokio::test]
async fn builder_rejects_whitespace_base_url() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({"base_url": "   "}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(runtime.err().unwrap().to_string().contains("non-empty URL"));
}

#[tokio::test]
async fn builder_rejects_relative_base_url() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({"base_url": "openrouter.ai/api/v1"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("absolute http(s) URL")
    );
}

#[tokio::test]
async fn builder_accepts_base_url_with_other_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(RemoteOpenAIProvider::new())
        .catalog(vec![openai_spec(
            ModelTask::Embed,
            serde_json::json!({
                "api_key_env": "MY_OPENAI_KEY",
                "base_url": "https://openrouter.ai/api/v1",
                "embedding_dimensions": 768
            }),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}
