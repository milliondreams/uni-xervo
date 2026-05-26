//! Fixture-level tests for `local/whisper-cpp` × `Transcribe`.
//!
//! v1 ships the provider scaffold; `recognize`/`transcribe` returns
//! `Unavailable` until the whisper-rs integration lands. These tests cover
//! capability advertising, catalog-registration, options validation, and
//! the documented `Unavailable` response.

#![cfg(feature = "provider-whisper-cpp")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalWhisperCppProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{AudioInput, TranscribeOptions};

fn transcribe_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Transcribe,
        provider_id: "local/whisper-cpp".to_string(),
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
async fn whisper_cpp_advertises_transcribe_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let tasks = LocalWhisperCppProvider::new()
        .capabilities()
        .supported_tasks;
    assert!(tasks.contains(&ModelTask::Transcribe));
}

#[tokio::test]
async fn whisper_cpp_alias_registers_with_defaults() {
    let result = ModelRuntime::builder()
        .register_provider(LocalWhisperCppProvider::new())
        .catalog(vec![transcribe_spec(
            "asr/whisper",
            "ggerganov/whisper.cpp",
            serde_json::Value::Null,
        )])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "lazy-warmup alias should build: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn whisper_cpp_alias_accepts_documented_options() {
    let opts = serde_json::json!({
        "model_path": "ggml-base.bin",
        "default_language": "en"
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalWhisperCppProvider::new())
        .catalog(vec![transcribe_spec(
            "asr/whisper",
            "ggerganov/whisper.cpp",
            opts,
        )])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "documented options should validate: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn whisper_cpp_alias_rejects_unknown_option() {
    let opts = serde_json::json!({ "not_a_real_option": "x" });
    let result = ModelRuntime::builder()
        .register_provider(LocalWhisperCppProvider::new())
        .catalog(vec![transcribe_spec(
            "asr/whisper",
            "ggerganov/whisper.cpp",
            opts,
        )])
        .build()
        .await;
    assert!(result.is_err(), "unknown option key must fail validation");
}

#[tokio::test]
async fn whisper_cpp_transcribe_returns_unavailable_in_v1() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalWhisperCppProvider::new())
        .catalog(vec![transcribe_spec(
            "asr/whisper",
            "ggerganov/whisper.cpp",
            serde_json::Value::Null,
        )])
        .build()
        .await
        .unwrap();
    let model = runtime.transcriber("asr/whisper").await.unwrap();
    let pcm = AudioInput::Pcm {
        sample_rate: 16000,
        channels: 1,
        samples: vec![0.0; 1600],
    };
    let res = model.transcribe(pcm, TranscribeOptions::default()).await;
    // The Instrumented wrapper considers Unavailable retryable. With no retry
    // configured it surfaces directly; with `retry` set it would retry once
    // and still surface. Either way the call returns Err.
    assert!(
        matches!(
            res,
            Err(RuntimeError::Unavailable) | Err(RuntimeError::Timeout)
        ),
        "v1 transcribe must surface Unavailable (or Timeout if retried)"
    );
}
