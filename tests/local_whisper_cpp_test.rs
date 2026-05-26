//! Fixture-level tests for `local/whisper-cpp` × `Transcribe`.
//!
//! These tests cover capability advertising, catalog-registration, and
//! options validation. They do not load a ggml model (lazy warmup) so the
//! suite runs without network access. The expensive end-to-end live test
//! (gated on `EXPENSIVE_TESTS=1`) downloads `ggml-tiny.en.bin` and
//! exercises the real whisper-rs path.

#![cfg(feature = "provider-whisper-cpp")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalWhisperCppProvider;
use uni_xervo::runtime::ModelRuntime;

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
async fn whisper_cpp_transcribe_fails_with_load_error_for_nonexistent_model() {
    // Use a deliberately invalid model_id so the HF download fails fast
    // (no real network round-trip succeeds). Confirms the load path
    // returns a useful error rather than panicking.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalWhisperCppProvider::new())
        .catalog(vec![transcribe_spec(
            "asr/whisper-bad",
            "uni-xervo-test/this-repo-does-not-exist",
            serde_json::json!({ "model_path": "no-such-file.bin" }),
        )])
        .build()
        .await
        .unwrap();
    let res = runtime.transcriber("asr/whisper-bad").await;
    assert!(
        res.is_err(),
        "transcriber resolution must fail when the ggml model can't be downloaded"
    );
}

// ---------------------------------------------------------------------------
// Live end-to-end test — gated on EXPENSIVE_TESTS=1
//
// Downloads `ggml-tiny.en.bin` (~75 MB) from `ggerganov/whisper.cpp` and
// transcribes a short synthetic silence buffer. Asserts the call returns
// a non-error result with at least zero segments (silence may produce 0
// segments — what we care about is that the path doesn't error).
// ---------------------------------------------------------------------------

fn should_run_expensive_tests() -> bool {
    match std::env::var("EXPENSIVE_TESTS") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

#[tokio::test]
#[ignore]
async fn whisper_cpp_end_to_end_with_ggml_tiny() {
    if !should_run_expensive_tests() {
        eprintln!("Skipping — set EXPENSIVE_TESTS=1 to run");
        return;
    }

    use uni_xervo::traits::{AudioInput, TranscribeOptions};

    let runtime = ModelRuntime::builder()
        .register_provider(LocalWhisperCppProvider::new())
        .catalog(vec![transcribe_spec(
            "asr/whisper-tiny",
            "ggerganov/whisper.cpp",
            serde_json::json!({
                "model_path": "ggml-tiny.en.bin",
                "default_language": "en"
            }),
        )])
        .build()
        .await
        .expect("runtime build");

    let model = runtime
        .transcriber("asr/whisper-tiny")
        .await
        .expect("load ggml-tiny");

    // 1 second of silence at 16 kHz mono. whisper.cpp can transcribe this;
    // typically returns 0 segments or a no-speech token. Either is fine —
    // we only assert the call succeeds end-to-end.
    let audio = AudioInput::Pcm {
        sample_rate: 16000,
        channels: 1,
        samples: vec![0.0; 16_000],
    };
    let result = model
        .transcribe(audio, TranscribeOptions::default())
        .await
        .expect("transcribe");
    assert_eq!(result.language, "en");
    // segments count is not asserted — silence may produce zero.
    let _ = result.segments.len();
}
