#![cfg(feature = "provider-mistralrs")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;

fn mistralrs_spec(options: serde_json::Value) -> ModelAliasSpec {
    mistralrs_spec_with_task(ModelTask::Embed, options)
}

fn mistralrs_spec_with_task(task: ModelTask, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "test/default".to_string(),
        task,
        provider_id: "local/mistralrs".to_string(),
        model_id: "test-model".to_string(),
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
async fn builder_accepts_valid_dtype_option() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": "f32"}))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_dtype_auto() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": "auto"}))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_dtype_value() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": "int8"}))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be one of")
    );
}

#[tokio::test]
async fn builder_rejects_non_string_dtype() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(serde_json::json!({"dtype": 16}))])
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
async fn builder_accepts_dtype_with_other_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec(
            serde_json::json!({"dtype": "f32", "force_cpu": true}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

// ---------------------------------------------------------------------------
// Pipeline validation tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_accepts_valid_pipeline_vision() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "vision"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_valid_pipeline_diffusion() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_valid_pipeline_speech() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "speech"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "audio"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be one of")
    );
}

#[tokio::test]
async fn builder_rejects_gguf_for_vision_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "vision", "gguf_files": ["model.gguf"]}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("not supported for the vision pipeline")
    );
}

#[tokio::test]
async fn builder_accepts_diffusion_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "diffusion_loader_type": "flux"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_diffusion_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "diffusion_loader_type": "invalid"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be one of")
    );
}

#[tokio::test]
async fn builder_rejects_isq_for_diffusion_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "isq": "Q4K"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("not supported for the diffusion pipeline")
    );
}

#[tokio::test]
async fn builder_accepts_speech_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "speech", "speech_loader_type": "dia"}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_speech_loader_type() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "speech", "speech_loader_type": "invalid"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be one of")
    );
}

// ---------------------------------------------------------------------------
// Shared validation tests (dtype / force_cpu across pipelines)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_rejects_invalid_dtype_for_vision_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "vision", "dtype": "int8"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be one of")
    );
}

#[tokio::test]
async fn builder_rejects_non_bool_force_cpu_for_diffusion_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "diffusion", "force_cpu": "yes"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a boolean")
    );
}

// ---------------------------------------------------------------------------
// Auto-device-mapper override knobs (max_seq_len, max_batch_size,
// max_image_shape, max_num_images).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_accepts_max_seq_len_and_batch_size() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"max_seq_len": 1024, "max_batch_size": 2}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok(), "expected runtime build to succeed");
}

#[tokio::test]
async fn builder_accepts_max_image_shape_for_vision_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({
                "pipeline": "vision",
                "max_seq_len": 1024,
                "max_image_shape": [224, 224],
                "max_num_images": 1,
            }),
        )])
        .build()
        .await;

    assert!(runtime.is_ok(), "expected runtime build to succeed");
}

#[tokio::test]
async fn builder_rejects_max_image_shape_with_wrong_arity() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"pipeline": "vision", "max_image_shape": [224]}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a 2-element array of positive integers")
    );
}

#[tokio::test]
async fn builder_rejects_max_image_shape_on_text_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"max_image_shape": [224, 224]}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("only supported for the vision pipeline")
    );
}

#[tokio::test]
async fn builder_rejects_max_seq_len_zero() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"max_seq_len": 0}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be greater than 0")
    );
}

// ---------------------------------------------------------------------------
// UQFF (mistralrs pre-quantized) loading.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_accepts_uqff_files_for_text_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"uqff_files": ["q4k-0.uqff"]}),
        )])
        .build()
        .await;

    assert!(runtime.is_ok(), "expected runtime build to succeed");
}

#[tokio::test]
async fn builder_accepts_uqff_files_for_vision_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({
                "pipeline": "vision",
                "uqff_files": ["q4k-0.uqff"],
            }),
        )])
        .build()
        .await;

    assert!(runtime.is_ok(), "expected runtime build to succeed");
}

#[tokio::test]
async fn builder_rejects_uqff_files_with_isq() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"uqff_files": ["q4k-0.uqff"], "isq": "Q4K"}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("incompatible with 'uqff_files'")
    );
}

#[tokio::test]
async fn builder_rejects_uqff_files_with_gguf_files() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({
                "uqff_files": ["q4k-0.uqff"],
                "gguf_files": ["model.gguf"],
            }),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("mutually exclusive")
    );
}

#[tokio::test]
async fn builder_rejects_uqff_files_on_diffusion_pipeline() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({
                "pipeline": "diffusion",
                "diffusion_loader_type": "flux",
                "uqff_files": ["q4k-0.uqff"],
            }),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("not supported for the diffusion pipeline")
    );
}

#[tokio::test]
async fn builder_rejects_empty_uqff_files_array() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"uqff_files": []}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a non-empty array of strings")
    );
}

#[tokio::test]
async fn builder_rejects_non_string_uqff_files_entry() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![mistralrs_spec_with_task(
            ModelTask::Generate,
            serde_json::json!({"uqff_files": [123]}),
        )])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("must be a non-empty array of strings")
    );
}
