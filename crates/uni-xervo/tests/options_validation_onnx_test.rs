#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;

fn onnx_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "raw/onnx".to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: "model.onnx".to_string(),
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
async fn builder_accepts_valid_onnx_options() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "artifact": "onnx/model.onnx",
            "max_batch_size": 8,
            "execution_providers": ["cpu"],
            "graph_optimization_level": "all",
            "inter_op_num_threads": 1,
            "intra_op_num_threads": 2
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_cuda_onnx_execution_provider() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "execution_providers": ["cuda"]
        }))])
        .build()
        .await;

    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_invalid_onnx_execution_provider() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "execution_providers": ["rocm"]
        }))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Execution provider 'rocm'")
    );
}

#[tokio::test]
async fn builder_rejects_non_string_onnx_artifact() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![onnx_spec(serde_json::json!({
            "artifact": 7
        }))])
        .build()
        .await;

    assert!(runtime.is_err());
    assert!(
        runtime
            .err()
            .unwrap()
            .to_string()
            .contains("Option 'artifact'")
    );
}

// ---------------------------------------------------------------------------
// OCR detection-stage option validation
// ---------------------------------------------------------------------------

fn ocr_spec(options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "ocr/test".to_string(),
        task: ModelTask::Ocr,
        provider_id: "local/onnx".to_string(),
        model_id: "test/repo".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

async fn build_ocr(
    options: serde_json::Value,
) -> Result<std::sync::Arc<ModelRuntime>, uni_xervo::error::RuntimeError> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec(options)])
        .build()
        .await
}

#[tokio::test]
async fn builder_accepts_ocr_without_detection() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt"
    }))
    .await;
    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_accepts_ocr_with_detection_options() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_model_id": "other/det",
        "det_limit_side": 960,
        "det_bin_threshold": 0.3,
        "det_box_score_threshold": 0.6,
        "det_unclip_ratio": 1.5,
        "det_min_box_size": 3,
        "det_input_name": "x"
    }))
    .await;
    assert!(runtime.is_ok());
}

#[tokio::test]
async fn builder_rejects_unknown_det_key() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_bogus": 1
    }))
    .await;
    assert!(runtime.is_err());
}

#[tokio::test]
async fn builder_rejects_det_bin_threshold_out_of_range() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_bin_threshold": 1.5
    }))
    .await;
    assert!(runtime.is_err());
}

#[tokio::test]
async fn builder_rejects_det_unclip_ratio_negative() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_unclip_ratio": -1.0
    }))
    .await;
    assert!(runtime.is_err());
}

#[tokio::test]
async fn builder_rejects_det_bin_threshold_not_a_number() {
    let runtime = build_ocr(serde_json::json!({
        "onnx_path": "rec.onnx",
        "char_dict_path": "dict.txt",
        "det_onnx_path": "det.onnx",
        "det_bin_threshold": "x"
    }))
    .await;
    assert!(runtime.is_err());
}
