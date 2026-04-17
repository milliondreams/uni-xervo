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
