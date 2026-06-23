#![cfg(feature = "provider-onnx")]

use ndarray::{Array1, Array2, arr1, arr2};
use std::path::{Path, PathBuf};
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{ModelProvider, TensorBatch, TensorValue};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("onnx")
}

fn make_spec(alias: &str, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::Raw,
        provider_id: "local/onnx".to_string(),
        model_id: model_id.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }
}

async fn runtime_for(spec: ModelAliasSpec) -> std::sync::Arc<ModelRuntime> {
    ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new().with_base_dir(fixture_dir()))
        .catalog(vec![spec])
        .build()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_local_onnx_identity_run() {
    let runtime = runtime_for(make_spec("raw/identity", "identity_f32.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/identity").await.unwrap();

    let input = arr2(&[[1.0_f32, 2.0, 3.0, 4.0]]).into_dyn();
    let mut batch = TensorBatch::new();
    batch.insert("input", TensorValue::F32(input.clone()));

    let output = runner.run(&batch).await.unwrap();
    assert_eq!(output.get("output"), Some(&TensorValue::F32(input)));
}

#[tokio::test]
async fn test_local_onnx_two_input_two_output() {
    let runtime = runtime_for(make_spec("raw/two-io", "two_input_two_output.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/two-io").await.unwrap();

    let mut batch = TensorBatch::new();
    batch.insert("lhs", TensorValue::F32(arr1(&[2.0_f32, 5.0]).into_dyn()));
    batch.insert("rhs", TensorValue::F32(arr1(&[1.0_f32, 3.0]).into_dyn()));

    let output = runner.run(&batch).await.unwrap();
    assert_eq!(
        output.get("sum"),
        Some(&TensorValue::F32(arr1(&[3.0_f32, 8.0]).into_dyn()))
    );
    assert_eq!(
        output.get("diff"),
        Some(&TensorValue::F32(arr1(&[1.0_f32, 2.0]).into_dyn()))
    );
}

#[tokio::test]
async fn test_local_onnx_dynamic_batch_run_batch() {
    let runtime = runtime_for(make_spec("raw/dynamic", "dynamic_batch_linear.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/dynamic").await.unwrap();

    let sample = |values: [f32; 4]| {
        let mut batch = TensorBatch::new();
        batch.insert(
            "input",
            TensorValue::F32(Array1::from_vec(values.to_vec()).into_dyn()),
        );
        batch
    };

    let samples = vec![sample([1.0, 1.0, 1.0, 1.0]), sample([2.0, 0.0, 0.0, 0.0])];
    let outputs = runner.run_batch(&samples).await.unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].get("output"),
        Some(&TensorValue::F32(arr1(&[10.5_f32]).into_dyn()))
    );
    assert_eq!(
        outputs[1].get("output"),
        Some(&TensorValue::F32(arr1(&[2.5_f32]).into_dyn()))
    );
}

#[tokio::test]
async fn test_local_onnx_static_batch_size_enforced() {
    let runtime = runtime_for(make_spec("raw/static", "static_batch_8.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/static").await.unwrap();

    let sample = || {
        let mut batch = TensorBatch::new();
        batch.insert(
            "input",
            TensorValue::F32(arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn()),
        );
        batch
    };

    let ok_samples = (0..8).map(|_| sample()).collect::<Vec<_>>();
    let ok = runner.run_batch(&ok_samples).await.unwrap();
    assert_eq!(ok.len(), 8);

    let bad_samples = (0..7).map(|_| sample()).collect::<Vec<_>>();
    let err = runner.run_batch(&bad_samples).await.unwrap_err();
    assert!(matches!(err, RuntimeError::OnnxBatchStackingFailure { .. }));
}

#[tokio::test]
async fn test_local_onnx_no_batch_sequential_fallback() {
    let runtime = runtime_for(make_spec("raw/linear", "linear_4in_1out.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/linear").await.unwrap();

    let sample = |values: [f32; 4]| {
        let mut batch = TensorBatch::new();
        batch.insert(
            "input",
            TensorValue::F32(Array1::from_vec(values.to_vec()).into_dyn()),
        );
        batch
    };

    let samples = vec![sample([1.0, 1.0, 1.0, 1.0]), sample([0.0, 1.0, 0.0, 0.0])];
    let outputs = runner.run_batch(&samples).await.unwrap();

    assert_eq!(
        outputs[0].get("output"),
        Some(&TensorValue::F32(arr1(&[10.5_f32]).into_dyn()))
    );
    assert_eq!(
        outputs[1].get("output"),
        Some(&TensorValue::F32(arr1(&[2.5_f32]).into_dyn()))
    );
}

#[tokio::test]
async fn test_local_onnx_missing_file_error() {
    let runtime = runtime_for(make_spec("raw/missing", "./does-not-exist.onnx")).await;
    let err = match runtime.raw_tensor_model("raw/missing").await {
        Ok(_) => panic!("expected missing model error"),
        Err(err) => err,
    };
    assert!(matches!(err, RuntimeError::OnnxModelNotFound { .. }));
}

#[tokio::test]
async fn test_local_onnx_malformed_model_error() {
    let runtime = runtime_for(make_spec("raw/malformed", "malformed.onnx")).await;
    let err = match runtime.raw_tensor_model("raw/malformed").await {
        Ok(_) => panic!("expected malformed model error"),
        Err(err) => err,
    };
    assert!(matches!(err, RuntimeError::OnnxLoadFailure { .. }));
}

#[tokio::test]
async fn test_local_onnx_bad_input_errors() {
    let runtime = runtime_for(make_spec("raw/identity-bad", "identity_f32.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/identity-bad").await.unwrap();

    let mut missing = TensorBatch::new();
    missing.insert("wrong", TensorValue::F32(Array2::zeros((1, 4)).into_dyn()));
    let err = runner.run(&missing).await.unwrap_err();
    assert!(matches!(err, RuntimeError::OnnxInputMissing { .. }));

    let mut wrong_shape = TensorBatch::new();
    wrong_shape.insert(
        "input",
        TensorValue::F32(arr1(&[1.0_f32, 2.0, 3.0, 4.0]).into_dyn()),
    );
    let err = runner.run(&wrong_shape).await.unwrap_err();
    assert!(matches!(err, RuntimeError::OnnxInputShapeMismatch { .. }));
}

#[tokio::test]
async fn test_local_onnx_batch_rejects_unexpected_inputs() {
    let runtime = runtime_for(make_spec("raw/dynamic-extra", "dynamic_batch_linear.onnx")).await;
    let runner = runtime.raw_tensor_model("raw/dynamic-extra").await.unwrap();

    let mut batch = TensorBatch::new();
    batch.insert(
        "input",
        TensorValue::F32(arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn()),
    );
    batch.insert("ignored", TensorValue::F32(arr1(&[1.0_f32]).into_dyn()));

    let mut valid = TensorBatch::new();
    valid.insert(
        "input",
        TensorValue::F32(arr1(&[2.0_f32, 0.0, 0.0, 0.0]).into_dyn()),
    );

    let batches = vec![batch.clone(), valid];
    let err = runner.run_batch(&batches).await.unwrap_err();

    assert!(matches!(err, RuntimeError::OnnxInvocationFailure { .. }));
}

#[test]
fn capabilities_advertise_sparse_and_multi_vector() {
    let caps = LocalOnnxProvider::new().capabilities();
    assert!(
        caps.supported_tasks.contains(&ModelTask::EmbedSparse),
        "local/onnx must advertise EmbedSparse"
    );
    assert!(
        caps.supported_tasks.contains(&ModelTask::EmbedMultiVector),
        "local/onnx must advertise EmbedMultiVector"
    );
}
