use ndarray::{ArrayD, arr1};
use uni_xervo::api::ModelTask;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{TensorBatch, TensorDtype, TensorValue};

mod common;
use common::mock_support::{MockProvider, make_spec};

#[test]
fn tensor_value_reports_dtype_and_shape() {
    let tensor = TensorValue::F32(ArrayD::from_elem(vec![2, 3], 1.0));
    assert_eq!(tensor.dtype(), TensorDtype::F32);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.ndim(), 2);
}

#[test]
fn tensor_batch_preserves_insertion_order() {
    let mut batch = TensorBatch::new();
    batch.insert("b", TensorValue::I64(arr1(&[1_i64]).into_dyn()));
    batch.insert("a", TensorValue::I64(arr1(&[2_i64]).into_dyn()));

    let keys: Vec<_> = batch.tensors.keys().cloned().collect();
    assert_eq!(keys, vec!["b".to_string(), "a".to_string()]);
}

#[tokio::test]
async fn runtime_resolves_mock_onnx_runner() {
    let runtime = ModelRuntime::builder()
        .register_provider(MockProvider::raw_only())
        .catalog(vec![make_spec(
            "raw/test",
            ModelTask::Raw,
            "mock/raw",
            "test-model",
        )])
        .build()
        .await
        .unwrap();

    let runner = runtime.onnx_runner("raw/test").await.unwrap();
    let mut batch = TensorBatch::new();
    batch.insert("input", TensorValue::I64(arr1(&[1_i64, 2, 3]).into_dyn()));

    let output = runner.run(&batch).await.unwrap();
    assert_eq!(output, batch);
    assert_eq!(runner.max_batch_size(), 1);
}
