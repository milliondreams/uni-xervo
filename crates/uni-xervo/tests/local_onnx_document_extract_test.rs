//! Fixture-level tests for `local/onnx` × `DocumentExtract`.

#![cfg(feature = "provider-onnx")]

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::error::RuntimeError;
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::DocExtractOptions;

fn doc_spec(alias: &str, model_id: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.to_string(),
        task: ModelTask::DocumentExtract,
        provider_id: "local/onnx".to_string(),
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
async fn local_onnx_advertises_document_extract_capability() {
    use uni_xervo::traits::ModelProvider as _;
    let tasks = LocalOnnxProvider::new().capabilities().supported_tasks;
    assert!(tasks.contains(&ModelTask::DocumentExtract));
}

#[tokio::test]
async fn doc_alias_registers_with_granite_docling_style() {
    let opts = serde_json::json!({
        "style": "granite-docling",
        "onnx_path": "onnx/model.onnx",
        "tokenizer_path": "tokenizer.json",
        "max_seq_len": 2048
    });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![doc_spec(
            "doc/granite",
            "ibm-granite/granite-docling",
            opts,
        )])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "granite-docling style should pass validation: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn doc_alias_registers_with_mineru_style() {
    let opts = serde_json::json!({ "style": "mineru" });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![doc_spec("doc/mineru", "opendatalab/MinerU", opts)])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "mineru style should pass validation: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn doc_alias_registers_with_olmocr_style() {
    let opts = serde_json::json!({ "style": "olmocr" });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![doc_spec("doc/olmocr", "allenai/olmocr", opts)])
        .build()
        .await;
    assert!(
        result.is_ok(),
        "olmocr style should pass validation: err={}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
}

#[tokio::test]
async fn doc_alias_rejects_unknown_style() {
    let opts = serde_json::json!({ "style": "bogus" });
    let result = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![doc_spec("doc/bad", "x/y", opts)])
        .build()
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn doc_alias_extract_returns_unavailable_in_v1() {
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![doc_spec(
            "doc/granite",
            "ibm-granite/granite-docling",
            serde_json::json!({ "style": "granite-docling" }),
        )])
        .build()
        .await
        .unwrap();
    let model = runtime.document_extractor("doc/granite").await.unwrap();
    let res = model
        .extract(
            vec![],
            DocExtractOptions {
                output: uni_xervo::traits::DocOutputFormat::Markdown,
                include_tables: true,
                include_formulas: true,
                include_bboxes: true,
            },
        )
        .await;
    assert!(
        matches!(
            res,
            Err(RuntimeError::Unavailable) | Err(RuntimeError::Timeout)
        ),
        "v1 extract must surface Unavailable (or Timeout if retried)"
    );
}
