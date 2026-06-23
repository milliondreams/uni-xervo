//! End-to-end (real model) test of the document-extraction path: olmOCR-2 on
//! the mistral.rs vision pipeline through `LocalMistralRsProvider`.
//!
//! This is the live coverage for the generative `DocumentExtractionModel` rung
//! (the `local/onnx` document-extract path is a scaffold that returns
//! `Unavailable` pending upstream ONNX exports, and is covered separately).
//!
//! ⚠ Unverified end-to-end / likely affected by the same upstream breakage.
//! olmOCR-2 is a Qwen2.5-VL fine-tune and runs on mistral.rs's vision pipeline.
//! Sibling vision-generation tests confirm that pipeline currently fails at
//! inference on `mistralrs-core` 0.8.1 (Idefics3 and Qwen2-VL both load but error
//! during the forward pass; see `mistralrs_vision_describe_expensive_test.rs` and
//! mistral.rs issues #1068 / #1025 / #935). This test has not been run to green;
//! it may hit the same class of bug until upstream is fixed.
//!
//! Gated by both:
//! - `#![cfg(feature = "provider-mistralrs")]`
//! - `EXPENSIVE_TESTS=1`.
//!
//! Hardware: `allenai/olmOCR-2-7B-1025` is a 7B-parameter model (multi-GB
//! download). A GPU is strongly recommended; CPU inference with ISQ `Q4K`
//! works but is slow. Output is generative and non-deterministic, so this is a
//! smoke test (non-empty Markdown), not an exact-text assertion.
//!
//! Run with:
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test --features provider-mistralrs \
//!     --test mistralrs_doc_extract_expensive_test -- --ignored
//! ```

#![cfg(feature = "provider-mistralrs")]

use std::env;

use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{DocExtractOptions, DocOutputFormat, ImageInput};

fn should_run_expensive_tests() -> bool {
    match env::var("EXPENSIVE_TESTS") {
        Ok(v) => matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

macro_rules! require_expensive_tests {
    () => {
        if !should_run_expensive_tests() {
            eprintln!("Skipping test - set EXPENSIVE_TESTS=1 to run");
            return;
        }
    };
}

const MODEL_ID: &str = "allenai/olmOCR-2-7B-1025";

fn olmocr_doc_spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "docext/olmocr".to_string(),
        task: ModelTask::DocumentExtract,
        provider_id: "local/mistralrs".to_string(),
        model_id: MODEL_ID.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        // Document extraction always runs on the vision pipeline; `style`
        // selects the output parser. ISQ Q4K keeps the footprint manageable.
        options: serde_json::json!({
            "isq": "Q4K",
            "style": "olmocr"
        }),
    }
}

fn page_image(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/ocr/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

/// olmOCR-2 reads a page image and returns non-empty Markdown.
#[tokio::test]
#[ignore]
async fn olmocr2_extracts_markdown_from_page() {
    require_expensive_tests!();

    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![olmocr_doc_spec()])
        .build()
        .await
        .expect("build runtime");
    let extractor = runtime
        .document_extractor("docext/olmocr")
        .await
        .expect("load document extractor");

    let page = ImageInput::Bytes {
        data: page_image("text_lines.png"),
        media_type: "image/png".to_string(),
    };
    let options = DocExtractOptions {
        output: DocOutputFormat::Markdown,
        include_tables: true,
        include_formulas: true,
        include_bboxes: false,
    };
    let results = extractor
        .extract(vec![page], options)
        .await
        .expect("extract");
    assert_eq!(results.len(), 1, "one result per input page");

    eprintln!("plain_markdown:\n{}", results[0].plain_markdown);
    assert!(
        !results[0].plain_markdown.trim().is_empty(),
        "expected non-empty Markdown from a page with text"
    );
}
