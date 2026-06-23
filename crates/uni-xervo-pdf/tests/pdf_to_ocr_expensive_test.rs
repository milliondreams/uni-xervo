//! End-to-end (real model) cross-crate test: PDF -> rasterize -> OCR.
//!
//! Generates a born-digital PDF, rasterizes it with `uni-xervo-pdf`'s hayro
//! backend, and feeds the page image into `uni-xervo`'s PP-OCRv5 `OcrModel` to
//! confirm the rasterized text is recognized. This exercises the whole
//! image-tier pipeline the two crates form together.
//!
//! Gated by both:
//! - `#![cfg(feature = "hayro")]` (the rasterizer); the OCR model is reached via
//!   `uni-xervo`'s `provider-onnx`, enabled through this crate's dev-dependency.
//! - `EXPENSIVE_TESTS=1` (downloads ~96 MB of ONNX weights from HuggingFace).
//!
//! Run with:
//! ```sh
//! EXPENSIVE_TESTS=1 cargo test -p uni-xervo-pdf \
//!     --test pdf_to_ocr_expensive_test -- --ignored
//! ```

#![cfg(all(feature = "pdf-input", feature = "hayro"))]

mod common;

use std::env;

use common::make_text_pdf;
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uni_xervo::provider::LocalOnnxProvider;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo_pdf::{HayroRasterizer, Rasterizer};

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

fn ocr_spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "ocr/ppocr-en".to_string(),
        task: ModelTask::Ocr,
        provider_id: "local/onnx".to_string(),
        model_id: "monkt/paddleocr-onnx".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "onnx_path": "languages/english/rec.onnx",
            "char_dict_path": "languages/english/dict.txt",
            "image_height": 48,
            "image_width": 320,
            "normalization": "siglip",
            "det_onnx_path": "detection/v5/det.onnx"
        }),
    }
}

/// A PDF rendered by `uni-xervo-pdf` is read back by `uni-xervo`'s OCR model.
#[tokio::test]
#[ignore]
async fn rasterized_pdf_text_is_recognized_by_ocr() {
    require_expensive_tests!();

    // Rasterize a generated single-page PDF to a page image.
    let pdf = make_text_pdf("Hello World OCR test 2026");
    let pages = HayroRasterizer::new()
        .rasterize_pages(&pdf, &[1])
        .expect("rasterize page 1");
    let page_image = pages[0].image.clone();

    // Recognize the rasterized page with PP-OCRv5.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec()])
        .build()
        .await
        .expect("build runtime");
    let ocr = runtime
        .ocr_model("ocr/ppocr-en")
        .await
        .expect("load ocr model");
    let results = ocr.recognize(vec![page_image]).await.expect("recognize");
    assert_eq!(results.len(), 1, "one result per input page");

    let text = results[0].plain_text.to_lowercase();
    eprintln!("recognized: {text:?}");
    // Allow OCR imperfection: require a few of the known tokens.
    let hits = ["hello", "world", "ocr", "test", "2026"]
        .iter()
        .filter(|t| text.contains(*t))
        .count();
    assert!(
        hits >= 2,
        "expected to recognize >= 2 known tokens, matched {hits} in {text:?}"
    );
}
