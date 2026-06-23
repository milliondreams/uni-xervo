//! OCR example: recognize text in an image with the local ONNX provider.
//!
//! Builds a [`ModelRuntime`] with a single PP-OCRv5 alias (two-stage DBNet
//! detection + CRNN/CTC recognition) and reads the bundled sample image,
//! printing the recognized text plus each block's bounding box and confidence.
//!
//! The model (`monkt/paddleocr-onnx`, Apache-2.0, ~96 MB) is downloaded from
//! HuggingFace on first run and cached for later runs.
//!
//! Run with:
//! ```sh
//! cargo run --example ocr --features provider-onnx
//! ```

#[cfg(feature = "provider-onnx")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-onnx")]
use uni_xervo::provider::LocalOnnxProvider;
#[cfg(feature = "provider-onnx")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-onnx")]
use uni_xervo::traits::ImageInput;

#[cfg(feature = "provider-onnx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the OCR model. Dropping `det_onnx_path` switches to
    //    single-stage recognition for images that are already cropped to a line.
    let spec = ModelAliasSpec {
        alias: "ocr/ppocr-en".to_string(),
        task: ModelTask::Ocr,
        provider_id: "local/onnx".to_string(),
        model_id: "monkt/paddleocr-onnx".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            // Recognition (English CRNN, height 48, SigLIP-style normalization).
            "onnx_path": "languages/english/rec.onnx",
            "char_dict_path": "languages/english/dict.txt",
            "image_height": 48,
            "image_width": 320,
            "normalization": "siglip",
            // Detection (PP-OCRv5 DBNet, same repo) — enables detect -> recognize.
            "det_onnx_path": "detection/v5/det.onnx"
        }),
    };

    // 2. Build the runtime with the local ONNX provider.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Load the bundled sample image (replace with your own as needed).
    let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/text_lines.png");
    let input = ImageInput::Bytes {
        data: std::fs::read(image_path)?,
        media_type: "image/png".to_string(),
    };

    // 4. Recognize. One `OcrResult` is returned per input image.
    let ocr = runtime.ocr_model("ocr/ppocr-en").await?;
    let results = ocr.recognize(vec![input]).await?;
    let result = &results[0];

    println!("recognized text:\n{}\n", result.plain_text);
    println!("{} block(s):", result.blocks.len());
    for block in &result.blocks {
        println!(
            "  bbox={:?} conf={:.3} text={:?}",
            block.bbox, block.confidence, block.text
        );
    }

    Ok(())
}

#[cfg(not(feature = "provider-onnx"))]
fn main() {
    eprintln!(
        "This example requires the `provider-onnx` feature.\n\
         Run with: cargo run --example ocr --features provider-onnx"
    );
}
