//! Document-extraction example: structured Markdown from a page image.
//!
//! Runs olmOCR-2 (a Qwen2.5-VL fine-tune) on the mistral.rs vision pipeline and
//! prints the page's Markdown. Unlike OCR — which only transcribes text — a
//! document VLM recovers tables, formulas, headings, and reading order.
//!
//! Hardware: `allenai/olmOCR-2-7B-1025` is a 7B-parameter model (multi-GB
//! download). A GPU is strongly recommended; CPU inference with ISQ `Q4K`
//! quantization works but is slow. Because it is generative, the output can
//! hallucinate — corroborate against a deterministic OCR tier for reliability
//! (see the tiered PDF extraction guide).
//!
//! Run with:
//! ```sh
//! cargo run --example document_extract --features provider-mistralrs
//! ```

#[cfg(feature = "provider-mistralrs")]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::provider::mistralrs::LocalMistralRsProvider;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::runtime::ModelRuntime;
#[cfg(feature = "provider-mistralrs")]
use uni_xervo::traits::{DocExtractOptions, DocOutputFormat, ImageInput};

#[cfg(feature = "provider-mistralrs")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Describe the document extractor. Document extraction always runs on the
    //    vision pipeline; `style` selects the output parser (olmocr by default).
    let spec = ModelAliasSpec {
        alias: "docext/olmocr".to_string(),
        task: ModelTask::DocumentExtract,
        provider_id: "local/mistralrs".to_string(),
        model_id: "allenai/olmOCR-2-7B-1025".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: true,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "isq": "Q4K",
            "style": "olmocr"
        }),
    };

    // 2. Build the runtime with the local mistral.rs provider.
    let runtime = ModelRuntime::builder()
        .register_provider(LocalMistralRsProvider::new())
        .catalog(vec![spec])
        .build()
        .await?;

    // 3. Load the bundled sample page (replace with your own as needed).
    let page_path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/text_lines.png");
    let page = ImageInput::Bytes {
        data: std::fs::read(page_path)?,
        media_type: "image/png".to_string(),
    };

    // 4. Extract. `DocExtractOptions` has no `Default` — set all fields. One page
    //    image per request (olmOCR-2 is single-image by design).
    let options = DocExtractOptions {
        output: DocOutputFormat::Markdown,
        include_tables: true,
        include_formulas: true,
        include_bboxes: false,
    };
    let extractor = runtime.document_extractor("docext/olmocr").await?;
    let pages = extractor.extract(vec![page], options).await?;

    println!("extracted Markdown:\n{}", pages[0].plain_markdown);
    println!("\n{} block(s) in reading order", pages[0].blocks.len());

    Ok(())
}

#[cfg(not(feature = "provider-mistralrs"))]
fn main() {
    eprintln!(
        "This example requires the `provider-mistralrs` feature.\n\
         Run with: cargo run --example document_extract --features provider-mistralrs"
    );
}
