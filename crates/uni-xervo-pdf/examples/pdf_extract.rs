//! Tiered PDF extraction example: drive `uni-xervo-pdf` end to end.
//!
//! Builds a `uni-xervo` runtime with an OCR rung, wraps it with the
//! [`PdfExt`](uni_xervo_pdf::PdfExt) extension trait, and extracts a small
//! generated PDF. The pipeline escalates per page only as far as needed: a
//! born-digital page (like the one generated here) is served by the free
//! [`Native`](uni_xervo_pdf::Tier::Native) tier without ever invoking OCR;
//! scanned pages would escalate to the [`Ocr`](uni_xervo_pdf::Tier::Ocr) rung.
//!
//! The OCR rung (`monkt/paddleocr-onnx`, ~96 MB) is resolved when the extractor
//! is built and downloaded on first run. If it cannot be loaded (e.g. offline),
//! the pipeline degrades gracefully to the highest available tier.
//!
//! Run with:
//! ```sh
//! cargo run -p uni-xervo-pdf --example pdf_extract
//! ```

#[cfg(all(feature = "pdf-input", feature = "hayro"))]
use uni_xervo::api::{ModelAliasSpec, ModelTask, WarmupPolicy};
#[cfg(all(feature = "pdf-input", feature = "hayro"))]
use uni_xervo::provider::LocalOnnxProvider;
#[cfg(all(feature = "pdf-input", feature = "hayro"))]
use uni_xervo::runtime::ModelRuntime;
#[cfg(all(feature = "pdf-input", feature = "hayro"))]
use uni_xervo_pdf::{DocExtractPolicy, DocInput, PdfConfig, PdfExt, Tier};

#[cfg(all(feature = "pdf-input", feature = "hayro"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Register an OCR rung the pipeline can escalate to for scanned pages.
    let ocr_spec = ModelAliasSpec {
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
    };

    let runtime = ModelRuntime::builder()
        .register_provider(LocalOnnxProvider::new())
        .catalog(vec![ocr_spec])
        .build()
        .await?;

    // 2. Point the pipeline's OCR rung at the alias we registered. There is no
    //    VLM rung here, so cap escalation at the OCR tier.
    let config = PdfConfig {
        ocr_alias: Some("ocr/ppocr-en".to_string()),
        vlm_alias: None,
        ..PdfConfig::auto()
    };
    let extractor = runtime.pdf_extractor(config).await?;

    // 3. Extract a generated single-page PDF, escalating up to the OCR tier.
    let pdf = make_text_pdf("Hello World OCR test 2026");
    let pages = extractor
        .extract(
            DocInput::Pdf {
                bytes: pdf,
                pages: None,
            },
            DocExtractPolicy::auto_up_to(Tier::Ocr),
        )
        .await?;

    for page in &pages {
        println!(
            "page {} produced by {:?}:\n{}\n",
            page.page_number, page.produced_by, page.plain_markdown
        );
    }

    Ok(())
}

/// Build a minimal single-page PDF whose text layer contains `text`.
///
/// Mirrors `lopdf`'s hello-world example so the real parse/rasterize paths are
/// exercised. (The crate's own `testutil::make_text_pdf` is `#[cfg(test)]
/// pub(crate)`, so it is not reachable from an example.)
#[cfg(all(feature = "pdf-input", feature = "hayro"))]
fn make_text_pdf(text: &str) -> Vec<u8> {
    use lopdf::content::{Content, Operation};
    use lopdf::{Document, Object, Stream, dictionary};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font_id },
    });
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![100.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal(text)]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    let mut buf = Vec::new();
    doc.save_to(&mut buf).unwrap();
    buf
}

#[cfg(not(all(feature = "pdf-input", feature = "hayro")))]
fn main() {
    eprintln!(
        "This example requires the `pdf-input` and `hayro` features (both on by default).\n\
         Run with: cargo run -p uni-xervo-pdf --example pdf_extract"
    );
}
