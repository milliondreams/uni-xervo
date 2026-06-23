//! Cross-crate bridge test: `uni-xervo-pdf` rasterizer output is a valid image.
//!
//! Rasterization is the precondition for the image tiers; this verifies that the
//! PNG `uni-xervo-pdf` produces is decodable by the same `image` crate that
//! `uni-xervo`'s ONNX preprocessing uses to load page pixels. Runs in CI with no
//! network or model download — only `pdf-input` + `hayro` (both on by default).
//!
//! Run with:
//! ```sh
//! cargo test -p uni-xervo-pdf --test rasterize_decodes_test
//! ```

#![cfg(all(feature = "pdf-input", feature = "hayro"))]

mod common;

use common::make_text_pdf;
use uni_xervo::traits::ImageInput;
use uni_xervo_pdf::{HayroRasterizer, Rasterizer};

/// The rendered page is PNG bytes tagged with the matching media type.
#[test]
fn rasterized_pdf_page_is_valid_png() {
    let pdf = make_text_pdf("Hello World OCR test 2026");
    let pages = HayroRasterizer::new()
        .rasterize_pages(&pdf, &[1])
        .expect("rasterize page 1");
    assert_eq!(pages.len(), 1, "one rendered page for a single-page PDF");
    assert_eq!(pages[0].page_number, 1);

    let ImageInput::Bytes { data, media_type } = &pages[0].image else {
        panic!("rasterizer should produce owned bytes, not a URL");
    };
    assert_eq!(media_type, "image/png");
    // PNG magic number.
    assert_eq!(
        &data[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
    );
}

/// The rendered PNG decodes with the same `image` crate `uni-xervo` preprocessing
/// uses, to non-degenerate dimensions — proving it is a feedable page image.
#[test]
fn rasterized_pdf_page_decodes_with_image_crate() {
    let pdf = make_text_pdf("Hello World OCR test 2026");
    let pages = HayroRasterizer::new()
        .rasterize_pages(&pdf, &[1])
        .expect("rasterize page 1");

    let ImageInput::Bytes { data, .. } = &pages[0].image else {
        panic!("rasterizer should produce owned bytes, not a URL");
    };
    let decoded = image::load_from_memory(data).expect("decode rasterized PNG");
    assert!(
        decoded.width() > 0 && decoded.height() > 0,
        "decoded page has non-zero dimensions, got {}x{}",
        decoded.width(),
        decoded.height()
    );
}
