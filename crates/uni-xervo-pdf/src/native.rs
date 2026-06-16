//! Native text seam — the [`Native`](crate::Tier::Native) tier and its signals.
//!
//! Native extraction reads the PDF's digital text layer directly (no image). It
//! serves a triple duty: it *is* the cheapest tier, it produces the cheap
//! routing signals ([`PageSignals`](crate::PageSignals)) that decide
//! `Native -> Ocr`, and it provides the deterministic reference text used to
//! corroborate the generative tier.
//!
//! The pure signal heuristics ([`text_coverage_proxy`], [`garble_ratio`]) are
//! always available; the concrete pure-Rust parser
//! ([`LopdfTextExtractor`](struct@LopdfTextExtractor)) lives behind the
//! `pdf-input` feature.

use crate::input::{NativeText, PageRange};
use uni_xervo::error::Result;

/// Non-whitespace character count treated as a "full" page for the coverage proxy.
///
/// True text-area coverage needs glyph geometry the text layer does not cheaply
/// expose, so the native tier approximates it with character density: a dense
/// page runs ~1800 non-whitespace characters. The proxy serves only routing
/// (is this page text-bearing or likely scanned?), so an approximation is fine;
/// lowering it makes sparse pages read as "well covered".
const EXPECTED_FULL_PAGE_CHARS: f32 = 1800.0;

/// Garble ratio above which a non-empty page is treated as mojibake.
///
/// Catches CID-keyed fonts without `ToUnicode`, whose extracted text is mostly
/// replacement/control characters. `0.2` (20%) tolerates the occasional stray
/// symbol while flagging genuinely unreadable output.
const GARBLE_THRESHOLD: f32 = 0.2;

/// A character-density proxy for text-area coverage, in `[0.0, 1.0]`.
///
/// Returns the count of non-whitespace characters relative to a
/// "full page" density (~1800 non-whitespace chars), capped at `1.0`.
#[must_use]
pub fn text_coverage_proxy(text: &str) -> f32 {
    let non_ws = text.chars().filter(|c| !c.is_whitespace()).count() as f32;
    (non_ws / EXPECTED_FULL_PAGE_CHARS).min(1.0)
}

/// Fraction of non-whitespace characters that look like mojibake, in `[0.0, 1.0]`.
///
/// Counts the Unicode replacement character and non-whitespace control
/// characters. Returns `0.0` for text with no readable characters.
#[must_use]
pub fn garble_ratio(text: &str) -> f32 {
    let mut total = 0usize;
    let mut bad = 0usize;
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        total += 1;
        if c == '\u{FFFD}' || c.is_control() {
            bad += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        bad as f32 / total as f32
    }
}

/// Assemble a [`NativeText`] (text + routing signals) from a page's raw text.
#[must_use]
pub fn native_text_from(text: String) -> NativeText {
    let coverage = text_coverage_proxy(&text);
    let garbled = !text.trim().is_empty() && garble_ratio(&text) > GARBLE_THRESHOLD;
    NativeText {
        text,
        coverage,
        garbled,
    }
}

/// One page's parsed text layer produced by a [`NativeExtractor`].
#[derive(Debug, Clone)]
pub struct NativePage {
    /// The page's 1-based number.
    pub page_number: u32,
    /// The parsed text and its derived signals.
    pub native: NativeText,
}

/// Parses a PDF's digital text layer into per-page text plus routing signals.
pub trait NativeExtractor: Send + Sync {
    /// Extract the text layer of `pdf`, optionally limited to a `pages` range.
    ///
    /// Returns one [`NativePage`] per processed page, in page order.
    ///
    /// # Errors
    /// Returns an error if the PDF cannot be parsed.
    fn extract(&self, pdf: &[u8], pages: Option<PageRange>) -> Result<Vec<NativePage>>;
}

/// Pure-Rust native text extractor backed by `lopdf`.
///
/// Enumerates pages and reads each page's text layer. A page whose text cannot
/// be extracted yields empty text (which routes it to OCR), rather than failing
/// the whole document.
#[cfg(feature = "pdf-input")]
#[derive(Debug, Clone, Default)]
pub struct LopdfTextExtractor;

#[cfg(feature = "pdf-input")]
impl LopdfTextExtractor {
    /// Create a new extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "pdf-input")]
impl NativeExtractor for LopdfTextExtractor {
    fn extract(&self, pdf: &[u8], pages: Option<PageRange>) -> Result<Vec<NativePage>> {
        use uni_xervo::error::RuntimeError;

        let doc = lopdf::Document::load_mem(pdf)
            .map_err(|e| RuntimeError::Config(format!("failed to parse PDF: {e}")))?;

        let mut out = Vec::new();
        for &page_number in doc.get_pages().keys() {
            if let Some(range) = pages
                && !range.contains(page_number)
            {
                continue;
            }
            // A page whose text cannot be read becomes empty -> routes to OCR.
            let text = doc.extract_text(&[page_number]).unwrap_or_default();
            out.push(NativePage {
                page_number,
                native: native_text_from(text),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_proxy_bounds() {
        assert_eq!(text_coverage_proxy(""), 0.0);
        assert!(text_coverage_proxy("a few words here") > 0.0);
        let dense: String = "x".repeat(5000);
        assert_eq!(text_coverage_proxy(&dense), 1.0);
    }

    #[test]
    fn garble_ratio_clean_vs_mojibake() {
        assert_eq!(garble_ratio("clean readable text"), 0.0);
        let mojibake = "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}a";
        assert!(garble_ratio(mojibake) > 0.5);
    }

    #[test]
    fn native_text_flags_garbled_only_when_nonempty() {
        let clean = native_text_from("hello world".to_string());
        assert!(!clean.garbled);
        assert!(clean.is_usable());

        let garbled = native_text_from("\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}".to_string());
        assert!(garbled.garbled);
        assert!(!garbled.is_usable());

        // Empty text is not "garbled" (it is simply absent -> scanned page).
        let empty = native_text_from(String::new());
        assert!(!empty.garbled);
        assert!(!empty.is_usable());
    }

    #[cfg(feature = "pdf-input")]
    mod lopdf_backend {
        use super::*;
        use crate::testutil::make_text_pdf;

        #[test]
        fn extracts_text_layer_from_real_pdf() {
            let pdf = make_text_pdf("Hello World");
            let pages = LopdfTextExtractor::new().extract(&pdf, None).unwrap();
            assert_eq!(pages.len(), 1);
            assert_eq!(pages[0].page_number, 1);
            assert!(
                pages[0].native.text.contains("Hello"),
                "extracted text was {:?}",
                pages[0].native.text
            );
            assert!(pages[0].native.is_usable());
        }

        #[test]
        fn rejects_malformed_pdf() {
            let err = LopdfTextExtractor::new().extract(b"not a pdf at all", None);
            assert!(err.is_err());
        }
    }
}
