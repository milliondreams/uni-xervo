//! Pipeline inputs — the per-call choice between owning or supplying pixels.
//!
//! [`DocInput`] resolves the "who rasterizes?" question per call rather than
//! as an architecture-wide commitment: pass [`Pdf`](DocInput::Pdf) to have the
//! pipeline parse + rasterize (needs the `pdf-input` + a rasterizer feature),
//! or [`Pages`](DocInput::Pages) to supply already-rendered pixels (and,
//! optionally, a pre-parsed text layer). This mirrors `uni_xervo`'s own
//! [`ImageInput::Bytes`](uni_xervo::traits::ImageInput) / `Url` enum.

use uni_xervo::traits::ImageInput;

/// A 1-based, inclusive range of pages.
///
/// `PageRange { start: 1, end: 3 }` selects pages 1, 2, and 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRange {
    /// First page to process (1-based, inclusive).
    pub start: u32,
    /// Last page to process (1-based, inclusive).
    pub end: u32,
}

impl PageRange {
    /// Create a 1-based inclusive range.
    ///
    /// # Panics
    /// Panics if `start` is `0` or `end < start` — a caller programming error.
    #[must_use]
    pub fn new(start: u32, end: u32) -> Self {
        assert!(start >= 1, "page numbers are 1-based; start must be >= 1");
        assert!(end >= start, "page range end must be >= start");
        Self { start, end }
    }

    /// Whether `page` (1-based) falls within this range.
    #[must_use]
    pub fn contains(self, page: u32) -> bool {
        page >= self.start && page <= self.end
    }
}

/// A page's digital text layer plus the cheap signals derived while parsing it.
///
/// Supplied by the native parser on the [`Pdf`](DocInput::Pdf) path, or by the
/// caller on the [`Pages`](DocInput::Pages) path. The signals drive
/// `Native -> Ocr` routing without any model work.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeText {
    /// The extracted digital text for the page (may be empty for scans).
    pub text: String,
    /// Fraction of page area covered by extractable text, in `[0.0, 1.0]`.
    ///
    /// Low coverage on a non-empty page suggests a scanned/image region.
    pub coverage: f32,
    /// Whether CID-keyed fonts without `ToUnicode` were seen (mojibake risk).
    pub garbled: bool,
}

impl NativeText {
    /// Whether this page has usable digital text.
    ///
    /// `false` when empty or flagged garbled — the trigger to escalate to OCR.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        !self.garbled && !self.text.trim().is_empty()
    }
}

/// One caller-supplied page on the [`Pages`](DocInput::Pages) path.
#[derive(Debug, Clone)]
pub struct PageInput {
    /// The rasterized page image (for the OCR / VLM tiers).
    pub image: ImageInput,
    /// The page's text layer, if the caller already parsed it.
    pub native_text: Option<NativeText>,
    /// The page's 1-based number, for provenance and ordering.
    pub page_number: u32,
}

/// Input to a tiered extraction call.
#[derive(Debug, Clone)]
pub enum DocInput {
    /// Raw PDF bytes; the pipeline parses + rasterizes (Option A).
    ///
    /// Requires the `pdf-input` feature and a rasterizer feature. `pages`
    /// optionally restricts processing to a sub-range.
    Pdf {
        /// The PDF file contents.
        bytes: Vec<u8>,
        /// Optional 1-based page sub-range; `None` processes all pages.
        pages: Option<PageRange>,
    },
    /// Caller-supplied page images (+ optional text layers) (Option B).
    ///
    /// No rasterizer is needed; the caller owns page rendering.
    Pages(Vec<PageInput>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_range_contains_is_inclusive() {
        let r = PageRange::new(2, 4);
        assert!(!r.contains(1));
        assert!(r.contains(2));
        assert!(r.contains(4));
        assert!(!r.contains(5));
    }

    #[test]
    #[should_panic(expected = "1-based")]
    fn page_range_rejects_zero_start() {
        let _ = PageRange::new(0, 3);
    }

    #[test]
    fn native_text_usability() {
        let usable = NativeText {
            text: "hello".into(),
            coverage: 0.5,
            garbled: false,
        };
        assert!(usable.is_usable());

        let empty = NativeText {
            text: "   ".into(),
            coverage: 0.0,
            garbled: false,
        };
        assert!(!empty.is_usable());

        let garbled = NativeText {
            text: "hello".into(),
            coverage: 0.5,
            garbled: true,
        };
        assert!(!garbled.is_usable());
    }
}
