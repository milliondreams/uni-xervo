//! Per-page routing signals and their cheap derivation.
//!
//! Signals are the inputs the [`router`](crate::router) uses to decide a page's
//! tier. They are deliberately cheap: most come straight from the native text
//! parse (no model work), which is why `Native -> Ocr` routing is near-free.
//! The `looks_structured` hint is the one signal that may require a layout
//! cue (OCR box geometry or, later, a layout classifier).

use crate::input::NativeText;
use crate::result::EscalationReason;

/// Coverage below which a page *with* a text layer is treated as likely scanned.
///
/// A born-digital text page covers a meaningful fraction of its area with glyph
/// boxes; a scanned page with only a stray digital watermark covers almost
/// none. `0.05` (5%) is a conservative default — high enough to catch
/// image-dominated pages, low enough not to misfire on sparse but real text.
/// Tune per corpus; lowering it makes the pipeline trust thin text layers more.
pub const LOW_TEXT_COVERAGE: f32 = 0.05;

/// Cheap, per-page signals that drive tier escalation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageSignals {
    /// Whether the page has any usable digital text.
    pub has_text_layer: bool,
    /// Fraction of page area covered by extractable text, in `[0.0, 1.0]`.
    pub text_coverage: f32,
    /// Whether CID-keyed fonts without `ToUnicode` were detected (mojibake).
    pub garbled: bool,
    /// Whether the page looks structurally complex (table / figure / columns).
    pub looks_structured: bool,
}

impl PageSignals {
    /// Derive the cheap signals from a parsed text layer.
    ///
    /// `looks_structured` is left `false`; it is set later from a layout cue
    /// via [`with_structured`](Self::with_structured) once one is available.
    #[must_use]
    pub fn from_native(native: &NativeText) -> Self {
        Self {
            has_text_layer: !native.text.trim().is_empty(),
            text_coverage: native.coverage,
            garbled: native.garbled,
            looks_structured: false,
        }
    }

    /// Return a copy with the `looks_structured` hint set.
    #[must_use]
    pub fn with_structured(mut self, structured: bool) -> Self {
        self.looks_structured = structured;
        self
    }

    /// Whether the native text layer is insufficient and OCR is needed.
    ///
    /// True when there is no text layer, the text is garbled, or coverage is
    /// below [`LOW_TEXT_COVERAGE`].
    #[must_use]
    pub fn needs_ocr(self) -> bool {
        !self.has_text_layer || self.garbled || self.text_coverage < LOW_TEXT_COVERAGE
    }

    /// The reason this page needs OCR, assuming [`needs_ocr`](Self::needs_ocr).
    ///
    /// Returns the most specific applicable reason; the order mirrors how a
    /// reader would diagnose the page (no text, then garbled, then sparse).
    #[must_use]
    pub fn ocr_reason(self) -> EscalationReason {
        if !self.has_text_layer {
            EscalationReason::NoTextLayer
        } else if self.garbled {
            EscalationReason::GarbledText
        } else {
            EscalationReason::LowTextCoverage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native(text: &str, coverage: f32, garbled: bool) -> NativeText {
        NativeText {
            text: text.to_string(),
            coverage,
            garbled,
        }
    }

    #[test]
    fn born_digital_page_does_not_need_ocr() {
        let s = PageSignals::from_native(&native("full page of text", 0.5, false));
        assert!(s.has_text_layer);
        assert!(!s.needs_ocr());
    }

    #[test]
    fn missing_text_layer_needs_ocr_with_reason() {
        let s = PageSignals::from_native(&native("", 0.0, false));
        assert!(s.needs_ocr());
        assert_eq!(s.ocr_reason(), EscalationReason::NoTextLayer);
    }

    #[test]
    fn low_coverage_needs_ocr_with_reason() {
        let s = PageSignals::from_native(&native("x", LOW_TEXT_COVERAGE - 0.01, false));
        assert!(s.needs_ocr());
        assert_eq!(s.ocr_reason(), EscalationReason::LowTextCoverage);
    }

    #[test]
    fn garbled_text_needs_ocr_with_reason() {
        let s = PageSignals::from_native(&native("text", 0.5, true));
        assert!(s.needs_ocr());
        assert_eq!(s.ocr_reason(), EscalationReason::GarbledText);
    }
}
