//! Provenance-bearing results — every block traceable to its producer.
//!
//! Downstream consumers (e.g. a knowledge graph) gate trust on provenance, so
//! each [`TieredBlock`] carries which [`Tier`] produced it and a
//! [`Confidence`] that is *honest about its source*: deterministic tiers report
//! [`Deterministic`](Confidence::Deterministic), the OCR tier a real
//! [`Measured`](Confidence::Measured) probability, and the generative tier a
//! [`Derived`](Confidence::Derived) heuristic (the VLM emits no native
//! confidence). Per-page [`Escalation`] records explain why a page cost more.

use crate::tier::Tier;
use uni_xervo::traits::DocBlockKind;

/// How a block's trust score was obtained — the source matters, not just value.
#[derive(Debug, Clone, PartialEq)]
pub enum Confidence {
    /// Deterministic extraction; the text was read, not generated (`~1.0`).
    Deterministic,
    /// A real model probability, e.g. CTC softmax from OCR recognition.
    Measured(f32),
    /// A heuristic for generative output, which carries no native confidence.
    Derived {
        /// Aggregate score in `[0.0, 1.0]`; higher is more trustworthy.
        score: f32,
        /// The signals the score was derived from.
        signals: DerivedSignals,
    },
}

impl Confidence {
    /// A comparable scalar in `[0.0, 1.0]` for simple trust gating.
    ///
    /// [`Deterministic`](Confidence::Deterministic) maps to `1.0`; the others
    /// return their underlying score. Prefer matching the variant when the
    /// *source* of trust matters.
    #[must_use]
    pub fn score(&self) -> f32 {
        match self {
            Self::Deterministic => 1.0,
            Self::Measured(s) => *s,
            Self::Derived { score, .. } => *score,
        }
    }
}

/// Heuristic signals behind a [`Derived`](Confidence::Derived) confidence.
///
/// These approximate trust for generative output: whether decoding finished
/// cleanly, how repetitive it was, and how well it agrees with a lower tier.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedSignals {
    /// Whether generation stopped on EOS (vs. truncation / token overflow).
    pub finished_cleanly: bool,
    /// Repeated n-gram fraction in `[0.0, 1.0]`; higher means more looping.
    pub repetition_ratio: f32,
    /// Output length relative to the lower-tier text, if one ran.
    pub length_ratio: Option<f32>,
    /// Disagreement with the deterministic tier in `[0.0, 1.0]`, if compared.
    pub cross_tier_divergence: Option<f32>,
}

/// Why the router climbed (or was capped) for a page — an audit record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationReason {
    /// The page had no digital text layer.
    NoTextLayer,
    /// Digital text coverage was below the threshold (likely scanned).
    LowTextCoverage,
    /// Garbled text (CID fonts without `ToUnicode`) was detected.
    GarbledText,
    /// Structured output was requested for a table/figure/multi-column page.
    StructureRequested,
    /// A lower tier returned empty output.
    EmptyOutput,
    /// OCR returned flat text where layout structure was expected.
    FlatWhereStructureExpected,
    /// Escalation was stopped because the budget was exhausted.
    BudgetExhausted,
    /// Escalation was capped by available hardware (e.g. no GPU for the VLM).
    HardwareCapped,
}

/// One routing decision for a page.
///
/// `from == to` denotes a *cap* (the router wanted to climb but could not),
/// e.g. [`BudgetExhausted`](EscalationReason::BudgetExhausted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Escalation {
    /// Tier the router was at.
    pub from: Tier,
    /// Tier the router moved to (equal to `from` for a cap).
    pub to: Tier,
    /// Why the decision was made.
    pub reason: EscalationReason,
}

/// One block of extracted content, tagged with full provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct TieredBlock {
    /// Kind of block (reuses `uni_xervo`'s taxonomy).
    pub kind: DocBlockKind,
    /// Block content (Markdown text, LaTeX formula, HTML table, etc.).
    pub content: String,
    /// Bounding box in page coordinates `[x0, y0, x1, y1]`, when available.
    pub bbox: Option<[f32; 4]>,
    /// Monotonic reading-order index within the page.
    pub reading_order: u32,
    /// The tier that produced this block.
    pub produced_by: Tier,
    /// Trust score, honest about its source.
    pub confidence: Confidence,
}

/// The result for one page.
#[derive(Debug, Clone, PartialEq)]
pub struct TieredPageResult {
    /// The page's 1-based number.
    pub page_number: u32,
    /// Extracted blocks in reading order.
    pub blocks: Vec<TieredBlock>,
    /// Concatenated Markdown view (always populated).
    pub plain_markdown: String,
    /// The tier that produced this page's result.
    pub produced_by: Tier,
    /// Audit trail of routing decisions for this page.
    pub escalations: Vec<Escalation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_score_maps_each_variant() {
        assert!((Confidence::Deterministic.score() - 1.0).abs() < 1e-6);
        assert!((Confidence::Measured(0.7).score() - 0.7).abs() < 1e-6);

        let derived = Confidence::Derived {
            score: 0.4,
            signals: DerivedSignals {
                finished_cleanly: true,
                repetition_ratio: 0.1,
                length_ratio: None,
                cross_tier_divergence: None,
            },
        };
        assert!((derived.score() - 0.4).abs() < 1e-6);
    }
}
