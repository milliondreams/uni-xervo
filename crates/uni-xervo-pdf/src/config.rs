//! Extraction policy and pipeline configuration.
//!
//! [`PdfConfig`] binds the pipeline's tiers to runtime aliases and picks a
//! rasterizer backend; [`DocExtractPolicy`] is the per-call reliability/cost
//! knob (how far to escalate, whether structure is wanted, budget, and
//! cross-tier verification). Both are plain data with public fields, matching
//! the style of `uni_xervo`'s own `GenerationOptions` / `DocExtractOptions`.

use crate::tier::Tier;
use std::time::Duration;

/// Default alias the [`Ocr`](Tier::Ocr) rung resolves to in [`PdfConfig::auto`].
///
/// Callers must register this alias on the runtime (an
/// [`OcrModel`](uni_xervo::traits::OcrModel)), or override
/// [`PdfConfig::ocr_alias`].
pub const DEFAULT_OCR_ALIAS: &str = "ocr/default";

/// Default alias the [`Vlm`](Tier::Vlm) rung resolves to in [`PdfConfig::auto`].
///
/// Callers must register this alias on the runtime (a
/// [`DocumentExtractionModel`](uni_xervo::traits::DocumentExtractionModel)), or
/// override [`PdfConfig::vlm_alias`].
pub const DEFAULT_VLM_ALIAS: &str = "docext/default";

/// What the caller needs out of extraction — drives `Ocr -> Vlm` escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OutputWant {
    /// Plain reading text. A page never escalates to [`Vlm`](Tier::Vlm) just
    /// for layout when only text is wanted.
    #[default]
    Text,
    /// Structured layout — tables, formulas, reading order. Table/figure pages
    /// may escalate to [`Vlm`](Tier::Vlm).
    Structure,
}

/// How far the router may climb the ladder, per page.
///
/// The reliability knob: a memory-critical caller sets `Ceiling(Tier::Ocr)` to
/// forbid generative extraction entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelPolicy {
    /// Always use exactly this tier; the router never escalates.
    Fixed(Tier),
    /// Escalate as needed, but never above this tier.
    Ceiling(Tier),
    /// Escalate within `[min, max]` per page.
    Auto {
        /// The lowest tier to attempt (escalation starts here).
        min: Tier,
        /// The highest tier the router may climb to.
        max: Tier,
    },
}

impl LevelPolicy {
    /// The lowest tier this policy permits.
    #[must_use]
    pub fn min_tier(self) -> Tier {
        match self {
            Self::Fixed(t) => t,
            // A ceiling caps the top but may start escalating from Native.
            Self::Ceiling(_) => Tier::Native,
            Self::Auto { min, .. } => min,
        }
    }

    /// The highest tier this policy permits.
    #[must_use]
    pub fn max_tier(self) -> Tier {
        match self {
            Self::Fixed(t) | Self::Ceiling(t) => t,
            Self::Auto { max, .. } => max,
        }
    }

    /// Whether `tier` is allowed under this policy.
    #[must_use]
    pub fn allows(self, tier: Tier) -> bool {
        match self {
            Self::Fixed(t) => tier == t,
            Self::Ceiling(max) => tier <= max,
            Self::Auto { min, max } => tier >= min && tier <= max,
        }
    }

    /// Clamp `tier` into the range this policy permits.
    ///
    /// Used by the hardware-adaptive default: a discovered ceiling (e.g. no
    /// GPU) narrows the effective `max` without erroring.
    #[must_use]
    pub fn clamp(self, tier: Tier) -> Tier {
        tier.min(self.max_tier()).max(self.min_tier())
    }
}

/// Cross-tier corroboration settings for generative output.
///
/// When a deterministic tier ran for the same page, the generative tier's
/// output can be diffed against it — a stronger anti-hallucination signal than
/// any confidence scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyPolicy {
    /// Diff generative output against the deterministic tier, when available.
    pub cross_tier: bool,
    /// Flag/quarantine numeric tokens that diverge from the deterministic tier.
    pub numeric_guard: bool,
}

impl Default for VerifyPolicy {
    /// Both guards on — reliability-first by default.
    fn default() -> Self {
        Self {
            cross_tier: true,
            numeric_guard: true,
        }
    }
}

/// A spend ceiling enforced mid-document, with graceful degradation.
///
/// When a limit is reached the router stops escalating and returns the
/// best-so-far result with an [`Escalation`](crate::Escalation) note — it does
/// not error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Budget {
    /// Maximum number of pages allowed to reach the [`Vlm`](Tier::Vlm) tier.
    pub max_vlm_pages: Option<u32>,
    /// Soft wall-clock ceiling for the whole call.
    pub max_latency: Option<Duration>,
}

/// Per-call reliability and cost policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DocExtractPolicy {
    /// How far the router may climb.
    pub level: LevelPolicy,
    /// What the caller needs (gates `Ocr -> Vlm`).
    pub want: OutputWant,
    /// Optional spend ceiling.
    pub budget: Option<Budget>,
    /// Cross-tier corroboration settings.
    pub verify: VerifyPolicy,
}

impl DocExtractPolicy {
    /// Auto-escalate from [`Native`](Tier::Native) up to `max`, text output.
    ///
    /// # Examples
    /// ```
    /// use uni_xervo_pdf::{DocExtractPolicy, Tier};
    /// let p = DocExtractPolicy::auto_up_to(Tier::Vlm);
    /// assert!(p.level.allows(Tier::Ocr));
    /// ```
    #[must_use]
    pub fn auto_up_to(max: Tier) -> Self {
        Self {
            level: LevelPolicy::Auto {
                min: Tier::Native,
                max,
            },
            want: OutputWant::Text,
            budget: None,
            verify: VerifyPolicy::default(),
        }
    }

    /// Pin extraction to exactly `tier` (no escalation).
    #[must_use]
    pub fn fixed(tier: Tier) -> Self {
        Self {
            level: LevelPolicy::Fixed(tier),
            ..Self::auto_up_to(tier)
        }
    }

    /// Auto-escalate but never above `ceiling` (forbids higher tiers).
    #[must_use]
    pub fn ceiling(ceiling: Tier) -> Self {
        Self {
            level: LevelPolicy::Ceiling(ceiling),
            ..Self::auto_up_to(ceiling)
        }
    }
}

impl Default for DocExtractPolicy {
    /// Auto from Native to Vlm, text output, verification on.
    fn default() -> Self {
        Self::auto_up_to(Tier::Vlm)
    }
}

/// Configuration binding the pipeline to runtime aliases.
///
/// Construct with [`PdfConfig::auto`] for batteries-included defaults, then
/// adjust fields as needed. The `ocr_alias` / `vlm_alias` name
/// [`OcrModel`](uni_xervo::traits::OcrModel) /
/// [`DocumentExtractionModel`](uni_xervo::traits::DocumentExtractionModel)
/// entries the caller has registered on the runtime. Page rasterization (when
/// `DocInput::Pdf` is used) is done by the pure-Rust `hayro` backend.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfConfig {
    /// Runtime alias for the [`Ocr`](Tier::Ocr) rung, if available.
    pub ocr_alias: Option<String>,
    /// Runtime alias for the [`Vlm`](Tier::Vlm) rung, if available.
    pub vlm_alias: Option<String>,
    /// Default policy applied when a call does not specify one.
    pub defaults: DocExtractPolicy,
}

impl PdfConfig {
    /// Batteries-included config: conventional aliases, auto policy.
    ///
    /// Uses [`DEFAULT_OCR_ALIAS`] and [`DEFAULT_VLM_ALIAS`]; override the
    /// `*_alias` fields to point at your own catalog entries.
    #[must_use]
    pub fn auto() -> Self {
        Self {
            ocr_alias: Some(DEFAULT_OCR_ALIAS.to_string()),
            vlm_alias: Some(DEFAULT_VLM_ALIAS.to_string()),
            defaults: DocExtractPolicy::default(),
        }
    }
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self::auto()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceiling_forbids_higher_tiers() {
        let p = LevelPolicy::Ceiling(Tier::Ocr);
        assert!(p.allows(Tier::Native));
        assert!(p.allows(Tier::Ocr));
        assert!(!p.allows(Tier::Vlm));
    }

    #[test]
    fn fixed_allows_only_itself() {
        let p = LevelPolicy::Fixed(Tier::Ocr);
        assert!(!p.allows(Tier::Native));
        assert!(p.allows(Tier::Ocr));
        assert!(!p.allows(Tier::Vlm));
    }

    #[test]
    fn auto_respects_bounds() {
        let p = LevelPolicy::Auto {
            min: Tier::Ocr,
            max: Tier::Vlm,
        };
        assert!(!p.allows(Tier::Native));
        assert!(p.allows(Tier::Ocr));
        assert!(p.allows(Tier::Vlm));
        assert_eq!(p.min_tier(), Tier::Ocr);
        assert_eq!(p.max_tier(), Tier::Vlm);
    }

    #[test]
    fn clamp_narrows_to_discovered_ceiling() {
        // Hardware-adaptive default: requested Vlm but clamp to an Ocr ceiling.
        let p = LevelPolicy::Ceiling(Tier::Ocr);
        assert_eq!(p.clamp(Tier::Vlm), Tier::Ocr);
        assert_eq!(p.clamp(Tier::Native), Tier::Native);
    }

    #[test]
    fn auto_config_has_conventional_aliases() {
        let c = PdfConfig::auto();
        assert_eq!(c.ocr_alias.as_deref(), Some(DEFAULT_OCR_ALIAS));
        assert_eq!(c.vlm_alias.as_deref(), Some(DEFAULT_VLM_ALIAS));
    }
}
