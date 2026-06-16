//! The extraction tier ladder — stable, ordered rungs of capability.
//!
//! A [`Tier`] is a *semantic* rung, not a concrete model. Routing policy
//! ([`LevelPolicy`](crate::LevelPolicy)) and provenance
//! ([`TieredBlock`](crate::TieredBlock)) refer to tiers, while the concrete
//! model bound to [`Ocr`](Tier::Ocr) / [`Vlm`](Tier::Vlm) is selected by alias
//! in [`PdfConfig`](crate::PdfConfig). Keeping the rungs semantic lets new
//! models slot in by re-binding an alias, with no change to this enum.
//!
//! The ordering `Native < Ocr < Vlm` is meaningful: it is cheapest-to-most-
//! capable, and the derived [`Ord`] is what [`LevelPolicy`](crate::LevelPolicy)
//! bounds compare against.

/// A rung on the extraction ladder, ordered cheapest-to-most-capable.
///
/// The two image tiers ([`Ocr`](Tier::Ocr), [`Vlm`](Tier::Vlm)) require a
/// rasterized page; [`Native`](Tier::Native) reads the PDF text layer directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Native digital-text extraction from the PDF text layer (no image).
    ///
    /// Deterministic and free; the result is the text the PDF already embeds.
    Native,
    /// Optical character recognition over a rasterized page image.
    ///
    /// Deterministic (it reads pixels); handles scanned/image-only pages.
    Ocr,
    /// Vision-language structured extraction over a rasterized page image.
    ///
    /// Generative — recovers tables/formulas/reading order, but can
    /// hallucinate, so its output is corroborated against lower tiers.
    Vlm,
}

impl Tier {
    /// Whether this tier needs a rasterized page image as input.
    ///
    /// `true` for [`Ocr`](Tier::Ocr) and [`Vlm`](Tier::Vlm); the gating
    /// rasterization step is only paid when escalating to these tiers.
    #[must_use]
    pub fn requires_raster(self) -> bool {
        matches!(self, Self::Ocr | Self::Vlm)
    }

    /// Whether this tier *generates* text (and so can hallucinate).
    ///
    /// Only [`Vlm`](Tier::Vlm) is generative; the deterministic tiers read
    /// existing text. Callers gate trust on this distinction.
    #[must_use]
    pub fn is_generative(self) -> bool {
        matches!(self, Self::Vlm)
    }
}

#[cfg(test)]
mod tests {
    use super::Tier;

    #[test]
    fn ordering_is_cheapest_to_most_capable() {
        assert!(Tier::Native < Tier::Ocr);
        assert!(Tier::Ocr < Tier::Vlm);
    }

    #[test]
    fn raster_and_generative_flags() {
        assert!(!Tier::Native.requires_raster());
        assert!(Tier::Ocr.requires_raster());
        assert!(Tier::Vlm.requires_raster());

        assert!(!Tier::Native.is_generative());
        assert!(!Tier::Ocr.is_generative());
        assert!(Tier::Vlm.is_generative());
    }
}
