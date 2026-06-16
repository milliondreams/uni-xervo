//! The `PdfExt` extension trait — a family-native accessor on `ModelRuntime`.
//!
//! This is how an *optional, separate* crate joins the `embed` / `generate` /
//! `nlp` family: an extension trait adds `runtime.pdf_extractor(..)`, which
//! reads exactly like the runtime's built-in accessors. Bring it into scope
//! with `use uni_xervo_pdf::PdfExt;`.
//!
//! Resolution is graceful: if a configured tier alias cannot be loaded (e.g.
//! no GPU for the VLM), that tier is simply left unbound and the pipeline caps
//! itself there, rather than failing — the hardware-adaptive default.

use async_trait::async_trait;
use uni_xervo::error::Result;
use uni_xervo::runtime::ModelRuntime;

#[cfg(any(feature = "pdf-input", feature = "hayro"))]
use std::sync::Arc;

use crate::config::PdfConfig;
use crate::pipeline::PdfExtractor;

/// Family-native construction of a [`PdfExtractor`] from a runtime.
#[async_trait]
pub trait PdfExt {
    /// Build a tiered PDF extractor, resolving its OCR/VLM rungs by alias.
    ///
    /// Aliases named in `config` are resolved via the runtime's
    /// [`ocr_model`](ModelRuntime::ocr_model) /
    /// [`document_extractor`](ModelRuntime::document_extractor) accessors, so the
    /// resulting sub-models are cached and instrumented like any other. A tier
    /// whose alias fails to resolve is left unbound (the pipeline degrades to
    /// the highest available tier).
    ///
    /// # Errors
    /// Returns an error only for non-recoverable setup problems; a missing or
    /// unloadable tier alias is downgraded, not surfaced as an error.
    async fn pdf_extractor(&self, config: PdfConfig) -> Result<PdfExtractor>;
}

#[async_trait]
impl PdfExt for ModelRuntime {
    async fn pdf_extractor(&self, config: PdfConfig) -> Result<PdfExtractor> {
        let mut builder = PdfExtractor::builder(config.clone());

        if let Some(alias) = config.ocr_alias.as_deref() {
            match self.ocr_model(alias).await {
                Ok(model) => builder = builder.ocr(model),
                Err(error) => {
                    tracing::warn!(
                        name: "pdf.tier.unavailable",
                        tier = "ocr",
                        alias,
                        %error,
                        "OCR tier alias did not resolve; degrading"
                    );
                }
            }
        }

        if let Some(alias) = config.vlm_alias.as_deref() {
            match self.document_extractor(alias).await {
                Ok(model) => builder = builder.vlm(model),
                Err(error) => {
                    tracing::warn!(
                        name: "pdf.tier.unavailable",
                        tier = "vlm",
                        alias,
                        %error,
                        "VLM tier alias did not resolve; degrading"
                    );
                }
            }
        }

        // Native text extraction (the `Native` tier), pure-Rust, default-on.
        #[cfg(feature = "pdf-input")]
        {
            builder = builder.native(Arc::new(crate::native::LopdfTextExtractor::new()));
        }

        // Rasterizer for the image tiers — the pure-Rust hayro backend.
        #[cfg(feature = "hayro")]
        {
            builder = builder.rasterizer(Arc::new(crate::raster::HayroRasterizer::new()));
        }

        Ok(builder.build())
    }
}
