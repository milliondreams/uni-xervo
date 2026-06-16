//! Tiered PDF document extraction for the uni-xervo runtime.
//!
//! This optional companion crate turns a PDF into structured, provenance-bearing
//! text by escalating per page only as far as needed along a capability ladder:
//!
//! - [`Tier::Native`] — digital text already embedded in the PDF (free).
//! - [`Tier::Ocr`] — optical character recognition over a rasterized page
//!   (handles scans; deterministic).
//! - [`Tier::Vlm`] — a document vision-language model that recovers tables,
//!   formulas, and reading order (generative; corroborated against lower tiers).
//!
//! # Design
//!
//! `uni-xervo` is a model-inference engine; this crate is the *orchestration*
//! layer on top of it. The OCR and VLM rungs are ordinary `uni-xervo` model
//! capabilities, resolved by alias through the runtime's own accessors and so
//! cached and instrumented for free. This crate adds rasterization, native
//! text parsing, per-page routing, cross-tier verification, and a unified
//! result — none of which belong inside an inference engine.
//!
//! It aligns with the `embed` / `generate` / `nlp` family via an extension
//! trait, so it reads like a native accessor while living in a separate,
//! optionally included crate:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use uni_xervo::runtime::ModelRuntime;
//! # async fn example(runtime: Arc<ModelRuntime>) -> uni_xervo::error::Result<()> {
//! use uni_xervo_pdf::{PdfExt, PdfConfig, DocInput, DocExtractPolicy, Tier};
//!
//! let pdf = runtime.pdf_extractor(PdfConfig::auto()).await?;
//! let pages = pdf
//!     .extract(
//!         DocInput::Pdf { bytes: std::fs::read("doc.pdf").unwrap(), pages: None },
//!         DocExtractPolicy::auto_up_to(Tier::Vlm),
//!     )
//!     .await?;
//! # let _ = pages;
//! # Ok(())
//! # }
//! ```
//!
//! # Features
//!
//! Heavy, target-specific dependencies are gated and additive: rasterizer
//! the pure-Rust `hayro` rasterizer and native PDF text parsing (`pdf-input`).
//! The inference providers themselves are configured on the `uni-xervo`
//! dependency by the consuming application.

#![forbid(unsafe_code)]

pub mod confidence;
pub mod config;
pub mod ext;
pub mod input;
pub mod native;
pub mod pipeline;
pub mod raster;
pub mod result;
pub mod router;
pub mod signals;
pub mod tier;
pub mod verify;

#[cfg(test)]
mod testutil;

#[doc(inline)]
pub use config::{Budget, DocExtractPolicy, LevelPolicy, OutputWant, PdfConfig, VerifyPolicy};
#[doc(inline)]
pub use ext::PdfExt;
#[doc(inline)]
pub use input::{DocInput, NativeText, PageInput, PageRange};
#[cfg(feature = "pdf-input")]
#[doc(inline)]
pub use native::LopdfTextExtractor;
#[doc(inline)]
pub use native::{NativeExtractor, NativePage};
#[doc(inline)]
pub use pipeline::{PdfExtractor, PdfExtractorBuilder};
#[cfg(feature = "hayro")]
#[doc(inline)]
pub use raster::HayroRasterizer;
#[doc(inline)]
pub use raster::{RasterPage, Rasterizer};
#[doc(inline)]
pub use result::{
    Confidence, DerivedSignals, Escalation, EscalationReason, TieredBlock, TieredPageResult,
};
#[doc(inline)]
pub use router::{TierObservation, TierPlan, escalate_on_failure, plan_tier};
#[doc(inline)]
pub use signals::PageSignals;
#[doc(inline)]
pub use tier::Tier;
#[doc(inline)]
pub use verify::{NumericDivergence, numeric_divergence};
