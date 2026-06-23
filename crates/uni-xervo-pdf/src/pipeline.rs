//! The tiered extractor — composes native / OCR / VLM into one provenance pass.
//!
//! [`PdfExtractor`] is the concrete pipeline. It holds resolved model handles
//! and the rasterizer/native seams, and runs each page through: derive signals
//! -> [`plan_tier`](crate::router::plan_tier) -> run the chosen tier ->
//! failure-driven escalation -> cross-tier corroboration -> assemble a
//! [`TieredPageResult`]. It is cheap to [`Clone`] (an `Arc` handle), matching
//! the runtime's other service handles.
//!
//! Construct it the family-native way via
//! [`PdfExt::pdf_extractor`](crate::PdfExt::pdf_extractor), or directly via
//! [`PdfExtractor::builder`] (which also accepts injected mock handles, the
//! basis for the deterministic reliability/routing tests).

use std::fmt;
use std::sync::Arc;

use uni_xervo::error::{Result, RuntimeError};
use uni_xervo::traits::{
    DocBlockKind, DocExtractOptions, DocOutputFormat, DocumentExtractionModel, ImageInput, OcrModel,
};

use crate::confidence;
use crate::config::{DocExtractPolicy, LevelPolicy, OutputWant, PdfConfig};
use crate::input::{DocInput, NativeText};
use crate::native::NativeExtractor;
use crate::raster::Rasterizer;
use crate::result::{
    Confidence, DerivedSignals, Escalation, EscalationReason, TieredBlock, TieredPageResult,
};
use crate::router::{self, TierObservation};
use crate::signals::PageSignals;
use crate::tier::Tier;
use crate::verify;

/// Trigram size used for the repetition-loop heuristic on generative output.
const REPETITION_NGRAM: usize = 3;

/// Shared inner state of a [`PdfExtractor`].
struct Inner {
    config: PdfConfig,
    ocr: Option<Arc<dyn OcrModel>>,
    vlm: Option<Arc<dyn DocumentExtractionModel>>,
    rasterizer: Option<Arc<dyn Rasterizer>>,
    native: Option<Arc<dyn NativeExtractor>>,
    /// The highest tier this extractor can actually run, given configured models.
    available_ceiling: Tier,
}

/// A tiered PDF extractor bound to a set of model handles and seams.
///
/// Cloning shares the underlying state (an `Arc`), so a single extractor can be
/// handed to many tasks cheaply.
#[derive(Clone)]
pub struct PdfExtractor {
    inner: Arc<Inner>,
}

impl fmt::Debug for PdfExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PdfExtractor")
            .field("config", &self.inner.config)
            .field("has_ocr", &self.inner.ocr.is_some())
            .field("has_vlm", &self.inner.vlm.is_some())
            .field("has_rasterizer", &self.inner.rasterizer.is_some())
            .field("has_native", &self.inner.native.is_some())
            .field("available_ceiling", &self.inner.available_ceiling)
            .finish()
    }
}

/// Builder for a [`PdfExtractor`].
///
/// Required configuration is passed to [`PdfExtractor::builder`]; the model
/// handles and seams are optional and set via the chainable methods. Omitted
/// tiers simply make the pipeline degrade to the highest available tier.
#[derive(Default)]
pub struct PdfExtractorBuilder {
    config: PdfConfig,
    ocr: Option<Arc<dyn OcrModel>>,
    vlm: Option<Arc<dyn DocumentExtractionModel>>,
    rasterizer: Option<Arc<dyn Rasterizer>>,
    native: Option<Arc<dyn NativeExtractor>>,
}

impl fmt::Debug for PdfExtractorBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PdfExtractorBuilder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PdfExtractorBuilder {
    /// Bind the [`Ocr`](Tier::Ocr) rung to an OCR model.
    #[must_use]
    pub fn ocr(mut self, model: Arc<dyn OcrModel>) -> Self {
        self.ocr = Some(model);
        self
    }

    /// Bind the [`Vlm`](Tier::Vlm) rung to a document-extraction model.
    #[must_use]
    pub fn vlm(mut self, model: Arc<dyn DocumentExtractionModel>) -> Self {
        self.vlm = Some(model);
        self
    }

    /// Set the rasterizer used to render PDF pages for the image tiers.
    #[must_use]
    pub fn rasterizer(mut self, rasterizer: Arc<dyn Rasterizer>) -> Self {
        self.rasterizer = Some(rasterizer);
        self
    }

    /// Set the native text extractor used for the [`Native`](Tier::Native) tier.
    #[must_use]
    pub fn native(mut self, native: Arc<dyn NativeExtractor>) -> Self {
        self.native = Some(native);
        self
    }

    /// Finish building the extractor.
    #[must_use]
    pub fn build(self) -> PdfExtractor {
        let available_ceiling = if self.vlm.is_some() {
            Tier::Vlm
        } else if self.ocr.is_some() {
            Tier::Ocr
        } else {
            Tier::Native
        };
        PdfExtractor {
            inner: Arc::new(Inner {
                config: self.config,
                ocr: self.ocr,
                vlm: self.vlm,
                rasterizer: self.rasterizer,
                native: self.native,
                available_ceiling,
            }),
        }
    }
}

/// Internal per-tier run output before provenance assembly.
struct RunOutcome {
    blocks: Vec<TieredBlock>,
    plain_markdown: String,
}

/// Everything the per-page routines need about one page.
///
/// Bundling these keeps the routing/run methods to a sane arity and threads the
/// (possibly borrowed) PDF bytes for lazy rasterization without re-plumbing.
struct PageCtx<'a> {
    page_number: u32,
    native: Option<NativeText>,
    ready_image: Option<ImageInput>,
    pdf: Option<&'a [u8]>,
}

impl PdfExtractor {
    /// Start building an extractor with the given configuration.
    #[must_use]
    pub fn builder(config: PdfConfig) -> PdfExtractorBuilder {
        PdfExtractorBuilder {
            config,
            ..PdfExtractorBuilder::default()
        }
    }

    /// The configuration this extractor was built with.
    #[must_use]
    pub fn config(&self) -> &PdfConfig {
        &self.inner.config
    }

    /// Extract structured, provenance-bearing content from `input`.
    ///
    /// Each page is routed independently under `policy`; the result preserves
    /// input page order. The policy is first narrowed to what the configured
    /// models can actually run (hardware-adaptive degradation), so a missing
    /// VLM silently caps `Auto`/`Ceiling` at OCR rather than erroring.
    ///
    /// # Errors
    /// Returns an error if a [`Fixed`](LevelPolicy::Fixed) tier is unavailable,
    /// if `Pdf` input is given without a native extractor, or if an underlying
    /// model fails.
    pub async fn extract(
        &self,
        input: DocInput,
        policy: DocExtractPolicy,
    ) -> Result<Vec<TieredPageResult>> {
        let effective = self.effective_policy(policy)?;
        let mut vlm_used: u32 = 0;

        match input {
            DocInput::Pages(pages) => {
                let mut out = Vec::with_capacity(pages.len());
                for page in pages {
                    let ctx = PageCtx {
                        page_number: page.page_number,
                        native: page.native_text,
                        ready_image: Some(page.image),
                        pdf: None,
                    };
                    out.push(self.process_page(ctx, &effective, &mut vlm_used).await?);
                }
                Ok(out)
            }
            DocInput::Pdf { bytes, pages } => {
                let native = self.inner.native.as_ref().ok_or_else(|| {
                    RuntimeError::Config(
                        "Pdf input requires a native extractor (enable the `pdf-input` feature)"
                            .to_string(),
                    )
                })?;
                let native_pages = native.extract(&bytes, pages)?;
                let mut out = Vec::with_capacity(native_pages.len());
                for np in native_pages {
                    let ctx = PageCtx {
                        page_number: np.page_number,
                        native: Some(np.native),
                        ready_image: None,
                        pdf: Some(&bytes),
                    };
                    out.push(self.process_page(ctx, &effective, &mut vlm_used).await?);
                }
                Ok(out)
            }
        }
    }

    /// Narrow a requested policy to what the configured models can run.
    fn effective_policy(&self, mut policy: DocExtractPolicy) -> Result<DocExtractPolicy> {
        let ceiling = self.inner.available_ceiling;
        policy.level = match policy.level {
            LevelPolicy::Fixed(t) => {
                if t > ceiling {
                    return Err(RuntimeError::CapabilityMismatch(format!(
                        "fixed tier {t:?} is unavailable (no model configured for it)"
                    )));
                }
                LevelPolicy::Fixed(t)
            }
            LevelPolicy::Ceiling(t) => LevelPolicy::Ceiling(t.min(ceiling)),
            LevelPolicy::Auto { min, max } => LevelPolicy::Auto {
                min: min.min(ceiling),
                max: max.min(ceiling),
            },
        };
        Ok(policy)
    }

    /// Route and run a single page.
    async fn process_page(
        &self,
        ctx: PageCtx<'_>,
        effective: &DocExtractPolicy,
        vlm_used: &mut u32,
    ) -> Result<TieredPageResult> {
        // Cheap signals from the native text, plus the v1 structure heuristic:
        // absent a layout classifier, "structure requested" is the structure
        // hint. Replace this single line with a classifier's per-page cue when
        // the layout-model upgrade lands.
        let base = ctx
            .native
            .as_ref()
            .map(PageSignals::from_native)
            .unwrap_or(PageSignals {
                has_text_layer: false,
                text_coverage: 0.0,
                garbled: false,
                looks_structured: false,
            });
        let signals = base.with_structured(effective.want == OutputWant::Structure);

        let plan = router::plan_tier(signals, effective);
        let mut escalations = plan.escalations;
        let mut tier = plan.target;

        // Budget cap: a VLM-targeted page beyond the page budget drops to the
        // highest deterministic tier still allowed (graceful, not an error).
        if tier == Tier::Vlm && !self.budget_allows_vlm(effective, *vlm_used) {
            let capped = if effective.level.allows(Tier::Ocr) && self.inner.ocr.is_some() {
                Tier::Ocr
            } else {
                Tier::Native
            };
            escalations.push(Escalation {
                from: Tier::Vlm,
                to: capped,
                reason: EscalationReason::BudgetExhausted,
            });
            tier = capped;
        }

        let mut outcome = self.run_tier(tier, &ctx, effective, vlm_used).await?;

        // One-step failure-driven escalation (ParseFixer pattern).
        let observation = TierObservation {
            empty: outcome.blocks.is_empty(),
            flat: tier == Tier::Ocr
                && effective.want == OutputWant::Structure
                && !outcome.blocks.is_empty(),
        };
        if let Some(esc) = router::escalate_on_failure(tier, observation, effective)
            && self.tier_runnable(esc.to, &ctx)
            && (esc.to != Tier::Vlm || self.budget_allows_vlm(effective, *vlm_used))
        {
            escalations.push(esc);
            tier = esc.to;
            outcome = self.run_tier(tier, &ctx, effective, vlm_used).await?;
        }

        Ok(TieredPageResult {
            page_number: ctx.page_number,
            blocks: outcome.blocks,
            plain_markdown: outcome.plain_markdown,
            produced_by: tier,
            escalations,
        })
    }

    /// Run exactly one tier for a page.
    async fn run_tier(
        &self,
        tier: Tier,
        ctx: &PageCtx<'_>,
        effective: &DocExtractPolicy,
        vlm_used: &mut u32,
    ) -> Result<RunOutcome> {
        match tier {
            Tier::Native => Ok(Self::run_native(&ctx.native)),
            Tier::Ocr => {
                let ocr = self.inner.ocr.as_ref().ok_or_else(|| {
                    RuntimeError::CapabilityMismatch(
                        "OCR tier requested but no OCR model is configured".to_string(),
                    )
                })?;
                let image = self.resolve_image(ctx)?;
                Self::run_ocr(ocr, image).await
            }
            Tier::Vlm => {
                let vlm = self.inner.vlm.as_ref().ok_or_else(|| {
                    RuntimeError::CapabilityMismatch(
                        "VLM tier requested but no document-extraction model is configured"
                            .to_string(),
                    )
                })?;
                let image = self.resolve_image(ctx)?;
                let reference = self.deterministic_reference(ctx, effective).await?;
                *vlm_used += 1;
                Self::run_vlm(vlm, image, reference, effective).await
            }
        }
    }

    /// Build the [`Native`](Tier::Native) result from a parsed text layer.
    fn run_native(native: &Option<NativeText>) -> RunOutcome {
        let text = native.as_ref().map(|n| n.text.clone()).unwrap_or_default();
        let blocks = split_paragraphs(&text)
            .into_iter()
            .enumerate()
            .map(|(i, para)| TieredBlock {
                kind: DocBlockKind::Text,
                content: para,
                bbox: None,
                reading_order: i as u32,
                produced_by: Tier::Native,
                confidence: Confidence::Deterministic,
            })
            .collect();
        RunOutcome {
            blocks,
            plain_markdown: text,
        }
    }

    /// Run OCR recognition for one page image.
    async fn run_ocr(ocr: &Arc<dyn OcrModel>, image: ImageInput) -> Result<RunOutcome> {
        let mut results = ocr.recognize(vec![image]).await?;
        let result = results
            .pop()
            .ok_or_else(|| RuntimeError::InferenceError("OCR returned no result".to_string()))?;
        let blocks = result
            .blocks
            .into_iter()
            .enumerate()
            .map(|(i, b)| TieredBlock {
                kind: DocBlockKind::Text,
                content: b.text,
                bbox: Some(b.bbox),
                reading_order: i as u32,
                produced_by: Tier::Ocr,
                // OCR recognition reports a real per-block probability.
                confidence: Confidence::Measured(b.confidence),
            })
            .collect();
        Ok(RunOutcome {
            blocks,
            plain_markdown: result.plain_text,
        })
    }

    /// Run VLM extraction for one page image, corroborating against `reference`.
    async fn run_vlm(
        vlm: &Arc<dyn DocumentExtractionModel>,
        image: ImageInput,
        reference: Option<String>,
        effective: &DocExtractPolicy,
    ) -> Result<RunOutcome> {
        let options = DocExtractOptions {
            output: DocOutputFormat::Markdown,
            include_tables: true,
            include_formulas: true,
            include_bboxes: true,
        };
        let mut results = vlm.extract(vec![image], options).await?;
        let result = results
            .pop()
            .ok_or_else(|| RuntimeError::InferenceError("VLM returned no result".to_string()))?;

        let blocks = result
            .blocks
            .into_iter()
            .map(|b| {
                // Per-block numeric corroboration against the deterministic tier.
                let divergence = if effective.verify.numeric_guard {
                    reference
                        .as_ref()
                        .map(|r| verify::numeric_divergence(r, &b.content).ratio)
                } else {
                    None
                };
                let signals = DerivedSignals {
                    // The core trait does not yet surface EOS / finish_reason;
                    // assume clean until that signal is plumbed through.
                    finished_cleanly: true,
                    repetition_ratio: confidence::repetition_ratio(&b.content, REPETITION_NGRAM),
                    length_ratio: None,
                    cross_tier_divergence: divergence,
                };
                TieredBlock {
                    kind: b.kind,
                    content: b.content,
                    bbox: b.bbox,
                    reading_order: b.reading_order,
                    produced_by: Tier::Vlm,
                    confidence: confidence::from_signals(signals),
                }
            })
            .collect();
        Ok(RunOutcome {
            blocks,
            plain_markdown: result.plain_markdown,
        })
    }

    /// The deterministic text to corroborate a VLM page against, if any.
    ///
    /// Prefers the native text layer; falls back to a fresh OCR pass when
    /// cross-tier verification is on and an OCR model is available. `None` means
    /// the page is *unverified* (no deterministic tier to compare with).
    async fn deterministic_reference(
        &self,
        ctx: &PageCtx<'_>,
        effective: &DocExtractPolicy,
    ) -> Result<Option<String>> {
        if let Some(n) = ctx.native.as_ref()
            && n.is_usable()
        {
            return Ok(Some(n.text.clone()));
        }
        if effective.verify.cross_tier
            && let Some(ocr) = self.inner.ocr.as_ref()
            && let Ok(image) = self.resolve_image(ctx)
        {
            let outcome = Self::run_ocr(ocr, image).await?;
            return Ok(Some(outcome.plain_markdown));
        }
        Ok(None)
    }

    /// Resolve a page image, lazily rasterizing from `pdf` when not pre-rendered.
    fn resolve_image(&self, ctx: &PageCtx<'_>) -> Result<ImageInput> {
        if let Some(image) = ctx.ready_image.as_ref() {
            return Ok(image.clone());
        }
        let page_number = ctx.page_number;
        match (self.inner.rasterizer.as_ref(), ctx.pdf) {
            (Some(rasterizer), Some(bytes)) => rasterizer
                .rasterize_pages(bytes, &[page_number])?
                .into_iter()
                .next()
                .map(|rp| rp.image)
                .ok_or_else(|| {
                    RuntimeError::InferenceError(format!(
                        "rasterizer produced no image for page {page_number}"
                    ))
                }),
            _ => Err(RuntimeError::Config(format!(
                "no image available for page {page_number} and no rasterizer configured"
            ))),
        }
    }

    /// Whether `tier` can actually be run for a page right now.
    fn tier_runnable(&self, tier: Tier, ctx: &PageCtx<'_>) -> bool {
        let have_image =
            ctx.ready_image.is_some() || (self.inner.rasterizer.is_some() && ctx.pdf.is_some());
        match tier {
            Tier::Native => true,
            Tier::Ocr => self.inner.ocr.is_some() && have_image,
            Tier::Vlm => self.inner.vlm.is_some() && have_image,
        }
    }

    /// Whether the per-document VLM page budget still permits another VLM page.
    fn budget_allows_vlm(&self, effective: &DocExtractPolicy, vlm_used: u32) -> bool {
        match effective.budget.and_then(|b| b.max_vlm_pages) {
            Some(max) => vlm_used < max,
            None => true,
        }
    }
}

/// Split text into paragraph blocks on blank lines, dropping empties.
fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};
    use uni_xervo::traits::{DocBlock, DocExtractResult, OcrBlock, OcrResult};

    // ---- companion-local mocks over the public traits (no runtime needed) ----

    struct MockOcr {
        result: OcrResult,
        calls: Arc<AtomicU32>,
    }

    impl uni_xervo::traits::ModelInfo for MockOcr {
        fn model_id(&self) -> &str {
            "mock-ocr"
        }
    }

    #[async_trait]
    impl OcrModel for MockOcr {
        async fn recognize(&self, images: Vec<ImageInput>) -> Result<Vec<OcrResult>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(images.iter().map(|_| self.result.clone()).collect())
        }
    }

    struct MockVlm {
        result: DocExtractResult,
        calls: Arc<AtomicU32>,
    }

    impl uni_xervo::traits::ModelInfo for MockVlm {
        fn model_id(&self) -> &str {
            "mock-vlm"
        }
    }

    #[async_trait]
    impl DocumentExtractionModel for MockVlm {
        async fn extract(
            &self,
            pages: Vec<ImageInput>,
            _options: DocExtractOptions,
        ) -> Result<Vec<DocExtractResult>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(pages.iter().map(|_| self.result.clone()).collect())
        }
    }

    fn image() -> ImageInput {
        ImageInput::Bytes {
            data: vec![0u8; 4],
            media_type: "image/png".to_string(),
        }
    }

    fn ocr_returning(text: &str) -> (Arc<dyn OcrModel>, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let model = MockOcr {
            result: OcrResult {
                blocks: vec![OcrBlock {
                    text: text.to_string(),
                    bbox: [0.0, 0.0, 1.0, 1.0],
                    confidence: 0.9,
                }],
                plain_text: text.to_string(),
            },
            calls: calls.clone(),
        };
        let model: Arc<dyn OcrModel> = Arc::new(model);
        (model, calls)
    }

    fn vlm_returning(content: &str) -> (Arc<dyn DocumentExtractionModel>, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let model = MockVlm {
            result: DocExtractResult {
                blocks: vec![DocBlock {
                    kind: DocBlockKind::Text,
                    content: content.to_string(),
                    bbox: None,
                    reading_order: 0,
                }],
                plain_markdown: content.to_string(),
            },
            calls: calls.clone(),
        };
        let model: Arc<dyn DocumentExtractionModel> = Arc::new(model);
        (model, calls)
    }

    fn native(text: &str) -> NativeText {
        NativeText {
            text: text.to_string(),
            coverage: 0.5,
            garbled: false,
        }
    }

    /// R6 invariant helper: every block is fully provenance-tagged.
    fn assert_provenance_complete(result: &TieredPageResult) {
        for block in &result.blocks {
            assert_eq!(
                block.produced_by, result.produced_by,
                "block producer must match page producer in v1"
            );
            assert!(
                (0.0..=1.0).contains(&block.confidence.score()),
                "confidence score must be in [0, 1]"
            );
        }
    }

    // C1: born-digital + want:Text -> Native; OCR & VLM never called.
    #[tokio::test]
    async fn c1_born_digital_uses_no_models() {
        let (ocr, ocr_calls) = ocr_returning("ignored");
        let (vlm, vlm_calls) = vlm_returning("ignored");
        let pdf = PdfExtractor::builder(PdfConfig::auto())
            .ocr(ocr)
            .vlm(vlm)
            .build();

        let pages = pdf
            .extract(
                DocInput::Pages(vec![crate::PageInput {
                    image: image(),
                    native_text: Some(native("a full page of digital text")),
                    page_number: 1,
                }]),
                DocExtractPolicy::auto_up_to(Tier::Vlm),
            )
            .await
            .unwrap();

        assert_eq!(pages[0].produced_by, Tier::Native);
        assert_eq!(ocr_calls.load(Ordering::SeqCst), 0);
        assert_eq!(vlm_calls.load(Ordering::SeqCst), 0);
        assert_provenance_complete(&pages[0]);
    }

    // R1: a VLM altering a number is flagged with near-zero derived confidence.
    #[tokio::test]
    async fn r1_altered_number_is_flagged() {
        let (vlm, _) = vlm_returning("Total: $1,284.56");
        let pdf = PdfExtractor::builder(PdfConfig::auto()).vlm(vlm).build();

        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.want = OutputWant::Structure;

        let pages = pdf
            .extract(
                DocInput::Pages(vec![crate::PageInput {
                    image: image(),
                    native_text: Some(native("Total: $1,234.56")),
                    page_number: 1,
                }]),
                policy,
            )
            .await
            .unwrap();

        assert_eq!(pages[0].produced_by, Tier::Vlm);
        let block = &pages[0].blocks[0];
        assert!(
            block.confidence.score() < 0.5,
            "divergent numeric content must score low, got {}",
            block.confidence.score()
        );
        assert_provenance_complete(&pages[0]);
    }

    // A matching VLM number is NOT penalized for divergence.
    #[tokio::test]
    async fn vlm_matching_number_scores_high() {
        let (vlm, _) = vlm_returning("Total: $1,234.56");
        let pdf = PdfExtractor::builder(PdfConfig::auto()).vlm(vlm).build();
        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.want = OutputWant::Structure;

        let pages = pdf
            .extract(
                DocInput::Pages(vec![crate::PageInput {
                    image: image(),
                    native_text: Some(native("Total: $1,234.56")),
                    page_number: 1,
                }]),
                policy,
            )
            .await
            .unwrap();
        assert!(pages[0].blocks[0].confidence.score() > 0.9);
    }

    // R5: Ceiling(Ocr) forbids the VLM even when structure is requested.
    #[tokio::test]
    async fn r5_ceiling_forbids_vlm() {
        let (ocr, ocr_calls) = ocr_returning("scanned text");
        let (vlm, vlm_calls) = vlm_returning("structured");
        let pdf = PdfExtractor::builder(PdfConfig::auto())
            .ocr(ocr)
            .vlm(vlm)
            .build();

        let mut policy = DocExtractPolicy::ceiling(Tier::Ocr);
        policy.want = OutputWant::Structure;

        let pages = pdf
            .extract(
                // No native text -> would need an image tier.
                DocInput::Pages(vec![crate::PageInput {
                    image: image(),
                    native_text: None,
                    page_number: 1,
                }]),
                policy,
            )
            .await
            .unwrap();

        assert_eq!(pages[0].produced_by, Tier::Ocr);
        assert_eq!(
            vlm_calls.load(Ordering::SeqCst),
            0,
            "VLM must not run under Ceiling(Ocr)"
        );
        assert!(ocr_calls.load(Ordering::SeqCst) >= 1);
    }

    // C10: a VLM page budget caps later pages, gracefully (no error).
    #[tokio::test]
    async fn c10_budget_caps_vlm_pages() {
        let (ocr, _) = ocr_returning("ocr fallback");
        let (vlm, vlm_calls) = vlm_returning("structured");
        let pdf = PdfExtractor::builder(PdfConfig::auto())
            .ocr(ocr)
            .vlm(vlm)
            .build();

        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.want = OutputWant::Structure;
        policy.budget = Some(crate::Budget {
            max_vlm_pages: Some(1),
            max_latency: None,
        });

        let page = |n| crate::PageInput {
            image: image(),
            native_text: None,
            page_number: n,
        };
        let pages = pdf
            .extract(DocInput::Pages(vec![page(1), page(2)]), policy)
            .await
            .unwrap();

        assert_eq!(pages[0].produced_by, Tier::Vlm);
        assert_eq!(pages[1].produced_by, Tier::Ocr, "second page capped to OCR");
        assert_eq!(vlm_calls.load(Ordering::SeqCst), 1);
        assert!(
            pages[1]
                .escalations
                .iter()
                .any(|e| e.reason == EscalationReason::BudgetExhausted)
        );
    }

    // C11: no VLM configured -> Auto silently caps at OCR, no error.
    #[tokio::test]
    async fn c11_no_gpu_degrades_to_ocr() {
        let (ocr, _) = ocr_returning("scanned text");
        let pdf = PdfExtractor::builder(PdfConfig::auto()).ocr(ocr).build();

        let mut policy = DocExtractPolicy::auto_up_to(Tier::Vlm);
        policy.want = OutputWant::Structure;

        let pages = pdf
            .extract(
                DocInput::Pages(vec![crate::PageInput {
                    image: image(),
                    native_text: None,
                    page_number: 1,
                }]),
                policy,
            )
            .await
            .unwrap();

        assert_eq!(pages[0].produced_by, Tier::Ocr);
        assert_provenance_complete(&pages[0]);
    }

    // A Fixed tier that is unavailable is a hard error (explicit request).
    #[tokio::test]
    async fn fixed_unavailable_tier_errors() {
        let pdf = PdfExtractor::builder(PdfConfig::auto()).build(); // no models
        let err = pdf
            .extract(
                DocInput::Pages(vec![crate::PageInput {
                    image: image(),
                    native_text: None,
                    page_number: 1,
                }]),
                DocExtractPolicy::fixed(Tier::Vlm),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, RuntimeError::CapabilityMismatch(_)));
    }

    // End-to-end on the real `Pdf` path: a born-digital page stays Native and
    // never touches the rasterizer or OCR (cost proof on real parsing).
    #[cfg(all(feature = "pdf-input", feature = "hayro"))]
    #[tokio::test]
    async fn pdf_input_born_digital_stays_native_without_raster() {
        use crate::native::LopdfTextExtractor;
        use crate::raster::HayroRasterizer;
        use crate::testutil::make_text_pdf;

        let (ocr, ocr_calls) = ocr_returning("unused");
        let dense = "Lorem ipsum dolor sit amet consectetur ".repeat(8);
        let bytes = make_text_pdf(&dense);
        let pdf = PdfExtractor::builder(PdfConfig::auto())
            .ocr(ocr)
            .native(Arc::new(LopdfTextExtractor::new()))
            .rasterizer(Arc::new(HayroRasterizer::new()))
            .build();

        let pages = pdf
            .extract(
                DocInput::Pdf { bytes, pages: None },
                DocExtractPolicy::auto_up_to(Tier::Ocr),
            )
            .await
            .unwrap();

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].produced_by, Tier::Native);
        assert_eq!(ocr_calls.load(Ordering::SeqCst), 0);
    }

    // End-to-end on the real `Pdf` path: a sparse page escalates to OCR, which
    // lazily drives the real hayro rasterizer.
    #[cfg(all(feature = "pdf-input", feature = "hayro"))]
    #[tokio::test]
    async fn pdf_input_sparse_page_rasterizes_and_ocrs() {
        use crate::native::LopdfTextExtractor;
        use crate::raster::HayroRasterizer;
        use crate::testutil::make_text_pdf;

        let (ocr, ocr_calls) = ocr_returning("scanned text");
        // Minimal text -> low coverage -> routes to OCR over a real render.
        let bytes = make_text_pdf("Hi");
        let pdf = PdfExtractor::builder(PdfConfig::auto())
            .ocr(ocr)
            .native(Arc::new(LopdfTextExtractor::new()))
            .rasterizer(Arc::new(HayroRasterizer::new()))
            .build();

        let pages = pdf
            .extract(
                DocInput::Pdf { bytes, pages: None },
                DocExtractPolicy::auto_up_to(Tier::Ocr),
            )
            .await
            .unwrap();

        assert_eq!(pages[0].produced_by, Tier::Ocr);
        assert_eq!(ocr_calls.load(Ordering::SeqCst), 1);
    }
}
