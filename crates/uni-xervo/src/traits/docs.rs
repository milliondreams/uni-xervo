//! Document understanding types and traits — VLM-based extraction and OCR.

use crate::error::Result;
use crate::traits::{ImageInput, ModelInfo};
use async_trait::async_trait;

/// Options for a [`DocumentExtractionModel::extract`] call.
#[derive(Debug, Clone)]
pub struct DocExtractOptions {
    /// Desired primary output format. The `plain_markdown` field on
    /// [`DocExtractResult`] is always populated; this controls the
    /// in-block `content` string format for non-text block kinds.
    pub output: DocOutputFormat,
    /// Whether to extract table blocks (some models skip tables for speed).
    pub include_tables: bool,
    /// Whether to extract math formulas (some models skip formulas for speed).
    pub include_formulas: bool,
    /// Whether to populate per-block bounding boxes (extra work for some VLMs).
    pub include_bboxes: bool,
}

/// Output format for the in-block `content` string of [`DocBlock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DocOutputFormat {
    /// Markdown — the default for text-shaped blocks.
    Markdown,
    /// JSON — structured representation for downstream parsing.
    Json,
    /// HTML — useful for tables and rich layout preservation.
    Html,
}

/// Per-page result of a document extraction call.
#[derive(Debug, Clone)]
pub struct DocExtractResult {
    /// Structured blocks in reading order.
    pub blocks: Vec<DocBlock>,
    /// Concatenated Markdown view (convenience field, always populated).
    pub plain_markdown: String,
}

/// One block within a [`DocExtractResult`].
#[derive(Debug, Clone)]
pub struct DocBlock {
    /// Kind of block.
    pub kind: DocBlockKind,
    /// Block content. Markdown for text / heading / table; LaTeX for formula;
    /// description for figure.
    pub content: String,
    /// Bounding box in page coordinates `[x0, y0, x1, y1]`, when requested
    /// via [`DocExtractOptions::include_bboxes`] and supported by the model.
    pub bbox: Option<[f32; 4]>,
    /// Monotonically increasing index within the page, in reading order.
    pub reading_order: u32,
}

/// Kind of block returned by a [`DocumentExtractionModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DocBlockKind {
    /// Paragraph or running text.
    Text,
    /// Heading (any level).
    Heading,
    /// List item.
    List,
    /// Tabular data.
    Table,
    /// Figure or image with a caption / description.
    Figure,
    /// Math formula.
    Formula,
    /// Figure / table caption.
    Caption,
    /// Page footer (e.g. page number, footnote).
    Footer,
    /// Page header (e.g. running title).
    Header,
}

/// Per-image result of an OCR call.
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Per-word or per-line text blocks with bounding boxes.
    pub blocks: Vec<OcrBlock>,
    /// Concatenated plain text (convenience field, always populated).
    pub plain_text: String,
}

/// One text region within an [`OcrResult`].
#[derive(Debug, Clone)]
pub struct OcrBlock {
    /// Recognized text.
    pub text: String,
    /// Bounding box in image coordinates `[x0, y0, x1, y1]`.
    pub bbox: [f32; 4],
    /// Provider-reported confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A vision-language model that extracts structured blocks from
/// document page images.
///
/// Targets MinerU-2.5, Granite-Docling, olmOCR-2, and similar VLMs that
/// understand page layout, tables, figures, and math. The caller is
/// responsible for rendering PDF pages to images upstream — uni-xervo
/// providers do not own PDF rendering.
#[async_trait]
pub trait DocumentExtractionModel: ModelInfo {
    /// Extract structured blocks from one or more page images.
    ///
    /// Returns one [`DocExtractResult`] per page, in input order.
    ///
    /// # Errors
    /// Returns an error if the provider rejects an input or fails internally.
    async fn extract(
        &self,
        pages: Vec<ImageInput>,
        options: DocExtractOptions,
    ) -> Result<Vec<DocExtractResult>>;

    /// Optional warmup hook. The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// A model that recognizes text in images (optical character recognition).
///
/// OCR is the document-blind counterpart to
/// [`DocumentExtractionModel`] — read text out of an image without
/// interpreting layout. Suited to EasyOCR, PaddleOCR, Tesseract-style models.
#[async_trait]
pub trait OcrModel: ModelInfo {
    /// Recognize text in a batch of images.
    ///
    /// Returns one [`OcrResult`] per input image.
    ///
    /// # Errors
    /// Returns an error if the provider rejects an input or fails internally.
    async fn recognize(&self, images: Vec<ImageInput>) -> Result<Vec<OcrResult>>;

    /// Optional warmup hook. The default is a no-op.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails to initialize.
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn document_extraction_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn DocumentExtractionModel>) {}
    }

    #[test]
    fn ocr_model_is_dyn_safe() {
        fn _accept(_: Arc<dyn OcrModel>) {}
    }
}
