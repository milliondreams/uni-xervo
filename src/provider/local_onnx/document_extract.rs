// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Document extraction task for `local/onnx`.
//!
//! Targets vision-language document parsers that ONNX-export — Granite-Docling
//! (reference, smallest, DocTags output), MinerU 2.5 (structured Markdown +
//! `$$`-delimited LaTeX), and olmOCR-2 (Markdown with inline `$`-delimited
//! LaTeX). Output schemas differ significantly per family; the `style`
//! option (default `"granite-docling"`) selects which parser to apply to
//! the VLM's text output.
//!
//! # v1 scope (this release)
//!
//! The style-aware output parsers are production-ready and live under
//! [`parse_doctags`], [`parse_mineru_markdown`], and
//! [`parse_olmocr_markdown`]. They convert the model's text output into
//! [`DocExtractResult`] regardless of whether the model is wired yet.
//!
//! The actual VLM inference loop (vision encoder forward + LLM decoder
//! generation) is deferred to a follow-up that picks concrete
//! ONNX-exported variants and validates them end-to-end. Until then,
//! `extract()` returns `RuntimeError::Unavailable` with a clear message.
//! Catalog authors can register `document_extract/*` aliases now and they
//! will pass validation, then begin returning real extractions as soon as
//! the follow-up lands.

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::api::ModelAliasSpec;
use crate::error::{Result, RuntimeError};
use crate::traits::{
    DocBlock, DocBlockKind, DocExtractOptions, DocExtractResult, DocumentExtractionModel,
    ImageInput,
};

/// Style of VLM output. Picks the parser to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DocStyle {
    /// Granite-Docling — typed DocTags (`<heading>`, `<table>`, …) with
    /// optional bounding boxes. Smallest VLM; reference target.
    GraniteDocling,
    /// MinerU 2.5 — structured Markdown with `$$..$$` LaTeX blocks.
    Mineru,
    /// olmOCR-2 — Markdown with `$..$` inline LaTeX.
    OlmOcr,
}

/// Entry point for `LocalOnnxProvider::load` when `spec.task == DocumentExtract`.
pub(super) async fn load_document_extractor(
    spec: &ModelAliasSpec,
) -> Result<Arc<dyn DocumentExtractionModel>> {
    let style = match spec
        .options
        .get("style")
        .and_then(Value::as_str)
        .unwrap_or("granite-docling")
    {
        "granite-docling" => DocStyle::GraniteDocling,
        "mineru" => DocStyle::Mineru,
        "olmocr" => DocStyle::OlmOcr,
        other => {
            return Err(RuntimeError::Config(format!(
                "Document extractor '{}' has unknown `style` value '{other}'; \
                 expected one of: granite-docling, mineru, olmocr",
                spec.alias
            )));
        }
    };
    let model = OnnxDocumentExtractor {
        alias: spec.alias.clone(),
        model_id: spec.model_id.clone(),
        style,
    };
    Ok(Arc::new(model) as Arc<dyn DocumentExtractionModel>)
}

struct OnnxDocumentExtractor {
    alias: String,
    model_id: String,
    style: DocStyle,
}

#[async_trait]
impl DocumentExtractionModel for OnnxDocumentExtractor {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn extract(
        &self,
        _pages: Vec<ImageInput>,
        _options: DocExtractOptions,
    ) -> Result<Vec<DocExtractResult>> {
        tracing::warn!(
            alias = %self.alias,
            model_id = %self.model_id,
            style = ?self.style,
            "local/onnx document_extract invoked but the VLM inference loop \
             is not yet wired (v1 scaffold-only release). Returning \
             Unavailable; the provider impl lands in a follow-up."
        );
        Err(RuntimeError::Unavailable)
    }
}

// ---------------------------------------------------------------------------
// Output parsers — convert a VLM's text output into DocExtractResult.
//
// These are the durable, testable units of PR-5. The follow-up that wires
// the actual VLM inference loop just feeds the generated string into the
// matching parser per `style`.
// ---------------------------------------------------------------------------

/// Parse a Granite-Docling DocTags string into a [`DocExtractResult`].
///
/// DocTags is XML-style; each block is wrapped in a typed tag (e.g.
/// `<heading>`, `<table>`, `<formula>`) with an optional `loc` attribute
/// of comma-separated `x0,y0,x1,y1` floats. Inline content is the tag body.
///
/// Unknown tags are mapped to [`DocBlockKind::Text`].
pub fn parse_doctags(input: &str) -> DocExtractResult {
    let mut blocks = Vec::new();
    let mut plain_md = String::new();
    let mut reading_order = 0u32;
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if bytes[i] != b'<' {
            // Untagged text — gobble until next `<` and emit as text.
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            let s = input[start..i].trim();
            if !s.is_empty() {
                blocks.push(DocBlock {
                    kind: DocBlockKind::Text,
                    content: s.to_string(),
                    bbox: None,
                    reading_order,
                });
                reading_order += 1;
                plain_md.push_str(s);
                plain_md.push('\n');
            }
            continue;
        }

        // Parse opening tag `<name attr="...">`.
        let tag_start = i;
        i += 1; // skip '<'
        let name_start = i;
        while i < bytes.len() && bytes[i] != b' ' && bytes[i] != b'>' && bytes[i] != b'/' {
            i += 1;
        }
        let name_end = i;
        let tag_name = &input[name_start..name_end];

        // Find end of opening tag.
        let mut attrs_end = i;
        while attrs_end < bytes.len() && bytes[attrs_end] != b'>' {
            attrs_end += 1;
        }
        if attrs_end >= bytes.len() {
            // Malformed — bail out, drop the trailing junk.
            break;
        }
        let attr_str = &input[name_end..attrs_end];
        let bbox = parse_loc_attr(attr_str);
        i = attrs_end + 1;

        // Self-closing tags (`<x/>`) — emit empty block, continue.
        if input[..attrs_end].ends_with('/') {
            blocks.push(DocBlock {
                kind: doctag_kind(tag_name),
                content: String::new(),
                bbox,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Find matching closing `</name>`.
        let close = format!("</{tag_name}>");
        let body_start = i;
        let close_pos = input[body_start..].find(&close).map(|p| body_start + p);
        let (body_end, after_close) = match close_pos {
            Some(p) => (p, p + close.len()),
            None => {
                // Unclosed — treat the rest as body.
                (input.len(), input.len())
            }
        };
        let body = &input[body_start..body_end];
        let body_trimmed = body.trim().to_string();

        let kind = doctag_kind(tag_name);
        if !body_trimmed.is_empty() {
            plain_md.push_str(&render_block_to_markdown(kind, &body_trimmed));
            plain_md.push('\n');
        }
        blocks.push(DocBlock {
            kind,
            content: body_trimmed,
            bbox,
            reading_order,
        });
        reading_order += 1;
        i = after_close;
        let _ = tag_start;
    }

    DocExtractResult {
        blocks,
        plain_markdown: plain_md.trim_end().to_string(),
    }
}

fn doctag_kind(tag: &str) -> DocBlockKind {
    match tag.to_ascii_lowercase().as_str() {
        "heading" | "title" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => DocBlockKind::Heading,
        "list" | "ul" | "ol" => DocBlockKind::List,
        "table" => DocBlockKind::Table,
        "figure" | "image" => DocBlockKind::Figure,
        "formula" | "math" => DocBlockKind::Formula,
        "caption" => DocBlockKind::Caption,
        "footer" => DocBlockKind::Footer,
        "header" => DocBlockKind::Header,
        _ => DocBlockKind::Text,
    }
}

fn parse_loc_attr(attrs: &str) -> Option<[f32; 4]> {
    let loc_key = "loc=\"";
    let start = attrs.find(loc_key)? + loc_key.len();
    let rest = &attrs[start..];
    let end = rest.find('"')?;
    let parts: Vec<f32> = rest[..end]
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if parts.len() == 4 {
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    }
}

fn render_block_to_markdown(kind: DocBlockKind, content: &str) -> String {
    match kind {
        DocBlockKind::Heading => format!("## {content}"),
        DocBlockKind::List => format!("- {content}"),
        DocBlockKind::Formula => format!("$$\n{content}\n$$"),
        DocBlockKind::Caption => format!("*{content}*"),
        _ => content.to_string(),
    }
}

/// Parse MinerU 2.5's structured Markdown into a [`DocExtractResult`].
///
/// Block-detection heuristics:
/// - Lines starting with `#`, `##`, `###`, … → `Heading`.
/// - Lines starting with `- ` / `* ` / `1.` (numeric prefix) → `List`.
/// - Blocks delimited by `$$..$$` → `Formula` (LaTeX body).
/// - Blocks starting with `| ` (Markdown table syntax) → `Table`.
/// - Blocks with `![...](...)` (image markdown) → `Figure`.
/// - Everything else → `Text`.
pub fn parse_mineru_markdown(input: &str) -> DocExtractResult {
    let mut blocks = Vec::new();
    let mut reading_order = 0u32;

    // First pass: split into double-newline-separated paragraphs.
    for paragraph in input.split("\n\n") {
        let p = paragraph.trim();
        if p.is_empty() {
            continue;
        }

        // Match $$..$$ as a formula block.
        if let Some(inner) = p.strip_prefix("$$").and_then(|r| r.strip_suffix("$$")) {
            blocks.push(DocBlock {
                kind: DocBlockKind::Formula,
                content: inner.trim().to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Image: ![alt](path). Treat as figure with the alt text as content.
        if p.starts_with("![") {
            let alt = p
                .strip_prefix("![")
                .and_then(|r| r.find(']').map(|i| &r[..i]))
                .unwrap_or("");
            blocks.push(DocBlock {
                kind: DocBlockKind::Figure,
                content: alt.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Table: every line starts with '|'.
        if p.lines().all(|line| line.trim_start().starts_with('|')) {
            blocks.push(DocBlock {
                kind: DocBlockKind::Table,
                content: p.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Heading: starts with one or more '#' followed by space.
        if let Some(rest) = p.strip_prefix('#')
            && (rest.starts_with('#') || rest.starts_with(' '))
        {
            let stripped = rest.trim_start_matches('#').trim_start();
            blocks.push(DocBlock {
                kind: DocBlockKind::Heading,
                content: stripped.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // List: first line starts with "- ", "* ", or "N. ".
        let first = p.lines().next().unwrap_or("").trim_start();
        let is_list = first.starts_with("- ")
            || first.starts_with("* ")
            || first.chars().take_while(|c| c.is_ascii_digit()).count() >= 1
                && first.contains(". ");
        if is_list {
            blocks.push(DocBlock {
                kind: DocBlockKind::List,
                content: p.to_string(),
                bbox: None,
                reading_order,
            });
            reading_order += 1;
            continue;
        }

        // Default: plain text.
        blocks.push(DocBlock {
            kind: DocBlockKind::Text,
            content: p.to_string(),
            bbox: None,
            reading_order,
        });
        reading_order += 1;
    }

    DocExtractResult {
        blocks,
        plain_markdown: input.trim().to_string(),
    }
}

/// Parse olmOCR-2's Markdown output. olmOCR-2 emits Markdown that's
/// very close to MinerU's, but inline LaTeX uses single `$..$` rather
/// than `$$..$$`. We reuse the MinerU parser since the block-level
/// heuristics still hold.
///
/// Inline `$..$` is preserved verbatim inside text blocks — no separate
/// formula block, matching the inline convention.
pub fn parse_olmocr_markdown(input: &str) -> DocExtractResult {
    parse_mineru_markdown(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctags_parses_typed_blocks() {
        let input = r#"<heading loc="0.1,0.1,0.9,0.2">Chapter 1</heading><text>Body.</text><formula>E = mc^2</formula>"#;
        let r = parse_doctags(input);
        assert_eq!(r.blocks.len(), 3);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(r.blocks[0].content, "Chapter 1");
        assert_eq!(r.blocks[0].bbox, Some([0.1, 0.1, 0.9, 0.2]));
        assert_eq!(r.blocks[1].kind, DocBlockKind::Text);
        assert_eq!(r.blocks[2].kind, DocBlockKind::Formula);
        assert!(r.plain_markdown.contains("## Chapter 1"));
    }

    #[test]
    fn doctags_handles_unknown_tag_as_text() {
        let r = parse_doctags("<unknown>hello</unknown>");
        assert_eq!(r.blocks.len(), 1);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Text);
        assert_eq!(r.blocks[0].content, "hello");
    }

    #[test]
    fn doctags_assigns_increasing_reading_order() {
        let r = parse_doctags("<heading>A</heading><text>B</text>");
        assert_eq!(r.blocks[0].reading_order, 0);
        assert_eq!(r.blocks[1].reading_order, 1);
    }

    #[test]
    fn mineru_parses_heading_text_formula_table() {
        let input = "# Title\n\nA paragraph.\n\n$$\n\\sum x_i\n$$\n\n| a | b |\n| 1 | 2 |";
        let r = parse_mineru_markdown(input);
        assert_eq!(r.blocks.len(), 4);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(r.blocks[0].content, "Title");
        assert_eq!(r.blocks[1].kind, DocBlockKind::Text);
        assert_eq!(r.blocks[2].kind, DocBlockKind::Formula);
        assert_eq!(r.blocks[3].kind, DocBlockKind::Table);
    }

    #[test]
    fn mineru_parses_lists() {
        let input = "- item one\n- item two";
        let r = parse_mineru_markdown(input);
        assert_eq!(r.blocks.len(), 1);
        assert_eq!(r.blocks[0].kind, DocBlockKind::List);
    }

    #[test]
    fn mineru_parses_figures() {
        let r = parse_mineru_markdown("![A cat](cat.png)");
        assert_eq!(r.blocks.len(), 1);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Figure);
        assert_eq!(r.blocks[0].content, "A cat");
    }

    #[test]
    fn olmocr_reuses_mineru_block_heuristics() {
        let r = parse_olmocr_markdown("# Heading\n\nText with $x = 1$ inline math.");
        assert_eq!(r.blocks.len(), 2);
        assert_eq!(r.blocks[0].kind, DocBlockKind::Heading);
        assert_eq!(r.blocks[1].kind, DocBlockKind::Text);
        // Inline $..$ stays verbatim in text — no separate formula block.
        assert!(r.blocks[1].content.contains("$x = 1$"));
    }
}
