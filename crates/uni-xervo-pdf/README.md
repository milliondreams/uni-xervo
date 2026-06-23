# uni-xervo-pdf

Tiered PDF document extraction for the [`uni-xervo`](../uni-xervo) runtime.

This optional companion crate turns a PDF into structured, provenance-bearing
text by escalating **per page** only as far as needed along a capability ladder:

- `Tier::Native` — digital text already embedded in the PDF text layer (free, no image).
- `Tier::Ocr` — optical character recognition over a rasterized page (handles scans; deterministic).
- `Tier::Vlm` — a document vision-language model that recovers tables, formulas, and reading order (generative; corroborated against lower tiers).

The ordering `Native < Ocr < Vlm` is meaningful — cheapest-to-most-capable.
Each page climbs only as high as its content demands, and the result carries
per-block provenance plus cross-tier verification signals.

## Design

`uni-xervo` is a model-inference engine; this crate is the *orchestration* layer
on top of it. The OCR and VLM rungs are ordinary `uni-xervo` model capabilities
(`ocr` / `document_extract`), resolved by alias through the runtime's own
accessors and so cached and instrumented for free. This crate adds
rasterization, native text parsing, per-page routing, cross-tier verification,
and a unified result — none of which belong inside an inference engine.

It aligns with the `embed` / `generate` / `nlp` family via the `PdfExt`
extension trait, so it reads like a native runtime accessor while living in a
separate, optionally included crate.

## Usage

```rust
use std::sync::Arc;
use uni_xervo::runtime::ModelRuntime;
use uni_xervo_pdf::{PdfExt, PdfConfig, DocInput, DocExtractPolicy, Tier};

async fn example(runtime: Arc<ModelRuntime>) -> uni_xervo::error::Result<()> {
    let pdf = runtime.pdf_extractor(PdfConfig::auto()).await?;
    let pages = pdf
        .extract(
            DocInput::Pdf { bytes: std::fs::read("doc.pdf").unwrap(), pages: None },
            DocExtractPolicy::auto_up_to(Tier::Vlm),
        )
        .await?;
    let _ = pages;
    Ok(())
}
```

The OCR (`ocr`) and VLM (`document_extract`) tiers are resolved by alias from the
`uni-xervo` runtime, so the consuming application registers and configures the
inference providers (and any GPU features) on the `uni-xervo` dependency itself.

## Features

Heavy, target-specific dependencies are gated and additive — each feature only
*adds* capability:

| Feature | Default | Description |
| --- | --- | --- |
| `pdf-input` | yes | Native PDF text-layer extraction (the `Native` tier) via pure-Rust `lopdf`. |
| `hayro` | yes | Pure-Rust PDF page rasterization (the default rasterizer backend), feeding the `Ocr` / `Vlm` tiers. |

The inference providers themselves are configured on the `uni-xervo` dependency
by the consuming application — this crate's library dependency keeps them off.

## License

Apache-2.0 (`LICENSE`).
