# Tiered Document Extraction for uni-xervo

**Status:** Draft — for discussion
**Created:** 2026-06-15
**Author:** xervo team (in response to the uniko `xervo-pdf-extraction-ask`)
**Consumers:** uniko-extract (PDF ingest into a knowledge graph); any future doc-AI consumer

> **Source ask:** uniko proposed a single tiered document-extraction API
> (`plain-text` → `ppocr` → `olmocr`, pinned/bounded/auto, escalate-only-as-needed,
> provenance-bearing). This proposal is xervo's design response. It accepts the
> intent of the ask but **re-homes the orchestration into a separate companion
> crate** and refines the API around xervo's existing idioms.

> **Note on paths:** as of the `crates/` workspace restructure, all
> `src/…:line` references in this document are now rooted at
> `crates/uni-xervo/` (core) and `crates/uni-xervo-pdf/` (companion).

---

## 1. The one decision that shapes everything

The ask frames its central fork (`§7.2`) as *"does xervo take `PdfBytes` end-to-end
(Option A) or `{native_text, page_images}` (Option B)?"* That framing misses a
structural fact about xervo:

> **xervo is a model-inference engine, not a document pipeline.**

Every existing capability is *one model behind a task-trait*, selected by alias,
deduplicated, and instrumented. Inputs are **already-decoded** (`ImageInput`
bytes, text, audio). There is no PDF parsing, no rasterization, and — crucially —
**no orchestration primitive** (no model-chaining, no per-item routing) anywhere
in the codebase.

The tiered router the ask describes is therefore a **different kind of object**
than anything xervo has. It is *orchestration*, not *inference*. It consumes
models; it is not a model. Cramming it into `Provider::load()` (which returns a
single model) would break xervo's core abstraction.

**Resolution:** a three-layer design that keeps xervo's core pure and puts the
orchestration in a **separate companion crate** (`uni-doc-extract`, name TBD).

```
┌─ Layer 3 — plug-and-play  ── extract_document(pdf) → results, sane defaults, hardware-adaptive
│            (in uni-doc-extract)
├─ Layer 2 — tiered router  ── policy · per-page escalation · rasterize · native-parse · provenance · verify
│            (in uni-doc-extract; takes Arc<ModelRuntime>, composes core capabilities)
└─ Layer 1 — atomic models  ── OcrModel (PP-OCRv6 det+rec) · DocumentExtractionModel (olmOCR-2)
             (in uni-xervo core — pure inference, selected by alias, deduplicated, instrumented)
```

### Why a separate crate (not in-core, not caller-owned)

- **Keeps core xervo pure.** The rasterizer (`hayro`, pure-Rust) and the PDF text
  parser (`pdf-extract`/`lopdf`) are **non-inference** dependencies. They do not
  belong in an inference engine. A companion crate isolates them.
- **Still plug-and-play.** A single `uni-doc-extract` dependency + one call gives
  every consumer the whole pipeline. uniko does **not** re-solve rasterization,
  native extraction, or routing (the failure mode of the ask's Option B).
- **Clean dependency direction.** `uni-doc-extract` depends on `uni-xervo`; core
  never depends on the companion. Core advances independently.

---

## 2. What core xervo already provides (verified against 0.14.0)

The ask claims this is "incremental." Audit confirms it largely is.

| Capability | State in core | Evidence |
|---|---|---|
| `DocumentExtractionModel` trait + `extract()` | **Wired, gated** — returns `Unavailable` (was waiting on ONNX exports that aren't coming) | `traits/docs.rs:111`, `provider/local_onnx/document_extract.rs:92` |
| olmOCR / MinerU / DocTags markdown parsers | **Present, tested** | `provider/local_onnx/document_extract.rs:155–424` (+9 passing tests) |
| Autoregressive greedy decoder | **Present, tested** | `provider/local_onnx/autoreg.rs:57` (+7 passing tests) |
| `OcrModel` — CTC recognition | **Present** (recognition only) | `provider/local_onnx/ocr.rs:177` |
| OCR **detection** (DBNet) | **Absent** — docstring defers to "v2" | `provider/local_onnx/ocr.rs:4–19` |
| mistral.rs vision path, arbitrary `model_id`, image bytes, deterministic sampling | **Present** — Qwen2.5-VL supported, ISQ Q4K | `provider/mistralrs.rs:334`, `:827` |
| `DocBlock` w/ `bbox`, `reading_order` | **Present** | `traits/docs.rs:46` |
| `DocBlock` w/ `confidence`, `produced_by` | **Absent** (only `OcrBlock` has `confidence`) | `traits/docs.rs:46,94` |
| `RawTensorModel` escape hatch (could host a detector) | **Present** | `traits/raw_tensor_model.rs:100` |
| `ModelTask::{Ocr, DocumentExtract}` + runtime accessors | **Present** (`#[non_exhaustive]` enum) | `api.rs:14`, `runtime.rs` |

**Implication for the split:**

- **Core work (inference):** complete `OcrModel` with detection (det + rec +
  box-merge + reading-order → full-page OCR); revive `DocumentExtractionModel::
  extract()` onto the mistral.rs vision path for olmOCR-2. Both already have
  traits, tasks, and accessors — this finishes existing surfaces.
- **Companion work (orchestration):** everything else (rasterize, native-parse,
  route, verify, provenance, plug-and-play entry).

---

## 3. The level ladder, and what each rung needs

| Tier | Answers | Mechanism | Modality | Hallucination | Cost | Home |
|---|---|---|---|---|---|---|
| **Native** | "what text is digitally embedded" | pure-Rust PDF text parse | PDF text layer | none | ~free | companion |
| **Ocr** | "what text is visually present" (scans) | PP-OCRv6 det→rec on `ort` | **page raster** | none (reads) | low, CPU-viable | core `OcrModel` |
| **Vlm** | "what is the document *structure*" (tables/formulas/order) | olmOCR-2 (Qwen2.5-VL-7B) on `mistral.rs` | **page raster** | **yes (generates)** | high, GPU | core `DocumentExtractionModel` |

> **The ladder mixes two modalities.** `Native` reads the PDF's text stream — no
> image. `Ocr`/`Vlm` read a **rasterized page image**. The `Native`→`Ocr` jump is
> therefore not "more effort" — it requires **PDF page → pixels** (§4), the
> gating dependency.

### Tiers are *bound capabilities*, not a hardcoded enum

The ask bakes `Level { PlainText, Ppocr, Olmocr }` into the types. That is
brittle — it cannot absorb olmOCR-3, a future PaddleOCR-VL tier, or a
layout-classifier rung without a breaking change. Instead, the *semantic* tier
is stable while the *concrete model* behind it is config (xervo's existing alias
mechanism):

```rust
/// Stable, ordered semantic rungs. Policy + provenance refer to these.
pub enum Tier { Native, Ocr, Vlm }

// The model behind each rung is catalog config, e.g.
//   Tier::Ocr -> alias "ocr/ppocr-v6"      (an OcrModel)
//   Tier::Vlm -> alias "docext/olmocr-2"    (a DocumentExtractionModel)
```

New models slot in by re-binding an alias — no API churn.

---

## 4. Rasterization — the gating dependency (resolved)

**Why a rasterizer at all (it is *not* "beyond" PP-OCR/VLM — it is *prior* to
them):** PP-OCR and olmOCR both take a **pixel image** as input; neither can open
a PDF. A PDF page is a vector/text *description*, not pixels. Something must
render the page to a bitmap before any image tier can run. That renderer adds
zero extraction capability — it is the bridge that lets the image tiers see the
page at all. Without it, the entire image half of the ask (Ocr **and** Vlm) is
unreachable. It is invoked **only on the escalation path** — born-digital text
pages stay on `Native` and are never rasterized.

**What we use today:** nothing rasterizes. uniko's current path is `pdf-extract`
(pure-Rust), which pulls the **text layer only** — which is exactly why scanned
PDFs return empty: no text layer to pull, and nothing renders the page to pixels.

**Decision:** `hayro` (pure-Rust, via `vello_cpu`) behind a `Rasterizer` trait
in the companion crate. **No FFI** — this project does not take C/C++ rasterizer
bindings. `hayro` has zero C dependencies and a permissive MIT/Apache license.

> The rasterizer's output is the **only** thing the OCR/VLM ever sees: a dropped
> glyph or mangled table at raster time propagates into extraction and is
> **unrecoverable** downstream. So rasterization fidelity matters — but the
> answer is to validate and, if needed, improve `hayro` (it is young and
> perf-unoptimized), not to reach for an FFI renderer. The `Rasterizer` trait
> leaves room for a second *pure-Rust* backend if one is ever needed.

**Avoided (FFI / copyleft):** `pdfium` (C++ FFI), MuPDF (AGPL), and Poppler
(GPL) are all disqualified — no FFI, and no copyleft landmines.

**Action item (gates the image tiers):** benchmark `hayro` fidelity on a real
document mix (tables, small fonts) before committing latency-sensitive paths to
it. This is the long pole.

---

## 5. The API (companion crate)

### 5.1 Input — resolve §7.2 with an enum, not an architecture-wide choice

The ask treats "xervo owns PDFs" vs "caller owns pixels" as mutually exclusive.
They are not — xervo already models this exact pattern with `ImageInput::Bytes |
Url`. Mirror it:

```rust
pub enum DocInput {
    /// Plug-and-play (Option A). The crate parses + rasterizes.
    /// Requires the `pdf-input` + a rasterizer feature.
    Pdf { bytes: Vec<u8>, pages: Option<PageRange> },

    /// Escape hatch (Option B). Caller already rendered pixels
    /// (+ optionally pre-parsed the text layer). No rasterizer needed.
    Pages(Vec<PageInput>),
}

pub struct PageInput {
    pub image: ImageInput,             // reuse core's type
    pub native_text: Option<NativeText>,
    pub page_number: u32,
}
```

A-vs-B becomes a **per-call** decision. Plug-and-play callers pass `Pdf`;
callers on a platform without the FFI, or who already rasterize, pass `Pages`.
One API serves both.

> Accepting `DocInput::Pdf` makes native-text extraction nearly free — it falls
> out of the parse we must do anyway, and pays for itself three times: it *is*
> `Native`, it powers the cheap routing signals (§5.4), and it enables
> cross-tier verification (§5.3). Feature-gated under `pdf-input` so
> `Pages`-only callers don't pay for it.

### 5.2 Policy — the reliability + cost knob

```rust
pub struct DocExtractPolicy {
    pub level:  LevelPolicy,
    pub want:   OutputWant,           // Text | Structure (does the caller need tables/markdown?)
    pub budget: Option<Budget>,       // max GPU pages, latency ceiling — enforced mid-stream
    pub verify: VerifyPolicy,         // cross-tier corroboration (§5.3)
}

pub enum LevelPolicy {
    Fixed(Tier),                      // always exactly this tier
    Ceiling(Tier),                    // auto, never exceed (e.g. Ceiling(Ocr) → no VLM, no hallucination)
    Auto { min: Tier, max: Tier },    // escalate within [min, max] per page
}
```

- **Pinned** (`Fixed`) — caller knows what it wants.
- **Bounded** (`Ceiling`) — the reliability knob. A memory-critical caller sets
  `Ceiling(Ocr)` to forbid generative extraction entirely.
- **Auto** — escalate per page using §5.4 signals.

### 5.3 Result — provenance with *honest* confidence

The ask requires `confidence: f32` on every block. But the three tiers produce
**fundamentally different trust signals**, and **olmOCR emits no confidence at
all** (verified: no logprobs, no confidence field). A bare `f32` invents a number
for the generative tier and invites false trust. Make the *source* of trust a
first-class part of the type:

```rust
pub struct TieredPageResult {
    pub page_number: u32,
    pub blocks: Vec<TieredBlock>,
    pub plain_markdown: String,
    pub produced_by: Tier,
    pub escalations: Vec<Escalation>,   // audit trail: why we climbed
}

pub struct TieredBlock {
    pub kind: DocBlockKind,             // reuse core's enum
    pub content: String,
    pub bbox: Option<[f32; 4]>,
    pub reading_order: u32,
    pub produced_by: Tier,
    pub confidence: Confidence,         // honest about its source
}

pub enum Confidence {
    /// Deterministic extraction — text was not generated (Native, or read-only OCR pass).
    Deterministic,
    /// Real model probability (e.g. CTC softmax from PP-OCR recognition).
    Measured(f32),
    /// Derived heuristic for generative output — no native model probability exists.
    Derived { score: f32, signals: DerivedSignals },  // EOS/finish_reason, n-gram repetition, length, ...
}
```

> The companion crate defines its own result types because provenance, tier, and
> escalation are **router-level** concepts. Core's `DocExtractResult` stays about
> "what one model produced"; the router enriches it into "what the pipeline
> decided." Core's `DocBlock` needs no breaking change.

**The ladder is corroboration, not just escalation.** When `Auto` runs a lower
tier first (it usually does), there is deterministic `Native`/`Ocr` text for the
*same page* the VLM just processed. Diffing them catches the exact failure the
ask fears — olmOCR silently altering a number — far better than any confidence
float:

```rust
pub struct VerifyPolicy {
    pub cross_tier: bool,    // diff generative output against the deterministic tier
    pub numeric_guard: bool, // flag/quarantine numeric tokens that diverge
}
```

This directly serves the "don't poison the knowledge graph" constraint.

### 5.4 Routing — pluggable signals + failure-driven escalation

Per-page escalation is a small state machine driven by **swappable signal
functions** (a trait), so v1 heuristics can be replaced by a v2 layout
classifier without touching the router.

- **Native → Ocr (cheap, ~free — the native parse already knows):** no text
  layer; text coverage below threshold (char count vs page area); garble signal
  (CID-keyed fonts without `ToUnicode` → mojibake risk).
- **Ocr → Vlm (needs a layout signal):** `want: Structure` **and** the page has
  table/figure/multi-column layout.
  - *v1:* cheap heuristic (image-area ratio, column count, ruling-line density
    from PP-OCR boxes).
  - *v2:* a real layout classifier (PP-DocLayout-class). *Honest: robust
    Ocr→Vlm auto has a cost floor — detecting "this is a table page" is itself
    model work.*
- **Failure-driven (the ParseFixer pattern):** `Native` empty/garbage → `Ocr`;
  `want: Structure` and `Ocr` gives flat text where layout was expected → `Vlm`.
  Every escalation is logged into `escalations` so callers see why a page cost
  more.

### 5.5 Usage

```rust
// Explicit construction — power users bind each rung.
let docs = DocumentPipeline::builder(runtime.clone())   // Arc<ModelRuntime>
    .native(NativeTextExtractor::default())  // Tier::Native — pure-Rust, `pdf-input`
    .rasterizer(Rasterizer::default())       // hayro (pure-Rust)
    .ocr("ocr/ppocr-v6")                      // Tier::Ocr  → OcrModel
    .vlm("docext/olmocr-2")                   // Tier::Vlm  → DocumentExtractionModel
    .build()?;

let result = docs
    .extract(DocInput::Pdf { bytes, pages: None },
             DocExtractPolicy::auto_up_to(Tier::Vlm))
    .await?;

// Plug-and-play — sane defaults, hardware-adaptive, batteries-included catalog.
let result = uni_doc_extract::extract_document(runtime, pdf_bytes).await?;
```

---

## 6. Dependency decisions

- **Rasterizer:** `hayro` (pure-Rust; no FFI), §4. Benchmark
  hayro first.
- **PP-OCRv6 detection:** adopt **`oar-ocr`** (Apache 2.0, same `ort` runtime,
  implements DBNet detection + the accuracy-sensitive Vatti-unclip box
  postprocessing + reading order — the exact missing piece). Keep existing CTC
  recognition via its model-adapter system. Porting just the postprocessing is
  the fallback if the dependency is unwelcome. Models: start on PP-OCRv5 ONNX for
  stability, move to PP-OCRv6 (~34.5M medium tier) for the accuracy/speed gain.
- **olmOCR-2:** `allenai/olmOCR-2-7B-1025` as Qwen2.5-VL on mistral.rs, ISQ Q4K.
  Verified contract: **single page per request** (multi-image KV-cache bug
  #1786 is open + `--max-num-images` defaults to 1); **image-only, no anchor
  text** (olmOCR-2 dropped anchoring), longest dim **1288px**; first pass
  low-temp (~0.1) with temp-escalating retries + repetition/EOS guards. The
  existing `parse_olmocr_markdown` already handles output.
- **Out of scope (confirmed):** PaddleOCR-VL / dots.ocr / MinerU2.5 — custom
  vision towers, not runnable on mistral.rs without per-model Candle ports.
  (MinerU2.5 is also AGPL.)

---

## 7. Plug-and-play packaging (the consumer's primary goal)

1. **Zero-config entry:** `extract_document(runtime, pdf_bytes)` → `Auto` policy +
   a batteries-included default catalog. Consumers don't hand-author det/rec/vlm
   aliases.
2. **Hardware-adaptive default:** no GPU → `Auto` silently caps at
   `Ceiling(Ocr)` and **still closes the scanned-PDF gap on CPU-only deploys**
   (Native + Ocr are CPU-viable). The Vlm tier lights up only where a GPU exists.
   Graceful degradation, never a hard error.
3. **Feature flags mapped to deployment reality:** `hayro` (pure-Rust raster),
   `pdf-input` (native PDF text parse). The inference providers (OCR/VLM) are
   enabled on the `uni-xervo` dependency by the consuming app. A CPU microservice
   ships small; a GPU box opts into the heavy provider.

---

## 8. Phasing

1. **Benchmark `hayro` fidelity** + finalize the rasterizer feature split.
   *Gates everything image-based.*
2. **Companion skeleton + Native + Ocr.** Stand up `uni-doc-extract`; `pdf-input`
   native parse (Tier::Native); complete core `OcrModel` with detection (PP-OCRv6
   via oar-ocr). Router with `Auto { Native, Ocr }`. Ships the scanned-PDF fix,
   fully deterministic / reliability-safe.
3. **Vlm tier.** Revive core `DocumentExtractionModel::extract()` on mistral.rs
   (olmOCR-2) + `want: Structure` routing + cross-tier verification +
   derived-confidence. Extend router to `Tier::Vlm`.
4. **Layout-classifier upgrade** for robust Ocr→Vlm auto (replace the v1
   heuristic).

---

## 9. Non-goals (v1)

- PaddleOCR-VL / dots.ocr / MinerU2.5 (custom vision towers — separate ask).
- Chart/figure → SVG/code reconstruction.
- Region-level (sub-page) escalation — v1 is per-page; the per-block
  `produced_by` leaves room for region granularity later.
- Cross-page table/section reconstruction.
- Handwriting-specialized models.
- Streaming per-page results (batch `Vec<TieredPageResult>` in v1; async stream
  is a likely v2 refinement for large/long PDFs and mid-stream budget control).

---

## 10. Open decisions

1. **Companion crate name** — `uni-doc-extract` is a placeholder. (`uni-xervo-docs`?
   `uni-docling`?)
2. **`oar-ocr`: adopt as dependency vs. port its postprocessing.** Adopt is
   faster and lower-risk; port gives tighter control / fewer deps.
3. **Does core `DocBlock` gain an optional `confidence`?** Not required by this
   design (the companion's `TieredBlock` carries it), but cheap and symmetric
   with `OcrBlock`. Decide when revising `traits/docs.rs`.
4. **Default `want`** for the zero-config entry — `Text` (cheaper, never reaches
   Vlm on a text page) vs `Structure` (richer, more GPU). Lean `Text`.

---

## 11. Implementation status (as built)

The companion crate landed as **`uni-xervo-pdf`** (a workspace member;
`default-features` off on its `uni-xervo` dependency). The earlier names in this
doc (`uni-doc-extract`, `DocumentPipeline::builder`) are superseded by what
shipped below.

### Done and verified (no GPU / no models needed)

- **Family-aligned entry:** `PdfExt` extension trait →
  `runtime.pdf_extractor(PdfConfig) -> PdfExtractor` (concrete, `Arc`-clone),
  reading like `runtime.document_extractor(..)`. Graceful degradation: an
  unresolved tier alias is downgraded, not surfaced as an error.
- **Type system:** `Tier`, `PdfConfig`/`DocExtractPolicy`/`LevelPolicy`/
  `OutputWant`/`Budget`/`VerifyPolicy`, `DocInput` (the per-call A-vs-B enum),
  and provenance-bearing `TieredPageResult`/`TieredBlock`/`Confidence`
  (`Deterministic` / `Measured` / `Derived`).
- **Pure routing** (`router`): `plan_tier` + `escalate_on_failure`, the full
  **C1–C9** matrix as unit tests; the `Ceiling(Ocr)` reliability knob proven.
- **Cross-tier verification** (`verify`) + **confidence derivation**
  (`confidence`): numeric divergence (format-insensitive), repetition/EOS
  heuristics.
- **Pipeline** (`pipeline`): composes native/OCR/VLM with injected handles
  (testability), lazy rasterization, budget + hardware caps. Reliability/cost
  tests **R1, R5, R6, C1, C10, C11** pass on mocks.
- **Native tier** (`pdf-input`, `lopdf`): per-page text + coverage/garble
  signals, verified on a real generated PDF.
- **Rasterizer** (`hayro`, default): pure-Rust PDF→PNG, verified rendering a
  real PDF; `DocInput::Pdf` works end-to-end (real lopdf + real hayro + mock
  OCR), including the born-digital "no raster, no OCR" cost proof.
- **Core olmOCR-2 path:** revived `DocumentExtractionModel` on the
  `local/mistralrs` vision pipeline — no-anchor olmOCR prompt, one page per
  request, low-temp first pass; parses via the hoisted `crate::doc_parse`
  module. Verified with a mock vision generator (prompt + parser wiring, no
  GPU). The doc-VLM output parsers were hoisted to `crate::doc_parse` so both
  `local/onnx` and `local/mistralrs` share them; all 9 parser tests stay green.
- **Core preprocess:** generalized to non-square via `preprocess_batch_hw`; the
  OCR square-resize workaround removed.
- **Core OCR detection (DBNet), ported — zero new deps.** `OcrModel` is now a
  two-stage reader when a detector is configured (`det_onnx_path` option):
  detect → crop → recognize → reading order; otherwise it keeps today's
  whole-image behavior (backward compatible). The accuracy-sensitive
  postprocessing (binarize → iterative 4-connectivity flood fill → mean-prob
  score → axis-aligned unclip → TB-YX reading order) lives in
  `provider/local_onnx/det.rs` and is covered by **19 positive/negative unit
  tests** on synthetic probability maps (connectivity, threshold inclusivity,
  scoring/filtering, unclip math, scale-back, reading order, blank/degenerate
  edges, large-map no-overflow). Detection preprocessing (`preprocess_det` —
  resize to a multiple of 32 + scale factors) and option validation (`det_*`
  keys + numeric ranges, positive + negative) are tested too. We chose the
  **port** over `oar-ocr` (~301 transitive crates + a Rust-1.95 MSRV bump vs.
  core's 1.88) — reusing core's existing `ndarray`/`ort`/CTC recognition with
  **no new dependency and no FFI**; the only model-dependent part is the two
  `session.run` calls, wired through the existing `ort` machinery.

- **Core OCR detection — end-to-end verified against a real model.** The gated
  `EXPENSIVE` test `tests/local_onnx_ocr_detection_expensive_test.rs` runs the
  full detect → crop → recognize → order path with **PP-OCRv5** (DBNet server
  detector + English CRNN recognizer, `monkt/paddleocr-onnx`) on a generated
  two-line image: it detects the lines, recognizes "Hello World" / "OCR" /
  "Test 2026" in reading order with confidences ≥ 0.98, and returns an empty
  result for a blank page. This run also surfaced and fixed a real bug —
  PP-OCR's recognizer emits an already-softmaxed distribution, so the prior
  `softmax_at` double-softmaxed it and collapsed confidence to ~1/438;
  `class_confidence` now detects a normalized row and reads the probability
  directly (unit-tested both ways).

### Deferred — decisions for the team

- **Rotated / multi-column text:** v1 is axis-aligned boxes + TB-YX reading
  order (handles horizontal single-/simple-column documents). Skewed text wants
  min-area-rect + polygon unclip (the pure-Rust `clipper2-rust`) and true
  multi-column wants an XY-cut — both future refinements.
- A spurious low-confidence box can appear between dense lines; per-box score
  thresholding or light NMS could trim it. The confidence signal already flags
  it (≈0.53 vs ≥0.98 for real text), so downstream trust-gating handles it.
- **olmOCR temperature-escalating retries:** the core path does a single
  low-temp pass; the companion's confidence + cross-tier verify already flag bad
  output. Retry-on-repetition is a future refinement.
- **hayro fidelity benchmark** (the Phase-1 gate) and the full F1–F14 committed
  fixture corpus: the suite currently generates PDFs inline via `lopdf`; the
  benchmark + committed corpus remain to be added before betting latency-
  sensitive paths on hayro.
