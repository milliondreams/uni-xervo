# Design — Multimodal / NLP / VLM / ASR / OCR Extension for uni-xervo

| | |
|---|---|
| **Status** | Draft for internal review |
| **Repo** | `uni-xervo` (this repo) |
| **Companion document** | [`xervo-multimodal-api-proposal.md`](./xervo-multimodal-api-proposal.md) (consumer-authored requirements) |
| **Audience** | uni-xervo maintainers |
| **Date** | 2026-05-25 |
| **Scope** | Trait surface, `ModelTask` extension, `ModelRuntime` resolvers, instrumentation wrappers, handle-cache slots, and options-validation hooks. No provider implementations. The downstream `UniXervo` facade is out of scope — that lives in a separate consumer crate. |

---

## 0. TL;DR

The consumer proposal asks for seven new capabilities (image / audio / multimodal embedding, structured NLP, VLM document extraction, ASR, OCR). The proposal was written against a downstream facade that doesn't live in this repo. This document **re-grounds the design in uni-xervo's actual architecture**:

- The public surface is `ModelRuntime` (`src/runtime.rs`) plus the typed trait handles it returns (`src/traits.rs`). There is no `UniXervo` struct here.
- Adding a capability means: (1) a new trait, (2) a new `ModelTask` variant, (3) a new `HandleCache` slot, (4) a new `Instrumented*Model` wrapper, (5) a new `ModelRuntime` resolver, (6) per-provider `options_validation.rs` opt-in, (7) a `cfg(test)` mock.

Net additions: 7 trait types, 7 `ModelTask` variants, 7 runtime resolvers, 7 instrumentation wrappers, 7 mock impls, plus 4 shared types (`AudioInput`, `NlpResult`, `DocExtractResult`, `TranscribeResult`) and supporting structs.

No existing public signature changes. `RawTensorModel` is preserved verbatim — see §2.5.

---

## 2.5 First-class principle: the raw tensor / ONNX escape hatch stays

`RawTensorModel` and the `ModelTask::Raw` variant are **load-bearing public API** for uni-xervo and remain so indefinitely. This is not a transitional state.

**Why this matters in writing.** The managed traits in this design (`EmbeddingModel`, `NlpModel`, `OcrModel`, etc.) cover the common cases — models whose input/output shape is well-typed enough to wrap in a trait. The raw path covers everything else:

- Research models with novel input shapes (multi-input graphs, dynamic-axis batching, custom op sets).
- Customer-supplied ONNX graphs where uni-xervo can't reasonably know the schema.
- Models with structured outputs the managed traits don't model yet — a new tagging scheme, a custom segmentation head, a regression head — without forcing a uni-xervo release every time.
- Pre/post-processing pipelines that must live in caller code for licensing, observability, or domain-specific reasons.

**What this means concretely:**

1. `RawTensorModel` keeps its signature, its `ModelTask::Raw` variant, its `provider-onnx` feature gate, and its position in `ModelRuntime::raw_tensor_model(alias)`. No deprecation.
2. The new managed traits are **siblings**, not replacements. A consumer migrating from raw to managed (e.g., uniko's kniv-deberta cascade moving to `NlpModel`) does so by choice — there is no forcing function.
3. New `ModelTask` variants do **not** subsume `Raw`. An alias declared as `Raw` continues to resolve through the existing path, untouched by this design.
4. Tests for `RawTensorModel` (`tests/raw_tensor_model_test.rs`) and its instrumentation wrapper (`InstrumentedRawTensorModel`) are preserved verbatim. PR-1 adds tests; it does not modify the existing raw-path tests.
5. The `tracing` / `metrics` / circuit-breaker instrumentation around `InstrumentedRawTensorModel` already in `src/reliability.rs` continues to apply — raw-path users get the same managed-observability layer that managed-trait users get, just without the typed input/output.

**Customer contract:** any customer running today against `raw_tensor_model(alias)` continues to work after PR-1 with zero code changes. The escape hatch is a public commitment.

---

## 1. Grounding — current uni-xervo architecture

### 1.1 Verified surface

`src/lib.rs` exports four modules consumers depend on: `api`, `runtime`, `traits`, `error`. The `provider` module is a registry of feature-gated implementations.

**Trait layer** (`src/traits.rs`):

```rust
pub trait ModelProvider:       Send + Sync   // src/traits.rs:38
pub trait EmbeddingModel:      Send + Sync   // src/traits.rs:74
pub trait RerankerModel:       Send + Sync   // src/traits.rs:108
pub trait GeneratorModel:      Send + Sync   // src/traits.rs:280
pub trait RawTensorModel:      ...           // src/traits/raw_tensor_model.rs
```

Every concrete model trait has:
- A `Send + Sync` bound.
- One business `async fn` (`embed`, `rerank`, `generate`, …).
- An `async fn warmup(&self) -> Result<()>` default-no-op hook.
- For `EmbeddingModel`: `dimensions() -> u32` and `model_id() -> &str` accessors.

**Runtime layer** (`src/runtime.rs`):

```rust
struct HandleCache {                          // src/runtime.rs:31
    embeddings:        DashMap<String, Arc<dyn EmbeddingModel>>,
    rerankers:         DashMap<String, Arc<dyn RerankerModel>>,
    generators:        DashMap<String, Arc<dyn GeneratorModel>>,
    raw_tensor_models: DashMap<String, Arc<dyn RawTensorModel>>,
}

impl ModelRuntime {
    pub async fn embedding(&self, alias: &str)      -> Result<Arc<dyn EmbeddingModel>>;
    pub async fn reranker(&self, alias: &str)       -> Result<Arc<dyn RerankerModel>>;
    pub async fn generator(&self, alias: &str)      -> Result<Arc<dyn GeneratorModel>>;
    pub async fn raw_tensor_model(&self, alias: &str)-> Result<Arc<dyn RawTensorModel>>;
}
```

Each resolver: cache-check → spec-lookup → load-or-reuse → downcast `LoadedModelHandle` (`Arc<dyn Any + Send + Sync>`) to `Arc<dyn TraitName>` → wrap in `Instrumented*Model` → insert into `HandleCache`.

**Reliability layer** (`src/reliability.rs`):

`InstrumentedEmbeddingModel`, `InstrumentedRerankerModel`, `InstrumentedGeneratorModel`, `InstrumentedRawTensorModel` — each carries `{ inner, alias, provider_id, timeout, retry }` and bolts on circuit-breaker / retry / timeout / metrics around the inner trait method.

**Catalog layer** (`src/api.rs`):

```rust
pub enum ModelTask {                          // src/api.rs:10
    Embed,
    Rerank,
    Generate,
    Raw,
}
```

(Note: `Raw` is what backs `raw_tensor_model` today — four variants, not three as the consumer proposal stated.)

**Options validation** (`src/options_validation.rs`):

`validate_provider_options(provider_id, task, options) -> Result<()>` is called by `ModelRuntimeBuilder::build` and `ModelRuntime::register`. Each provider has a per-task validator. Adding new tasks means each provider must either opt in (recognize and validate) or implicitly reject by absence.

**Provider capability advertising**:

`ProviderCapabilities { supported_tasks: Vec<ModelTask> }` is returned by `ModelProvider::capabilities()`. The runtime cross-checks an alias's declared task against the resolved provider's advertised capabilities at register / load time.

### 1.2 What the consumer proposal got wrong about this repo

| Claim in proposal | Reality |
|---|---|
| Public facade lives at `crates/uni/src/api/xervo.rs` as `struct UniXervo` | No such file or struct in this repo. That facade lives in a downstream `uni` crate. |
| `ModelTask` is `Generate \| Embed \| Rerank` | Has 4 variants including `Raw`. |
| `#[non_exhaustive]` is the convention on `DataType` | `DataType` is a uni-db type, not uni-xervo. No `#[non_exhaustive]` precedent here. |
| New traits "automatically get managed observability" | Only if each trait ships its own `Instrumented*Model` wrapper. Not automatic. |
| Facade methods (`embed_image`, `transcribe`, …) | Out of scope for this repo. Consumer-side mechanical wrappers over the resolvers. |

This document scopes only the uni-xervo work.

---

## 2. Requirements re-stated (mapped from consumer proposal §2)

Functional requirements R1–R7 unchanged. Non-functional requirements reinterpreted for this repo:

| # | Requirement | This-repo translation |
|---|---|---|
| N1 | Pattern consistency | Mirror `EmbeddingModel` / `RerankerModel` / `GeneratorModel` shape exactly: `*Model` suffix, one business async fn, `warmup()` hook, `Send + Sync`. |
| N2 | One business method per trait | Honored. |
| N3 | Selective provider opt-in | Already supported via `ProviderCapabilities`. Existing providers gain new traits over time; missing traits return `CapabilityMismatch` / `ProviderCapabilityMissing`. |
| N4 | No breakage to existing public surface | No changes to existing trait method signatures, `ModelRuntime` resolver signatures, or `RawTensorModel`. `ModelTask` gains variants — mitigated by adding `#[non_exhaustive]` (new precedent here; document in CHANGELOG). |
| N5 | Async + Send + Sync | Honored. |
| N6 | Feature gating | Traits are unconditional; provider impls are gated. Matches existing pattern. |
| N7 | Reuse existing types | `ImageInput`, `ContentBlock`, `Message`, `TokenUsage` reused. New types added only where the existing shape doesn't fit. |
| N8 | Per-call cost reporting | Addressed via `EmbedResult` and `usage: Option<TokenUsage>` on new methods — without breaking existing `EmbeddingModel::embed`. See §6 for the decision. |

---

## 3. Naming conventions used in this design

Aligning with the existing codebase, departing from the consumer proposal where it diverged:

- **Trait names**: `*Model` suffix, capability-prefixed.
  - `ImageEmbeddingModel`, `AudioEmbeddingModel`, `MultimodalEmbeddingModel`
  - `NlpModel`, `OcrModel`
  - `DocumentExtractionModel` (proposal had `VlmExtractor`)
  - `TranscriptionModel` (proposal had `Transcriber`)
- **Trait method names**: verb only, no modality (mirrors `EmbeddingModel::embed`, `GeneratorModel::generate`).
  - `ImageEmbeddingModel::embed(Vec<ImageInput>)`
  - `AudioEmbeddingModel::embed(Vec<AudioInput>)`
  - `TranscriptionModel::transcribe(AudioInput, TranscribeOptions)`
  - `NlpModel::analyze(Vec<NlpRequest<'_>>)`
  - `DocumentExtractionModel::extract(Vec<ImageInput>, DocExtractOptions)`
  - `OcrModel::recognize(Vec<ImageInput>)`
- **Field naming**: match existing types.
  - `AudioInput::Bytes { data: Vec<u8>, media_type: String }` (matches `ImageInput::Bytes`).
- **Numeric types**: match existing usage.
  - `dimensions() -> u32` (matches `EmbeddingModel::dimensions`).

---

## 4. `ModelTask` extension

```rust
// src/api.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]                          // NEW — see §10.1
pub enum ModelTask {
    Embed,
    Rerank,
    Generate,
    Raw,
    // === New ===
    EmbedImage,
    EmbedAudio,
    EmbedMultimodal,
    Nlp,
    DocumentExtract,
    Transcribe,
    Ocr,
}
```

`#[non_exhaustive]` is added in the **same** PR to give downstream catalog-matching code a forward-compat path. Documented as a one-time hardening in CHANGELOG.

---

## 5. New traits

All traits live in `src/traits.rs` (or a `src/traits/multimodal.rs` submodule if file size becomes an issue; mechanical decision). Every trait follows the existing template: `async_trait`, `Send + Sync`, one business method, `warmup()` hook.

### 5.1 Image embedding

```rust
#[async_trait]
pub trait ImageEmbeddingModel: Send + Sync + Any {
    async fn embed(&self, images: Vec<ImageInput>) -> Result<EmbedResult>;
    fn dimensions(&self) -> u32;
    fn model_id(&self) -> &str;
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

Reuses `ImageInput` (`src/traits.rs:147`). Returns `EmbedResult` (§6) rather than bare `Vec<Vec<f32>>` — the new embed traits are designed forward; the existing `EmbeddingModel::embed` is left untouched.

### 5.2 Audio embedding

```rust
#[async_trait]
pub trait AudioEmbeddingModel: Send + Sync + Any {
    async fn embed(&self, audios: Vec<AudioInput>) -> Result<EmbedResult>;
    fn dimensions(&self) -> u32;
    fn model_id(&self) -> &str;
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

### 5.3 Multimodal embedding

```rust
#[async_trait]
pub trait MultimodalEmbeddingModel: Send + Sync + Any {
    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<EmbedResult>;
    fn dimensions(&self) -> u32;
    fn model_id(&self) -> &str;
    fn supported_modalities(&self) -> &[Modality];
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

### 5.4 Structured NLP

```rust
#[async_trait]
pub trait NlpModel: Send + Sync + Any {
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>) -> Result<Vec<NlpResult>>;
    fn supported_tasks(&self) -> NlpTasks;
    fn model_id(&self) -> &str;
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

#### 5.4.1 Canonical default model: `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`

The reference / default implementation of `NlpModel` is the published kniv-deberta cascade:

| Property | Value |
|---|---|
| HuggingFace ID | `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` |
| Heads | POS, NER, DEP, SRL, CLS (full `NlpTasks::ALL`) |
| Base encoder | DeBERTa-v3-xsmall (384d, 12 layers, 74.7M params) |
| ONNX artifact | `onnx/cascade.onnx` (FP32 300 MB / INT8 92 MB) |
| ONNX inputs | `input_ids[batch,seq]: i64`, `attention_mask[batch,seq]: i64`, `predicate_idx[batch]: i64` |
| ONNX outputs | 6 tensors: `pos`, `ner`, `arc`, `label`, `srl`, `cls` (`arc`+`label` jointly produce DEP) |
| Tagsets | POS: UD English EWT (17 classes). NER: OntoNotes (18 entity types). DEP: UD English EWT arc+relation. SRL: PropBank (42 classes). CLS: 8 dialog acts. |
| Language | English only |
| Max sequence length | 128 tokens (provider-side chunking responsibility — see below) |
| License | CC-BY-SA-4.0 |

This model maps cleanly onto the `NlpModel` trait shape — one forward pass produces every head — and is the existence proof the trait was designed around. The kniv cascade currently runs in uniko's `uniko-extract/src/nlp/mod.rs` via the `raw_tensor_model` escape hatch; this design promotes it to a first-class managed `NlpModel` impl.

**Catalog spec (canonical default alias):**

```json
{
  "alias": "nlp/default",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
  "revision": "main",
  "warmup": "background",
  "options": {
    "onnx_path": "onnx/cascade.onnx",
    "max_seq_len": 128
  }
}
```

A second alias `nlp/default-int8` pointing at the same repo with `onnx_path: "onnx/cascade-int8.onnx"` (or whatever the INT8 file name resolves to) is the recommended low-memory variant; both share dedup-keyed loading if revisions match.

#### 5.4.2 Provider-impl notes (informative, lands in PR-3)

The kniv ONNX graph imposes three impl-time decisions the trait does **not** surface to callers:

1. **SRL is per-predicate.** The `predicate_idx` input identifies one verb per forward pass; getting all SRL frames means running the forward N times, once per verb in the sentence. The provider impl:
   - First runs with `predicate_idx = 0` (sentinel "no SRL") to populate POS / NER / DEP / CLS.
   - Then, if `NlpTasks::SRL` is requested, identifies verbs from the POS output and re-runs once per verb with the appropriate `predicate_idx`, accumulating frames into `NlpResult::frames`.
   - This entire dance is invisible to the `NlpModel::analyze` caller — the trait contract stays at "ask for the heads you want, get them back populated."

2. **Max sequence length is 128 tokens.** Inputs longer than 128 must be chunked at sentence boundaries (or token-aware sliding windows for runaway sentences). The provider impl owns chunking + result-stitching, including translating per-chunk token offsets back to whole-document UTF-8 byte offsets in `NlpToken::{start, end}`. Token offset bookkeeping is the most error-prone part of the impl; cover it with golden-output tests in PR-3.

3. **The `cascade.onnx` graph has a strict opset / transformers-version provenance.** The model card pins `transformers==5.6.2` for re-export. uni-xervo doesn't re-export; the published `onnx/cascade.onnx` is consumed verbatim. Document this in the provider impl's doc comment so anyone tempted to "regenerate" the graph from PyTorch weights knows the exact pinning.

None of this changes PR-1. The trait, types, and `ModelTask::Nlp` variant land here; the kniv impl lands in PR-3 (now promoted from "optional reference impl" to "ships with the canonical `nlp/default` catalog entry").

### 5.5 Document extraction

```rust
#[async_trait]
pub trait DocumentExtractionModel: Send + Sync + Any {
    async fn extract(
        &self,
        pages: Vec<ImageInput>,
        options: DocExtractOptions,
    ) -> Result<Vec<DocExtractResult>>;
    fn model_id(&self) -> &str;
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

### 5.6 Transcription

```rust
#[async_trait]
pub trait TranscriptionModel: Send + Sync + Any {
    /// Transcribe a single audio stream. The canonical primitive — every
    /// provider must implement this.
    async fn transcribe(
        &self,
        audio: AudioInput,
        options: TranscribeOptions,
    ) -> Result<TranscribeResult>;

    /// Transcribe a batch of audio inputs. Providers that can genuinely
    /// batch (e.g. whisper-large server pools, future GPU-batched ASR)
    /// override this; the default fans out to `transcribe` concurrently
    /// via `futures::future::try_join_all`, which is what most providers
    /// (whisper.cpp, single-stream remote APIs) want anyway.
    ///
    /// Returns results in the same order as `audios`.
    async fn transcribe_many(
        &self,
        audios: Vec<AudioInput>,
        options: TranscribeOptions,
    ) -> Result<Vec<TranscribeResult>> {
        use futures::future::try_join_all;
        let opts = &options;
        try_join_all(
            audios.into_iter().map(|a| self.transcribe(a, opts.clone()))
        ).await
    }

    fn supported_languages(&self) -> &[String];
    fn model_id(&self) -> &str;
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

Two-method shape per the ingest-batching requirement: `transcribe` is the primitive, `transcribe_many` is what ingest pipelines call. The default fan-out gives every provider correct batch semantics on day one; provider-internal batching is an opt-in override. `TranscribeOptions` is shared across the batch — per-item options can be added later as a `transcribe_many_with_options(Vec<(AudioInput, TranscribeOptions)>)` if it turns out to be needed, without breaking either current method.

The runtime resolver returns the wrapped trait, so callers get retry / timeout / circuit-breaker on both methods. The `Instrumented<TranscriptionModel>` wrapper (§8.3) instruments both `transcribe` and `transcribe_many` — the latter is a batched operation, so per-batch timeout / retry semantics apply (the fan-out happens inside the inner impl, not above the instrumentation).

### 5.7 OCR

```rust
#[async_trait]
pub trait OcrModel: Send + Sync + Any {
    async fn recognize(&self, images: Vec<ImageInput>) -> Result<Vec<OcrResult>>;
    fn model_id(&self) -> &str;
    async fn warmup(&self) -> Result<()> { Ok(()) }
}
```

---

## 6. Cost reporting — `EmbedResult` (decided)

The consumer proposal raised this in §9.1 with three options. Decision for this repo:

**Adopt option C** (only new embed traits return `EmbedResult`; existing `EmbeddingModel::embed` keeps its `Vec<Vec<f32>>` signature).

Rationale:
- uni-xervo is at 0.12.0, published to crates.io, has external users.
- A breaking signature change to the most-used method needs more justification than internal convenience.
- The new embed traits are net-new; designing them with cost reporting from day one costs nothing.
- A future 0.x-to-1.0 cycle can unify by promoting `EmbeddingModel::embed` to return `EmbedResult`.

```rust
#[derive(Debug, Clone)]
pub struct EmbedResult {
    pub vectors: Vec<Vec<f32>>,
    /// `Some` for remote providers that report usage; `None` for local.
    pub usage: Option<TokenUsage>,
}
```

`TokenUsage` (`src/traits.rs:264`) is reused verbatim.

---

## 7. New shared types

### 7.1 Audio input

```rust
/// An audio input to a transcription or audio embedding model.
///
/// No `Path` variant: providers in uni-xervo do not perform file I/O.
/// Callers read bytes themselves. This mirrors `ImageInput`, which also
/// has no `Path` variant.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Raw container bytes (WAV, MP3, FLAC, …). Provider decides decoding
    /// based on `media_type`.
    Bytes { data: Vec<u8>, media_type: String },
    /// Pre-decoded PCM. Provider skips decode.
    Pcm { sample_rate: u32, channels: u16, samples: Vec<f32> },
}
```

Departure from the consumer proposal: dropped `Path` variant (providers stay I/O-free), renamed `mime` → `media_type` (consistency with `ImageInput::Bytes`).

### 7.2 Multimodal input

`ContentBlock` today is `Text | Image`. The multimodal embedder needs to carry audio blocks too. Two paths:

- **Path A**: Add `ContentBlock::Audio(AudioInput)`. Affects every `GeneratorModel` impl — they would receive `Audio` blocks they currently cannot handle. Requires per-provider "reject Audio block" handling.
- **Path B**: Introduce a separate `MultimodalBlock` type for the embedding path only. Generators stay on `ContentBlock`; embedders use `MultimodalBlock`.

**Decision: Path B.** Cleaner separation, no cross-cutting impact on generators. `MultimodalBlock` shares the same shape as `ContentBlock` plus audio — semantically distinct from a conversation message.

```rust
#[derive(Debug, Clone)]
pub enum MultimodalBlock {
    Text(String),
    Image(ImageInput),
    Audio(AudioInput),
}

#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub blocks: Vec<MultimodalBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}
```

If a future generator needs audio input (e.g., Gemini multimodal chat), the same `MultimodalBlock` can be used as input to a new generator trait without disturbing today's `GeneratorModel`. That cost is deferred and isolated.

### 7.3 NLP types

```rust
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NlpTasks: u32 {
        const POS = 1 << 0;
        const NER = 1 << 1;
        const DEP = 1 << 2;
        const SRL = 1 << 3;
        const CLS = 1 << 4;
        const ALL = Self::POS.bits() | Self::NER.bits() | Self::DEP.bits()
                  | Self::SRL.bits() | Self::CLS.bits();
    }
}

pub struct NlpRequest<'a> { pub text: &'a str, pub tasks: NlpTasks }

pub struct NlpResult {
    pub tokens: Vec<NlpToken>,
    pub sentences: Vec<NlpSentence>,
    pub frames: Vec<SrlFrame>,
    pub speech_acts: Vec<SpeechAct>,
}

pub struct NlpToken {
    pub text: String,
    pub start: usize, pub end: usize,        // UTF-8 byte offsets, [start, end)
    pub pos: Option<String>,                 // Universal Dependencies tagset
    pub ner: Option<String>,
    pub dep: Option<DepLink>,
}

pub struct NlpSentence { pub token_range: (usize, usize), pub start: usize, pub end: usize }
pub struct DepLink   { pub head: usize, pub relation: String }
pub struct SrlFrame  { pub predicate_token: usize, pub predicate_sense: Option<String>, pub roles: Vec<SrlRole> }
pub struct SrlRole   { pub span: (usize, usize), pub label: String }
pub struct SpeechAct { pub sentence_index: usize, pub label: String, pub confidence: f32 }
```

`bitflags` is a new dep in `uni-xervo` but minimal-weight (~5 KB compiled). If the team prefers zero new deps, a plain `struct NlpTasks { pub pos: bool, pub ner: bool, … }` is a drop-in replacement with no semantic change.

### 7.4 Document extraction types

```rust
pub struct DocExtractOptions {
    pub output: DocOutputFormat,
    pub include_tables: bool,
    pub include_formulas: bool,
    pub include_bboxes: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum DocOutputFormat { Markdown, Json, Html }

pub struct DocExtractResult {
    pub blocks: Vec<DocBlock>,
    pub plain_markdown: String,
}

pub struct DocBlock {
    pub kind: DocBlockKind,
    pub content: String,             // MD for text/heading/table; LaTeX for formula
    pub bbox: Option<[f32; 4]>,      // x0, y0, x1, y1 in page coords
    pub reading_order: u32,
}

#[non_exhaustive]
pub enum DocBlockKind { Text, Heading, List, Table, Figure, Formula, Caption, Footer, Header }
```

`DocBlockKind` ships `#[non_exhaustive]` from day one — VLM taxonomies evolve, and this enum is consumer-pattern-matched.

### 7.5 Transcription types

```rust
pub struct TranscribeOptions {
    pub language: Option<String>,             // None → auto-detect
    pub word_timestamps: bool,
    pub diarize: bool,
    pub initial_prompt: Option<String>,       // whisper.cpp biasing
}

pub struct TranscribeResult {
    pub language: String,
    pub segments: Vec<TranscribeSegment>,
}

pub struct TranscribeSegment {
    pub start_ms: u64, pub end_ms: u64,
    pub text: String,
    pub speaker: Option<String>,              // iff diarize=true
    pub words: Vec<TranscribeWord>,           // iff word_timestamps=true
}

pub struct TranscribeWord {
    pub start_ms: u64, pub end_ms: u64,
    pub text: String,
    pub confidence: Option<f32>,
}
```

### 7.6 OCR types

```rust
pub struct OcrResult {
    pub blocks: Vec<OcrBlock>,
    pub plain_text: String,
}

pub struct OcrBlock {
    pub text: String,
    pub bbox: [f32; 4],
    pub confidence: f32,
}
```

---

## 8. Runtime integration

### 8.1 `HandleCache` extension

```rust
// src/runtime.rs

#[derive(Default)]
struct HandleCache {
    embeddings:        DashMap<String, Arc<dyn EmbeddingModel>>,
    rerankers:         DashMap<String, Arc<dyn RerankerModel>>,
    generators:        DashMap<String, Arc<dyn GeneratorModel>>,
    raw_tensor_models: DashMap<String, Arc<dyn RawTensorModel>>,
    // === New ===
    image_embedders:       DashMap<String, Arc<dyn ImageEmbeddingModel>>,
    audio_embedders:       DashMap<String, Arc<dyn AudioEmbeddingModel>>,
    multimodal_embedders:  DashMap<String, Arc<dyn MultimodalEmbeddingModel>>,
    nlp_models:            DashMap<String, Arc<dyn NlpModel>>,
    doc_extractors:        DashMap<String, Arc<dyn DocumentExtractionModel>>,
    transcribers:          DashMap<String, Arc<dyn TranscriptionModel>>,
    ocr_models:            DashMap<String, Arc<dyn OcrModel>>,
}
```

### 8.2 `ModelRuntime` resolvers

Seven new resolver methods, each cloning the existing template at `src/runtime.rs:148-178` (the `embedding` method). Each:

1. Fast-path cache hit on `handle_cache.<slot>`.
2. `lookup_spec(alias).await?`.
3. `resolve_and_load_internal(&spec).await?` → returns `LoadedModelHandle`.
4. `downcast_ref::<Arc<dyn TraitName>>()`.
5. Wrap in `Instrumented<TraitName>` (see §8.3).
6. Insert into cache, return.
7. Downcast failure → `RuntimeError::CapabilityMismatch` (or `ProviderCapabilityMissing` if we want to preserve `raw_tensor_model`'s richer error variant — bikeshed).

```rust
impl ModelRuntime {
    pub async fn image_embedder(&self, alias: &str)       -> Result<Arc<dyn ImageEmbeddingModel>>;
    pub async fn audio_embedder(&self, alias: &str)       -> Result<Arc<dyn AudioEmbeddingModel>>;
    pub async fn multimodal_embedder(&self, alias: &str)  -> Result<Arc<dyn MultimodalEmbeddingModel>>;
    pub async fn nlp_model(&self, alias: &str)            -> Result<Arc<dyn NlpModel>>;
    pub async fn document_extractor(&self, alias: &str)   -> Result<Arc<dyn DocumentExtractionModel>>;
    pub async fn transcriber(&self, alias: &str)          -> Result<Arc<dyn TranscriptionModel>>;
    pub async fn ocr_model(&self, alias: &str)            -> Result<Arc<dyn OcrModel>>;
}
```

### 8.3 Instrumentation wrappers

Seven new `Instrumented*Model` structs in `src/reliability.rs`, each modeled on the existing four. Each holds `{ inner, alias, provider_id, timeout, retry }` and wraps the business method with:

- `tokio::time::timeout` (if `timeout.is_some()`)
- Retry-with-backoff loop (if `retry.is_some()`) — via existing helper logic
- Circuit-breaker call (existing `CircuitBreakerWrapper`)
- `tracing` span + `metrics` emission keyed on `(alias, provider_id, task)`

The `Instrumented*Model::warmup()` impl forwards to inner with a `tracing` span — matches existing pattern.

This is **the** central piece of work for PR-1 — without these wrappers, the headline benefit ("managed dispatch with retry / timeout / metrics, unlike `raw_tensor_model`") doesn't materialize.

### 8.4 Provider capability matching

`ProviderCapabilities::supported_tasks` already enforces the contract. A provider only ever serves aliases whose `ModelTask` it advertises. Adding new tasks doesn't require any change here — existing providers are silently unsupported for new tasks (their `capabilities()` returns the old set). Each provider extends incrementally.

Catalog validation flow (already in place at `src/runtime.rs:76`):
- `ModelRuntime::register(spec)` → `validate_provider_options(provider_id, task, options)` → if the provider doesn't recognize the task, return `Config` error.
- At load time, the runtime downcasts `LoadedModelHandle` to the expected trait. If the provider's load result doesn't implement the trait, `CapabilityMismatch` error fires.

### 8.5 Options validation

`src/options_validation.rs` dispatches per `(provider_id, task)`. Adding 7 new tasks means each provider's validator function must explicitly accept or reject them. Strategy:

- For every existing provider, add an explicit "task X not supported by this provider" match arm. Returns `Config` error at register time.
- This is mechanical but mandatory — without it, a misconfigured catalog could pass validation and fail mysteriously at load time.

Concrete: `validate_provider_options` becomes the dispatch funnel; each provider's validator pattern-matches on `task` against its known set.

---

## 9. Testing strategy

### 9.1 Trait dyn-safety and object construction

One test per trait that constructs `Arc<dyn TraitName>` from a no-op `cfg(test)` mock impl. Confirms object safety, `Send + Sync`, and `Any` blanket impl all line up.

### 9.2 `ModelTask` round-trip

JSON serialize/deserialize every new variant. Confirms catalog wire-format stability.

### 9.3 `#[non_exhaustive]` regression

Add a test that pattern-matches on `ModelTask` with a wildcard arm and confirms compilation. Documents the convention.

### 9.4 Resolver error paths

For each new resolver:
- `alias_not_found` (catalog missing) → `AliasNotFound`.
- Provider loads a model that doesn't impl the expected trait → `CapabilityMismatch` / `ProviderCapabilityMissing`.
- `handle_cache` hit on second call returns the same `Arc` (pointer equality via `Arc::ptr_eq`).

### 9.5 Instrumentation parity

For each new `Instrumented*Model`:
- Timeout path: inner sleeps > timeout → `Timeout` error.
- Retry path: inner fails N-1 times, succeeds Nth → returns Ok with N attempts logged.
- Circuit-breaker path: N consecutive failures opens the breaker → next call returns `Unavailable`.

These mirror existing `tests/reliability_test.rs`.

### 9.6 Options validation per provider

For every (provider, new-task) pair, assert `validate_provider_options` returns a `Config` error until the provider opts in. Prevents silent misconfiguration in PR-1.

### 9.7 Mock provider extension

`src/mock.rs` gains seven mock trait impls (`MockImageEmbedder`, etc.) for the test mock provider. Used by the resolver and instrumentation tests above.

---

## 10. Backwards compatibility

| Change | Breaking? | Mitigation |
|---|---|---|
| New traits in `uni_xervo::traits` | No | Additive. |
| New `ModelTask` variants | Soft yes (exhaustive match against `ModelTask` in downstream code stops compiling) | Land `#[non_exhaustive]` in the same PR. Document in CHANGELOG. |
| `#[non_exhaustive]` on `ModelTask` | Yes (downstream `match` without wildcard breaks) | Major-version-aware change but uni-xervo is pre-1.0; acceptable for 0.13. Call out in release notes. |
| New `HandleCache` fields | No | Internal struct. |
| New `ModelRuntime` resolver methods | No | Additive. |
| New `Instrumented*Model` structs | No | Internal. |
| `EmbedResult` for new embed traits | No | Net-new return type on net-new traits. Existing `EmbeddingModel::embed` unchanged. |
| Existing public surface (`embed`, `rerank`, `generate`, `raw_tensor_model`, etc.) | Unchanged | None needed. |
| `bitflags` new dep | No | Cheap; replaceable with plain struct if rejected. |

---

## 11. Open decisions

### 11.1 `#[non_exhaustive]` on `ModelTask` — yes or no?

**Recommendation: yes**, landing with this PR. Pre-1.0 is the right moment; deferring to 1.0 forces a louder release. Downstream remediation is trivial (add `_ =>` arm).

### 11.2 `bitflags` dependency

**Recommendation: yes.** Crate is mature, ~1.4k stars, used by countless ecosystem crates including `tokio`. Bench impact zero. If rejected, the fallback is mechanical (struct of bools).

### 11.3 `MultimodalBlock` vs extending `ContentBlock`

**Recommendation: separate `MultimodalBlock`.** Decided in §7.2. Generators stay unchanged. If a future multimodal generator emerges (e.g., Gemini chat with audio inputs), introduce a new generator trait that takes `MultimodalBlock` rather than retrofit the existing one.

### 11.4 `traits.rs` file growth

Adding 7 traits + ~25 supporting types nearly doubles `src/traits.rs` from 293 lines to ~600+. Split into:

```
src/traits.rs                  // existing core (Embedder, Reranker, Generator, ...) + re-exports
src/traits/raw_tensor_model.rs // existing
src/traits/multimodal.rs       // NEW — image/audio/multimodal embedders
src/traits/nlp.rs              // NEW — NlpModel + types
src/traits/docs.rs             // NEW — DocumentExtractionModel, OcrModel + types
src/traits/asr.rs              // NEW — TranscriptionModel + types
```

Re-export every new symbol at `crate::traits::*` to keep import sites flat. **Recommendation: do the split in PR-1** — easier to do now than after the impls land.

### 11.5 Error variant strategy

Existing errors used by current resolvers: `CapabilityMismatch(String)` (used by `embedding`/`reranker`/`generator`) and `ProviderCapabilityMissing { alias, provider_id, capability }` (used by `raw_tensor_model`). The latter is strictly richer. **Recommendation:** new resolvers use `ProviderCapabilityMissing`, and a follow-up PR migrates the legacy three to it for consistency. Not part of this PR.

### 11.6 `dimensions()` on multimodal embedder — single value or per-modality?

A multimodal embedder produces one vector per input regardless of modality. Single `dimensions() -> u32` is correct. Confirmed in §5.3.

---

## 12. PR breakdown

### PR-1 (this design — uni-xervo only):

1. `src/api.rs`: add 7 `ModelTask` variants, add `#[non_exhaustive]`.
2. `src/traits.rs` + new submodules: 7 new traits, ~25 new types, `EmbedResult`.
3. `src/runtime.rs`: 7 new `HandleCache` slots, 7 new resolver methods.
4. `src/reliability.rs`: 7 new `Instrumented*Model` wrappers.
5. `src/options_validation.rs`: per-provider task acceptance/rejection arms for new tasks (all start as "unsupported" since no provider impls land here).
6. `src/mock.rs`: 7 mock impls + extended mock provider capabilities for tests.
7. `tests/`: new dyn-safety, resolver, instrumentation, options-validation tests (mirrors existing `embedding_model_test.rs`, `reranker_model_test.rs`, `reliability_test.rs`).
8. CHANGELOG entry covering `#[non_exhaustive]` on `ModelTask` and the new symbols.

### Subsequent PRs (out of scope for this design):

- PR-2: `local/onnx` extends `ProviderCapabilities` + implements `ImageEmbeddingModel` (SigLIP-2) and `OcrModel`.
- PR-3: `local/onnx` implements `NlpModel` against the canonical default `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` (see §5.4.1). Ships the `nlp/default` catalog entry. uniko's `uniko-extract/src/nlp/mod.rs` migrates from `raw_tensor_model` to `nlp_model` in a co-landing consumer PR.
- PR-4: New `local/whisper-cpp` provider implementing `TranscriptionModel`.
- PR-5: `local/onnx` or `local/candle` implements `DocumentExtractionModel`.
- PR-6: `remote/cohere` and `remote/gemini` implement `MultimodalEmbeddingModel`.

Each is independent.

---

## 13. Risks specific to this repo

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| `traits.rs` becomes unwieldy | High | Low | Submodule split in PR-1 (§11.4). |
| `#[non_exhaustive]` breaks downstream pre-1.0 code | Medium | Low | Trivial remediation; CHANGELOG. |
| Adding 7 `Instrumented*` wrappers explodes `reliability.rs` | Medium | Low | Consider a generic `Instrumented<T>` helper in a follow-up; for now duplicate-and-paste matches the existing pattern, which is the right local convention. |
| `bitflags` dep rejected | Low | Low | Plain struct fallback. |
| Capability advertising drift | Low | Medium | Test in §9.6 catches misconfiguration at register time. |
| `EmbedResult` divergence from existing `embed` confuses users | Low | Low | Document at §6 in release notes; plan unification at 1.0. |

---

## 13.1 Dependency note

`transcribe_many`'s default impl uses `futures::future::try_join_all`. uni-xervo already pulls `futures` transitively via `tokio` and `async_trait`, but if `futures` isn't a direct dep yet, add `futures = "0.3"` to `Cargo.toml` in PR-1. Cheap, ubiquitous, no concern.

---

## 14. Mapping consumer requirements → this design

| Consumer §2 req | Where addressed |
|---|---|
| R1 Image embedding | §5.1 `ImageEmbeddingModel` |
| R2 Audio embedding (bytes/PCM/path) | §5.2 + §7.1 (path dropped — see §7.1 rationale) |
| R3 Mixed-modality embedding | §5.3 + §7.2 (new `MultimodalBlock`) |
| R4 Structured NLP cascade | §5.4 + §7.3 |
| R5 VLM document extraction | §5.5 + §7.4 |
| R6 Speech-to-text | §5.6 (single + batch via `transcribe_many`) + §7.5 |
| (new) Ingest batching for ASR | §5.6 `transcribe_many` with default fan-out |
| (new) Preserve raw ONNX / tensor escape hatch | §2.5 first-class principle |
| R7 OCR | §5.7 + §7.6 |
| N1 Pattern consistency | §3 naming, §5 trait template, §8.2 resolver template |
| N2 One-method traits | §5 |
| N3 Selective opt-in | §8.4 |
| N4 Backwards compat | §10 |
| N5 Async + Send + Sync | §5 |
| N6 Feature gating | Inherited; impls in follow-up PRs |
| N7 Reuse existing types | `ImageInput`, `TokenUsage`, `Message` reused; new types in §7 only where needed |
| N8 Per-call cost reporting | §6 `EmbedResult` |

---

## Appendix A — Departures from the consumer proposal

| Consumer proposal | This design | Reason |
|---|---|---|
| Trait names `*Embedder`, `Transcriber`, `VlmExtractor` | `*EmbeddingModel`, `TranscriptionModel`, `DocumentExtractionModel` | Match existing `*Model` suffix. |
| Trait methods `embed_images`, `embed_audios`, `recognize` | `embed`, `embed`, `recognize` | Match existing verb-only convention. |
| `AudioInput::Path` variant | Dropped | Providers stay I/O-free; parity with `ImageInput`. |
| `AudioInput::Bytes { mime, … }` | `AudioInput::Bytes { data, media_type }` | Parity with `ImageInput::Bytes`. |
| `dimensions() -> usize` | `dimensions() -> u32` | Parity with `EmbeddingModel::dimensions`. |
| `MultimodalInput` reuses `ContentBlock` (implies audio extension) | Separate `MultimodalBlock` | Don't disturb `GeneratorModel`. |
| Recommendation 9.1 = A (break `embed`) | Option C (keep `embed`, new traits return `EmbedResult`) | Crates.io users; no signature break warranted. |
| `ModelTask` is 3 variants today | Acknowledged 4 (incl. `Raw`) | Factual correction. |
| `#[non_exhaustive]` precedent on `DataType` | Acknowledged as new in this repo | Factual correction. |
| Facade methods `embed_image`, `transcribe`, etc. | Out of scope (downstream `uni` crate) | Repo boundary. |

## Appendix B — Files touched in PR-1

```
src/api.rs                          # ModelTask variants + non_exhaustive
src/traits.rs                       # Re-exports + EmbedResult
src/traits/multimodal.rs            # NEW
src/traits/nlp.rs                   # NEW
src/traits/docs.rs                  # NEW
src/traits/asr.rs                   # NEW
src/runtime.rs                      # HandleCache + 7 resolvers
src/reliability.rs                  # 7 Instrumented*Model wrappers
src/options_validation.rs           # per-provider task arms
src/mock.rs                         # 7 mock impls
Cargo.toml                          # bitflags dep
CHANGELOG.md                        # non_exhaustive + new symbols
tests/multimodal_traits_test.rs     # NEW — §9.1, §9.4
tests/multimodal_instrumentation_test.rs  # NEW — §9.5
tests/options_validation_multimodal_test.rs  # NEW — §9.6
```

No new crate dependencies beyond `bitflags`. No new feature flags in PR-1 (provider-specific features arrive with their impls).
