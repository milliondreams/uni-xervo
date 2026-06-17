# Multimodal trait surface

uni-xervo 0.13.0 adds seven new traits alongside the original
`EmbeddingModel` / `RerankerModel` / `GeneratorModel` / `RawTensorModel`
quartet. Each new trait keeps the same one-business-method-per-trait,
`Send + Sync`, `async_trait` shape — and gains a matching
`ModelRuntime` resolver, instrumented wrapper (timeout + retry +
metrics), and `ModelTask` variant.

## The seven new traits

| Trait | Method | Resolver | `ModelTask` |
| --- | --- | --- | --- |
| `ImageEmbeddingModel` | `embed(Vec<ImageInput>) -> EmbedResult` | `runtime.image_embedder(alias)` | `EmbedImage` |
| `AudioEmbeddingModel` | `embed(Vec<AudioInput>) -> EmbedResult` | `runtime.audio_embedder(alias)` | `EmbedAudio` |
| `MultimodalEmbeddingModel` | `embed(Vec<MultimodalInput>) -> EmbedResult` | `runtime.multimodal_embedder(alias)` | `EmbedMultimodal` |
| [`NlpModel`](nlp.md) | `analyze(Vec<NlpRequest>) -> Vec<NlpResult>` | `runtime.nlp_model(alias)` | `Nlp` |
| `DocumentExtractionModel` | `extract(Vec<ImageInput>, DocExtractOptions) -> Vec<DocExtractResult>` | `runtime.document_extractor(alias)` | `DocumentExtract` |
| `TranscriptionModel` | `transcribe(AudioInput, TranscribeOptions) -> TranscribeResult` + `transcribe_many(...)` | `runtime.transcriber(alias)` | `Transcribe` |
| `OcrModel` | `recognize(Vec<ImageInput>) -> Vec<OcrResult>` | `runtime.ocr_model(alias)` | `Ocr` |

`ModelTask` is now `#[non_exhaustive]` — downstream pattern matches must
add a wildcard arm.

## Result types

### `EmbedResult`

New embed traits return `EmbedResult { vectors, usage: Option<TokenUsage> }`
rather than the bare `Vec<Vec<f32>>` of `EmbeddingModel::embed`. Remote
providers (Cohere, Gemini) populate `usage` when their APIs report it;
local providers leave it `None`. The existing `EmbeddingModel::embed`
signature is unchanged for backwards compatibility.

### `NlpResult`

`NlpModel::analyze` returns one `NlpResult` per request, carrying per-token
POS / NER / DEP, sentence boundaries, SRL frames, merged named-entity spans,
and dialog-act classifications. See the dedicated
[Structured NLP guide](nlp.md) for the full result-type reference, a worked
example, and the 0.14.0 migration notes.

### `DocExtractResult`

Structured document blocks with reading order, optional bounding boxes,
and a concatenated `plain_markdown` field:

```rust
pub struct DocBlock {
    pub kind: DocBlockKind,  // Text | Heading | List | Table | Figure | Formula | …
    pub content: String,
    pub bbox: Option<[f32; 4]>,
    pub reading_order: u32,
}
```

### `TranscribeResult`

```rust
pub struct TranscribeResult {
    pub language: String,
    pub segments: Vec<TranscribeSegment>,
}
```

Each segment has `start_ms` / `end_ms` / `text` / optional `speaker` /
optional `words`. Word-level timestamps populate when
`TranscribeOptions::word_timestamps = true`.

### `OcrResult`

```rust
pub struct OcrResult {
    pub blocks: Vec<OcrBlock>,  // text + bbox + confidence
    pub plain_text: String,
}
```

## Inputs

### `AudioInput`

```rust
pub enum AudioInput {
    Bytes { data: Vec<u8>, media_type: String },
    Pcm { sample_rate: u32, channels: u16, samples: Vec<f32> },
}
```

No `Path` variant — providers stay I/O-free; callers fetch files
themselves and pass bytes.

### `MultimodalBlock` + `MultimodalInput`

```rust
pub enum MultimodalBlock {
    Text(String),
    Image(ImageInput),
    Audio(AudioInput),
}
pub struct MultimodalInput {
    pub blocks: Vec<MultimodalBlock>,
}
```

Distinct from `ContentBlock` (used by `GeneratorModel`) so generators
stay unchanged.

### `NlpRequest`

`NlpModel::analyze` takes a batch of `NlpRequest { text, tasks }`, where `tasks`
is an `NlpTasks` bitflag selecting which heads to populate. See the
[Structured NLP guide](nlp.md) for details.

## Built-in provider coverage

Today's provider matrix for the new traits:

| Provider | Image embed | Audio embed | Multimodal embed | NLP | Doc extract | Transcribe | OCR |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `local/onnx` | ✓ | | | ✓ | scaffold | | ✓ |
| `local/mistralrs` | | | | | ✓ | | |
| `remote/cohere` | | | ✓ | | | | |
| `remote/gemini` | | | ✓ | | | | |
| `local/whisper-cpp` | | | | | | ✓ | |

`local/mistralrs` **Doc extract** is the live olmOCR-2 path on its vision
pipeline — see [`local/mistralrs` → Document extraction](../reference/providers/mistralrs.md#document-extraction-olmocr-2).

`local/onnx` **OCR** supports an optional DBNet detection stage for full-page
detect→recognize (set `det_onnx_path`); without it, OCR is single-stage
recognition — see [`local/onnx` → OCR-only keys](../reference/providers/onnx.md#ocr-only-keys-task-ocr).

`scaffold` (here, `local/onnx` **Doc extract**) means catalog wiring + options
validation are production-ready, but the inference path returns
`RuntimeError::Unavailable` until an upstream prerequisite ships (a canonical
ONNX export of Granite-Docling / MinerU / olmOCR). The reusable building blocks
(`provider::local_onnx::autoreg::greedy_decode`, the shared `doc_parse` DocTags
/ MinerU / olmOCR output parsers) are tested and available — and are exactly
what the live olmOCR-2 path on `local/mistralrs` reuses.

## Instrumentation

Each new resolver wraps the loaded handle in an `Instrumented*Model`
adapter — same shape as the existing wrappers around `EmbeddingModel`,
`RerankerModel`, etc. The wrapper applies:

- **Timeout** — per-call deadline from `ModelAliasSpec::timeout`. A
  hit surfaces as `RuntimeError::Timeout`.
- **Retry** — exponential backoff on retryable errors
  (`RateLimited` / `Timeout` / `Unavailable`) up to
  `RetryConfig::max_attempts`.
- **Metrics** — `model_inference.duration_seconds` (histogram) and
  `model_inference.total` (counter), labeled with `alias` / `task` /
  `provider` / `status`.

For `TranscriptionModel`, both `transcribe` and `transcribe_many` are
instrumented; the batched timeout applies batch-wide, not per-item.

## Migration: `ModelTask` is `#[non_exhaustive]`

The 0.13.0 trait surface added seven `ModelTask` variants. Existing
exhaustive matches against `ModelTask` no longer compile downstream.
Add a wildcard arm:

```rust
match spec.task {
    ModelTask::Embed => { /* ... */ }
    ModelTask::Rerank => { /* ... */ }
    ModelTask::Generate => { /* ... */ }
    ModelTask::Raw => { /* ... */ }
    _ => return Err(/* unknown task for this provider */),
}
```

## See also

- [Provider reference](../reference/providers/index.md) — per-provider
  capability matrix and option keys.
- [Feature flags](../reference/feature-flags.md) — which providers
  ship which tasks.
- [API reference (rustdoc)](../api/uni_xervo/index.html) — full trait
  signatures.
