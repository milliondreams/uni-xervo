# Execution Plan — PR-1 Multimodal API Surface

| | |
|---|---|
| **Status** | Ready to execute |
| **Companion documents** | [`xervo-multimodal-api-design.md`](./xervo-multimodal-api-design.md), [`xervo-multimodal-api-proposal.md`](./xervo-multimodal-api-proposal.md) |
| **Target release** | uni-xervo 0.13.0 |
| **Date** | 2026-05-25 |

This document turns the design into an ordered, file-level work plan. Each phase has a single goal, an explicit exit criterion, and a validation command that must pass before the next phase begins. Phases are sequenced by compile-graph dependency — earlier phases must build before later phases can reference them.

---

## 0. Pre-flight

### 0.1 Branch + baseline

```bash
git checkout -b feat/multimodal-api-surface
cargo check --all-features
cargo test --all-features --no-run    # confirm a clean baseline build
```

**Exit criterion:** `cargo check --all-features` is clean on `main` and on the new branch.

### 0.2 Snapshot the existing public surface

Capture the current `cargo public-api` (or equivalent `cargo doc` JSON) so the final diff at the end of PR-1 can be reviewed as a single "what was added, nothing was changed" artifact.

```bash
cargo public-api --simplified > /tmp/uni-xervo-public-api-before.txt
```

If `cargo public-api` isn't installed, the rg-based fallback:

```bash
rg -nF "pub " src/ > /tmp/uni-xervo-public-surface-before.txt
```

**Exit criterion:** baseline file saved.

---

## Phase 1 — Foundation: `ModelTask` extension + `EmbedResult`

Nothing else in this PR compiles without these. Keep this phase small and self-contained so it can be reviewed in isolation.

### 1.1 `src/api.rs`

- Add `#[non_exhaustive]` to `ModelTask`.
- Add 7 new variants in declared order: `EmbedImage, EmbedAudio, EmbedMultimodal, Nlp, DocumentExtract, Transcribe, Ocr`.
- Confirm `#[serde(rename_all = "snake_case")]` produces stable wire names: `embed_image`, `embed_audio`, `embed_multimodal`, `nlp`, `document_extract`, `transcribe`, `ocr`.
- Add unit tests in the existing `tests` module: round-trip each new variant through `serde_json`.

### 1.2 `src/traits.rs`

Add `EmbedResult` near the existing `TokenUsage` declaration:

```rust
#[derive(Debug, Clone)]
pub struct EmbedResult {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Option<TokenUsage>,
}
```

### 1.3 Validation

```bash
cargo check --all-features
cargo test --all-features model_task   # serde round-trip tests pass
```

**Exit criterion:** existing tests still pass; new serde round-trip tests pass; no clippy regressions.

**Estimated effort:** 1–2 hours.

---

## Phase 2 — Trait module split

Splits `src/traits.rs` into a submodule layout *before* adding new traits, so the diff in Phase 3 is purely additive within submodules rather than a tangled split-plus-add.

### 2.1 Convert `src/traits.rs` to `src/traits/mod.rs`

```
src/traits/
  mod.rs                # existing trait definitions, re-exports
  raw_tensor_model.rs   # unchanged (was already a submodule)
```

Move existing trait definitions into `mod.rs`. Keep `pub use raw_tensor_model::{...}` re-exports intact. No content changes — this is a file move.

### 2.2 Verify no public-API drift

```bash
cargo check --all-features
cargo test --all-features --no-run
diff <(rg -nF "pub " src/) /tmp/uni-xervo-public-surface-before.txt | grep -v "ModelTask\|EmbedResult"
```

The diff must be empty (modulo Phase 1 additions). If any other public symbol moved or changed, fix before continuing.

### 2.3 Add empty submodule scaffolding

Create empty files referenced later:

```
src/traits/multimodal.rs   # ImageEmbeddingModel, AudioEmbeddingModel, MultimodalEmbeddingModel
src/traits/nlp.rs          # NlpModel + types
src/traits/docs.rs         # DocumentExtractionModel, OcrModel + types
src/traits/asr.rs          # TranscriptionModel + types
```

Each file: a module-level doc comment and nothing else. Wire each into `mod.rs` with `pub mod multimodal;` etc., and re-export everything via `pub use multimodal::*;`.

### 2.4 Validation

```bash
cargo check --all-features
cargo test --all-features
```

**Exit criterion:** zero behavior change; submodule scaffolding compiles.

**Estimated effort:** 1 hour. Keep this PR-commit isolated; it's a clean mechanical move that reviewers can skim.

---

## Phase 3 — New types (no traits yet)

Trait definitions in Phase 4 depend on these types existing. Adding them first keeps Phase 4 a pure trait-shape phase.

### 3.1 `src/traits/multimodal.rs`

- `AudioInput` enum (Bytes + Pcm variants, no Path).
- `MultimodalBlock` enum (Text, Image, Audio).
- `MultimodalInput { blocks: Vec<MultimodalBlock> }`.
- `Modality` enum (Text, Image, Audio, Video).

All `#[derive(Debug, Clone)]`. `Modality` also derives `Copy, PartialEq, Eq, Hash`.

### 3.2 `src/traits/nlp.rs`

- `NlpTasks` bitflag (via `bitflags!`).
- `NlpRequest<'a>`, `NlpResult`, `NlpToken`, `NlpSentence`, `DepLink`, `SrlFrame`, `SrlRole`, `SpeechAct`.

### 3.3 `src/traits/docs.rs`

- `DocExtractOptions`, `DocOutputFormat` (`#[non_exhaustive]`), `DocExtractResult`, `DocBlock`, `DocBlockKind` (`#[non_exhaustive]`).
- `OcrResult`, `OcrBlock`.

### 3.4 `src/traits/asr.rs`

- `TranscribeOptions`, `TranscribeResult`, `TranscribeSegment`, `TranscribeWord`.

### 3.5 `Cargo.toml`

Add deps:

```toml
bitflags = "2"
futures = "0.3"    # only if not already a direct dep
```

Confirm `futures` is needed at the direct-dep level by trying `cargo check` after Phase 4.2 and observing whether the `try_join_all` import resolves transitively. If it does, drop the direct dep.

### 3.6 Validation

```bash
cargo check --all-features
cargo doc --all-features --no-deps    # confirm all new types render
```

**Exit criterion:** all new types compile and render in rustdoc; no functional change yet.

**Estimated effort:** 3–4 hours including doc comments.

---

## Phase 4 — Trait definitions

Add the 7 new traits. This is the core "contract" of PR-1.

### 4.1 `src/traits/multimodal.rs`

- `ImageEmbeddingModel`
- `AudioEmbeddingModel`
- `MultimodalEmbeddingModel`

Each: `Send + Sync + Any`, `async fn embed(...) -> Result<EmbedResult>`, `fn dimensions() -> u32`, `fn model_id() -> &str`, `async fn warmup() -> Result<()> { Ok(()) }`. `MultimodalEmbeddingModel` also has `fn supported_modalities() -> &[Modality]`.

### 4.2 `src/traits/asr.rs`

`TranscriptionModel` with `transcribe`, `transcribe_many` (default fan-out via `futures::future::try_join_all`), `supported_languages`, `model_id`, `warmup`.

### 4.3 `src/traits/nlp.rs`

`NlpModel` with `analyze`, `supported_tasks`, `model_id`, `warmup`.

### 4.4 `src/traits/docs.rs`

- `DocumentExtractionModel` with `extract`, `model_id`, `warmup`.
- `OcrModel` with `recognize`, `model_id`, `warmup`.

### 4.5 Re-export in `src/traits/mod.rs`

```rust
pub use multimodal::*;
pub use nlp::*;
pub use docs::*;
pub use asr::*;
```

### 4.6 Validation

Dyn-safety smoke tests in each new submodule (or one shared `tests/dyn_safety_test.rs`):

```rust
#[test]
fn image_embedding_model_is_dyn_safe() {
    fn _accept(_: std::sync::Arc<dyn crate::traits::ImageEmbeddingModel>) {}
}
// ... one per trait
```

```bash
cargo check --all-features
cargo test --all-features --lib dyn_safety
cargo clippy --all-features -- -D warnings
```

**Exit criterion:** all 7 traits compile, all 7 dyn-safety stubs accept `Arc<dyn Trait>`.

**Estimated effort:** 4–6 hours including doc comments. This is the most-scrutinized phase; budget time for trait-shape iteration if review pushes back.

---

## Phase 5 — Mocks

Mocks must land before runtime integration (Phase 6) because the resolver tests need them.

### 5.1 `src/mock.rs`

Extend with 7 mock impls, one per new trait. Each:

- Returns deterministic placeholder data sized correctly (e.g., `vec![vec![0.0; 384]; n]` for embedders).
- Reports `dimensions() = 384` (or trait-appropriate constant — matches the canonical kniv encoder hidden size, so dimension-matching downstream code is consistent across mock and real).
- `model_id() = "mock"`.
- `warmup()` is no-op.
- For `MultimodalEmbeddingModel::supported_modalities`, return `&[Modality::Text, Modality::Image]`.
- For `NlpModel::supported_tasks`, return `NlpTasks::ALL`.
- For `TranscriptionModel::supported_languages`, return `&["en".to_string()]`.

### 5.2 Extend the mock provider's capabilities

The existing `MockProvider` in `src/mock.rs` advertises a fixed `supported_tasks`. Extend it to advertise all 7 new tasks, and extend its `load()` impl to return `Arc<dyn TraitName>` for each new task.

### 5.3 Validation

```bash
cargo test --all-features --lib mock
```

**Exit criterion:** mock provider loads each new task and returns a downcastable handle.

**Estimated effort:** 2–3 hours.

---

## Phase 6 — Runtime integration

### 6.1 `src/runtime.rs` — extend `HandleCache`

Add 7 new `DashMap<String, Arc<dyn TraitName>>` fields. Update `HandleCache::default()` automatically via `#[derive(Default)]` (already present).

### 6.2 `src/runtime.rs` — add 7 resolver methods

Each resolver follows the existing `embedding` template at `src/runtime.rs:148-178`:

1. Cache-check.
2. `lookup_spec(alias)`.
3. `resolve_and_load_internal(&spec)`.
4. `downcast_ref::<Arc<dyn TraitName>>()`.
5. Wrap in `Instrumented<TraitName>` (added in Phase 7).
6. Insert into cache, return.
7. Downcast failure → `RuntimeError::ProviderCapabilityMissing` (the richer variant — see design §11.5).

Resolver order in the impl block: `image_embedder`, `audio_embedder`, `multimodal_embedder`, `nlp_model`, `document_extractor`, `transcriber`, `ocr_model`.

**Compilation note:** this phase introduces calls to `Instrumented*` types that don't exist yet. Two options:

- **Option A (preferred):** stub each instrumented wrapper as a pass-through in Phase 6, then add real instrumentation in Phase 7. This keeps the phase compilable end-to-end.
- **Option B:** swap Phases 6 and 7. Less clean because the instrumented wrappers are easier to test against a working resolver.

Use Option A. Pass-through stub looks like:

```rust
// src/reliability.rs (temporary stub)
pub struct InstrumentedImageEmbeddingModel {
    pub inner: Arc<dyn ImageEmbeddingModel>,
    pub alias: String,
    pub provider_id: String,
    pub timeout: Option<Duration>,
    pub retry: Option<RetryConfig>,
}

#[async_trait]
impl ImageEmbeddingModel for InstrumentedImageEmbeddingModel {
    async fn embed(&self, images: Vec<ImageInput>) -> Result<EmbedResult> {
        self.inner.embed(images).await       // TODO Phase 7: timeout / retry / cb / metrics
    }
    fn dimensions(&self) -> u32 { self.inner.dimensions() }
    fn model_id(&self) -> &str { self.inner.model_id() }
    async fn warmup(&self) -> Result<()> { self.inner.warmup().await }
}
```

Mark each stub `// TODO Phase 7` so the work is impossible to miss.

### 6.3 Validation

```bash
cargo check --all-features
cargo test --all-features --test handle_cache_test    # existing tests still pass
```

Add new tests in `tests/multimodal_resolver_test.rs`:

- `image_embedder_resolves_via_mock`
- `image_embedder_cache_hit_returns_same_arc` (use `Arc::ptr_eq`)
- `image_embedder_alias_not_found` returns `AliasNotFound`
- `image_embedder_capability_mismatch` (alias declared `EmbedImage` but provider returns wrong handle) returns `ProviderCapabilityMissing`
- ... one set per new trait

**Exit criterion:** all 7 resolvers cache-on-success and error-on-mismatch.

**Estimated effort:** 4–6 hours (mostly mechanical replication of the existing pattern).

---

## Phase 7 — Real instrumentation wrappers

Replace the Phase 6 pass-through stubs with real `Instrumented*Model` impls.

### 7.1 `src/reliability.rs`

For each new trait, implement timeout + retry + circuit-breaker + metrics around the business method. Model after the existing `InstrumentedEmbeddingModel` impl.

**Shape per wrapper:**

- Struct fields: `inner`, `alias`, `provider_id`, `timeout`, `retry`, optional `breaker: CircuitBreakerWrapper`.
- `async fn business_method(...)`:
  1. `tracing::info_span!("op", alias = %self.alias, provider = %self.provider_id)`.
  2. Compose `breaker.call(|| async { with_retry(|| with_timeout(self.inner.business(...))) })`.
  3. On Ok: emit `metrics::counter!("uni_xervo.ops.success", "alias" => …)`.
  4. On Err: emit `metrics::counter!("uni_xervo.ops.failure", "alias" => …, "kind" => err.kind())`.
- Forwarded accessors (`dimensions`, `model_id`, `supported_*`) pass through unchanged.
- `warmup()` wraps in a tracing span; no retry/timeout (matches existing pattern).

### 7.2 Special case: `TranscriptionModel`

Both `transcribe` and `transcribe_many` get instrumented. The default `transcribe_many` fan-out happens *inside* the inner impl, so the instrumented batch operation has one timeout / retry envelope covering the whole batch. Document this clearly in the wrapper's doc comment — operators tuning `timeout` for batched transcription need to know it applies batch-wide.

### 7.3 Validation

Add `tests/multimodal_instrumentation_test.rs`:

- `embed_with_timeout_returns_timeout_err`
- `embed_with_retry_recovers_on_third_attempt`
- `circuit_breaker_opens_after_n_failures`
- `transcribe_many_timeout_applies_to_whole_batch`

Each test uses the mock to inject controlled failures / sleeps.

```bash
cargo test --all-features --test multimodal_instrumentation_test
```

**Exit criterion:** parity with existing instrumentation tests in `tests/reliability_test.rs`.

**Estimated effort:** 6–8 hours. This is the highest-value phase — without real instrumentation, the design's "managed observability" claim is hollow.

---

## Phase 8 — Options validation

### 8.1 `src/options_validation.rs`

For each registered provider, extend its per-task validator to explicitly accept-or-reject every new `ModelTask` variant.

Per-provider strategy in PR-1:

- All existing providers reject all 7 new tasks with a clear `Config` error (e.g., `"Provider 'local/onnx' does not support task 'embed_image'"`).
- This is intentional: PR-1 contains no provider implementations. Catalog misconfiguration fails fast at register time.
- Subsequent PRs (PR-2 through PR-6 in design §12) flip individual provider+task pairs from "rejected" to "validated" as the impls land.

### 8.2 Helper

Add a small utility to keep the boilerplate tight:

```rust
fn reject_unsupported_task(provider_id: &str, task: ModelTask) -> RuntimeError {
    RuntimeError::Config(format!(
        "Provider '{}' does not support task '{}'",
        provider_id, task_wire_name(task),
    ))
}

fn task_wire_name(task: ModelTask) -> &'static str {
    match task {
        ModelTask::Embed => "embed",
        ModelTask::Rerank => "rerank",
        ModelTask::Generate => "generate",
        ModelTask::Raw => "raw",
        ModelTask::EmbedImage => "embed_image",
        ModelTask::EmbedAudio => "embed_audio",
        ModelTask::EmbedMultimodal => "embed_multimodal",
        ModelTask::Nlp => "nlp",
        ModelTask::DocumentExtract => "document_extract",
        ModelTask::Transcribe => "transcribe",
        ModelTask::Ocr => "ocr",
    }
}
```

### 8.3 Validation

Add `tests/options_validation_multimodal_test.rs`:

- For every (existing_provider, new_task) pair, assert `validate_provider_options` returns a `Config` error.
- Mock provider's validator accepts all new tasks (so mock-backed resolver tests pass).

```bash
cargo test --all-features --test options_validation_multimodal_test
cargo test --all-features                       # full suite still green
```

**Exit criterion:** every real provider rejects new tasks; mock provider accepts them; mock-backed integration tests still pass.

**Estimated effort:** 2–3 hours.

---

## Phase 9 — Documentation

### 9.1 Module-level doc comments

Every new submodule gets a top-of-file doc explaining the trait it defines and a usage example modeled on the lib.rs quick-start. Each new public type gets a one-line summary minimum.

### 9.2 `src/lib.rs` quick-start

Update the "Key concepts" bullet list to mention the new traits in one line each. Do not bloat the quick-start example — it stays on `EmbeddingModel`.

### 9.3 `CHANGELOG.md`

Section under `## [0.13.0] - YYYY-MM-DD`:

```markdown
### Added
- Seven new model traits for multimodal inference: `ImageEmbeddingModel`,
  `AudioEmbeddingModel`, `MultimodalEmbeddingModel`, `NlpModel`,
  `DocumentExtractionModel`, `TranscriptionModel`, `OcrModel`. No provider
  implementations in this release — see the multimodal API design doc.
- `ModelTask` gains variants: `EmbedImage`, `EmbedAudio`, `EmbedMultimodal`,
  `Nlp`, `DocumentExtract`, `Transcribe`, `Ocr`.
- `EmbedResult { vectors, usage }` for new embed traits. Existing
  `EmbeddingModel::embed` signature is unchanged.
- New `ModelRuntime` resolvers: `image_embedder`, `audio_embedder`,
  `multimodal_embedder`, `nlp_model`, `document_extractor`, `transcriber`,
  `ocr_model`.

### Changed
- `ModelTask` is now `#[non_exhaustive]`. Downstream pattern matches without
  a wildcard arm will not compile. Add `_ => { ... }` to fix.

### Notes
- `RawTensorModel` and `ModelTask::Raw` are unchanged and remain
  first-class public API. Existing customers using `raw_tensor_model(alias)`
  require no code changes.
```

### 9.4 Public API snapshot

```bash
cargo public-api --simplified > /tmp/uni-xervo-public-api-after.txt
diff /tmp/uni-xervo-public-api-before.txt /tmp/uni-xervo-public-api-after.txt > /tmp/pr1-public-api.diff
```

Attach the diff to the PR description. Expect: only additions, plus the `#[non_exhaustive]` marker on `ModelTask`.

**Estimated effort:** 2–3 hours.

---

## Phase 9.5 — Canonical default catalog entry (docs only)

PR-1 ships no provider implementations, so no functional `nlp/default` alias yet exists. But the design pins `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` as the canonical default NLP model (design §5.4.1). PR-1 records that decision in writing so anyone reading the trait can find the reference impl pointer.

### 9.5.1 `src/traits/nlp.rs` — trait-level doc comment

Add a section to the `NlpModel` doc comment:

```rust
/// # Canonical default model
///
/// The reference implementation is
/// [`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall):
/// a DeBERTa-v3-xsmall multi-head cascade producing POS / NER / DEP / SRL / CLS
/// in a single forward pass. Ships as the `nlp/default` catalog alias once the
/// `local/onnx` provider gains an `NlpModel` impl (planned for the follow-up
/// release after the API surface lands).
///
/// Tagsets:
/// - POS: Universal Dependencies English EWT (17 classes)
/// - NER: OntoNotes (18 entity types)
/// - DEP: Universal Dependencies English EWT
/// - SRL: PropBank (42 classes)
/// - CLS: 8 dialog acts
///
/// Language: English. Maximum input: 128 tokens (provider-side chunking
/// is the impl's responsibility — the trait contract is offset-stable
/// regardless of internal chunking).
```

### 9.5.2 `docs/` or `examples/` — example catalog snippet

If `docs/catalog-examples.md` exists, add the canonical `nlp/default` JSON spec from design §5.4.1. Otherwise create `examples/catalog-multimodal.json` with all 7 task aliases stubbed (only `nlp/default` carries a real model_id; the others are placeholders with `provider_id: "local/onnx"` and a `// TODO: model_id pending provider impl` comment via JSON-with-comments). Mark this example as illustrative — it won't load until provider impls ship.

### 9.5.3 CHANGELOG addendum

Append to the `## [0.13.0]` entry from Phase 9.3:

```markdown
### Reference models

The canonical default for `NlpModel` is
`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`. It will be wired up as
`nlp/default` once the `local/onnx` provider gains `NlpModel` support in
a follow-up release. No catalog entry ships in this release.
```

### 9.5.4 Validation

```bash
cargo doc --all-features --no-deps                # confirm doc renders
rg "kniv-deberta-nlp-base-en-xsmall" docs/ src/ examples/ CHANGELOG.md
```

The grep must return at least three hits: doc comment, CHANGELOG, example/docs.

**Estimated effort:** 0.5 h. Pure documentation; no code change.

---

## Phase 10 — Final validation gate

### 10.1 Full test suite

```bash
cargo test --all-features
cargo test --no-default-features
cargo test --features provider-onnx
cargo test --features provider-candle
cargo test --features provider-mistralrs
```

Each feature combination must build and pass. The new traits are unconditional; only impls would be gated, and no impls land here.

### 10.2 Clippy + fmt

```bash
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --check
```

### 10.3 Docs build

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

### 10.4 Bench smoke

```bash
cargo bench --bench runtime_bench --no-run    # confirms benches still compile
```

### 10.5 Public-API review

Walk through `/tmp/pr1-public-api.diff` line by line in the PR description. Confirm every line is an addition (or the documented `#[non_exhaustive]` change).

**Exit criterion:** all of the above pass. Open the PR.

**Estimated effort:** 1–2 hours, plus iteration on review feedback.

---

## Summary timeline

| Phase | Goal | Effort | Cumulative |
|---|---|---|---|
| 0 | Pre-flight, branch, baseline snapshot | 0.5 h | 0.5 h |
| 1 | `ModelTask` + `EmbedResult` | 1–2 h | 2.5 h |
| 2 | Trait module split (mechanical move) | 1 h | 3.5 h |
| 3 | New shared types | 3–4 h | 7.5 h |
| 4 | Trait definitions | 4–6 h | 13.5 h |
| 5 | Mocks | 2–3 h | 16.5 h |
| 6 | Runtime: HandleCache + resolvers (stub instrumentation) | 4–6 h | 22.5 h |
| 7 | Real instrumentation wrappers | 6–8 h | 30.5 h |
| 8 | Options validation rejection arms | 2–3 h | 33.5 h |
| 9 | Documentation + CHANGELOG | 2–3 h | 36.5 h |
| 9.5 | Canonical kniv default — docs + example catalog | 0.5 h | 37 h |
| 10 | Final validation gate + PR | 1–2 h | 39 h |

**Total: ~35–40 engineering hours** (≈ one focused engineer-week). Each phase exits with a green `cargo test --all-features`, so progress is unambiguous and the branch is always shippable as a partial increment if needed.

---

## Parallelization notes

The phase ordering above is strict on the compile-graph axis. If two engineers split the work:

- **Phases 3 + 4** can be parallelized *per submodule* once the empty scaffolding from Phase 2 lands. Engineer A takes `multimodal.rs` + `asr.rs` (most interconnected, gets `AudioInput` right early); Engineer B takes `nlp.rs` + `docs.rs`.
- **Phases 6 + 7**: a single engineer should own both, because Phase 7 finishes Phase 6's stubs. Splitting risks subtle wrapper-vs-resolver mismatches.
- **Phase 8** is independent of Phases 6–7 once the trait + `ModelTask` exist. Can start in parallel with Phase 6.

For a single engineer, the linear order is correct and minimizes context-switching.

---

## Risk register specific to execution

| Risk | Phase | Mitigation |
|---|---|---|
| `cargo public-api` shows accidental changes to existing symbols | 10 | Phase 2's "no public-API drift" gate catches this early. |
| `bitflags` v2 macro-syntax surprises | 3 | Lock to `bitflags = "2.6"` and reference an existing crate's usage (e.g., `tracing-subscriber`). |
| Instrumentation wrapper boilerplate becomes copy-paste rot | 7 | After PR-1, consider a generic `Instrumented<T>` macro or trait-impl helper as a follow-up. Don't try it in PR-1 — keep the diff readable. |
| `transcribe_many` default impl borrow-checker fight with `TranscribeOptions: Clone` | 4 | `TranscribeOptions: Clone` is already in §7.5 of the design via `#[derive(Clone)]`. Confirm at Phase 3 and don't forget the derive. |
| Reviewer pushback on `#[non_exhaustive]` for `ModelTask` | 1 | Land Phase 1 as its own commit, prepare the CHANGELOG note early, point reviewers to the design §10 + §11.1 rationale. If rejected, revert is one-line; subsequent phases don't depend on the marker. |
| Provider validators in Phase 8 explode from 1 arm per provider to 7 | 8 | Use the `task_wire_name` helper + a single `_ => reject_unsupported_task(...)` fallback arm; do not enumerate all 7 per provider. |

---

## Phase exit checklist (apply at every phase)

Before declaring a phase done:

- [ ] `cargo check --all-features` clean.
- [ ] `cargo test --all-features` passes (no skipped, no ignored other than pre-existing).
- [ ] `cargo clippy --all-features -- -D warnings` clean.
- [ ] No public symbol from `main` changed shape (verified by surface diff).
- [ ] New public symbols have doc comments.
- [ ] Phase-specific exit criterion (named in each phase above) met.
- [ ] Single focused commit, message references the phase: `feat(traits): phase N — <goal>`.

This per-phase rigor makes the PR easy to land in chunks if reviewers want to merge incrementally, and impossible to land if anything regressed.
