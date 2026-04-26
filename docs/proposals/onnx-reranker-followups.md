# ONNX Cross-Encoder Reranker — Deferred Follow-ups

**Status:** Open
**Created:** 2026-04-25
**Update (2026-04-26, uni-xervo `0.7.0`):** The cross-encoder reranker is no longer a separate provider. `LocalOnnxRerankerProvider` and the `local/onnx-reranker` provider id have been removed; the rerank task is served by `LocalOnnxProvider` (capabilities `[Raw, Rerank]`, dispatching in `load()`). The implementation lives at `src/provider/local_onnx/rerank.rs`. Path references below referring to `local_onnx_reranker.rs` should be read as `local_onnx/rerank.rs`. Type references to `LocalOnnxRerankerProvider` should be read as the rerank-task arm of `LocalOnnxProvider`.
**Context:** Companion doc to the addition of `LocalOnnxRerankerProvider` in uni-xervo `0.5.2`. The provider was ported from `uni-db` (originally landed there in commit `328c3e5a`) into `uni-xervo/src/provider/local_onnx_reranker.rs` so all uni-xervo consumers can do local cross-encoder reranking without re-implementing it. The four items below were considered during that hoist and deliberately deferred. Each is captured here so future contributors can pick them up with full context.

---

## 1. Score normalization API on `RerankerModel`

### Status

Deferred. Not yet implemented in any uni-xervo provider. Each consumer normalizes (or doesn't) on its own:
- `LocalOnnxRerankerProvider` — returns **raw logits** from the cross-encoder.
- `RemoteCohereProvider`, `RemoteVoyageAIProvider` — return whatever the remote API gives back (typically already in `[0, 1]`).
- uni-db (`crates/uni/src/query/df_graph/procedure_call.rs::sigmoid`, line ~1336) wraps the local-provider output in `sigmoid()` to map raw logits into `[0, 1]`.

### Why deferred

Trait-touching change. Adding normalization to `RerankerModel` would force every existing implementor (Cohere, Voyage, the new local ONNX) to re-spec their semantics in the same PR. Easier to review separately, after the hoist has landed.

### Motivation

Cross-encoder logits are unnormalized; sigmoid-mapping into `[0, 1]` is what almost every consumer wants. Today the choice of normalization is duplicated and inconsistent across providers — local ONNX returns logits, remote APIs return normalized scores, and consumers have to know which is which. Centralizing the choice makes the score domain consistent and lets users opt in to a different normalization (softmax, min-max) without rewriting consumer code.

### Sketch

Add to `uni-xervo/src/traits.rs`:

```rust
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScoreNormalization {
    #[default]
    Raw,
    Sigmoid,
    MinMax,
    Softmax,
}
```

Two competing API shapes:

**Option A — additive trait method (no breaking change):**
```rust
pub trait RerankerModel: Send + Sync {
    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>>;

    /// Default impl: calls `rerank`, then post-processes scores per `norm`.
    async fn rerank_normalized(
        &self,
        query: &str,
        docs: &[&str],
        norm: ScoreNormalization,
    ) -> Result<Vec<ScoredDoc>> { /* default sigmoid/softmax/min-max */ }
}
```

**Option B — struct-level helper (no trait change):**
```rust
impl ScoredDoc {
    /// Normalize this doc's score in-place against the full batch.
    pub fn normalize_in_place(&mut self, norm: ScoreNormalization, batch: &[ScoredDoc]) { /* ... */ }
}

/// Convenience: normalize a whole batch.
pub fn normalize_scores(docs: &mut [ScoredDoc], norm: ScoreNormalization) { /* ... */ }
```

**Recommendation:** lean **Option B**. Keeps the trait surface minimal, doesn't bloat existing remote-provider implementations that already return normalized scores, and gives the caller full control. Sigmoid is per-element so it doesn't need batch context; min-max and softmax do, hence the batch param.

### Files touched

- `uni-xervo/src/traits.rs` — new enum + (Option B) free function and impl on `ScoredDoc`.
- `uni-xervo/tests/score_normalization_test.rs` — *new*, unit tests for each variant.
- (consumer) `uni-db/crates/uni-query/src/query/df_graph/procedure_call.rs` — drop the inline `sigmoid()` (line ~1336), replace with `uni_xervo::normalize_scores(&mut scored, ScoreNormalization::Sigmoid)`.

### Risks

- Wrong default. Picking `Raw` as default keeps current behavior; picking `Sigmoid` would silently change downstream score distributions for any caller that doesn't explicitly opt out. Default `Raw` is the safe choice.
- Score-domain confusion. Documenting clearly that *the score domain returned by `rerank` depends on the provider* — and that `normalize_scores` is the way to get a consistent domain — is essential.

---

## 2. Shared `download_onnx_repo_files` helper

### Status

Deferred. HF-download logic is currently duplicated in two places:
- `uni-xervo/src/provider/local_onnx.rs` (~lines 320–352) — embedding/raw ONNX provider.
- `uni-xervo/src/provider/local_onnx_reranker.rs::download_model_files` — cross-encoder reranker provider (newly added).

### Why deferred

A third caller hasn't materialized yet. Premature extraction risks a wrong abstraction; the two existing callers are simple enough that inline duplication is reviewable.

### Sketch

New module `uni-xervo/src/provider/onnx_common.rs`:

```rust
use hf_hub::api::tokio::ApiRepo;

pub(crate) struct DownloadedOnnxFiles {
    pub model: PathBuf,
    pub extras: Vec<PathBuf>, // tokenizer, config, preprocessor, ...
}

/// Download an ONNX model + companion files from a HuggingFace repo.
///
/// `model_filenames` is a fallback chain of candidate model-file names
/// inside the repo (e.g. `["onnx/model.onnx", "model.onnx"]`). The first
/// one that downloads successfully wins.
///
/// `extras` are mandatory companion files (tokenizer, config) that all
/// must download or the call fails.
pub(crate) async fn download_onnx_repo_files(
    api_repo: &ApiRepo,
    alias: &str,
    model_filenames: &[&str],
    extras: &[&str],
) -> Result<DownloadedOnnxFiles> { /* ... */ }
```

Both `local_onnx` and `local_onnx_reranker` switch to it.

### Trigger to do this

When **either** of these happens:
- A third consumer of HF ONNX downloads is added (e.g. a sentence-transformer-style multi-file model).
- The two existing copies start diverging in ways that matter (e.g. one fixes a bug the other doesn't).

### Risks

- Premature shape. `model_filenames` as a fallback chain is the only generalization needed today; resist adding `revision`, `auth-token`, `progress-callback`, etc. until someone asks for them.

---

## 3. Shared `BertPairTokenizer` helper

### Status

Deferred. Single consumer today: `LocalOnnxRerankerProvider::tokenize_batch` (previously `uni-db/crates/uni/src/api/reranker.rs:142`, now in `uni-xervo/src/provider/local_onnx_reranker.rs`).

### Why deferred

Hoisting prematurely creates an abstraction with one user. The existing inline implementation is ~40 LOC and read clearly in context.

### Sketch

New module `uni-xervo/src/onnx_tokenizer.rs`:

```rust
use ndarray::Array2;
use tokenizers::Tokenizer;

/// Encode a batch of (query, document) pairs into the BERT-shape tensor
/// triple `(input_ids, attention_mask, token_type_ids)`, padded to the
/// longest sequence in the batch (capped at `max_seq_len`).
///
/// Used by any BERT-family ONNX model that takes paired text input —
/// cross-encoders for reranking, but also some bi-encoder embedders that
/// ingest `(query, passage)` together.
pub fn encode_pair_batch(
    tokenizer: &Tokenizer,
    query: &str,
    docs: &[&str],
    max_seq_len: usize,
) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>)> { /* ... */ }
```

### Trigger to do this

When a second BERT-pair consumer appears in uni-xervo. Likely candidates:
- A local sentence-transformer embedding provider that handles `(query, passage)` jointly.
- A new local cross-encoder variant (e.g. multi-stage rerankers).

Until then, leave the helper inline in `local_onnx_reranker.rs`.

### Risks

- API drift between BERT pair-encoding (cross-encoder shape) and BERT single-text encoding (typical embedding shape). The helper should focus on pair encoding and resist the temptation to be a general "BERT tokenizer wrapper".

---

## 4. ORT session pool

### Status

Deferred. Both `LocalOnnxProvider` (embedding) and `LocalOnnxRerankerProvider` (reranker) wrap their ORT session in `Mutex<Session>`. This serializes inference across concurrent queries.

### Why deferred

Larger surface change that touches the embedding path too. Needs its own design pass on concurrency semantics — pool sizing, memory budget, fairness, model-warmup interaction. Out of scope for the hoist PR.

### Motivation

`Mutex<Session>` is the production bottleneck after model loading. For workloads with many concurrent rerank/embed calls hitting the same alias, every call serializes through one ORT session even though the model weights could be replicated and run in parallel. ORT itself is thread-safe for `run` calls on **distinct** sessions, but the `Session` type isn't `Sync`.

### Sketch

New module `uni-xervo/src/provider/onnx_session_pool.rs`:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use ort::session::Session;

pub(crate) struct OnnxSessionPool {
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
}

impl OnnxSessionPool {
    /// Build `pool_size` independent sessions from the same model file.
    pub fn from_model_path(path: &Path, pool_size: usize) -> Result<Self> { /* ... */ }

    /// Round-robin pick a session and run a closure with exclusive access.
    pub fn with_session<F, R>(&self, f: F) -> R
    where F: FnOnce(&mut Session) -> R { /* ... */ }
}
```

Pool size from `ModelAliasSpec.options["session_pool_size"]`, default `1` (preserves today's behavior).

Both `LocalOnnxProvider` and `LocalOnnxRerankerProvider` switch from `Mutex<Session>` to `Arc<OnnxSessionPool>`.

### Risks

- **Memory.** ORT session memory is non-trivial — model weights are duplicated per pool slot. Default `pool_size = 1` preserves today's footprint; users opt in to higher concurrency.
- **Fairness.** Pure round-robin is naive; under contention some sessions may queue while others sit idle. `tokio::sync::Semaphore` + a per-session async lock would be fairer at the cost of complexity.
- **Warmup.** Currently the provider warms up one session; a pool needs to either warm up all `N` slots up front (long startup) or lazily on first hit (latency spike).

### Trigger to do this

When **either** of these happens:
- Profile data shows `Mutex<Session>` contention is a real bottleneck in a production workload.
- A user explicitly requests concurrent rerank/embed throughput.

---

## 5. Remove ort load-dynamic deadlock workaround

### Status

Workaround landed in `0.5.5`. As of `0.6.0` the workaround is **only present under load-dynamic modes** (`provider-onnx-dynamic`, the `_ort-fetched-base` group, and the un-bundleable `gpu-rocm`/`gpu-openvino` features which now activate `_ort-fetched-base`). All bundled modes (`provider-onnx`, `gpu-cuda`, `gpu-tensorrt`, `gpu-coreml`, `gpu-wgpu`) statically link ort and are immune to the deadlock by construction. The `preflight_ort_dylib` and `default_dylib_name` symbols are gated to load-dynamic modes only.

Upstream `pykeio/ort#560` is fixed in `17ed7277` but not yet released as of `=2.0.0-rc.12`. When `rc.13` ships, the preflight can be downgraded from "deadlock workaround" to "fast-fail UX improvement" or removed.

### Why this exists

`ort` 2.0.0-rc.12's load-dynamic error path has a re-entrant `OnceLock` deadlock: when `libloading::Library::new()` fails inside `setup_api`, the failure path constructs an `ort::Error` whose formatter calls back into `ort::api()` — which is exactly the `OnceLock` whose initializer just failed. The thread blocks on the same futex forever and the panic that `setup_api` intends to throw never fires.

We worked around this with `provider::onnx_ep::preflight_ort_dylib`, called before any ORT API touch in both providers' `load` paths. It attempts `libloading::Library::new` against `ORT_DYLIB_PATH` (or the platform default) and surfaces a clear `RuntimeError::Config` in <10ms instead of hanging.

### Upstream tracking

- Issue: [pykeio/ort#560](https://github.com/pykeio/ort/issues/560) (filed 2026-03-26, closed same day).
- Fix commit: [`17ed7277`](https://github.com/pykeio/ort/commit/17ed7277) — `fix: use separate error type for load-dynamic init`.
- Latest released ort: `v2.0.0-rc.12` (2026-03-05) — predates the fix by 21 days. No `rc.13` published as of 2026-04-25; `main` has 10 commits since the fix.

### Trigger to do this

When ort `rc.13` (or later) ships:

1. Bump `=2.0.0-rc.12` → new version in `Cargo.toml`.
2. Decide whether to remove or keep the preflight:
   - **Keep (default recommendation)**: it's only ~30 LOC and gives a clearer/faster error than letting ort's loader walk default paths and fail. Reposition its purpose in source comments from "deadlock workaround" to "fast-fail UX".
   - **Remove**: cleaner; trust the upstream fix's error message. Saves ~30 LOC and one optional dep (`libloading`).
3. Either way, drop the inline references to `pykeio/ort#560` once the bug is no longer load-bearing.

### Files to revisit

- `src/provider/onnx_ep.rs` — `preflight_ort_dylib` + `default_dylib_name` + tests block.
- `src/provider/local_onnx.rs` and `src/provider/local_onnx_reranker.rs` — call sites + the inline deadlock-workaround comments.
- `Cargo.toml` — `libloading` optional dep.
- `tests/preflight_dylib_test.rs` — keep or repurpose as a generic missing-dylib test.
- `CHANGELOG.md` — note the bump in the new version's entry.

---

## Tracking

Each item should become its own GitHub issue when picked up, with a link back to this doc for context. Cross-link from `uni-xervo/CHANGELOG.md` when an item lands.
