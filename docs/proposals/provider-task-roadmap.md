# Provider × Task Coverage Roadmap

**Status:** Draft
**Created:** 2026-04-26
**Author:** Rohit Rai
**Context:** uni-xervo `0.5.6` exposes four tasks (`embed`, `rerank`, `generate`, `raw`) across five local providers (`candle`, `fastembed`, `onnx` raw, `onnx` reranker, `mistralrs`) and eight remote providers. The local-provider task surface has grown organically — most notably the `LocalOnnxRerankerProvider` hoist in `0.5.2` (see [onnx-reranker-followups.md](onnx-reranker-followups.md)) — and now diverges meaningfully from what the underlying frameworks can actually deliver. This document is an audit of that divergence and a proposed sequencing of the work to close it.

This is a **roadmap**, not an implementation spec. Each phase below lists the items, the rationale, the expected effort tier, and the open design questions that need to be resolved before code is written. Per-task design specs are deferred to companion documents created when each phase begins.

---

## Goals

- Make the gap between "what uni-xervo exposes" and "what the local backends ship" visible and actionable.
- Sequence the work so that low-effort, high-leverage additions (encoder tasks on ORT, unused fastembed-rs APIs, mistral.rs feature surface) precede multi-month projects (ORT-hosted text generation, multi-graph diffusion pipelines).
- Identify hard caps — capabilities the underlying frameworks genuinely don't provide — so consumers don't expect uni-xervo to fill them.

## Non-goals

- Adding new *providers*. This document only addresses task coverage on the providers that already exist.
- Touching the trait surface. Score-normalization on `RerankerModel` and similar trait-level work is tracked separately in [onnx-reranker-followups.md](onnx-reranker-followups.md).
- Remote provider gaps. Scope is local providers only.

---

## Current state

### Local provider × task matrix

Legend:
- **Present** — exposed by uni-xervo today
- **Easy** — backend supports it; mechanical wrap (forward pass plus pre/post)
- **Medium** — backend supports it; some engineering (multi-stage pipelines, schedulers, preprocessing)
- **Hard** — backend supports it; real project (KV-cache loops, multi-graph orchestration, framework-level features)
- **Cannot** — backend fundamentally doesn't ship this capability
- **N/A** — wrong abstraction layer

| Task | `local/candle` | `local/fastembed` | `local/onnx` | `local/mistralrs` |
|---|---|---|---|---|
| Dense text embedding | Present | Present | Easy | Cannot (LLM-pooling only) |
| Sparse text embedding (SPLADE) | Cannot | Easy (`SparseTextEmbedding`) | Easy | Cannot |
| Image embedding (CLIP/SigLIP) | Medium (CLIP/SigLIP shipped) | Easy (`ImageEmbedding`) | Easy | Cannot |
| Cross-encoder reranking | Medium (BERT + head, no shipped module) | Easy (`TextRerank`, four models) | Present | Cannot |
| Text generation | Hard (zoo present, no serving loop) | Cannot | Hard (no `onnxruntime-genai` Rust binding) | Present |
| Vision LLM (VLM) | Hard (LLaVA/Moondream/Qwen3-VL shipped) | Cannot | Hard (multi-graph) | Present |
| Image generation (SD/Flux) | Hard (SD/Flux/MMDiT shipped) | Cannot | Hard (UNet+VAE+scheduler) | Present (FLUX) |
| ASR (Whisper) | Medium (Whisper + quantized shipped) | Cannot | Easy (well-trodden ONNX path) | Present |
| TTS | Medium (Parler, MetaVoice, CSM, …) | Cannot | Medium (Piper, Kokoro) | Present (Dia, Parler) |
| Tool calling | Cannot (no template/parser framework) | Cannot | Hard (DIY parser) | Easy (exists, not surfaced) |
| Structured output (JSON/regex/CFG) | Cannot (no logits processor framework) | Cannot | Hard (DIY sampler) | Easy (`llguidance`, not surfaced) |
| LoRA / adapter hot-swap | Cannot | Cannot | Cannot (no framework) | Easy (X-LoRA, not surfaced) |
| Speculative decoding | Cannot | Cannot | Cannot | Easy (draft + n-gram, not surfaced) |
| ISQ (in-situ quantize any HF model) | Cannot | Cannot | Cannot | Easy (flagship feature, not surfaced) |
| Raw ONNX tensor execution | N/A | N/A | Present | N/A |

### Reading the matrix

Three structurally different gap classes:

1. **Provider-level coverage gaps** (Easy / Medium cells). Backend already ships it; uni-xervo doesn't call into it. Highest density on `local/onnx` (every encoder-shaped task is one wrapper away) and `local/mistralrs` (the *features* — tool calling, structured output, LoRA, speculative decoding, ISQ — never made it through `provider/mistralrs.rs` even though the `generate` task is exposed).

2. **Hard-but-doable gaps** (Hard cells). Real engineering, not plumbing. Two clusters: (a) any LLM generation path that isn't mistral.rs (Candle has no serving loop; ORT has no `onnxruntime-genai` Rust binding), (b) multi-graph pipelines (VLM, Stable Diffusion) on Candle/ORT.

3. **Hard caps** (Cannot cells). Won't change without upstream framework changes: mistral.rs has no encoder embeddings or rerankers; fastembed has no generation; Candle has no logits-processor framework.

---

## Phased roadmap

### Phase 1 — Encoder coverage on ORT and fastembed (Easy)

**Goal:** Eliminate the awkwardness of needing two local providers to do hybrid retrieval.

| # | Item | Provider | Effort | Notes |
|---|---|---|---|---|
| 1.1 | Dense text embedding via ORT | `local/onnx` | Easy | Same plumbing as `LocalOnnxRerankerProvider`. Forward pass + mean/cls pooling. Reuses the existing `ort::Session` + `tokenizers` machinery and the `load-dynamic` setup from `f2a46774`. |
| 1.2 | Cross-encoder reranking via fastembed-rs | `local/fastembed` | Easy | Wraps `TextRerank` (BGE-reranker-base, BGE-reranker-v2-m3, Jina-reranker-v1-turbo, Jina-reranker-v2-multilingual). Question for design: does this *replace* `LocalOnnxRerankerProvider` or sit alongside? See open question #1. |
| 1.3 | Sparse text embedding via fastembed-rs | `local/fastembed` | Easy | `SparseTextEmbedding` API; SPLADE++ and BGE-M3 sparse. Requires deciding the trait shape for sparse vectors — currently no `SparseEmbeddingModel` trait. See open question #2. |
| 1.4 | Image embedding via fastembed-rs | `local/fastembed` | Easy | `ImageEmbedding` API; CLIP ViT-B/32, ResNet50, Unicom, Nomic-Vision-v1.5. Requires `ImageEmbeddingModel` trait or extending `EmbeddingModel` to accept an image input. See open question #2. |
| 1.5 | Sparse text embedding via ORT | `local/onnx` | Easy | After (1.3) lands the trait. SPLADE exports cleanly to ONNX. |
| 1.6 | Image embedding via ORT | `local/onnx` | Easy | After (1.4) lands the trait. |

**Why this phase first:**
- Items 1.1–1.4 are all single-forward-pass wrappers over machinery already in the dependency graph — no new crates, no framework projects.
- The encoder side of a RAG/agent pipeline gets called far more often than the generator side. Locally hosted embedding closes the most common production hot-path.
- Resolves the awkwardness from the matrix where `local/onnx` has rerank but not embed (the hand-off through fastembed for embeddings is purely historical).

**Phase exit criteria:**
- `uni`'s hybrid-retrieval path can complete embed + rerank using only `local/onnx` (no fastembed dependency required).
- Sparse and image embedding traits exist and at least one provider implements each.

### Phase 2 — Mistral.rs feature surface (Easy / Medium)

**Goal:** Surface the mistral.rs capabilities that are already production-grade in the framework but invisible through the uni-xervo abstraction.

| # | Item | Provider | Effort | Notes |
|---|---|---|---|---|
| 2.1 | Tool calling on `GeneratorModel` | `local/mistralrs` (+ all remote `generate` providers) | Medium | Trait change. Need an `OpenAI-style ToolSpec` + `ToolCall` shape that maps onto mistral.rs, OpenAI, Anthropic, Gemini, Cohere — all of which already have tool-calling APIs. This is the single biggest agentic-workload gap. |
| 2.2 | Structured output (JSON schema, regex, CFG) | `local/mistralrs` (+ remotes that support it) | Medium | mistral.rs uses `llguidance`. OpenAI/Gemini have JSON-schema modes. Trait surface needs a `StructuredOutput` enum. |
| 2.3 | LoRA / X-LoRA hot-swap | `local/mistralrs` | Medium | Adapter selection at request time. Fits as an `options` field on the generation request, not a trait change. |
| 2.4 | Speculative decoding configuration | `local/mistralrs` | Easy | Configuration plumbing — draft model + n-gram. Mostly a `ProviderOptions` schema addition. |
| 2.5 | ISQ (in-situ quantization) | `local/mistralrs` | Easy | Configuration plumbing — quantization spec at load time. `ProviderOptions` schema addition. Flagship mistral.rs feature; high user value. |

**Why this phase second:**
- Items 2.1 and 2.2 are *trait-touching*, so they need to land before downstream code starts depending on the new shapes.
- Items 2.3–2.5 are pure options-schema additions; near zero risk, immediate value.
- Tool calling and structured output are table stakes for agentic workloads; without them, downstream code has to bypass uni-xervo's abstraction for any non-trivial generation.

**Phase exit criteria:**
- `GeneratorModel` supports tool calls and structured output across all generation-capable providers (or returns a typed "not supported" error).
- mistral.rs's full feature set is reachable through `provider/mistralrs.rs` configuration.

### Phase 3 — Speech and vision encoders on ORT (Medium)

**Goal:** Add the two highest-leverage non-text encoder paths that ORT can host today.

| # | Item | Provider | Effort | Notes |
|---|---|---|---|---|
| 3.1 | Whisper ASR | `local/onnx` | Easy–Medium | Mel-spectrogram preprocessing + encoder-decoder loop. Well-trodden in the ONNX ecosystem (Murmure, SilentKeys). Requires an `AsrModel` trait. |
| 3.2 | TTS (Piper, Kokoro) | `local/onnx` | Medium | Text → mel → vocoder. Multi-stage. Requires a `TtsModel` trait. |
| 3.3 | Whisper ASR via Candle | `local/candle` | Medium | After (3.1) lands the trait. Candle ships Whisper; would mostly be wiring. |
| 3.4 | TTS via Candle | `local/candle` | Medium | After (3.2) lands the trait. Candle ships Parler-TTS, MetaVoice, CSM. |

**Why this phase third:**
- Speech tasks unlock new product surfaces, not just performance for existing ones — different prioritization than Phase 1.
- ORT-hosted Whisper is the only credible local-only ASR path that doesn't depend on mistral.rs's recently-added (and still experimental) Whisper support.
- Candle parity comes for free once the traits exist.

**Phase exit criteria:**
- `AsrModel` and `TtsModel` traits exist with at least one local provider each.
- mistral.rs and ORT both implement at least the ASR trait.

### Phase 4 — Hard projects (Hard)

**Goal:** Decide whether these should ever be built; if so, they each become standalone proposals.

| # | Item | Decision needed | Notes |
|---|---|---|---|
| 4.1 | Text generation on `local/onnx` | Build or defer? | Requires either a Rust binding for `onnxruntime-genai` (ideally upstream-maintained) or a hand-rolled KV-cache loop, sampler, chat templating. Multi-month project. *Recommendation: defer until upstream binding exists. Track the upstream issue, don't start from scratch.* |
| 4.2 | VLM on `local/onnx` | Build or defer? | Multi-graph orchestration (vision encoder + LLM decoder). Real engineering. *Recommendation: defer. mistral.rs covers the production cases.* |
| 4.3 | Image generation on `local/onnx` | Build or defer? | UNet + VAE + scheduler glue. *Recommendation: defer. mistral.rs FLUX is the current path.* |
| 4.4 | Text generation on `local/candle` | Build or defer? | Candle ships the model code but no serving loop (no PagedAttention, no continuous batching). *Recommendation: defer indefinitely. mistral.rs is the right tool; Candle is a model zoo, not an inference engine.* |

**Why this phase exists at all:** to make explicit that these gaps are *known* and *deliberately deferred*, so future contributors don't relitigate the decision.

### Hard caps (won't address)

These are framework-level limits, not work items. Document them so consumers don't ask uni-xervo to fill them:

- **No encoder embeddings or rerankers from mistral.rs.** Encoder tasks stay on Candle / fastembed / ORT.
- **No generation from fastembed-rs.** It is an embedding library by design.
- **No structured output / tool calling / logits processors from Candle.** Candle is a model-zoo crate.
- **No first-class reranker from Candle.** Possible via a BERT + classification head wrapper, but not shipped as a module.

---

## Cross-cutting decisions

Several items above depend on broader design questions. These need to be resolved before the relevant phase begins.

### D1. Consolidate `local/onnx` and `local/fastembed`?

After Phase 1, `local/onnx` and `local/fastembed` will have nearly identical task coverage on encoders (embed, rerank, sparse, image-embed). fastembed-rs is *itself* ORT under the hood with a curated model registry. The architectural question is whether the two providers should remain distinct (their value being a model-ID registry) or whether `local/onnx` should absorb the registry and become the single ORT-backed encoder provider.

Tradeoffs:
- **Keep both:** zero migration cost; users opt into whichever vocabulary they prefer; some duplication in pre/post-processing code.
- **Merge:** one tokenizer/preprocessing pipeline; fewer provider IDs to document; breaking change for existing fastembed users.

*Recommendation: keep both through Phase 1; revisit after.*

### D2. File layout for `local/onnx`

**Resolved (2026-04-26, uni-xervo `0.7.0`).** Implemented ahead of Phase 1: provider split (`local/onnx` and `local/onnx-reranker`) collapsed into a single `LocalOnnxProvider` declaring `[Raw, Rerank]` capabilities and dispatching in `load()`. File layout reorganized to:

```
provider/local_onnx.rs           — unified provider entry (capabilities + load() dispatch)
provider/local_onnx/raw.rs       — raw tensor execution task
provider/local_onnx/rerank.rs    — cross-encoder rerank task
```

Sibling-file convention matches `traits.rs` + `traits/raw_tensor_model.rs`. Phase 1 additions (`embed.rs`, `sparse.rs`, `image_embed.rs`) and Phase 3 (`asr.rs`, `tts.rs`) will land alongside in the same module.

### D3. Trait additions

Phases 1 and 3 add new task traits. These are public-API surface and should be designed as a single batch, not individually:

- `SparseEmbeddingModel` — output is sparse vector (token-id → weight)
- `ImageEmbeddingModel` — input is image bytes / `image::DynamicImage`, output is dense vector. Or extend `EmbeddingModel` with an enum input?
- `AsrModel` — input is audio bytes + sample rate, output is text + optional timestamps
- `TtsModel` — input is text + voice ID, output is audio bytes + sample rate

Open question: do CLIP-family models that produce *both* text and image embeddings live as two model handles or one bimodal handle? Affects how `local/fastembed` exposes CLIP.

### D4. Remote provider parity for new tasks

Several Phase 1/3 tasks have remote-provider equivalents that should be implemented in lockstep to keep the alias-swap promise of uni-xervo intact:

- Sparse embedding — Voyage AI has `voyage-rerank-sparse`; Cohere has multilingual sparse.
- Image embedding — Voyage AI multimodal, Cohere multimodal.
- ASR — OpenAI Whisper API, Gemini.
- TTS — OpenAI, Gemini, ElevenLabs (would require a new remote provider).

The new task traits should land with at least one remote implementation each, so the abstraction doesn't ossify around local-only assumptions. Track this as part of each phase, not a separate phase.

---

## Open questions

1. **Reranker consolidation.** `LocalOnnxRerankerProvider` (Phase 0, already shipped) and a fastembed-rs `TextRerank` wrapper (Phase 1.2) cover overlapping models. Do we want both? If only one, which?
2. **Sparse/image-embed trait shape.** New traits or polymorphic input on `EmbeddingModel`? The latter is closer to what fastembed-rs does internally but loses static type safety at the call site.
3. **mistral.rs upgrade cadence.** Phase 2 depends on mistral.rs versions that ship the relevant features. Confirm minimum version requirements before starting.
4. **Tool-call schema unification.** Does uni-xervo adopt OpenAI's tool-call shape verbatim, or define a neutral one and translate per provider? The former is faster; the latter is cleaner but adds a translation layer for every provider.
5. **`onnxruntime-genai` Rust binding.** Is there an upstream effort worth contributing to, or should uni-xervo treat ORT-hosted generation as permanently deferred (Phase 4.1)?

---

## Sequencing summary

```
Phase 1 (Easy)        →  encoder coverage on ORT + fastembed-rs    →  weeks
Phase 2 (Easy/Medium) →  mistral.rs feature surface                →  weeks
Phase 3 (Medium)      →  speech + vision encoders                  →  weeks–months
Phase 4 (Hard)        →  decide; mostly defer                      →  N/A
Hard caps             →  document and move on                      →  N/A
```

Phases 1 and 2 are independent and can run in parallel. Phase 3 depends on the trait scaffolding from Phase 1 (`EmbeddingModel` extensions, new task-trait conventions). Phase 4 should not start before all of Phases 1–3 are complete or explicitly cancelled.

---

## Appendix: framework capability evidence

The classifications above are grounded in source-level audits of each backend (April 2026):

- **Candle (`candle-transformers` 0.10.2):** model zoo includes Llama 1–3, Mistral, Mixtral, Phi/Phi-3, Qwen2/3+MoE, Gemma 1–3, DeepSeek-v2, Mamba, RWKV, Whisper, Parler-TTS, MetaVoice, SD 1.5/2.1/SDXL, Flux, MMDiT, LLaVA, Moondream, Qwen3-VL, ColPali, CLIP/SigLIP. Does not ship: continuous batching, paged attention, speculative decoding, structured-output sampler, dedicated reranker module.

- **fastembed-rs (latest 2026):** four first-class APIs (`TextEmbedding`, `SparseTextEmbedding`, `TextRerank`, `ImageEmbedding`), all backed by `ort` 2.0.0-rc.12. Model registry covers BGE family, E5, Jina, Nomic, ModernBERT, Snowflake-Arctic, EmbeddingGemma, Qwen3-VL multimodal embedding (feature-gated via Candle).

- **`ort` (pykeio 2.x):** generic ONNX Runtime binding; can host anything exported to ONNX. Execution providers: CPU, CUDA, TensorRT, CoreML, DirectML, OpenVINO, ROCm, oneDNN, XNNPACK, QNN, plus more. No tokenizer (compose with HF `tokenizers`). **No Rust binding for `onnxruntime-genai` exists as of early 2026** — this is the single biggest blocker for ORT-hosted text generation.

- **mistral.rs (latest 2026):** core text generation across Llama 2–3, Mistral, Mixtral, Phi 2–4, Qwen 2/2.5/3, Gemma 1–3, DeepSeek V2/V3/R1. Quantization: GGUF, ISQ, AWQ, GPTQ, HQQ, FP8, BNB, AFQ. Tool calling, structured output (`llguidance`), speculative decoding, PagedAttention, X-LoRA, multi-GPU TP. VLMs: Phi-3.5-Vision, Phi-4-Multimodal, Idefics 2/3, LLaVA-Next, Qwen2.5-VL, Llama-3.2-Vision, MiniCPM-O, Gemma-3 vision. Newer (2025–early 2026): FLUX diffusion, Dia/Parler TTS, Whisper ASR, dedicated embedding endpoint. **Does not ship:** encoder embeddings (BGE/E5), rerankers, ROCm.
