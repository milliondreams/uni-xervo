# Changelog

All notable changes to this project are documented in this file.

## [0.5.5] - 2026-04-25

### Fixed

- **`provider-onnx` no longer hangs indefinitely when the ONNX Runtime dynamic library is missing.** Previously, a misconfigured `ORT_DYLIB_PATH` (or its absence on a host without system ORT) would cause any ORT-backed session creation to block forever in `futex_wait`. Root cause: an upstream re-entrant `OnceLock` deadlock in `ort` 2.0.0-rc.12's load-dynamic error path ([pykeio/ort#560](https://github.com/pykeio/ort/issues/560), fixed upstream in [`17ed7277`](https://github.com/pykeio/ort/commit/17ed7277) but **not yet released** as of `=2.0.0-rc.12`).
- New pre-flight check `provider::onnx_ep::preflight_ort_dylib` runs before any ORT API call in `LocalOnnxProvider::load` and `OnnxCrossEncoder::load`. It attempts `libloading::Library::new` against `ORT_DYLIB_PATH` (or the platform default) and converts a load failure into a clear `RuntimeError::Config` in milliseconds, with a pointer to `docs/migrations/0.5.4-load-dynamic.md`.
- New regression test `tests/preflight_dylib_test.rs` — guards against re-introducing the hang. Both tests complete in <10ms.

### Added

- `libloading = "0.8"` as an optional dep gated by `provider-onnx`. Already present transitively via ort's `load-dynamic`; we now expose it directly so the preflight can use it.

## [0.5.4] - 2026-04-25

### Breaking

- The `ort` crate is now compiled with `load-dynamic` instead of `download-binaries`. The CPU-only ONNX Runtime binary that previously shipped with the Rust crate is no longer bundled. Consumers must download Microsoft's official ONNX Runtime release tarball matching their target hardware and set `ORT_DYLIB_PATH` (or rely on system library paths) before any ORT-backed provider session is created. See `docs/migrations/0.5.4-load-dynamic.md` for step-by-step instructions and `docs/proposals/gpu-runtime-architecture.md` for the rationale.
- `fastembed` similarly switched from `ort-download-binaries` to `ort-load-dynamic`. Same deployment requirement.
- The previous "build with `gpu-cuda`, get CUDA at runtime" promise was already broken (the bundled ORT had no CUDA EP); load-dynamic makes the contract honest. Existing `LocalOnnxProvider` / `LocalOnnxRerankerProvider` users on CPU-only hosts must now point `ORT_DYLIB_PATH` at the CPU tarball before tests/binaries run.

### Added

- **Runtime-selectable execution providers via ORT load-dynamic.** A single uni-xervo binary can now run on NVIDIA / AMD / Apple / DirectML / Intel hardware by swapping the ORT tarball at deploy time — no rebuild required. Per-spec EP selection through `options.execution_providers` (already supported) is now the canonical knob.
- New cargo features as **default-EP-list hints** (additive over the universal load-dynamic base):
  - `gpu-rocm = ["ort/rocm"]` — Linux only.
  - `gpu-openvino = ["ort/openvino"]` — Linux + Windows.
  - `gpu-qnn = ["ort/qnn"]` — Windows ARM64 (Snapdragon Hexagon NPU).
- `gpu-metal` is no longer commented out — declaring the feature is safe on all platforms; the `build.rs` guard fails fast if you try to *activate* it on a non-Apple target.
- `OnnxRunner::active_execution_providers()` and `RerankerModel::active_execution_providers()` — return the requested EP list as stable string ids (e.g. `["cuda", "cpu"]`). Default impls return empty for non-ORT runners. Documents the "requested vs. attached" distinction (the ORT 2.0 Rust binding doesn't yet expose `Session::providers()`).
- Build script (`build.rs`) now guards `gpu-metal`, `gpu-coreml`, `gpu-directml`, `gpu-rocm`, `gpu-qnn` to their respective target operating systems with copy-pasteable error messages.
- New tests: `tests/onnx_ep_resolution_test.rs` (3 tests, no I/O) verifies feature-gated EP validation runs *before* any HF download — wrong EP requests fail fast with a clear `Config` error.
- Inline `active_execution_providers()` assertion added to the existing reranker e2e test.

### Changed

- `gpu-cuda` is now an opt-in *hint* rather than a load-bearing build flag for the ORT path. The CUDA EP is available at runtime to any load-dynamic binary whose ORT tarball includes it. The candle/mistralrs CUDA paths still require `gpu-cuda` (and `nvcc` at build time) — that's a structural property of those upstream crates.
- `OnnxCrossEncoder::load` now validates the requested execution-provider list **before** downloading any HF assets. A misconfigured spec (e.g. CUDA requested without `gpu-cuda` enabled) errors out instantly instead of after a multi-megabyte download.
- The `Cargo.toml` GPU feature block is reorganized into "Tier 1" (runtime-selectable through ORT) and "Tier 2" (build-locked through candle/mistralrs) groups with inline rationale.

## [0.5.3] - 2026-04-25

### Added
- Execution-provider selection for `LocalOnnxRerankerProvider`. Specify `"execution_providers": ["cuda"]` (or `["cuda", "cpu"]`, etc.) in the catalog spec options to run reranking on CUDA / CoreML / DirectML, mirroring the knob already on `LocalOnnxProvider`. Without the option, defaults to `[Cuda, Cpu]` when `gpu-cuda` is enabled, `[Cpu]` otherwise.
- New `gpu_cuda_inference_test` module with `EXPENSIVE_TESTS=1`-gated tests that exercise reranker, embed, and mistralrs generate on CUDA. Includes inline doc on host-side requirements (ORT-CUDA shared libs, candle-cuda PTX-toolchain match).

### Changed
- Extracted shared ONNX execution-provider selection (`OnnxExecutionProvider` enum, `build_execution_providers`, `default_execution_providers`, `parse_execution_providers_option`) into a new `provider::onnx_ep` module. Both `LocalOnnxProvider` and `LocalOnnxRerankerProvider` import from it; the duplicated copy that previously lived in `local_onnx.rs` is gone.
- `LocalOnnxOptions` (used by `LocalOnnxProvider`) now accepts `execution_providers` as either a JSON array (`["cuda", "cpu"]`) or a single string (`"cuda"`); previously only the array form parsed.

## [0.5.2] - 2026-04-25

### Added
- `LocalOnnxRerankerProvider` (`local/onnx-reranker`) — local ONNX cross-encoder reranker (BERT-style models such as `cross-encoder/ms-marco-MiniLM-L6-v2`). Handles HF download, WordPiece tokenization, and batched ORT inference. Returns raw logits as `ScoredDoc`s sorted descending; the caller is responsible for any normalization (sigmoid, etc.). Hoisted from uni-db so all uni-xervo consumers get local rerank without re-implementing it. Gated by the existing `provider-onnx` feature.
- Deferred-followups proposal at `docs/proposals/onnx-reranker-followups.md` capturing four enhancements considered during the hoist (score-normalization API, shared HF download helper, BERT-pair tokenizer extraction, ORT session pool).

### Changed
- `provider-onnx` feature now also pulls `tokenizers` (previously only gated by `provider-candle`), required by the new reranker provider.

## [0.4.0] - 2026-04-15

### Added
- Configurable `embedding_dimensions` option for all remote embedding providers (OpenAI, Gemini, Cohere, Mistral, Azure OpenAI, Voyage AI). Overrides the default dimensions reported by the model handle, enabling immediate adoption of new upstream models without a crate release.
- Configurable `api_version` option for the Gemini provider (default: `v1beta`), allowing users to switch API versions as Google promotes endpoints to stable.

### Fixed
- Corrected default embedding dimensions for newer upstream models: `gemini-embedding-001` (3072), Cohere `embed-v4.0` (1536), Mistral `codestral-embed` (1536), Voyage AI `voyage-3-lite` (512).

## [0.3.0] - 2026-04-13

### Breaking Changes
- Updated all dependencies to latest versions and migrated to mistralrs 0.8 API.

### Changed
- Upgraded `mistralrs` from 0.4 to 0.8.
- Upgraded `candle-core`, `candle-nn`, `candle-transformers` to 0.10.
- Upgraded `tokenizers` to 0.22.
- Upgraded `fastembed` to 5.9.0.
- Upgraded `reqwest` to 0.13.
- Upgraded `thiserror` to 2.
- Upgraded `criterion` (dev) to 0.8.

## [0.2.0] - 2026-03-12

### Breaking Changes
- `GeneratorModel::generate()` signature changed: `messages: &[String]` → `messages: &[Message]`.
- `GenerationResult` now has two additional required fields: `images: Vec<GeneratedImage>`, `audio: Option<AudioOutput>`.
- All provider implementations updated accordingly.
- Migration: replace `&["text".to_string()]` with `&[Message::user("text")]`.

### Added
- **Multimodal message types**: `Message`, `MessageRole`, `ContentBlock`, `ImageInput` for structured conversation input.
- **Vision generation**: Process images + text via mistralrs vision pipeline (`"pipeline": "vision"`).
- **Image generation**: Diffusion pipeline (FLUX) via mistralrs (`"pipeline": "diffusion"`).
- **Speech synthesis**: Audio generation (Dia) via mistralrs (`"pipeline": "speech"`).
- **GGUF model support**: Load quantized GGUF models in mistralrs text pipeline.
- **dtype control**: Configure model precision (`f32`, `f16`, `bf16`, `auto`) for mistralrs pipelines.
- **ISQ quantization**: In-situ quantization support for text and vision pipelines.
- **Embedding validation**: NaN/Inf detection in embedding outputs.
- **Explicit message roles**: `System`, `User`, `Assistant` roles replace index-based role inference.
- `GenerationOptions` gains `width` and `height` fields for diffusion image sizing.

## [0.1.1] - 2026-02-23

### Changed
- Upgraded `thiserror` from 1.0 to 2, aligning with the transitive dependency ecosystem.
- Upgraded `reqwest` from 0.11 to 0.12, replacing the legacy `hyper` 0.14 / `http` 0.2 HTTP stack with the modern `hyper` 1.x stack and eliminating several duplicate transitive dependencies.

## [0.1.0] - 2026-02-19

### Added
- Provider options schema files and runtime validation for provider-specific options.
- Dedicated public API and error taxonomy improvements with clearer error variants.
- Minimal GitHub Actions workflows for CI and crates.io publishing.
- Expanded website documentation content aligned with Uni-Xervo branding.

### Changed
- Improved runtime load timeout handling and reliability behavior.
- Refined packaging metadata and include list for cleaner crates.io distribution.

### Fixed
- Correct provider capability declarations for Gemini embedding support.
- Gated mock module export for test/testing builds only.
