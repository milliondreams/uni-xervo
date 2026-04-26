# GPU Runtime Architecture for uni-xervo

**Status:** Implemented in `0.5.4`.
**Companion docs:** [`docs/migrations/0.5.4-load-dynamic.md`](../migrations/0.5.4-load-dynamic.md), [`docs/proposals/onnx-reranker-followups.md`](onnx-reranker-followups.md)

## The problem this solves

Through `0.5.3`, uni-xervo's GPU story was:

- A single `gpu-cuda` cargo feature activated CUDA across `candle-core`, `fastembed`, `mistralrs`, and `ort` simultaneously.
- `ort` was built with `download-binaries`, which ships only a CPU-only static binary. Adding `ort/cuda` on top of `download-binaries` did **not** inject a CUDA execution provider into that binary — the CUDA EP only exists in Microsoft's *separate* GPU-flavored ONNX Runtime tarball.
- Users who enabled `gpu-cuda` got a build that *compiled* successfully and *ran* on CUDA hosts, but silently fell back to CPU at session creation because the CUDA EP DLL wasn't where ORT looked. There was no diagnostic API to detect this.

The result was a feature flag that lied. This proposal restructures GPU support around the natural fault line in the upstream Rust ML ecosystem: some crates support runtime device selection from a single binary, others bake the GPU backend into the type system at compile time.

## The split

| Provider | GPU selection | Build-time choice required |
|---|---|---|
| `LocalOnnxProvider` (ort) | runtime, per-session | only "use load-dynamic" once |
| `LocalOnnxRerankerProvider` (ort) | runtime, per-session | only "use load-dynamic" once |
| `LocalFastEmbedProvider` (ort wrapper) | runtime, per-session | only "use load-dynamic" once |
| `LocalCandleProvider` (candle) | runtime device choice (CPU vs GPU) | yes — pick CUDA *xor* Metal at build |
| `LocalMistralRsProvider` (mistralrs) | runtime device choice (CPU vs GPU) | yes — same as candle |

### Why ort does it

`ort` with the `load-dynamic` feature compiles the Rust crate to a tiny shim that `dlopen`s the ONNX Runtime shared library at first session creation. The library itself is provided at deploy time, not build time. Microsoft's official ONNX Runtime tarballs are *fat binaries* — one tarball contains pre-built CUDA, ROCm, TensorRT, DirectML, CoreML, OpenVINO, QNN execution providers as separate `.so` / `.dll` files.

At session creation, ORT iterates the EP list given to `Session::builder().with_execution_providers(...)`, silently skipping providers whose backing DLL isn't present, and attaches the first one that succeeds. Result: a single uni-xervo binary built with `ort/load-dynamic` (no `gpu-cuda`, no `gpu-rocm`) can run on NVIDIA Linux today, AMD Linux tomorrow, and Apple Silicon next week — by swapping `ORT_DYLIB_PATH`.

### Why candle and mistralrs don't

`candle-core` encodes the backend in the Rust type system. `Device::Cuda` exists only when `candle-core/cuda` is on at compile time; `Device::Metal` only when `metal` is on. The feature activates a different kernel binary baked into the artifact (PTX for CUDA, MSL `.metallib` for Metal). The two are mutually exclusive and not runtime-swappable.

Within a chosen backend, candle does still dynamic-load (`libcuda.so.1` via `cudarc`) — so a `cuda`-built binary doesn't crash on a CPU-only host, it just returns `Err` from `Device::new_cuda(0)`. mistralrs builds on candle and adds its own kernels with the same constraint.

## Two-tier architecture

The cleanest expression of this split:

### Tier 1 — universal binary

A uni-xervo build with **only `ort/load-dynamic`** (no GPU cargo features at all). One Rust artifact handles every supported vendor — the deploy-time ORT tarball and the per-spec `execution_providers` option pick the actual EP. Embed and rerank work everywhere. LLM generation through mistralrs is **CPU-only** in this tier.

This is the recommended default for libraries / CLIs distributed to end-users.

### Tier 2 — platform-specific binaries

Tier 1 + candle/mistralrs GPU. Choose one:

- **Linux + Windows + NVIDIA** (build host has `nvcc`): `gpu-cuda`. candle / mistralrs use CUDA; ort still picks at runtime.
- **macOS Apple Silicon**: `gpu-metal`. candle / mistralrs use Metal; ort still picks at runtime (typically Core ML).
- **Linux + AMD**: stay on Tier 1. mistralrs has no ROCm path; embed/rerank go through ort+ROCm.

The ORT runtime decision still happens per-session in Tier 2; only the LLM-gen path is build-locked.

## Cargo feature surface (`0.5.4`)

```toml
# Tier 2: NVIDIA. Build-host needs nvcc + matching driver.
gpu-cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda",
            "fastembed/cuda", "mistralrs/cuda", "ort/cuda"]

# Tier 2: Apple. macOS / iOS only. build.rs guards.
gpu-metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal",
             "fastembed/metal", "mistralrs/metal"]

# Tier 1 hints (ORT default-EP-list contributors).
gpu-rocm = ["ort/rocm"]              # Linux only
gpu-coreml = ["ort/coreml"]          # macOS / iOS only
gpu-directml = ["ort/directml"]      # Windows only
gpu-openvino = ["ort/openvino"]      # Linux + Windows
gpu-qnn = ["ort/qnn"]                # Windows ARM64
```

The Tier 1 hint features are **additive and optional**. Without any of them, the default EP list is `[Cpu]`; with them, the corresponding EP is added to the default list. Either way, the per-spec `options.execution_providers` overrides the default. A build with **no** GPU cargo features still runs CUDA at runtime when given an appropriate ORT tarball and a spec that requests CUDA.

The `build.rs` script enforces target-OS compatibility for each feature with copy-pasteable error messages.

## Per-spec EP configuration

Every ONNX-backed spec accepts an `execution_providers` option:

```json
{
  "alias": "rerank/minilm",
  "task": "Rerank",
  "provider_id": "local/onnx-reranker",
  "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
  "options": {
    "execution_providers": ["cuda", "coreml", "cpu"]
  }
}
```

Accepted forms: a JSON array of strings (`["cuda", "cpu"]`), or a single string (`"cuda"`). Recognized values: `cpu`, `cuda`, `coreml`, `directml`. Unknown values fall back to CPU (intentionally permissive).

ORT tries each EP in order and silently skips ones whose provider DLL is missing. To force-fail when a specific EP isn't available, omit `cpu` from the list — `build_execution_providers` flags the last entry with `error_on_failure()` and ORT will refuse to load.

## Runtime introspection

Both `OnnxRunner` and `RerankerModel` traits expose:

```rust
fn active_execution_providers(&self) -> Vec<String>;
```

Returns the requested EP list as stable string ids (e.g. `["cuda", "cpu"]`). **Caveats**:

- This reflects what was *requested*, not what *attached*. ORT 2.0's Rust binding doesn't yet expose `Session::providers()`. A session whose request list is `[cuda, cpu]` may have silently fallen back to CPU because the loaded ORT tarball lacks the CUDA EP DLL.
- For verifying actual attachment, watch ORT's tracing logs (`Successfully created CUDAExecutionProvider`) or run a smoke test that crashes-not-falls-back when the EP isn't real.

The trait method is sufficient for catching the most common misconfiguration (catalog spec didn't ask for the GPU).

## Cross-platform compatibility matrix

| OS / arch / GPU | Recommended path |
|---|---|
| Linux x64 + NVIDIA | Tier 2 `gpu-cuda` + ORT GPU tarball; or Tier 1 + ORT GPU tarball + `["cuda", "cpu"]` |
| Linux x64 + AMD | Tier 1 + ORT ROCm tarball + `["rocm", "cpu"]`; embed/rerank only |
| Linux x64 + Intel Arc | Tier 1 + ORT OpenVINO tarball + `["openvino", "cpu"]` |
| Linux x64 + integrated | Tier 1 + ORT CPU tarball |
| macOS Apple Silicon | Tier 2 `gpu-metal` + ORT macOS tarball; ort EPs `["coreml", "cpu"]` for ANE |
| Windows x64 + NVIDIA | Tier 2 `gpu-cuda` + ORT GPU tarball; or Tier 1 + DirectML |
| Windows x64 + AMD / Intel | Tier 1 + ORT GPU tarball + `["directml", "cpu"]` |
| Windows ARM64 (Snapdragon X) | Tier 1 + ORT ARM64 tarball + `["qnn", "cpu"]` for NPU or `["directml", "cpu"]` for GPU |
| iOS | Tier 1 + ORT iOS framework + `["coreml", "cpu"]` |

## What's still build-time

Documented for clarity:

1. **candle CUDA vs candle Metal** — different kernel binaries, mutually exclusive.
2. **mistralrs CUDA vs mistralrs Metal** — same.
3. **`gpu-cuda` with candle/mistralrs** — requires `nvcc` on the build host, requires `CUDA_COMPUTE_CAP` for cross-compile / GPU-less CI, hits the PTX-version trap if `nvcc` and the runtime driver mismatch (see [`onnx-reranker-followups.md`](onnx-reranker-followups.md) §4 and the inline doc in `tests/gpu_cuda_inference_test.rs`).

These are properties of the upstream crates, not uni-xervo's design. The migration shifts as much GPU decision-making as possible *out* of compile time and into the deploy-time tarball selection.

## Roadmap

- `wgpu` / `burn-wgpu` integration as a "universal fallback" (works on consumer NVIDIA / AMD / Intel / Apple without any vendor SDK). Out of scope for `0.5.4`.
- `Session::providers()` once ort exposes it — would let us return *attached* EPs, not just *requested*.
- An `ort` autofetch helper that downloads the right tarball on first run (Path C in the migration doc). Optional `provider-onnx-autofetch` feature.
