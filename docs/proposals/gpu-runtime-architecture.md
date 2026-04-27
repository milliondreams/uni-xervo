# GPU Runtime Architecture for uni-xervo

**Status:** Final, settled in `0.6.0`. This is the canonical design.
**Migration:** [`docs/migrations/0.6.0-final-feature-surface.md`](../migrations/0.6.0-final-feature-surface.md)

## Design intent

Every supported (vendor, target_os, target_arch) combination is reachable through a single cargo feature with no manual setup at build or runtime. Specifically:

- The user picks **one** GPU/CPU feature based on their target hardware.
- `cargo build --features <feature>` produces a working artifact (binary plus, where unavoidable, sidecar `.so`/`.dylib`/`.dll` files staged automatically next to the binary).
- The artifact runs on its target host with no `ORT_DYLIB_PATH`, no manual tarball download, no env-var setup beyond what the GPU vendor's driver itself requires (e.g. cuDNN for CUDA, ROCm for AMD).

## Per-provider GPU semantics

uni-xervo exposes three families of providers that each handle GPU differently:

### `LocalCandleProvider`, `LocalFastEmbedProvider` (candle path), `LocalMistralRsProvider`

candle and the candle-backed fastembed embed/rerank paths bundle their CUDA / Metal kernels into the Rust binary at build time as PTX or `.metallib`. At runtime they need only the GPU vendor's driver (`libcuda.so.1` for NVIDIA, the OS-level Metal framework for Apple). No separate ONNX Runtime in the picture.

These paths are fully bundled via the `gpu-cuda` (Linux+Windows+NVIDIA) and `gpu-metal` (macOS+iOS) cargo features. AMD ROCm and Intel are not supported because candle/mistralrs upstream don't.

### `LocalOnnxProvider`, `LocalFastEmbedProvider` (ONNX path)

These go through the `ort` crate, which itself wraps Microsoft's ONNX Runtime C++ library. The runtime is compiled as one of:

- **Static** (`ort/download-binaries`): pyke's prebuilt `libonnxruntime.a` is linked into the Rust binary. The accompanying provider-EP shared libraries (`libonnxruntime_providers_cuda.so` etc.) are staged into `target/<profile>/` by `ort/copy-dylibs` with `$ORIGIN` rpath.

- **Dynamic** (`ort/load-dynamic`): the Rust binary contains a small `dlopen` shim that loads a runtime-supplied `libonnxruntime.so` / `.dylib` / `.dll` via `ORT_DYLIB_PATH`. No build-time fetch.

uni-xervo's job is to drive the right combination based on which `gpu-*` cargo feature the user picked. The decision tree:

```
user picks gpu-X
├── pyke ships an X-flavored prebuilt? (gpu-cuda, gpu-tensorrt, gpu-coreml, gpu-wgpu)
│   → ort/download-binaries + ort/cuda (or ort/coreml etc.) + ort/copy-dylibs
│   → pyke-bundled, fully self-contained, no further uni-xervo work
│
├── Microsoft ships an X-flavored prebuilt? (gpu-directml, gpu-qnn)
│   → ort/load-dynamic + uni-xervo's build/ort_vendor.rs fetcher
│   → fetcher downloads MS's tarball at build time, stages dylibs in OUT_DIR
│     with absolute rpath into the binary. Functionally equivalent to pyke
│     bundle but requires custom build-script logic per vendor.
│
└── neither pyke nor Microsoft ships an X-flavored prebuilt? (gpu-rocm, gpu-openvino)
    → ort/load-dynamic + cargo:warning=
    → uni-xervo cannot bundle these. The vendor (AMD / Intel) ships their
      own onnxruntime build through their own distribution. User must add
      `provider-onnx-dynamic` to features and set ORT_DYLIB_PATH at deploy.
```

Each branch is mutually exclusive with the others — `build.rs` panics if more than one is active.

## Cargo feature surface (`0.6.0`)

```toml
default = ["provider-candle"]

# ─── Mode A — bundled CPU (pyke download-binaries) ───────────────────────────
provider-onnx = [
    "dep:ort", "dep:hf-hub", "dep:tokenizers",
    "ort/download-binaries", "ort/copy-dylibs", "ort/api-24",
    "fastembed?/ort-download-binaries",
]

# ─── Mode B — pyke-bundled GPU bundles ───────────────────────────────────────
gpu-cuda     = ["provider-onnx", "ort/cuda",     "candle-core/cuda", "candle-nn/cuda",
                "candle-transformers/cuda", "fastembed/cuda", "mistralrs/cuda"]
gpu-tensorrt = ["provider-onnx", "ort/nvrtx",    "candle-core/cuda", "candle-nn/cuda",
                "candle-transformers/cuda", "fastembed/cuda", "mistralrs/cuda"]
gpu-coreml   = ["provider-onnx", "ort/coreml"]
gpu-wgpu     = ["provider-onnx", "ort/webgpu"]

# ─── Mode C — Microsoft-fetched GPU bundles (build/ort_vendor.rs) ────────────
gpu-directml = ["_ort-fetched-base", "ort/directml"]
gpu-qnn      = ["_ort-fetched-base", "ort/qnn"]

# ─── Mode D — un-bundleable GPU vendors (build.rs warns; needs dynamic) ──────
gpu-rocm     = ["_ort-fetched-base", "ort/rocm"]
gpu-openvino = ["_ort-fetched-base", "ort/openvino"]

# ─── Mode E — power-user escape (BYO tarball) ────────────────────────────────
provider-onnx-dynamic = [
    "dep:ort", "dep:hf-hub", "dep:tokenizers", "dep:libloading",
    "ort/load-dynamic", "ort/api-24",
    "fastembed?/ort-load-dynamic",
]

# ─── Apple Metal — orthogonal to ort path (candle/mistralrs only) ────────────
gpu-metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal",
             "fastembed/metal", "mistralrs/metal"]

# Internal scaffolding (not user-facing).
_ort-fetched-base = [
    "dep:ort", "dep:hf-hub", "dep:tokenizers", "dep:libloading",
    "ort/load-dynamic", "ort/api-24",
    "fastembed?/ort-load-dynamic",
]
```

`build.rs` enforces:
1. Mutual exclusion among Modes A/C/E (Mode B activates A; Mode D activates C; the `_ort-fetched-base` marker keeps the panic message readable).
2. Target-OS guards: `gpu-metal` & `gpu-coreml` Apple-only; `gpu-directml` & `gpu-qnn` Windows-only; `gpu-rocm` Linux-only.

## Build-script architecture

`build/ort_vendor.rs` (~280 LOC). Responsibilities:

1. **Detect active vendor.** Scans `CARGO_FEATURE_GPU_*` env vars.
2. **For un-bundleable vendors** (`gpu-rocm`, `gpu-openvino`): emit `cargo:warning=` directives explaining the deployment requirement (`ORT_DYLIB_PATH` + vendor's runtime). No fetch attempt.
3. **For Microsoft-fetched vendors** (`gpu-directml`, `gpu-qnn`): look up the spec from the `VENDOR_SPECS` table for the current `(target_os, target_arch)`. Spec includes the artifact filename, SHA256, archive format, and lib subdirectory.
4. **Cache**: `~/.cache/uni-xervo/ort/<ORT_VERSION>/`. Idempotent across builds and crates.
5. **Bypass**: `UNI_XERVO_ORT_DIST_DIR=<dir>` skips the network step entirely. For sandboxed CI / offline builds.
6. **Download**: `ureq` (sync HTTP) with native-tls.
7. **Verify**: SHA256 against the pinned hash in the spec.
8. **Extract**: `flate2` + `tar` for `.tgz`, `zip` for `.zip`.
9. **Stage + emit cargo: directives**:
   - `cargo:rustc-link-search=native=<lib_dir>`
   - `cargo:rustc-link-arg=-Wl,-rpath,<absolute lib_dir path>`
   - `cargo:rustc-env=UNI_XERVO_ORT_RUNTIME_DIR=<lib_dir>` (informational)

The pyke-bundled vendors (`gpu-cuda`, `gpu-tensorrt`, `gpu-coreml`, `gpu-wgpu`) flow through `ort/download-binaries` + `ort/copy-dylibs` and require zero work from `build/ort_vendor.rs`.

## Deployment artifact shapes

| Mode | Build artifact |
|---|---|
| `provider-onnx` | One binary (~70 MB extra from static-linked CPU ORT). |
| `gpu-cuda` / `gpu-tensorrt` | Binary + 4-5 EP `.so` sidecars in the same dir (`$ORIGIN` rpath via ort/copy-dylibs). |
| `gpu-coreml` | One binary (CoreML EP is built into pyke's macOS bundle). |
| `gpu-wgpu` | One binary (WebGPU EP is built into pyke's wgpu bundle). |
| `gpu-directml` / `gpu-qnn` | Binary + dylibs in OUT_DIR (absolute rpath). For deployment to another host, copy dylibs alongside the binary (or use the `uni-xervo-bundle` helper, planned). |
| `gpu-rocm` / `gpu-openvino` | Tiny binary (load-dynamic shim). User supplies vendor's ORT lib + `ORT_DYLIB_PATH`. |
| `provider-onnx-dynamic` | Tiny binary. User supplies any ORT lib + `ORT_DYLIB_PATH`. |
| `gpu-metal` (alone) | One binary (candle/mistralrs Metal kernels embedded as `.metallib`). |

## Cross-platform support matrix

| Target | Bundled features that work |
|---|---|
| linux-x86_64 + NVIDIA | `gpu-cuda`, `gpu-tensorrt`, `gpu-wgpu`, `provider-onnx` |
| linux-x86_64 + AMD | `gpu-rocm` (load-dynamic + AMD's ORT), `gpu-wgpu` (vendor-neutral), `provider-onnx` |
| linux-x86_64 + Intel | `gpu-openvino` (load-dynamic + Intel's ORT), `gpu-wgpu`, `provider-onnx` |
| linux-aarch64 | `provider-onnx` only (no GPU pyke prebuilts for ARM Linux yet) |
| macos-aarch64 (Apple Silicon) | `gpu-coreml`, `gpu-metal`, `provider-onnx` |
| macos-x86_64 (Intel Mac) | `provider-onnx`, `gpu-metal` (intel iGPU/AMD discrete) |
| windows-x86_64 + NVIDIA | `gpu-cuda`, `gpu-tensorrt`, `gpu-directml`, `gpu-wgpu`, `provider-onnx` |
| windows-x86_64 + AMD/Intel | `gpu-directml`, `gpu-wgpu`, `provider-onnx` |
| windows-aarch64 (Snapdragon X) | `gpu-directml` (Adreno GPU), `gpu-qnn` (Hexagon NPU), `provider-onnx` |
| iOS / iPadOS | `gpu-coreml`, `gpu-metal`, `provider-onnx` |

## Why we settled here

The 0.5.x series went through three iterations trying to get the deployment story right:

- `0.5.4`: ort `load-dynamic`. Forced *every* user — CPU or GPU — to manage a tarball at deploy. Rejected the `download-binaries` path because pyke's CPU lib was thought to lack GPU EPs.
- `0.5.5`: deadlock workaround for the load-dynamic OnceLock bug. Useful but didn't address the deployment UX.
- `0.5.6`: split `provider-onnx` (bundled CPU) from `provider-onnx-dynamic`. Fixed the CPU UX but left every `gpu-*` user still managing tarballs.

`0.6.0` resolves it by **discovering that pyke actually does ship GPU bundles** (CUDA, TensorRT-RTX, CoreML built into macOS, WebGPU) — and by **filling the remaining gap with a build-script fetcher** that pulls Microsoft's prebuilts for the vendors pyke doesn't cover. The intersection of "user wanted this to work" and "uni-xervo can deliver it without external coordination" maps cleanly onto the cargo feature surface above.

`gpu-rocm` and `gpu-openvino` remain explicitly load-dynamic because there's no central prebuilt to fetch — that's an upstream coordination problem (AMD and Intel publish their own packages) and not something uni-xervo can solve unilaterally.

## What's deliberately not in this design

- **Bundling the GPU vendor driver itself.** We bundle ONNX Runtime + EP libraries, not the NVIDIA driver, ROCm kernel module, etc. Driver remains the host's responsibility (per Rust ML ecosystem norm; PyTorch doesn't bundle the CUDA driver either).
- **Static linking of GPU EPs.** Microsoft's GPU EPs (CUDA, ROCm, DirectML) are dynamic libraries by design. Static linking would require upstream changes we can't make.
- **Bundled binaries inside the crate.** The crate published to crates.io stays small (~few MB). Builds fetch from pyke / Microsoft on first run. Users without network at build time use `UNI_XERVO_ORT_DIST_DIR` to point at a pre-populated cache.
- **Per-platform Python wheels.** This is a uni-db concern (separate distribution layer). Outside this proposal's scope.

## Tracking

When upstream ORT releases (currently pinned to 1.25.0) update, `VENDOR_SPECS` in `build/ort_vendor.rs` needs the new SHA256s. Same for any new artifact filename pattern Microsoft introduces. The vendor specs table is the single source of truth.
