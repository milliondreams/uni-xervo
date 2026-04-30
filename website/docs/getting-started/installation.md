# Installation

Add Uni-Xervo to your Rust project.

## Default install — everything except GPU

```toml
[dependencies]
uni-xervo = "0.9"
```

Defaults give you all three local backends (`provider-candle`, `provider-mistralrs`, `provider-onnx`) and all eight remote providers (`provider-openai`, `provider-gemini`, `provider-vertexai`, `provider-mistral`, `provider-anthropic`, `provider-voyageai`, `provider-cohere`, `provider-azure-openai`) on CPU.

## Lean build — pick what you need

```toml
[dependencies]
uni-xervo = { version = "0.9", default-features = false, features = [
  "provider-candle",
  "provider-onnx",
] }
```

Pass `default-features = false` and select explicitly to shrink build time and binary size. Remote-only and local-only footprints are both common.

## GPU acceleration

GPU is opt-in and additive — CPU is always the fallback. Pick one based on your target hardware:

```toml
# NVIDIA on Linux / Windows
uni-xervo = { version = "0.9", features = ["gpu-cuda"] }

# Apple GPU + Neural Engine on macOS / iOS
uni-xervo = { version = "0.9", features = ["gpu-metal"] }
```

- `gpu-cuda` requires `nvcc` and `CUDA_COMPUTE_CAP` on the build host (for candle / mistralrs PTX generation). The ORT side is fetched automatically by pyke.
- `gpu-metal` only builds on Apple targets (`build.rs` panics elsewhere). Activates the CoreML EP for ORT plus Metal kernels for candle / mistralrs.

For ORT-backed providers, CUDA builds default `execution_providers` to `["cuda", "cpu"]` and Metal builds default to `["coreml", "cpu"]` unless you override per alias.

For other GPU vendors (AMD ROCm, Intel OpenVINO, Microsoft DirectML, Qualcomm QNN, TensorRT, WebGPU), build with `provider-onnx-dynamic`, point `ORT_DYLIB_PATH` at a vendor-supplied ORT library at deploy, and request the EP per alias by setting `execution_providers` to `"rocm"`, `"directml"`, `"openvino"`, `"qnn"`, `"tensorrt"`, or `"webgpu"` (typically chained with `"cpu"` as a fallback). See [`docs/migrations/0.9.0-feature-surface.md`](https://github.com/rustic-ai/uni-xervo/blob/main/docs/migrations/0.9.0-feature-surface.md).

## Remote auth environment variables

Set the variables for providers you use:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `VERTEX_AI_TOKEN`
- `VERTEX_AI_PROJECT` (optional fallback for Vertex project)
- `MISTRAL_API_KEY`
- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY`
- `CO_API_KEY`
- `AZURE_OPENAI_API_KEY`

You can override key variable names per alias with provider options such as `api_key_env` or `api_token_env`.

## Next

- [Quick Start](quickstart.md)
- [Feature Flags](../reference/feature-flags.md)
