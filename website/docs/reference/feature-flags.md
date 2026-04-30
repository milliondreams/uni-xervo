# Feature Flags

Uni-Xervo providers are feature-gated.

## Provider features

- `provider-candle`
- `provider-onnx` (covers raw, rerank, and embed tasks via `local/onnx`; replaces the retired `provider-fastembed`)
- `provider-onnx-dynamic` (same surface as `provider-onnx`, load-dynamic ORT linking mode)
- `provider-mistralrs`
- `provider-openai`
- `provider-gemini`
- `provider-vertexai`
- `provider-mistral`
- `provider-anthropic`
- `provider-voyageai`
- `provider-cohere`
- `provider-azure-openai`

## Acceleration features

- `gpu-cuda`

## Cargo examples

```toml
# Local-only footprint
uni-xervo = { version = "0.5.0", default-features = false, features = [
  "provider-candle",
  "provider-onnx"
] }

# Remote-only footprint
uni-xervo = { version = "0.5.0", default-features = false, features = [
  "provider-openai",
  "provider-cohere",
  "provider-vertexai"
] }
```

## Runtime registration reminder

Enabling features compiles provider code; it does not auto-register providers.

Register each provider in `ModelRuntime::builder()` before `build()`.
