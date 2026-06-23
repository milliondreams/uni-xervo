# Error Taxonomy

Uni-Xervo uses `RuntimeError` to separate config, load, API, and inference failures.

## Variants

- `Config(String)`
- `AliasNotFound { alias }`
- `ProviderNotFound(String)`
- `CapabilityMismatch(String)`
- `ProviderCapabilityMissing { alias, provider_id, capability }`
- `Load(String)`
- `ApiError(String)`
- `InferenceError(String)`
- `RateLimited`
- `Unauthorized`
- `Timeout`
- `Unavailable`

### ONNX-specific variants

`local/onnx` surfaces structured failures through dedicated variants:

- `OnnxModelNotFound { alias, path }`
- `OnnxArtifactSelectionFailure { alias, cause }`
- `OnnxDownloadFailure { alias, cause }`
- `OnnxLoadFailure { alias, path, cause }`
- `OnnxSignatureIntrospectionFailure { alias, cause }`
- `OnnxInputMissing { alias, required_input }`
- `OnnxInputTypeMismatch { alias, input_name, expected, got }`
- `OnnxInputShapeMismatch { alias, input_name, expected, got }`
- `OnnxInvocationFailure { alias, cause }`
- `OnnxBatchStackingFailure { alias, cause }`

## Retryability

`RuntimeError::is_retryable()` returns `true` for:

- `RateLimited`
- `Timeout`
- `Unavailable`

These are the only variants retried by instrumented wrappers when `retry` is configured.

## Remote HTTP mapping

Remote providers map HTTP status to runtime errors:

- `429` -> `RateLimited`
- `401`, `403` -> `Unauthorized`
- `5xx` -> `Unavailable`
- Other non-2xx -> `ApiError`

## Typical diagnosis workflow

1. `Config`: catalog/provider setup bug.
2. `ProviderNotFound`: provider not registered or not compiled.
3. `CapabilityMismatch`: requested typed handle does not match alias task/provider capability.
4. `Load`: provider initialization or model materialization failure.
5. `ApiError`/`InferenceError`: inspect provider response body and model input assumptions.

## ONNX-specific notes

`local/onnx` may also surface more specific runtime failures through `RuntimeError`, including:

- artifact selection failures,
- Hugging Face snapshot/download failures,
- missing inputs,
- tensor dtype mismatches,
- tensor shape mismatches,
- batch stacking/splitting failures.

In practice, ONNX issues usually fall into one of two buckets:

1. `Config` or `Load`: wrong `artifact`, unsupported execution provider, missing model files.
2. `InferenceError`: wrong tensor names, dtypes, or shapes for a loaded graph.
