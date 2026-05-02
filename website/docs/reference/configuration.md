# Configuration

Catalogs are JSON arrays of `ModelAliasSpec` entries.

## Canonical schema shape

```json
[
  {
    "alias": "embed/default",
    "task": "embed",
    "provider_id": "local/candle",
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": null,
    "warmup": "lazy",
    "required": false,
    "timeout": 10,
    "load_timeout": 600,
    "retry": {
      "max_attempts": 3,
      "initial_backoff_ms": 100
    },
    "options": null
  }
]
```

## Field constraints

- `alias`: string matching `.+/.+`.
- `task`: one of `embed`, `rerank`, `generate`, `raw`.
- `warmup`: one of `eager`, `lazy`, `background`.
- `timeout`, `load_timeout`: integer >= 1.
- `retry.max_attempts`, `retry.initial_backoff_ms`: integer >= 1.
- `options`: object or null, strict provider-specific keys only.

## Provider options reference

| Provider ID | Allowed option keys | Notes |
| --- | --- | --- |
| `local/candle` | `cache_dir` | Per-model local cache path |
| `local/onnx` | `artifact`, `max_batch_size`, `execution_providers`, `graph_optimization_level`, `inter_op_num_threads`, `intra_op_num_threads`, `cache_dir` (all tasks); `max_seq_len`, `style`, `instruction` (rerank); `pooling` (`cls`/`mean`/`max`/`last-token`), `normalize`, `dimensions`, `token_type_ids`, `tokenizer_path`, `output_name`, `max_seq_len` (embed) | Per-task ONNX Runtime configuration. Rerank `style` is `cross-encoder` (default; single-logit BERT-style models like `BAAI/bge-reranker-base`) or `generative` (decoder-LM rerankers like Qwen3-Reranker that emit `yes`/`no` token logits). Embed `pooling: "last-token"` is for decoder-style embedders (Qwen3-Embedding, NV-Embed). The embed lane subsumes the retired `local/fastembed` provider as of 0.8.0; FastEmbed alias strings such as `BGESmallENV15` still resolve via the built-in preset table |
| `local/mistralrs` | `pipeline`, `dtype`, `isq`, `uqff_files`, `force_cpu`, `paged_attention`, `max_num_seqs`, `max_seq_len`, `max_batch_size`, `max_image_shape`, `max_num_images`, `chat_template`, `tokenizer_json`, `embedding_dimensions`, `gguf_files`, `diffusion_loader_type`, `speech_loader_type` | Multimodal pipelines (text, vision, diffusion, speech), quantization (in-situ via `isq` or pre-quantized via `uqff_files`), auto-device-mapper overrides, and local runtime tuning |
| `remote/openai` | `api_key_env`, `base_url`, `embedding_dimensions` | Override env var name for API key; override base URL (for OpenAI-compatible servers); override embedding dimensions |
| `remote/gemini` | `api_key_env`, `api_version`, `embedding_dimensions` | `api_version` defaults to `v1beta`; override embedding dimensions |
| `remote/vertexai` | `api_token_env`, `project_id`, `location`, `publisher`, `embedding_dimensions` | OAuth token + project/location metadata |
| `remote/mistral` | `api_key_env`, `embedding_dimensions` | Override env var name for API key; override embedding dimensions |
| `remote/anthropic` | `api_key_env`, `anthropic_version` | `anthropic_version` defaults to `2023-06-01` |
| `remote/voyageai` | `api_key_env`, `embedding_dimensions` | Override env var name for API key; override embedding dimensions |
| `remote/cohere` | `api_key_env`, `input_type`, `embedding_dimensions` | `input_type` used for embedding mode; override embedding dimensions |
| `remote/azure-openai` | `api_key_env`, `resource_name`, `api_version`, `embedding_dimensions` | `resource_name` required; `api_version` default `2024-10-21`; override embedding dimensions |

Provider-specific model/config links:

- [Provider Reference Pages](providers/index.md)

## Runtime builder paths

- Programmatic catalog: `.catalog(Vec<ModelAliasSpec>)`
- JSON string catalog: `.catalog_from_str(&str)`
- JSON file catalog: `.catalog_from_file(path)`

## Helpful APIs

- `runtime.contains_alias(alias)`
- `runtime.prefetch_all()`
- `runtime.prefetch(&[aliases])`
- `runtime.embedding(alias)`
- `runtime.reranker(alias)`
- `runtime.generator(alias)`
- `runtime.raw_tensor_model(alias)`
