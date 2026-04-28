# local/onnx

## Uni-Xervo support

- Provider ID: `local/onnx`
- Feature flag: `provider-onnx` (or `provider-onnx-dynamic`)
- Capabilities: `raw`, `rerank`, `embed`

A single ONNX-Runtime-backed provider that dispatches by `task` to three task implementations:

- **`raw`** — arbitrary tensor execution (`RawTensorModel` trait).
- **`rerank`** — cross-encoder rerankers via `RerankerModel`.
- **`embed`** — dense text embeddings via `EmbeddingModel` (replaces the retired `local/fastembed` provider as of 0.8.0; the same alias strings still resolve).

## Provider options

Options are validated per task (unknown keys are rejected with a precise `RuntimeError::Config`).

### Common keys (all tasks)

- `artifact` (string) — explicit `.onnx` filename within an HF repo (auto-detected if a single match exists).
- `max_batch_size` (integer)
- `execution_providers` — array of `"cpu"`, `"cuda"`, `"coreml"`. Defaults to a feature-aware list: `["cuda", "cpu"]` under `gpu-cuda`, `["coreml", "cpu"]` under `gpu-metal`, `["cpu"]` otherwise. For other vendor EPs (ROCm, DirectML, OpenVINO, QNN, TensorRT, WebGPU), use `provider-onnx-dynamic` with a vendor-supplied ORT build via `ORT_DYLIB_PATH` — see `docs/migrations/0.9.0-feature-surface.md`.
- `graph_optimization_level` — `"disable" | "basic" | "extended" | "all"`.
- `inter_op_num_threads`, `intra_op_num_threads` (integers)
- `cache_dir` (string) — overrides `UNI_CACHE_DIR` and the default `.uni_cache/onnx-…/` location.

### Embed-only keys

- `pooling` — `"cls" | "mean" | "max"`. Required when no preset matches.
- `normalize` (bool, default `true`) — apply L2 normalization after pooling.
- `dimensions` (integer) — required when no preset matches; validated against the model's actual output.
- `max_seq_len` (integer, default `512`) — truncation cap for tokenizer input.
- `token_type_ids` (bool) — whether the model accepts a `token_type_ids` tensor (BERT-family yes; MPNet, ModernBERT, BAAI's M3 export, Qdrant's E5-large export — no). Required for pass-through models.
- `tokenizer_path` (string, default `"tokenizer.json"`) — relative path within the HF repo.
- `output_name` (string) — explicit ONNX output to read; defaults to the first session output (typically `last_hidden_state`).

### Rerank-only keys

- `max_seq_len` (integer, default `512`)

Authoritative option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/onnx.schema.json>

## Model IDs

`model_id` can be:

- a Hugging Face repo ID (e.g. `"BAAI/bge-small-en-v1.5"`), or
- a built-in alias for the embed task (e.g. `"BGESmallENV15"`, `"all-MiniLM-L6-v2"` — see [Embedding presets](#embedding-presets) below), or
- a local path to a `.onnx` file (raw / rerank tasks).

HF-backed aliases download into the per-task cache directory (`onnx-raw`, `onnx-reranker`, or `onnx-embed`) before ORT session creation. If the repo contains multiple `.onnx` files, set `options.artifact`.

## Embedding presets

The embed task ships with a preset table covering 25 popular text-embedding models. When `model_id` matches a preset alias, the HF repo, ONNX path, pooling kind, dimensions, and `token_type_ids` flag are filled in automatically; per-spec `options` may still override any field.

| Alias (canonical) | HF repo | Dim | Pooling | `token_type_ids` |
|---|---|---:|---|:---:|
| `AllMiniLML6V2` (`all-MiniLM-L6-v2`) | Qdrant/all-MiniLM-L6-v2-onnx | 384 | mean | ✓ |
| `AllMiniLML6V2Q` | Xenova/all-MiniLM-L6-v2 (quantized) | 384 | mean | ✓ |
| `AllMiniLML12V2` | Xenova/all-MiniLM-L12-v2 | 384 | mean | ✓ |
| `AllMiniLML12V2Q` | Xenova/all-MiniLM-L12-v2 (quantized) | 384 | mean | ✓ |
| `AllMpnetBaseV2` (`all-mpnet-base-v2`) | Xenova/all-mpnet-base-v2 | 768 | mean | ✗ |
| `BGESmallENV15` (`bge-small-en-v1.5`) | Xenova/bge-small-en-v1.5 | 384 | cls | ✓ |
| `BGESmallENV15Q` | Qdrant/bge-small-en-v1.5-onnx-Q | 384 | cls | ✓ |
| `BGEBaseENV15` (`bge-base-en-v1.5`) | Xenova/bge-base-en-v1.5 | 768 | cls | ✓ |
| `BGEBaseENV15Q` | Qdrant/bge-base-en-v1.5-onnx-Q | 768 | cls | ✓ |
| `BGELargeENV15` (`bge-large-en-v1.5`) | Xenova/bge-large-en-v1.5 | 1024 | cls | ✓ |
| `BGELargeENV15Q` | Qdrant/bge-large-en-v1.5-onnx-Q | 1024 | cls | ✓ |
| `BGESmallZHV15` | Xenova/bge-small-zh-v1.5 | 512 | cls | ✓ |
| `BGELargeZHV15` | Xenova/bge-large-zh-v1.5 | 1024 | cls | ✓ |
| `BGEM3` | BAAI/bge-m3 (external data) | 1024 | cls | ✗ |
| `NomicEmbedTextV1` | nomic-ai/nomic-embed-text-v1 | 768 | mean | ✓ |
| `NomicEmbedTextV15` (`nomic-embed-text-v1.5`) | nomic-ai/nomic-embed-text-v1.5 | 768 | mean | ✓ |
| `NomicEmbedTextV15Q` | nomic-ai/nomic-embed-text-v1.5 (quantized) | 768 | mean | ✓ |
| `ParaphraseMLMiniLML12V2` | Xenova/paraphrase-multilingual-MiniLM-L12-v2 | 384 | mean | ✓ |
| `ParaphraseMLMiniLML12V2Q` | Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q | 384 | mean | ✓ |
| `ParaphraseMLMpnetBaseV2` | Xenova/paraphrase-multilingual-mpnet-base-v2 | 768 | mean | ✗ |
| `MultilingualE5Small` (`multilingual-e5-small`) | intfloat/multilingual-e5-small | 384 | mean | ✓ |
| `MultilingualE5Base` (`multilingual-e5-base`) | intfloat/multilingual-e5-base | 768 | mean | ✗ |
| `MultilingualE5Large` (`multilingual-e5-large`) | Qdrant/multilingual-e5-large-onnx (external data) | 1024 | mean | ✗ |
| `MxbaiEmbedLargeV1` (`mxbai-embed-large-v1`) | mixedbread-ai/mxbai-embed-large-v1 | 1024 | cls | ✓ |
| `ModernBertEmbedLarge` | lightonai/modernbert-embed-large | 1024 | mean | ✗ |

All 25 presets normalize output to L2 norm ≈ 1.0 and use `max_seq_len: 512` to match `fastembed-rs` defaults.

## Runtime contract

By task:

- **`raw`** → `runtime.raw_tensor_model(alias)` returns `Arc<dyn RawTensorModel>`. Methods: `run`, `run_batch`, `input_signature`, `output_signature`, `max_batch_size`, `active_execution_providers`.
- **`rerank`** → `runtime.reranker(alias)` returns `Arc<dyn RerankerModel>`. Method: `rerank(query, docs)` → `Vec<ScoredDoc>`.
- **`embed`** → `runtime.embedding(alias)` returns `Arc<dyn EmbeddingModel>`. Method: `embed(texts)` → `Vec<Vec<f32>>` (each row of length `dimensions()`, L2-normalized when `normalize: true`).

## Example catalog entries

### Embed (preset alias)

```json
{
  "alias": "embed/local",
  "task": "Embed",
  "provider_id": "local/onnx",
  "model_id": "BGESmallENV15"
}
```

### Embed (pass-through, custom HF model)

```json
{
  "alias": "embed/custom",
  "task": "Embed",
  "provider_id": "local/onnx",
  "model_id": "Snowflake/snowflake-arctic-embed-m",
  "options": {
    "artifact": "onnx/model.onnx",
    "pooling": "cls",
    "dimensions": 768,
    "token_type_ids": true,
    "execution_providers": ["cuda", "cpu"]
  }
}
```

### Rerank

```json
{
  "alias": "rerank/cross",
  "task": "Rerank",
  "provider_id": "local/onnx",
  "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2"
}
```

### Raw tensor

```json
{
  "alias": "raw/classifier",
  "task": "Raw",
  "provider_id": "local/onnx",
  "model_id": "smokxy/sequence_classification_onnx",
  "options": {
    "artifact": "model.onnx",
    "execution_providers": ["cpu"]
  }
}
```

## External references

- ONNX Runtime docs: <https://onnxruntime.ai/docs/>
- Hugging Face ONNX docs: <https://huggingface.co/docs/optimum/exporters/onnx/overview>
