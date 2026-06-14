# local/onnx

## Uni-Xervo support

- Provider ID: `local/onnx`
- Feature flag: `provider-onnx` (or `provider-onnx-dynamic`)
- Capabilities: `raw`, `rerank`, `embed`, `embed_image`, `nlp`, `ocr`,
  `document_extract` (scaffold)

A single ONNX-Runtime-backed provider that dispatches by `task`:

- **`raw`** — arbitrary tensor execution (`RawTensorModel` trait).
- **`rerank`** — cross-encoder rerankers via `RerankerModel`.
- **`embed`** — dense text embeddings via `EmbeddingModel` (replaces the retired `local/fastembed` provider as of 0.8.0; the same alias strings still resolve).
- **`embed_image`** — ViT-style image embedders (SigLIP-2, CLIP, OpenCLIP)
  via `ImageEmbeddingModel`. New in 0.13.0.
- **`nlp`** — multi-head NLP cascades via `NlpModel`. Reference target:
  `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` (POS / NER / DEP / SRL
  / dialog-act CLS in one forward pass). New in 0.13.0; result surface
  extended in 0.14.0 (merged entities, full dialog-act distribution,
  word alignment, exposed label vocabularies). See the
  [Structured NLP guide](../../guides/nlp.md).
- **`ocr`** — CRNN + CTC-style text recognition via `OcrModel`. New in
  0.13.0.
- **`document_extract`** — VLM-based document parsing via
  `DocumentExtractionModel`. Scaffold only in 0.13.0: catalog wiring,
  options validation, the reusable `autoreg::greedy_decode` decoder
  helper, and three style-specific output parsers (Granite-Docling
  DocTags / MinerU Markdown / olmOCR Markdown) all ship and are unit
  tested. `extract()` returns `RuntimeError::Unavailable` until an
  upstream canonical ONNX export of one of the three target VLMs ships.

## Provider options

Options are validated per task (unknown keys are rejected with a precise `RuntimeError::Config`).

### Common keys (all tasks)

- `artifact` (string) — explicit `.onnx` filename within an HF repo (auto-detected if a single match exists).
- `max_batch_size` (integer)
- `execution_providers` — array of EP identifiers. Always recognized: `"cpu"`, `"cuda"`, `"coreml"`. When built with `provider-onnx-dynamic`, also recognized: `"rocm"`, `"directml"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"` (the user must supply a matching ORT library via `ORT_DYLIB_PATH`). Defaults to a feature-aware list: `["cuda", "cpu"]` under `gpu-cuda`, `["coreml", "cpu"]` under `gpu-metal`, `["cpu"]` otherwise. Vendor EPs are never default — opt in explicitly. **Per-spec**, so a single catalog can mix GPU and CPU placement on a `gpu-cuda` build (e.g. embedder on `["cpu"]`, reranker on `["cuda", "cpu"]`) — see the [mixed-catalog recipe](../../guides/onnx.md#mixing-gpu-and-cpu-placement-in-one-catalog). For `local/mistralrs` aliases the analogous per-spec switch is [`force_cpu`](mistralrs.md#device-placement) (mistralrs is candle-backed, not ORT-backed, so it doesn't accept this option). If a requested EP fails to initialize (missing cuDNN, wrong CUDA major version, etc.), ORT silently falls through to the next EP in the list — see [GPU setup → verifying the GPU EP actually loaded](../../guides/gpu-setup.md#verifying-the-gpu-ep-actually-loaded) for how to detect this. See `docs/migrations/0.9.0-feature-surface.md` for vendor-build setup.
- `graph_optimization_level` — `"disable" | "basic" | "extended" | "all"`.
- `inter_op_num_threads`, `intra_op_num_threads` (integers)
- `cache_dir` (string) — overrides `UNI_CACHE_DIR` and the default `.uni_cache/onnx-…/` location.

### Embed-only keys

- `pooling` — `"cls" | "mean" | "max" | "last-token"`. Required when no preset matches. Use `"last-token"` for decoder-style LLM embedders (Qwen3-Embedding, NV-Embed) that take the rightmost non-pad hidden state as the sentence representation; classic BERT-family embedders use `"cls"`, sentence-transformers use `"mean"`.
- `normalize` (bool, default `true`) — apply L2 normalization after pooling.
- `dimensions` (integer) — required when no preset matches; validated against the model's actual output.
- `max_seq_len` (integer, default `512`) — truncation cap for tokenizer input.
- `token_type_ids` (bool) — whether the model accepts a `token_type_ids` tensor (BERT-family yes; MPNet, ModernBERT, BAAI's M3 export, Qdrant's E5-large export, Qwen3-Embedding — no). Required for pass-through models.
- `tokenizer_path` (string, default `"tokenizer.json"`) — relative path within the HF repo.
- `output_name` (string) — explicit ONNX output to read; defaults to the first session output (typically `last_hidden_state`).

Decoder-LM embedders that declare `position_ids` and `past_key_values.*` placeholder inputs (e.g. `onnx-community/Qwen3-Embedding-0.6B-ONNX`) are auto-detected from the session metadata and fed correctly — no extra option is needed beyond `pooling: "last-token"`.

### Rerank-only keys

- `max_seq_len` (integer, default `512`)
- `style` — `"cross-encoder"` (default) or `"generative"`. Cross-encoder is for single-logit BERT-family models like `cross-encoder/ms-marco-MiniLM-L6-v2` and `BAAI/bge-reranker-base` (auto-detects whether the model takes a `token_type_ids` input). Generative is for decoder-LM rerankers like `Qwen3-Reranker-0.6B` that score by emitting `yes`/`no` token logits.
- `instruction` (string, generative only) — task description shown to the reranker. Default: `"Given a web search query, retrieve relevant passages that answer the query"`. Customize for domain-specific tasks (code search, medical literature, etc.).

### NLP-only keys (`task = nlp`)

- `onnx_path` (string, default `"onnx/cascade.onnx"`) — path within the HF
  repo to the cascade ONNX file.
- `tokenizer_path` (string, default `"tokenizer.json"`).
- `label_maps_path` (string, default `"label_maps.json"`) — JSON file
  containing the model's per-head id-to-label arrays (`pos`, `ner`,
  `srl`, `cls`, `deprel`). Each entry's position is the model's class
  index.
- `max_seq_len` (integer, default `128`) — chunking cap. Inputs longer
  than this are split into non-overlapping token windows; per-token byte
  offsets in `NlpToken::{start, end}` stay correct because tokenization
  runs once over the full text.

Canonical default model: `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`
(DeBERTa-v3-xsmall, 5 heads, ~75 MB INT8 ONNX). SRL multi-pass
orchestration (one forward per detected verb via `predicate_idx`) is
handled provider-side and invisible to callers.

### Image-embed-only keys (`task = embed_image`)

- `onnx_path` (string, required) — `.onnx` path within the HF repo.
- `image_size` (integer, required) — square input edge in pixels (e.g.
  384 for SigLIP-2-So400m-patch16-384).
- `dimensions` (integer, required) — expected output embedding dim.
- `normalization` (`"siglip"` | `"imagenet"`, default `"siglip"`) —
  SigLIP/SigLIP-2 mean=std=(0.5, 0.5, 0.5); ImageNet stats otherwise.
- `pool` (`"none"` | `"mean"`, default `"none"`) — `"none"` for
  pre-pooled `[batch, dim]` outputs; `"mean"` to mean-pool a
  `[batch, tokens, dim]` patch sequence.
- `normalize` (bool, default `true`) — L2-normalize each row.
- `output_name` (string, optional) — explicit output tensor name override.

Images may be supplied as `ImageInput::Bytes` (PNG/JPEG/WebP);
`ImageInput::Url` is rejected — callers fetch URLs upstream.

### OCR-only keys (`task = ocr`)

- `onnx_path` (string, required).
- `char_dict_path` (string, required) — character dictionary file within
  the HF repo, one entry per line. Class 0 is the CTC blank by default.
- `image_height` (integer, default 48), `image_width` (integer, default 320).
- `normalization` (`"siglip"` | `"imagenet"`, default `"imagenet"`).
- `blank_class` (integer, default 0).
- `output_name` (string, optional).

v1 is single-stage recognition only — callers pre-crop their text regions
and pass one image per region. Each call returns one `OcrBlock` per image
with normalized whole-image bbox and average softmax confidence.

### Document-extract-only keys (`task = document_extract`)

- `style` (`"granite-docling"` | `"mineru"` | `"olmocr"`, default
  `"granite-docling"`) — selects which output parser to apply.
- `onnx_path` (string, optional once the upstream model ships).
- `tokenizer_path` (string, optional).
- `max_seq_len` (integer, optional).

The shipped output parsers convert the VLM's decoded text into typed
`DocBlock` entries with reading order, optional bboxes, and a
concatenated `plain_markdown` field. End-to-end inference is gated on
an upstream canonical ONNX export — see the module doc comment in the
source for the expected ONNX input schema.

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
- **`embed_image`** → `runtime.image_embedder(alias)` returns
  `Arc<dyn ImageEmbeddingModel>`. Method: `embed(images)` → `EmbedResult`
  (vectors + optional `TokenUsage`).
- **`nlp`** → `runtime.nlp_model(alias)` returns `Arc<dyn NlpModel>`.
  Method: `analyze(requests)` → `Vec<NlpResult>` with populated `tokens`,
  `sentences`, `frames`, `speech_acts`, and merged `entities` per the requested
  `NlpTasks` bitflag. `NlpModel::label_maps()` exposes the model's per-head
  label vocabularies. See the [Structured NLP guide](../../guides/nlp.md) for
  the full result-type reference and a worked example.
- **`ocr`** → `runtime.ocr_model(alias)` returns `Arc<dyn OcrModel>`.
  Method: `recognize(images)` → `Vec<OcrResult>`.
- **`document_extract`** → `runtime.document_extractor(alias)` returns
  `Arc<dyn DocumentExtractionModel>`. Method: `extract(pages, options)`
  → `Vec<DocExtractResult>` (scaffold today; returns `Unavailable`).

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

### NLP (kniv-deberta canonical default)

```json
{
  "alias": "nlp/default",
  "task": "nlp",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
  "warmup": "background",
  "options": {
    "onnx_path": "onnx/cascade.onnx",
    "max_seq_len": 128
  }
}
```

### Image embed (SigLIP-2-style ViT)

```json
{
  "alias": "embed/siglip2",
  "task": "embed_image",
  "provider_id": "local/onnx",
  "model_id": "google/siglip2-so400m-patch16-384",
  "options": {
    "onnx_path": "onnx/model.onnx",
    "image_size": 384,
    "dimensions": 1152,
    "normalization": "siglip",
    "pool": "none"
  }
}
```

### OCR (CRNN + CTC recognition)

```json
{
  "alias": "ocr/paddle-rec",
  "task": "ocr",
  "provider_id": "local/onnx",
  "model_id": "OpenCV/paddleocr-rec-en",
  "options": {
    "onnx_path": "model.onnx",
    "char_dict_path": "char_dict.txt",
    "image_height": 48,
    "image_width": 320,
    "normalization": "imagenet",
    "blank_class": 0
  }
}
```

### Document extract (scaffold; returns Unavailable until wired)

```json
{
  "alias": "doc/granite",
  "task": "document_extract",
  "provider_id": "local/onnx",
  "model_id": "ibm-granite/granite-docling-258M",
  "options": {
    "style": "granite-docling"
  }
}
```

## External references

- ONNX Runtime docs: <https://onnxruntime.ai/docs/>
- Hugging Face ONNX docs: <https://huggingface.co/docs/optimum/exporters/onnx/overview>
