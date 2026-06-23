# ONNX Usage Patterns

This page maps common ONNX use cases to the Uni-Xervo developer flow.

## Tabular regression or classification

Best fit for `local/onnx`.

Developer flow:

1. Build numeric feature tensors in app code.
2. Put them into `TensorBatch`.
3. Call `run()`.
4. Read scalar or logits outputs.

Why it works well:

- no tokenizer,
- usually one or two tensors,
- minimal postprocessing.

## Sequence classification

Examples:

- sentiment
- topic classification
- moderation

Developer flow:

1. Tokenize text in app code.
2. Build `input_ids` and `attention_mask`.
3. Call `run()`.
4. Read logits and map labels.

This is one of the best HF ONNX patterns for Uni-Xervo today.

## Token classification / NER

Examples:

- NER
- PII tagging
- slot extraction

Developer flow:

1. Tokenize with offsets.
2. Build transformer input tensors.
3. Call `run()`.
4. Read token logits.
5. Reconstruct spans in app code.

Uni-Xervo handles execution cleanly here, but the token-to-span logic still belongs in your application.

## Dense text embedding (`Embed` task)

As of 0.8.0, `local/onnx` serves the `Embed` task directly — this replaces the retired `local/fastembed` provider. All 25 popular text-embedding aliases (`BGESmallENV15`, `AllMiniLML6V2`, `NomicEmbedTextV15`, `MultilingualE5Base`, …) ship as built-in presets that resolve the HF repo, ONNX path, pooling kind, dimensions, and `token_type_ids` automatically.

Catalog config (preset alias):

```json
{
  "alias": "embed/local",
  "task": "Embed",
  "provider_id": "local/onnx",
  "model_id": "BGESmallENV15"
}
```

Catalog config (custom HF model, pass-through):

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
    "token_type_ids": true
  }
}
```

Developer flow:

1. Resolve the typed handle: `let embedder = runtime.embedding("embed/local").await?;`
2. Call `embedder.embed(&["hello world", "second doc"]).await?` — Uni-Xervo handles tokenization, ORT session execution, pooling, and L2 normalization.
3. Each row of `result.vectors` is `Vec<f32>` of length `embedder.dimensions()`.

If you have a custom export that returns hidden states and you want to handle pooling yourself, use the `Raw` task instead and pool in app code.

## Cross-encoder reranking (`Rerank` task)

Catalog config:

```json
{
  "alias": "rerank/cross",
  "task": "Rerank",
  "provider_id": "local/onnx",
  "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2"
}
```

Developer flow:

1. Resolve the handle: `let reranker = runtime.reranker("rerank/cross").await?;`
2. Call `reranker.rerank("query", &["doc a", "doc b"]).await?` to get scored documents back.

## Image classification

Developer flow:

1. Resize/normalize image in app code.
2. Build the expected image tensor layout.
3. Call `run()`.
4. Decode logits.

The runtime is a good fit; preprocessing remains model-specific.

## Object detection or complex multimodal graphs

These are possible, but more manual.

Developer flow:

1. Inspect signatures carefully.
2. Build custom tensors.
3. Run inference.
4. Decode boxes, masks, spans, or custom outputs outside Uni-Xervo.

That is the point where `local/onnx` intentionally stays low-level.

## Which ONNX cases feel easiest today

Best current DX:

- tabular models
- regression
- sequence classification
- NER
- custom numeric graphs

More manual but still supported:

- image models
- object detection
- audio models
- unusual multi-input graphs
