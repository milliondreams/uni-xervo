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

## Embedding exports

Some HF ONNX exports expose hidden states rather than finished pooled embeddings.

Developer flow:

1. Tokenize inputs.
2. Run the graph.
3. Pool or normalize outputs if the export requires it.

If you want a higher-level local embedding provider, prefer `local/candle` or `local/fastembed`.

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
