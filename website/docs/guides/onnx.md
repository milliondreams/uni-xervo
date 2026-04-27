# ONNX Runtime Guide

`local/onnx` is Uni-Xervo's raw tensor runtime for ONNX models.

Use it when you want Uni-Xervo to handle:

- alias-based model resolution,
- Hugging Face snapshot download and caching,
- ONNX Runtime session loading,
- tensor signature introspection,
- batch validation and splitting,
- timeout/retry/warmup wrappers.

Your application still owns preprocessing and postprocessing.

## What `local/onnx` is for

`local/onnx` is best for:

- custom numeric models,
- tabular regression/classification,
- HF transformer exports where you already control tokenization,
- NER / classification / embedding pipelines that need raw tensor access,
- any ONNX graph where app code should interpret outputs.

It is not a high-level task wrapper like Python `transformers`.

## Catalog shape

Use:

- `provider_id: "local/onnx"`
- `task: "raw"`

Example:

```json
{
  "alias": "raw/minilm",
  "task": "raw",
  "provider_id": "local/onnx",
  "model_id": "nixiesearch/all-MiniLM-L6-v2-onnx",
  "options": {
    "artifact": "model.onnx"
  }
}
```

## Local path vs Hugging Face repo

`model_id` can be either:

- a local filesystem path to an ONNX file, or
- a Hugging Face repo ID

For HF-backed aliases, Uni-Xervo snapshots the full repo revision into cache before loading the selected artifact. That means repo-local sidecars such as tokenizer files, config files, and external data files are available next to the ONNX model on disk.

If the repo contains multiple `.onnx` files, set `options.artifact`.

## Runtime API

Resolve raw ONNX handles with `runtime.raw_tensor_model(alias)`.

```rust
use ndarray::{arr2, ArrayD};
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::{TensorBatch, TensorValue};

let runner = runtime.raw_tensor_model("raw/minilm").await?;

let mut batch = TensorBatch::new();
batch.insert(
    "input_ids".to_string(),
    TensorValue::I64(arr2(&[[101_i64, 7592, 2088, 102]]).into_dyn()),
);
batch.insert(
    "attention_mask".to_string(),
    TensorValue::I64(arr2(&[[1_i64, 1, 1, 1]]).into_dyn()),
);

let outputs = runner.run(&batch).await?;
let logits = outputs.get("last_hidden_state").unwrap();
```

Useful APIs:

- `runtime.raw_tensor_model(alias)`
- `runner.input_signature()`
- `runner.output_signature()`
- `runner.max_batch_size()`
- `runner.run(&batch)`
- `runner.run_batch(&batches)`

## Tensor contract

Uni-Xervo does not interpret model semantics. It only validates names, dtypes, and shapes against the ONNX graph signature.

Core types:

- `TensorBatch`: ordered tensor map keyed by input/output name
- `TensorValue`: raw `ndarray::ArrayD<T>` over supported ONNX tensor dtypes
- `TensorSpec`: name, dtype, and dimension metadata

Supported `TensorValue` variants cover:

- `f32`
- `f64`
- `i32`
- `i64`
- `bool`
- `string`

## Provider options

`local/onnx` accepts these `options` keys:

- `artifact`
- `max_batch_size`
- `execution_providers`
- `graph_optimization_level`
- `inter_op_num_threads`
- `intra_op_num_threads`

Example:

```json
{
  "artifact": "model.onnx",
  "max_batch_size": 32,
  "execution_providers": ["cuda", "cpu"],
  "graph_optimization_level": "extended",
  "intra_op_num_threads": 4
}
```

## Execution providers

`execution_providers` is optional.

Defaults:

- CPU-only builds: `["cpu"]`
- `gpu-cuda` builds: ORT-backed providers prefer `["cuda", "cpu"]`

Supported names:

- `cpu`
- `cuda`
- `coreml`
- `directml`

## Developer workflow

Typical flow for HF transformer exports:

1. Configure `local/onnx` in the catalog.
2. Resolve `runtime.raw_tensor_model(alias)`.
3. Load `tokenizer.json` / `config.json` from the same cached snapshot if needed.
4. Build `TensorBatch` inputs in app code.
5. Run inference.
6. Interpret outputs in app code.

See the live test coverage in `tests/local_onnx_hf_e2e_test.rs` for real classification and NER examples.
