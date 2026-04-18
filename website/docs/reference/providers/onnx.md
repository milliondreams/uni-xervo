# local/onnx

## Uni-Xervo support

- Provider ID: `local/onnx`
- Feature flag: `provider-onnx`
- Capabilities: `raw`

## Uni-Xervo provider options

- `artifact` (string)
- `max_batch_size` (integer)
- `execution_providers` (`cpu`, `cuda`, `coreml`, `directml`)
- `graph_optimization_level` (`disable`, `basic`, `extended`, `all`)
- `inter_op_num_threads` (integer)
- `intra_op_num_threads` (integer)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/onnx.schema.json>

## Model IDs

`model_id` can be:

- a local path to a `.onnx` file, or
- a Hugging Face repo ID

HF-backed aliases download a full repo snapshot into the Uni-Xervo cache before ORT session creation.

If the repo contains multiple `.onnx` files, set `options.artifact`.

## Runtime contract

`local/onnx` exposes the raw tensor API:

- `runtime.onnx_runner(alias)`
- `TensorBatch`
- `TensorValue`
- `TensorSpec`
- `runner.run(&batch)`
- `runner.run_batch(&batches)`

Uni-Xervo validates input names, dtypes, and shapes, but does not assign semantic meaning to tensors.

## Example catalog entry

```json
{
  "alias": "raw/classifier",
  "task": "raw",
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
