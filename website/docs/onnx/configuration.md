# ONNX Configuration

## Required catalog fields

Every ONNX alias needs:

- `alias`
- `task: "raw"`
- `provider_id: "local/onnx"`
- `model_id`

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

## `model_id`

Supported forms:

- absolute local path
- relative local path
- Hugging Face repo ID

HF repos are snapshotted into cache before ONNX Runtime loads the artifact.

## Provider options

`local/onnx` supports:

- `artifact`
- `max_batch_size`
- `execution_providers`
- `graph_optimization_level`
- `inter_op_num_threads`
- `intra_op_num_threads`

## `artifact`

Use `artifact` when the HF repo contains more than one `.onnx` file.

```json
{
  "artifact": "model.onnx"
}
```

If a repo has exactly one `.onnx` file, `artifact` can be omitted.

## `execution_providers`

`execution_providers` is an ordered list.

Examples:

```json
{
  "execution_providers": ["cpu"]
}
```

```json
{
  "execution_providers": ["cuda", "cpu"]
}
```

Supported names:

- `cpu`
- `cuda`
- `coreml`
- `directml`

Defaults:

- CPU-only builds: `["cpu"]`
- `gpu-cuda` builds: ORT-backed providers prefer `["cuda", "cpu"]`

## Batch tuning

`max_batch_size` controls the ceiling for dynamic-batch models.

```json
{
  "max_batch_size": 32
}
```

For non-batch models, Uni-Xervo falls back to sequential execution in `run_batch()`.

## ORT tuning

Example:

```json
{
  "graph_optimization_level": "extended",
  "inter_op_num_threads": 2,
  "intra_op_num_threads": 4
}
```

Accepted optimization levels:

- `disable`
- `basic`
- `extended`
- `all`

## Validation behavior

Uni-Xervo validates ONNX options at runtime build/register time.

It rejects:

- unknown option keys,
- wrong value types,
- unsupported execution provider names,
- invalid batch/thread counts.

Schema files:

- `schemas/model-catalog.schema.json`
- `schemas/provider-options/onnx.schema.json`
