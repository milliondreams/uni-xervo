# local/mistralrs

## Uni-Xervo support

- Provider ID: `local/mistralrs`
- Feature flag: `provider-mistralrs`
- Capabilities: `embed`, `generate`
- Pipeline types: `text` (default), `vision`, `diffusion`, `speech`

## Pipeline types

| Pipeline | Description | Output |
| --- | --- | --- |
| `text` | Standard LLM text generation (default) | `result.text` |
| `vision` | Image + text understanding | `result.text` |
| `diffusion` | Text-to-image generation | `result.images` |
| `speech` | Text-to-audio synthesis | `result.audio` |

## Uni-Xervo provider options

### Common options (all pipelines)

| Option | Type | Description |
| --- | --- | --- |
| `pipeline` | string | Pipeline type: `text`, `vision`, `diffusion`, `speech`. Default: `text` |
| `dtype` | string | Model precision: `auto`, `f16`, `bf16`, `f32`. See [dtype](#dtype) |
| `force_cpu` | boolean | Force CPU inference |

### Auto-device-mapper overrides (text and vision pipelines)

The auto-device-mapper plans layer placement using the *unquantized* dtype
footprint, ignoring future ISQ shrinkage. On small GPUs the default
reservation can leave zero layers on-device; lower these knobs to fit.

| Option | Type | Description |
| --- | --- | --- |
| `max_seq_len` | integer > 0 | Override max sequence length used for KV-cache reservation (default 4096). Text, vision, embedding |
| `max_batch_size` | integer > 0 | Override max batch size used for KV-cache reservation (default 1). Text, vision, embedding |
| `max_image_shape` | `[h, w]` integers > 0 | Override max image shape (default `[1024, 1024]`). Vision only |
| `max_num_images` | integer > 0 | Override max number of images per request (default 1). Vision only |

### Text pipeline options

| Option | Type | Description |
| --- | --- | --- |
| `isq` | string | In-situ quantization type (e.g. `Q4K`, `Q8_0`) |
| `paged_attention` | boolean | Enable paged attention |
| `max_num_seqs` | integer > 0 | Maximum concurrent sequences |
| `chat_template` | string | Custom chat template |
| `tokenizer_json` | string | Path to tokenizer.json |
| `embedding_dimensions` | integer > 0 | Override output dimensions for embeddings (embed task only) |
| `gguf_files` | array of strings | GGUF filenames to load in GGUF mode |
| `uqff_files` | array of strings | UQFF (mistralrs pre-quantized) filenames. See [UQFF](#uqff). Mutually exclusive with `gguf_files` and `isq` |

### Diffusion pipeline options

| Option | Type | Description |
| --- | --- | --- |
| `diffusion_loader_type` | string | Required. One of: `flux`, `flux_offloaded` |

### Speech pipeline options

| Option | Type | Description |
| --- | --- | --- |
| `speech_loader_type` | string | Required. One of: `dia` |

### Pipeline-specific option validity

| Option | text | vision | diffusion | speech |
| --- | --- | --- | --- | --- |
| `pipeline` | Yes | Yes | Yes | Yes |
| `dtype` | Yes | Yes | Yes | Yes |
| `force_cpu` | Yes | Yes | Yes | Yes |
| `isq` | Yes | No | No | No |
| `paged_attention` | Yes | No | No | No |
| `max_num_seqs` | Yes | No | No | No |
| `chat_template` | Yes | No | No | No |
| `tokenizer_json` | Yes | No | No | No |
| `embedding_dimensions` | Yes | No | No | No |
| `gguf_files` | Yes | No | No | No |
| `uqff_files` | Yes | Yes | No | No |
| `diffusion_loader_type` | No | No | Yes | No |
| `speech_loader_type` | No | No | No | Yes |
| `max_seq_len` | Yes | Yes | No | No |
| `max_batch_size` | Yes | Yes | No | No |
| `max_image_shape` | No | Yes | No | No |
| `max_num_images` | No | Yes | No | No |

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/mistralrs.schema.json>

## Dtype

Model precision control. Available on all four pipeline types.

| Value | Description |
| --- | --- |
| `auto` | Automatic selection (BF16 on GPU, F32 on CPU) |
| `f16` | 16-bit floating point |
| `bf16` | Brain floating point 16 |
| `f32` | 32-bit floating point |

**Default resolution logic:**

1. Explicit `dtype` value in catalog options, if set
2. `f32` when running on CPU or without GPU support
3. `auto` otherwise

## UQFF

UQFF is mistralrs's native pre-quantized format. Unlike `isq` — which loads
full-precision weights and quantizes them in memory at load time — UQFF
files store already-quantized tensors and are loaded directly. This
bypasses the bf16 (or f16) load step entirely, which matters when the
unquantized footprint exceeds available VRAM.

When to use UQFF:

- Loading a multi-billion-parameter multimodal model on a small GPU where
  the bf16 load step OOMs (e.g. Gemma 4 E2B on 8 GB cards).
- Faster startup: no in-memory quantization pass.

How to use:

1. Set `model_id` to a UQFF HuggingFace repo, e.g.
   `mistralrs-community/gemma-4-E2B-it-UQFF`.
2. Set `uqff_files` to the *first* shard's filename, e.g. `["q4k-0.uqff"]`.
   Remaining shards are auto-discovered from the same repo by mistralrs.
3. The quantization variant (Q4K, Q5K, AFQ8, …) is selected by *which*
   file you name.

Compatible pipelines: `text`, `vision`, and the embedding task. Not
supported on `diffusion` / `speech` (the underlying mistralrs builders
don't expose `from_uqff` for those).

Conflicts:

- `uqff_files` ✕ `gguf_files` — both load pre-quantized weights;
  mutually exclusive.
- `uqff_files` ✕ `isq` — UQFF files embed their own quantization; setting
  `isq` alongside is rejected to avoid silent ignore.

`force_cpu`, `dtype`, `paged_attention`, `chat_template`, `tokenizer_json`,
`max_num_seqs`, and the auto-device-mapper overrides
(`max_seq_len`, `max_image_shape`, …) all continue to apply.

The canonical source of UQFF repos is the
[`mistralrs-community`](https://huggingface.co/mistralrs-community) HF
namespace.

## Available models

`local/mistralrs` delegates model support to the upstream `mistral.rs` engine.

Authoritative model/support references:

- mistral.rs docs: <https://ericlbuehler.github.io/mistral.rs/>
- mistral.rs repository: <https://github.com/EricLBuehler/mistral.rs>

## Generation API

Uni-Xervo generation API exposes:

- `max_tokens`
- `temperature`
- `top_p`
- `width` (diffusion only)
- `height` (diffusion only)

`GenerationResult` output fields:

- `text` — generated text (text and vision pipelines)
- `usage` — optional token usage stats
- `images` — generated images (diffusion pipeline)
- `audio` — generated audio (speech pipeline)

## Example catalog entries

### Text generation (basic)

```json
{
  "alias": "generate/local",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "options": {
    "isq": "Q4K",
    "paged_attention": true,
    "max_num_seqs": 8
  }
}
```

### Text generation with GGUF

```json
{
  "alias": "generate/gguf",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
  "options": {
    "gguf_files": ["mistral-7b-instruct-v0.2.Q4_K_M.gguf"]
  }
}
```

### Vision generation with UQFF

UQFF loads pre-quantized weights directly, bypassing the bf16 load step
that ISQ uses. This is the practical path for fitting larger multimodal
models on commodity hardware. Only the first shard needs to be named;
remaining shards are auto-discovered.

```json
{
  "alias": "vision/gemma-4-uqff",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "mistralrs-community/gemma-4-E2B-it-UQFF",
  "options": {
    "pipeline": "vision",
    "uqff_files": ["q4k-0.uqff"]
  }
}
```

### Vision on a small GPU (auto-device-mapper overrides)

The auto-device-mapper sizes its KV-cache and image-buffer reservation
from `max_seq_len` and `max_image_shape` using the *unquantized* dtype.
On 8 GB cards the default reservation can leave zero layers on-device;
lowering these knobs frees room for layer placement.

```json
{
  "alias": "vision/gemma-3n-8gb",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "google/gemma-3n-E2B-it",
  "options": {
    "pipeline": "vision",
    "isq": "Q4K",
    "dtype": "bf16",
    "max_seq_len": 1024,
    "max_image_shape": [224, 224],
    "max_num_images": 1
  }
}
```

### Text generation with ISQ + dtype

```json
{
  "alias": "generate/isq",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
  "options": {
    "isq": "Q8_0",
    "dtype": "bf16"
  }
}
```

### Vision

```json
{
  "alias": "vision/qwen",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "Qwen/Qwen2-VL-2B-Instruct",
  "options": {
    "pipeline": "vision",
    "dtype": "bf16"
  }
}
```

### Diffusion (image generation)

```json
{
  "alias": "image/flux",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "options": {
    "pipeline": "diffusion",
    "diffusion_loader_type": "flux"
  }
}
```

### Speech synthesis

```json
{
  "alias": "tts/dia",
  "task": "generate",
  "provider_id": "local/mistralrs",
  "model_id": "nari-labs/Dia-1.6B",
  "options": {
    "pipeline": "speech",
    "speech_loader_type": "dia"
  }
}
```
