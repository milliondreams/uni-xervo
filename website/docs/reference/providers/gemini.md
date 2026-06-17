# remote/gemini

## Uni-Xervo support

- Provider ID: `remote/gemini`
- Feature flag: `provider-gemini`
- Capabilities: `embed`, `generate`, `embed_multimodal` (Gemini
  Embedding 2 / `batchEmbedContents`, new in 0.13.0)

## Authentication

Default key env var:

- `GEMINI_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string, optional env var override)
- `api_version` (string, API version path segment, default `v1beta`)
- `embedding_dimensions` (integer, override embedding dimensions)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/crates/uni-xervo/schemas/provider-options/gemini.schema.json>

## Authoritative model and config docs

- Model catalog: <https://ai.google.dev/gemini-api/docs/models>
- Text generation docs/config: <https://ai.google.dev/gemini-api/docs/text-generation>
- Embeddings docs/config: <https://ai.google.dev/gemini-api/docs/embeddings>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/gemini",
  "task": "generate",
  "provider_id": "remote/gemini",
  "model_id": "gemini-2.0-flash",
  "options": {
    "api_key_env": "GEMINI_API_KEY"
  }
}
```

### Multimodal embed (Gemini Embedding 2)

Gemini Embedding 2 accepts `content.parts: [text | inline_data | file_data]`
via `batchEmbedContents`. Uni-Xervo converts all four `MultimodalBlock`
variants:

- `Text` → `{ "text": "..." }`
- `Image(Url)` → `{ "file_data": { "file_uri": "..." } }`
- `Image(Bytes { data, media_type })` → `{ "inline_data": { "mime_type": ..., "data": "<base64>" } }`
- `Audio(Bytes { data, media_type })` → `{ "inline_data": { ... } }`
- `Audio(Pcm { sample_rate, samples, .. })` → encoded to 16-bit mono WAV
  in memory, then sent as `inline_data` with `mime_type: "audio/wav"`.

```json
{
  "alias": "embed/gemini-mm",
  "task": "embed_multimodal",
  "provider_id": "remote/gemini",
  "model_id": "gemini-embedding-001",
  "options": {
    "api_key_env": "GEMINI_API_KEY"
  }
}
```

`supported_modalities()` reports `[Text, Image, Audio, Video]`. Gemini's
embed endpoint does not return usage info, so `EmbedResult::usage` is
always `None`.
