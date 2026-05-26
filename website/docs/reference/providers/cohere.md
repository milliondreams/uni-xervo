# remote/cohere

## Uni-Xervo support

- Provider ID: `remote/cohere`
- Feature flag: `provider-cohere`
- Capabilities: `embed`, `rerank`, `generate`, `embed_multimodal`
  (Cohere Embed v4, new in 0.13.0)

## Authentication

Default key env var:

- `CO_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string)
- `input_type` (string, embedding requests)
- `embedding_dimensions` (integer, override embedding dimensions)

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/schemas/provider-options/cohere.schema.json>

## Authoritative model and config docs

- Model catalog: <https://docs.cohere.com/v2/docs/models>
- Chat/generation request config: <https://docs.cohere.com/v2/reference/chat>
- Embeddings request config: <https://docs.cohere.com/v2/reference/embed>
- Rerank request config: <https://docs.cohere.com/v2/reference/rerank>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "embed/cohere",
  "task": "embed",
  "provider_id": "remote/cohere",
  "model_id": "embed-english-v3.0",
  "options": {
    "api_key_env": "CO_API_KEY",
    "input_type": "search_document"
  }
}
```

### Multimodal embed (Cohere Embed v4)

Cohere Embed v4 takes a mixed `inputs: [{ "content": [...] }]` payload
of text and image blocks. Uni-Xervo converts `MultimodalBlock::Text` and
`MultimodalBlock::Image` (URL or raw bytes — bytes are base64-encoded
into a `data:image/...` URL upstream). `MultimodalBlock::Audio` is
rejected because Cohere Embed v4 does not accept audio inputs.

```json
{
  "alias": "embed/cohere-mm",
  "task": "embed_multimodal",
  "provider_id": "remote/cohere",
  "model_id": "embed-v4.0",
  "options": {
    "api_key_env": "CO_API_KEY",
    "input_type": "search_document"
  }
}
```

`supported_modalities()` reports `[Text, Image]`. `EmbedResult::usage`
is populated from `meta.billed_units.input_tokens` when Cohere returns
it.
