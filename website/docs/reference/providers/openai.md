# remote/openai

## Uni-Xervo support

- Provider ID: `remote/openai`
- Feature flag: `provider-openai`
- Capabilities: `embed`, `generate`

## Authentication

Default key env var:

- `OPENAI_API_KEY`

## Uni-Xervo provider options

- `api_key_env` (string, optional env var override)
- `embedding_dimensions` (integer, override embedding dimensions)
- `base_url` (string, optional) — override the API base URL. Defaults to `https://api.openai.com/v1`. Set this to target an OpenAI-compatible server (OpenRouter, vLLM, LM Studio, Ollama, internal proxies). The value should include the version path segment (e.g. `/v1`).

Authoritative Uni-Xervo option schema:

- <https://github.com/rustic-ai/uni-xervo/blob/main/crates/uni-xervo/schemas/provider-options/openai.schema.json>

## Authoritative model and config docs

- Model catalog: <https://platform.openai.com/docs/models>
- Chat generation request params: <https://platform.openai.com/docs/api-reference/chat/create>
- Embeddings request params: <https://platform.openai.com/docs/api-reference/embeddings/create>

## Uni-Xervo generation options exposed

- `max_tokens`
- `temperature`
- `top_p`

## Example catalog entry

```json
{
  "alias": "generate/chat",
  "task": "generate",
  "provider_id": "remote/openai",
  "model_id": "gpt-4o-mini",
  "options": {
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

### Targeting an OpenAI-compatible server

Any server that speaks the OpenAI wire protocol (OpenRouter, vLLM, LM Studio, Ollama's `/v1` endpoint, internal proxies) can be reached by setting `base_url`:

```json
{
  "alias": "generate/local",
  "task": "generate",
  "provider_id": "remote/openai",
  "model_id": "llama-3.1-8b-instruct",
  "options": {
    "api_key_env": "LOCAL_LLM_KEY",
    "base_url": "http://localhost:8000/v1"
  }
}
```
