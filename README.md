# uni-xervo

A Rust workspace for local and remote ML model inference, and document
extraction built on top of it.

## Crates

| Crate | Path | Description |
| --- | --- | --- |
| [`uni-xervo`](crates/uni-xervo) | `crates/uni-xervo` | Unified runtime for local and remote embedding, reranking, generation, OCR, and document-extraction models across pluggable providers (Candle, ONNX Runtime, mistral.rs, OpenAI, Gemini, …). |
| [`uni-xervo-pdf`](crates/uni-xervo-pdf) | `crates/uni-xervo-pdf` | Optional companion: tiered PDF document extraction that escalates per page across native text → OCR → doc-VLM, with provenance and cross-tier verification. |

## Documentation

- Guides and reference: <https://rustic-ai.github.io/uni-xervo> (sources under [`website/`](website)).
- API reference (rustdoc): <https://rustic-ai.github.io/uni-xervo/api/uni_xervo/>.
- Changelog: [`CHANGELOG.md`](CHANGELOG.md).

## Development

```sh
./scripts/test.sh      # fmt check + cargo check/test across the workspace
```

Licensed under Apache-2.0 ([`LICENSE`](LICENSE)).
