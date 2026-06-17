# Guides

Practical guides for operating Uni-Xervo in production environments.

- [Provider Selection](provider-selection.md): choose providers by capability, latency, and control.
- [ONNX Runtime](onnx.md): use `local/onnx` for raw tensor execution, HF snapshots, and ONNX Runtime session management.
- [Config Validation](config-validation.md): enforce schema correctness in CI and startup.
- [Multimodal Generation](multimodal-generation.md): vision, diffusion, and speech pipelines with `local/mistralrs`.
- [Structured NLP](nlp.md): POS / NER / dependency / SRL / dialog-act analysis via `NlpModel` and the kniv-deberta cascade.
- [Multimodal Trait Surface](multimodal-traits.md): the seven new traits added in 0.13.0 (image / audio / multimodal embed, NLP, document extract, transcription, OCR).
- [Tiered PDF Extraction](pdf-extraction.md): the `uni-xervo-pdf` companion crate — escalate per page across native text → OCR → doc-VLM, with provenance and cross-tier verification.

For full ONNX developer documentation, use the dedicated [ONNX](../onnx/index.md) section.
