// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Embedding-model presets for `local/onnx`.
//!
//! Maps short alias strings (and HF repo IDs) to concrete ONNX embedding
//! configuration: file layout within the repo, pooling strategy,
//! dimensionality, and tokenization shape. Catalog entries that match a
//! preset alias inherit these defaults; per-spec `options` may override.
//!
//! Presets are looked up case-sensitively via [`lookup`] against either the
//! `model_id` from a [`crate::api::ModelAliasSpec`] or any alias string in
//! the entry's `aliases` table.
//!
//! The current table mirrors the 25 text-embedding models exposed by the
//! (now-retired) `provider-fastembed` wrapper, so existing catalogs that
//! used FastEmbed alias strings (e.g. `"BGESmallENV15"`,
//! `"all-MiniLM-L6-v2"`) keep resolving without changes.
//!
//! ## How dimensions, pooling, and `token_type_ids` were chosen
//!
//! - **Dimensions** match `LocalFastEmbedProvider::dimension_for` exactly
//!   (most are 384 / 768 / 1024; BGESmallZHV15 is 512 — the architectural
//!   exception in the BGE-small family).
//! - **Pooling**: BERT-family models (BGE, MiniLM, MPNet, Mxbai, ModernBERT)
//!   use CLS; sentence-transformer-style models trained for mean-pool
//!   (E5, Paraphrase, Nomic) use Mean.
//! - **`token_type_ids`**: BERT-family models accept it; Nomic Embed (built
//!   on Nomic-BERT, no type ids) does not.
//! - **`max_seq_len`** is 512 across the board to match `fastembed-rs`'s
//!   `DEFAULT_MAX_LENGTH`. Long-context users can override via
//!   `options.max_seq_len`.
//! - **Normalization** is `true` for every preset (matches FastEmbed).

/// How a preset's hidden states are reduced to a single vector per input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PoolingKind {
    /// First-token (`[CLS]`) pool: take `hidden[:, 0, :]`.
    Cls,
    /// Mean pool over unmasked positions.
    Mean,
    /// Per-dimension max over unmasked positions.
    #[allow(dead_code)] // No preset uses Max today; included for completeness.
    Max,
}

/// A canonical embedding-model configuration that matches one or more aliases.
#[derive(Debug, Clone)]
pub(super) struct EmbeddingPreset {
    /// Strings that resolve to this preset (HF repo IDs and short aliases).
    pub aliases: &'static [&'static str],
    /// HuggingFace repo to download from.
    pub hf_repo: &'static str,
    /// Path to the `.onnx` file within the repo.
    pub onnx_path: &'static str,
    /// Path to `tokenizer.json` within the repo.
    pub tokenizer_path: &'static str,
    /// Extra files to download alongside the `.onnx` file (e.g. external-data
    /// `.onnx_data` sidecars for models too large to fit in a single protobuf,
    /// or auxiliary constant tensors). Files land in the same cache directory
    /// so ONNX Runtime can resolve them via relative paths embedded in the
    /// graph. Use an empty slice if none are required.
    pub additional_files: &'static [&'static str],
    /// Output embedding dimensionality.
    pub dimensions: u32,
    /// Pooling strategy applied to the model's last hidden state.
    pub pooling: PoolingKind,
    /// Whether outputs are L2-normalized.
    pub normalize: bool,
    /// Truncation cap for tokenized inputs.
    pub max_seq_len: usize,
    /// Whether the model expects a `token_type_ids` input tensor.
    pub token_type_ids: bool,
}

const PRESETS: &[EmbeddingPreset] = &[
    // ---- All-MiniLM family (sentence-transformers; mean-pool) -------------
    EmbeddingPreset {
        aliases: &["AllMiniLML6V2", "all-MiniLM-L6-v2"],
        hf_repo: "Qdrant/all-MiniLM-L6-v2-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["AllMiniLML6V2Q"],
        hf_repo: "Xenova/all-MiniLM-L6-v2",
        onnx_path: "onnx/model_quantized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["AllMiniLML12V2"],
        hf_repo: "Xenova/all-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["AllMiniLML12V2Q"],
        hf_repo: "Xenova/all-MiniLM-L12-v2",
        onnx_path: "onnx/model_quantized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    // ---- All-MPNet (mean-pool; MPNet has no token_type_ids) ---------------
    EmbeddingPreset {
        aliases: &["AllMpnetBaseV2", "all-mpnet-base-v2"],
        hf_repo: "Xenova/all-mpnet-base-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: false,
    },
    // ---- BGE English ------------------------------------------------------
    EmbeddingPreset {
        aliases: &[
            "BGESmallENV15",
            "bge-small-en-v1.5",
            "BAAI/bge-small-en-v1.5",
        ],
        hf_repo: "Xenova/bge-small-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["BGESmallENV15Q"],
        hf_repo: "Qdrant/bge-small-en-v1.5-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["BGEBaseENV15", "bge-base-en-v1.5", "BAAI/bge-base-en-v1.5"],
        hf_repo: "Xenova/bge-base-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["BGEBaseENV15Q"],
        hf_repo: "Qdrant/bge-base-en-v1.5-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &[
            "BGELargeENV15",
            "bge-large-en-v1.5",
            "BAAI/bge-large-en-v1.5",
        ],
        hf_repo: "Xenova/bge-large-en-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["BGELargeENV15Q"],
        hf_repo: "Qdrant/bge-large-en-v1.5-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    // ---- BGE Chinese ------------------------------------------------------
    EmbeddingPreset {
        aliases: &["BGESmallZHV15", "BAAI/bge-small-zh-v1.5"],
        hf_repo: "Xenova/bge-small-zh-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        // bge-small-zh-v1.5 has hidden_size=512 (architectural exception in
        // the BGE-small family — English variant is 384).
        dimensions: 512,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["BGELargeZHV15", "BAAI/bge-large-zh-v1.5"],
        hf_repo: "Xenova/bge-large-zh-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    // ---- BGE M3 (multilingual, multifunc) ---------------------------------
    EmbeddingPreset {
        aliases: &["BGEM3", "BAAI/bge-m3"],
        hf_repo: "BAAI/bge-m3",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        // BGE-M3 ships its weights as ONNX external-data: the `.onnx_data`
        // sidecar holds the bulk tensor blob, and `Constant_7_attr__value`
        // is a separate constant tensor referenced by the graph.
        additional_files: &["onnx/model.onnx_data", "onnx/Constant_7_attr__value"],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        // BAAI's M3 ONNX export omits token_type_ids.
        token_type_ids: false,
    },
    // ---- Nomic (mean-pool; ONNX export DOES consume token_type_ids) ------
    EmbeddingPreset {
        aliases: &["NomicEmbedTextV1", "nomic-ai/nomic-embed-text-v1"],
        hf_repo: "nomic-ai/nomic-embed-text-v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &[
            "NomicEmbedTextV15",
            "nomic-embed-text-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
        ],
        hf_repo: "nomic-ai/nomic-embed-text-v1.5",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["NomicEmbedTextV15Q"],
        hf_repo: "nomic-ai/nomic-embed-text-v1.5",
        onnx_path: "onnx/model_quantized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    // ---- Paraphrase Multilingual -----------------------------------------
    EmbeddingPreset {
        aliases: &["ParaphraseMLMiniLML12V2"],
        hf_repo: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["ParaphraseMLMiniLML12V2Q"],
        hf_repo: "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
        onnx_path: "model_optimized.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &["ParaphraseMLMpnetBaseV2"],
        hf_repo: "Xenova/paraphrase-multilingual-mpnet-base-v2",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        // MPNet architecture: no token_type_ids.
        token_type_ids: false,
    },
    // ---- Multilingual E5 --------------------------------------------------
    EmbeddingPreset {
        aliases: &[
            "MultilingualE5Small",
            "multilingual-e5-small",
            "intfloat/multilingual-e5-small",
        ],
        hf_repo: "intfloat/multilingual-e5-small",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 384,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    EmbeddingPreset {
        aliases: &[
            "MultilingualE5Base",
            "multilingual-e5-base",
            "intfloat/multilingual-e5-base",
        ],
        hf_repo: "intfloat/multilingual-e5-base",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 768,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        // intfloat's E5-base ONNX export omits token_type_ids (unlike the
        // Small variant, which accepts it).
        token_type_ids: false,
    },
    EmbeddingPreset {
        aliases: &[
            "MultilingualE5Large",
            "multilingual-e5-large",
            "intfloat/multilingual-e5-large",
        ],
        hf_repo: "Qdrant/multilingual-e5-large-onnx",
        onnx_path: "model.onnx",
        tokenizer_path: "tokenizer.json",
        // E5-large is exported with external data — model.onnx is the graph,
        // model.onnx_data is the weight blob.
        additional_files: &["model.onnx_data"],
        dimensions: 1024,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        // Qdrant's E5-large ONNX export omits token_type_ids (matches
        // intfloat/multilingual-e5-base behavior; only Small accepts them).
        token_type_ids: false,
    },
    // ---- Mxbai ------------------------------------------------------------
    EmbeddingPreset {
        aliases: &[
            "MxbaiEmbedLargeV1",
            "mxbai-embed-large-v1",
            "mixedbread-ai/mxbai-embed-large-v1",
        ],
        hf_repo: "mixedbread-ai/mxbai-embed-large-v1",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Cls,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: true,
    },
    // ---- ModernBERT (mean-pool; no token_type_ids — ModernBERT replaces
    //                  segment ids with rotary position embeddings) --------
    EmbeddingPreset {
        aliases: &["ModernBertEmbedLarge", "lightonai/modernbert-embed-large"],
        hf_repo: "lightonai/modernbert-embed-large",
        onnx_path: "onnx/model.onnx",
        tokenizer_path: "tokenizer.json",
        additional_files: &[],
        dimensions: 1024,
        pooling: PoolingKind::Mean,
        normalize: true,
        max_seq_len: 512,
        token_type_ids: false,
    },
];

/// Look up a preset by alias or HF repo ID. Returns `None` for unknown keys
/// so callers can fall back to pass-through (user-supplied options).
pub(super) fn lookup(key: &str) -> Option<&'static EmbeddingPreset> {
    PRESETS.iter().find(|p| p.aliases.contains(&key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Every alias must be unique across the table — if two presets claim
    /// the same alias, lookup is order-dependent and surprising.
    #[test]
    fn aliases_are_unique() {
        let mut seen: HashSet<&'static str> = HashSet::new();
        for preset in PRESETS {
            for alias in preset.aliases {
                assert!(
                    seen.insert(alias),
                    "Duplicate alias '{}' across presets",
                    alias
                );
            }
        }
    }

    /// Spot-check: the three popular FastEmbed aliases should resolve.
    #[test]
    fn popular_aliases_resolve() {
        for alias in &[
            "BGESmallENV15",
            "bge-small-en-v1.5",
            "AllMiniLML6V2",
            "all-MiniLM-L6-v2",
            "MultilingualE5Small",
            "NomicEmbedTextV15",
        ] {
            assert!(
                lookup(alias).is_some(),
                "Expected preset for alias '{}'",
                alias
            );
        }
    }
}
