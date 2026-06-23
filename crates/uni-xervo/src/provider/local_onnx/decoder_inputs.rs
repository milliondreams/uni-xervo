// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared session-input discovery for decoder-LM ONNX exports
//! (Qwen3-Embedding, Qwen3-Reranker, NV-Embed, etc.).
//!
//! HuggingFace `optimum`/`transformers.js` exports of decoder-only
//! causal LMs declare a tail of inputs the user rarely cares about:
//! `position_ids` (caller-built `0..seq` per row) and one
//! `past_key_values.{layer}.{key|value}` per transformer layer
//! (empty zero tensors for a one-shot prefill pass).
//!
//! Both the generative-reranker code path and the last-token embedding
//! path need to feed all of these or ORT errors with
//! `Missing Input: past_key_values.0.value`. This module centralises:
//!
//! 1. classifying every `session.inputs()` entry by role,
//! 2. resolving each dynamic dim to runtime batch / current-seq /
//!    past-seq (always 0 in a one-shot pass),
//! 3. allocating zero-filled placeholder tensors of the right shape
//!    and dtype for `past_key_values.*` inputs.
//!
//! Inputs the *caller* supplies (`input_ids`, `attention_mask`,
//! `position_ids`, `token_type_ids`) are classified but not built
//! here — each caller has its own tokenization path and feeds those
//! tensors with real data.

use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::value::{DynTensor, Tensor, TensorElementType, ValueType};

use crate::error::{Result, RuntimeError};

/// Classified session input. `role` tells the caller how to populate
/// the tensor on each forward pass; `dims` is the resolved per-axis
/// behaviour (constant or dynamic).
#[derive(Debug, Clone)]
pub(super) struct InputSchema {
    pub(super) name: String,
    pub(super) dtype: TensorElementType,
    pub(super) dims: Vec<DimRole>,
    pub(super) role: InputRole,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DimRole {
    /// Concrete dim baked into the ONNX graph (e.g. `num_kv_heads=8`).
    Fixed(usize),
    /// Dynamic dim that takes the runtime batch size.
    Batch,
    /// Dynamic dim that takes the runtime *current* sequence length
    /// (the input being prefilled).
    Seq,
    /// Dynamic dim for past sequence length — always 0 in a one-shot
    /// prefill pass with an empty KV cache.
    PastSeq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputRole {
    InputIds,
    AttentionMask,
    PositionIds,
    TokenTypeIds,
    /// `past_key_values.{N}.{key|value}` — caller treats as opaque,
    /// passes to [`build_empty_past_kv`].
    PastKeyValue,
}

/// Classify an input name. Returns `None` if the name doesn't match
/// any of the known decoder-LM input patterns; callers should error
/// rather than silently feeding garbage to the model.
pub(super) fn classify_input(name: &str) -> Option<InputRole> {
    match name {
        "input_ids" => Some(InputRole::InputIds),
        // `attention_mask` is the HF-standard name; `input_mask` is used by
        // some BERT exports (e.g. SPLADE++ `onnx/model.onnx`).
        "attention_mask" | "input_mask" => Some(InputRole::AttentionMask),
        "position_ids" => Some(InputRole::PositionIds),
        // `token_type_ids` is standard; `segment_ids` is the TF/BERT-legacy
        // name used by some exports (e.g. SPLADE++).
        "token_type_ids" | "segment_ids" => Some(InputRole::TokenTypeIds),
        n if n.starts_with("past_key_values") => Some(InputRole::PastKeyValue),
        _ => None,
    }
}

/// Walk `session.inputs()` and classify every entry. Errors if any
/// input has an unrecognised name (catches schema drift early rather
/// than at first run).
pub(super) fn build_input_schema(session: &Session, alias: &str) -> Result<Vec<InputSchema>> {
    session
        .inputs()
        .iter()
        .map(|input| {
            let name = input.name().to_string();
            let role = classify_input(&name).ok_or_else(|| RuntimeError::OnnxLoadFailure {
                alias: alias.to_string(),
                path: PathBuf::new(),
                cause: format!(
                    "Decoder-LM input '{name}' is not recognised. Expected one of: \
                     input_ids, attention_mask, position_ids, token_type_ids, or \
                     past_key_values.*"
                ),
            })?;
            let (dtype, dims) = resolve_dims(input.dtype(), role, &name, alias)?;
            Ok(InputSchema {
                name,
                dtype,
                dims,
                role,
            })
        })
        .collect()
}

fn resolve_dims(
    value_type: &ValueType,
    role: InputRole,
    input_name: &str,
    alias: &str,
) -> Result<(TensorElementType, Vec<DimRole>)> {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
        ..
    } = value_type
    else {
        return Err(RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: PathBuf::new(),
            cause: format!("Input '{input_name}' is not a tensor"),
        });
    };

    let dims: Vec<DimRole> = shape
        .iter()
        .enumerate()
        .map(|(idx, dim)| {
            if *dim >= 0 {
                DimRole::Fixed(*dim as usize)
            } else {
                let symbol = dimension_symbols.get(idx).cloned().unwrap_or_default();
                classify_dynamic_dim(role, idx, &symbol)
            }
        })
        .collect();

    Ok((*ty, dims))
}

/// Map a dynamic dimension to its runtime role. Uses dim symbols when
/// the export labels them (`batch_size`, `sequence_length`,
/// `past_sequence_length`) and falls back to positional heuristics
/// matching transformers/optimum decoder-LM conventions.
fn classify_dynamic_dim(role: InputRole, idx: usize, symbol: &str) -> DimRole {
    let s = symbol.to_lowercase();
    if s.contains("past") || s.contains("kv") {
        return DimRole::PastSeq;
    }
    if s.contains("batch") {
        return DimRole::Batch;
    }
    if s.contains("seq") {
        return match role {
            InputRole::PastKeyValue => DimRole::PastSeq,
            _ => DimRole::Seq,
        };
    }

    // Positional fallback — order matches HF decoder-LM exports:
    //   input_ids/attention_mask/position_ids/token_type_ids: [batch, seq]
    //   past_key_values.*: [batch, num_kv_heads, past_seq, head_dim]
    match (role, idx) {
        (
            InputRole::InputIds
            | InputRole::AttentionMask
            | InputRole::PositionIds
            | InputRole::TokenTypeIds,
            0,
        ) => DimRole::Batch,
        (
            InputRole::InputIds
            | InputRole::AttentionMask
            | InputRole::PositionIds
            | InputRole::TokenTypeIds,
            1,
        ) => DimRole::Seq,
        (InputRole::PastKeyValue, 0) => DimRole::Batch,
        (InputRole::PastKeyValue, 2) => DimRole::PastSeq,
        // Any other un-named dynamic dim — treat as PastSeq (collapses
        // to 0). If this picks the wrong axis, the ORT shape-mismatch
        // error message will name the input.
        _ => DimRole::PastSeq,
    }
}

/// Build an empty (zero-filled) tensor for a `past_key_values.*`
/// input. Resolves [`DimRole::Batch`] to the runtime batch size and
/// [`DimRole::PastSeq`]/[`DimRole::Seq`] to 0; fixed dims pass through.
pub(super) fn build_empty_past_kv(
    schema: &InputSchema,
    batch: usize,
    alias: &str,
) -> Result<DynTensor> {
    let resolved: Vec<usize> = schema
        .dims
        .iter()
        .map(|d| match d {
            DimRole::Fixed(n) => *n,
            DimRole::Batch => batch,
            DimRole::PastSeq | DimRole::Seq => 0,
        })
        .collect();

    match schema.dtype {
        TensorElementType::Float32 => {
            let arr = ArrayD::<f32>::zeros(IxDyn(&resolved));
            Ok(Tensor::from_array(arr)
                .map_err(|e| RuntimeError::OnnxInvocationFailure {
                    alias: alias.to_string(),
                    cause: format!("'{}' empty f32 KV: {e}", schema.name),
                })?
                .upcast())
        }
        // f16 / bf16 KV caches would require the `ort/half` feature
        // flag (and the `half` crate as a direct dep). Q4-quantized
        // exports for the Qwen3 family keep KV in f32 in practice;
        // fail loud rather than silently disagree with the runtime.
        other => Err(RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: PathBuf::new(),
            cause: format!(
                "past_key_values input '{}' has dtype {other:?}; the decoder-LM \
                 path currently handles only f32 KV caches. Re-export with \
                 f32 KV or open an issue if this variant should be supported.",
                schema.name
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_standard_input_names() {
        assert_eq!(classify_input("input_ids"), Some(InputRole::InputIds));
        assert_eq!(
            classify_input("attention_mask"),
            Some(InputRole::AttentionMask)
        );
        assert_eq!(classify_input("position_ids"), Some(InputRole::PositionIds));
        assert_eq!(
            classify_input("token_type_ids"),
            Some(InputRole::TokenTypeIds)
        );
        assert_eq!(
            classify_input("past_key_values.0.key"),
            Some(InputRole::PastKeyValue)
        );
    }

    #[test]
    fn classifies_nonstandard_aliases() {
        // SPLADE++ `onnx/model.onnx` declares these legacy names.
        assert_eq!(classify_input("input_mask"), Some(InputRole::AttentionMask));
        assert_eq!(classify_input("segment_ids"), Some(InputRole::TokenTypeIds));
    }

    #[test]
    fn rejects_unknown_input_name() {
        assert_eq!(classify_input("mystery_input"), None);
    }
}
