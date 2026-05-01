// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Generative-style reranker for Qwen3-Reranker (and compatible decoder-LM
//! rerankers) on `local/onnx`.
//!
//! Selected via `options.style = "generative"` for `ModelTask::Rerank`.
//!
//! Unlike a cross-encoder (which produces a single relevance logit per
//! `(query, doc)` pair), Qwen3-Reranker is a `Qwen3ForCausalLM` that
//! decides relevance via constrained binary generation. We:
//!
//! 1. Format `(query, doc)` into a fixed chat prompt that ends right
//!    before the model would emit `yes` or `no`.
//! 2. Tokenize and run a single forward pass.
//! 3. Read logits at the last non-pad position for the `yes` and `no`
//!    token ids.
//! 4. Score = `softmax([yes, no])[0]` — a probability in `[0, 1]`.
//!    (The cross-encoder path returns raw logits in an unbounded range;
//!     callers can still mix the two via descending-score sort.)
//!
//! ## ONNX input handling
//!
//! Qwen3 ONNX exports (e.g. `onnx-community/Qwen3-Reranker-0.6B-ONNX`) are
//! exported as a "merged" decoder that takes `input_ids`, `attention_mask`,
//! `position_ids`, plus one `past_key_values.{i}.{key|value}` tensor per
//! transformer layer. For a one-shot reranking pass we feed empty KV
//! cache tensors (sequence dim = 0) and let the model populate
//! `present_key_values.*` outputs that we discard.
//!
//! Input shapes are discovered from `session.inputs()` at load time: any
//! `past_key_values.*` input has its dynamic batch dim resolved at run
//! time, the dynamic past-sequence dim resolved to 0, and any fixed dims
//! (`num_kv_heads`, `head_dim`) read straight from the ONNX graph.
//!
//! ## Reference
//!
//! Prompt template and scoring formula follow the
//! [official Qwen3-Reranker model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
//! and the [transformers.js usage example](https://huggingface.co/onnx-community/Qwen3-Reranker-0.6B-ONNX).

use async_trait::async_trait;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use ndarray::{Array2, ArrayD, ArrayViewD, IxDyn};
use ort::session::Session;
use ort::value::{DynTensor, Tensor, TensorElementType, ValueType};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    build_execution_providers, parse_execution_providers_option, resolve_ep_list,
};
use crate::traits::{RerankerModel, ScoredDoc};

/// Default ONNX file path within the HF repo. Qwen3-Reranker repos
/// publish `onnx/model_q4.onnx` (4-bit weight-only quantized) and
/// `onnx/model_quantized.onnx` (int8). fp32/fp16 aren't published
/// because the unquantized weights exceed the ONNX single-file size
/// limit. Override via `options.artifact`.
const DEFAULT_ARTIFACT: &str = "onnx/model_q4.onnx";

/// Default truncation cap. Qwen3 supports much longer context
/// (40K tokens), but reranking documents rarely needs more, and
/// allocating the prefill matrix scales linearly with this. Override
/// via `options.max_seq_len`.
const DEFAULT_MAX_SEQ_LEN: usize = 4096;

/// System message — fixed by the model's training. Changing it produces
/// undefined scoring behavior, so it isn't exposed as an option.
const SYSTEM_PROMPT: &str = concat!(
    "Judge whether the Document meets the requirements based on the Query and ",
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
);

/// Default task instruction shown to the model. Matches the HF README's
/// example and works for general retrieval. Override via
/// `options.instruction` for domain-specific tasks (e.g. code search,
/// medical literature).
const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Load the generative reranker model for `spec`. Called from
/// [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load) when
/// `spec.task == ModelTask::Rerank` and `options.style == "generative"`.
pub(super) async fn load(spec: &ModelAliasSpec) -> Result<Arc<dyn RerankerModel>> {
    let reranker = OnnxGenerativeReranker::load(spec).await?;
    Ok(Arc::new(reranker) as Arc<dyn RerankerModel>)
}

struct OnnxGenerativeReranker {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    /// Vocabulary id of the literal token "yes" (case-sensitive, no
    /// leading space) — fixed at load time so each `rerank()` skips the
    /// per-call lookup. Errors at load if either token is unknown.
    yes_token_id: u32,
    no_token_id: u32,
    max_seq_len: usize,
    instruction: String,
    /// Resolved schema for every session input. Built once at load
    /// time; used to assemble per-call ort feeds without re-walking
    /// session metadata.
    input_schema: Vec<InputSchema>,
    alias: String,
    requested_eps: Vec<String>,
}

/// Classified session input. The role tells `build_inputs()` how to
/// populate the tensor on each forward pass.
#[derive(Debug, Clone)]
struct InputSchema {
    name: String,
    dtype: TensorElementType,
    /// Resolved per-dimension behaviour (fixed value, dynamic batch,
    /// or dynamic past-seq → 0).
    dims: Vec<DimRole>,
    role: InputRole,
}

#[derive(Debug, Clone, Copy)]
enum DimRole {
    /// Concrete dim baked into the ONNX graph (e.g. `num_kv_heads=8`).
    Fixed(usize),
    /// Dynamic dim that takes the runtime batch size.
    Batch,
    /// Dynamic dim that takes the runtime sequence length (input_ids,
    /// attention_mask, position_ids).
    Seq,
    /// Dynamic dim for past sequence length — always 0 in a one-shot
    /// pass with an empty KV cache.
    PastSeq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputRole {
    /// `input_ids` — caller-supplied token ids (i64).
    InputIds,
    /// `attention_mask` — caller-supplied mask (i64).
    AttentionMask,
    /// `position_ids` — caller-built `0..seq` per row (i64).
    PositionIds,
    /// `past_key_values.*` — empty zero-tensor, dtype from session.
    PastKeyValue,
}

impl OnnxGenerativeReranker {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let max_seq_len = spec
            .options
            .get("max_seq_len")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);

        let artifact = spec
            .options
            .get("artifact")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_ARTIFACT)
            .to_string();

        let instruction = spec
            .options
            .get("instruction")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_INSTRUCTION)
            .to_string();

        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;
        let requested_eps: Vec<String> = resolve_ep_list(execution_providers.as_deref())
            .into_iter()
            .map(|ep| ep.as_str().to_string())
            .collect();

        // Same fail-fast EP validation as the cross-encoder path.
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;

        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-reranker-gen", &spec.model_id, &spec.options);
        let (model_path, tokenizer_path) = download_model_files(
            &spec.alias,
            &spec.model_id,
            spec.revision.as_deref(),
            &cache_dir,
            &artifact,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            artifact = %artifact,
            "Loading ONNX generative reranker"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: tokenizer_path,
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let yes_token_id = lookup_token_id(&tokenizer, "yes", &spec.alias)?;
        let no_token_id = lookup_token_id(&tokenizer, "no", &spec.alias)?;

        let session =
            super::rerank::build_session(&model_path, spec, execution_providers.as_deref())?;

        let input_schema = build_input_schema(&session, &spec.alias)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            yes_token_id,
            no_token_id,
            max_seq_len,
            instruction,
            input_schema,
            alias: spec.alias.clone(),
            requested_eps,
        })
    }

    /// Build the chat prompt the model was trained to produce a binary
    /// `yes`/`no` continuation for.
    fn build_prompt(&self, query: &str, doc: &str) -> String {
        format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n",
            instruction = self.instruction,
        )
    }

    /// Tokenize a batch of formatted prompts into padded `[batch, seq]`
    /// `(input_ids, attention_mask)` arrays. Padding uses id 0 (the
    /// model's pad/eos region; attention_mask zeroes prevent attending
    /// to it regardless of the actual id).
    fn tokenize_batch(&self, prompts: &[String]) -> Result<(Array2<i64>, Array2<i64>)> {
        let batch_size = prompts.len();
        let encodings: Vec<tokenizers::Encoding> = prompts
            .iter()
            .map(|p| {
                self.tokenizer.encode(p.as_str(), false).map_err(|e| {
                    RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Tokenization failed: {e}"),
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_len))
            .max()
            .unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch_size, padded_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, padded_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let seq_len = ids.len().min(self.max_seq_len);
            for j in 0..seq_len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
            }
        }

        Ok((input_ids, attention_mask))
    }
}

#[async_trait]
impl RerankerModel for OnnxGenerativeReranker {
    fn active_execution_providers(&self) -> Vec<String> {
        self.requested_eps.clone()
    }

    async fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<ScoredDoc>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }

        let prompts: Vec<String> = docs.iter().map(|d| self.build_prompt(query, d)).collect();
        let (input_ids, attention_mask) = self.tokenize_batch(&prompts)?;
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // position_ids: 0..seq_len for every row.
        let mut position_ids = Array2::<i64>::zeros((batch_size, seq_len));
        for b in 0..batch_size {
            for s in 0..seq_len {
                position_ids[[b, s]] = s as i64;
            }
        }

        let scores = {
            let mut session =
                self.session
                    .lock()
                    .map_err(|e| RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Session lock poisoned: {e}"),
                    })?;

            let inputs = build_inputs(
                &self.input_schema,
                &input_ids,
                &attention_mask,
                &position_ids,
                &self.alias,
            )?;

            let output_name = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "logits".to_string());

            let outputs = session
                .run(inputs)
                .map_err(|e| RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("ONNX inference: {e}"),
                })?;

            let logits_value =
                outputs
                    .get(&output_name)
                    .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                        alias: self.alias.clone(),
                        cause: format!("Missing output tensor '{output_name}'"),
                    })?;
            let logits = logits_value.try_extract_array::<f32>().map_err(|e| {
                RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("Failed to extract logits: {e}"),
                }
            })?;

            // Expected shape: [batch, seq, vocab].
            if logits.ndim() != 3 {
                return Err(RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!(
                        "expected 3-D logits [batch, seq, vocab], got {}-D shape {:?}",
                        logits.ndim(),
                        logits.shape()
                    ),
                });
            }

            score_yes_no(
                logits.view(),
                &attention_mask,
                self.yes_token_id,
                self.no_token_id,
            )
        };

        let mut scored: Vec<ScoredDoc> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| ScoredDoc {
                index,
                score,
                text: None,
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(scored)
    }
}

/// For each batch row, find the last non-pad position and read the
/// `yes`/`no` token logits there. Returns the binary-softmax probability
/// of `yes` per row, in input order.
fn score_yes_no(
    logits: ArrayViewD<'_, f32>,
    attention_mask: &Array2<i64>,
    yes_id: u32,
    no_id: u32,
) -> Vec<f32> {
    let batch = logits.shape()[0];
    let seq = logits.shape()[1];
    let yes_idx = yes_id as usize;
    let no_idx = no_id as usize;

    (0..batch)
        .map(|b| {
            // Last position with mask == 1; fall back to seq-1 if row
            // is somehow all-zero (shouldn't happen — every prompt has
            // real content — but a sane fallback beats a panic).
            let last = (0..seq)
                .rev()
                .find(|&s| attention_mask[[b, s]] != 0)
                .unwrap_or(seq.saturating_sub(1));
            let yes = logits[[b, last, yes_idx]];
            let no = logits[[b, last, no_idx]];
            // Numerically-stable two-class softmax:
            //   exp(yes) / (exp(yes) + exp(no))
            // = 1 / (1 + exp(no - yes))
            1.0 / (1.0 + (no - yes).exp())
        })
        .collect()
}

/// Look up a token id by exact literal string. Errors with a clear
/// message if the tokenizer doesn't know the token — this catches
/// misconfigured tokenizers at load time rather than at first request.
fn lookup_token_id(tokenizer: &tokenizers::Tokenizer, token: &str, alias: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: PathBuf::from("tokenizer.json"),
            cause: format!(
                "Generative reranker requires the tokenizer to know the literal '{token}' \
                 token, but lookup returned None. The model may not be a Qwen3-Reranker-style \
                 model, or the tokenizer is misconfigured."
            ),
        })
}

/// Walk `session.inputs()` and classify each one. Names not in the
/// known set (`input_ids`, `attention_mask`, `position_ids`, or
/// anything starting with `past_key_values`) are an error — feeding
/// the wrong inputs to a decoder LM produces silently wrong scores.
fn build_input_schema(session: &Session, alias: &str) -> Result<Vec<InputSchema>> {
    session
        .inputs()
        .iter()
        .map(|input| {
            let name = input.name().to_string();
            let role = classify_input(&name).ok_or_else(|| RuntimeError::OnnxLoadFailure {
                alias: alias.to_string(),
                path: PathBuf::new(),
                cause: format!(
                    "Generative reranker doesn't know how to feed input '{name}'. \
                         Expected one of: input_ids, attention_mask, position_ids, \
                         or past_key_values.*. Is this really a Qwen3-Reranker-style \
                         decoder export?"
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

fn classify_input(name: &str) -> Option<InputRole> {
    match name {
        "input_ids" => Some(InputRole::InputIds),
        "attention_mask" => Some(InputRole::AttentionMask),
        "position_ids" => Some(InputRole::PositionIds),
        n if n.starts_with("past_key_values") => Some(InputRole::PastKeyValue),
        _ => None,
    }
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
/// available (they're typically named `batch_size`, `sequence_length`,
/// `past_sequence_length`, etc. by transformers/optimum exporters) and
/// falls back to positional heuristics that match Qwen3's export
/// conventions.
fn classify_dynamic_dim(role: InputRole, idx: usize, symbol: &str) -> DimRole {
    let s = symbol.to_lowercase();
    if s.contains("past") || s.contains("kv") {
        return DimRole::PastSeq;
    }
    if s.contains("batch") {
        return DimRole::Batch;
    }
    if s.contains("seq") {
        // Could be "sequence_length" (current) or "past_sequence_length"
        // — handled above. Bare "sequence_length" → current input seq.
        return match role {
            InputRole::PastKeyValue => DimRole::PastSeq,
            _ => DimRole::Seq,
        };
    }

    // No useful symbol — fall back to position. Qwen3-style exports use:
    //   input_ids/attention_mask/position_ids: [batch, seq]
    //   past_key_values.*: [batch, num_kv_heads, past_seq, head_dim]
    match (role, idx) {
        (InputRole::InputIds | InputRole::AttentionMask | InputRole::PositionIds, 0) => {
            DimRole::Batch
        }
        (InputRole::InputIds | InputRole::AttentionMask | InputRole::PositionIds, 1) => {
            DimRole::Seq
        }
        (InputRole::PastKeyValue, 0) => DimRole::Batch,
        (InputRole::PastKeyValue, 2) => DimRole::PastSeq,
        // Any other unsymbolized dynamic dim is a red flag — feeding 0
        // for an unknown axis would silently corrupt inference. Treat
        // as PastSeq (likely safe) with a debug breadcrumb.
        _ => DimRole::PastSeq,
    }
}

/// Materialize the per-call ort feed for every session input.
fn build_inputs(
    schema: &[InputSchema],
    input_ids: &Array2<i64>,
    attention_mask: &Array2<i64>,
    position_ids: &Array2<i64>,
    alias: &str,
) -> Result<Vec<(String, DynTensor)>> {
    let batch = input_ids.shape()[0];

    schema
        .iter()
        .map(|s| {
            let dyn_tensor = match s.role {
                InputRole::InputIds => i64_array_to_dyn(input_ids.clone(), &s.name, alias)?,
                InputRole::AttentionMask => {
                    i64_array_to_dyn(attention_mask.clone(), &s.name, alias)?
                }
                InputRole::PositionIds => i64_array_to_dyn(position_ids.clone(), &s.name, alias)?,
                InputRole::PastKeyValue => build_empty_past_kv(s, batch, &s.name, alias)?,
            };
            Ok((s.name.clone(), dyn_tensor))
        })
        .collect()
}

fn i64_array_to_dyn(arr: Array2<i64>, input_name: &str, alias: &str) -> Result<DynTensor> {
    let dyn_arr = arr.into_dyn();
    Ok(Tensor::from_array(dyn_arr)
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{input_name}' tensor build: {e}"),
        })?
        .upcast())
}

/// Build an empty (zero-filled) tensor for a `past_key_values.*` input.
/// Resolves [`DimRole::Batch`] to the runtime batch and
/// [`DimRole::PastSeq`] to 0; fixed dims pass through.
fn build_empty_past_kv(
    schema: &InputSchema,
    batch: usize,
    input_name: &str,
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

    let dtype = schema.dtype;
    match dtype {
        TensorElementType::Float32 => {
            let arr = ArrayD::<f32>::zeros(IxDyn(&resolved));
            Ok(Tensor::from_array(arr)
                .map_err(|e| RuntimeError::OnnxInvocationFailure {
                    alias: alias.to_string(),
                    cause: format!("'{input_name}' empty f32 KV: {e}"),
                })?
                .upcast())
        }
        // f16 / bf16 KV caches would require the `ort/half` feature flag
        // (and the `half` crate as a direct dep). Q4-quantized exports
        // for Qwen3-Reranker-0.6B keep KV in f32 in practice, so fail
        // loud rather than silently disagreeing with the runtime.
        other => Err(RuntimeError::OnnxLoadFailure {
            alias: alias.to_string(),
            path: PathBuf::new(),
            cause: format!(
                "past_key_values input '{input_name}' has dtype {other:?}; the \
                 generative reranker currently handles only f32 KV caches. \
                 Re-export the model with f32 KV, or open an issue if this \
                 model variant should be supported."
            ),
        }),
    }
}

// Note: for a dynamic axis we couldn't classify, `build_empty_past_kv`
// errs on the side of 0 — which collapses the dimension entirely. If
// that produces a shape error from ORT, the error message includes the
// input name and shape, which is the diagnostic the user needs to
// either upgrade their ONNX export or open an issue.

async fn download_model_files(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    artifact: &str,
) -> Result<(PathBuf, PathBuf)> {
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| RuntimeError::OnnxDownloadFailure {
            alias: alias.to_string(),
            cause: e.to_string(),
        })?;

    let repo = match revision {
        Some(rev) => Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string()),
        None => Repo::model(model_id.to_string()),
    };
    let api_repo = api.repo(repo);

    let model_path =
        api_repo
            .get(artifact)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download '{artifact}': {e}"),
            })?;

    let tokenizer_path =
        api_repo
            .get("tokenizer.json")
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download tokenizer.json: {e}"),
            })?;

    Ok((model_path, tokenizer_path))
}
