// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Structured NLP task for `local/onnx`.
//!
//! Implements [`NlpModel`] against the kniv-deberta cascade family
//! (canonical default: `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`).
//! One DeBERTa-v3 encoder forward pass produces all five heads in parallel:
//! POS / NER / DEP arc + label / SRL / dialog-act CLS.
//!
//! Per the model card, SRL is per-predicate: `predicate_idx` selects one
//! verb to score. To populate every frame in a sentence the provider runs
//! the forward once with `predicate_idx = 0` (sentinel "no SRL") to extract
//! POS / NER / DEP / CLS, then identifies verbs from the POS output and
//! re-runs once per verb with the matching index. This multi-pass dance is
//! invisible to callers; the `analyze` contract just asks for `NlpTasks`
//! and returns populated heads.
//!
//! Inputs longer than `max_seq_len` (default 128 — the model's training
//! cap) are chunked into non-overlapping token windows. Each chunk is
//! decoded independently; per-token byte offsets stay correct because the
//! tokenizer is run once over the full text with `Tokenizer::encode` and
//! the resulting offsets are absolute. Dependency arcs do not cross chunk
//! boundaries — each chunk gets its own local dependency tree.

use async_trait::async_trait;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, parse_execution_providers_option,
};
use crate::traits::{
    DepLink, NlpModel, NlpRequest, NlpResult, NlpSentence, NlpTasks, NlpToken, SpeechAct, SrlFrame,
    SrlRole,
};

// Defaults — match the canonical kniv-deberta repo layout.
const DEFAULT_ONNX_PATH: &str = "onnx/cascade.onnx";
const DEFAULT_TOKENIZER_PATH: &str = "tokenizer.json";
const DEFAULT_LABEL_MAPS_PATH: &str = "label_maps.json";
const DEFAULT_MAX_SEQ_LEN: usize = 128;

/// Entry point called from [`LocalOnnxProvider::load`](super::LocalOnnxProvider::load)
/// when `spec.task == ModelTask::Nlp`.
pub(super) async fn load_nlp(spec: &ModelAliasSpec) -> Result<Arc<dyn NlpModel>> {
    let model = OnnxNlpModel::load(spec).await?;
    Ok(Arc::new(model) as Arc<dyn NlpModel>)
}

/// Resolved configuration after option parsing.
struct NlpConfig {
    hf_repo: String,
    onnx_path: String,
    tokenizer_path: String,
    label_maps_path: String,
    max_seq_len: usize,
}

impl NlpConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let opts = &spec.options;
        let get_str = |key: &str| -> Option<String> {
            opts.get(key).and_then(Value::as_str).map(str::to_string)
        };
        Ok(Self {
            hf_repo: spec.model_id.clone(),
            onnx_path: get_str("onnx_path").unwrap_or_else(|| DEFAULT_ONNX_PATH.to_string()),
            tokenizer_path: get_str("tokenizer_path")
                .unwrap_or_else(|| DEFAULT_TOKENIZER_PATH.to_string()),
            label_maps_path: get_str("label_maps_path")
                .unwrap_or_else(|| DEFAULT_LABEL_MAPS_PATH.to_string()),
            max_seq_len: opts
                .get("max_seq_len")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_MAX_SEQ_LEN),
        })
    }
}

/// Authoritative tagsets parsed from `label_maps.json`.
#[derive(Debug, Clone)]
struct LabelMaps {
    pos: Vec<String>,
    ner: Vec<String>,
    srl: Vec<String>,
    cls: Vec<String>,
    deprel: Vec<String>,
}

impl LabelMaps {
    /// Parse the kniv `label_maps.json` schema.
    ///
    /// Required keys: `pos`, `ner`, `srl`, `cls`, `deprel`. Each maps to an
    /// array of strings whose position is the model's class index.
    fn parse_from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            RuntimeError::Config(format!(
                "Failed to read NLP label maps at {}: {e}",
                path.display(),
            ))
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| {
            RuntimeError::Config(format!(
                "Failed to parse NLP label maps at {}: {e}",
                path.display(),
            ))
        })?;
        let take = |key: &str| -> Result<Vec<String>> {
            value
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "NLP label maps at {} is missing required array '{key}'",
                        path.display(),
                    ))
                })?
                .iter()
                .map(|v| {
                    v.as_str().map(str::to_string).ok_or_else(|| {
                        RuntimeError::Config(format!(
                            "NLP label maps at {}: '{key}' contains non-string entries",
                            path.display(),
                        ))
                    })
                })
                .collect()
        };
        Ok(Self {
            pos: take("pos")?,
            ner: take("ner")?,
            srl: take("srl")?,
            cls: take("cls")?,
            deprel: take("deprel")?,
        })
    }
}

/// ONNX-backed multi-head structured NLP model.
struct OnnxNlpModel {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    labels: LabelMaps,
    alias: String,
    model_id: String,
    max_seq_len: usize,
}

impl OnnxNlpModel {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = NlpConfig::resolve(spec)?;
        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;

        // Validate EP list FIRST so misconfigurations fail with a precise error.
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;

        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-nlp", &cfg.hf_repo, &spec.options);
        let (model_path, tokenizer_path, label_maps_path) = download_nlp_artifacts(
            &spec.alias,
            &cfg.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &cfg.onnx_path,
            &cfg.tokenizer_path,
            &cfg.label_maps_path,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            model_path = %model_path.display(),
            max_seq_len = cfg.max_seq_len,
            "Loading ONNX NLP cascade"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: tokenizer_path,
                cause: format!("Failed to load tokenizer: {e}"),
            }
        })?;

        let labels = LabelMaps::parse_from_file(&label_maps_path)?;
        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            labels,
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            max_seq_len: cfg.max_seq_len,
        })
    }

    /// Run the cascade for one request, handling chunking and SRL multi-pass.
    fn analyze_one(&self, request: &NlpRequest<'_>) -> Result<NlpResult> {
        // Single tokenization over the whole text — offsets returned are
        // absolute UTF-8 byte offsets in `request.text`, which keeps
        // NlpToken::{start,end} stable across any chunk slicing below.
        let encoding = self.tokenizer.encode(request.text, true).map_err(|e| {
            RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Tokenization failed: {e}"),
            }
        })?;
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let offsets = encoding.get_offsets();
        let special_mask = encoding.get_special_tokens_mask();
        let total = ids.len();

        // Walk the token sequence in non-overlapping windows of size
        // `max_seq_len`. The last window may be shorter than the cap; the
        // ONNX graph declares dynamic seq so the smaller batch is fine.
        let mut all_tokens: Vec<NlpToken> = Vec::with_capacity(total);
        let mut all_frames: Vec<SrlFrame> = Vec::new();
        let mut cls_logits_first: Option<Array1<f32>> = None;

        let mut start = 0;
        while start < total {
            let end = (start + self.max_seq_len).min(total);
            let chunk_ids: Vec<i64> = ids[start..end].iter().map(|&v| v as i64).collect();
            let chunk_mask: Vec<i64> = mask[start..end].iter().map(|&v| v as i64).collect();
            let chunk_offsets = &offsets[start..end];
            let chunk_special = &special_mask[start..end];

            // Pass 1 — predicate_idx = 0 — produces POS / NER / DEP / CLS
            // for this chunk plus the CLS-level dialog act for the first
            // chunk only (subsequent chunks' CLS is dropped since dialog
            // act is a whole-utterance signal).
            let outputs = self.forward(&chunk_ids, &chunk_mask, 0)?;

            let pos_indices = argmax_last_axis(&outputs.pos_logits)?;
            let ner_indices = argmax_last_axis(&outputs.ner_logits)?;
            let dep_heads = argmax_axis(&outputs.arc_scores, 2)?; // [seq, seq] -> head per token
            // For each (token, head), look up the relation argmax over the
            // 53 deprel classes.
            let dep_relations = dep_relation_per_token(
                &outputs.label_scores,
                &dep_heads,
                self.labels.deprel.len(),
            )?;

            // Build NlpTokens for non-special tokens in this chunk, with
            // absolute byte offsets and chunk-local DEP head indices.
            let mut chunk_token_global_indices: Vec<Option<usize>> = vec![None; chunk_ids.len()];
            let chunk_start_global = all_tokens.len();
            for (i, (&off, &is_special)) in chunk_offsets.iter().zip(chunk_special).enumerate() {
                if is_special != 0 || off == (0, 0) {
                    continue;
                }
                let pos = request
                    .tasks
                    .contains(NlpTasks::POS)
                    .then(|| label_at(&self.labels.pos, pos_indices[i]));
                let ner = request
                    .tasks
                    .contains(NlpTasks::NER)
                    .then(|| label_at(&self.labels.ner, ner_indices[i]));
                let dep = if request.tasks.contains(NlpTasks::DEP) {
                    Some(DepLink {
                        head: dep_heads[i],
                        relation: label_at(&self.labels.deprel, dep_relations[i]),
                    })
                } else {
                    None
                };
                // Surface form from the absolute byte range.
                let text = request.text.get(off.0..off.1).unwrap_or("").to_string();
                chunk_token_global_indices[i] = Some(
                    chunk_start_global
                        + chunk_token_global_indices
                            .iter()
                            .filter(|x| x.is_some())
                            .count(),
                );
                all_tokens.push(NlpToken {
                    text,
                    start: off.0,
                    end: off.1,
                    pos,
                    ner,
                    dep,
                });
            }

            // Keep CLS logits from the very first chunk only.
            if start == 0 {
                cls_logits_first = Some(outputs.cls_logits);
            }

            // SRL multi-pass: identify chunk-local verb tokens via POS
            // argmax and re-run forward once per verb with the matching
            // predicate_idx. Frames are appended to the running list.
            if request.tasks.contains(NlpTasks::SRL) {
                let verb_id = self
                    .labels
                    .pos
                    .iter()
                    .position(|tag| tag == "VERB")
                    .unwrap_or(usize::MAX);
                for (local_i, &is_special) in chunk_special.iter().enumerate() {
                    if is_special != 0 {
                        continue;
                    }
                    // predicate_idx == 0 is the sentinel "no SRL"; the
                    // model card guarantees position 0 is always a special
                    // token ([CLS]). Skip it explicitly.
                    if local_i == 0 {
                        continue;
                    }
                    if pos_indices[local_i] != verb_id {
                        continue;
                    }
                    let pred_outputs = self.forward(&chunk_ids, &chunk_mask, local_i as i64)?;
                    let srl_indices = argmax_last_axis(&pred_outputs.srl_logits)?;
                    let frame = decode_srl_frame(
                        local_i,
                        &srl_indices,
                        &chunk_token_global_indices,
                        chunk_offsets,
                        chunk_special,
                        &self.labels.srl,
                    );
                    if let Some(frame) = frame {
                        all_frames.push(frame);
                    }
                }
            }

            start = end;
        }

        let mut sentences = Vec::new();
        if !all_tokens.is_empty() {
            sentences.push(NlpSentence {
                token_range: (0, all_tokens.len() - 1),
                start: all_tokens.first().map(|t| t.start).unwrap_or(0),
                end: all_tokens
                    .last()
                    .map(|t| t.end)
                    .unwrap_or(request.text.len()),
            });
        }

        let speech_acts = if request.tasks.contains(NlpTasks::CLS) {
            cls_logits_first
                .as_ref()
                .map(|logits| {
                    let id = argmax_1d(logits);
                    vec![SpeechAct {
                        sentence_index: 0,
                        label: label_at(&self.labels.cls, id),
                        confidence: softmax_at(logits, id),
                    }]
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(NlpResult {
            tokens: all_tokens,
            sentences,
            frames: all_frames,
            speech_acts,
        })
    }

    /// One ONNX forward pass on a single chunk.
    fn forward(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        predicate_idx: i64,
    ) -> Result<CascadeOutputs> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: "Empty input chunk".to_string(),
            });
        }

        let input_ids_arr = Array2::<i64>::from_shape_vec((1, seq_len), input_ids.to_vec())
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("input_ids shape error: {e}"),
            })?;
        let attention_mask_arr =
            Array2::<i64>::from_shape_vec((1, seq_len), attention_mask.to_vec()).map_err(|e| {
                RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: format!("attention_mask shape error: {e}"),
                }
            })?;
        let predicate_idx_arr = Array1::<i64>::from_vec(vec![predicate_idx]);

        let mut session = self
            .session
            .lock()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Session lock poisoned: {e}"),
            })?;

        let map_err = |e: ort::Error, ctx: &str| RuntimeError::OnnxInvocationFailure {
            alias: self.alias.clone(),
            cause: format!("{ctx}: {e}"),
        };

        let ids_tensor = ort::value::Tensor::from_array(input_ids_arr.into_dyn())
            .map_err(|e| map_err(e, "input_ids tensor"))?;
        let mask_tensor = ort::value::Tensor::from_array(attention_mask_arr.into_dyn())
            .map_err(|e| map_err(e, "attention_mask tensor"))?;
        let pred_tensor = ort::value::Tensor::from_array(predicate_idx_arr.into_dyn())
            .map_err(|e| map_err(e, "predicate_idx tensor"))?;

        let inputs: Vec<(String, ort::value::DynTensor)> = vec![
            ("input_ids".to_string(), ids_tensor.upcast()),
            ("attention_mask".to_string(), mask_tensor.upcast()),
            ("predicate_idx".to_string(), pred_tensor.upcast()),
        ];

        let outputs = session
            .run(inputs)
            .map_err(|e| map_err(e, "ONNX inference"))?;

        let pos_logits = extract_3d_first_batch(&outputs, "pos_logits", &self.alias)?;
        let ner_logits = extract_3d_first_batch(&outputs, "ner_logits", &self.alias)?;
        let arc_scores = extract_3d_first_batch(&outputs, "arc_scores", &self.alias)?;
        let label_scores = extract_4d_first_batch(&outputs, "label_scores", &self.alias)?;
        let srl_logits = extract_3d_first_batch(&outputs, "srl_logits", &self.alias)?;
        let cls_logits = extract_2d_first_batch_1d(&outputs, "cls_logits", &self.alias)?;

        Ok(CascadeOutputs {
            pos_logits,
            ner_logits,
            arc_scores,
            label_scores,
            srl_logits,
            cls_logits,
        })
    }
}

/// Per-chunk ONNX outputs (batch dim stripped — batch is always 1).
struct CascadeOutputs {
    pos_logits: Array2<f32>,   // [seq, 17]
    ner_logits: Array2<f32>,   // [seq, 37]
    arc_scores: Array2<f32>,   // [seq, seq]
    label_scores: Array3<f32>, // [seq, seq, 53]
    srl_logits: Array2<f32>,   // [seq, 42]
    cls_logits: Array1<f32>,   // [8]
}

#[async_trait]
impl NlpModel for OnnxNlpModel {
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>) -> Result<Vec<NlpResult>> {
        // The kniv graph accepts dynamic batch in principle, but our SRL
        // multi-pass orchestration is per-sentence — running each request
        // serially keeps the post-processing straightforward and matches
        // typical caller batch sizes (1–8).
        let mut results = Vec::with_capacity(requests.len());
        for req in &requests {
            results.push(self.analyze_one(req)?);
        }
        Ok(results)
    }

    fn supported_tasks(&self) -> NlpTasks {
        NlpTasks::ALL
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Output decoding helpers
// ---------------------------------------------------------------------------

fn label_at(labels: &[String], id: usize) -> String {
    labels
        .get(id)
        .cloned()
        .unwrap_or_else(|| format!("UNK_{id}"))
}

/// Argmax over the last axis of a 2-D array `[seq, classes]`.
fn argmax_last_axis(arr: &Array2<f32>) -> Result<Vec<usize>> {
    Ok((0..arr.shape()[0])
        .map(|i| {
            let row = arr.row(i);
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect())
}

/// Argmax over `axis` of a 2-D matrix; works only when axis=1 (per-row argmax).
fn argmax_axis(arr: &Array2<f32>, axis: usize) -> Result<Vec<usize>> {
    if axis != 1 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: "nlp".to_string(),
            cause: format!("argmax_axis only supports axis=1, got {axis}"),
        });
    }
    argmax_last_axis(arr)
}

/// Argmax over a 1-D array.
fn argmax_1d(arr: &Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// For each token, look up the deprel label given its already-decoded head.
///
/// `label_scores` is `[seq, seq, n_rel]`; for token `i` with head `h`,
/// `label_scores[[i, h, k]]` is the score for relation class `k`.
fn dep_relation_per_token(
    label_scores: &Array3<f32>,
    dep_heads: &[usize],
    n_rel: usize,
) -> Result<Vec<usize>> {
    let seq = label_scores.shape()[0];
    if dep_heads.len() != seq {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: "nlp".to_string(),
            cause: format!(
                "DEP shape mismatch: arc_scores seq {seq} vs heads len {}",
                dep_heads.len()
            ),
        });
    }
    if label_scores.shape()[2] != n_rel {
        warn!(
            label_scores_last = label_scores.shape()[2],
            expected = n_rel,
            "DEP label_scores last dim does not match deprel label map size; \
             will still decode but UNK_<id> may appear in output"
        );
    }
    Ok((0..seq)
        .map(|i| {
            let h = dep_heads[i].min(seq.saturating_sub(1));
            let slice = label_scores.slice(ndarray::s![i, h, ..]);
            slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect())
}

fn softmax_at(logits: &Array1<f32>, idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.get(idx).copied().unwrap_or(0.0) / sum
    } else {
        0.0
    }
}

/// Decode a BIO-tagged SRL span sequence into one [`SrlFrame`].
///
/// Returns `None` if no role span was detected. Token spans use **global**
/// NlpResult token indices via `chunk_token_global_indices`.
fn decode_srl_frame(
    predicate_local: usize,
    srl_indices: &[usize],
    chunk_token_global_indices: &[Option<usize>],
    chunk_offsets: &[(usize, usize)],
    chunk_special: &[u32],
    srl_labels: &[String],
) -> Option<SrlFrame> {
    let predicate_global = *chunk_token_global_indices.get(predicate_local)?.as_ref()?;
    let mut roles: Vec<SrlRole> = Vec::new();
    let mut current: Option<(String, usize, usize)> = None; // (label, start_global, end_global)

    let close = |cur: &Option<(String, usize, usize)>, roles: &mut Vec<SrlRole>| {
        if let Some((label, s, e)) = cur {
            roles.push(SrlRole {
                span: (*s, *e),
                label: label.clone(),
            });
        }
    };

    for (local_i, &class_id) in srl_indices.iter().enumerate() {
        if local_i >= chunk_special.len() || chunk_special[local_i] != 0 {
            close(&current, &mut roles);
            current = None;
            continue;
        }
        let global = chunk_token_global_indices.get(local_i).and_then(|x| *x);
        if global.is_none() || chunk_offsets[local_i] == (0, 0) {
            close(&current, &mut roles);
            current = None;
            continue;
        }
        let global = global.unwrap();
        let raw_label = srl_labels
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("UNK_{class_id}"));

        if raw_label == "O" || raw_label == "V" {
            close(&current, &mut roles);
            current = None;
            continue;
        }

        // BIO scheme: "B-FOO" starts a new span; "I-FOO" continues if the
        // previous tag was matching "B-FOO" / "I-FOO".
        let (prefix, base) = raw_label
            .split_once('-')
            .map(|(a, b)| (a, b.to_string()))
            .unwrap_or(("B", raw_label.clone()));

        match prefix {
            "B" => {
                close(&current, &mut roles);
                current = Some((base, global, global));
            }
            "I" => {
                if let Some((ref existing_label, _, ref mut end)) = current {
                    if existing_label == &base {
                        *end = global;
                    } else {
                        // Mismatched I- tag — treat as a fresh span.
                        close(&current, &mut roles);
                        current = Some((base, global, global));
                    }
                } else {
                    current = Some((base, global, global));
                }
            }
            _ => {
                close(&current, &mut roles);
                current = None;
            }
        }
    }
    close(&current, &mut roles);

    if roles.is_empty() {
        None
    } else {
        Some(SrlFrame {
            predicate_token: predicate_global,
            predicate_sense: None,
            roles,
        })
    }
}

// ---------------------------------------------------------------------------
// HF download + session build (parallel structure to embedding.rs)
// ---------------------------------------------------------------------------

async fn download_nlp_artifacts(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    onnx_path: &str,
    tokenizer_path: &str,
    label_maps_path: &str,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
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

    let model_file =
        api_repo
            .get(onnx_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download ONNX model '{onnx_path}': {e}"),
            })?;
    let tokenizer_file =
        api_repo
            .get(tokenizer_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download tokenizer '{tokenizer_path}': {e}"),
            })?;
    let label_file =
        api_repo
            .get(label_maps_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download label maps '{label_maps_path}': {e}"),
            })?;
    Ok((model_file, tokenizer_file, label_file))
}

fn build_session(
    path: &Path,
    spec: &ModelAliasSpec,
    execution_providers: Option<&[OnnxExecutionProvider]>,
) -> Result<Session> {
    let builder = Session::builder().map_err(|e| RuntimeError::OnnxLoadFailure {
        alias: spec.alias.clone(),
        path: path.to_path_buf(),
        cause: e.to_string(),
    })?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;
    let dispatch = build_execution_providers(execution_providers, &spec.alias, "local/onnx")?;
    builder =
        builder
            .with_execution_providers(dispatch)
            .map_err(|e| RuntimeError::OnnxLoadFailure {
                alias: spec.alias.clone(),
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
    builder
        .commit_from_file(path)
        .map_err(|e| RuntimeError::OnnxLoadFailure {
            alias: spec.alias.clone(),
            path: path.to_path_buf(),
            cause: e.to_string(),
        })
}

// ---------------------------------------------------------------------------
// Output-tensor extraction helpers — strip the always-1 batch dim.
// ---------------------------------------------------------------------------

fn extract_3d_first_batch(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    alias: &str,
) -> Result<Array2<f32>> {
    let val = outputs
        .get(name)
        .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Missing output tensor '{name}'"),
        })?;
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Extract '{name}': {e}"),
        })?;
    let arr3 = view.into_dimensionality::<ndarray::Ix3>().map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' expected rank 3: {e}"),
        }
    })?;
    let (b, s, c) = (arr3.shape()[0], arr3.shape()[1], arr3.shape()[2]);
    if b == 0 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' has zero batch dim"),
        });
    }
    // Take batch 0 — always 1 in our serial-per-request orchestration.
    let mut out = Array2::<f32>::zeros((s, c));
    for i in 0..s {
        for j in 0..c {
            out[[i, j]] = arr3[[0, i, j]];
        }
    }
    Ok(out)
}

fn extract_4d_first_batch(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    alias: &str,
) -> Result<Array3<f32>> {
    let val = outputs
        .get(name)
        .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Missing output tensor '{name}'"),
        })?;
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Extract '{name}': {e}"),
        })?;
    let arr4: Array4<f32> = view
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' expected rank 4: {e}"),
        })?
        .to_owned();
    let (b, s1, s2, c) = (
        arr4.shape()[0],
        arr4.shape()[1],
        arr4.shape()[2],
        arr4.shape()[3],
    );
    if b == 0 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' has zero batch dim"),
        });
    }
    let mut out = Array3::<f32>::zeros((s1, s2, c));
    for i in 0..s1 {
        for j in 0..s2 {
            for k in 0..c {
                out[[i, j, k]] = arr4[[0, i, j, k]];
            }
        }
    }
    Ok(out)
}

fn extract_2d_first_batch_1d(
    outputs: &ort::session::SessionOutputs,
    name: &str,
    alias: &str,
) -> Result<Array1<f32>> {
    let val = outputs
        .get(name)
        .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Missing output tensor '{name}'"),
        })?;
    let view = val
        .try_extract_array::<f32>()
        .map_err(|e| RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("Extract '{name}': {e}"),
        })?;
    let arr2 = view.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
        RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' expected rank 2: {e}"),
        }
    })?;
    let (b, c) = (arr2.shape()[0], arr2.shape()[1]);
    if b == 0 {
        return Err(RuntimeError::OnnxInvocationFailure {
            alias: alias.to_string(),
            cause: format!("'{name}' has zero batch dim"),
        });
    }
    let mut out = Array1::<f32>::zeros(c);
    for j in 0..c {
        out[j] = arr2[[0, j]];
    }
    Ok(out)
}
