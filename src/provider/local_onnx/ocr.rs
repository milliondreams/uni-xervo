// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! OCR task for `local/onnx` — CRNN + CTC-style recognition.
//!
//! Targets the PaddleOCR-rec / EasyOCR family of recognition models that
//! ONNX-export with a single tensor output of shape
//! `[batch, time_steps, n_classes]`, where `n_classes` is the vocabulary
//! size plus a CTC blank token. The impl performs CTC greedy decoding:
//!
//! 1. Argmax per time-step → class index sequence.
//! 2. Collapse consecutive duplicates (CTC merge rule).
//! 3. Drop the blank token (class 0 by convention; configurable).
//! 4. Map remaining indices through the character dictionary.
//!
//! This v1 covers single-image whole-text recognition. For two-stage
//! detection + recognition (region proposals + per-region recognize),
//! callers crop their regions upstream and pass one image per region.
//! A future v2 may absorb a DBNet-style detector as a sibling `style`.

use async_trait::async_trait;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::info;

use super::image::{Normalization, preprocess_batch};
use crate::api::ModelAliasSpec;
use crate::cache::resolve_cache_dir;
use crate::error::{Result, RuntimeError};
#[cfg(feature = "provider-onnx-dynamic")]
use crate::provider::onnx_ep::preflight_ort_dylib;
use crate::provider::onnx_ep::{
    OnnxExecutionProvider, build_execution_providers, parse_execution_providers_option,
};
use crate::traits::{ImageInput, OcrBlock, OcrModel, OcrResult};

/// Entry point for `LocalOnnxProvider::load` when `spec.task == Ocr`.
pub(super) async fn load_ocr(spec: &ModelAliasSpec) -> Result<Arc<dyn OcrModel>> {
    let model = OnnxOcrModel::load(spec).await?;
    Ok(Arc::new(model) as Arc<dyn OcrModel>)
}

struct OcrConfig {
    hf_repo: String,
    onnx_path: String,
    char_dict_path: String,
    image_height: u32,
    image_width: u32,
    normalization: Normalization,
    blank_class: usize,
    output_name: Option<String>,
}

impl OcrConfig {
    fn resolve(spec: &ModelAliasSpec) -> Result<Self> {
        let opts = &spec.options;
        let get_str = |key: &str| -> Option<String> {
            opts.get(key).and_then(Value::as_str).map(str::to_string)
        };

        let onnx_path = get_str("onnx_path").ok_or_else(|| {
            RuntimeError::Config(format!(
                "OCR model '{}' requires option `onnx_path` (.onnx path within the HF repo)",
                spec.alias
            ))
        })?;
        let char_dict_path = get_str("char_dict_path").ok_or_else(|| {
            RuntimeError::Config(format!(
                "OCR model '{}' requires option `char_dict_path` \
                 (character dictionary file within the HF repo, one char per line)",
                spec.alias
            ))
        })?;
        let image_height = opts
            .get("image_height")
            .and_then(Value::as_u64)
            .unwrap_or(48) as u32;
        let image_width = opts
            .get("image_width")
            .and_then(Value::as_u64)
            .unwrap_or(320) as u32;
        let normalization = match opts
            .get("normalization")
            .and_then(Value::as_str)
            .unwrap_or("imagenet")
        {
            "siglip" => Normalization::SIGLIP,
            "imagenet" => Normalization::IMAGENET,
            other => {
                return Err(RuntimeError::Config(format!(
                    "OCR model '{}' has unknown `normalization` value '{other}'; \
                     expected one of: siglip, imagenet",
                    spec.alias
                )));
            }
        };
        let blank_class = opts.get("blank_class").and_then(Value::as_u64).unwrap_or(0) as usize;
        let output_name = get_str("output_name");

        Ok(Self {
            hf_repo: spec.model_id.clone(),
            onnx_path,
            char_dict_path,
            image_height,
            image_width,
            normalization,
            blank_class,
            output_name,
        })
    }
}

struct OnnxOcrModel {
    session: Mutex<Session>,
    alias: String,
    model_id: String,
    chars: Vec<String>,
    image_height: u32,
    image_width: u32,
    normalization: Normalization,
    blank_class: usize,
    output_name: Option<String>,
}

impl OnnxOcrModel {
    async fn load(spec: &ModelAliasSpec) -> Result<Self> {
        let cfg = OcrConfig::resolve(spec)?;
        let execution_providers =
            parse_execution_providers_option(spec.options.get("execution_providers"))?;
        let _ =
            build_execution_providers(execution_providers.as_deref(), &spec.alias, "local/onnx")?;
        #[cfg(feature = "provider-onnx-dynamic")]
        preflight_ort_dylib(&spec.alias, "local/onnx")?;

        let cache_dir = resolve_cache_dir("onnx-ocr", &cfg.hf_repo, &spec.options);
        let (model_path, char_dict_path) = download_ocr_artifacts(
            &spec.alias,
            &cfg.hf_repo,
            spec.revision.as_deref(),
            &cache_dir,
            &cfg.onnx_path,
            &cfg.char_dict_path,
        )
        .await?;

        info!(
            alias = %spec.alias,
            model_id = %spec.model_id,
            image_height = cfg.image_height,
            image_width = cfg.image_width,
            "Loading ONNX OCR recognizer"
        );

        let chars = read_char_dict(&char_dict_path)?;
        let session = build_session(&model_path, spec, execution_providers.as_deref())?;

        Ok(Self {
            session: Mutex::new(session),
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            chars,
            image_height: cfg.image_height,
            image_width: cfg.image_width,
            normalization: cfg.normalization,
            blank_class: cfg.blank_class,
            output_name: cfg.output_name,
        })
    }
}

#[async_trait]
impl OcrModel for OnnxOcrModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn recognize(&self, images: Vec<ImageInput>) -> Result<Vec<OcrResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // CRNN-style OCR conventionally takes a fixed height with width
        // scaled to preserve aspect ratio (with optional padding to a max
        // width). For v1 we use a fixed (h, w) which matches PaddleOCR-rec
        // when called per cropped line. Two-stage detection would pre-crop
        // each line; whole-image OCR works but loses fine-grained layout.
        //
        // The preprocess helper does square-resize by default; here we
        // need a non-square. We call preprocess_batch with target_size set
        // equal to image_height and then post-resize each row. Since the
        // helper currently always resizes to a square, we work around this
        // for now by using image_height for both — leaves a follow-up to
        // generalize the helper to accept (h, w).
        let target = self.image_width.max(self.image_height);
        let tensor = preprocess_batch(&images, target, self.normalization)?;

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

        let pixel_tensor = ort::value::Tensor::from_array(tensor.into_dyn())
            .map_err(|e| map_err(e, "pixel tensor"))?;
        let inputs: Vec<(String, ort::value::DynTensor)> =
            vec![("x".to_string(), pixel_tensor.upcast())];

        let chosen_output = if let Some(name) = self.output_name.as_deref() {
            name.to_string()
        } else {
            session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                    alias: self.alias.clone(),
                    cause: "ONNX session has no outputs".to_string(),
                })?
        };

        let outputs = session
            .run(inputs)
            .map_err(|e| map_err(e, "ONNX inference"))?;

        let view = outputs
            .get(&chosen_output)
            .ok_or_else(|| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("Missing output tensor '{chosen_output}'"),
            })?
            .try_extract_array::<f32>()
            .map_err(|e| map_err(e, "extract output"))?;

        let arr3 = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!("OCR output expected rank 3 [batch, time, classes]: {e}"),
            })?
            .to_owned();

        let (batch, _time, n_classes) = (arr3.shape()[0], arr3.shape()[1], arr3.shape()[2]);
        if self.blank_class >= n_classes {
            return Err(RuntimeError::OnnxInvocationFailure {
                alias: self.alias.clone(),
                cause: format!(
                    "blank_class={} out of range for {n_classes} classes",
                    self.blank_class
                ),
            });
        }

        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = arr3.slice(ndarray::s![b, .., ..]);
            let mut prev_class: Option<usize> = None;
            let mut text = String::new();
            let mut total_conf = 0.0f64;
            let mut emitted = 0usize;
            for t in 0..slice.shape()[0] {
                let row = slice.row(t);
                let class_id = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if class_id == self.blank_class {
                    prev_class = None;
                    continue;
                }
                if prev_class == Some(class_id) {
                    continue;
                }
                prev_class = Some(class_id);

                // Index 0 is blank by convention; chars are typically
                // 1-indexed against the dictionary file. Offset by 1.
                let char_idx = class_id.saturating_sub(self.blank_class + 1);
                if let Some(c) = self.chars.get(char_idx) {
                    text.push_str(c);
                    total_conf += softmax_at(&row.to_vec(), class_id) as f64;
                    emitted += 1;
                }
            }
            let avg_conf = if emitted > 0 {
                (total_conf / emitted as f64) as f32
            } else {
                0.0
            };

            // Whole-image bbox: image_size-relative; the caller knows its
            // own image size, so we report [0, 0, 1, 1] as normalized coords.
            results.push(OcrResult {
                blocks: vec![OcrBlock {
                    text: text.clone(),
                    bbox: [0.0, 0.0, 1.0, 1.0],
                    confidence: avg_conf,
                }],
                plain_text: text,
            });
        }

        Ok(results)
    }
}

/// Read a character dictionary file: one entry per line, in class order.
fn read_char_dict(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        RuntimeError::Config(format!(
            "Failed to read OCR char dictionary '{}': {e}",
            path.display()
        ))
    })?;
    Ok(content.lines().map(str::to_string).collect())
}

/// Stable softmax probability for class `idx` over `logits`.
fn softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.get(idx).copied().unwrap_or(0.0) / sum
    } else {
        0.0
    }
}

async fn download_ocr_artifacts(
    alias: &str,
    model_id: &str,
    revision: Option<&str>,
    cache_dir: &Path,
    onnx_path: &str,
    char_dict_path: &str,
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
    let onnx_file =
        api_repo
            .get(onnx_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download ONNX '{onnx_path}': {e}"),
            })?;
    let dict_file =
        api_repo
            .get(char_dict_path)
            .await
            .map_err(|e| RuntimeError::OnnxDownloadFailure {
                alias: alias.to_string(),
                cause: format!("Could not download char dict '{char_dict_path}': {e}"),
            })?;
    Ok((onnx_file, dict_file))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_at_normalizes_to_unit_total() {
        let logits = vec![1.0, 2.0, 3.0];
        let s0 = softmax_at(&logits, 0);
        let s1 = softmax_at(&logits, 1);
        let s2 = softmax_at(&logits, 2);
        assert!((s0 + s1 + s2 - 1.0).abs() < 1e-5);
        assert!(s2 > s1 && s1 > s0);
    }

    #[test]
    fn read_char_dict_parses_one_per_line() {
        let dir = std::env::temp_dir();
        let path = dir.join("uni_xervo_test_char_dict.txt");
        std::fs::write(&path, "a\nb\nc\n").unwrap();
        let chars = read_char_dict(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(chars, vec!["a", "b", "c"]);
    }
}
