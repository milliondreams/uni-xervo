// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Local ONNX Runtime provider (`local/onnx`).
//!
//! Single backend, multiple tasks. The provider declares
//! [`ModelTask::Raw`] and [`ModelTask::Rerank`] in [`ProviderCapabilities`]
//! and dispatches inside [`ModelProvider::load`] — matching the pattern
//! used by `cohere.rs`, `mistralrs.rs`, etc.
//!
//! Task implementations live in private submodules:
//!
//! - `raw` — arbitrary ONNX tensor execution via [`RawTensorModel`].
//! - `rerank` — cross-encoder reranking via [`RerankerModel`].
//! - `embedding` — dense text embeddings via [`EmbeddingModel`].

mod decoder_inputs;
mod embedding;
mod nlp;
mod presets;
mod raw;
mod rerank;
mod rerank_generative;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    EmbeddingModel, LoadedModelHandle, ModelProvider, NlpModel, ProviderCapabilities,
    ProviderHealth, RawTensorModel, RerankerModel,
};

pub struct LocalOnnxProvider {
    base_dir: Option<PathBuf>,
    sessions: DashMap<String, Arc<raw::LoadedOnnxSession>>,
}

impl Default for LocalOnnxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalOnnxProvider {
    pub fn new() -> Self {
        Self {
            base_dir: None,
            sessions: DashMap::new(),
        }
    }

    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(base_dir.into());
        self
    }
}

#[async_trait]
impl ModelProvider for LocalOnnxProvider {
    fn provider_id(&self) -> &'static str {
        "local/onnx"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![
                ModelTask::Raw,
                ModelTask::Rerank,
                ModelTask::Embed,
                ModelTask::Nlp,
            ],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Raw => {
                let runner: Arc<dyn RawTensorModel> =
                    raw::load_raw(spec, self.base_dir.as_deref(), &self.sessions).await?;
                Ok(Arc::new(runner) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let style = spec
                    .options
                    .get("style")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cross-encoder");
                let reranker: Arc<dyn RerankerModel> = match style {
                    "cross-encoder" => rerank::load_rerank(spec).await?,
                    "generative" => rerank_generative::load(spec).await?,
                    other => {
                        return Err(RuntimeError::Config(format!(
                            "Unknown rerank style '{other}' for alias '{}'; expected one of cross-encoder, generative",
                            spec.alias
                        )));
                    }
                };
                Ok(Arc::new(reranker) as LoadedModelHandle)
            }
            ModelTask::Embed => {
                let embedder: Arc<dyn EmbeddingModel> = embedding::load_embedding(spec).await?;
                Ok(Arc::new(embedder) as LoadedModelHandle)
            }
            ModelTask::Nlp => {
                let model: Arc<dyn NlpModel> = nlp::load_nlp(spec).await?;
                Ok(Arc::new(model) as LoadedModelHandle)
            }
            _ => Err(RuntimeError::CapabilityMismatch(format!(
                "ONNX provider does not support task {:?}",
                spec.task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}
