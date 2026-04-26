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
//! - `raw` — arbitrary ONNX tensor execution via [`OnnxRunner`].
//! - `rerank` — cross-encoder reranking via [`RerankerModel`].

mod raw;
mod rerank;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    LoadedModelHandle, ModelProvider, OnnxRunner, ProviderCapabilities, ProviderHealth,
    RerankerModel,
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
            supported_tasks: vec![ModelTask::Raw, ModelTask::Rerank],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Raw => {
                let runner: Arc<dyn OnnxRunner> =
                    raw::load_raw(spec, self.base_dir.as_deref(), &self.sessions).await?;
                Ok(Arc::new(runner) as LoadedModelHandle)
            }
            ModelTask::Rerank => {
                let reranker: Arc<dyn RerankerModel> = rerank::load_rerank(spec).await?;
                Ok(Arc::new(reranker) as LoadedModelHandle)
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
