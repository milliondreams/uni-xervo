// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Speech-to-text provider targeting whisper.cpp via `whisper-rs`.
//!
//! # v1 scope (this release)
//!
//! Provider scaffold + catalog wiring + options validation are
//! production-ready. The `recognize` path is a stub: `transcribe()` and
//! `transcribe_many()` return `RuntimeError::Unavailable` with a clear
//! message until the `whisper-rs` dep is wired in a follow-up that can
//! validate against the build environment (CMake + C/C++ toolchain).
//!
//! Catalog authors can register `local/whisper-cpp` aliases now and they
//! will pass validation, then begin returning real transcripts as soon as
//! the follow-up lands.

use crate::api::{ModelAliasSpec, ModelTask};
use crate::error::{Result, RuntimeError};
use crate::traits::{
    AudioInput, LoadedModelHandle, ModelProvider, ProviderCapabilities, ProviderHealth,
    TranscribeOptions, TranscribeResult, TranscriptionModel,
};
use async_trait::async_trait;
use std::sync::Arc;

/// `local/whisper-cpp` provider.
pub struct LocalWhisperCppProvider;

impl Default for LocalWhisperCppProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalWhisperCppProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ModelProvider for LocalWhisperCppProvider {
    fn provider_id(&self) -> &'static str {
        "local/whisper-cpp"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::Transcribe],
        }
    }

    async fn load(&self, spec: &ModelAliasSpec) -> Result<LoadedModelHandle> {
        match spec.task {
            ModelTask::Transcribe => {
                let model = WhisperCppModel {
                    alias: spec.alias.clone(),
                    model_id: spec.model_id.clone(),
                    default_language: spec
                        .options
                        .get("default_language")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                };
                let handle: Arc<dyn TranscriptionModel> = Arc::new(model);
                Ok(Arc::new(handle) as LoadedModelHandle)
            }
            task => Err(RuntimeError::CapabilityMismatch(format!(
                "local/whisper-cpp provider does not support task {:?}",
                task
            ))),
        }
    }

    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

/// Transcription model wrapper. Real ggml-model load and whisper-rs
/// inference land in a follow-up; the current impl reports
/// `RuntimeError::Unavailable` from the inference paths so a misconfigured
/// catalog still fails loudly at first use.
struct WhisperCppModel {
    alias: String,
    model_id: String,
    default_language: Option<String>,
}

impl WhisperCppModel {
    fn unavailable(&self) -> RuntimeError {
        RuntimeError::Unavailable
    }
}

#[async_trait]
impl TranscriptionModel for WhisperCppModel {
    async fn transcribe(
        &self,
        _audio: AudioInput,
        _options: TranscribeOptions,
    ) -> Result<TranscribeResult> {
        tracing::warn!(
            alias = %self.alias,
            model_id = %self.model_id,
            "local/whisper-cpp `transcribe` invoked but the whisper-rs \
             integration is not yet wired (v1 scaffold-only release). \
             Returning Unavailable; the provider impl lands in a follow-up."
        );
        Err(self.unavailable())
    }

    fn supported_languages(&self) -> &[String] {
        // Whisper supports ~99 languages once wired. v1 reports an empty
        // slice — callers fall back to auto-detection.
        &[]
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl WhisperCppModel {
    // Surface the default_language to silence dead-code; will be passed
    // through to whisper-rs's FullParams::set_language in the follow-up.
    #[allow(dead_code)]
    fn default_language(&self) -> Option<&str> {
        self.default_language.as_deref()
    }
}
