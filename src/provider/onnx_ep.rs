// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared ORT execution-provider selection used by both
//! [`LocalOnnxProvider`](super::LocalOnnxProvider) and
//! [`LocalOnnxRerankerProvider`](super::LocalOnnxRerankerProvider).
//!
//! The `execution_providers` option in a model alias spec is a list of
//! string identifiers (`"cpu"`, `"cuda"`, `"coreml"`, `"directml"`).
//! This module parses those strings into an internal enum and turns the
//! list into the `Vec<ExecutionProviderDispatch>` that `ort::Session`
//! expects.
//!
//! # Default behavior
//!
//! When the spec doesn't specify `execution_providers`, we fall back to
//! a feature-aware default:
//!
//! - With `gpu-cuda`: `[Cuda, Cpu]` (CUDA preferred, CPU fallback).
//! - Otherwise: `[Cpu]`.
//!
//! # Strict-vs-fallback semantics
//!
//! When the *configured* list contains GPU EPs but does **not** include
//! `Cpu`, the last GPU EP is built with `error_on_failure()` so we don't
//! silently fall back to CPU. Otherwise every EP is built with
//! `fail_silently()` so ORT can chain through the list as written.

use ort::execution_providers::ExecutionProviderDispatch;

use ort::ep::CPU;
#[cfg(feature = "gpu-cuda")]
use ort::ep::CUDA;
#[cfg(feature = "gpu-coreml")]
use ort::ep::CoreML;
#[cfg(feature = "gpu-directml")]
use ort::ep::DirectML;

use crate::error::{Result, RuntimeError};

/// Parsed `execution_providers` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    CoreMl,
    DirectMl,
}

impl OnnxExecutionProvider {
    /// Parse a single execution-provider string. Unknown values fall
    /// back to `Cpu` (matches the historical permissive behavior of
    /// `LocalOnnxProvider`).
    pub(crate) fn from_str(value: &str) -> Self {
        match value {
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "coreml" => Self::CoreMl,
            "directml" => Self::DirectMl,
            _ => Self::Cpu,
        }
    }

    /// Human-readable label used in error messages.
    fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::CoreMl => "CoreML",
            Self::DirectMl => "DirectML",
        }
    }

    /// Stable string id used by spec options and surfaced through
    /// [`OnnxRunner::active_execution_providers`](crate::traits::OnnxRunner::active_execution_providers).
    /// Round-trips with [`OnnxExecutionProvider::from_str`].
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
        }
    }
}

/// Resolve the EP list that will be used for a session: either the
/// caller-provided list verbatim, or the feature-aware default.
pub(crate) fn resolve_ep_list(
    configured: Option<&[OnnxExecutionProvider]>,
) -> Vec<OnnxExecutionProvider> {
    configured
        .map(<[OnnxExecutionProvider]>::to_vec)
        .unwrap_or_else(default_execution_providers)
}

/// Default EP list when the spec doesn't specify one.
pub(crate) fn default_execution_providers() -> Vec<OnnxExecutionProvider> {
    #[cfg(feature = "gpu-cuda")]
    {
        vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
    }
    #[cfg(not(feature = "gpu-cuda"))]
    {
        vec![OnnxExecutionProvider::Cpu]
    }
}

/// Build the `Vec<ExecutionProviderDispatch>` to hand to
/// `Session::builder().with_execution_providers(...)`.
///
/// `configured` is the user-supplied list (or `None` to use defaults).
/// `provider_label` is the provider id string used only in error
/// messages (e.g. `"local/onnx"` or `"local/onnx-reranker"`) so failures
/// point at the right alias.
pub(crate) fn build_execution_providers(
    configured: Option<&[OnnxExecutionProvider]>,
    alias: &str,
    provider_label: &str,
) -> Result<Vec<ExecutionProviderDispatch>> {
    let providers = configured
        .map(|value| value.to_vec())
        .unwrap_or_else(default_execution_providers);
    let cpu_present = providers.contains(&OnnxExecutionProvider::Cpu);
    let last_index = providers.len().saturating_sub(1);

    providers
        .into_iter()
        .enumerate()
        .map(|(index, provider)| {
            let strict = configured.is_some() && !cpu_present && index == last_index;
            execution_provider_dispatch(provider, strict, alias, provider_label)
        })
        .collect()
}

fn execution_provider_dispatch(
    provider: OnnxExecutionProvider,
    strict: bool,
    alias: &str,
    provider_label: &str,
) -> Result<ExecutionProviderDispatch> {
    let dispatch = match provider {
        OnnxExecutionProvider::Cpu => CPU::default().build(),
        OnnxExecutionProvider::Cuda => {
            #[cfg(feature = "gpu-cuda")]
            {
                CUDA::default().build()
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                return Err(feature_not_enabled(provider, alias, provider_label));
            }
        }
        OnnxExecutionProvider::CoreMl => {
            #[cfg(feature = "gpu-coreml")]
            {
                CoreML::default().build()
            }
            #[cfg(not(feature = "gpu-coreml"))]
            {
                return Err(feature_not_enabled(provider, alias, provider_label));
            }
        }
        OnnxExecutionProvider::DirectMl => {
            #[cfg(feature = "gpu-directml")]
            {
                DirectML::default().build()
            }
            #[cfg(not(feature = "gpu-directml"))]
            {
                return Err(feature_not_enabled(provider, alias, provider_label));
            }
        }
    };

    Ok(if strict {
        dispatch.error_on_failure()
    } else {
        dispatch.fail_silently()
    })
}

fn feature_not_enabled(
    provider: OnnxExecutionProvider,
    alias: &str,
    provider_label: &str,
) -> RuntimeError {
    let feature = match provider {
        OnnxExecutionProvider::Cuda => "gpu-cuda",
        OnnxExecutionProvider::CoreMl => "gpu-coreml",
        OnnxExecutionProvider::DirectMl => "gpu-directml",
        OnnxExecutionProvider::Cpu => unreachable!("CPU is always available"),
    };
    RuntimeError::Config(format!(
        "Alias '{alias}' requested {} execution for {provider_label}, but {feature} is not enabled",
        provider.label()
    ))
}

/// Convenience: parse a `serde_json::Value` array into a list of
/// `OnnxExecutionProvider`s. Accepts either a JSON array of strings or
/// a single string. Returns `None` when the value isn't present.
pub(crate) fn parse_execution_providers_option(
    value: Option<&serde_json::Value>,
) -> Option<Vec<OnnxExecutionProvider>> {
    let value = value?;
    if let Some(arr) = value.as_array() {
        Some(
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(OnnxExecutionProvider::from_str)
                .collect(),
        )
    } else {
        value
            .as_str()
            .map(|s| vec![OnnxExecutionProvider::from_str(s)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_known_values() {
        assert_eq!(
            OnnxExecutionProvider::from_str("cpu"),
            OnnxExecutionProvider::Cpu
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("cuda"),
            OnnxExecutionProvider::Cuda
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("coreml"),
            OnnxExecutionProvider::CoreMl
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("directml"),
            OnnxExecutionProvider::DirectMl
        );
    }

    #[test]
    fn from_str_unknown_falls_back_to_cpu() {
        assert_eq!(
            OnnxExecutionProvider::from_str("rocm"),
            OnnxExecutionProvider::Cpu
        );
        assert_eq!(
            OnnxExecutionProvider::from_str(""),
            OnnxExecutionProvider::Cpu
        );
    }

    #[test]
    fn parse_array_form() {
        let v = serde_json::json!(["cuda", "cpu"]);
        let parsed = parse_execution_providers_option(Some(&v));
        assert_eq!(
            parsed,
            Some(vec![
                OnnxExecutionProvider::Cuda,
                OnnxExecutionProvider::Cpu
            ])
        );
    }

    #[test]
    fn parse_string_form() {
        let v = serde_json::json!("cuda");
        let parsed = parse_execution_providers_option(Some(&v));
        assert_eq!(parsed, Some(vec![OnnxExecutionProvider::Cuda]));
    }

    #[test]
    fn parse_missing_returns_none() {
        assert!(parse_execution_providers_option(None).is_none());
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn default_prefers_cuda_when_enabled() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(not(feature = "gpu-cuda"))]
    #[test]
    fn default_is_cpu_only_without_cuda() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::Cpu]
        );
    }

    #[test]
    fn unsupported_feature_returns_config_error() {
        // CoreML on Linux without gpu-coreml
        #[cfg(not(feature = "gpu-coreml"))]
        {
            let result = build_execution_providers(
                Some(&[OnnxExecutionProvider::CoreMl]),
                "test/alias",
                "local/test",
            );
            assert!(matches!(result, Err(RuntimeError::Config(_))));
        }
    }
}
