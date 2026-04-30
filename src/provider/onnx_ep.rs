// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared ORT execution-provider selection used by both task implementations
//! of [`LocalOnnxProvider`](super::LocalOnnxProvider) (raw and rerank).
//!
//! The `execution_providers` option in a model alias spec is a list of
//! string identifiers. The always-recognized values are `"cpu"`, `"cuda"`,
//! `"coreml"`. When built with `provider-onnx-dynamic`, the vendor strings
//! `"rocm"`, `"directml"`, `"openvino"`, `"qnn"`, `"tensorrt"`, `"webgpu"`
//! are also accepted — they require the user to supply a matching ORT
//! library via `ORT_DYLIB_PATH`. This module parses those strings into an
//! internal enum and turns the list into the `Vec<ExecutionProviderDispatch>`
//! that `ort::Session` expects.
//!
//! # Default behavior
//!
//! When the spec doesn't specify `execution_providers`, we fall back to
//! a feature-aware default:
//!
//! - With `gpu-cuda`:  `[Cuda, Cpu]`   (CUDA preferred, CPU fallback).
//! - With `gpu-metal`: `[CoreMl, Cpu]` (CoreML preferred, CPU fallback).
//! - Otherwise:        `[Cpu]`.
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
#[cfg(feature = "gpu-metal")]
use ort::ep::CoreML;
#[cfg(feature = "provider-onnx-dynamic")]
use ort::ep::{DirectML, OpenVINO, QNN, ROCm, TensorRT, WebGPU};

use crate::error::{Result, RuntimeError};

/// Parsed `execution_providers` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnnxExecutionProvider {
    Cpu,
    Cuda,
    CoreMl,
    Rocm,
    DirectMl,
    OpenVino,
    Qnn,
    TensorRt,
    WebGpu,
}

impl OnnxExecutionProvider {
    /// Parse a single execution-provider string. Returns `None` for
    /// unrecognized values so the caller can surface a precise
    /// configuration error rather than silently falling back to CPU.
    pub(crate) fn from_str(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "coreml" => Some(Self::CoreMl),
            "rocm" => Some(Self::Rocm),
            "directml" => Some(Self::DirectMl),
            "openvino" => Some(Self::OpenVino),
            "qnn" => Some(Self::Qnn),
            "tensorrt" => Some(Self::TensorRt),
            "webgpu" => Some(Self::WebGpu),
            _ => None,
        }
    }

    /// Human-readable label used in error messages.
    fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::CoreMl => "CoreML",
            Self::Rocm => "ROCm",
            Self::DirectMl => "DirectML",
            Self::OpenVino => "OpenVINO",
            Self::Qnn => "QNN",
            Self::TensorRt => "TensorRT",
            Self::WebGpu => "WebGPU",
        }
    }

    /// Stable string id used by spec options and surfaced through
    /// [`RawTensorModel::active_execution_providers`](crate::traits::RawTensorModel::active_execution_providers).
    /// Round-trips with [`OnnxExecutionProvider::from_str`].
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::Rocm => "rocm",
            Self::DirectMl => "directml",
            Self::OpenVino => "openvino",
            Self::Qnn => "qnn",
            Self::TensorRt => "tensorrt",
            Self::WebGpu => "webgpu",
        }
    }

    /// True for execution providers that require `provider-onnx-dynamic`
    /// (a vendor-supplied ORT library loaded via `ORT_DYLIB_PATH`). The
    /// pyke-bundled binary used by `provider-onnx` doesn't include them.
    fn requires_dynamic(&self) -> bool {
        matches!(
            self,
            Self::Rocm
                | Self::DirectMl
                | Self::OpenVino
                | Self::Qnn
                | Self::TensorRt
                | Self::WebGpu
        )
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
    #[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
    {
        vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
    }
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    {
        vec![OnnxExecutionProvider::Cpu]
    }
}

/// Build the `Vec<ExecutionProviderDispatch>` to hand to
/// `Session::builder().with_execution_providers(...)`.
///
/// `configured` is the user-supplied list (or `None` to use defaults).
/// `provider_label` is the provider id string used only in error
/// messages (e.g. `"local/onnx"`) so failures point at the right alias.
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

/// Build a vendor-EP dispatch when `provider-onnx-dynamic` is active, or
/// return a Config error pointing the user at the right feature otherwise.
/// Implemented as a macro so the early-return flows through the caller's
/// `match` arm.
macro_rules! vendor_dispatch {
    ($provider:ident, $alias:ident, $provider_label:ident, $ep:ident) => {{
        #[cfg(feature = "provider-onnx-dynamic")]
        {
            $ep::default().build()
        }
        #[cfg(not(feature = "provider-onnx-dynamic"))]
        {
            return Err(feature_not_enabled($provider, $alias, $provider_label));
        }
    }};
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
            #[cfg(feature = "gpu-metal")]
            {
                CoreML::default().build()
            }
            #[cfg(not(feature = "gpu-metal"))]
            {
                return Err(feature_not_enabled(provider, alias, provider_label));
            }
        }
        OnnxExecutionProvider::Rocm => vendor_dispatch!(provider, alias, provider_label, ROCm),
        OnnxExecutionProvider::DirectMl => {
            vendor_dispatch!(provider, alias, provider_label, DirectML)
        }
        OnnxExecutionProvider::OpenVino => {
            vendor_dispatch!(provider, alias, provider_label, OpenVINO)
        }
        OnnxExecutionProvider::Qnn => vendor_dispatch!(provider, alias, provider_label, QNN),
        OnnxExecutionProvider::TensorRt => {
            vendor_dispatch!(provider, alias, provider_label, TensorRT)
        }
        OnnxExecutionProvider::WebGpu => {
            vendor_dispatch!(provider, alias, provider_label, WebGPU)
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
    if provider.requires_dynamic() {
        return RuntimeError::Config(format!(
            "Alias '{alias}' requested {} execution for {provider_label}, but vendor execution providers require the `provider-onnx-dynamic` feature plus a vendor-supplied ONNX Runtime library via ORT_DYLIB_PATH",
            provider.label()
        ));
    }
    let feature = match provider {
        OnnxExecutionProvider::Cuda => "gpu-cuda",
        OnnxExecutionProvider::CoreMl => "gpu-metal",
        OnnxExecutionProvider::Cpu => unreachable!("CPU is always available"),
        _ => unreachable!("vendor EPs handled above"),
    };
    RuntimeError::Config(format!(
        "Alias '{alias}' requested {} execution for {provider_label}, but {feature} is not enabled",
        provider.label()
    ))
}

/// Parse a `serde_json::Value` array into a list of
/// `OnnxExecutionProvider`s. Accepts either a JSON array of strings or
/// a single string.
///
/// Returns:
/// - `Ok(None)` when the value isn't present, or when an explicit empty
///   array is given (caller falls back to the feature-aware default).
/// - `Err(RuntimeError::Config)` when an entry is unrecognized or has the
///   wrong JSON shape — surfaces typos and stale docs at load time
///   instead of letting them silently degrade to CPU.
pub(crate) fn parse_execution_providers_option(
    value: Option<&serde_json::Value>,
) -> Result<Option<Vec<OnnxExecutionProvider>>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let raw: Vec<&str> = if let Some(arr) = value.as_array() {
        arr.iter()
            .map(|v| {
                v.as_str().ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "execution_providers entries must be strings, got {v}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else if let Some(s) = value.as_str() {
        vec![s]
    } else {
        return Err(RuntimeError::Config(format!(
            "execution_providers must be a string or array of strings, got {value}"
        )));
    };

    let providers: Vec<OnnxExecutionProvider> = raw
        .into_iter()
        .map(|s| {
            OnnxExecutionProvider::from_str(s).ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Unknown execution_providers entry `{s}`. \
                     Recognized values: cpu, cuda, coreml, rocm, directml, openvino, qnn, tensorrt, webgpu."
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if providers.is_empty() {
        Ok(None)
    } else {
        Ok(Some(providers))
    }
}

/// Default ONNX Runtime dylib filename for the current platform. Mirrors
/// what ort itself searches for when `ORT_DYLIB_PATH` is unset (see
/// ort 2.0.0-rc.12 `src/lib.rs:188-194`).
///
/// Only present under `provider-onnx-dynamic`. Under `provider-onnx`
/// (bundled CPU) the lib is statically linked into the binary; there's
/// no dlopen and the preflight is meaningless.
#[cfg(feature = "provider-onnx-dynamic")]
fn default_dylib_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        "libonnxruntime.so"
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    )))]
    {
        "libonnxruntime.so"
    }
}

/// Pre-flight check that fails fast if the ONNX Runtime dynamic library
/// can't be loaded.
///
/// # Why this exists
///
/// `ort` 2.0.0-rc.12 has a re-entrant `OnceLock` deadlock in its
/// load-dynamic error path: when `libloading::Library::new()` fails,
/// the failure is wrapped in `ort::Error::new(...)`, whose constructor
/// calls back into `ort::api()` to format the status message — which
/// is exactly the `OnceLock` whose initializer just failed. The thread
/// blocks on the same futex forever and the panic that `setup_api`
/// intends to throw never fires.
///
/// This was reported as [pykeio/ort#560] and fixed by [`17ed7277`] but
/// not yet released as of `rc.12`. Until a new ort release ships, we
/// intercept the missing-dylib case here — *before* any code touches
/// the ort API — and surface a clear `Config` error instead.
///
/// # Behavior
///
/// 1. If `ORT_DYLIB_PATH` is set, attempt `libloading::Library::new(path)`
///    and immediately drop the handle.
/// 2. Otherwise, attempt the platform-default name
///    (`libonnxruntime.so` / `.dylib` / `onnxruntime.dll`). The dynamic
///    loader walks `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / standard
///    system paths, so brew/system installs are auto-detected.
/// 3. On failure, return `RuntimeError::Config` naming the alias, the
///    attempted path, the OS error, and the migration-doc reference.
/// 4. On success, drop the handle. The lib stays in the loader's
///    in-memory cache (refcount), so ort's subsequent `dlopen` finds it
///    instantly without a second disk read.
///
/// Idempotent: safe to call from multiple loads. Cheap (one syscall on
/// the success path; one syscall + one error format on the failure path).
///
/// # Limitations
///
/// Does **not** catch ABI-version mismatches — those still hit the ort
/// deadlock if they happen. Mitigation: the pinned `ort = "=2.0.0-rc.12"`
/// + matched ORT runtime tarball makes version mismatches unlikely.
///
/// [pykeio/ort#560]: https://github.com/pykeio/ort/issues/560
/// [`17ed7277`]: https://github.com/pykeio/ort/commit/17ed7277
#[cfg(feature = "provider-onnx-dynamic")]
pub(crate) fn preflight_ort_dylib(alias: &str, provider_label: &str) -> Result<()> {
    let (path_str, source) = match std::env::var("ORT_DYLIB_PATH") {
        Ok(s) if !s.is_empty() => (s, "ORT_DYLIB_PATH env var"),
        _ => (default_dylib_name().to_string(), "platform default"),
    };

    // SAFETY: libloading::Library::new is unsafe because the loaded library
    // may run init code (DT_INIT, DllMain). For ONNX Runtime this is well-defined.
    match unsafe { libloading::Library::new(&path_str) } {
        Ok(lib) => {
            drop(lib);
            Ok(())
        }
        Err(e) => Err(RuntimeError::Config(format!(
            "Alias '{alias}' ({provider_label}): cannot load ONNX Runtime dynamic library `{path_str}` (from {source}): {e}.\n\
             Set ORT_DYLIB_PATH to a downloaded ONNX Runtime release tarball matching your hardware. \
             See `docs/migrations/0.9.0-feature-surface.md` for setup instructions."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_known_values() {
        assert_eq!(
            OnnxExecutionProvider::from_str("cpu"),
            Some(OnnxExecutionProvider::Cpu)
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("cuda"),
            Some(OnnxExecutionProvider::Cuda)
        );
        assert_eq!(
            OnnxExecutionProvider::from_str("coreml"),
            Some(OnnxExecutionProvider::CoreMl)
        );
    }

    #[test]
    fn from_str_unknown_returns_none() {
        assert_eq!(OnnxExecutionProvider::from_str("bogus"), None);
        assert_eq!(OnnxExecutionProvider::from_str(""), None);
    }

    #[test]
    fn parse_array_form() {
        let v = serde_json::json!(["cuda", "cpu"]);
        let parsed = parse_execution_providers_option(Some(&v)).unwrap();
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
        let parsed = parse_execution_providers_option(Some(&v)).unwrap();
        assert_eq!(parsed, Some(vec![OnnxExecutionProvider::Cuda]));
    }

    #[test]
    fn parse_missing_returns_none() {
        assert!(parse_execution_providers_option(None).unwrap().is_none());
    }

    #[test]
    fn parse_empty_array_returns_none() {
        let v = serde_json::json!([]);
        assert!(
            parse_execution_providers_option(Some(&v))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parse_unknown_entry_errors() {
        let v = serde_json::json!(["cuda", "bogus"]);
        assert!(matches!(
            parse_execution_providers_option(Some(&v)),
            Err(RuntimeError::Config(_))
        ));
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn default_prefers_cuda_when_enabled() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::Cuda, OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
    #[test]
    fn default_prefers_coreml_when_metal_enabled() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::CoreMl, OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
    #[test]
    fn default_is_cpu_only_without_gpu() {
        assert_eq!(
            default_execution_providers(),
            vec![OnnxExecutionProvider::Cpu]
        );
    }

    #[cfg(not(feature = "gpu-metal"))]
    #[test]
    fn unsupported_feature_returns_config_error() {
        // CoreML requested without gpu-metal active
        let result = build_execution_providers(
            Some(&[OnnxExecutionProvider::CoreMl]),
            "test/alias",
            "local/test",
        );
        assert!(matches!(result, Err(RuntimeError::Config(_))));
    }

    #[test]
    fn vendor_eps_parse_to_expected_variants() {
        for (s, expected) in [
            ("rocm", OnnxExecutionProvider::Rocm),
            ("directml", OnnxExecutionProvider::DirectMl),
            ("openvino", OnnxExecutionProvider::OpenVino),
            ("qnn", OnnxExecutionProvider::Qnn),
            ("tensorrt", OnnxExecutionProvider::TensorRt),
            ("webgpu", OnnxExecutionProvider::WebGpu),
        ] {
            assert_eq!(OnnxExecutionProvider::from_str(s), Some(expected));
            assert_eq!(expected.as_str(), s);
        }
    }

    #[cfg(not(feature = "provider-onnx-dynamic"))]
    #[test]
    fn vendor_eps_fail_under_bundled_provider() {
        // Each vendor EP must error with a Config message that points the
        // user at provider-onnx-dynamic when the bundled feature is active.
        for ep in [
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
            OnnxExecutionProvider::Qnn,
            OnnxExecutionProvider::TensorRt,
            OnnxExecutionProvider::WebGpu,
        ] {
            let result = build_execution_providers(Some(&[ep]), "test/alias", "local/test");
            match result {
                Err(RuntimeError::Config(msg)) => {
                    assert!(
                        msg.contains("provider-onnx-dynamic"),
                        "expected message to mention provider-onnx-dynamic, got: {msg}"
                    );
                }
                other => panic!("expected Config error for {ep:?}, got {other:?}"),
            }
        }
    }

    #[cfg(feature = "provider-onnx-dynamic")]
    #[test]
    fn vendor_eps_dispatch_under_dynamic() {
        for ep in [
            OnnxExecutionProvider::Rocm,
            OnnxExecutionProvider::DirectMl,
            OnnxExecutionProvider::OpenVino,
            OnnxExecutionProvider::Qnn,
            OnnxExecutionProvider::TensorRt,
            OnnxExecutionProvider::WebGpu,
        ] {
            // CPU as fallback so the strict path doesn't kick in. We only
            // assert the dispatch builds — actual EP registration happens
            // at session-build time, which we don't reach here.
            let result = build_execution_providers(
                Some(&[ep, OnnxExecutionProvider::Cpu]),
                "test/alias",
                "local/test",
            );
            assert!(result.is_ok(), "expected dispatch to build for {ep:?}");
        }
    }
}
