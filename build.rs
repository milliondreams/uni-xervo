//! Build script that validates feature-flag / target-platform compatibility.
//!
//! Two classes of checks:
//!
//! 1. **Mutual exclusion** between `provider-onnx` (bundled CPU) and
//!    `provider-onnx-dynamic` (load-dynamic). They activate conflicting
//!    `ort` features (`download-binaries` vs `load-dynamic`) — ort itself
//!    will eventually error, but with a confusing low-level message; we
//!    catch the misconfiguration here instead.
//!
//! 2. **Target-OS guards** for GPU features that only make sense on a
//!    specific operating system:
//!    - `gpu-metal` (candle/mistralrs Metal kernels) — Apple only.
//!    - `gpu-coreml` (ort Core ML EP) — Apple only.
//!    - `gpu-directml` (ort DirectML EP) — Windows only.
//!    - `gpu-rocm` (ort ROCm EP) — Linux only.
//!    - `gpu-qnn` (ort Qualcomm QNN EP) — Windows only (primarily ARM64).
//!
//! Both classes fail fast here with a clear, copy-pasteable message rather
//! than letting the user wait through a long build for a confusing error.

fn main() {
    // 1. Mutual exclusion between the two ONNX linking modes.
    let bundled = std::env::var("CARGO_FEATURE_PROVIDER_ONNX").is_ok();
    let dynamic = std::env::var("CARGO_FEATURE_PROVIDER_ONNX_DYNAMIC").is_ok();
    if bundled && dynamic {
        panic!(
            "Features `provider-onnx` (bundled CPU, ort/download-binaries) and\n\
             `provider-onnx-dynamic` (load-dynamic) are mutually exclusive.\n\
             Pick exactly one based on your deployment model:\n\
             \n\
             - `provider-onnx`         : self-contained binary, CPU only,\n\
                                         zero runtime configuration.\n\
             - `provider-onnx-dynamic` : small binary + user-supplied\n\
                                         ONNX Runtime tarball at runtime\n\
                                         (`ORT_DYLIB_PATH` env var); required\n\
                                         for any GPU EP.\n\
             \n\
             Note: the `gpu-cuda` / `gpu-rocm` / `gpu-coreml` / `gpu-directml` /\n\
             `gpu-openvino` / `gpu-qnn` features all imply `provider-onnx-dynamic`\n\
             — combining any of them with `provider-onnx` is the same configuration\n\
             error.\n\
             \n\
             See `docs/proposals/gpu-runtime-architecture.md` for guidance."
        );
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // (cargo-feature-env-var, feature name, allowed targets, reason)
    let guards: &[(&str, &str, &[&str], &str)] = &[
        (
            "CARGO_FEATURE_GPU_METAL",
            "gpu-metal",
            &["macos", "ios"],
            "candle-metal-kernels and the Objective-C bindings only build on Apple targets",
        ),
        (
            "CARGO_FEATURE_GPU_COREML",
            "gpu-coreml",
            &["macos", "ios"],
            "the ONNX Runtime Core ML execution provider links Apple frameworks",
        ),
        (
            "CARGO_FEATURE_GPU_DIRECTML",
            "gpu-directml",
            &["windows"],
            "DirectML is a DirectX 12 component shipped only on Windows",
        ),
        (
            "CARGO_FEATURE_GPU_ROCM",
            "gpu-rocm",
            &["linux"],
            "AMD ROCm is officially supported only on Linux",
        ),
        (
            "CARGO_FEATURE_GPU_QNN",
            "gpu-qnn",
            &["windows"],
            "Qualcomm QNN is shipped through Windows ARM64 ONNX Runtime tarballs",
        ),
    ];

    for (env_var, feature, allowed, reason) in guards {
        if std::env::var(env_var).is_ok() && !allowed.contains(&target_os.as_str()) {
            panic!(
                "The `{feature}` feature is only supported on {allowed:?}.\n\
                 Reason: {reason}.\n\
                 Remove `{feature}` from your feature list when building for `{target_os}`."
            );
        }
    }
}
