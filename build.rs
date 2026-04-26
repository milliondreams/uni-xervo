//! Build script that validates feature-flag / target-platform compatibility.
//!
//! Cargo features are target-independent — the same `--features ...` selection
//! flows to every target. But several uni-xervo GPU features only make sense on
//! specific operating systems:
//!
//! - `gpu-metal` (candle/mistralrs Metal kernels) — Apple only.
//! - `gpu-coreml` (ort Core ML EP) — Apple only.
//! - `gpu-directml` (ort DirectML EP) — Windows only.
//! - `gpu-rocm` (ort ROCm EP) — Linux only.
//! - `gpu-qnn` (ort Qualcomm QNN EP) — Windows only (today; primarily ARM64).
//!
//! Activating one of these on the wrong OS would produce confusing link errors
//! at the end of a long build. We fail fast here with a clear, copy-pasteable
//! message instead.

fn main() {
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
