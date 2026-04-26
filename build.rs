//! Build script. Two responsibilities:
//!
//! 1. **Feature-flag validation** — mutual exclusion of the three ONNX
//!    linking modes plus target-OS guards for vendor-specific features.
//! 2. **ONNX Runtime vendor fetcher** — when one of the Microsoft-distributed
//!    GPU features (`gpu-rocm`, `gpu-coreml`, `gpu-directml`, `gpu-openvino`,
//!    `gpu-qnn`) is active, downloads the appropriate Microsoft prebuilt
//!    tarball, verifies its SHA256, extracts it, and stages the dynamic libs
//!    so the resulting binary finds them at runtime without
//!    `ORT_DYLIB_PATH`.
//!
//! The vendor fetcher is implemented in `build/ort_vendor.rs`.

#[path = "build/ort_vendor.rs"]
mod ort_vendor;

fn main() {
    feature_guards();

    if let Err(e) = ort_vendor::fetch_and_stage() {
        // Build script panics surface as compile errors. Format multi-line so
        // the user sees the structured message even after Cargo's prefixing.
        panic!("\n\nuni-xervo build script: ONNX Runtime vendor fetch failed:\n  {e}\n");
    }

    // Re-run if any of the relevant env vars change. Cargo invalidates the
    // build script automatically on feature changes, but `UNI_XERVO_ORT_DIST_DIR`
    // is an external override worth tracking.
    println!("cargo:rerun-if-env-changed=UNI_XERVO_ORT_DIST_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/ort_vendor.rs");
}

/// Enforce mutual-exclusion between the three ONNX linking modes plus the
/// target-OS gates on vendor-specific GPU features.
///
/// The three modes:
/// - `provider-onnx` (bundled CPU; pyke `download-binaries`) — also
///   activated by `gpu-cuda` / `gpu-tensorrt`.
/// - `_ort-fetched-base` (Microsoft-fetched bundle; `load-dynamic` + our
///   build-time fetcher) — activated by `gpu-rocm` / `gpu-coreml` /
///   `gpu-directml` / `gpu-openvino` / `gpu-qnn`.
/// - `provider-onnx-dynamic` (`load-dynamic`, BYO tarball) — activated only
///   by the user explicitly.
///
/// Combining any two is a configuration error. ort itself would eventually
/// error with a confusing low-level message; we catch it here instead.
fn feature_guards() {
    // 1. Three-mode mutual exclusion.
    let bundled = std::env::var("CARGO_FEATURE_PROVIDER_ONNX").is_ok();
    let fetched = std::env::var("CARGO_FEATURE__ORT_FETCHED_BASE").is_ok();
    let dynamic = std::env::var("CARGO_FEATURE_PROVIDER_ONNX_DYNAMIC").is_ok();

    let active: Vec<&str> = [
        ("provider-onnx", bundled),
        (
            "_ort-fetched-base (via gpu-rocm/coreml/directml/openvino/qnn)",
            fetched,
        ),
        ("provider-onnx-dynamic", dynamic),
    ]
    .iter()
    .filter_map(|(name, on)| if *on { Some(*name) } else { None })
    .collect();

    if active.len() > 1 {
        panic!(
            "Multiple mutually-exclusive ONNX linking modes are active simultaneously:\n  - {}\n\n\
             Pick exactly one based on your deployment model:\n  \
             - `provider-onnx`              : self-contained CPU binary (default).\n  \
             - `gpu-cuda` / `gpu-tensorrt`  : self-contained NVIDIA binary, sidecar EP libs.\n  \
             - `gpu-rocm` / `gpu-coreml` / `gpu-directml` / `gpu-openvino` / `gpu-qnn`:\n    \
               Microsoft-fetched bundle, sidecar EP libs.\n  \
             - `provider-onnx-dynamic`      : load-dynamic, you supply ORT_DYLIB_PATH.\n\n\
             Note that the gpu-cuda/gpu-tensorrt features auto-activate `provider-onnx`,\n\
             and gpu-rocm/coreml/directml/openvino/qnn auto-activate the internal\n\
             `_ort-fetched-base`. So `provider-onnx + gpu-rocm` is the same configuration\n\
             error as activating both modes directly.\n\n\
             See `docs/proposals/gpu-runtime-architecture.md` for guidance.",
            active.join("\n  - ")
        );
    }

    // 1b. provider-fastembed needs an ORT linking mode. Without one, fastembed
    // pulls `ort` in with no linking strategy selected and the build fails with
    // a confusing ort-internal error. Surface it here.
    let fastembed = std::env::var("CARGO_FEATURE_PROVIDER_FASTEMBED").is_ok();
    if fastembed && !(bundled || fetched || dynamic) {
        panic!(
            "`provider-fastembed` requires an ONNX Runtime linking mode but none is enabled.\n\
             Add one of:\n  \
             - `provider-onnx`         (bundled CPU, self-contained binary)\n  \
             - `provider-onnx-dynamic` (load-dynamic, BYO ORT_DYLIB_PATH)\n  \
             - any `gpu-*`             (auto-selects the appropriate mode)\n\n\
             Example: --features \"provider-fastembed,provider-onnx\""
        );
    }

    // 2. Target-OS guards.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

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
